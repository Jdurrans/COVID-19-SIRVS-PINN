"""
x_sweep.py  —  Data/physics weight sensitivity sweep for the SIRS→SIRVS PINN

59 minute run time, but then models saved so plots can be altered using learned models

This script trains the PINN with a reduced number of epochs for each value of X for 0.1 - 0.9 (increasing by 0.1),
Finding the value with the smallest overall loss value.

The loss function is X * L_data + (1-X) * L_physics.
So larger values of X should, in theory, favour the data, and be less consistent with the ODE forwards pass.

Usage:
    python x_sweep.py --data cases.csv --vax vax.csv --N 56000000 --I0 25000 --S0_frac 0.90
which then saves models to allow quicker plotting changes
retrain by first running
    del sweep_model_x*.pt
then the usage line again
"""

import os
import csv
import argparse
from copy import deepcopy
from datetime import timedelta
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import SIRS_SIRVS_PINN as fi

torch.manual_seed(13)
np.random.seed(13)
DEVICE = fi.DEVICE

# -- sweep configuration ---------------
X_VALUES          = [round(x * 0.1, 1) for x in range(1, 10)]   # 0.1 … 0.9
SWEEP_BASE_EPOCHS = 3000
WARMUP_FRAC       = 0.25
SWEEP_LR          = 5e-4
WARMUP_LR         = 1e-3


# -- command line arguments ---------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        type=str,   required=True)
    p.add_argument("--vax",         type=str,   required=True)
    p.add_argument("--N",           type=int,   default=56_000_000)
    p.add_argument("--I0",          type=int,   default=50)
    p.add_argument("--S0_frac",     type=float, default=1.0)
    p.add_argument("--start",       type=str,   default="2020-01-30")
    p.add_argument("--wave_start",  type=str,   default="2020-07-01")
    p.add_argument("--gamma",       type=float, default=fi.GAMMA)
    p.add_argument("--xi_i",        type=float, default=fi.XI_I)
    p.add_argument("--xi_v",        type=float, default=fi.XI_V)
    p.add_argument("--epsilon",     type=float, default=fi.EPSILON)
    p.add_argument("--base_epochs", type=int,   default=SWEEP_BASE_EPOCHS)
    p.add_argument("--lr",          type=float, default=SWEEP_LR)
    p.add_argument("--beta_max",    type=float, default=None)
    return p.parse_args()


# -- training ---------------
def train_one(wave, N, gamma, xi_i, xi_v, epsilon,
              base_epochs, lr, wave_idx, x, beta_max=None):

    if beta_max is None:
        bmax = fi.BETA_MAX_PER_WAVE[min(wave_idx, len(fi.BETA_MAX_PER_WAVE)-1)]
    else:
        bmax = float(beta_max)

    T      = wave["T"]
    mtype  = wave["type"]
    n_phys = fi.N_PHYS_PER_WAVE[min(wave_idx, len(fi.N_PHYS_PER_WAVE) - 1)]
    w_d    = x
    w_p    = 1.0 - x

    epochs = fi._scaled_epochs(T, base_epochs)

    t_obs_np = np.arange(T) / (T - 1)
    t_obs  = torch.tensor(t_obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    t_phys = torch.linspace(0, 1, n_phys, device=DEVICE).unsqueeze(1)

    inc_t = torch.tensor(wave["inc"]    / N, dtype=torch.float32, device=DEVICE)
    i_t   = torch.tensor(wave["I_est"]  / N, dtype=torch.float32, device=DEVICE)
    ri_t  = torch.tensor(wave["Ri_est"] / N, dtype=torch.float32, device=DEVICE)

    if mtype == "SIRS":
        model = fi.SIRSPINN(gamma, xi_i, N,
                            wave["I0"], wave["S0"], wave["Ri0"],
                            beta_max=bmax).to(DEVICE)
        def step(mdl):
            l_d = fi.sirs_loss_data(mdl, t_obs, i_t, ri_t, inc_t)
            l_p = fi.sirs_loss_physics(mdl, t_phys, float(T - 1))
            return l_d, l_p
    else:
        rv_t = torch.tensor(wave["Rv_est"] / N, dtype=torch.float32, device=DEVICE)
        vax_phys_np = np.interp(
            np.linspace(0, 1, n_phys) * (T - 1),
            np.arange(T), wave["vax"] / N)
        vax_t = torch.tensor(vax_phys_np, dtype=torch.float32, device=DEVICE)
        model = fi.SIRVSPINN(gamma, xi_i, xi_v, epsilon, N,
                             wave["I0"], wave["S0"], wave["Ri0"], wave["Rv0"],
                             beta_max=bmax).to(DEVICE)
        def step(mdl):
            l_d = fi.sirvs_loss_data(mdl, t_obs, i_t, ri_t, rv_t, inc_t)
            l_p = fi.sirvs_loss_physics(mdl, t_phys, vax_t, float(T - 1))
            return l_d, l_p

    warmup_epochs = max(1, int(epochs * WARMUP_FRAC))
    main_epochs   = epochs - warmup_epochs

    # phase 1: warmup — high fixed LR to escape flat β
    opt_w = torch.optim.Adam(model.parameters(), lr=WARMUP_LR)
    for _ in range(warmup_epochs):
        model.train(); opt_w.zero_grad()
        l_d, l_p = step(model)
        loss = w_d * l_d + w_p * l_p
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_w.step()

    # phase 2: main training with LR scheduler
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=400, factor=0.5, min_lr=1e-5)

    last_ld = last_lp = last_loss = None
    for _ in range(main_epochs):
        model.train(); opt.zero_grad()
        l_d, l_p = step(model)
        loss = w_d * l_d + w_p * l_p
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step(loss.detach())
        last_ld   = l_d.item()
        last_lp   = l_p.item()
        last_loss = loss.item()

    return model, last_loss, last_ld, last_lp


# -- evaluate ---------------
def get_curves(model, wave, N, n=300):
    model.eval()
    t_e = torch.linspace(0, 1, n, device=DEVICE).unsqueeze(1)
    with torch.no_grad():
        out  = model(t_e)
        beta = model.beta(t_e).cpu().numpy().squeeze()
    s = out[0].cpu().numpy().squeeze()
    i = out[1].cpu().numpy().squeeze()
    inc = beta * s * i * N
    days = np.linspace(0, wave["T"] - 1, n)
    dates_fine = [wave["dates"][0] + timedelta(days=float(d)) for d in days]
    return dates_fine, beta, inc


# -- sweep ---------------
def run_sweep(waves_template, N, gamma, xi_i, xi_v, epsilon,
              base_epochs, lr, beta_max=None, out_dir="."):
    results = {}
    records = []
    n_total = len(X_VALUES) * len(waves_template)
    run_no  = 0

    for x in X_VALUES:
        results[x] = []
        waves = deepcopy(waves_template)
        models_so_far = []

        for wi, wave in enumerate(waves):
            run_no += 1
            wlabel    = wave["label"].replace("\n", " ")
            scaled    = fi._scaled_epochs(wave["T"], base_epochs)
            save_path = os.path.join(out_dir, f"sweep_model_x{x:.1f}_wave{wi}.pt")

            # determine beta_max for this wave
            if beta_max is None:
                bmax = fi.BETA_MAX_PER_WAVE[min(wi, len(fi.BETA_MAX_PER_WAVE)-1)]
            else:
                bmax = float(beta_max)

            if os.path.exists(save_path):
                print(f"[{run_no}/{n_total}]  x={x:.1f}  {wlabel}  "
                      f"— loading saved model …", end="  ", flush=True)

                # Reconstruct the correct model class with current wave ICs
                if wave["type"] == "SIRS":
                    model = fi.SIRSPINN(gamma, xi_i, N,
                                        wave["I0"], wave["S0"], wave["Ri0"],
                                        beta_max=bmax).to(DEVICE)
                else:
                    model = fi.SIRVSPINN(gamma, xi_i, xi_v, epsilon, N,
                                         wave["I0"], wave["S0"], wave["Ri0"],
                                         wave["Rv0"], beta_max=bmax).to(DEVICE)
                model.load_state_dict(torch.load(save_path, map_location=DEVICE))
                model.eval()
                loss = l_d = l_p = float("nan")
                print("done")

            else:
                print(f"[{run_no}/{n_total}]  x={x:.1f}  {wlabel}  "
                      f"({wave['T']}d  epochs={scaled}) …", end="  ", flush=True)
                model, loss, l_d, l_p = train_one(
                    wave, N, gamma, xi_i, xi_v, epsilon,
                    base_epochs, lr, wi, x, beta_max=beta_max)
                torch.save(model.state_dict(), save_path)
                print(f"total={loss:.3e}  data={l_d:.3e}  phys={l_p:.3e}")

            models_so_far.append(model)
            fi.chain_wave_ics(waves, models_so_far, wave_idx=wi,
                              N=N, gamma=gamma, xi_i=xi_i,
                              xi_v=xi_v, epsilon=epsilon)

            dates_fine, beta, inc = get_curves(model, wave, N)
            results[x].append({
                "loss": loss, "l_data": l_d, "l_phys": l_p,
                "dates_fine": dates_fine, "beta": beta, "inc": inc,
            })
            records.append({
                "x": x, "wave": wlabel, "type": wave["type"],
                "T": wave["T"], "epochs": scaled,
                "total_loss": loss, "data_loss": l_d, "phys_loss": l_p,
            })

    return results, records


# -- plotting ---------------
def plot_sweep(results, records, waves_template, N, gamma,
               base_epochs, out_dir):

    # sum losses across all waves for each x value
    ldata_by_x = {x: 0.0 for x in X_VALUES}
    lphys_by_x = {x: 0.0 for x in X_VALUES}
    for rec in records:
        ldata_by_x[rec["x"]] += rec["data_loss"]
        lphys_by_x[rec["x"]] += rec["phys_loss"]

    # scale-free normalised score: normalising each component by its value at x=0.5
    ref_x = min(X_VALUES, key=lambda x: abs(x - 0.5))
    ref_d = ldata_by_x[ref_x] or 1.0
    ref_p = lphys_by_x[ref_x] or 1.0
    norm_score = {x: (ldata_by_x[x] / ref_d) + (lphys_by_x[x] / ref_p)
                  for x in X_VALUES}
    best_x = min(X_VALUES, key=lambda x: norm_score[x])

    fig, (ax_norm, ax_dat, ax_phy) = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.38, top=0.88, bottom=0.14,
                        left=0.07, right=0.97)

    xs = X_VALUES
    ax_norm.plot(xs, [norm_score[x] for x in xs], "o-", color="purple", lw=2.5)
    ax_dat.plot( xs, [ldata_by_x[x] for x in xs], "o-", color="blue", lw=2.5)
    ax_phy.plot( xs, [lphys_by_x[x] for x in xs], "o-", color="green", lw=2.5)

    for ax in [ax_norm, ax_dat, ax_phy]:
        ax.axvline(best_x, color="red", lw=2.0, ls="--",
                   label=f"Best x = {best_x:.1f}")

    for ax, title, ylabel in zip(
        [ax_norm, ax_dat, ax_phy],
        ["Normalised loss (scale-free)",
         "Data loss $L_{data}$ (summed over waves)",
         "Physics loss $L_{physics}$ (summed over waves)"],
        ["$L_{data}$/ref + $L_{phys}$/ref", "$L_{data}$", "$L_{physics}$"]
    ):
        ax.set_xlabel("x  (data weight)", fontsize=15)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(xs)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, alpha=0.25)
        ax.set_yscale("log")
        ax.legend(fontsize=13)

    path = os.path.join(out_dir, "x_sweep_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nFigure → {path}")
    plt.show()
    return best_x


# -- csv ---------------
#def save_csv(records, out_dir):
#    path = os.path.join(out_dir, "x_sweep_summary.csv")
#    fields = ["x", "wave", "type", "T", "epochs",
#              "total_loss", "data_loss", "phys_loss"]
#    with open(path, "w", newline="") as f:
#        w = csv.DictWriter(f, fieldnames=fields)
#        w.writeheader(); w.writerows(records)
#    print(f"CSV   → {path}")


# --entry point ---------------
if __name__ == "__main__":
    args    = parse_args()
    N       = args.N
    gamma   = args.gamma
    xi_i    = args.xi_i
    xi_v    = args.xi_v
    epsilon = args.epsilon
    base_ep = args.base_epochs
    lr      = args.lr
    out_dir = os.path.dirname(os.path.abspath(__file__))

    bmax_desc = (f"{args.beta_max} (all waves)"
                 if args.beta_max is not None
                 else f"{fi.BETA_MAX_PER_WAVE} (per wave)")

    print("=" * 64)
    print(f"x sweep:  {X_VALUES}")
    print(f"Base epochs (Wave 1): {base_ep}  "
          f"(warmup={int(base_ep*WARMUP_FRAC)}, "
          f"main={base_ep - int(base_ep*WARMUP_FRAC)})")
    print(f"S0_frac={args.S0_frac:.2f}  β_max={bmax_desc}")
    print(f"Loss = x·L_data + (1-x)·L_physics  — two terms only")
    print(f"Total training runs:  {len(X_VALUES) * len(fi.WAVE_DEFS)}  "
          f"(Wave 4 will use ~{fi._scaled_epochs(184, base_ep)} epochs)")
    print("=" * 64)

    cases, vax, dates_all = fi.load_data(args.data, args.vax, args.start)

    waves_template = fi.split_waves(
        cases, vax, dates_all, args.wave_start, fi.WAVE_DEFS,
        N, args.I0, gamma, xi_i, xi_v, epsilon,
        S0_frac=args.S0_frac)

    print(f"\nRunning {len(X_VALUES)} × {len(waves_template)} = "
          f"{len(X_VALUES) * len(waves_template)} training runs …\n")

    results, records = run_sweep(
        waves_template, N, gamma, xi_i, xi_v, epsilon,
        base_ep, lr, beta_max=args.beta_max, out_dir=out_dir)

    best_x = plot_sweep(
        results, records, waves_template, N, gamma, base_ep, out_dir)
    save_csv(records, out_dir)

    print("\n" + "=" * 64)
    print(f"Recommended x:  {best_x:.1f}")
    print(f"Full run:  python lag2.py --data {args.data} "
          f"--vax {args.vax} --N {N} --I0 {args.I0} "
          f"--S0_frac {args.S0_frac} --x {best_x:.1f}")
    print("=" * 64)