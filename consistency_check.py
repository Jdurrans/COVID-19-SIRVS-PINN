"""
consistency_check.py
23 minute run time

Validates each fixed epidemiological parameter by learning it from data, while keeping all others fixed, then comparing
 the learned value against the fixed choice. Produces one output figure per parameter.

Parameters checked:
  gamma - recovery rate (fixed = 0.1, mean infectious period 10 days)
  xi_i - infection immunity waning rate (fixed = 1/200)
  xi_v - vaccine immunity waning rate (fixed = 1/70)
  epsilon - vaccine efficacy (fixed = 0.8)

Each parameter is learned independently on Wave 2 (Alpha) which has the
cleanest PINN/ODE agreement and is therefore most informative.

Usage:
    python consistency_check.py --data cases.csv --vax vax.csv --N 56000000 --I0 25000 --S0_frac 0.90
"""

import os
import argparse
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import SIRS_SIRVS_PINN as fi

torch.manual_seed(13)
np.random.seed(13)
DEVICE = fi.DEVICE

OUT_DIR      = os.path.dirname(os.path.abspath(__file__))
CHECK_WAVE   = 1        # Wave 2 Alpha (indexed to 0)
CHECK_EPOCHS = 15_000
CHECK_LR     = 5e-4
X_CHECK      = 0.7
CONSISTENCY_THRESHOLD = 0.15   # 15% relative difference means its consistent


# --- learnable-parameter SIRVS PINN ---------------
class SIRVSPINNLearnParam(nn.Module):

    def __init__(self, gamma, xi_i, xi_v, epsilon,
                 N, I0, S0, Ri0, Rv0,
                 param_name, param_init,
                 hidden=64, n_layers=5, beta_max=0.5):
        super().__init__()
        self.N_pop      = float(N)
        self.param_name = param_name

        #store the parameters
        self._gamma_fixed   = float(gamma)
        self._xi_i_fixed    = float(xi_i)
        self._xi_v_fixed    = float(xi_v)
        self._epsilon_fixed = float(epsilon)

        #learned parameter with softplus keeping it positive
        self._raw_param = nn.Parameter(
            torch.tensor(float(np.log(np.exp(param_init) - 1)),
                         dtype=torch.float32))

        #compartment network
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 4)]
        self.net      = nn.Sequential(*layers)
        self.beta_net = fi.BetaNet(beta_max=beta_max)

        s0  = float(S0/N);  i0  = max(float(I0/N),  1e-6)
        ri0 = max(float(Ri0/N), 1e-6); rv0 = max(float(Rv0/N), 1e-6)
        total = s0 + i0 + ri0 + rv0
        s0 /= total; i0 /= total; ri0 /= total; rv0 /= total
        with torch.no_grad():
            self.net[-1].bias.data = torch.tensor(
                [float(np.log(s0)), float(np.log(i0)),
                 float(np.log(ri0)), float(np.log(rv0))],
                dtype=torch.float32)

    @property
    def learned_value(self):
        return nn.functional.softplus(self._raw_param)

    @property
    def gamma(self):
        return self.learned_value if self.param_name == "gamma" \
               else torch.tensor(self._gamma_fixed, device=DEVICE)
    @property
    def xi_i(self):
        return self.learned_value if self.param_name == "xi_i" \
               else torch.tensor(self._xi_i_fixed, device=DEVICE)

    @property
    def xi_v(self):
        return self.learned_value if self.param_name == "xi_v" \
               else torch.tensor(self._xi_v_fixed, device=DEVICE)

    @property
    def epsilon(self):
        return self.learned_value if self.param_name == "epsilon" \
               else torch.tensor(self._epsilon_fixed, device=DEVICE)

    def forward(self, t):
        raw = self.net(t)
        out = torch.softmax(raw, dim=-1)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]

    def beta(self, t):
        return self.beta_net(t)


# -- loss functions ------------
def data_loss(model, t_obs, i_t, ri_t, rv_t, inc_t):
    s, i, ri, rv = model(t_obs)
    pred_inc = (model.beta(t_obs) * s * i).squeeze()
    return (torch.mean((i.squeeze()  - i_t)  ** 2) +
            torch.mean((ri.squeeze() - ri_t) ** 2) +
            torch.mean((rv.squeeze() - rv_t) ** 2) +
            torch.mean((pred_inc     - inc_t) ** 2))


def physics_loss(model, t_phys, vax_t, T_days):
    t = t_phys.clone().requires_grad_(True)
    s, i, ri, rv = model(t)
    b   = model.beta(t)
    g   = model.gamma
    xi_i = model.xi_i; xi_v = model.xi_v; eps = model.epsilon
    vt  = vax_t.unsqueeze(1)

    dsdt  = torch.autograd.grad(s,  t, torch.ones_like(s),  create_graph=True)[0] / T_days
    didt  = torch.autograd.grad(i,  t, torch.ones_like(i),  create_graph=True)[0] / T_days
    dridt = torch.autograd.grad(ri, t, torch.ones_like(ri), create_graph=True)[0] / T_days
    drvdt = torch.autograd.grad(rv, t, torch.ones_like(rv), create_graph=True)[0] / T_days

    f_s  = dsdt  + b*s*i + eps*vt - xi_i*ri - xi_v*rv
    f_i  = didt  - b*s*i + g*i
    f_ri = dridt - g*i   + xi_i*ri
    f_rv = drvdt - eps*vt + xi_v*rv

    floor = lambda x: x.detach().clamp(min=1e-3)
    return torch.mean((f_s/floor(s))**2 + (f_i/floor(i))**2 +
                      (f_ri/floor(ri))**2 + (f_rv/floor(rv))**2)


# -- training ---------------
def train_check(wave, N, gamma, xi_i, xi_v, epsilon,
                param_name, param_init, param_fixed,
                epochs, lr, x, beta_max):
    T      = wave["T"]
    n_phys = 500
    w_d    = x; w_p = 1.0 - x

    t_obs_np = np.arange(T) / (T - 1)
    t_obs  = torch.tensor(t_obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    t_phys = torch.linspace(0, 1, n_phys, device=DEVICE).unsqueeze(1)

    inc_t = torch.tensor(wave["inc"]    / N, dtype=torch.float32, device=DEVICE)
    i_t   = torch.tensor(wave["I_est"]  / N, dtype=torch.float32, device=DEVICE)
    ri_t  = torch.tensor(wave["Ri_est"] / N, dtype=torch.float32, device=DEVICE)
    rv_t  = torch.tensor(wave["Rv_est"] / N, dtype=torch.float32, device=DEVICE)
    vax_np = np.interp(np.linspace(0, 1, n_phys) * (T-1),
                       np.arange(T), wave["vax"] / N)
    vax_t = torch.tensor(vax_np, dtype=torch.float32, device=DEVICE)

    model = SIRVSPINNLearnParam(
        gamma, xi_i, xi_v, epsilon, N,
        wave["I0"], wave["S0"], wave["Ri0"], wave["Rv0"],
        param_name=param_name, param_init=param_init,
        beta_max=beta_max).to(DEVICE)

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=500, factor=0.5, min_lr=1e-5)

    param_history = []
    loss_history  = []
    print_every   = max(1, epochs // 8)

    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        l_d = data_loss(model, t_obs, i_t, ri_t, rv_t, inc_t)
        l_p = physics_loss(model, t_phys, vax_t, float(T - 1))
        loss = w_d * l_d + w_p * l_p
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(loss)

        p_val = model.learned_value.item()
        param_history.append(p_val)
        loss_history.append(loss.item())


    return model, param_history, loss_history


# -- per-parameter plot ---------------
PARAM_META = {
    "gamma": {
        "label":    "γ (recovery rate)",
        "unit":     "day⁻¹",
        "period":   "mean infectious period",
        "period_fn": lambda v: 1/v,
        "period_unit": "days",
        "color":    "purple",
    },
    "xi_i": {
        "label":    "ξᵢ (infection immunity waning)",
        "unit":     "day⁻¹",
        "period":   "mean immunity duration",
        "period_fn": lambda v: 1/v,
        "period_unit": "days",
        "color":    "green",
    },
    "xi_v": {
        "label":    "ξᵥ (vaccine immunity waning)",
        "unit":     "day⁻¹",
        "period":   "mean vaccine protection duration",
        "period_fn": lambda v: 1/v,
        "period_unit": "days",
        "color":    "blue",
    },
    "epsilon": {
        "label":    "ε (vaccine efficacy)",
        "unit":     "—",
        "period":   "efficacy",
        "period_fn": lambda v: v * 100,
        "period_unit": "%",
        "color":    "red",
    },
}


def plot_param(wave, model, param_history, loss_history,
               param_name, param_fixed, N):
    meta          = PARAM_META[param_name]
    param_learned = param_history[-1]
    rel_diff      = abs(param_learned - param_fixed) / abs(param_fixed)
    consistent    = rel_diff < CONSISTENCY_THRESHOLD
    col           = meta["color"]
    dates         = wave["dates"]
    T             = wave["T"]

    #evaluate PINN
    model.eval()
    t_e = torch.linspace(0, 1, 300, device=DEVICE).unsqueeze(1)
    with torch.no_grad():
        out  = model(t_e)
        beta = model.beta(t_e).cpu().numpy().squeeze()
    s_p, i_p, ri_p, rv_p = [v.cpu().numpy().squeeze()*N for v in out]
    days_f     = np.linspace(0, T-1, 300)
    dates_fine = [dates[0] + timedelta(days=float(d)) for d in days_f]
    inc_pred   = beta * (s_p/N) * (i_p/N) * N

    fmt = mdates.DateFormatter("%b %Y")
    loc = mdates.MonthLocator(interval=1)

    def ax_fmt(ax):
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(loc)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.22)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # -- parameter convergence ---------------
    ax = axes[0, 0]
    ax.plot(param_history, color=col, lw=2, label=f"Learned {param_name}")
    ax.axhline(param_fixed,   color="red", lw=1.5, ls="--",
               label=f"Fixed = {param_fixed:.5f}")
    ax.axhline(param_learned, color="green", lw=1.5, ls=":",
               label=f"Learned = {param_learned:.5f}")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(f"{param_name} ({meta['unit']})", fontsize=12)
    ax.set_title(f"{meta['label']} — convergence", fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.22)

    # -- loss convergence ---------------
    ax = axes[0, 1]
    ax.semilogy(loss_history, color="gray", lw=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Total loss (log)", fontsize=12)
    ax.set_title("Loss convergence", fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.22)

    # -- incidence fit ---------------
    ax = axes[0, 2]
    ax.bar(dates, wave["inc"], alpha=0.30, color="red", label="Observed")
    ax.plot(dates_fine, inc_pred, color=col, lw=2,
            label=f"PINN (learned {param_name})")
    ax.set_ylabel("Daily cases", fontsize=12)
    ax.set_title("Incidence fit", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax_fmt(ax)

    # -- compartments ----------------
    ax = axes[1, 0]
    ax.plot(dates_fine, s_p/N*100,  color="blue", lw=2, label="S")
    ax.plot(dates_fine, i_p/N*100,  color="red", lw=2, label="I")
    ax.plot(dates_fine, ri_p/N*100, color="green", lw=2, label="Rᵢ")
    ax.plot(dates_fine, rv_p/N*100, color="purple", lw=2, label="Rᵥ")
    ax.set_ylabel("Population (%)", fontsize=12)
    ax.set_title("Compartments", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax_fmt(ax)

    # -- beta(t)---------------
    ax = axes[1, 1]
    ax.plot(dates_fine, beta, color=col, lw=2)
    ax.set_ylabel("β(t)", fontsize=12)
    ax.set_title("Learned β(t)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, None)
    ax_fmt(ax)

    # -- summary panel ---------------
    ax = axes[1, 2]
    ax.axis("off")
    pf = meta["period_fn"]
    pu = meta["period_unit"]
    verdict_col = "green" if consistent else "red"
    verdict     = "CONSISTENT" if consistent else "INCONSISTENT"

    summary = (
        f"Parameter: {param_name}\n"
        f"{'─'*36}\n\n"
        f"Fixed value:    {param_fixed:.5f} {meta['unit']}\n"
        f"Learned value:  {param_learned:.5f} {meta['unit']}\n\n"
        f"Abs difference: {abs(param_learned-param_fixed):.5f}\n"
        f"Rel difference: {rel_diff*100:.1f}%\n\n"
        f"{meta['period']} (fixed):    {pf(param_fixed):.1f} {pu}\n"
        f"{meta['period']} (learned):  {pf(param_learned):.1f} {pu}\n\n"
        f"Threshold: ±{CONSISTENCY_THRESHOLD*100:.0f}% relative\n\n"
        f"Verdict: {verdict}\n"
    )
    if consistent:
        summary += "Fixed value is supported\nby the observed data."
    else:
        summary += "Fixed value deviates from\ndata-informed estimate.\nConsider revising."

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=13, verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor=verdict_col, alpha=0.12))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"consistency_{param_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Figure → {path}")
    plt.show()



# -- command line arguments ---------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Consistency check for all fixed epidemiological parameters")
    p.add_argument("--data",       type=str, required=True)
    p.add_argument("--vax",        type=str, required=True)
    p.add_argument("--N",          type=int,   default=56_000_000)
    p.add_argument("--I0",         type=int,   default=50)
    p.add_argument("--S0_frac",    type=float, default=1.0)
    p.add_argument("--start",      type=str,   default="2020-01-30")
    p.add_argument("--wave_start", type=str,   default="2020-07-01")
    p.add_argument("--gamma",      type=float, default=fi.GAMMA)
    p.add_argument("--xi_i",       type=float, default=fi.XI_I)
    p.add_argument("--xi_v",       type=float, default=fi.XI_V)
    p.add_argument("--epsilon",    type=float, default=fi.EPSILON)
    p.add_argument("--epochs",     type=int,   default=CHECK_EPOCHS)
    p.add_argument("--lr",         type=float, default=CHECK_LR)
    p.add_argument("--x",          type=float, default=X_CHECK)
    p.add_argument("--wave_idx",   type=int,   default=CHECK_WAVE,
                   help="Wave to use for check (0–3). Default 1 = Wave 2.")
    p.add_argument("--params",     type=str,   nargs="+",
                   default=["gamma", "xi_i", "xi_v", "epsilon"],
                   help="Which parameters to check. Default: all four.")
    return p.parse_args()


# -- entry point ---------------
if __name__ == "__main__":
    args = parse_args()
    N    = args.N

    # Fixed values dict for easy lookup
    fixed_vals = {
        "gamma":   args.gamma,
        "xi_i":    args.xi_i,
        "xi_v":    args.xi_v,
        "epsilon": args.epsilon,
    }


    cases, vax, dates_all = fi.load_data(args.data, args.vax, args.start)

    waves_template = fi.split_waves(
        cases, vax, dates_all, args.wave_start, fi.WAVE_DEFS,
        N, args.I0, args.gamma, args.xi_i, args.xi_v, args.epsilon,
        S0_frac=args.S0_frac)

    #pre-train earlier waves to get correct chained ICs
    if args.wave_idx > 0:
        print(f"\nPre-training waves 0–{args.wave_idx-1} for ICs …")
        models_pre = []
        for i in range(args.wave_idx):
            m = fi.train_wave(
                waves_template[i], N, args.gamma,
                args.xi_i, args.xi_v, args.epsilon,
                base_epochs=4_000, LR=args.lr,
                wave_idx=i, x=args.x)
            models_pre.append(m)
            fi.chain_wave_ics(
                waves_template, models_pre, wave_idx=i, N=N,
                gamma=args.gamma, xi_i=args.xi_i,
                xi_v=args.xi_v, epsilon=args.epsilon)

    wave = waves_template[args.wave_idx]
    bmax = fi.BETA_MAX_PER_WAVE[min(args.wave_idx, len(fi.BETA_MAX_PER_WAVE)-1)]

    summary_rows = []

    for param_name in args.params:
        param_fixed = fixed_vals[param_name]

        model, param_history, loss_history = train_check(
            wave, N,
            args.gamma, args.xi_i, args.xi_v, args.epsilon,
            param_name=param_name,
            param_init=param_fixed,   # start from fixed value
            param_fixed=param_fixed,
            epochs=args.epochs, lr=args.lr, x=args.x,
            beta_max=bmax)

        param_learned = param_history[-1]
        rel_diff      = abs(param_learned - param_fixed) / abs(param_fixed)
        consistent    = rel_diff < CONSISTENCY_THRESHOLD

        summary_rows.append({
            "param":         param_name,
            "fixed":         param_fixed,
            "learned":       param_learned,
            "rel_diff_pct":  rel_diff * 100,
            "consistent":    consistent,
        })

        plot_param(wave, model, param_history, loss_history,
                   param_name, param_fixed, N)


    #