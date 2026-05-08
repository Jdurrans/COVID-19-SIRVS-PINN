"""
SIRS SIRVS Physics-Informed Neural Network
27 minute runtime learning the model

Trains two neural networks, for compartmental and betanet, to learn parameter beta(t), to find Re(t)
Usage:
    pip install torch numpy matplotlib odfpy
First run and train:
    python SIRS_SIRVS_PINN.py --data cases.csv --vax vax.csv --N 56000000 --I0 25000 --S0_frac 0.90 --x 0.8 --ukhsa_r "ukhsa_r.ods"
Subsequent using loaded model:
    python SIRS_SIRVS_PINN.py --data cases.csv --vax vax.csv --N 56000000 --I0 25000 --S0_frac 0.90 --x 0.8 --load_models --ukhsa_r "ukhsa_r.ods"

Delete saved waves:
    del model_wave1.pt; del model_wave2.pt; del model_wave3.pt; del model_wave4.pt
"""

import os
import argparse
from datetime import date, timedelta
import numpy as np
import torch
import torch.nn as nn
#pytorch used for layers, activation functions, (building the neural network)
import matplotlib
#save plots rather than show them every time
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.gridspec import GridSpec

torch.manual_seed(13)
np.random.seed(13)
#fixing the seeds meaning every run uses the same random weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- wave defenitions ---------------
VAX_START = date(2020, 12, 8)

WAVE_DEFS = [
    ("Wave 1\n(pre-vax)",  date(2020, 12,  8), "SIRS"),
    ("Wave 2\n(Alpha)",    date(2021,  7,  1), "SIRVS"),
    ("Wave 3\n(Delta)",    date(2021, 12,  1), "SIRVS"),
    ("Wave 4\n(Omicron)",  date(2022,  5,  1), "SIRVS"),
]

# -- fixed parameters ---------------
GAMMA   = 1 / 10
XI_I    = 1 / 200
XI_V    = 1 / 70
EPSILON = 8 / 10

# -- training hyperparamters ---------------
# longer waves get: int(BASE_EPOCHS * T / 160), minimum BASE_EPOCHS.
BASE_EPOCHS = 15_000
LR          = 5e-4
PRINT_EVERY = 3_000
#base epochs 15000, based on wave 1 with duration 160 days. With other waves getting max((15,000*T/160),15,000)

# beta ceilings per wave
BETA_MAX_PER_WAVE = [0.5, 0.5, 0.7, 1.2]  # [Wave1, Wave2, Wave3, Wave4]

# underreporting factors per wave (true infections / reported cases)
# Wave 1: 7x
# Wave 2: 3.5x
# Wave 3: 2.5x
# Wave 4: 1.5x
UNDERREPORTING_PER_WAVE = [7.0, 3.5, 2.5, 1.5]

X_DATA = 0.8   # overwritten by --x

#approx 2.5 collocation points per wave (wave 2 has ~1.5)
N_PHYS_PER_WAVE = [400, 320, 380, 375]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
#sets up directory for saving files

# initial run line arguments (command line arguments)
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",        type=str,   required=True)
    p.add_argument("--vax",         type=str,   required=True)
    p.add_argument("--N",           type=int,   default=56_000_000)
    p.add_argument("--I0",          type=int,   default=50)
    p.add_argument("--S0_frac",     type=float, default=1.0,
                   help="Susceptible fraction at Wave 1 start (0-1). "
                        "Set to ~0.90 to account for ~10%% prior immunity "
                        "from the spring 2020 wave. Default 1.0.")
    p.add_argument("--start",       type=str,   default="2020-01-30")
    p.add_argument("--wave_start",  type=str,   default="2020-07-01")
    p.add_argument("--gamma",       type=float, default=GAMMA)
    p.add_argument("--xi_i",        type=float, default=XI_I)
    p.add_argument("--xi_v",        type=float, default=XI_V)
    p.add_argument("--epsilon",     type=float, default=EPSILON)
    p.add_argument("--base_epochs", type=int,   default=BASE_EPOCHS)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--x",           type=float, nargs="+", default=[0.5])
    p.add_argument("--beta_max",    type=float, default=None)
    p.add_argument("--ukhsa_r", type=str, default=None)
    p.add_argument("--load_models", action="store_true")
    return p.parse_args()

# -- data loading and compartment estimates ---------------
def smooth7(x):
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        lo, hi = max(0, i-3), min(len(x), i+4)
        out[i] = x[lo:hi].mean()
    return out
#7 day rolling average, with truncted ends

def load_data(cases_path, vax_path, start_str):
    cases_raw = np.loadtxt(cases_path, delimiter=",", encoding="utf-8-sig").flatten()
    cases_raw = np.clip(cases_raw, 0, None)
    cases     = smooth7(cases_raw)

    vax_raw = np.loadtxt(vax_path, delimiter=",", encoding="utf-8-sig").flatten()
    vax_raw = np.clip(vax_raw, 0, None)
    vax     = smooth7(vax_raw)

    start  = date.fromisoformat(start_str)
    dates  = [start + timedelta(days=i) for i in range(min(len(cases), len(vax)))]

    for i, d in enumerate(dates):
        if d < VAX_START:
            vax[i] = 0.0

    T = len(dates)
    cases = cases[:T]; vax = vax[:T]

    print(f"Cases: {T} days, smoothed peak={cases.max():.0f}")
    print(f"Vax:   {T} days, peak={vax.max():.0f}/day, zero before {VAX_START}")
    return np.array(cases), np.array(vax), dates
#reading case and vax data from csv files, with for loop adding 0s before the vax start date
#len(dates) matches the lenth of the two data sets

# making targets for the data loss
def reconstruct_sirs(inc, N, I0, gamma, xi_i, T,
                     S_init=None, I_init=None, Ri_init=None):
    S  = float(S_init  if S_init  is not None else N - I0)
    I  = float(I_init  if I_init  is not None else I0)
    Ri = float(Ri_init if Ri_init is not None else 0.0)
    Ss, Is, Ris = [S], [I], [Ri]

    for t in range(T - 1):
        new_inf = float(np.clip(inc[t], 0, S))
        new_rec = gamma * I
        wan_i   = xi_i * Ri
        S  = np.clip(S  - new_inf + wan_i, 0, N)
        I  = np.clip(I  + new_inf - new_rec, 0, N)
        Ri = np.clip(Ri + new_rec - wan_i, 0, N)
        total = S + I + Ri
        if total > 0:
            S *= N/total; I *= N/total; Ri *= N/total
        Ss.append(S); Is.append(I); Ris.append(Ri)

    return np.array(Ss), np.array(Is), np.array(Ris)
#creates estimates for the the trajectories, which is then compared by the pinn estiamtes for the data loss

def reconstruct_sirvs(inc, vax, N, I0, gamma, xi_i, xi_v, epsilon, T,
                      S_init=None, I_init=None, Ri_init=None, Rv_init=None):
    S  = float(S_init  if S_init  is not None else N - I0)
    I  = float(I_init  if I_init  is not None else I0)
    Ri = float(Ri_init if Ri_init is not None else 0.0)
    Rv = float(Rv_init if Rv_init is not None else 0.0)
    Ss, Is, Ris, Rvs = [S], [I], [Ri], [Rv]

    for t in range(T - 1):
        new_inf = float(np.clip(inc[t], 0, S))
        new_vax = float(np.clip(epsilon * vax[t], 0, max(S - new_inf, 0)))
        new_rec = gamma * I
        wan_i   = xi_i * Ri
        wan_v   = xi_v * Rv
        S  = np.clip(S  - new_inf - new_vax + wan_i + wan_v, 0, N)
        I  = np.clip(I  + new_inf - new_rec, 0, N)
        Ri = np.clip(Ri + new_rec - wan_i, 0, N)
        Rv = np.clip(Rv + new_vax - wan_v, 0, N)
        total = S + I + Ri + Rv
        if total > 0:
            S *= N/total; I *= N/total; Ri *= N/total; Rv *= N/total
        Ss.append(S); Is.append(I); Ris.append(Ri); Rvs.append(Rv)

    return np.array(Ss), np.array(Is), np.array(Ris), np.array(Rvs)

#creates estimates for the the trajectories, which is then compared by the pinn estiamtes for the data loss


# -- wave splitting ---------------
def split_waves(cases, vax, dates_all, wave_start_str, wave_defs, N, I0, gamma, xi_i, xi_v, epsilon, S0_frac=1.0):
  
    wave_start = date.fromisoformat(wave_start_str)
    ws_idx     = next((i for i, d in enumerate(dates_all) if d >= wave_start), 0)

    # sets s0 as 90
    S0_frac = float(np.clip(S0_frac, 0.0, 1.0))
    S0   = S0_frac * float(N - I0)
    Ri0  = (1.0 - S0_frac) * float(N - I0)   # prior immune → Ri
    I0_w = float(I0)
    Rv0  = 0.0

    print(f"Wave 1 ICs: S={S0/N*100:.2f}%  I={I0_w:.0f}  "
          f"Ri={Ri0/N*100:.2f}% (prior immunity)  Rv=0.00%")

    cases_w = cases[ws_idx:]; vax_w = vax[ws_idx:]
    dates_w = dates_all[ws_idx:]
    waves   = []
    prev    = 0

    for label, end_date, mtype in wave_defs:
        end_idx = next((i for i, d in enumerate(dates_w) if d >= end_date),
                       len(dates_w))
        end_idx = min(end_idx, len(cases_w))
        if end_idx <= prev:
            continue

        wc = cases_w[prev:end_idx]
        wv = vax_w[prev:end_idx]
        wd = dates_w[prev:end_idx]
        T_w = len(wc)

        #aply underreporting correction — scale reported cases up to estimated true incidence before passing to the PINN
        wave_idx_local = len(waves)  # current wave index (0-based)
        ur = UNDERREPORTING_PER_WAVE[min(wave_idx_local,
                                         len(UNDERREPORTING_PER_WAVE) - 1)]
        wc = wc * ur
        print(f"  Underreporting factor for {label.replace(chr(10), ' ')}: "
              f"{ur}x  (peak true incidence ≈ {wc.max():.0f})")

        if mtype == "SIRS":
            S_e, I_e, Ri_e = reconstruct_sirs(
                wc, N, I0_w, gamma, xi_i, T_w, S0, I0_w, Ri0)
            Rv_e = np.zeros(T_w)
        else:
            S_e, I_e, Ri_e, Rv_e = reconstruct_sirvs(
                wc, wv, N, I0_w, gamma, xi_i, xi_v, epsilon, T_w,
                S0, I0_w, Ri0, Rv0)

        waves.append({
            "label": label, "type": mtype,
            "inc": wc, "vax": wv, "dates": wd, "T": T_w,
            "S_est": S_e, "I_est": I_e, "Ri_est": Ri_e, "Rv_est": Rv_e,
            "S0": S0, "I0": I0_w, "Ri0": Ri0, "Rv0": Rv0,
        })

        #placeholder end-state — overwritten by chain_wave_ics() after training
        S0   = float(S_e[-1])
        I0_w = float(I_e[-1])
        Ri0  = float(Ri_e[-1])
        Rv0  = float(Rv_e[-1])
        prev = end_idx

    print("Waves: " + ", ".join(
        f"{w['label'].replace(chr(10),' ')} [{w['type']}] ({w['T']}d)"
        for w in waves))
    return waves


def chain_wave_ics(waves, models, wave_idx, N, gamma, xi_i, xi_v, epsilon):
    #using end conditions of wave k as intiial conditions of wave k+1
    if wave_idx + 1 >= len(waves):
        return

    model = models[wave_idx]
    wave  = waves[wave_idx]
    model.eval()

    t_end = torch.ones(1, 1, device=DEVICE)
    with torch.no_grad():
        out = model(t_end)

    if wave["type"] == "SIRS":
        S_f  = out[0].item() * N
        I_f  = out[1].item() * N
        Ri_f = out[2].item() * N
        Rv_f = 0.0
    else:
        S_f  = out[0].item() * N
        I_f  = out[1].item() * N
        Ri_f = out[2].item() * N
        Rv_f = out[3].item() * N

    nw = waves[wave_idx + 1]
    nw["S0"]  = S_f
    nw["I0"]  = I_f
    nw["Ri0"] = Ri_f
    nw["Rv0"] = Rv_f

    print(f"\nChained ICs → Wave {wave_idx + 2}:")
    print(f"  S={S_f/N*100:.3f}%  I={I_f:.1f}  "
          f"Ri={Ri_f/N*100:.4f}%  Rv={Rv_f/N*100:.4f}%")

    T_nw  = nw["T"]
    mtype = nw["type"]
    if mtype == "SIRS":
        S_e, I_e, Ri_e = reconstruct_sirs(
            nw["inc"], N, I_f, gamma, xi_i, T_nw, S_f, I_f, Ri_f)
        nw["S_est"]  = S_e
        nw["I_est"]  = I_e
        nw["Ri_est"] = Ri_e
        nw["Rv_est"] = np.zeros(T_nw)
    else:
        S_e, I_e, Ri_e, Rv_e = reconstruct_sirvs(
            nw["inc"], nw["vax"], N, I_f, gamma, xi_i, xi_v, epsilon, T_nw,
            S_f, I_f, Ri_f, Rv_f)
        nw["S_est"]  = S_e
        nw["I_est"]  = I_e
        nw["Ri_est"] = Ri_e
        nw["Rv_est"] = Rv_e


# -- shared architecture used by SIRS and SIRVS ---------------
    y = max(y, eps)
    return float(np.log(np.exp(y) - 1 + eps))
#inverse softplus, finding an input that would give the desired outcome

class BetaNet(nn.Module):
    def __init__(self, hidden=64, n_layers=5, beta0=0.2, beta_max=0.5):
        super().__init__()
        self.beta_max = float(beta_max)   # FIX A
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].bias.data.fill_(_inv_sp(beta0))

    def forward(self, t):
        return nn.functional.softplus(self.net(t)).clamp(max=self.beta_max)


# -- SIRS PINN---------------
class SIRSPINN(nn.Module):

    def __init__(self, gamma, xi_i, N, I0, S0, Ri0,
                 hidden=96, n_layers=5, beta_max=0.5):
        super().__init__()
        self.N_pop = float(N); self.I0 = float(I0)
        self.S0 = float(S0);   self.Ri0 = float(Ri0)
        self.gamma = float(gamma); self.xi_i = float(xi_i)

        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 3)]
        self.net      = nn.Sequential(*layers)
        self.beta_net = BetaNet(beta_max=beta_max)

        #initialise output bias so softmax(bias) ≈ [S0, I0, Ri0]
        s0 = float(S0/N); i0 = max(float(I0/N), 1e-6); ri0 = max(float(Ri0/N), 1e-6)
        total = s0 + i0 + ri0
        s0 /= total; i0 /= total; ri0 /= total   # normalise to sum=1
        with torch.no_grad():
            self.net[-1].bias.data = torch.tensor(
                [float(np.log(s0)), float(np.log(i0)), float(np.log(ri0))],
                dtype=torch.float32)

    def forward(self, t):
        #softmax enforces S+I+Ri = 1 everywhere
        raw = self.net(t)
        out = torch.softmax(raw, dim=-1)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def beta(self, t): return self.beta_net(t)


# -- SIRVS PINN ---------------
class SIRVSPINN(nn.Module):

    def __init__(self, gamma, xi_i, xi_v, epsilon,
                 N, I0, S0, Ri0, Rv0,
                 hidden=96, n_layers=5, beta_max=0.5):
        super().__init__()
        self.N_pop = float(N); self.I0 = float(I0)
        self.S0 = float(S0); self.Ri0 = float(Ri0); self.Rv0 = float(Rv0)
        self.gamma = float(gamma); self.xi_i = float(xi_i)
        self.xi_v  = float(xi_v);  self.epsilon = float(epsilon)

        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 4)]
        self.net      = nn.Sequential(*layers)
        self.beta_net = BetaNet(beta_max=beta_max)

        #iitialise bias so softmax(bias) ≈ [S0, I0, Ri0, Rv0]
        s0  = float(S0/N);  i0  = max(float(I0/N),  1e-6)
        ri0 = max(float(Ri0/N), 1e-6); rv0 = max(float(Rv0/N), 1e-6)
        total = s0 + i0 + ri0 + rv0
        s0 /= total; i0 /= total; ri0 /= total; rv0 /= total
        with torch.no_grad():
            self.net[-1].bias.data = torch.tensor(
                [float(np.log(s0)), float(np.log(i0)),
                 float(np.log(ri0)), float(np.log(rv0))],
                dtype=torch.float32)

    def forward(self, t):
        #softmax enforces S+I+Ri+Rv = 1 everywhere
        raw = self.net(t)
        out = torch.softmax(raw, dim=-1)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]

    def beta(self, t): return self.beta_net(t)


# --loss functions ---------------
def _phys_floor(x):
    return x.detach().clamp(min=1e-3)

def sirs_loss_data(model, t_obs, i_t, ri_t, inc_t):
    s, i, ri  = model(t_obs)
    pred_inc  = (model.beta(t_obs) * s * i).squeeze()
    return (torch.mean((i.squeeze()  - i_t)  ** 2) +
            torch.mean((ri.squeeze() - ri_t) ** 2) +
            torch.mean((pred_inc     - inc_t) ** 2))


def sirs_loss_physics(model, t_phys, T_days):
    t = t_phys.clone().requires_grad_(True)
    s, i, ri = model(t)
    b = model.beta(t); g = model.gamma; xi = model.xi_i

    dsdt  = torch.autograd.grad(s,  t, torch.ones_like(s),  create_graph=True)[0] / T_days
    didt  = torch.autograd.grad(i,  t, torch.ones_like(i),  create_graph=True)[0] / T_days
    dridt = torch.autograd.grad(ri, t, torch.ones_like(ri), create_graph=True)[0] / T_days

    f_s  = dsdt  + b*s*i - xi*ri
    f_i  = didt  - b*s*i + g*i
    f_ri = dridt - g*i   + xi*ri

    return torch.mean((f_s/_phys_floor(s))**2 +
                      (f_i/_phys_floor(i))**2 +
                      (f_ri/_phys_floor(ri))**2)


def sirvs_loss_data(model, t_obs, i_t, ri_t, rv_t, inc_t):
    s, i, ri, rv = model(t_obs)
    pred_inc = (model.beta(t_obs) * s * i).squeeze()
    return (torch.mean((i.squeeze()  - i_t)  ** 2) +
            torch.mean((ri.squeeze() - ri_t) ** 2) +
            torch.mean((rv.squeeze() - rv_t) ** 2) +
            torch.mean((pred_inc     - inc_t) ** 2))


def sirvs_loss_physics(model, t_phys, vax_t, T_days):
    t = t_phys.clone().requires_grad_(True)
    s, i, ri, rv = model(t)
    b    = model.beta(t); g = model.gamma
    xi_i = model.xi_i; xi_v = model.xi_v; eps = model.epsilon
    vt   = vax_t.unsqueeze(1)

    dsdt  = torch.autograd.grad(s,  t, torch.ones_like(s),  create_graph=True)[0] / T_days
    didt  = torch.autograd.grad(i,  t, torch.ones_like(i),  create_graph=True)[0] / T_days
    dridt = torch.autograd.grad(ri, t, torch.ones_like(ri), create_graph=True)[0] / T_days
    drvdt = torch.autograd.grad(rv, t, torch.ones_like(rv), create_graph=True)[0] / T_days

    f_s  = dsdt  + b*s*i + eps*vt - xi_i*ri - xi_v*rv
    f_i  = didt  - b*s*i + g*i
    f_ri = dridt - g*i   + xi_i*ri
    f_rv = drvdt - eps*vt + xi_v*rv

    return torch.mean((f_s/_phys_floor(s))**2   +
                      (f_i/_phys_floor(i))**2   +
                      (f_ri/_phys_floor(ri))**2 +
                      (f_rv/_phys_floor(rv))**2)


# -- train one wave ---------------
def _scaled_epochs(wave_T, base_epochs, ref_T=160):
    return max(base_epochs, int(base_epochs * wave_T / ref_T))


def train_wave(wave, N, gamma, xi_i, xi_v, epsilon,
               base_epochs, LR, wave_idx, x, beta_max=None):
    T      = wave["T"]
    mtype  = wave["type"]
    n_phys = N_PHYS_PER_WAVE[min(wave_idx, len(N_PHYS_PER_WAVE)-1)]
    w_data = x
    w_phys = 1.0 - x

    if beta_max is None:
        bmax = BETA_MAX_PER_WAVE[min(wave_idx, len(BETA_MAX_PER_WAVE)-1)]
    else:
        bmax = float(beta_max)

    EPOCHS = _scaled_epochs(T, base_epochs)

    t_obs_np = np.arange(T) / (T-1)
    t_obs    = torch.tensor(t_obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    t_phys   = torch.linspace(0, 1, n_phys, device=DEVICE).unsqueeze(1)

    inc_t = torch.tensor(wave["inc"]    / N, dtype=torch.float32, device=DEVICE)
    i_t   = torch.tensor(wave["I_est"]  / N, dtype=torch.float32, device=DEVICE)
    ri_t  = torch.tensor(wave["Ri_est"] / N, dtype=torch.float32, device=DEVICE)

    label = wave["label"].replace("\n", " ")

    if mtype == "SIRS":
        model = SIRSPINN(gamma, xi_i, N,
                         wave["I0"], wave["S0"], wave["Ri0"],
                         beta_max=bmax).to(DEVICE)
        def step(mdl):
            l_d = sirs_loss_data(mdl, t_obs, i_t, ri_t, inc_t)
            l_p = sirs_loss_physics(mdl, t_phys, float(T-1))
            return l_d, l_p

    else:  # SIRVS
        rv_t  = torch.tensor(wave["Rv_est"] / N, dtype=torch.float32, device=DEVICE)
        vax_phys_np = np.interp(np.linspace(0, 1, n_phys) * (T-1),
                                np.arange(T), wave["vax"] / N)
        vax_t = torch.tensor(vax_phys_np, dtype=torch.float32, device=DEVICE)

        model = SIRVSPINN(gamma, xi_i, xi_v, epsilon, N,
                          wave["I0"], wave["S0"], wave["Ri0"], wave["Rv0"],
                          beta_max=bmax).to(DEVICE)
        def step(mdl):
            l_d = sirvs_loss_data(mdl, t_obs, i_t, ri_t, rv_t, inc_t)
            l_p = sirvs_loss_physics(mdl, t_phys, vax_t, float(T-1))
            return l_d, l_p

    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=500, factor=0.5, min_lr=1e-5)

    print(f"\n── {label} [{mtype}]  ({T}d)  x={w_data:.2f}  "
          f"N_PHYS={n_phys}  EPOCHS={EPOCHS}  β_max={bmax} ──")
    print(f"  ICs: S0={wave['S0']/N*100:.3f}%  I0={wave['I0']:.1f}  "
          f"Ri0={wave['Ri0']/N*100:.4f}%  Rv0={wave['Rv0']/N*100:.4f}%")
    print(f"  Loss = {w_data:.2f}·L_data + {w_phys:.2f}·L_physics")
    print(f"{'Epoch':>6}  {'Total':>10}  {'Data':>10}  {'Physics':>10}  {'D/P':>8}")
    print("-" * 54)

    for epoch in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad()
        l_d, l_p = step(model)
        loss = w_data * l_d + w_phys * l_p
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(loss)

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            ratio = l_d.item() / (l_p.item() + 1e-12)
            print(f"{epoch:>6}  {loss.item():>10.3e}  {l_d.item():>10.3e}  "
                  f"{l_p.item():>10.3e}  {ratio:>8.1f}")

    return model


# -- evaluate ---------------
def evaluate_wave(model, wave, n=400):
    model.eval(); N = model.N_pop; T = wave["T"]
    t_e = torch.linspace(0, 1, n, device=DEVICE).unsqueeze(1)
    with torch.no_grad():
        out  = model(t_e)
        beta = model.beta(t_e).cpu().numpy().squeeze()
    days = np.linspace(0, T-1, n)
    if wave["type"] == "SIRS":
        s, i, ri = [x.cpu().numpy().squeeze()*N for x in out]
        rv = np.zeros(n)
    else:
        s, i, ri, rv = [x.cpu().numpy().squeeze()*N for x in out]
    return days, s, i, ri, rv, beta


# -- forward ODE validation---------------
def ode_validate(wave, model, N, gamma, xi_i, xi_v, epsilon):
    T   = wave["T"]
    vax = wave["vax"]
    model.eval()
    t_d = torch.linspace(0, 1, T, device=DEVICE).unsqueeze(1)
    with torch.no_grad():
        beta_d = model.beta(t_d).cpu().numpy().squeeze()

    s  = wave["S0"]  / N; i  = wave["I0"]  / N
    ri = wave["Ri0"] / N; rv = wave["Rv0"] / N
    ss, is_, ris, rvs = [s], [i], [ri], [rv]

    for t in range(T-1):
        b0 = float(beta_d[t]); b1 = float(beta_d[min(t+1, T-1)])
        v0 = float(vax[t]/N);  v1 = float(vax[min(t+1, T-1)]/N)

        if wave["type"] == "SIRS":
            def f(s_, i_, ri_, rv_, b, v):
                return (-b*s_*i_ + xi_i*ri_,
                         b*s_*i_ - gamma*i_,
                         gamma*i_ - xi_i*ri_, 0.0)
        else:
            def f(s_, i_, ri_, rv_, b, v):
                return (-b*s_*i_ - epsilon*v + xi_i*ri_ + xi_v*rv_,
                         b*s_*i_ - gamma*i_,
                         gamma*i_ - xi_i*ri_,
                         epsilon*v - xi_v*rv_)

        k1 = f(s, i, ri, rv, b0, v0)
        k2 = f(s+.5*k1[0], i+.5*k1[1], ri+.5*k1[2], rv+.5*k1[3], .5*(b0+b1), .5*(v0+v1))
        k3 = f(s+.5*k2[0], i+.5*k2[1], ri+.5*k2[2], rv+.5*k2[3], .5*(b0+b1), .5*(v0+v1))
        k4 = f(s+k3[0], i+k3[1], ri+k3[2], rv+k3[3], b1, v1)

        s  += (k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
        i  += (k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
        ri += (k1[2]+2*k2[2]+2*k3[2]+k4[2])/6
        rv += (k1[3]+2*k2[3]+2*k3[3]+k4[3])/6

        s=max(s,0); i=max(i,0); ri=max(ri,0); rv=max(rv,0)
        tot = s+i+ri+rv
        if tot > 1e-9: s/=tot; i/=tot; ri/=tot; rv/=tot

        ss.append(s); is_.append(i); ris.append(ri); rvs.append(rv)

    S_o  = np.array(ss)*N;  I_o  = np.array(is_)*N
    Ri_o = np.array(ris)*N; Rv_o = np.array(rvs)*N
    inc_o = beta_d * np.array(ss) * np.array(is_) * N
    return S_o, I_o, Ri_o, Rv_o, inc_o, beta_d


# -- plotting -------------------
def _fmt_ax(ax, loc_interval=3):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=loc_interval))
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.2)
#axis formatting

def plot_combined(waves, models, N, gamma):
    all_d, all_b = [], []
    all_S, all_I, all_Ri, all_Rv, all_inc = [], [], [], [], []
    all_od, all_oi = [], []

    for wave, model in zip(waves, models):
        days_f, S_h, I_h, Ri_h, Rv_h, beta_a = evaluate_wave(model, wave)
        ws = wave["dates"][0]
        df = [ws + timedelta(days=float(d)) for d in days_f]
        all_d.extend(df); all_b.extend(beta_a.tolist())
        all_S.extend((S_h/N*100).tolist())
        all_I.extend((I_h/N*100).tolist())
        all_Ri.extend((Ri_h/N*100).tolist())
        all_Rv.extend((Rv_h/N*100).tolist())
        all_inc.extend((beta_a*(S_h/N)*(I_h/N)*N).tolist())
        all_od.extend(wave["dates"]); all_oi.extend(wave["inc"].tolist())

    ba = np.array(all_b)
    S_arr = np.array(all_S) / 100
    Re = (ba / gamma) * S_arr
    bd = [w["dates"][0] for w in waves[1:]]

    fig = plt.figure(figsize=(16, 14))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.46, wspace=0.30)

    def vl(ax):
        for b in bd: ax.axvline(b, color="gray", lw=0.8, ls="--", alpha=0.5)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(all_d, all_S,  color="blue", lw=2, label="S")
    ax1.plot(all_d, all_I,  color="red", lw=2, label="I")
    ax1.plot(all_d, all_Ri, color="green", lw=2, label="Rᵢ (infection)")
    ax1.plot(all_d, all_Rv, color="purple", lw=2, label="Rᵥ (vaccine)")
    ax1.bar(all_od, np.array(all_oi)/N*100, alpha=0.22,
            color="red", label="Daily cases")
    ax1.axvline(VAX_START, color="orange", lw=1.5, ls="-.",
                label=f"Vaccination start ({VAX_START})")
    vl(ax1)
    ax1.set_ylabel("Population (%)", fontsize=13)
    ax1.set_title(f"SIRS→SIRVS PINN  |  γ={gamma}  ξᵢ={XI_I:.4f}  "
                  f"ξᵥ={XI_V:.4f}  ε={EPSILON}", fontsize=13)
    ax1.legend(ncol=5, fontsize=11, framealpha=0.5)
    ax1.tick_params(axis="both", labelsize=11)
    _fmt_ax(ax1)

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(all_d, ba, color="purple", lw=2.5, label="Learned β(t)")
    ax2.fill_between(all_d, ba, alpha=0.12, color="purple")
    ax2.axhline(gamma, color="red", lw=1.2, ls=":",
                label=f"R=1 threshold (β=γ={gamma})")
    ax2.axvline(VAX_START, color="orange", lw=1.5, ls="-.",
                label="Vaccination start")
    vl(ax2)
    ax2.set_ylabel("β(t)", fontsize=13)
    ax2.set_title("Learned β(t)  [SIRS → SIRVS at vaccination start]", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.tick_params(axis="both", labelsize=11)
    _fmt_ax(ax2)
    ax2.set_ylim(0, None)

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.bar(all_od, all_oi, alpha=0.30, color="red", label="Observed")
    ax3.plot(all_d, all_inc, color="blue", lw=2, label="PINN prediction")
    ax3.axvline(VAX_START, color="orange", lw=1.5, ls="-.")
    vl(ax3)
    ax3.set_xlabel("Date", fontsize=13)
    ax3.set_ylabel("Daily new cases", fontsize=13)
    ax3.set_title("Daily infection cases", fontsize=13)
    ax3.legend(fontsize=11)
    ax3.tick_params(axis="both", labelsize=11)
    _fmt_ax(ax3)

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(all_d, Re, color="red", lw=2, label="Re(t)=β(t)S(t)/γN")
    ax4.axhline(1.0, color="gray", lw=1.2, ls="--", label="Re=1")
    ax4.fill_between(all_d, Re, 1, where=Re>=1, alpha=0.15, color="red",
                     label="Growing")
    ax4.fill_between(all_d, Re, 1, where=Re<1,  alpha=0.15, color="green",
                     label="Shrinking")
    ax4.axvline(VAX_START, color="orange", lw=1.5, ls="-.")
    vl(ax4)
    ax4.set_xlabel("Date", fontsize=13)
    ax4.set_ylabel("Re(t)", fontsize=13)
    ax4.set_title("Effective Re(t)", fontsize=13)
    ax4.legend(fontsize=11)
    ax4.tick_params(axis="both", labelsize=11)
    _fmt_ax(ax4)

    path = os.path.join(OUT_DIR, "sirs_sirvs_combined.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Combined plot → {path}")
    plt.show()


def compute_npi_impact(all_d, Re, NPI_EVENTS, wave_boundaries, window_days=21,
                       lag_days=7, boundary_warn_days=30):

    import csv
    all_d_arr  = np.array([d.toordinal() for d in all_d])
    bd_ords    = [d.toordinal() for d in wave_boundaries]

    def near_boundary(d):
        return any(abs(d.toordinal() - b) <= boundary_warn_days
                   for b in bd_ords)

    rows = []
    print(f"\n{'NPI Event':<32} {'Date':<12} {'Dir':<8} {'Re pre':>8} {'Re post':>8} {'ΔRe':>8} {'Δ%':>8} {'Note'}")
    print("─" * 92)

    for d, lbl, dr in NPI_EVENTS:
        d_ord = d.toordinal()
        pre_mask = (all_d_arr >= d_ord - window_days) & (all_d_arr < d_ord)
        post_mask = (all_d_arr >= d_ord + lag_days) & (all_d_arr < d_ord + lag_days + window_days)

        if pre_mask.sum() == 0 or post_mask.sum() == 0:
            continue

        re_pre  = Re[pre_mask].mean()
        re_post = Re[post_mask].mean()
        delta   = re_post - re_pre
        pct     = delta / re_pre * 100
        warn    = "* boundary artefact" if near_boundary(d) else ""

        rows.append({
            "event":      lbl,
            "date":       d.isoformat(),
            "direction":  dr,
            "Re_pre":     round(re_pre,  3),
            "Re_post":    round(re_post, 3),
            "delta_Re":   round(delta,   3),
            "delta_pct":  round(pct,     3),
            "note":       warn,
        })
        print(f"{lbl:<32} {d.isoformat():<12} {dr:<8} "
              f"{re_pre:>8.3f} {re_post:>8.3f} "
              f"{delta:>+8.3f} {pct:>+7.1f}%  {warn}")

    path = os.path.join(OUT_DIR, "npi_impact.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["event","date","direction", "Re_pre","Re_post", "delta_Re","delta_pct","note"])
        w.writeheader(); w.writerows(rows)
    print(f"\nNPI impact table → {path}")
    print(
        f"(Window = {window_days} days   Lag = {lag_days} days   * = within {boundary_warn_days} days of wave boundary)")
    print("\n── Narrative interpretation ─────────────────────────────────────")
    print("Note: overlapping interventions and epidemic trends mean precise")
    print("attribution of individual NPI effects is not possible.\n")
    for row in rows:
        d = row["delta_Re"]
        pct = row["delta_pct"]
        dr = row["direction"]
        lbl = row["event"]
        warn = row["note"]
        trend = "decreased" if d < 0 else "increased"
        expected = ((dr == "tighten" and d < 0) or
                    (dr == "ease" and d > 0) or
                    (dr == "variant" and d > 0))
        consistent = "consistent" if expected else "inconsistent"
        boundary = f"  [WARNING: {warn}]" if warn else ""
        print(f"  {lbl} ({row['date']}, {dr}): "
              f"Re {trend} by {abs(pct):.1f}% — {consistent} with expected direction.{boundary}")

    return rows


def plot_npi_overlay(waves, models, N, gamma):
    NPI_EVENTS = [

        (date(2020,  9, 14), "Rule of Six",            "tighten"),
        (date(2020, 10, 14), "Three-tier system",      "tighten"),
        (date(2020, 11,  5), "Lockdown 2 begins",      "tighten"),
        (date(2020, 12,  2), "Lockdown 2 ends",        "ease"),
        (date(2020, 12,  8), "Alpha variant",          "variant"), #check date
        (date(2021,  1,  6), "Lockdown 3 begins",      "tighten"),
        (date(2021,  3,  8), "Step 1 (schools)",       "ease"),
        (date(2021,  4, 12), "Step 2 (outdoor hosp.)", "ease"),
        (date(2021,  5, 17), "Step 3 (indoor hosp.)",  "ease"),
        (date(2021,  5, 28), "Delta variant",          "variant"),
        (date(2021,  7, 19), "Freedom Day",            "ease"), #easing?
        (date(2021, 12,  1), "Omicron",                "variant"),
        (date(2021, 12,  8), "Plan B begins",          "tighten"),
        (date(2022,  1, 27), "Plan B ends",            "ease"),
        (date(2022,  2,  7), "Omicron subvariant",     "variant"),
        (date(2022,  2, 24), "All restrictions lifted", "ease"),
    ]

    all_d, all_b, all_S_frac = [], [], []
    all_od, all_oi = [], []
    for wave, model in zip(waves, models):
        days_f, S_h, I_h, _, _, beta_a = evaluate_wave(model, wave)
        ws = wave["dates"][0]
        df = [ws + timedelta(days=float(d)) for d in days_f]
        all_S_frac.extend((S_h / N).tolist())
        all_d.extend(df)
        all_b.extend(beta_a.tolist())
        all_od.extend(wave["dates"])
        all_oi.extend(wave["inc"].tolist())

    ba = np.array(all_b)
    Re = (ba / gamma) * np.array(all_S_frac)
    bd = [w["dates"][0] for w in waves[1:]]

    d_min = all_d[0]; d_max = all_d[-1]
    npis  = [(d, lbl, dr) for d, lbl, dr in NPI_EVENTS
             if d_min <= d <= d_max]

    compute_npi_impact(all_d, Re, npis, wave_boundaries=bd)

    col_tighten = "green"
    col_ease    = "red"
    col_variant = "black"
    col_wave    = "gray"

    fig, axes = plt.subplots(3, 1, figsize=(22, 18),
                             gridspec_kw={"height_ratios": [1, 2, 2]})

    def get_col(dr):
        if dr == "tighten": return col_tighten
        if dr == "ease":    return col_ease
        return col_variant

    def add_npis(ax):
        for d, lbl, dr in npis:
            ax.axvline(d, color=get_col(dr), lw=2.0, ls="-", alpha=0.9, zorder=3)
        for b in bd:
            ax.axvline(b, color=col_wave, lw=1.2, ls="--", alpha=0.6)

    def add_npi_labels(ax, y_pos):
        for d, lbl, dr in npis:
            ax.text(d, y_pos, lbl,
                    rotation=90, va="top", ha="right",
                    fontsize=13, color=get_col(dr), alpha=0.95,
                    fontweight="bold", zorder=4)

    # -- top plot, daily incidence ---------------
    ax0 = axes[0]
    ax0.bar(all_od, all_oi, alpha=0.35, color="red", label="Observed cases")
    add_npis(ax0)
    ax0.set_ylabel("Daily cases", fontsize=16)
    ax0.set_title("Observed daily incidence", fontsize=17, fontweight="bold")
    ax0.legend(fontsize=14)
    ax0.tick_params(axis="both", labelsize=14)
    _fmt_ax(ax0, loc_interval=1)

    # -- centre, β(t) ---------------
    ax1 = axes[1]
    ax1.plot(all_d, ba, color="purple", lw=2.5, label="Learned β(t)", zorder=2)
    ax1.fill_between(all_d, ba, alpha=0.12, color="purple")
    ax1.axhline(gamma, color="gray", lw=1.2, ls=":",
                label=f"R0=1 threshold (β=γ={gamma})")
    add_npis(ax1)
    add_npi_labels(ax1, ba.max() * 0.97)
    ax1.set_ylabel("β(t)", fontsize=16)
    ax1.set_title("Learned transmission rate β(t)", fontsize=17, fontweight="bold")
    ax1.legend(fontsize=14)
    ax1.set_ylim(0, None)
    ax1.tick_params(axis="both", labelsize=14)
    _fmt_ax(ax1, loc_interval=1)

    # -- bottom, Re(t) ---------------
    ax2 = axes[2]
    ax2.plot(all_d, Re, color="red", lw=2.5,
             label="Re(t) = β(t)S(t)/γN", zorder=2)
    ax2.fill_between(all_d, Re, 1,
                     where=Re >= 1, alpha=0.15, color="red",
                     label="Growing (Re>1)")
    ax2.fill_between(all_d, Re, 1,
                     where=Re < 1, alpha=0.15, color="green",
                     label="Shrinking (Re<1)")
    ax2.axhline(1.0, color="gray", lw=1.2, ls="--", label="Re=1")
    add_npis(ax2)
    add_npi_labels(ax2, Re.max() * 0.97)
    ax2.set_ylabel("Re(t)", fontsize=16)
    ax2.set_xlabel("Date", fontsize=16)
    ax2.set_title("Effective reproduction number Re(t)", fontsize=17, fontweight="bold")
    ax2.legend(fontsize=14)
    ax2.set_ylim(0, None)
    ax2.tick_params(axis="both", labelsize=14)
    _fmt_ax(ax2, loc_interval=1)

    # -- bottom legend---------------
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color=col_tighten, lw=2.5,
               label="Restriction introduced"),
        Line2D([0], [0], color=col_ease,    lw=2.5,
               label="Restriction eased / lifted"),
        Line2D([0], [0], color=col_variant, lw=2.5,
               label="Variant arrival"),
        Line2D([0], [0], color=col_wave,    lw=1.5, ls="--",
               label="Wave boundary"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=4,
               fontsize=14, framealpha=0.8,
               bbox_to_anchor=(0.5, 0.02))

    plt.subplots_adjust(left=0.07, bottom=0.10, right=0.983,
                        top=0.96, wspace=0.2, hspace=0.42)
    path = os.path.join(OUT_DIR, "npi_overlay.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"NPI overlay → {path}")
    plt.show()

def plot_ode_validation(waves, models, N, gamma, xi_i, xi_v, epsilon):
    fmt1 = mdates.DateFormatter("%b %Y")
    loc1 = mdates.MonthLocator(interval=1)

    for wave, model in zip(waves, models):
        lc    = wave["label"].replace("\n", " ")
        mtype = wave["type"]
        S_o, I_o, Ri_o, Rv_o, inc_o, beta_d = ode_validate(
            wave, model, N, gamma, xi_i, xi_v, epsilon)
        _, S_p, I_p, Ri_p, Rv_p, _ = evaluate_wave(model, wave, n=wave["T"])
        inc_p = beta_d * (S_p/N) * (I_p/N) * N
        ws    = wave["dates"][0]
        dd    = [ws + timedelta(days=i) for i in range(wave["T"])]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"ODE validation [{mtype}] — {lc}",
                     fontsize=15, fontweight="bold")

        def ax_fmt(ax):
            ax.xaxis.set_major_formatter(fmt1)
            ax.xaxis.set_major_locator(loc1)
            ax.tick_params(axis="both", labelsize=11)
            ax.grid(True, alpha=0.2)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        for ax, pinn, ode, col, title, ylab in [
            (axes[0,0], S_p,  S_o,  "blue", "S(t)",  "Population (%)"),
            (axes[0,1], I_p,  I_o,  "red", "I(t)",  "Population (%)"),
            (axes[0,2], Ri_p, Ri_o, "green", "Rᵢ(t)", "Population (%)"),
            (axes[1,0], Rv_p, Rv_o, "purple", "Rᵥ(t)", "Population (%)"),
        ]:
            ax.plot(dd, pinn/N*100, color=col, lw=2, label="PINN")
            ax.plot(dd, ode/N*100,  color=col, lw=1.5, ls="--",
                    alpha=0.8, label="ODE (RK4)")
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_ylabel(ylab, fontsize=12)
            ax.legend(fontsize=11)
            ax_fmt(ax)

        ax = axes[1, 1]
        ax.bar(wave["dates"], wave["inc"], alpha=0.28, color="red",
               label="Observed")
        ax.plot(dd, inc_p, color="blue", lw=2, label="PINN")
        ax.plot(dd, inc_o, color="green", lw=1.8, ls="--",
                label="ODE (RK4)")
        ax.set_title("Incidence", fontsize=13, fontweight="bold")
        ax.set_ylabel("Daily cases", fontsize=12)
        ax.legend(fontsize=11)
        ax_fmt(ax)

        ax = axes[1, 2]
        ax.bar(wave["dates"], wave["vax"], alpha=0.5, color="orange",
               label="Daily 1st doses")
        ax.set_title("Vaccination v(t)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Daily doses", fontsize=12)
        ax.legend(fontsize=11)
        ax_fmt(ax)

        plt.tight_layout()
        safe = (lc.replace(" ","_").replace("/","-")
                  .replace("(","").replace(")",""))
        path = os.path.join(OUT_DIR, f"ode_val_{safe}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"ODE validation → {path}")
        plt.show()

def plot_re_vs_ukhsa(waves, models, N, gamma, ukhsa_path):
    import pandas as pd

    # -- load UKHSA R estimates ---------------
    df = pd.read_excel(ukhsa_path, engine="odf",
                       sheet_name="Table1_-_R", header=None)

    raw = df.iloc[9:122, [1, 4, 5]].copy()
    raw.columns = ["date", "R_lower", "R_upper"]
    raw["date"] = pd.to_datetime(raw["date"], format="mixed", errors="coerce")
    raw["R_lower"] = pd.to_numeric(raw["R_lower"], errors="coerce")
    raw["R_upper"] = pd.to_numeric(raw["R_upper"], errors="coerce")
    raw = raw.dropna(subset=["date", "R_lower"]).sort_values("date").reset_index(drop=True)
    raw["R_upper"] = raw["R_upper"].fillna(raw["R_lower"])
    raw["R_mid"] = (raw["R_lower"] + raw["R_upper"]) / 2
    print(f"UKHSA R data: {len(raw)} estimates  "
          f"({raw['date'].iloc[0].date()} → {raw['date'].iloc[-1].date()})")



    # -- collect Re(t) across all waves ---------------
    all_dates, all_re = [], []
    all_obs_d, all_inc = [], []

    for wave, model in zip(waves, models):
        days_f, S_h, I_h, _, _, beta_a = evaluate_wave(model, wave)
        ws = wave["dates"][0]
        df_dates = [ws + timedelta(days=float(d)) for d in days_f]
        all_dates.extend(df_dates)
        all_re.extend(((beta_a / gamma) * (S_h / N)).tolist())
        all_obs_d.extend(wave["dates"])
        all_inc.extend(wave["inc"].tolist())

    all_re = np.array(all_re)

    # -- find UKHSA within model date range---------------
    d_min, d_max = all_dates[0], all_dates[-1]
    ukhsa = raw[(raw["date"] >= pd.Timestamp(d_min)) &
            (raw["date"] <= pd.Timestamp(d_max))].reset_index(drop=True)

    # -- load TrackingR estimates ---------------
    tracking = pd.read_csv("crondonm.csv")
    tracking["date"] = pd.to_datetime(tracking["date"], dayfirst=True)
    tracking = tracking.rename(columns={
        "r": "R_mid",
        "ci_95_u": "R_upper",
        "ci_95_l": "R_lower"})
    tracking = tracking[["date", "R_mid", "R_upper", "R_lower"]]
    tracking["R_lower"] = pd.to_numeric(tracking["R_lower"], errors="coerce")
    tracking["R_upper"] = pd.to_numeric(tracking["R_upper"], errors="coerce")
    tracking["R_mid"] = pd.to_numeric(tracking["R_mid"], errors="coerce")
    tracking = tracking.dropna().sort_values("date").reset_index(drop=True)
    tracking = tracking[(tracking["date"] >= pd.Timestamp(d_min)) &
                        (tracking["date"] <= pd.Timestamp(d_max))].reset_index(drop=True)
    print(f"TrackingR data: {len(tracking)} estimates  "
          f"({tracking['date'].iloc[0].date()} → {tracking['date'].iloc[-1].date()})")

    # -- wave boundaries and NPI markers ---------------
    wave_boundaries = [w["dates"][0] for w in waves[1:]]
    NPI_EVENTS = [
        (date(2020, 11,  5), "Lockdown 2",             "tighten"),
        (date(2021,  1,  6), "Lockdown 3",             "tighten"),
        (date(2021,  7, 19), "Freedom Day",            "ease"),
        (date(2021, 12,  8), "Plan B",                 "tighten"),
        (date(2022,  2, 24), "All restrictions lifted","ease"),
    ]
    col_tight = "green"
    col_ease  = "red"
    col_wave  = "gray"

    # -- figure ---------------
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 5))
    fig.suptitle("PINN Re(t) vs UKHSA/SPI-M Consensus R and Arroyo-Marioli et al. TrackingR — England COVID-19",
                 fontsize=13, fontweight="bold")

    # UKHSA shaded band
    ax1.fill_between(ukhsa["date"], ukhsa["R_lower"], ukhsa["R_upper"],
                     color="amber orange", alpha=0.25, zorder=1,
                     label="UKHSA 90% CI band")
    # upper and lower as dashed lines
    ax1.plot(ukhsa["date"], ukhsa["R_upper"], color="amber orange",
             lw=1.5, ls="--", zorder=2)
    ax1.plot(ukhsa["date"], ukhsa["R_lower"], color="amber orange",
             lw=1.5, ls="--", zorder=2)
    # midpoint dots
    ax1.plot(ukhsa["date"], ukhsa["R_mid"],
             "o", color="dark amber", ms=4, zorder=4,
             label="UKHSA midpoint")

    # TrackingR shaded band + midpoint
    ax1.fill_between(tracking["date"], tracking["R_lower"], tracking["R_upper"],
                     color="mediumpurple", alpha=0.15, zorder=1,
                     label="TrackingR 95% CI (UK)")
    ax1.plot(tracking["date"], tracking["R_mid"],
             color="mediumpurple", lw=1.5, ls="-.", zorder=2,
             label="TrackingR midpoint (UK)")

    # PINN Re(t)
    ax1.plot(all_dates, all_re, color="blue", lw=2.2,
             zorder=3, label="PINN Re(t) = β(t)S(t)/γN")

    # R=1 line
    ax1.axhline(1.0, color="gray", lw=1.2, ls=":")

    #wave boundaries
    for b in wave_boundaries:
        ax1.axvline(b, color=col_wave, lw=1.0, ls="--", alpha=0.5)

    y_top = max(float(all_re.max()), float(ukhsa["R_upper"].max())) * 1.02

    for wave in waves:
        mid_d = wave["dates"][len(wave["dates"]) // 2]
        ax1.text(mid_d, y_top * 1.005,
                 wave["label"].replace("\n", " "),
                 ha="center", va="bottom", fontsize=8.5,
                 color="#444444", style="italic")

    ax1.set_ylim(0, y_top * 1.10)
    ax1.set_ylabel("Re(t)", fontsize=11)
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_title(
        "PINN Re(t) vs UKHSA consensus (England, 90% CI) and TrackingR (UK, 95% CI)",
        fontsize=10)
    ax1.grid(True, alpha=0.18)

    # combined legend
    ax1.legend(fontsize=9, loc="upper left", framealpha=0.7)

    # shared x-axis formatting
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "re_vs_ukhsa.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Re vs UKHSA plot → {path}")
    plt.show()

    # -- alignment table in terminal ---------------
    d_arr = np.array([d.toordinal() for d in all_dates])
    inside = 0
    print(f"\n{'Date':<14} {'UKHSA lo':>9} {'mid':>7} {'hi':>7} "
          f"{'PINN Re':>9} {'In CI?':>7}")
    print("─" * 58)
    for _, row in ukhsa.iterrows():
        idx = np.argmin(np.abs(d_arr - row["date"].toordinal()))
        pv  = all_re[idx]
        ok  = row["R_lower"] <= pv <= row["R_upper"]
        if ok: inside += 1
        print(f"{str(row['date'].date()):<14} "
              f"{row['R_lower']:>9.2f} {row['R_mid']:>7.2f} "
              f"{row['R_upper']:>7.2f} {pv:>9.3f}  "
              f"{'✓' if ok else '✗'}")
    above = sum(1 for _, row in ukhsa.iterrows()
                if all_re[np.argmin(np.abs(d_arr - row["date"].toordinal()))] > row["R_upper"])
    below = sum(1 for _, row in ukhsa.iterrows()
                if all_re[np.argmin(np.abs(d_arr - row["date"].toordinal()))] < row["R_lower"])

    pct = inside / len(ukhsa) * 100
    pct_above = above / len(ukhsa) * 100
    pct_below = below / len(ukhsa) * 100
    print(f"In UKHSA 90% CI:    {inside}/{len(ukhsa)} weeks ({pct:.1f}%)")
    print(f"Above UKHSA 90% CI: {above}/{len(ukhsa)} weeks ({pct_above:.1f}%)")
    print(f"Below UKHSA 90% CI: {below}/{len(ukhsa)} weeks ({pct_below:.1f}%)")

    # TrackingR alignment
    tr_arr = np.array([d.toordinal() for d in tracking["date"]])
    inside_tr = 0
    for _, row in tracking.iterrows():
        idx = np.argmin(np.abs(d_arr - row["date"].toordinal()))
        pv = all_re[idx]
        if row["R_lower"] <= pv <= row["R_upper"]:
            inside_tr += 1
    above_tr = sum(1 for _, row in tracking.iterrows()
                   if all_re[np.argmin(np.abs(d_arr - row["date"].toordinal()))] > row["R_upper"])
    below_tr = sum(1 for _, row in tracking.iterrows()
                   if all_re[np.argmin(np.abs(d_arr - row["date"].toordinal()))] < row["R_lower"])

    pct_tr = inside_tr / len(tracking) * 100
    pct_above_tr = above_tr / len(tracking) * 100
    pct_below_tr = below_tr / len(tracking) * 100
    print(f"In TrackingR 95% CI:    {inside_tr}/{len(tracking)} estimates ({pct_tr:.1f}%)")
    print(f"Above TrackingR 95% CI: {above_tr}/{len(tracking)} estimates ({pct_above_tr:.1f}%)")
    print(f"Below TrackingR 95% CI: {below_tr}/{len(tracking)} estimates ({pct_below_tr:.1f}%)")


# -- entry point ---------------
if __name__ == "__main__":
    args    = parse_args()
    N       = args.N
    GAMMA   = args.gamma
    XI_I    = args.xi_i
    XI_V    = args.xi_v
    EPSILON = args.epsilon
    beta_max_override = args.beta_max

    #resolve per-wave x: single value → apply to all waves
    #multiple values → one per wave (must match number of waves)
    n_waves = len(WAVE_DEFS)
    if len(args.x) == 1:
        x_per_wave = [args.x[0]] * n_waves
    elif len(args.x) == n_waves:
        x_per_wave = args.x
    else:
        raise ValueError(
            f"--x expects 1 value or {n_waves} values (one per wave), "
            f"got {len(args.x)}: {args.x}")

    bmax_desc = (f"{beta_max_override} (all waves)"
                 if beta_max_override is not None
                 else f"{BETA_MAX_PER_WAVE} (per wave)")
    print(f"Per-wave x:  {[f'{v:.2f}' for v in x_per_wave]}")
    print(f"  → each wave: x·L_data + (1-x)·L_physics")
    print(f"β_max: {bmax_desc}")
    print(f"S0_frac={args.S0_frac:.2f}  →  "
          f"prior immunity={(1-args.S0_frac)*100:.1f}% of N")

    cases, vax, dates_all = load_data(args.data, args.vax, args.start)

    waves = split_waves(cases, vax, dates_all, args.wave_start, WAVE_DEFS,
                        N, args.I0, GAMMA, XI_I, XI_V, EPSILON,
                        S0_frac=args.S0_frac)

    # -- model save paths (one .pt file per wave) ---------------
    model_paths = [
        os.path.join(OUT_DIR, f"model_wave{i+1}.pt")
        for i in range(len(waves))]

    models = []
    for i, wave in enumerate(waves):
        if args.load_models and os.path.exists(model_paths[i]):
            # ── Load saved model, skip retraining ─────────────────────────────
            print(f"\nLoading saved model for Wave {i+1} "
                  f"from {model_paths[i]} …")
            bmax = (beta_max_override if beta_max_override is not None
                    else BETA_MAX_PER_WAVE[min(i, len(BETA_MAX_PER_WAVE)-1)])
            if wave["type"] == "SIRS":
                m = SIRSPINN(GAMMA, XI_I, N,
                             wave["I0"], wave["S0"], wave["Ri0"],
                             beta_max=bmax).to(DEVICE)
            else:
                m = SIRVSPINN(GAMMA, XI_I, XI_V, EPSILON, N,
                              wave["I0"], wave["S0"], wave["Ri0"], wave["Rv0"],
                              beta_max=bmax).to(DEVICE)
            m.load_state_dict(torch.load(model_paths[i],
                                         map_location=DEVICE))
            m.eval()
        else:
            # -- train ---------------
            m = train_wave(wave, N, GAMMA, XI_I, XI_V, EPSILON,
                           base_epochs=args.base_epochs, LR=args.lr,
                           wave_idx=i, x=x_per_wave[i],
                           beta_max=beta_max_override)
            torch.save(m.state_dict(), model_paths[i])
            print(f"Model saved → {model_paths[i]}")

        models.append(m)
        chain_wave_ics(waves, models, wave_idx=i, N=N,
                       gamma=GAMMA, xi_i=XI_I, xi_v=XI_V, epsilon=EPSILON)

    plot_combined(waves, models, N, GAMMA)
    plot_ode_validation(waves, models, N, GAMMA, XI_I, XI_V, EPSILON)
    plot_npi_overlay(waves, models, N, GAMMA)
    if args.ukhsa_r:
        plot_re_vs_ukhsa(waves, models, N, GAMMA, args.ukhsa_r)