"""
utilities_classical.py
======================
Classical PINN utilities for NS shock-tube training.
Provides: UnifiedPINN, build_unified, unified_loss, load_ns_data,
          NI, NO, PRESETS, _make_act, _make_weighting, StaticWeighting
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Constants ─────────────────────────────────────────────────────────────────
NI = 7   # input features:  [x, t, p_ratio, mu, rho_L, rho_R, p_R]
NO = 3   # output features: [rho, u, p]

PRESETS = {
    "mlp":            {"hidden": 128, "depth": 4, "activation": "tanh",  "use_physics": False},
    "pinn":           {"hidden": 128, "depth": 4, "activation": "tanh",  "use_physics": True},
    "fourier_pinn":   {"hidden": 128, "depth": 4, "activation": "sin",   "use_physics": True},
    "hardbc_pinn":    {"hidden": 128, "depth": 4, "activation": "tanh",  "use_physics": True},
    "softadapt_pinn": {"hidden": 128, "depth": 4, "activation": "tanh",  "use_physics": True},
    "dynratio_pinn":  {"hidden": 128, "depth": 4, "activation": "tanh",  "use_physics": True},
    "rar_pinn":       {"hidden": 128, "depth": 4, "activation": "tanh",  "use_physics": True},
    "best_classical": {"hidden": 256, "depth": 6, "activation": "tanh",  "use_physics": True},
    "swish_pinn":     {"hidden": 128, "depth": 4, "activation": "swish", "use_physics": True},
    "gelu_pinn":      {"hidden": 128, "depth": 4, "activation": "gelu",  "use_physics": True},
}


# ── Activations ───────────────────────────────────────────────────────────────

class _Sin(nn.Module):
    def forward(self, x): return torch.sin(x)

class _Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

def _make_act(name: str) -> nn.Module:
    return {
        "tanh":  nn.Tanh(),
        "relu":  nn.ReLU(),
        "gelu":  nn.GELU(),
        "sin":   _Sin(),
        "swish": _Swish(),
    }.get(name, nn.Tanh())


# ── Loss weighting ────────────────────────────────────────────────────────────

class StaticWeighting:
    """Fixed 1:1 weighting — no adaptation."""
    def get_lams(self): return {}
    def update(self, epoch, log): pass


class _DynRatioWeighting:
    """Dynamically rebalance data vs physics losses."""
    def __init__(self):
        self._lam_phys = 1.0

    def get_lams(self): return {"lam_phys": self._lam_phys}

    def update(self, epoch, log):
        data  = log.get("data", 1e-8)
        phys  = log.get("total_physics", 1e-8)
        if phys > 1e-10:
            self._lam_phys = float(data / (phys + 1e-10))
            self._lam_phys = max(0.01, min(self._lam_phys, 100.0))


def _make_weighting(mode: str):
    if mode in ("static", "hardbc", "rar"):
        return StaticWeighting()
    if mode in ("dynratio", "softadapt"):
        return _DynRatioWeighting()
    return StaticWeighting()


# ── UnifiedPINN model ─────────────────────────────────────────────────────────

class UnifiedPINN(nn.Module):
    """
    Classical MLP / PINN for NS shock-tube data.
    Input:  [x, t, p_ratio, mu, rho_L, rho_R, p_R]  (NI=7)
    Output: [rho, u, p]                               (NO=3)
    """

    def __init__(
        self,
        hidden:      int  = 128,
        depth:       int  = 4,
        activation:  str  = "tanh",
        use_physics: bool = False,
        loss_mode:   str  = "static",
        model_type:  str  = "mlp",
    ):
        super().__init__()
        self.use_physics = use_physics
        self.model_type  = model_type if not use_physics else "pinn"
        self._weighting  = _make_weighting(loss_mode)

        layers = [nn.Linear(NI, hidden), _make_act(activation)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), _make_act(activation)]
        layers += [nn.Linear(hidden, NO)]
        self.net = nn.Sequential(*layers)

        # placeholders — attached by main()
        self._Xm = self._Xs = self._Ym = self._Ys = None

    def forward(self, x: torch.Tensor,
                y: torch.Tensor = None) -> tuple:
        """
        Forward pass. Returns (pred, loss).

        If y is provided: loss = MSE(pred, y)   (training)
        If y is None:     loss = 0              (inference)

        Consistent interface with TorchPINN and QAPINN.
        """
        pred = self.net(x)
        if y is not None:
            loss = nn.functional.mse_loss(pred, y)
            return pred, loss
        return pred, torch.tensor(0., device=x.device)

    def get_lam_overrides(self):
        return self._weighting.get_lams()

    def update(self, epoch, log):
        self._weighting.update(epoch, log)

    def describe(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"\nUnifiedPINN: use_physics={self.use_physics}  params={n:,}")


def build_unified(preset: str = "mlp") -> UnifiedPINN:
    cfg = PRESETS.get(preset, PRESETS["mlp"])
    return UnifiedPINN(
        hidden      = cfg["hidden"],
        depth       = cfg["depth"],
        activation  = cfg["activation"],
        use_physics = cfg["use_physics"],
        model_type  = "pinn" if cfg["use_physics"] else "mlp",
    )


# ── Unified loss (data + optional physics) ────────────────────────────────────

def unified_loss(model, x, y, epoch, Xm, Xs, p_range, mu_range):
    """
    Compute loss for UnifiedPINN.
    Returns (loss_scalar, log_dict).
    """
    pred, _ = model(x)
    l_data = nn.functional.mse_loss(pred, y)

    log = {"data": float(l_data.detach())}

    if getattr(model, "use_physics", False):
        # Simple physics proxy: encourage smooth predictions
        # (full NS residual requires autograd w.r.t. x,t — expensive)
        l_phys = _ns_residual(model, x, Xm, Xs)
        lams   = model.get_lam_overrides()
        # Start gently: an untrained network has large derivatives, so a full
        # physics weight from epoch 1 can flatten the supervised solution.
        # Static PINNs use 0.01 after a 50-epoch linear warm-up; adaptive
        # presets retain their learned weight but use the same warm-up.
        lam_target = lams.get("lam_phys", 0.01)
        lam = lam_target * min(1.0, float(epoch + 1) / 50.0)
        loss   = l_data + lam * l_phys
        log["total_physics"] = float(l_phys.detach())
        log["lam_phys"]      = lam
    else:
        loss = l_data

    log["total"] = float(loss.detach())
    return loss, log


def _ns_residual(model, x, Xm, Xs, gamma=1.4):
    """Pointwise 1-D compressible Navier--Stokes residual.

    Derivatives are obtained with autograd at each collocation/data point, so
    a shuffled mini-batch may safely contain different times and scenarios.
    Inputs and outputs are converted back to physical units before applying
    the chain rule.  This replaces the old second-difference proxy, which
    incorrectly treated unrelated rows (same x, different scenario/time) as
    spatial neighbours.

    The three primitive-variable equations used are continuity, momentum and
    pressure evolution for a calorically perfect gas with viscous heating.
    """
    xn = x.detach().clone().requires_grad_(True)
    pred_n, _ = model(xn)

    # Model outputs are normalised; physics is evaluated in physical units.
    Ym = model._Ym.to(xn.device)
    Ys = model._Ys.to(xn.device)
    Xmean = Xm.to(xn.device)
    Xstd = Xs.to(xn.device)
    pred = pred_n * Ys + Ym
    rho, vel, pressure = pred[:, 0], pred[:, 1], pred[:, 2]
    mu = xn[:, 3] * Xstd[3] + Xmean[3]

    def grad_phys(value, feature):
        g = torch.autograd.grad(value, xn, torch.ones_like(value),
                                create_graph=True, retain_graph=True)[0]
        return g[:, feature] / Xstd[feature]

    rho_x, rho_t = grad_phys(rho, 0), grad_phys(rho, 1)
    u_x, u_t = grad_phys(vel, 0), grad_phys(vel, 1)
    p_x, p_t = grad_phys(pressure, 0), grad_phys(pressure, 1)
    u_xx = grad_phys(u_x, 0)

    mass_terms = (rho_t, vel * rho_x, rho * u_x)
    mom_terms = (rho * u_t, rho * vel * u_x, p_x, -mu * u_xx)
    pressure_terms = (p_t, vel * p_x, gamma * pressure * u_x,
                      -(gamma - 1.0) * mu * u_x.square())
    residuals = (sum(mass_terms), sum(mom_terms), sum(pressure_terms))

    # Balance equations by the detached RMS magnitude of their constituent
    # terms. This prevents pressure/momentum units from dominating continuity.
    loss = pred_n.new_zeros(())
    for residual, terms in zip(residuals,
                               (mass_terms, mom_terms, pressure_terms)):
        scale2 = sum(term.detach().square().mean() for term in terms)
        loss = loss + residual.square().mean() / (scale2 + 1e-8)
    return loss / len(residuals)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ns_data(
    data_dir:    str,
    n_scenarios: int  = 100,
    t_stride:    int  = 2,
    train_frac:  float = 0.70,
    val_frac:    float = 0.15,
    seed:        int  = 0,
):
    """
    Load NS shock-tube scenarios from data_dir (output of Step1 sweep).

    Each scenario is a .npz file with fields:
        x        (N,)              spatial positions
        t_snaps  (n_snaps,)        snapshot times
        rho      (n_snaps, N)      density field
        u        (n_snaps, N)      velocity field
        p        (n_snaps, N)      pressure field
        p_ratio, mu, rho_L, rho_R, p_R  — scalar parameters

    Returns dict with:
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te  — torch tensors
        Xm, Xs, Ym, Ys                        — normalisation stats (torch)
        feature_names, output_names
        p_range, mu_range
    """
    index_path = os.path.join(data_dir, "index.json")
    with open(index_path) as f:
        index = json.load(f)

    scenarios = index["scenarios"][:n_scenarios]
    print(f"  Loading {len(scenarios)} scenarios from {data_dir} ...")

    rows_X, rows_Y = [], []

    for sc in scenarios:
        fpath = os.path.join(data_dir, sc["filename"])
        d     = np.load(fpath)

        x_pos   = d["x"].astype(np.float32)          # (N,)
        t_snaps = d["t_snaps"].astype(np.float32)    # (n_snaps,)
        rho     = d["rho"].astype(np.float32)         # (n_snaps, N)
        u_arr   = d["u"].astype(np.float32)
        p_arr   = d["p"].astype(np.float32)

        p_ratio = float(d["p_ratio"])
        mu      = float(d["mu"])
        rho_L   = float(d["rho_L"])
        rho_R   = float(d["rho_R"])
        p_R     = float(d["p_R"])

        n_snaps, N = rho.shape
        t_idx = range(0, n_snaps, t_stride)

        for ti in t_idx:
            t_val = t_snaps[ti]
            for xi in range(N):
                # Input:  [x, t, p_ratio, mu, rho_L, rho_R, p_R]
                row_x = [x_pos[xi], t_val, p_ratio, mu, rho_L, rho_R, p_R]
                # Output: [rho, u, p]
                row_y = [rho[ti, xi], u_arr[ti, xi], p_arr[ti, xi]]
                rows_X.append(row_x)
                rows_Y.append(row_y)

    X = np.array(rows_X, dtype=np.float32)
    Y = np.array(rows_Y, dtype=np.float32)
    print(f"  Total samples: {len(X):,}  (features={X.shape[1]}, outputs={Y.shape[1]})")

    # Normalise
    Xm = X.mean(axis=0);  Xs = X.std(axis=0) + 1e-8
    Ym = Y.mean(axis=0);  Ys = Y.std(axis=0) + 1e-8
    Xn = (X - Xm) / Xs
    Yn = (Y - Ym) / Ys

    # Shuffle and split
    rng  = np.random.RandomState(seed)
    perm = rng.permutation(len(Xn))
    Xn, Yn = Xn[perm], Yn[perm]

    n_tr = int(len(Xn) * train_frac)
    n_va = int(len(Xn) * val_frac)

    X_tr, Y_tr = Xn[:n_tr],        Yn[:n_tr]
    X_va, Y_va = Xn[n_tr:n_tr+n_va], Yn[n_tr:n_tr+n_va]
    X_te, Y_te = Xn[n_tr+n_va:],  Yn[n_tr+n_va:]

    print(f"  train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}")

    # Parameter ranges (for physics loss scaling)
    p_ratios = [float(s["p_ratio"]) for s in scenarios]
    mus      = [float(s["mu"])      for s in scenarios]
    p_range  = (min(p_ratios), max(p_ratios))
    mu_range = (min(mus),      max(mus))

    def _t(a): return torch.tensor(a, dtype=torch.float32)

    return {
        "X_tr": _t(X_tr), "Y_tr": _t(Y_tr),
        "X_va": _t(X_va), "Y_va": _t(Y_va),
        "X_te": _t(X_te), "Y_te": _t(Y_te),
        "Xm":   _t(Xm),   "Xs":   _t(Xs),
        "Ym":   _t(Ym),   "Ys":   _t(Ys),
        "feature_names": ["x", "t", "p_ratio", "mu", "rho_L", "rho_R", "p_R"],
        "output_names":  ["rho", "u", "p"],
        "p_range":  p_range,
        "mu_range": mu_range,
    }
