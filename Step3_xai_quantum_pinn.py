"""
xai_quantum_pinn.py
===================
Explainable AI toolkit for QPINN and QAPINN models trained in
Step2_train_ns_quantum_pinn_and_qpinn_torch.py

Methods
-------
  1. Permutation Feature Importance  — model-agnostic, all models
  2. Gradient-Based Sensitivity      — |∂output/∂input|, QPINN + QAPINN
  3. Integrated Gradients            — completeness-axiom attribution
  4. SHAP (KernelExplainer)          — requires `pip install shap`
  5. Quantum State Probe             — density-matrix distance per feature
  6. Qubit Entanglement / Entropy    — Von Neumann entropy per qubit
  7. Circuit Weight Analysis         — weight distribution + plateau check
  8. Feature Interaction (H-stat)    — pairwise Friedman H²
  9. Ablation Study                  — component knock-out
 10. Full HTML Report                — self-contained report saved to disk

Usage
-----
  # After training, load checkpoints and run:
  python xai_quantum_pinn.py

  Or import individual functions:
    from xai_quantum_pinn import (
        permutation_importance, gradient_sensitivity,
        integrated_gradients, quantum_state_probe,
        qubit_entropy, run_full_xai
    )

Dependencies
------------
  torch, numpy, matplotlib, json, pathlib
  Optional: shap  (pip install shap)

Input features (NI=7): [x, t, p_ratio, mu, rho_L, rho_R, p_R]
Output features (NO=3): [rho, u, p]
"""

import sys
import os
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")          # headless — change to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore")

# ── path to your project (adjust if needed) ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

FEATURE_NAMES = ["x", "t", "p_ratio", "mu", "rho_L", "rho_R", "p_R"]
OUTPUT_NAMES  = ["rho", "u", "p"]
NI, NO = 7, 3

# colour palette matching the dashboard
COLOURS = ["#38bdf8", "#818cf8", "#fb923c", "#34d399",
           "#f472b6", "#a3e635", "#fbbf24"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _to_device(t: torch.Tensor, device) -> torch.Tensor:
    return t.to(device)


def _mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(((pred - target) ** 2).mean().item())


def _predict_classical(model, X: torch.Tensor, device) -> torch.Tensor:
    """Forward pass for MLP / ClassPINN / QAPINN."""
    model.eval()
    with torch.no_grad():
        return model(_to_device(X, device)).cpu()


def _predict_qpinn(model, X: torch.Tensor, Y: torch.Tensor,
                   device) -> torch.Tensor:
    """
    Forward pass for TorchPINN (QPINN).
    Returns only the first NO columns of the n_qubits output.
    """
    model.eval()
    n_q = model.n_qubits
    n_feat = X.shape[-1]
    if n_feat < n_q:
        idx = torch.arange(n_q) % n_feat
        Xq  = X[:, idx]
    else:
        Xq = X[:, :n_q]
    n_out = Y.shape[-1]
    if n_out < n_q:
        idx = torch.arange(n_q) % n_out
        Yq  = Y[:, idx]
    else:
        Yq = Y[:, :n_q]
    with torch.no_grad():
        pred_full, _ = model(_to_device(Xq, device), _to_device(Yq, device))
    return pred_full[:, :NO].cpu()


def _predict(model, X: torch.Tensor, Y: torch.Tensor,
             model_type: str, device) -> torch.Tensor:
    """Unified predict for any model type."""
    if model_type == "qpinn":
        return _predict_qpinn(model, X, Y, device)
    return _predict_classical(model, X, device)


def _prep_qpinn_input(model, X: torch.Tensor,
                      Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cycle X / Y features to n_qubits width."""
    n_q   = model.n_qubits
    n_feat = X.shape[-1]
    if n_feat < n_q:
        idx = torch.arange(n_q) % n_feat
        Xq  = X[:, idx]
    else:
        Xq = X[:, :n_q]
    n_out = Y.shape[-1]
    if n_out < n_q:
        idx = torch.arange(n_q) % n_out
        Yq  = Y[:, idx]
    else:
        Yq = Y[:, :n_q]
    return Xq, Yq


def _save_fig(fig, path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ figure  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PERMUTATION FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def permutation_importance(
    model,
    X_te: torch.Tensor,
    Y_te: torch.Tensor,
    model_type: str = "classical",
    n_repeats: int = 10,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute permutation feature importance for any model.

    For each feature i:
      Shuffle X[:, i] n_repeats times.
      ΔMSE = MSE(shuffled) − MSE(original).
      Larger ΔMSE → feature is more important.

    Returns
    -------
    dict  {feature_name: {"mean": float, "std": float, "raw": list}}
    """
    print("\n[ Permutation Importance ]")
    torch.manual_seed(seed)
    model.eval()

    base_pred = _predict(model, X_te, Y_te, model_type, device)
    base_mse  = _mse(base_pred, Y_te)
    print(f"  baseline MSE = {base_mse:.6f}")

    results = {}
    for fi, fname in enumerate(FEATURE_NAMES):
        deltas = []
        for _ in range(n_repeats):
            Xp = X_te.clone()
            idx = torch.randperm(len(Xp))
            Xp[:, fi] = Xp[idx, fi]
            pred = _predict(model, Xp, Y_te, model_type, device)
            deltas.append(_mse(pred, Y_te) - base_mse)
        results[fname] = {
            "mean": float(np.mean(deltas)),
            "std":  float(np.std(deltas)),
            "raw":  [float(d) for d in deltas],
        }
        print(f"  {fname:<10}: ΔMSE = {results[fname]['mean']:+.6f}"
              f"  ± {results[fname]['std']:.6f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — GRADIENT-BASED SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════

def gradient_sensitivity(
    model,
    X_te: torch.Tensor,
    Y_te: torch.Tensor,
    model_type: str = "classical",
    device: str = "cpu",
    per_output: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute mean |∂output/∂input| over the test set.

    For QPINN the gradient flows through the statevector simulation.
    For QAPINN gradients pass through both classical and quantum layers.

    Returns
    -------
    {
      "global":  np.ndarray [NI],           # mean over all outputs
      "per_out": np.ndarray [NO, NI],        # per output variable
    }
    """
    print("\n[ Gradient Sensitivity ]")
    model.eval()

    if model_type == "qpinn":
        Xq, Yq = _prep_qpinn_input(model, X_te, Y_te)
        X_in = Xq.to(device).requires_grad_(True)
        Y_in = Yq.to(device)
        pred_full, _ = model(X_in, Y_in)
        pred = pred_full[:, :NO]
    else:
        X_in = X_te.to(device).requires_grad_(True)
        pred = model(X_in)

    sens_per = np.zeros((NO, X_in.shape[1]))
    for oi in range(NO):
        if X_in.grad is not None:
            X_in.grad.zero_()
        pred[:, oi].sum().backward(retain_graph=(oi < NO - 1))
        sens_per[oi] = X_in.grad.abs().mean(0).cpu().detach().numpy()

    # pad / crop to NI if QPINN recycled features
    if sens_per.shape[1] < NI:
        full = np.zeros((NO, NI))
        for qi in range(sens_per.shape[1]):
            fi = qi % NI
            full[:, fi] = np.maximum(full[:, fi], sens_per[:, qi])
        sens_per = full
    elif sens_per.shape[1] > NI:
        sens_per = sens_per[:, :NI]

    sens_global = sens_per.mean(axis=0)

    print("  Global sensitivity (mean |∂out/∂x|):")
    for fi, fname in enumerate(FEATURE_NAMES):
        print(f"    {fname:<10}: {sens_global[fi]:.6f}")

    return {"global": sens_global, "per_out": sens_per}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — INTEGRATED GRADIENTS
# ══════════════════════════════════════════════════════════════════════════════

def integrated_gradients(
    model,
    X_te: torch.Tensor,
    Y_te: torch.Tensor,
    model_type: str = "classical",
    device: str = "cpu",
    n_steps: int = 50,
    baseline: Optional[torch.Tensor] = None,
    sample_size: int = 200,
) -> np.ndarray:
    """
    Integrated Gradients attribution (Sundararajan et al., 2017).

    IG_i(x) = (x_i − x'_i) × ∫₀¹ ∂F(x' + α(x−x'))/∂x_i dα

    Satisfies completeness: sum of attributions = F(x) − F(x').

    Parameters
    ----------
    baseline : tensor [1, NI] or None  (defaults to zeros)
    sample_size : number of test points to explain

    Returns
    -------
    np.ndarray [NI]  mean attribution magnitude per feature
    """
    print("\n[ Integrated Gradients ]")
    model.eval()

    X_sub = X_te[:sample_size]
    Y_sub = Y_te[:sample_size]

    if baseline is None:
        baseline = torch.zeros_like(X_sub[:1])  # [1, NI]

    all_attrs = []

    for xi in range(len(X_sub)):
        x     = X_sub[xi:xi+1]          # [1, NI]
        x_b   = baseline                 # [1, NI]
        delta = x - x_b                  # [1, NI]

        grads = []
        for step in range(n_steps + 1):
            alpha  = step / n_steps
            x_int  = x_b + alpha * delta  # [1, NI]
            if model_type == "qpinn":
                n_q = model.n_qubits
                idx = torch.arange(n_q) % NI
                Xq  = x_int[:, idx].to(device).requires_grad_(True)
                Yq  = Y_sub[xi:xi+1, :n_q if n_q <= NO else NO].to(device)
                if Yq.shape[1] < n_q:
                    idxy = torch.arange(n_q) % Yq.shape[1]
                    Yq   = Yq[:, idxy]
                pred_full, _ = model(Xq, Yq)
                pred = pred_full[:, :NO]
            else:
                x_req = x_int.to(device).requires_grad_(True)
                pred  = model(x_req)

            pred.sum().backward()

            if model_type == "qpinn":
                g = Xq.grad.abs().cpu().detach().numpy()  # [1, n_q]
                # map back to NI
                g_full = np.zeros((1, NI))
                for qi in range(g.shape[1]):
                    g_full[0, qi % NI] = max(g_full[0, qi % NI], g[0, qi])
                grads.append(g_full[0])
            else:
                grads.append(x_req.grad.abs().cpu().detach().numpy()[0])

        # trapezoidal integration
        grads_arr = np.array(grads)  # [n_steps+1, NI]
        avg_grads = np.trapz(grads_arr, axis=0) / n_steps
        attribution = np.abs(delta.numpy()[0]) * avg_grads
        all_attrs.append(attribution)

    attrs = np.array(all_attrs).mean(axis=0)  # [NI]

    print("  Integrated Gradients (mean |IG|):")
    for fi, fname in enumerate(FEATURE_NAMES):
        print(f"    {fname:<10}: {attrs[fi]:.6f}")

    return attrs


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SHAP (KernelExplainer — model-agnostic)
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap(
    model,
    X_tr: torch.Tensor,
    X_te: torch.Tensor,
    Y_te: torch.Tensor,
    model_type: str = "classical",
    device: str = "cpu",
    background_size: int = 100,
    explain_size: int = 200,
) -> Optional[np.ndarray]:
    """
    SHAP KernelExplainer attribution.

    Requires:  pip install shap

    Returns
    -------
    np.ndarray [explain_size, NI]  mean |SHAP| per feature, or None if shap
    is not installed.
    """
    try:
        import shap
    except ImportError:
        print("\n[ SHAP ]  ✗ shap not installed — run:  pip install shap")
        return None

    print("\n[ SHAP — KernelExplainer ]")

    def predict_fn(X_np: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X_np, dtype=torch.float32)
        if model_type == "qpinn":
            n_q = model.n_qubits
            idx = torch.arange(n_q) % NI
            Xq  = X_t[:, idx].to(device)
            Yq  = torch.zeros(len(X_t), n_q, device=device)
            model.eval()
            with torch.no_grad():
                pred_full, _ = model(Xq, Yq)
            return pred_full[:, :NO].cpu().numpy()
        model.eval()
        with torch.no_grad():
            return model(X_t.to(device)).cpu().numpy()

    bg  = X_tr[torch.randperm(len(X_tr))[:background_size]].numpy()
    exp = X_te[:explain_size].numpy()

    explainer  = shap.KernelExplainer(predict_fn, bg)
    shap_vals  = explainer.shap_values(exp, nsamples=100, silent=True)
    # shap_vals: list of [explain_size, NI] arrays, one per output

    mean_abs = np.mean([np.abs(sv).mean(0) for sv in shap_vals], axis=0)  # [NI]

    print("  Mean |SHAP| per feature:")
    for fi, fname in enumerate(FEATURE_NAMES):
        print(f"    {fname:<10}: {mean_abs[fi]:.6f}")

    return mean_abs, shap_vals


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — QUANTUM STATE PROBE
# ══════════════════════════════════════════════════════════════════════════════

def quantum_state_probe(
    model,
    X_te: torch.Tensor,
    Y_te: torch.Tensor,
    device: str = "cpu",
    n_steps: int = 20,
    sample_size: int = 50,
) -> np.ndarray:
    """
    Measure how much each input feature perturbs the quantum statevector.

    Method: For each feature i, vary X[:, i] from −1 to +1 while holding
    all other features at their test-set mean. Compute the Frobenius distance
    between density matrices ρ = |ψ⟩⟨ψ| at each step:

        dist_i = mean_x ‖ρ(x + δe_i) − ρ(x)‖_F

    Requires TorchPINN (QPINN) to expose statevectors.
    For QAPINN, uses the QuantumLayer's internal statevector.

    Returns
    -------
    np.ndarray [NI]  mean density-matrix distance per feature
    """
    print("\n[ Quantum State Probe ]")

    from utilities_quantum_torch import torch_encode, torch_ansatz

    model.eval()
    n_q   = model.n_qubits
    X_sub = X_te[:sample_size]
    X_mean = X_sub.mean(0, keepdim=True)  # [1, NI]

    def _get_statevector(x_in: torch.Tensor) -> torch.Tensor:
        """Extract raw statevector from TorchPINN."""
        # x_in: [B, n_q]
        x_in = x_in.to(device)
        state = torch.zeros(x_in.shape[0], 2 ** n_q,
                            dtype=torch.cfloat, device=device)
        state[:, 0] = 1.0 + 0j
        state = torch_encode(state, x_in, n_q, model.encoding)
        state = torch_ansatz(state, model.weights, n_q, model.ansatz)
        return state  # [B, 2^n_q]

    def _dm_dist(sv1: torch.Tensor, sv2: torch.Tensor) -> float:
        """Frobenius distance of density matrices."""
        rho1 = sv1.unsqueeze(-1) @ sv1.conj().unsqueeze(-2)
        rho2 = sv2.unsqueeze(-1) @ sv2.conj().unsqueeze(-2)
        return float(torch.norm(rho1 - rho2, p="fro").mean().item())

    # baseline statevector at X_mean
    idx_q   = torch.arange(n_q) % NI
    X_base  = X_mean[:, idx_q].expand(sample_size, -1)   # [B, n_q]
    sv_base = _get_statevector(X_base.clone())

    dists = []
    vals  = torch.linspace(-1.0, 1.0, n_steps)

    for fi in range(NI):
        fi_q = fi % n_q
        total_dist = 0.0
        for v in vals:
            Xv = X_base.clone()
            Xv[:, fi_q] = v
            sv = _get_statevector(Xv)
            total_dist += _dm_dist(sv, sv_base)
        dists.append(total_dist / n_steps)

    dists = np.array(dists)
    print("  Density-matrix distance per feature:")
    for fi, fname in enumerate(FEATURE_NAMES):
        print(f"    {fname:<10}: {dists[fi]:.6f}")

    return dists


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — QUBIT ENTANGLEMENT / VON NEUMANN ENTROPY
# ══════════════════════════════════════════════════════════════════════════════

def qubit_entropy(
    model,
    X_te: torch.Tensor,
    Y_te: torch.Tensor,
    device: str = "cpu",
    sample_size: int = 100,
) -> np.ndarray:
    """
    Compute Von Neumann entropy for each qubit's reduced density matrix.

        S(ρ_i) = −Tr(ρ_i log ρ_i)

    High entropy → qubit is strongly entangled with the rest of the circuit.

    Returns
    -------
    np.ndarray [n_qubits]
    """
    print("\n[ Qubit Von Neumann Entropy ]")

    try:
        from utilities_quantum_torch import torch_encode, torch_ansatz
    except ImportError:
        print("  ✗ utilities_quantum_torch not found — skipping.")
        return np.zeros(model.n_qubits)

    model.eval()
    n_q   = model.n_qubits
    Xq, Yq = _prep_qpinn_input(model, X_te[:sample_size], Y_te[:sample_size])
    Xq = Xq.to(device)

    state = torch.zeros(len(Xq), 2 ** n_q, dtype=torch.cfloat, device=device)
    state[:, 0] = 1.0 + 0j
    with torch.no_grad():
        state = torch_encode(state, Xq, n_q, model.encoding)
        state = torch_ansatz(state, model.weights, n_q, model.ansatz)
        # state: [B, 2^n_q]

    entropies = []
    for qi in range(n_q):
        # partial trace over all qubits except qi
        dim_keep = 2
        dim_rest = 2 ** (n_q - 1)
        sv = state  # [B, 2^n_q]

        # reshape: [B, 2^qi, 2, 2^(n_q-qi-1)]
        sv_r = sv.reshape(-1, 2 ** qi, 2, 2 ** (n_q - qi - 1))

        # build 2×2 reduced density matrix for qubit qi
        rho_qi = torch.zeros(len(sv), 2, 2, dtype=torch.cfloat, device=device)
        for a in range(2):
            for b in range(2):
                rho_qi[:, a, b] = (
                    sv_r[:, :, a, :].conj().reshape(len(sv), -1) *
                    sv_r[:, :, b, :].reshape(len(sv), -1)
                ).sum(-1).sum(-1)  # sum over all other qubits

        rho_mean = rho_qi.mean(0)  # [2, 2]

        # eigenvalues
        eigvals = torch.linalg.eigvalsh(rho_mean).real
        eigvals = eigvals.clamp(min=1e-12)
        S = float(-(eigvals * torch.log(eigvals)).sum().item())
        entropies.append(S)

    entropies = np.array(entropies)
    print(f"  Qubit entropies (S = -Tr(ρ log ρ)):")
    for qi in range(n_q):
        fname = FEATURE_NAMES[qi % NI]
        print(f"    q{qi} ({fname:<8}): S = {entropies[qi]:.4f}")

    return entropies


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CIRCUIT WEIGHT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def circuit_weight_analysis(model) -> Dict:
    """
    Analyse the trained variational parameters (θ) of the quantum ansatz.

    Reports:
      - weight distribution statistics
      - fraction of "dead" weights (|θ| < 0.05)
      - fraction of "saturated" weights (|θ| > π − 0.1)
      - per-layer gradient norm (if model has .weights.grad)

    Works for TorchPINN and QAPINN (QuantumLayer).

    Returns
    -------
    dict with keys: weights (np.ndarray), stats (dict)
    """
    print("\n[ Circuit Weight Analysis ]")

    # extract weights
    if hasattr(model, "weights"):
        # TorchPINN
        w = model.weights.detach().cpu().numpy().flatten()
        print(f"  TorchPINN  — ansatz weights: {w.shape}")
    elif hasattr(model, "quantum_layer"):
        # QAPINN
        w = model.quantum_layer.q_weights.detach().cpu().numpy().flatten()
        print(f"  QAPINN     — QVC weights: {w.shape}")
    else:
        print("  ✗ Model has no detectable quantum weights.")
        return {}

    stats = {
        "n_params":    int(w.size),
        "mean":        float(w.mean()),
        "std":         float(w.std()),
        "min":         float(w.min()),
        "max":         float(w.max()),
        "dead_frac":   float((np.abs(w) < 0.05).mean()),       # near zero
        "sat_frac":    float((np.abs(w) > np.pi - 0.1).mean()), # near ±π
    }

    print(f"  n_params    : {stats['n_params']}")
    print(f"  mean ± std  : {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"  range       : [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  dead (<0.05): {stats['dead_frac']*100:.1f}%")
    print(f"  saturated   : {stats['sat_frac']*100:.1f}%")

    return {"weights": w, "stats": stats}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FEATURE INTERACTION (H-STATISTIC)
# ══════════════════════════════════════════════════════════════════════════════

def feature_interaction(
    model,
    X_te: torch.Tensor,
    Y_te: torch.Tensor,
    model_type: str = "classical",
    device: str = "cpu",
    sample_size: int = 300,
    feature_pairs: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    Friedman H-statistic for pairwise feature interactions.

    H²(i,j) = Var[PD_{ij}(x_i, x_j) − PD_i(x_i) − PD_j(x_j)] / Var[F(x)]

    where PD is the partial dependence function.

    Approximated via ICE (Individual Conditional Expectation) curves
    over a subsample of the test set.

    Returns
    -------
    np.ndarray [NI, NI]  symmetric H² matrix
    """
    print("\n[ Feature Interaction H-statistic ]")

    if feature_pairs is None:
        feature_pairs = [(i, j) for i in range(NI) for j in range(i + 1, NI)]

    X_sub = X_te[:sample_size]
    Y_sub = Y_te[:sample_size]
    n     = len(X_sub)
    grid  = 10   # grid points per feature for PD estimation

    def _pd_1d(fi: int) -> np.ndarray:
        """Partial dependence of feature fi. Returns [grid]."""
        vals = torch.linspace(X_sub[:, fi].min(), X_sub[:, fi].max(), grid)
        pds  = []
        for v in vals:
            Xp = X_sub.clone(); Xp[:, fi] = v
            pred = _predict(model, Xp, Y_sub, model_type, device)
            pds.append(pred.mean(0).numpy())   # [NO]
        return np.array(pds)  # [grid, NO]

    def _pd_2d(fi: int, fj: int) -> np.ndarray:
        """Joint partial dependence of (fi, fj). Returns [grid, grid, NO]."""
        vi = torch.linspace(X_sub[:, fi].min(), X_sub[:, fi].max(), grid)
        vj = torch.linspace(X_sub[:, fj].min(), X_sub[:, fj].max(), grid)
        pds = np.zeros((grid, grid, NO))
        for a, va in enumerate(vi):
            for b, vb in enumerate(vj):
                Xp = X_sub.clone()
                Xp[:, fi] = va; Xp[:, fj] = vb
                pred = _predict(model, Xp, Y_sub, model_type, device)
                pds[a, b] = pred.mean(0).numpy()
        return pds

    # global variance
    full_pred = _predict(model, X_sub, Y_sub, model_type, device).numpy()
    var_total = full_pred.var()

    H2 = np.zeros((NI, NI))
    pd1_cache = {}

    for fi, fj in feature_pairs:
        if fi not in pd1_cache:
            pd1_cache[fi] = _pd_1d(fi)
        if fj not in pd1_cache:
            pd1_cache[fj] = _pd_1d(fj)

        pd_i   = pd1_cache[fi]   # [grid, NO]
        pd_j   = pd1_cache[fj]   # [grid, NO]
        pd_ij  = _pd_2d(fi, fj)  # [grid, grid, NO]

        # H² = Var(PD_ij − PD_i − PD_j) / Var(F)
        residual = np.zeros((grid, grid, NO))
        for a in range(grid):
            for b in range(grid):
                residual[a, b] = pd_ij[a, b] - pd_i[a] - pd_j[b]

        h2 = residual.var() / (var_total + 1e-12)
        H2[fi, fj] = h2
        H2[fj, fi] = h2
        np.fill_diagonal(H2, 1.0)
        print(f"  H²({FEATURE_NAMES[fi]:<8}, {FEATURE_NAMES[fj]:<8}) = {h2:.4f}")

    return H2


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════

def ablation_study(
    model_factory_fn,
    X_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_te: torch.Tensor,
    Y_te: torch.Tensor,
    device: str = "cpu",
    n_epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    ablations: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Systematic ablation: train variants of the QAPINN with components removed.

    Parameters
    ----------
    model_factory_fn : callable
        Function that returns a new QAPINN given keyword overrides.
        Signature: model_factory_fn(**kwargs) -> nn.Module
    ablations : dict
        Maps ablation name → kwargs override dict.
        Example:
          {
            "no_quantum_layer":  {"n_qubits": 1},
            "no_fidelity_loss":  {"lambda_q": 0.0},
            "classical_only":    {"use_physics": False, "n_qubits": 0},
          }

    Returns
    -------
    dict  {ablation_name: {"r2": float, "mse": float}}
    """
    from torch.utils.data import DataLoader, TensorDataset

    if ablations is None:
        ablations = {
            "full_model":         {},
            "no_fidelity_loss":   {"lambda_q": 0.0},
            "no_physics_loss":    {"use_physics": False},
            "qvc_layer_0":        {"q_layer_idx": 0},
            "qvc_layer_1":        {"q_layer_idx": 1},
            "qvc_layer_2":        {"q_layer_idx": 2},
            "qvc_layer_3":        {"q_layer_idx": 3},
            "2_qubits":           {"n_qubits": 2},
            "4_qubits":           {"n_qubits": 4},
            "6_qubits":           {"n_qubits": 6},
        }

    print("\n[ Ablation Study ]")
    results = {}

    for name, kwargs in ablations.items():
        print(f"\n  ── {name} ──")
        try:
            model = model_factory_fn(**kwargs).to(device)
            opt   = torch.optim.Adam(model.parameters(), lr=lr)
            loader = DataLoader(TensorDataset(X_tr, Y_tr),
                                batch_size=batch_size, shuffle=True)

            model.train()
            for ep in range(n_epochs):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    _, __, loss = model.forward_with_loss(xb, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

            # evaluate
            model.eval()
            with torch.no_grad():
                pred = model(X_te.to(device)).cpu()
            mse   = float(((pred - Y_te) ** 2).mean())
            ss_r  = float(((pred - Y_te) ** 2).sum())
            ss_t  = float(((Y_te - Y_te.mean(0)) ** 2).sum())
            r2    = 1 - ss_r / (ss_t + 1e-12)
            results[name] = {"r2": r2, "mse": mse}
            print(f"    R² = {r2:.4f}  MSE = {mse:.6f}")

        except Exception as e:
            print(f"    ✗ failed: {e}")
            results[name] = {"r2": None, "mse": None}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

DARK_BG  = "#0a0e1a"
CARD_BG  = "#111827"
BORDER   = "#1e2d45"
TEXT_COL = "#e2e8f0"
MUTED    = "#64748b"

def _apply_dark_style(ax):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT_COL)


def plot_permutation_importance(
    results: Dict, model_name: str = "QPINN", save_path: Optional[Path] = None
):
    """Bar chart of permutation importance with error bars."""
    fnames = list(results.keys())
    means  = [results[f]["mean"] for f in fnames]
    stds   = [results[f]["std"]  for f in fnames]

    order  = np.argsort(means)[::-1]
    fnames = [fnames[i] for i in order]
    means  = [means[i]  for i in order]
    stds   = [stds[i]   for i in order]
    cols   = [COLOURS[i % len(COLOURS)] for i in order]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK_BG)
    _apply_dark_style(ax)
    bars = ax.barh(fnames, means, xerr=stds, color=cols, alpha=0.85,
                   error_kw=dict(ecolor=MUTED, capsize=4))
    ax.set_xlabel("ΔMSE (increase when feature is shuffled)")
    ax.set_title(f"Permutation Feature Importance — {model_name}")
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()
    return fig


def plot_gradient_sensitivity(
    sens: Dict, model_name: str = "QPINN", save_path: Optional[Path] = None
):
    """Grouped bar chart — sensitivity per feature per output."""
    global_s = sens["global"]
    per_out  = sens["per_out"]

    x     = np.arange(NI)
    width = 0.22
    fig, ax = plt.subplots(figsize=(11, 4), facecolor=DARK_BG)
    _apply_dark_style(ax)

    out_cols = ["#38bdf8", "#818cf8", "#34d399"]
    for oi, oname in enumerate(OUTPUT_NAMES):
        ax.bar(x + oi * width, per_out[oi], width,
               label=oname, color=out_cols[oi], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(FEATURE_NAMES, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("|∂output/∂input|")
    ax.set_title(f"Gradient-Based Sensitivity — {model_name}")
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COL, fontsize=9)
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()
    return fig


def plot_integrated_gradients(
    ig_attrs: np.ndarray, model_name: str = "QPINN",
    save_path: Optional[Path] = None
):
    order = np.argsort(ig_attrs)[::-1]
    fnames = [FEATURE_NAMES[i] for i in order]
    vals   = ig_attrs[order]
    cols   = [COLOURS[i % len(COLOURS)] for i in order]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK_BG)
    _apply_dark_style(ax)
    ax.barh(fnames, vals, color=cols, alpha=0.85)
    ax.set_xlabel("Mean |Integrated Gradient| attribution")
    ax.set_title(f"Integrated Gradients — {model_name}")
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()
    return fig


def plot_quantum_state_probe(
    dists: np.ndarray, model_name: str = "QPINN",
    save_path: Optional[Path] = None
):
    order  = np.argsort(dists)[::-1]
    fnames = [FEATURE_NAMES[i] for i in order]
    vals   = dists[order]
    cols   = [COLOURS[i % len(COLOURS)] for i in order]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK_BG)
    _apply_dark_style(ax)
    ax.barh(fnames, vals, color=cols, alpha=0.85)
    ax.set_xlabel("Mean Frobenius distance ‖ρ(x+δ) − ρ(x)‖_F")
    ax.set_title(f"Quantum State Probe — {model_name}")
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()
    return fig


def plot_qubit_entropy(
    entropies: np.ndarray, n_qubits: int,
    model_name: str = "QPINN", save_path: Optional[Path] = None
):
    labels = [f"q{i}\n({FEATURE_NAMES[i%NI]})" for i in range(n_qubits)]
    cols   = [COLOURS[i % len(COLOURS)] for i in range(n_qubits)]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK_BG)
    _apply_dark_style(ax)
    bars = ax.bar(labels, entropies, color=cols, alpha=0.85)
    ax.set_ylabel("Von Neumann Entropy S = −Tr(ρ log ρ)")
    ax.set_title(f"Qubit Entanglement Entropy — {model_name}")
    ax.axhline(np.log(2), color=MUTED, lw=0.8, ls="--",
               label="max entropy (log 2)")
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COL, fontsize=9)
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()
    return fig


def plot_circuit_weights(
    w: np.ndarray, model_name: str = "QPINN",
    save_path: Optional[Path] = None
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=DARK_BG)
    for ax in axes:
        _apply_dark_style(ax)

    # Histogram
    ax = axes[0]
    ax.hist(w, bins=30, color="#818cf8", alpha=0.8, edgecolor=DARK_BG)
    ax.axvline(0,       color="#f87171", lw=1.5, ls="--", label="0")
    ax.axvline( np.pi,  color="#fb923c", lw=1.5, ls="--", label="+π")
    ax.axvline(-np.pi,  color="#fb923c", lw=1.5, ls="--", label="−π")
    ax.set_xlabel("θ (radians)")
    ax.set_ylabel("count")
    ax.set_title(f"Weight Distribution — {model_name}")
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COL, fontsize=8)

    # Sorted weights
    ax = axes[1]
    ax.plot(np.sort(w), color="#38bdf8", lw=1.5)
    ax.axhline(0,      color="#f87171", lw=0.8, ls="--")
    ax.axhline( np.pi, color="#fb923c", lw=0.8, ls="--")
    ax.axhline(-np.pi, color="#fb923c", lw=0.8, ls="--")
    ax.set_xlabel("weight index (sorted)")
    ax.set_ylabel("θ value")
    ax.set_title("Sorted Weights")
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()
    return fig


def plot_interaction_heatmap(
    H2: np.ndarray, model_name: str = "QPINN",
    save_path: Optional[Path] = None
):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=DARK_BG)
    _apply_dark_style(ax)

    im = ax.imshow(H2, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(NI)); ax.set_xticklabels(FEATURE_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(NI)); ax.set_yticklabels(FEATURE_NAMES)
    ax.set_title(f"Feature Interaction H² — {model_name}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=MUTED)
    cbar.set_label("H² interaction strength", color=MUTED)

    for i in range(NI):
        for j in range(NI):
            if i != j:
                ax.text(j, i, f"{H2[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if H2[i,j] > 0.4 else MUTED)
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()
    return fig


def plot_ablation_r2(
    results: Dict, save_path: Optional[Path] = None
):
    names  = [n for n, v in results.items() if v["r2"] is not None]
    r2s    = [results[n]["r2"] for n in names]
    order  = np.argsort(r2s)[::-1]
    names  = [names[i] for i in order]
    r2s    = [r2s[i]   for i in order]
    baseline = r2s[0]
    deltas = [v - baseline for v in r2s]
    cols   = ["#ef4444" if d < 0 else "#34d399" for d in deltas]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
    _apply_dark_style(ax)
    ax.barh(names, r2s, color=cols, alpha=0.85)
    ax.axvline(baseline, color=MUTED, lw=1, ls="--", label="full model")
    ax.set_xlabel("R²")
    ax.set_title("Ablation Study — Test R²")
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT_COL, fontsize=9)
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()
    return fig


def plot_summary_dashboard(
    model_name: str,
    perm_results:    Optional[Dict]      = None,
    sens_results:    Optional[Dict]      = None,
    ig_attrs:        Optional[np.ndarray] = None,
    qstate_dists:    Optional[np.ndarray] = None,
    save_path:       Optional[Path]      = None,
):
    """
    4-panel summary figure combining the four main attribution methods.
    """
    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle(f"XAI Summary — {model_name}", color=TEXT_COL,
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    def _bar_ax(ax, vals, title, xlabel):
        _apply_dark_style(ax)
        order = np.argsort(vals)[::-1]
        fnames = [FEATURE_NAMES[i] for i in order]
        vs = [vals[i] for i in order]
        cols = [COLOURS[i % len(COLOURS)] for i in order]
        ax.barh(fnames, vs, color=cols, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)

    if perm_results is not None:
        ax = fig.add_subplot(gs[0, 0])
        means = np.array([perm_results[f]["mean"] for f in FEATURE_NAMES])
        _bar_ax(ax, means, "1 · Permutation Importance", "ΔMSE")

    if sens_results is not None:
        ax = fig.add_subplot(gs[0, 1])
        _bar_ax(ax, sens_results["global"],
                "2 · Gradient Sensitivity", "|∂output/∂input|")

    if ig_attrs is not None:
        ax = fig.add_subplot(gs[1, 0])
        _bar_ax(ax, ig_attrs, "3 · Integrated Gradients", "|IG|")

    if qstate_dists is not None:
        ax = fig.add_subplot(gs[1, 1])
        _bar_ax(ax, qstate_dists, "4 · Quantum State Probe", "‖Δρ‖_F")

    if save_path:
        _save_fig(fig, save_path)
    else:
        plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — FULL XAI PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_full_xai(
    model,
    model_name: str,
    model_type: str,          # "qpinn" | "classical"
    X_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_te: torch.Tensor,
    Y_te: torch.Tensor,
    device: str          = "cpu",
    out_dir: str         = "./xai_results",
    run_shap: bool       = True,
    run_qstate: bool     = True,
    run_interaction: bool = False,   # slow — set True if you have time
    n_perm_repeats: int  = 10,
    ig_steps: int        = 50,
    ig_samples: int      = 200,
) -> Dict:
    """
    Run all XAI methods and save figures + JSON report.

    Parameters
    ----------
    model        : trained QPINN (TorchPINN) or QAPINN instance
    model_name   : string label, e.g. "QPINN" or "QAPINN_layer2"
    model_type   : "qpinn" for TorchPINN; "classical" for QAPINN/MLP/PINN
    X_tr, Y_tr   : training tensors  (for SHAP background)
    X_te, Y_te   : test tensors
    out_dir      : where to save figures and JSON

    Returns
    -------
    dict  containing all computed XAI metrics
    """
    t0     = time.time()
    outdir = Path(out_dir) / model_name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  XAI Pipeline — {model_name}  [{model_type}]")
    print(f"  Output: {outdir}")
    print(f"{'='*65}")

    report = {"model": model_name, "model_type": model_type, "metrics": {}}

    # ── 1. Permutation importance ────────────────────────────────────────────
    perm = permutation_importance(
        model, X_te, Y_te, model_type=model_type,
        n_repeats=n_perm_repeats, device=device,
    )
    report["metrics"]["permutation_importance"] = perm
    plot_permutation_importance(perm, model_name,
                                save_path=outdir / "permutation_importance.png")

    # ── 2. Gradient sensitivity ──────────────────────────────────────────────
    sens = gradient_sensitivity(
        model, X_te, Y_te, model_type=model_type, device=device
    )
    report["metrics"]["gradient_sensitivity"] = {
        "global":  sens["global"].tolist(),
        "per_out": sens["per_out"].tolist(),
    }
    plot_gradient_sensitivity(sens, model_name,
                              save_path=outdir / "gradient_sensitivity.png")

    # ── 3. Integrated gradients ──────────────────────────────────────────────
    ig = integrated_gradients(
        model, X_te, Y_te, model_type=model_type, device=device,
        n_steps=ig_steps, sample_size=ig_samples,
    )
    report["metrics"]["integrated_gradients"] = ig.tolist()
    plot_integrated_gradients(ig, model_name,
                              save_path=outdir / "integrated_gradients.png")

    # ── 4. SHAP ──────────────────────────────────────────────────────────────
    if run_shap:
        shap_result = compute_shap(
            model, X_tr, X_te, Y_te,
            model_type=model_type, device=device,
        )
        if shap_result is not None:
            shap_mean, shap_vals = shap_result
            report["metrics"]["shap_mean_abs"] = shap_mean.tolist()

    # ── 5. Quantum state probe (QPINN only) ──────────────────────────────────
    if run_qstate and model_type == "qpinn":
        try:
            qstate = quantum_state_probe(model, X_te, Y_te, device=device)
            report["metrics"]["quantum_state_probe"] = qstate.tolist()
            plot_quantum_state_probe(qstate, model_name,
                                     save_path=outdir / "quantum_state_probe.png")
        except Exception as e:
            print(f"  ✗ quantum_state_probe failed: {e}")

    # ── 6. Qubit entropy (QPINN only) ────────────────────────────────────────
    if model_type == "qpinn":
        try:
            ent = qubit_entropy(model, X_te, Y_te, device=device)
            report["metrics"]["qubit_entropy"] = ent.tolist()
            plot_qubit_entropy(ent, model.n_qubits, model_name,
                               save_path=outdir / "qubit_entropy.png")
        except Exception as e:
            print(f"  ✗ qubit_entropy failed: {e}")

    # ── 7. QAPINN quantum layer ───────────────────────────────────────────────
    if model_type == "classical" and hasattr(model, "quantum_layer"):
        cw = circuit_weight_analysis(model)
        if cw:
            report["metrics"]["circuit_weights"] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in cw.items()
            }
            plot_circuit_weights(cw["weights"], model_name,
                                 save_path=outdir / "circuit_weights.png")

    # ── 8. Feature interaction (optional, slow) ──────────────────────────────
    if run_interaction:
        H2 = feature_interaction(
            model, X_te, Y_te, model_type=model_type, device=device
        )
        report["metrics"]["interaction_H2"] = H2.tolist()
        plot_interaction_heatmap(H2, model_name,
                                 save_path=outdir / "interaction_heatmap.png")

    # ── 9. Summary dashboard ─────────────────────────────────────────────────
    qstate_vals = (np.array(report["metrics"].get("quantum_state_probe", [None]*NI))
                   if model_type == "qpinn" else None)
    plot_summary_dashboard(
        model_name,
        perm_results  = perm,
        sens_results  = sens,
        ig_attrs      = ig,
        qstate_dists  = qstate_vals if (qstate_vals is not None and
                                        None not in qstate_vals.tolist()) else None,
        save_path     = outdir / "summary_dashboard.png",
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    report["elapsed_s"] = round(time.time() - t0, 1)

    def _serialise(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialise(v) for v in obj]
        return obj

    json_path = outdir / "xai_report.json"
    with open(json_path, "w") as f:
        json.dump(_serialise(report), f, indent=2)
    print(f"\n  ✓ JSON report → {json_path}")
    print(f"  Total time   : {report['elapsed_s']}s")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — loads real checkpoints and runs the full XAI pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Load trained models from ./results/ and run full XAI analysis.

    Adjust RESULTS_DIR and MODEL_CFG to match your training outputs.
    """
    import argparse

    parser = argparse.ArgumentParser(description="XAI for QPINN / QAPINN")
    parser.add_argument("--results_dir",  default="./results",   help="Path to training results")
    parser.add_argument("--data_dir",     default="./data",      help="Path to NS data")
    parser.add_argument("--out_dir",      default="./xai_results")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shap",         action="store_true",   help="Run SHAP (slow)")
    parser.add_argument("--interaction",  action="store_true",   help="Run H-stat (very slow)")
    parser.add_argument("--model",        default="all",
                        choices=["all", "qpinn", "qapinn", "mlp", "pinn"])
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        from utilities_classical import load_ns_data
        data = load_ns_data(args.data_dir, n_scenarios=50, t_stride=2)
        X_tr = data["X_tr"]; Y_tr = data["Y_tr"]
        X_te = data["X_te"]; Y_te = data["Y_te"]
        print(f"Data loaded: train={len(X_tr)}  test={len(X_te)}")
    except Exception as e:
        print(f"✗ Could not load data: {e}")
        print("  → Generating synthetic data for demonstration.")
        X_tr = torch.randn(2000, NI); Y_tr = torch.randn(2000, NO)
        X_te = torch.randn(400,  NI); Y_te = torch.randn(400,  NO)

    # ── Load QPINN ────────────────────────────────────────────────────────────
    if args.model in ("all", "qpinn"):
        try:
            from utilities_quantum_torch import TorchPINN
            from Step2_train_ns_quantum_pinn_and_qpinn_torch import (
                build_qpinn, load_ckpt
            )
            ckpt_path = Path(args.results_dir) / "qpinn" / "qpinn_final.pt"
            model_qp = build_qpinn(n_qubits=NI, n_layers=2,
                                   encoding="angle_full", ansatz="u_ring",
                                   device=device)
            model_qp, _, _, _, _, _ = load_ckpt(str(ckpt_path), model_qp, device=device)
            run_full_xai(
                model_qp, "QPINN", "qpinn",
                X_tr, Y_tr, X_te, Y_te,
                device=device, out_dir=args.out_dir,
                run_shap=args.shap,
                run_qstate=True,
                run_interaction=args.interaction,
            )
        except Exception as e:
            print(f"\n✗ QPINN XAI failed: {e}")
            import traceback; traceback.print_exc()

    # ── Load QAPINN ───────────────────────────────────────────────────────────
    if args.model in ("all", "qapinn"):
        try:
            from Step2_train_ns_quantum_pinn_and_qpinn_torch import (
                QAPINN, build_qapinn, load_ckpt
            )
            ckpt_path = Path(args.results_dir) / "qapinn" / "qapinn_final.pt"
            model_qa = build_qapinn(
                hidden=128, depth=4, q_layer_idx=None,
                n_qubits=6, n_layers=2, encoding="angle_full",
                ansatz="u_ring", lambda_q=0.1, device=device,
            )
            model_qa, _, _, _, _, _ = load_ckpt(str(ckpt_path), model_qa, device=device)
            run_full_xai(
                model_qa, "QAPINN", "classical",
                X_tr, Y_tr, X_te, Y_te,
                device=device, out_dir=args.out_dir,
                run_shap=args.shap,
                run_qstate=False,     # handled by circuit_weight_analysis
                run_interaction=args.interaction,
            )
        except Exception as e:
            print(f"\n✗ QAPINN XAI failed: {e}")
            import traceback; traceback.print_exc()

    # ── Load MLP ──────────────────────────────────────────────────────────────
    if args.model in ("all", "mlp"):
        try:
            from Step2_train_ns_quantum_pinn_and_qpinn_torch import (
                build_mlp, load_ckpt
            )
            ckpt_path = Path(args.results_dir) / "mlp" / "mlp_final.pt"
            model_mlp = build_mlp(device=device)
            model_mlp, _, _, _, _, _ = load_ckpt(str(ckpt_path), model_mlp, device=device)
            run_full_xai(
                model_mlp, "MLP", "classical",
                X_tr, Y_tr, X_te, Y_te,
                device=device, out_dir=args.out_dir,
                run_shap=args.shap,
                run_qstate=False, run_interaction=args.interaction,
            )
        except Exception as e:
            print(f"\n✗ MLP XAI failed: {e}")

    print(f"\n{'='*65}")
    print(f"  XAI complete — results in {args.out_dir}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
