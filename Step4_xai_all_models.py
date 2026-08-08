"""
Step4_xai_all_models.py
=======================
Run the Step-3 XAI toolkit on **every** model Step 2 trains — MLP, ClassPINN,
QPINN (including every combo of the encoder x ansatz sweep) and TorchQAPINN —
without hard-coding a single hyper-parameter.

Why this exists
---------------
Step3_xai_quantum_pinn_v3.main() only reaches QPINN / QAPINN / MLP, and it
reaches them through hard-coded paths and hard-coded architectures that no
longer match what Step 2 writes. Every failure is swallowed by a bare
`except`, so the script prints "XAI complete" having explained nothing.

This driver instead:

  1. Walks results_dir for `*_final.pt`.
  2. Reads the `config` dict that Step 2 already embeds in every checkpoint,
     and rebuilds the architecture *from that config* — so encoding, ansatz,
     n_qubits, n_layers, n_enc_layers, depth and q_layer_idx always match the
     weights being loaded.
  3. Wraps each model in XAIAdapter, which normalises the forward signature
     so the v3 XAI functions stop crashing on models that return a bare
     tensor instead of a (pred, loss) tuple.
  4. Runs run_full_xai() per model, then builds cross-model comparison
     figures that v3 never produced.

Nothing in Step3_xai_quantum_pinn_v3.py is modified — it is imported and
reused.

Usage
-----
  python Step4_xai_all_models.py \
      --results_dir /mnt/d/QCFD/WISER/BQC/github/results \
      --data_dir    /mnt/d/QCFD/WISER/BQC/github/data \
      --out_dir     /mnt/d/QCFD/WISER/BQC/github/xai_results

  # fast pass, skip the slow methods
  python Step4_xai_all_models.py --no-ig

  # everything, including SHAP and the H-statistic
  python Step4_xai_all_models.py --shap --interaction

  # only the sweep winners
  python Step4_xai_all_models.py --only qpinn --top_k 3
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NumPy 2 removed the `trapz` alias in favour of `trapezoid`. v3's
# integrated_gradients still calls np.trapz, so it raises AttributeError on
# any NumPy >= 2.0. Restore the alias rather than editing v3.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — LOCATE AND IMPORT THE STEP 2 / STEP 3 MODULES
# ══════════════════════════════════════════════════════════════════════════════
#
# Step 3 v3 does `from Step2_train_ns_quantum_pinn_and_qpinn_torch import ...`
# in three places, but the file on disk is `..._torch_v5.py`. That import has
# never resolved, which is why the QPINN / QAPINN / field-comparison blocks all
# fall into their `except` branches. We find whatever Step 2 file actually
# exists and register it under BOTH names in sys.modules, so v3's stale import
# resolves without editing v3.

STEP2_CANDIDATES = [
    "Step2_train_ns_quantum_pinn_and_qpinn_torch_v5",
    "Step2_train_ns_quantum_pinn_and_qpinn_torch_v4",
    "Step2_train_ns_quantum_pinn_and_qpinn_torch",
    "Step3_train_all_models",
]
STEP3_CANDIDATES = [
    "Step3_xai_quantum_pinn_v3",
    "Step3_xai_quantum_pinn_v2",
    "Step3_xai_quantum_pinn",
]
STEP2_LEGACY_ALIAS = "Step2_train_ns_quantum_pinn_and_qpinn_torch"


def _load_module(candidates: List[str], search_dirs: List[Path], label: str):
    """Import the first candidate module name that exists in search_dirs."""
    for d in search_dirs:
        for name in candidates:
            path = d / f"{name}.py"
            if not path.exists():
                continue
            spec = importlib.util.spec_from_file_location(name, str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            print(f"  [{label}] loaded {path}")
            return mod, name
    raise ImportError(
        f"Could not find any of {candidates} in {[str(d) for d in search_dirs]}"
    )


def load_pipeline_modules(script_dir: Path, extra_dirs: List[Path]):
    """Import Step 2 + Step 3 and patch v3's stale Step 2 import."""
    search = [script_dir] + extra_dirs + [Path.cwd()]
    search = [d for d in dict.fromkeys(search) if d.is_dir()]
    for d in search:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))

    step2, step2_name = _load_module(STEP2_CANDIDATES, search, "Step2")
    # v3 imports Step 2 under its old name in three functions — alias it.
    if step2_name != STEP2_LEGACY_ALIAS:
        sys.modules[STEP2_LEGACY_ALIAS] = step2
        print(f"  [Step2] aliased as '{STEP2_LEGACY_ALIAS}' for Step 3 imports")

    step3, _ = _load_module(STEP3_CANDIDATES, search, "Step3")
    return step2, step3


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FORWARD-SIGNATURE ADAPTER
# ══════════════════════════════════════════════════════════════════════════════

class XAIAdapter(nn.Module):
    """
    Normalise every model to a single calling convention:

        pred, aux = adapter(x)          # classical models
        pred, aux = adapter(x, y)       # qpinn models

    v3 assumes `pred, _ = model(X)` everywhere. That holds for TorchQPINN and
    TorchQAPINN but not necessarily for UnifiedPINN — Step 2's own evaluate()
    guards with `if isinstance(output, tuple)`, which is the tell that a bare
    tensor comes back sometimes. Unpacking a [N, 3] tensor into two names
    raises "too many values to unpack" for any N != 2, which is exactly the
    kind of error v3's bare `except Exception` hides. The adapter removes the
    ambiguity once, for all call sites.

    It also controls the y-register for QPINN. TorchQPINN.forward(x, y) uses y
    for the fidelity term. Step 2's own `predict()` / `predict_field()` pass
    y = zeros at inference, but v3's `_predict_qpinn` passes the true test
    labels. If any part of the prediction path reads y, attributions computed
    with true labels are contaminated by target leakage and the resulting
    feature rankings are not trustworthy. Default here is zeros, matching
    inference; `--qpinn_y true` restores v3's behaviour for comparison.

    Attribute lookups (n_qubits, weights, encoding, ansatz, n_out,
    quantum_layer, ...) fall through to the wrapped model, so
    quantum_state_probe / qubit_entropy / circuit_weight_analysis keep working.
    """

    def __init__(self, model: nn.Module, model_type: str, qpinn_y: str = "zero"):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.qpinn_y = qpinn_y

    def forward(self, x, y=None):
        inner = self.model
        if self.model_type == "qpinn":
            if y is None or self.qpinn_y == "zero":
                n_out = getattr(inner, "n_out", 3)
                y = torch.zeros(x.shape[0], n_out,
                                device=x.device, dtype=x.dtype)
            out = inner(x, y)
        else:
            out = inner(x) if y is None else inner(x)

        if isinstance(out, (tuple, list)):
            pred = out[0]
            aux = out[1] if len(out) > 1 else None
            return pred, aux
        return out, None

    def __getattr__(self, name):
        # nn.Module.__getattr__ handles params/buffers/submodules of the
        # adapter itself; anything else is delegated to the wrapped model.
        try:
            return super().__getattr__(name)
        except AttributeError:
            mods = self.__dict__.get("_modules", {})
            inner = mods.get("model")
            if inner is not None and hasattr(inner, name):
                return getattr(inner, name)
            raise


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CHECKPOINT DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def discover_checkpoints(results_dir: Path) -> List[Dict]:
    """
    Find every `*_final.pt` under results_dir and read its embedded config.

    Step 2's save_results() always writes `{model_name}_final.pt` with a
    `config` dict, including inside sweep_qpinn (save_ckpts=False only
    disables the *per-epoch* checkpoints — the final one is still written).
    So one rglob covers the single runs and every sweep combo alike.
    """
    found = []
    if not results_dir.is_dir():
        print(f"  ! results_dir does not exist: {results_dir}")
        return found

    for path in sorted(results_dir.rglob("*_final.pt")):
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  ! unreadable checkpoint {path.name}: {e}")
            continue

        cfg = ckpt.get("config", {}) or {}
        family = _classify(cfg, path)
        if family is None:
            print(f"  ! unknown model family for {path} (config={cfg})")
            continue

        metrics = _sidecar_metrics(path)
        found.append({
            "path": path,
            "config": cfg,
            "family": family,
            "run_tag": path.parent.name,
            "epoch": ckpt.get("epoch"),
            "r2": metrics.get("r2"),
            "mse": metrics.get("mse"),
        })
    return found


def _classify(cfg: Dict, path: Path) -> Optional[str]:
    """Map a checkpoint to one of: mlp | classpinn | qpinn | qapinn."""
    name = str(cfg.get("model", "")).lower()
    if name.startswith("classpinn"):
        return "classpinn"
    if name == "qapinn":
        return "qapinn"
    if name == "qpinn":
        return "qpinn"
    if name == "mlp":
        return "mlp"

    # Fall back to the filename if config was empty/older format.
    stem = path.stem.lower()
    for fam in ("qapinn", "classpinn", "qpinn", "mlp"):
        if stem.startswith(fam):
            return fam
    return None


def _sidecar_metrics(final_path: Path) -> Dict:
    """Read the *_metrics.json Step 2 writes next to the checkpoint."""
    stem = final_path.name.replace("_final.pt", "")
    mp = final_path.parent / f"{stem}_metrics.json"
    if not mp.exists():
        return {}
    try:
        with open(mp) as f:
            return json.load(f).get("evaluation", {}) or {}
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — REBUILD MODELS FROM THEIR OWN CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def rebuild_model(step2, entry: Dict, device, data: Optional[Dict] = None):
    """
    Reconstruct the architecture from the checkpoint's config and load weights.

    This is the fix for the class of bug where v3 builds a 7-qubit QAPINN and
    loads 14-qubit weights into it, or builds an angle_full QPINN and loads
    arctan weights: architecture always comes from the config that Step 2
    saved alongside those exact weights, never from a literal in this file.
    """
    cfg, fam = entry["config"], entry["family"]

    if fam == "mlp":
        model = step2.build_mlp(device)
        mtype = "classical"

    elif fam == "classpinn":
        preset = cfg.get("preset") or str(cfg.get("model", "")).replace(
            "classpinn_", "") or "pinn"
        model = step2.build_classpinn(preset, device)
        mtype = "classical"

    elif fam == "qpinn":
        model = step2.build_qpinn(
            n_qubits     = int(cfg.get("n_qubits", 7)),
            n_layers     = int(cfg.get("n_layers", 4)),
            n_enc_layers = int(cfg.get("n_enc_layers", 1)),
            encoding     = cfg.get("encoding", "arctan"),
            ansatz       = cfg.get("ansatz", "u_ring"),
            device       = device,
        )
        mtype = "qpinn"

    elif fam == "qapinn":
        model = step2.build_qapinn(
            hidden      = int(cfg.get("hidden", 128)),
            depth       = int(cfg.get("depth", 4)),
            q_layer_idx = cfg.get("q_layer_idx"),
            n_qubits    = int(cfg.get("n_qubits", 6)),
            n_layers    = int(cfg.get("n_layers", 2)),
            encoding    = cfg.get("encoding", "angle_full"),
            ansatz      = cfg.get("ansatz", "hardware_efficient"),
            activation  = cfg.get("activation", "tanh"),
            lambda_q    = float(cfg.get("lambda_q", 0.1)),
            use_physics = bool(cfg.get("use_physics", False)),
            device      = device,
        )
        mtype = "classical"
    else:
        raise ValueError(f"unhandled family {fam}")

    # Normalisation stats — the physics-loss path expects them on the model.
    if data is not None:
        for attr, key in (("_Xm", "Xm"), ("_Xs", "Xs"),
                          ("_Ym", "Ym"), ("_Ys", "Ys")):
            if key in data:
                setattr(model, attr, data[key].to(device))

    ckpt = torch.load(str(entry["path"]), map_location=device,
                      weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state"],
                                                strict=False)
    if missing or unexpected:
        # strict=False plus an explicit report: a silent partial load is how
        # you end up explaining a randomly-initialised circuit.
        print(f"    ! state_dict mismatch  missing={len(missing)}  "
              f"unexpected={len(unexpected)}")
        for k in list(missing)[:6]:
            print(f"        missing:    {k}")
        for k in list(unexpected)[:6]:
            print(f"        unexpected: {k}")
        if len(missing) > len(list(model.state_dict())) // 2:
            raise RuntimeError(
                "more than half the parameters failed to load — the config "
                "in this checkpoint does not describe these weights")

    model.eval()
    return model, mtype


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EXTRA PLOTS v3 DOES NOT DRAW
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_bar(step3, mean_abs: np.ndarray, model_name: str, save_path: Path):
    """v3 computes SHAP but never plots it. Same visual language as the rest."""
    order = np.argsort(mean_abs)[::-1]
    fnames = [step3.FEATURE_NAMES[i] for i in order]
    vals = np.asarray(mean_abs)[order]
    cols = [step3.COLOURS[i % len(step3.COLOURS)] for i in order]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor=step3.DARK_BG)
    step3._apply_dark_style(ax)
    ax.barh(fnames, vals, color=cols, alpha=0.85)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"SHAP (KernelExplainer) — {model_name}")
    fig.tight_layout()
    step3._save_fig(fig, save_path)


def plot_cross_model_importance(step3, reports: Dict[str, Dict],
                                method: str, save_path: Path,
                                title: str, xlabel: str):
    """
    Grouped bars: one cluster per input feature, one bar per model.

    This is the plot the whole exercise is for. A single model's importance
    bars tell you what that model latched onto; putting MLP, ClassPINN, QPINN
    and QAPINN side by side tells you whether the physics loss and the
    quantum circuit changed *what the model uses*, or only how well it fits.
    """
    models = [m for m in reports
              if _extract_importance(reports[m], method) is not None]
    if not models:
        return

    NI = step3.NI
    vals = np.zeros((len(models), NI))
    for mi, m in enumerate(models):
        v = np.asarray(_extract_importance(reports[m], method), dtype=float)
        v = np.nan_to_num(v[:NI], nan=0.0, posinf=0.0, neginf=0.0)
        denom = np.abs(v).sum()
        vals[mi] = v / denom if denom > 0 else v      # normalise → shares

    x = np.arange(NI)
    width = min(0.8 / len(models), 0.25)
    fig, ax = plt.subplots(figsize=(max(11, 1.6 * NI), 4.6),
                           facecolor=step3.DARK_BG)
    step3._apply_dark_style(ax)
    for mi, m in enumerate(models):
        ax.bar(x + mi * width, vals[mi], width, label=m,
               color=step3.COLOURS[mi % len(step3.COLOURS)], alpha=0.88)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(step3.FEATURE_NAMES, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(xlabel)
    ax.set_title(title)
    ax.legend(facecolor=step3.CARD_BG, edgecolor=step3.BORDER,
              labelcolor=step3.TEXT_COL, fontsize=8, ncol=min(len(models), 4))
    fig.tight_layout()
    step3._save_fig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3b — CORRECTED QUANTUM METRICS
# ══════════════════════════════════════════════════════════════════════════════
#
# Two of v3's quantum methods return values that are not physically possible,
# for reasons that live in the tensor algebra and so occur with the real
# utilities_quantum_torch just as much as with any stand-in.
#
#   qubit_entropy       — the reduced density matrix is built with
#                         `(...).sum(-1).sum(-1)`. The first sum contracts the
#                         other-qubit index and yields [B]; the second then
#                         contracts the *batch* to a scalar, which broadcasts
#                         into rho[:, a, b]. Every entry of rho becomes the sum
#                         over the whole batch, so Tr(rho) ≈ B instead of 1,
#                         the eigenvalues are ≫ 1, and S = −Σ λ log λ comes out
#                         large and negative. Von Neumann entropy is bounded in
#                         [0, log 2] for one qubit, so any negative number here
#                         is a defect, not a finding.
#
#   quantum_state_probe — `torch.norm(rho1 - rho2, p="fro")` reduces over the
#                         batch axis as well as the two matrix axes, so the
#                         reported distance grows like sqrt(B) and is not
#                         comparable between runs with different sample_size.
#                         It also materialises B copies of a 2^n x 2^n matrix,
#                         which is 4^n memory and dies above ~10 qubits.
#
# Both are rewritten below. The probe uses the exact identity for pure states,
#     ‖ |psi1><psi1| − |psi2><psi2| ‖_F = sqrt(2 (1 − |<psi1|psi2>|²)),
# which gives the same number in O(2^n) instead of O(4^n) and stays inside
# [0, sqrt(2)] where it belongs.

def compute_shap_fixed(model, X_tr, X_te, Y_te, model_type="classical",
                       device="cpu", background_size=50, explain_size=50,
                       nsamples=100, **_):
    """
    KernelExplainer attribution, tolerant of both shap return conventions.

    v3 does `np.mean([np.abs(sv).mean(0) for sv in shap_vals], axis=0)`, which
    assumes shap_values returns a *list* of [n_explain, n_features] arrays, one
    per output — the old multi-output convention. Modern shap returns a single
    ndarray of shape [n_explain, n_features, n_outputs]. Iterating that array
    iterates over samples, so each `sv` is [n_features, n_outputs], `.mean(0)`
    collapses to length n_outputs = 3, and the caller's loop over 7 feature
    names raises IndexError. Both shapes are handled here.
    """
    try:
        import shap
    except ImportError:
        print("\n[ SHAP ]  ✗ shap not installed — run:  pip install shap")
        return None

    print("\n[ SHAP — KernelExplainer ]")
    NI_ = X_te.shape[-1]

    def predict_fn(X_np):
        X_t = torch.tensor(X_np, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            if model_type == "qpinn":
                n_q = model.n_qubits
                Xq = X_t[:, torch.arange(n_q) % NI_].to(device)
                pred, _ = model(Xq, None)
            else:
                pred, _ = model(X_t.to(device))
        return pred.cpu().numpy()

    bg = X_tr[torch.randperm(len(X_tr))[:background_size]].numpy()
    exp = X_te[:explain_size].numpy()

    explainer = shap.KernelExplainer(predict_fn, bg)
    sv = explainer.shap_values(exp, nsamples=nsamples, silent=True)

    arr = np.asarray(sv) if not isinstance(sv, list) else np.stack(
        [np.asarray(s) for s in sv], axis=-1)
    # arr is now [n_explain, n_features] or [n_explain, n_features, n_outputs]
    if arr.ndim == 3:
        mean_abs = np.abs(arr).mean(axis=(0, 2))
    elif arr.ndim == 2:
        mean_abs = np.abs(arr).mean(axis=0)
    else:
        print(f"  ✗ unexpected SHAP shape {arr.shape} — skipping")
        return None

    if mean_abs.shape[0] != NI_:
        print(f"  ✗ SHAP returned {mean_abs.shape[0]} values for {NI_} "
              f"features — skipping")
        return None

    for fi in range(NI_):
        print(f"    {FEATURE_NAMES_FALLBACK[fi]:<10}: {mean_abs[fi]:.6f}")
    return mean_abs, arr


def detach_tensor_args(fn):
    """
    Give every XAI method its own detached copies of X and Y.

    v3 writes `X_in = X_te.to(device).requires_grad_(True)`. When device is
    CPU, `.to("cpu")` returns *the same object*, so requires_grad_ mutates the
    caller's test tensor in place. gradient_sensitivity runs first, so by the
    time integrated_gradients slices that tensor the slice is a non-leaf, its
    `.grad` is never populated, and IG dies on
    `'NoneType' object has no attribute 'abs'`.

    On CUDA `.to()` copies, so the bug is invisible on a GPU box and appears
    the moment anyone runs the same script on CPU. Detaching at the boundary
    fixes it for every method at once and stops the caller's tensors being
    silently modified.
    """
    def wrapper(model, *args, **kwargs):
        args = tuple(a.detach().clone() if torch.is_tensor(a) else a
                     for a in args)
        kwargs = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                  for k, v in kwargs.items()}
        return fn(model, *args, **kwargs)
    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    return wrapper


def _statevector(model, x_in, device):
    """Encode + ansatz, honouring n_enc_layers re-uploading if present."""
    from utilities_quantum_torch import torch_encode, torch_ansatz
    x_in = x_in.to(device)
    n_q = model.n_qubits
    state = torch.zeros(x_in.shape[0], 2 ** n_q,
                        dtype=torch.cfloat, device=device)
    state[:, 0] = 1.0 + 0j
    n_rep = int(getattr(model, "n_enc_layers", 1) or 1)
    for _ in range(n_rep):
        state = torch_encode(state, x_in, n_q, model.encoding)
        state = torch_ansatz(state, model.weights, n_q, model.ansatz)
    return state


def qubit_entropy_fixed(model, X_te, Y_te, device="cpu", sample_size=100,
                        **_) -> np.ndarray:
    """Mean per-sample Von Neumann entropy of each qubit. Bounded [0, log 2]."""
    print("\n[ Qubit Von Neumann Entropy — corrected ]")
    model.eval()
    n_q = model.n_qubits
    Xq = X_te[:sample_size]
    idx = torch.arange(n_q) % Xq.shape[-1]
    Xq = Xq[:, idx] if Xq.shape[-1] != n_q else Xq

    with torch.no_grad():
        state = _statevector(model, Xq, device)
        state = state / (state.norm(dim=-1, keepdim=True) + 1e-12)
        B = state.shape[0]

        ents = []
        for qi in range(n_q):
            sv_r = state.reshape(B, 2 ** qi, 2, 2 ** (n_q - qi - 1))
            rho = torch.zeros(B, 2, 2, dtype=torch.cfloat, device=state.device)
            for a in range(2):
                for b in range(2):
                    # one sum only: contract the other-qubit index, keep batch
                    rho[:, a, b] = (
                        sv_r[:, :, a, :].reshape(B, -1) *
                        sv_r[:, :, b, :].conj().reshape(B, -1)
                    ).sum(-1)
            ev = torch.linalg.eigvalsh(rho).real.clamp(min=1e-12)
            S = -(ev * torch.log(ev)).sum(-1)      # [B]
            ents.append(float(S.mean().item()))

    ents = np.array(ents)
    for qi in range(n_q):
        print(f"    q{qi} ({FEATURE_NAMES_FALLBACK[qi % 7]:<8}): "
              f"S = {ents[qi]:.4f}   (max {np.log(2):.4f})")
    return ents


def quantum_state_probe_fixed(model, X_te, Y_te, device="cpu", n_steps=20,
                              sample_size=50, **_) -> np.ndarray:
    """Mean per-sample ‖Δρ‖_F via the pure-state fidelity identity."""
    print("\n[ Quantum State Probe — corrected ]")
    model.eval()
    n_q = model.n_qubits
    NI_ = X_te.shape[-1]
    X_mean = X_te[:sample_size].mean(0, keepdim=True)
    idx = torch.arange(n_q) % NI_
    X_base = X_mean[:, idx].expand(max(sample_size, 1), -1).contiguous()

    with torch.no_grad():
        sv_base = _statevector(model, X_base.clone(), device)
        sv_base = sv_base / (sv_base.norm(dim=-1, keepdim=True) + 1e-12)

        dists = []
        for fi in range(NI_):
            tot = 0.0
            for v in torch.linspace(-1.0, 1.0, n_steps):
                Xv = X_base.clone()
                Xv[:, fi % n_q] = v
                sv = _statevector(model, Xv, device)
                sv = sv / (sv.norm(dim=-1, keepdim=True) + 1e-12)
                ov = (sv_base.conj() * sv).sum(-1).abs() ** 2       # [B]
                tot += float(torch.sqrt((2 * (1 - ov)).clamp(min=0))
                             .mean().item())
            dists.append(tot / n_steps)

    dists = np.array(dists)
    for fi in range(min(NI_, 7)):
        print(f"    {FEATURE_NAMES_FALLBACK[fi]:<10}: {dists[fi]:.6f}"
              f"   (max {np.sqrt(2):.4f})")
    return dists


FEATURE_NAMES_FALLBACK = ["x", "t", "p_ratio", "mu", "rho_L", "rho_R", "p_R"]


def find_quantum_weights(model: nn.Module) -> Optional[np.ndarray]:
    """
    Fallback for circuit_weight_analysis.

    v3 looks for exactly `model.weights` or `model.quantum_layer.q_weights`.
    If TorchQAPINN names its quantum layer anything else, the weight histogram
    is silently skipped. This scans named_parameters for the usual suspects
    instead of depending on one attribute path.
    """
    keys = ("q_weight", "qweight", "theta", "ansatz", "circuit", "quantum")
    hits = [p.detach().cpu().numpy().ravel()
            for n, p in model.named_parameters()
            if any(k in n.lower() for k in keys)]
    if not hits:
        return None
    return np.concatenate(hits)


def _extract_importance(report: Dict, method: str):
    m = report.get("metrics", {})
    if method == "permutation":
        perm = m.get("permutation_importance")
        if not perm:
            return None
        return [perm[f]["mean"] for f in perm]
    if method == "gradient":
        g = m.get("gradient_sensitivity")
        return g["global"] if g else None
    if method == "integrated_gradients":
        return m.get("integrated_gradients")
    if method == "shap":
        return m.get("shap_mean_abs")
    return None


def plot_rank_agreement(step3, reports: Dict[str, Dict], save_path: Path):
    """
    Spearman rank correlation between every pair of (model, method) rankings.

    Attribution methods disagree; that is normal and informative. Permutation
    importance measures what breaks the *fit*, gradients measure local
    slope, IG measures path-integrated contribution from a zero baseline. A
    feature that ranks top on all three is a real dependency. A feature that
    ranks top on gradients only is usually a sharp local response, not a
    globally important variable. This heatmap makes that visible instead of
    leaving four separate bar charts for the reader to reconcile by eye.
    """
    methods = ["permutation", "gradient", "integrated_gradients", "shap"]
    labels, vectors = [], []
    for m in sorted(reports):
        for meth in methods:
            v = _extract_importance(reports[m], meth)
            if v is None:
                continue
            v = np.nan_to_num(np.asarray(v, dtype=float)[:step3.NI])
            labels.append(f"{m}·{meth[:4]}")
            vectors.append(v)

    if len(vectors) < 2:
        return

    def _rank(v):
        order = np.argsort(np.argsort(-v))
        return order.astype(float)

    R = np.array([_rank(v) for v in vectors])
    n = len(R)
    C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = R[i] - R[i].mean(), R[j] - R[j].mean()
            d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
            c = float((a * b).sum() / d) if d > 0 else 0.0
            C[i, j] = C[j, i] = c

    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * n, 0.9 + 0.5 * n),
                           facecolor=step3.DARK_BG)
    step3._apply_dark_style(ax)
    im = ax.imshow(C, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=60,
                                                ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Rank agreement across models and attribution methods",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spearman ρ of feature ranking", color=step3.MUTED)
    cbar.ax.tick_params(colors=step3.MUTED)
    fig.tight_layout()
    step3._save_fig(fig, save_path)


def plot_accuracy_vs_models(step3, entries: List[Dict], save_path: Path):
    """Test R² per run — context for every attribution figure."""
    rows = [(e["label"], e.get("r2")) for e in entries if e.get("r2") is not None]
    if not rows:
        return
    rows.sort(key=lambda r: r[1], reverse=True)
    names = [r[0] for r in rows]
    r2s = [r[1] for r in rows]
    cols = ["#34d399" if v > 0 else "#ef4444" for v in r2s]

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 2),
                           facecolor=step3.DARK_BG)
    step3._apply_dark_style(ax)
    ax.barh(names, r2s, color=cols, alpha=0.85)
    ax.axvline(0, color=step3.MUTED, lw=0.8, ls="--")
    ax.set_xlabel("test R²")
    ax.set_title("Model accuracy — explanations are only meaningful "
                 "for models that fit")
    fig.tight_layout()
    step3._save_fig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PER-MODEL XAI RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_one(step2, step3, entry, data, device, args) -> Optional[Dict]:
    label = entry["label"]
    outdir = Path(args.out_dir) / label
    report_path = outdir / "xai_report.json"
    print(f"\n{'━' * 72}")
    print(f"  {label}   [{entry['family']}]   {entry['path']}")
    if entry.get("r2") is not None:
        print(f"  reported test R² = {entry['r2']:.4f}")
    print(f"{'━' * 72}")

    # Resume at model granularity. A completed JSON report is the durable
    # marker written by run_full_xai; load it into the combined summary and
    # immediately continue to the next checkpoint instead of recomputing XAI.
    if report_path.is_file() and not args.force:
        try:
            with open(report_path, encoding="utf-8") as handle:
                report = json.load(handle)
            if not isinstance(report, dict) or not isinstance(
                    report.get("metrics"), dict):
                raise ValueError("missing metrics object")
            print(f"  ✓ existing JSON report — skipping XAI: {report_path}")
            # Field plots need a live model even when its XAI was cached.
            if args.fields:
                model, mtype = rebuild_model(step2, entry, device, data)
                adapter = XAIAdapter(
                    model, mtype, qpinn_y=args.qpinn_y).to(device).eval()
                entry["model"] = model
                entry["mtype"] = mtype
                entry["adapter"] = adapter
            return report
        except Exception as e:
            print(f"  ! existing report is invalid ({e}); recomputing")

    try:
        model, mtype = rebuild_model(step2, entry, device, data)
    except Exception as e:
        print(f"  ✗ rebuild failed: {type(e).__name__}: {e}")
        if args.traceback:
            traceback.print_exc()
        return None

    adapter = XAIAdapter(model, mtype, qpinn_y=args.qpinn_y).to(device)
    adapter.eval()

    X_tr, Y_tr = data["X_tr"], data["Y_tr"]
    X_te, Y_te = data["X_te"], data["Y_te"]
    if args.max_test and len(X_te) > args.max_test:
        X_te, Y_te = X_te[:args.max_test], Y_te[:args.max_test]

    try:
        report = step3.run_full_xai(
            adapter, label, mtype,
            X_tr, Y_tr, X_te, Y_te,
            device=device,
            out_dir=args.out_dir,
            run_shap=args.shap,
            run_qstate=(mtype == "qpinn"),
            run_interaction=args.interaction,
            n_perm_repeats=args.perm_repeats,
            ig_steps=args.ig_steps,
            ig_samples=args.ig_samples,
        )
    except Exception as e:
        print(f"  ✗ XAI failed: {type(e).__name__}: {e}")
        if args.traceback:
            traceback.print_exc()
        return None

    # SHAP plot — v3 stores the numbers but never draws them.
    shap_vals = report.get("metrics", {}).get("shap_mean_abs")
    if shap_vals:
        try:
            plot_shap_bar(step3, np.asarray(shap_vals), label,
                          outdir / "shap_importance.png")
        except Exception as e:
            print(f"  ! shap plot failed: {e}")

    # Circuit weights. v3 gates this behind
    # `model_type == "classical" and hasattr(model, "quantum_layer")`, so pure
    # QPINN never got its weight histogram, and a QAPINN whose layer is named
    # anything else got skipped silently. Cover both here.
    has_quantum = (hasattr(model, "weights") or hasattr(model, "quantum_layer")
                   or find_quantum_weights(model) is not None)
    if has_quantum and "circuit_weights" not in report.get("metrics", {}):
        try:
            cw = step3.circuit_weight_analysis(model)
            w = cw.get("weights") if cw else None
            stats = cw.get("stats") if cw else None
            if w is None:
                w = find_quantum_weights(model)
                stats = None if w is None else {
                    "n_params": int(w.size),
                    "mean": float(w.mean()), "std": float(w.std()),
                    "min": float(w.min()), "max": float(w.max()),
                    "dead_frac": float((np.abs(w) < 0.05).mean()),
                    "sat_frac": float((np.abs(w) > np.pi - 0.1).mean()),
                }
            if w is not None:
                report.setdefault("metrics", {})["circuit_weights"] = {
                    "stats": stats}
                step3.plot_circuit_weights(
                    w, label, save_path=outdir / "circuit_weights.png")
        except Exception as e:
            print(f"  ! circuit weight analysis failed: {e}")

    entry["model"] = model
    entry["mtype"] = mtype
    entry["adapter"] = adapter
    return report


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CSV TABLES (formerly xai_table.py)
# ══════════════════════════════════════════════════════════════════════════════

TABLE_FEATURES = ["x", "t", "p_ratio", "mu", "rho_L", "rho_R", "p_R"]
TABLE_METHODS = ["permutation", "gradient", "integrated_gradients", "shap"]


def _table_extract(metrics, method):
    """Return one XAI method's raw feature-importance vector."""
    if method == "permutation":
        values = metrics.get("permutation_importance")
        if not values:
            return None
        return np.asarray([
            item.get("mean", np.nan) if isinstance(item, dict) else item
            for item in values.values()
        ], dtype=float)
    if method == "gradient":
        values = metrics.get("gradient_sensitivity")
        if not values:
            return None
        values = values.get("global") if isinstance(values, dict) else values
        return None if values is None else np.asarray(values, dtype=float)
    key = "shap_mean_abs" if method == "shap" else method
    values = metrics.get(key)
    return None if values is None else np.asarray(values, dtype=float)


def _table_shares(values, n_features):
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0,
                           posinf=0.0, neginf=0.0)[:n_features]
    if len(values) < n_features:
        values = np.pad(values, (0, n_features - len(values)))
    values = np.abs(values)
    total = values.sum()
    return values / total if total > 0 else values


def _table_spearman(a, b):
    ra = np.argsort(np.argsort(-a)).astype(float)
    rb = np.argsort(np.argsort(-b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2))
    return float(np.sum(ra * rb) / denom) if denom > 0 else None


def _table_top3(share, names):
    order = np.argsort(-share)[:3]
    return ", ".join(f"{names[i]}({share[i]:.0%})" for i in order
                     if share[i] > 0)


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  ✓ table → {path}")


def write_xai_tables(summary, out_dir, feature_names=None, sort_by="r2"):
    """Convert the in-memory combined summary into four analysis tables."""
    models = summary.get("models", summary)
    if not models:
        print("  ! no successful models; CSV tables not written")
        return []
    names = list(feature_names or TABLE_FEATURES)
    n_features = len(names)
    items = list(models.items())
    if sort_by == "r2":
        items.sort(key=lambda kv: -(kv[1].get("test_r2")
                                    if kv[1].get("test_r2") is not None
                                    else -1e9))
    else:
        items.sort(key=lambda kv: kv[0])

    model_rows, wide_rows, long_rows, quantum_rows = [], [], [], []
    for label, model in items:
        metrics = model.get("metrics", {}) or {}
        config = model.get("config", {}) or {}
        family = model.get("family", "")
        vectors = {}
        for method in TABLE_METHODS:
            raw = _table_extract(metrics, method)
            if raw is None:
                continue
            share = _table_shares(raw, n_features)
            vectors[method] = share
            rank = np.argsort(np.argsort(-share)) + 1
            for i, feature in enumerate(names):
                long_rows.append([
                    label, family, method, feature,
                    float(raw[i]) if i < len(raw) and np.isfinite(raw[i]) else "",
                    float(share[i]), int(rank[i]),
                ])
            wide_rows.append([label, family, method]
                             + [float(value) for value in share])

        permutation = vectors.get("permutation")
        agreement = {}
        if permutation is not None:
            for method in TABLE_METHODS[1:]:
                if method in vectors:
                    agreement[method] = _table_spearman(
                        permutation, vectors[method])
        model_rows.append([
            label, family, model.get("test_r2"), model.get("test_mse"),
            config.get("n_qubits"), config.get("n_layers"),
            config.get("n_enc_layers"), config.get("encoding"),
            config.get("ansatz"), config.get("n_params"),
            *[_table_top3(vectors[m], names) if m in vectors else ""
              for m in TABLE_METHODS],
            agreement.get("gradient"), agreement.get("integrated_gradients"),
            agreement.get("shap"),
        ])

        weight_stats = ((metrics.get("circuit_weights") or {}).get("stats")
                        or {})
        entropy = metrics.get("qubit_entropy")
        probe = metrics.get("quantum_state_probe")
        if weight_stats or entropy is not None or probe is not None:
            entropy = np.asarray(entropy, float) if entropy is not None else None
            probe = np.asarray(probe, float) if probe is not None else None
            quantum_rows.append([
                label, family, config.get("n_qubits"), config.get("n_layers"),
                weight_stats.get("n_params"), weight_stats.get("mean"),
                weight_stats.get("std"), weight_stats.get("dead_frac"),
                weight_stats.get("sat_frac"),
                float(entropy.mean()) if entropy is not None and entropy.size else "",
                float(entropy.max()) if entropy is not None and entropy.size else "",
                float(probe.mean()) if probe is not None and probe.size else "",
            ])

    out_dir = Path(out_dir)
    paths = [out_dir / "xai_models.csv", out_dir / "xai_importance.csv",
             out_dir / "xai_importance_long.csv"]
    _write_csv(paths[0],
               ["model", "family", "test_r2", "test_mse", "n_qubits",
                "n_layers", "n_enc_layers", "encoding", "ansatz", "n_params",
                "top3_permutation", "top3_gradient", "top3_ig", "top3_shap",
                "rho_perm_vs_grad", "rho_perm_vs_ig", "rho_perm_vs_shap"],
               model_rows)
    _write_csv(paths[1], ["model", "family", "method"]
               + [f"share_{name}" for name in names], wide_rows)
    _write_csv(paths[2],
               ["model", "family", "method", "feature", "value", "share",
                "rank"], long_rows)
    if quantum_rows:
        paths.append(out_dir / "xai_quantum.csv")
        _write_csv(paths[-1],
                   ["model", "family", "n_qubits", "n_layers", "weight_params",
                    "weight_mean", "weight_std", "dead_frac", "sat_frac",
                    "entropy_mean", "entropy_max", "state_probe_mean"],
                   quantum_rows)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the Step-3 XAI toolkit on every Step-2 model")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out_dir", default="./xai_results")
    p.add_argument("--force", action="store_true",
                   help="recompute XAI even when a model's xai_report.json "
                        "already exists")
    p.add_argument("--code_dir", default=None,
                   help="directory holding Step2/Step3/utilities (default: "
                        "this script's directory)")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--only", nargs="*", default=None,
                   choices=["mlp", "classpinn", "qpinn", "qapinn"],
                   help="restrict to these families (default: all found)")
    p.add_argument("--top_k", type=int, default=0,
                   help="within each family keep only the top-k runs by test "
                        "R² (0 = keep all). Useful for the QPINN sweep.")
    p.add_argument("--skip_sweep", action="store_true",
                   help="ignore anything under a qpinn_sweep/ directory")

    p.add_argument("--n_scenarios", type=int, default=30)
    p.add_argument("--t_stride", type=int, default=6)
    p.add_argument("--max_test", type=int, default=4000,
                   help="cap test-set size used for XAI (0 = no cap)")

    p.add_argument("--shap", action="store_true", help="run SHAP (slow)")
    p.add_argument("--shap_nsamples", type=int, default=100,
                   help="KernelExplainer coalition samples per point")
    p.add_argument("--shap_background", type=int, default=50,
                   help="SHAP background sample count (v3 hard-codes 100)")
    p.add_argument("--shap_explain", type=int, default=50,
                   help="SHAP points to explain (v3 hard-codes 200)")
    p.add_argument("--interaction", action="store_true",
                   help="run the H-statistic (very slow: O(NI² · grid²))")
    p.add_argument("--no-ig", dest="no_ig", action="store_true",
                   help="skip integrated gradients (the slow default method)")
    p.add_argument("--perm_repeats", type=int, default=10)
    p.add_argument("--ig_steps", type=int, default=50)
    p.add_argument("--ig_samples", type=int, default=200)

    p.add_argument("--qpinn_y", choices=["zero", "true"], default="zero",
                   help="what to feed TorchQPINN's y-register during XAI. "
                        "'zero' matches Step 2's predict() and avoids target "
                        "leakage; 'true' reproduces v3's behaviour.")
    p.add_argument("--raw_quantum", action="store_true",
                   help="keep v3's original qubit_entropy / quantum_state_probe "
                        "instead of the corrected versions")
    p.add_argument("--fields", action="store_true",
                   help="also draw the ground-truth field comparison")
    p.add_argument("--scenario_idx", type=int, default=0)
    p.add_argument("--no_tables", action="store_true",
                   help="do not create the CSV summary tables")
    p.add_argument("--tables_dir", default=None,
                   help="CSV destination (default: <out_dir>/tables)")
    p.add_argument("--table_sort", choices=["r2", "name"], default="r2",
                   help="row ordering in the generated CSV tables")
    p.add_argument("--traceback", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    extra = [Path(args.code_dir).resolve()] if args.code_dir else []

    print("=" * 72)
    print("  XAI — all Step 2 models")
    print("=" * 72)
    step2, step3 = load_pipeline_modules(script_dir, extra)

    if args.no_ig:
        # run_full_xai always calls integrated_gradients; passing sample_size=0
        # would hand np.array([]).mean(axis=0) an empty array. Stub the
        # function instead so the pipeline shape stays intact.
        def _ig_stub(*a, **k):
            print("\n[ Integrated Gradients ]  skipped (--no-ig)")
            return np.zeros(step3.NI)
        step3.integrated_gradients = _ig_stub

    if not args.raw_quantum:
        step3.qubit_entropy = qubit_entropy_fixed
        step3.quantum_state_probe = quantum_state_probe_fixed
        print("  [patch] using corrected qubit_entropy / quantum_state_probe "
              "(--raw_quantum to keep v3's originals)")

    if args.shap:
        # Two problems with v3's compute_shap:
        #   * it assumes the old shap return convention (a list of arrays, one
        #     per output) and IndexErrors on modern shap, which returns a
        #     single [n_explain, n_features, n_outputs] ndarray;
        #   * background_size=100 / explain_size=200 are hard-coded and
        #     run_full_xai does not pass them through, so a statevector QPINN
        #     faces 100 x 200 x nsamples circuit evaluations — hours of work.
        # Swap in the corrected version with the user's sizes bound.
        import functools
        step3.compute_shap = functools.partial(
            compute_shap_fixed,
            background_size=args.shap_background,
            explain_size=args.shap_explain,
            nsamples=args.shap_nsamples)
        print(f"  [patch] SHAP corrected  background={args.shap_background} "
              f"explain={args.shap_explain} nsamples={args.shap_nsamples}")

    for _fn in ("permutation_importance", "gradient_sensitivity",
                "integrated_gradients", "compute_shap", "feature_interaction",
                "quantum_state_probe", "qubit_entropy"):
        if hasattr(step3, _fn):
            setattr(step3, _fn, detach_tensor_args(getattr(step3, _fn)))
    print("  [patch] XAI methods now receive detached input copies")

    device = torch.device(args.device)
    print(f"  device      : {device}")
    print(f"  results_dir : {args.results_dir}")
    print(f"  out_dir     : {args.out_dir}")

    # ── data ──────────────────────────────────────────────────────────────────
    from utilities_classical import load_ns_data
    data = load_ns_data(args.data_dir,
                        n_scenarios=args.n_scenarios,
                        t_stride=args.t_stride)
    print(f"  data        : train={len(data['X_tr'])}  test={len(data['X_te'])}")
    if args.max_test == 0:
        args.max_test = None

    # ── discover ──────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}\nScanning for checkpoints ...")
    entries = discover_checkpoints(Path(args.results_dir))
    if args.skip_sweep:
        entries = [e for e in entries
                   if "qpinn_sweep" not in str(e["path"]).replace("\\", "/")]
    if args.only:
        entries = [e for e in entries if e["family"] in args.only]

    if args.top_k:
        kept = []
        for fam in {e["family"] for e in entries}:
            fam_e = [e for e in entries if e["family"] == fam]
            fam_e.sort(key=lambda e: (e["r2"] is None, -(e["r2"] or 0)))
            kept.extend(fam_e[:args.top_k])
        entries = kept

    if not entries:
        print("\n  No checkpoints found. Step 2 writes `*_final.pt` inside "
              "results/<run_tag>/ — check --results_dir.")
        return

    # unique, readable labels
    seen = {}
    for e in entries:
        base = f"{e['family'].upper()}_{e['run_tag']}"
        n = seen.get(base, 0)
        seen[base] = n + 1
        e["label"] = base if n == 0 else f"{base}_{n}"

    print(f"  found {len(entries)} checkpoint(s):")
    for e in entries:
        r2 = f"R²={e['r2']:.4f}" if e.get("r2") is not None else "R²=?"
        print(f"    {e['label']:<52} {r2}")

    # ── run ───────────────────────────────────────────────────────────────────
    t0 = time.time()
    reports: Dict[str, Dict] = {}
    for e in entries:
        rep = run_one(step2, step3, e, data, device, args)
        if rep is not None:
            reports[e["label"]] = rep

    if not reports:
        print("\n  Nothing succeeded — rerun with --traceback for details.")
        return

    # ── cross-model figures ───────────────────────────────────────────────────
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'─' * 72}\nCross-model comparison ...")

    plot_accuracy_vs_models(step3, entries, out / "cmp_accuracy.png")
    plot_cross_model_importance(
        step3, reports, "permutation", out / "cmp_permutation_importance.png",
        "Permutation importance across models (normalised share of ΔMSE)",
        "share of total ΔMSE")
    plot_cross_model_importance(
        step3, reports, "gradient", out / "cmp_gradient_sensitivity.png",
        "Gradient sensitivity across models (normalised)",
        "share of total |∂out/∂x|")
    plot_cross_model_importance(
        step3, reports, "integrated_gradients", out / "cmp_integrated_gradients.png",
        "Integrated gradients across models (normalised)",
        "share of total |IG|")
    if args.shap:
        plot_cross_model_importance(
            step3, reports, "shap", out / "cmp_shap.png",
            "SHAP across models (normalised)", "share of total |SHAP|")
    plot_rank_agreement(step3, reports, out / "cmp_rank_agreement.png")

    # ── optional field comparison ─────────────────────────────────────────────
    if args.fields:
        try:
            loaded = {e["label"]: (e["adapter"], e["mtype"])
                      for e in entries if "adapter" in e}
            step3.plot_field_comparison(
                models=loaded, data_dir=args.data_dir,
                Xm=data["Xm"], Xs=data["Xs"], Ym=data["Ym"], Ys=data["Ys"],
                device=device, scenario_idx=args.scenario_idx,
                t_indices=None, n_x=256,
                save_path=out / "field_comparison.png")
            step3.plot_field_error(
                models=loaded, data_dir=args.data_dir,
                Xm=data["Xm"], Xs=data["Xs"], Ym=data["Ym"], Ys=data["Ys"],
                device=device, scenario_idx=args.scenario_idx,
                t_indices=None, n_x=256,
                save_path=out / "field_error.png")
        except Exception as e:
            print(f"  ! field comparison failed: {type(e).__name__}: {e}")
            if args.traceback:
                traceback.print_exc()

    # ── combined summary ──────────────────────────────────────────────────────
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results_dir": str(args.results_dir),
        "qpinn_y_register": args.qpinn_y,
        "models": {},
    }
    for e in entries:
        if e["label"] not in reports:
            continue
        rep = reports[e["label"]]
        perm = rep["metrics"].get("permutation_importance", {})
        ranked = sorted(perm.items(), key=lambda kv: kv[1]["mean"], reverse=True)
        summary["models"][e["label"]] = {
            "family": e["family"],
            "config": e["config"],
            "test_r2": e.get("r2"),
            "test_mse": e.get("mse"),
            "top_features_permutation": [k for k, _ in ranked[:3]],
            "metrics": rep["metrics"],
        }

    sp = out / "xai_all_models_summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n  ✓ summary → {sp}")

    # Generate publication/statistics-ready tables in the same run.  This
    # replaces the former second command, xai_table.py.
    if not args.no_tables:
        tables_dir = (Path(args.tables_dir) if args.tables_dir
                      else out / "tables")
        write_xai_tables(summary, tables_dir,
                         feature_names=data.get("feature_names"),
                         sort_by=args.table_sort)

    print(f"\n{'=' * 72}")
    print(f"  Done — {len(reports)}/{len(entries)} models explained "
          f"in {time.time() - t0:.0f}s")
    print(f"  Figures in {out}/")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
