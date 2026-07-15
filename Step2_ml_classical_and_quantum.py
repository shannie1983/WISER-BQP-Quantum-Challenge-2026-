"""
main.py — NS-PINN Training and Prediction Runner
=================================================
Runs multiple model configurations continuously, saves all outputs,
and produces a single combined plot of loss curves + R² results.

Usage
-----
    # Run all default configurations
    python main.py

    # Run specific models only
    python main.py --models mlp pinn fourier_pinn qpinn_nq7

    # Custom data directory and epochs
    python main.py --data ./data --epochs 60 --outdir ./results

    # List all available configurations and exit
    python main.py --list

Output files (all in --outdir)
-------------------------------
    predictions_{name}.npy   — dict: X_te, Y_te, pred, r2_test, Xm, Xs, Ym, Ys
    history_{name}.npy        — dict: tr, va, r2v, lcu (quantum), time, config
    r2_summary.csv            — one row per model: name, r2_test, params, time, epochs
    training_curves.png       — combined plot: loss curves + R² bar chart

Model setup sweep (MODEL_CONFIGS at bottom of file)
----------------------------------------------------
  Classical:
    mlp, pinn, fourier_pinn, hardbc_pinn,
    softadapt_pinn, dynratio_pinn, rar_pinn,
    best_classical, adaptive_combined, deep_rar

  Hybrid quantum (QAPINN) — nq=8 DEFAULT, no ancilla:
    qapinn_nq8         — nq=8 ★ DEFAULT (encoder 7→8→VQC→decoder)
    qapinn_nq5         — nq=5 ablation
    qapinn_nq3         — nq=3 ablation

  Full quantum (QPINN) — nq=8 DEFAULT, no ancilla:
    qpinn_nq8          — nq=8 ★ DEFAULT (Hilbert dim=256, all 7 features)
    qpinn_nq7          — nq=7 (dim=128)
    qpinn_nq3          — nq=3 ablation

  Full quantum + LCU loss — nq=8 DEFAULT, no ancilla:
    qpinn_nq8_lcu      — nq=8 ★ DEFAULT FLAGSHIP + NSQuantumPDELoss
    qpinn_nq7_lcu      — nq=7 + NSQuantumPDELoss

  NOTE: 'LCU loss' = classical Pauli block-encoding on VQC outputs.
        NOT ancilla qubits. Zero extra circuit qubits.
"""

from __future__ import annotations
import argparse, os, time, csv, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, TensorDataset

# ── Local imports ─────────────────────────────────────────────────────────────
from utilities_classical import (
    build_unified, unified_loss, load_ns_data, list_unified, PRESETS,
    UnifiedPINN, NI as NI_C, NO,
)
from utilities_quantum import (
    ScalableQPINN, ScalableQAPINN, forward_with_raw,
    NSQuantumPDELoss, QPINN_PRESETS,
    get_device, model_to_device, estimate_vram, qubit_info, NI as NI_Q,
)

assert NI_C == NI_Q == 7, "NI mismatch between utilities files"
NI = 7

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS
# Each entry: (display_name, build_fn, train_fn, extra_kwargs)
# ══════════════════════════════════════════════════════════════════════════════

def _build_classical(preset, **kw):
    return build_unified(preset, **kw)

def _build_qpinn(n_qubits, n_layers, **kw):
    return ScalableQPINN(n_qubits=n_qubits, n_layers=n_layers)

def _build_qapinn(n_qubits, n_layers, **kw):
    return ScalableQAPINN(n_qubits=n_qubits, n_layers=n_layers)

# MODEL_CONFIGS: dict of name → config dict
# Keys used by runner:
#   type       : 'classical' | 'qpinn' | 'qapinn'
#   build      : callable() → nn.Module
#   lcu        : bool — use NSQuantumPDELoss as extra loss
#   lcu_weight : float — scale factor for LCU loss
#   lr         : float — learning rate
#   batch_size : int or 'auto'
#   n_batches_ep: int or None — budget limit (quantum on CPU)
#   desc       : str  — human-readable description

MODEL_CONFIGS = {
    # ── Classical ──────────────────────────────────────────────────────────────
    'mlp': dict(
        type='classical', build=lambda: build_unified('mlp'),
        lcu=False, lr=2e-3, batch_size='auto', n_batches_ep=None,
        desc="MLP baseline",
    ),
    'pinn': dict(
        type='classical', build=lambda: build_unified('pinn'),
        lcu=False, lr=2e-3, batch_size='auto', n_batches_ep=None,
        desc="PINN static-λ (mass+mom+IC+BC)",
    ),
    'fourier_pinn': dict(
        type='classical', build=lambda: build_unified('fourier_pinn'),
        lcu=False, lr=2e-3, batch_size='auto', n_batches_ep=None,
        desc="FourierPINN (n_freq=64, σ=2.0)",
    ),
    'hardbc_pinn': dict(
        type='classical', build=lambda: build_unified('hardbc_pinn'),
        lcu=False, lr=2e-3, batch_size='auto', n_batches_ep=None,
        desc="HardBC PINN",
    ),
    'softadapt_pinn': dict(
        type='classical', build=lambda: build_unified('softadapt_pinn'),
        lcu=False, lr=2e-3, batch_size='auto', n_batches_ep=None,
        desc="SoftAdapt PINN",
    ),
    'dynratio_pinn': dict(
        type='classical', build=lambda: build_unified('dynratio_pinn'),
        lcu=False, lr=2e-3, batch_size='auto', n_batches_ep=None,
        desc="DynRatio PINN",
    ),
    'rar_pinn': dict(
        type='classical', build=lambda: build_unified('rar_pinn'),
        lcu=False, lr=2e-3, batch_size='auto', n_batches_ep=None,
        desc="RAR PINN (residual-adaptive collocation)",
    ),
    'best_classical': dict(
        type='classical', build=lambda: build_unified('best_classical'),
        lcu=False, lr=1e-3, batch_size='auto', n_batches_ep=None,
        desc="Best classical: AdaptFourier + Combined + HardBC + RAR",
    ),
    'adaptive_combined': dict(
        type='classical', build=lambda: build_unified('adaptive_combined'),
        lcu=False, lr=1e-3, batch_size='auto', n_batches_ep=None,
        desc="AdaptiveFourier + Combined loss",
    ),
    'deep_rar': dict(
        type='classical', build=lambda: build_unified('deep_rar'),
        lcu=False, lr=1e-3, batch_size='auto', n_batches_ep=None,
        desc="Deep Fourier + SoftAdapt + HardBC + RAR",
    ),
    # ─────────────────────────────────────────────────────────────────────────
    # HYBRID QAPINN  (classical encoder sees all NI=7 features → nq=8 VQC)
    # No ancilla qubits. Classical enc: Linear(7→128→8) → rich VQC → decoder.
    # ─────────────────────────────────────────────────────────────────────────
    'qapinn_nq8': dict(
        type='qapinn', build=lambda: ScalableQAPINN(n_qubits=8, n_layers=4),
        lcu=False, lr=8e-4, batch_size='auto', n_batches_ep=None,
        desc="ScalableQAPINN nq=8 ★ DEFAULT HYBRID (enc 7→8→VQC→dec, no ancilla)",
    ),
    # Ablation variants (smaller nq for speed comparison)
    'qapinn_nq3': dict(
        type='qapinn', build=lambda: ScalableQAPINN(n_qubits=3, n_layers=4),
        lcu=False, lr=8e-4, batch_size='auto', n_batches_ep=None,
        desc="ScalableQAPINN nq=3 ablation (enc 7→3→VQC→dec)",
    ),
    'qapinn_nq5': dict(
        type='qapinn', build=lambda: ScalableQAPINN(n_qubits=5, n_layers=6),
        lcu=False, lr=8e-4, batch_size='auto', n_batches_ep=None,
        desc="ScalableQAPINN nq=5 ablation",
    ),
    # ─────────────────────────────────────────────────────────────────────────
    # FULL QUANTUM QPINN  (nq=8 system qubits, NO ancilla)
    # Hilbert dim = 2^8 = 256.  Encodes all 7 features (nq=8 ≥ NI=7) ✓
    # ─────────────────────────────────────────────────────────────────────────
    'qpinn_nq8': dict(
        type='qpinn', build=lambda: ScalableQPINN(n_qubits=8, n_layers=4),
        lcu=False, lr=5e-4, batch_size=64, n_batches_ep=20,
        desc="ScalableQPINN nq=8 ★ DEFAULT PURE QUANTUM (dim=256, no ancilla)",
    ),
    # Ablation variants
    'qpinn_nq7': dict(
        type='qpinn', build=lambda: ScalableQPINN(n_qubits=7, n_layers=4),
        lcu=False, lr=8e-4, batch_size=64, n_batches_ep=20,
        desc="ScalableQPINN nq=7 (dim=128, all 7 features, no ancilla)",
    ),
    'qpinn_nq3': dict(
        type='qpinn', build=lambda: ScalableQPINN(n_qubits=3, n_layers=4),
        lcu=False, lr=8e-4, batch_size=4096, n_batches_ep=None,
        desc="ScalableQPINN nq=3 ablation (dim=8)",
    ),
    # ─────────────────────────────────────────────────────────────────────────
    # FULL QUANTUM + LCU LOSS  (nq=8 system qubits, NO ancilla)
    # NSQuantumPDELoss applies LCU block-encoding to raw VQC ⟨Z⟩ outputs.
    # The 4 LCU channels are classical Pauli arithmetic — NOT extra qubits.
    # ─────────────────────────────────────────────────────────────────────────
    'qpinn_nq8_lcu': dict(
        type='qpinn', build=lambda: ScalableQPINN(n_qubits=8, n_layers=4),
        lcu=True, lcu_weight=0.1, lr=5e-4, batch_size=64, n_batches_ep=20,
        desc="ScalableQPINN nq=8 + LCU loss ★ DEFAULT FLAGSHIP (no ancilla)",
    ),
    'qpinn_nq7_lcu': dict(
        type='qpinn', build=lambda: ScalableQPINN(n_qubits=7, n_layers=4),
        lcu=True, lcu_weight=0.1, lr=8e-4, batch_size=64, n_batches_ep=20,
        desc="ScalableQPINN nq=7 + LCU loss (no ancilla)",
    ),
}

DEFAULT_RUN = [
    # Classical
    'mlp', 'pinn', 'fourier_pinn', 'hardbc_pinn',
    'softadapt_pinn', 'rar_pinn',
    # Hybrid quantum — nq=8 default, no ancilla
    'qapinn_nq8',
    # Full quantum — nq=8 default, no ancilla
    'qpinn_nq8',
    # Full quantum + LCU loss — nq=8 default, no ancilla
    'qpinn_nq8_lcu',
]


# ══════════════════════════════════════════════════════════════════════════════
# AUTO BATCH SIZE
# ══════════════════════════════════════════════════════════════════════════════

def auto_batch(model, cfg, device):
    if cfg.get('batch_size') not in (None, 'auto'):
        return cfg['batch_size']
    is_q = isinstance(model, (ScalableQPINN, ScalableQAPINN))
    if not is_q:
        return 4096
    info = estimate_vram(model, 512)
    if device.type == 'cuda':
        # Use 60% of VRAM
        try:
            free = torch.cuda.get_device_properties(0).total_memory
            nq = model.n_q; nl = model.n_l; n_pairs = nq*(nq-1)//2
            n_gates = (nq*6+n_pairs*3)*nl
            bps = 2**nq * 8 * (1+n_gates)
            B = int(free*0.6/bps)
            return max(16, min(4096, 2**int(np.log2(max(B,1)))))
        except Exception:
            return 256
    else:
        # CPU: conservative
        dim = 2**model.n_q
        if dim<=32: return 4096
        if dim<=128: return 64
        return 32


# ══════════════════════════════════════════════════════════════════════════════
# R² HELPER
# ══════════════════════════════════════════════════════════════════════════════

def r2_score(yp, yt):
    with torch.no_grad():
        ss_res = ((yt - yp)**2).sum().item()
        ss_tot = ((yt - yt.mean(0))**2).sum().item() + 1e-12
    return 1 - ss_res / ss_tot


def eval_r2(model, X, Y, device, chunk=512, is_qpinn=False):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            xb = X[i:i+chunk].to(device, non_blocking=True)
            if is_qpinn:
                pred, _ = forward_with_raw(model, xb)
            else:
                pred = model(xb)
            preds.append(pred.cpu())
    yp = torch.cat(preds)
    return r2_score(yp, Y)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def train_classical(model, cfg, data, epochs, device, print_every=5):
    """Train a UnifiedPINN (or any classical model)."""
    Xm=data['Xm'].to(device); Xs=data['Xs'].to(device)
    model._Xm=Xm; model._Xs=Xs
    model._Ym=data['Ym'].to(device); model._Ys=data['Ys'].to(device)
    B = auto_batch(model, cfg, device)
    loader = DataLoader(TensorDataset(data['X_tr'], data['Y_tr']),
                        batch_size=B, shuffle=True, pin_memory=(device.type=='cuda'))
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 2e-3))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    hist = {'tr':[], 'va':[], 'r2v':[], 'time':[]}
    t0 = time.time()

    for ep in range(1, epochs+1):
        model.train(); el = 0.; log_acc = {}
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss, log = unified_loss(model, xb, yb, ep, Xm, Xs,
                                     data['p_range'], data['mu_range'])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += log['data']
            for k,v in log.items(): log_acc[k] = log_acc.get(k,0.)+v

        sched.step()
        # Update adaptive weights
        if hasattr(model, 'update'):
            avg_log = {k:v/len(loader) for k,v in log_acc.items()}
            model.update(ep, avg_log)

        r2v = eval_r2(model, data['X_va'], data['Y_va'], device)
        vm  = float(nn.functional.mse_loss(
            model(data['X_va'][:512].to(device)).cpu(), data['Y_va'][:512]))
        hist['tr'].append(el/len(loader)); hist['va'].append(vm)
        hist['r2v'].append(r2v); hist['time'].append(time.time()-t0)

        #if ep % print_every == 0 or ep == 1:
        #    elapsed = time.time()-t0; eta=(elapsed/ep)*(epochs-ep)
        #    print(f"    ep{ep:4d}/{epochs}  R²={r2v:.4f}  val={vm:.5f}  "
        #          f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
        # BEFORE (single line):
        if ep % print_every == 0 or ep == 1:
            print(f"    ep{ep:4d}/{epochs}  R²=...  val=...  elapsed=...  ETA=...")

        # AFTER (two lines — line 1: accuracy/timing, line 2: loss breakdown):
        if ep % print_every == 0 or ep == 1:
            print(f"    ep{ep:4d}/{epochs}  R2={r2v:.4f}  val={vm:.5f}  lr={lr_cur:.2e}  elapsed=...  ETA=...")
            print(f"              L_data={L_data:.5f}  L_phys={L_phys:.5f}  ramp={ramp:.2f}"
                  f"  [mass=...  mom=...  ic=...  bc=...]")   # physics terms shown only when ramp > 0

    return hist


def train_qpinn(model, cfg, data, epochs, device, print_every=5):
    """Train a ScalableQPINN (or ScalableQAPINN), optionally with LCU loss."""
    model._Ym = data['Ym']; model._Ys = data['Ys']
    use_lcu = cfg.get('lcu', False)
    lcu_weight = cfg.get('lcu_weight', 0.1)
    is_qpinn = isinstance(model, ScalableQPINN)
    B = auto_batch(model, cfg, device)
    n_batches_ep = cfg.get('n_batches_ep')
    RAMP_LCU = max(1, epochs // 4)   # start LCU after 25% of epochs

    # Build LCU loss module if needed
    lcu_loss_fn = None
    if use_lcu:
        lcu_loss_fn = NSQuantumPDELoss(
            n_qubits=model.n_q, mode='separate',
            w_phys=0.5, w_fid=0.5, w_recon=0.1).to(device)

    all_params = list(model.parameters())
    if lcu_loss_fn: all_params += list(lcu_loss_fn.parameters())
    opt   = torch.optim.Adam(all_params, lr=cfg.get('lr', 8e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    loader = DataLoader(TensorDataset(data['X_tr'], data['Y_tr']),
                        batch_size=B, shuffle=True)
    loader_it = iter(loader)
    hist = {'tr':[], 'va':[], 'r2v':[], 'lcu':[], 'time':[]}
    t0 = time.time()

    for ep in range(1, epochs+1):
        model.train()
        if lcu_loss_fn: lcu_loss_fn.train()
        el = 0.; lcu_el = 0.; nb = 0
        limit = n_batches_ep or len(loader)
        r_lcu = max(0., min(1., (ep-RAMP_LCU)/max(epochs-RAMP_LCU,1)))

        for _ in range(limit):
            try: xb, yb = next(loader_it)
            except StopIteration: loader_it=iter(loader); xb,yb=next(loader_it)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()

            if is_qpinn:
                pred, raw = forward_with_raw(model, xb)
            else:
                pred = model(xb); raw = None

            L_data = nn.functional.mse_loss(pred, yb)
            loss = L_data

            if use_lcu and lcu_loss_fn and r_lcu > 0 and raw is not None:
                L_lcu = lcu_loss_fn(raw, y_true=yb)
                loss = loss + r_lcu * lcu_weight * L_lcu
                lcu_el += float(L_lcu.detach())

            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 0.5)
            opt.step()
            el += float(L_data.detach()); nb += 1

        sched.step()
        r2v = eval_r2(model, data['X_va'], data['Y_va'], device,
                      is_qpinn=is_qpinn)
        with torch.no_grad():
            xv = data['X_va'][:min(128,len(data['X_va']))].to(device)
            yv = data['Y_va'][:min(128,len(data['Y_va']))]
            if is_qpinn: pv,_ = forward_with_raw(model,xv)
            else: pv = model(xv)
            vm = float(nn.functional.mse_loss(pv.cpu(), yv))

        hist['tr'].append(el/nb); hist['va'].append(vm)
        hist['r2v'].append(r2v); hist['lcu'].append(lcu_el/max(nb,1))
        hist['time'].append(time.time()-t0)

        if ep % print_every == 0 or ep == 1:
            elapsed = time.time()-t0; eta=(elapsed/ep)*(epochs-ep)
            lcu_str = f"  lcu={lcu_el/max(nb,1):.4f}" if use_lcu else ""
            print(f"    ep{ep:4d}/{epochs}  R²={r2v:.4f}  val={vm:.5f}"
                  f"{lcu_str}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    return hist, lcu_loss_fn


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION + SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_predictions(name, model, data, device, outdir, is_qpinn=False):
    """Predict on test set and save to predictions_{name}.npy."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(data['X_te']), 512):
            xb = data['X_te'][i:i+512].to(device, non_blocking=True)
            if is_qpinn:
                p, _ = forward_with_raw(model, xb)
            else:
                p = model(xb)
            preds.append(p.cpu())
    yp = torch.cat(preds)
    r2_te = r2_score(yp, data['Y_te'])

    out = {
        'name': name,
        'r2_test': r2_te,
        'pred':    yp.numpy(),
        'Y_te':    data['Y_te'].numpy(),
        'X_te':    data['X_te'].numpy(),
        'Xm': data['Xm'].numpy(), 'Xs': data['Xs'].numpy(),
        'Ym': data['Ym'].numpy(), 'Ys': data['Ys'].numpy(),
        'feature_names': data['feature_names'],
        'output_names':  data['output_names'],
    }
    path = os.path.join(outdir, f'predictions_{name}.npy')
    np.save(path, out)
    return r2_te, path


def save_history(name, hist, cfg, epochs, outdir):
    """Save training history to history_{name}.npy."""
    h = dict(hist)
    h['config'] = {k: str(v) for k,v in cfg.items() if k != 'build'}
    h['epochs'] = epochs
    path = os.path.join(outdir, f'history_{name}.npy')
    np.save(path, h)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED PLOT
# ══════════════════════════════════════════════════════════════════════════════

def make_plot(results, outdir):
    """
    Single figure with:
      Left  — training curves (val loss per model)
      Middle — R² convergence curves per model
      Right  — final R² bar chart
    """
    n = len(results)
    if n == 0: return

    cmap = plt.cm.tab20
    colors = [cmap(i/max(n,1)) for i in range(n)]

    fig = plt.figure(figsize=(18, max(5, 2+n*0.4)))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])   # val loss curves
    ax2 = fig.add_subplot(gs[1])   # R² curves
    ax3 = fig.add_subplot(gs[2])   # final R² bar

    names   = [r['name']    for r in results]
    r2_vals = [r['r2_test'] for r in results]
    times   = [r['time']    for r in results]

    for i, r in enumerate(results):
        hist = r['hist']; c = colors[i]; label = r['name']
        if 'va' in hist and hist['va']:
            ax1.semilogy(range(1, len(hist['va'])+1), hist['va'],
                         color=c, linewidth=1.5, label=label)
        if 'r2v' in hist and hist['r2v']:
            ax2.plot(range(1, len(hist['r2v'])+1), hist['r2v'],
                     color=c, linewidth=1.5, label=label)

    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Val Loss (log scale)')
    ax1.set_title('Validation Loss Curves')
    ax1.legend(fontsize=6, loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch'); ax2.set_ylabel('R²')
    ax2.set_title('R² During Training')
    ax2.axhline(0, color='gray', lw=0.8, ls='--')
    ax2.legend(fontsize=6, loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Bar chart
    bars = ax3.barh(range(n), r2_vals, color=colors, edgecolor='white', linewidth=0.5)
    ax3.set_yticks(range(n)); ax3.set_yticklabels(names, fontsize=7)
    ax3.set_xlabel('R² (test set)')
    ax3.set_title('Final R² by Model')
    ax3.axvline(0, color='gray', lw=0.8)
    ax3.set_xlim(min(-0.05, min(r2_vals)-0.02), 1.02)
    ax3.grid(True, axis='x', alpha=0.3)
    for bar, val, t in zip(bars, r2_vals, times):
        ax3.text(max(val, 0)+0.01, bar.get_y()+bar.get_height()/2,
                 f'{val:.4f}  ({t:.0f}s)', va='center', fontsize=6)

    fig.suptitle('NS Shock-Tube PINN Comparison — 7-Feature Input',
                 fontsize=12, fontweight='bold')
    path = os.path.join(outdir, 'training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_sweep(model_names, data, epochs, device, outdir, print_every=5):
    """
    Run each model in sequence. Saves per-model files and a combined CSV.
    Returns list of result dicts.
    """
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, 'r2_summary.csv')
    all_results = []

    # Load previous results from CSV if it exists (resume mode)
    done_names = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done_names.add(row['name'])
        if done_names:
            print(f"[Resume] Already done: {sorted(done_names)}")

    csv_file = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=['name','r2_test','params','time_s','epochs','desc'])
    if not done_names:
        writer.writeheader()
    csv_file.flush()

    for mi, name in enumerate(model_names):
        if name in done_names:
            print(f"\n[{mi+1}/{len(model_names)}] {name} — already done, skipping")
            continue
        if name not in MODEL_CONFIGS:
            print(f"\n[{mi+1}/{len(model_names)}] {name} — UNKNOWN config, skipping")
            continue

        cfg = MODEL_CONFIGS[name]
        print(f"\n{'='*62}")
        print(f"[{mi+1}/{len(model_names)}] {name}")
        print(f"  {cfg['desc']}")
        print(f"{'='*62}")

        # ── Build model ──────────────────────────────────────────────────────
        model = cfg['build']()
        n_params = sum(p.numel() for p in model.parameters())
        is_qpinn = isinstance(model, ScalableQPINN)
        is_quantum = isinstance(model, (ScalableQPINN, ScalableQAPINN))

        if is_quantum:
            print(f"  nq={model.n_q}  dim=2^{model.n_q}={2**model.n_q}  params={n_params}")
            if is_qpinn and model.n_q < NI:
                print(f"  WARNING: nq={model.n_q} < NI=7, missing features {['rho_L','rho_R','p_R'][-(NI-model.n_q):]}")
        else:
            print(f"  params={n_params}")

        model = model_to_device(model, device)
        print(f"  device={device}  batch={auto_batch(model, cfg, device)}")

        # ── Train ────────────────────────────────────────────────────────────
        t_start = time.time()
        lcu_fn = None
        try:
            if cfg['type'] == 'classical':
                hist = train_classical(model, cfg, data, epochs, device, print_every)
            else:
                hist, lcu_fn = train_qpinn(model, cfg, data, epochs, device, print_every)
        except Exception as e:
            import traceback
            print(f"  ERROR during training: {e}")
            traceback.print_exc()
            continue

        elapsed = time.time() - t_start

        # ── Save predictions ─────────────────────────────────────────────────
        r2_te, pred_path = save_predictions(name, model, data, device, outdir, is_qpinn)
        hist_path = save_history(name, hist, cfg, epochs, outdir)

        print(f"\n  ✓  R²_test = {r2_te:.4f}   time = {elapsed:.0f}s")
        print(f"     Saved predictions → {pred_path}")
        print(f"     Saved history     → {hist_path}")

        # ── Append CSV ───────────────────────────────────────────────────────
        writer.writerow({'name':name,'r2_test':f"{r2_te:.6f}",
                         'params':n_params,'time_s':f"{elapsed:.0f}",
                         'epochs':epochs,'desc':cfg['desc']})
        csv_file.flush()

        all_results.append({'name':name,'r2_test':r2_te,
                             'hist':hist,'time':elapsed,'params':n_params})

    csv_file.close()
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='NS-PINN model sweep — trains models continuously and saves all outputs.')
    parser.add_argument('--models', nargs='+', default=None,
        help=f'Models to run (default: {DEFAULT_RUN}). Use --list to see all.')
    parser.add_argument('--data',   default='./data',
        help='Path to data directory (default: ./data)')
    parser.add_argument('--outdir', default='./results',
        help='Output directory (default: ./results)')
    parser.add_argument('--epochs', type=int, default=40,
        help='Training epochs per model (default: 40)')
    parser.add_argument('--scenarios', type=int, default=70,
        help='Number of scenarios to load (default: 70)')
    parser.add_argument('--t_stride', type=int, default=2,
        help='Time snapshot stride (default: 2 → 21 snaps/scenario)')
    parser.add_argument('--device', default='cuda',
        help='Device: cuda/cpu/mps (default: auto-detect)')
    parser.add_argument('--print_every', type=int, default=5,
        help='Print progress every N epochs (default: 5)')
    parser.add_argument('--list', action='store_true',
        help='List all available model configurations and exit')
    parser.add_argument('--list_classical', action='store_true',
        help='List classical UnifiedPINN presets and exit')
    args = parser.parse_args()

    if args.list:
        print("\nAvailable model configurations in MODEL_CONFIGS:\n")
        print(f"  {'Name':<25} {'Type':<12} {'LCU':>5}  Description")
        print("─"*75)
        for name, cfg in MODEL_CONFIGS.items():
            lcu = '✓' if cfg.get('lcu') else ''
            print(f"  {name:<25} {cfg['type']:<12} {lcu:>5}  {cfg['desc']}")
        print(f"\nDefault run: {DEFAULT_RUN}")
        return

    if args.list_classical:
        list_unified()
        return

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
        print(f"[Device] {device}")
    else:
        device = get_device()

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\n[Data] Loading from {args.data} ...")
    data = load_ns_data(args.data, n_scenarios=args.scenarios, t_stride=args.t_stride)
    print(f"  {data['n_samples']:,} samples  "
          f"train={len(data['X_tr']):,}  val={len(data['X_va']):,}  test={len(data['X_te']):,}")
    print(f"  features: {data['feature_names']}")
    print(f"  p_range={data['p_range']}  mu_range={data['mu_range']}")

    # Move normalisation stats to device
    for k in ('Xm','Xs','Ym','Ys'):
        data[k] = data[k].to(device)

    # ── Model list ────────────────────────────────────────────────────────────
    model_names = args.models or DEFAULT_RUN
    print(f"\n[Sweep] {len(model_names)} models × {args.epochs} epochs → {args.outdir}")
    print(f"  Models: {model_names}")

    # ── Run ───────────────────────────────────────────────────────────────────
    results = run_sweep(model_names, data, args.epochs, device,
                        args.outdir, args.print_every)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if results:
        plot_path = make_plot(results, args.outdir)
        print(f"\n[Plot] Saved → {plot_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        print(f"\n{'='*55}")
        print(f"{'Model':<25} {'R²_test':>8}  {'Time':>8}  {'Params':>8}")
        print(f"{'─'*55}")
        for r in sorted(results, key=lambda x: -x['r2_test']):
            print(f"  {r['name']:<23} {r['r2_test']:>8.4f}  {r['time']:>7.0f}s  {r['params']:>8,}")
        print(f"{'='*55}")
        print(f"\nAll outputs saved to: {args.outdir}/")
        print(f"  r2_summary.csv       — accuracy table")
        print(f"  predictions_*.npy    — test set predictions")
        print(f"  history_*.npy        — loss + R² training curves")
        print(f"  training_curves.png  — combined plot")


if __name__ == '__main__':
    main()
