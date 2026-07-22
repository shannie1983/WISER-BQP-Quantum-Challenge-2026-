"""
Step3_train_all_models.py
=========================
Train and compare four model architectures on 1D NS shock-tube data.

Models
------
  MLP        — classical MLP baseline (no physics loss)
               UnifiedPINN(model_type='mlp')
               Input:  [x, t, p_ratio, mu, rho_L, rho_R, p_R]  shape (7,)
               Output: [rho, u, p]  shape (3,)

  ClassPINN  — classical PINN with NS physics residuals via autograd
               UnifiedPINN(model_type='pinn', loss_mode='static')
               Same I/O as MLP, adds PDE loss terms

  QPINN      — pure quantum-inspired PINN (TorchPINN statevector sim)
               TorchPINN(n_qubits=7)
               Input:  [p_ratio, mu, rho_L, rho_R, p_R, t/t_end]  shape (6,)
               Output: field values  shape (n_qubits,)

  QAPINN     — hybrid Quantum-Augmented PINN
               classical encoder → quantum circuit → classical decoder
               Input:  [x, t, p_ratio, mu, rho_L, rho_R, p_R]  shape (7,)
               Output: [rho, u, p]  shape (3,)
               Quantum circuit acts as a nonlinear feature extractor
               between encoder and decoder MLP heads

Data
----
  Uses utilities_classical.load_ns_data() which expects the sweep output
  from Step1 (./data/ directory with scenario .npz files + index.json).

  Input features (NI=7): [x, t, p_ratio, mu, rho_L, rho_R, p_R]
  Output features (NO=3): [rho, u, p]

Structure
---------
  build_mlp()         — MLP model
  build_classpinn()   — Classical PINN
  build_qpinn()       — Quantum PINN (TorchPINN)
  build_qapinn()      — Hybrid QAPINN
  train_classical()   — training loop for MLP / ClassPINN
  train_qpinn()       — training loop for QPINN
  train_qapinn()      — training loop for QAPINN
  evaluate()          — R², MSE on test set
  save_ckpt()         — save checkpoint
  load_ckpt()         — load checkpoint
  save_results()      — save metrics JSON + predictions npz
  main()              — config + run all models

Dependencies
------------
  utilities_classical.py  — UnifiedPINN, unified_loss, load_ns_data
  utilities_quantum_torch.py — TorchPINN, torch_encode, torch_ansatz,
                               torch_measure, post_decode, fidelity_loss,
                               ansatz_weight_shape
"""

import sys
sys.path.insert(0, '/mnt/d/QCFD/WISER/BQC/github/')

import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utilities_classical import (
    UnifiedPINN, build_unified, unified_loss,
    load_ns_data, NI, NO, PRESETS
)
from utilities_quantum_torch import (
    TorchQPINN, TorchQaPINN, QuantumLayer, torch_encode, torch_ansatz, torch_measure,
    post_decode, fidelity_loss, pde_loss, ansatz_weight_shape,
    _DEFAULT_DEVICE,
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_mlp(device: torch.device) -> UnifiedPINN:
    """MLP baseline — data loss only, no physics."""
    return build_unified('mlp').to(device)


def build_classpinn(preset: str = 'pinn',
                    device: torch.device = None) -> UnifiedPINN:
    """
    Classical PINN from preset.
    Available presets: pinn, fourier_pinn, hardbc_pinn, softadapt_pinn,
                       dynratio_pinn, rar_pinn, best_classical, ...
    """
    return build_unified(preset).to(device)


def build_qpinn(n_qubits:    int   = 7,
                n_layers:    int   = 4,
                encoding:    str   = "angle_full",
                ansatz:      str   = "u_ring",
                device:      torch.device = None) -> TorchQPINN:
    """
    Pure quantum PINN with LCU loss (fidelity + NS Pauli residuals).

    lambda tuning:
        fidelity starts near 1.0 (psi and phi barely overlap initially)
        PDE Pauli terms start near 0 (small expval contributions)
        -> weight PDE terms much higher so they provide training signal
    """
    if device is None:
        device = _DEFAULT_DEVICE
    return TorchQPINN(
        n_qubits        = n_qubits,
        n_layers        = n_layers,
        encoding        = encoding,
        ansatz          = ansatz,
        pde             = "navier_stokes",
        nu              = 0.005,
        lambda_fidelity = 1.0,
        lambda_pde      = 50.0,  # boost PDE branches — they start tiny
        lambda_data     = 10.0,  # direct MSE supervision — main convergence signal
        n_out           = 3,     # NS outputs: [rho, u, p]
        device          = device,
    )


def build_qapinn(
    hidden:      int   = 128,
    depth:       int   = 4,
    q_layer_idx: int   = None,     # None → depth // 2  (middle layer)
    n_qubits:    int   = 6,
    n_layers:    int   = 2,
    encoding:    str   = "angle_full",
    ansatz:      str   = "u_ring",
    activation:  str   = "tanh",
    lambda_q:    float = 0.1,
    use_physics: bool  = False,
    loss_mode:   str   = "static",
    device:      torch.device = None,
) -> TorchQaPINN:
    """
    Hybrid Quantum-Augmented PINN.

    Standard MLP of `depth` hidden layers where layer `q_layer_idx`
    is replaced by a QVC (QuantumLayer).  All other layers are classical
    Linear + activation.  Compatible with UnifiedPINN interface.

    Args:
        hidden:      MLP hidden dimension (same across all layers)
        depth:       number of hidden layers (2–6)
        q_layer_idx: which hidden layer (0-indexed) to replace with QVC
                     default = depth // 2  (middle layer)
        n_qubits:    number of qubits in the QVC
        n_layers:    QVC ansatz depth
        encoding:    quantum encoding  (angle_full, fft, iqp, ...)
        ansatz:      quantum ansatz    (u_ring, strongly, ...)
        activation:  classical activation for non-QVC layers
        lambda_q:    weight for quantum fidelity loss
        use_physics: add classical NS physics residuals via unified_loss
        loss_mode:   adaptive weighting scheme  (static, softadapt, ...)
        device:      torch.device
    """
    if device is None:
        device = _DEFAULT_DEVICE
    return TorchQaPINN(
        hidden      = hidden,
        depth       = depth,
        q_layer_idx = q_layer_idx,
        n_qubits    = n_qubits,
        n_layers    = n_layers,
        encoding    = encoding,
        ansatz      = ansatz,
        activation  = activation,
        lambda_q    = lambda_q,
        use_physics = use_physics,
        loss_mode   = loss_mode,
        device      = device,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_ckpt(path, epoch, model, optimizer,
              loss_history, metric_history, config):
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss_history":    loss_history,
        "metric_history":  metric_history,
        "config":          config,
    }, path)


def load_ckpt(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  Loaded checkpoint: {path}  (epoch {ckpt['epoch']})")
    return (model, optimizer,
            ckpt["epoch"],
            ckpt.get("loss_history", []),
            ckpt.get("metric_history", []),
            ckpt.get("config", {}))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRAINING LOOPS
# ══════════════════════════════════════════════════════════════════════════════

def train_classical(
    model, optimizer, X_tr, Y_tr,
    n_epochs, batch_size,
    Xm, Xs, p_range, mu_range,
    checkpoint_dir, checkpoint_every, config,
    start_epoch=0, loss_history=None, device=None,
):
    """
    Training loop for MLP and ClassPINN (UnifiedPINN).

    Uses unified_loss() which adds NS physics terms for model_type='pinn'.
    model.update(epoch, log) is called each epoch for adaptive weighting.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if loss_history is None:
        loss_history = []
    loader = DataLoader(TensorDataset(X_tr, Y_tr),
                        batch_size=batch_size, shuffle=True)
    t0 = time.time()

    for epoch in range(start_epoch, n_epochs):
        model.train()
        ep_loss = 0.0; ep_log = {}; n = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss, log = unified_loss(
                model, xb, yb, epoch,
                Xm.to(device), Xs.to(device),
                p_range, mu_range,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(xb)
            for k, v in log.items():
                ep_log[k] = ep_log.get(k, 0.) + v * len(xb)
            n += len(xb)

        avg_loss = ep_loss / max(n, 1)
        avg_log  = {k: v / max(n, 1) for k, v in ep_log.items()}
        loss_history.append(avg_loss)
        model.update(epoch, avg_log)

        elapsed = time.time() - t0
        phys    = avg_log.get("total_physics", 0.)
        print(f"  Epoch {epoch+1:4d}/{n_epochs}  "
              f"loss={avg_loss:.6f}  data={avg_log.get('data',0.):.6f}  "
              f"phys={phys:.6f}  t={elapsed:.1f}s")

        if (epoch + 1) % checkpoint_every == 0:
            p = os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch+1:04d}.pt")
            save_ckpt(p, epoch+1, model, optimizer,
                      loss_history, [], config)
            print(f"  ✓ checkpoint → {p}")

    return loss_history


def train_qpinn(
    model, optimizer, X_tr, Y_tr,
    n_epochs, batch_size,
    checkpoint_dir, checkpoint_every, config,
    start_epoch=0, loss_history=None, mse_history=None, device=None,
):
    """
    Training loop for TorchPINN (QPINN).

    TorchPINN.forward(x, y) returns (pred, loss) where:
        x: [batch, n_qubits]  — input features (encode → ansatz → pred)
        y: [batch, n_qubits]  — target features (encode only, for fidelity)

    Note: TorchPINN output dim = n_qubits.
          For NS data (output = [rho,u,p], dim=3), we use only the first
          NO=3 output features for MSE tracking. The quantum fidelity loss
          operates in the full n_qubits statevector space.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if loss_history is None: loss_history = []
    if mse_history  is None: mse_history  = []

    # x: cycle NI=7 features to n_qubits for x-register encoding
    # y: pass raw [batch, n_out=3] — model uses n_y_qubits=n_out for phi
    n_q = model.n_qubits
    def _prep_x(X):
        n_feat = X.shape[-1]
        if n_feat < n_q:
            return X[:, torch.arange(n_q) % n_feat]
        return X[:, :n_q]

    Xq_tr = _prep_x(X_tr)
    loader = DataLoader(TensorDataset(Xq_tr, Y_tr),
                        batch_size=batch_size, shuffle=True)
    t0 = time.time()

    for epoch in range(start_epoch, n_epochs):
        model.train()
        ep_loss = 0.; ep_mse = 0.; n = 0

        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, loss = model(xb, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                # pred is [batch, n_out=3] from out_head, yb cycled to n_qubits
                # compare pred vs first NO cols of yb (which was cycled from Y_tr)
                n_out_pred = pred.shape[-1]
                ep_mse += ((pred - yb[:, :n_out_pred])**2).mean().item() * len(xb)
            ep_loss += loss.item() * len(xb)
            n += len(xb)
            print(epoch, i, ep_loss/n)
        avg_loss = ep_loss / max(n, 1)
        avg_mse  = ep_mse  / max(n, 1)
        grad_n   = (model.weights.grad.norm().item()
                    if model.weights.grad is not None else 0.)
        loss_history.append(avg_loss)
        mse_history.append(avg_mse)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:4d}/{n_epochs}  "
              f"loss={avg_loss:.6f}  pred_mse={avg_mse:.6f}  "
              f"grad={grad_n:.4f}  t={elapsed:.1f}s")

        if (epoch + 1) % checkpoint_every == 0:
            p = os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch+1:04d}.pt")
            save_ckpt(p, epoch+1, model, optimizer,
                      loss_history, mse_history, config)
            print(f"  ✓ checkpoint → {p}")

    return loss_history, mse_history


def train_qapinn(
    model, optimizer, X_tr, Y_tr,
    n_epochs, batch_size,
    checkpoint_dir, checkpoint_every, config,
    start_epoch=0, loss_history=None, mse_history=None, device=None,
):
    """
    Training loop for QAPINN (hybrid).

    Uses model.forward_with_loss(x, y) which returns:
        (pred, log_dict, loss_tensor)
    model.forward(x) is compatible with UnifiedPINN for evaluation.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if loss_history is None: loss_history = []
    if mse_history  is None: mse_history  = []

    loader = DataLoader(TensorDataset(X_tr, Y_tr),
                        batch_size=batch_size, shuffle=True)
    t0 = time.time()

    for epoch in range(start_epoch, n_epochs):
        model.train()
        ep_loss = 0.; ep_mse = 0.; ep_q = 0.; n = 0

        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, log, loss = model.forward_with_loss(xb, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                ep_mse += ((pred.detach() - yb)**2).mean().item() * len(xb)
            ep_loss += loss.item()  * len(xb)
            ep_q    += log["quantum"] * len(xb)
            n       += len(xb)
            
            #print(epoch, i/len(loader), ep_loss/n)

        avg_loss = ep_loss / max(n, 1)
        avg_mse  = ep_mse  / max(n, 1)
        avg_q    = ep_q    / max(n, 1)
        loss_history.append(avg_loss)
        mse_history.append(avg_mse)
        
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:4d}/{n_epochs}  "
              f"loss={avg_loss:.6f}  mse={avg_mse:.6f}  "
              f"q_fid={avg_q:.6f}  t={elapsed:.1f}s")

        if (epoch + 1) % checkpoint_every == 0:
            p = os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch+1:04d}.pt")
            save_ckpt(p, epoch+1, model, optimizer,
                      loss_history, mse_history, config)
            print(f"  ✓ checkpoint → {p}")

    return loss_history, mse_history


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_te, Y_te, device, model_type="classical"):
    """
    Evaluate model on test set.

    model_type:
        'classical'  — model(x) returns pred directly (MLP, ClassPINN, QAPINN)
        'qpinn'      — model(x, y) returns (pred, loss); pred has n_qubits cols

    Returns dict: mse, rmse, r2, pred (tensor)
    """
    model.eval()
    with torch.no_grad():
        X = X_te.to(device)
        Y = Y_te.to(device)

        if model_type == "qpinn":
            # x: cycle to n_qubits for x-register; y: raw [batch, n_out]
            n_q    = model.n_qubits
            n_feat = X.shape[-1]
            Xq     = X[:, torch.arange(n_q) % n_feat] if n_feat < n_q else X[:, :n_q]
            pred, _ = model(Xq, Y)   # y passed raw, model uses n_y_qubits internally
        else:
            pred = model(X)

        mse  = ((pred - Y)**2).mean().item()
        rmse = mse ** 0.5
        ss_res = ((pred - Y)**2).sum().item()
        ss_tot = ((Y - Y.mean(0, keepdim=True))**2).sum().item()
        r2   = 1 - ss_res / (ss_tot + 1e-12)

    return {"mse": mse, "rmse": rmse, "r2": r2, "pred": pred}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def save_results(out_dir, model_name, model, optimizer,
                 n_epochs, loss_history, mse_history,
                 eval_metrics, X_te, Y_te, config):
    """Save checkpoint_final.pt + metrics.json + predictions.npz."""
    os.makedirs(out_dir, exist_ok=True)

    # checkpoint
    ckpt_path = os.path.join(out_dir, f"{model_name}_final.pt")
    save_ckpt(ckpt_path, n_epochs, model, optimizer,
              loss_history, mse_history, config)
    print(f"  ✓ checkpoint  → {ckpt_path}")

    # metrics JSON
    metrics = {
        "model": model_name,
        "config": config,
        "training": {
            "loss_history": [round(l, 6) for l in loss_history],
            "mse_history":  [round(m, 6) for m in mse_history],
            "start_loss":   round(loss_history[0],  6) if loss_history else None,
            "end_loss":     round(loss_history[-1], 6) if loss_history else None,
        },
        "evaluation": {
            "mse":  round(eval_metrics["mse"],  6),
            "rmse": round(eval_metrics["rmse"], 6),
            "r2":   round(eval_metrics["r2"],   6),
        },
    }
    metrics_path = os.path.join(out_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ metrics     → {metrics_path}")

    # predictions npz
    pred = eval_metrics["pred"]
    npz_path = os.path.join(out_dir, f"{model_name}_predictions.npz")
    np.savez(
        npz_path,
        x_norm   = X_te.cpu().numpy(),
        y_norm   = Y_te.cpu().numpy(),
        pred_norm= pred.cpu().numpy(),
        residual = (pred - Y_te.to(pred.device)).cpu().numpy(),
    )
    print(f"  ✓ predictions → {npz_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def get_free_gpu() -> int:
    """Return the index of the GPU with the most free memory."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=True,
        )
        free = [int(x) for x in result.stdout.strip().split("\n")]
        best = free.index(max(free))
        print(f"  GPU free memory : {[f'{x} MiB' for x in free]}")
        print(f"  → selected GPU {best}  ({max(free)} MiB free)")
        return best
    except Exception as e:
        print(f"  nvidia-smi failed ({e}), defaulting to cuda:0")
        return 0


def main():
    # ── Configuration ──────────────────────────────────────────────────────────
    DATA_DIR         = "/mnt/d/QCFD/WISER/BQC/github/data"    # Step1 sweep output directory
    OUT_DIR          = "/mnt/d/QCFD/WISER/BQC/github/results" # output directory for all models
    N_SCENARIOS      = 100                    # number of scenarios to load (100 requested, 95 completed)
    T_STRIDE         = 4                      # time snapshot stride, change to 6 if you want it go faster
    N_EPOCHS         = 200                     # training epochs per model
    BATCH_SIZE       = 128
    LR               = 1e-3
    CHECKPOINT_EVERY = 10

    # ── Auto-select the GPU with most free memory ──────────────────────────────
    if torch.cuda.is_available():
        _gpu_id = get_free_gpu()
        DEVICE  = torch.device(f"cuda:{_gpu_id}")
        torch.cuda.set_device(_gpu_id)
    else:
        DEVICE  = torch.device("cpu")

    # which models to train (set False to skip)
    RUN_MLP        = False
    RUN_CLASSPINN  = False
    RUN_QPINN      = True
    RUN_QAPINN     = True

    # ClassPINN preset — any key from PRESETS or 'best_classical'
    CLASSPINN_PRESET = "pinn"

    # QPINN config
    QPINN_N_QUBITS  = 7           # matches NI=7 input features; 2^14 OOM on GPU
    QPINN_N_LAYERS  = 4           # more layers = more expressivity
    QPINN_ENCODING  = "angle_full"
    QPINN_ANSATZ    = "u_ring"

    # QAPINN config
    QAPINN_HIDDEN      = 128      # MLP hidden dim (same as MLP/ClassPINN)
    QAPINN_DEPTH       = 4        # total hidden layers
    QAPINN_Q_LAYER_IDX = 2     # None = middle layer (depth//2 = layer 2)
    QAPINN_N_QUBITS    = 14        # qubits in QVC
    QAPINN_N_LAYERS    = 2        # QVC ansatz depth
    QAPINN_ENCODING    = "angle_full"
    QAPINN_ANSATZ      = "u_ring"
    QAPINN_ACTIVATION  = "tanh"   # classical layer activation
    QAPINN_LAMBDA_Q    = 0.1      # quantum fidelity loss weight
    QAPINN_PHYSICS     = False    # add NS physics residuals

    print("=" * 65)
    print("  NS Shock Tube — Model Comparison")
    print("=" * 65)
    print(f"  device    : {DEVICE}")
    print(f"  data_dir  : {DATA_DIR}")
    print(f"  models    : MLP={RUN_MLP}  ClassPINN={RUN_CLASSPINN}"
          f"  QPINN={RUN_QPINN}  QAPINN={RUN_QAPINN}")
    print(f"  epochs    : {N_EPOCHS}  batch={BATCH_SIZE}  lr={LR}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"Loading NS data from {DATA_DIR} ...")
    data = load_ns_data(DATA_DIR, n_scenarios=N_SCENARIOS, t_stride=T_STRIDE)
    X_tr, Y_tr = data["X_tr"].to(DEVICE), data["Y_tr"].to(DEVICE)
    X_va, Y_va = data["X_va"].to(DEVICE), data["Y_va"].to(DEVICE)
    X_te, Y_te = data["X_te"].to(DEVICE), data["Y_te"].to(DEVICE)
    Xm, Xs     = data["Xm"],              data["Xs"]
    p_range    = data["p_range"]
    mu_range   = data["mu_range"]

    print(f"  train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")
    print(f"  input features  : {data['feature_names']}")
    print(f"  output features : {data['output_names']}")

    # attach normalisation stats to models (needed by physics loss)
    def _attach_stats(model):
        model._Xm = Xm.to(DEVICE); model._Xs = Xs.to(DEVICE)
        model._Ym = data["Ym"].to(DEVICE); model._Ys = data["Ys"].to(DEVICE)

    results = {}

    # ══════════════════════════════════════════════════════════════════════════
    # MLP
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_MLP:
        print(f"\n{'─'*65}")
        print("[ MLP ]  Classical MLP baseline — data loss only")
        cfg = {"model": "mlp", "epochs": N_EPOCHS, "batch": BATCH_SIZE, "lr": LR}
        ckpt_dir = os.path.join(OUT_DIR, "mlp")

        model_mlp = build_mlp(DEVICE)
        _attach_stats(model_mlp)
        model_mlp.describe()
        opt_mlp = optim.Adam(model_mlp.parameters(), lr=LR)

        loss_h = train_classical(
            model_mlp, opt_mlp, X_tr, Y_tr,
            N_EPOCHS, BATCH_SIZE, Xm, Xs, p_range, mu_range,
            ckpt_dir, CHECKPOINT_EVERY, cfg, device=DEVICE,
        )
        metrics = evaluate(model_mlp, X_te, Y_te, DEVICE, "classical")
        print(f"\n  MLP test  MSE={metrics['mse']:.6f}  "
              f"RMSE={metrics['rmse']:.6f}  R²={metrics['r2']:.4f}")
        save_results(ckpt_dir, "mlp", model_mlp, opt_mlp,
                     N_EPOCHS, loss_h, [], metrics, X_te, Y_te, cfg)
        results["mlp"] = metrics

    # ══════════════════════════════════════════════════════════════════════════
    # ClassPINN
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_CLASSPINN:
        print(f"\n{'─'*65}")
        print(f"[ ClassPINN ]  Classical PINN — preset='{CLASSPINN_PRESET}'")
        cfg = {"model": f"classpinn_{CLASSPINN_PRESET}",
               "preset": CLASSPINN_PRESET,
               "epochs": N_EPOCHS, "batch": BATCH_SIZE, "lr": LR}
        ckpt_dir = os.path.join(OUT_DIR, f"classpinn_{CLASSPINN_PRESET}")

        model_cp = build_classpinn(CLASSPINN_PRESET, DEVICE)
        _attach_stats(model_cp)
        model_cp.describe()
        opt_cp = optim.Adam(model_cp.parameters(), lr=LR)

        loss_h = train_classical(
            model_cp, opt_cp, X_tr, Y_tr,
            N_EPOCHS, BATCH_SIZE, Xm, Xs, p_range, mu_range,
            ckpt_dir, CHECKPOINT_EVERY, cfg, device=DEVICE,
        )
        metrics = evaluate(model_cp, X_te, Y_te, DEVICE, "classical")
        print(f"\n  ClassPINN test  MSE={metrics['mse']:.6f}  "
              f"RMSE={metrics['rmse']:.6f}  R²={metrics['r2']:.4f}")
        save_results(ckpt_dir, f"classpinn_{CLASSPINN_PRESET}",
                     model_cp, opt_cp,
                     N_EPOCHS, loss_h, [], metrics, X_te, Y_te, cfg)
        results["classpinn"] = metrics

    # ══════════════════════════════════════════════════════════════════════════
    # QPINN
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_QPINN:
        print(f"\n{'─'*65}")
        print(f"[ QPINN ]  Pure quantum-inspired PINN  "
              f"n_qubits={QPINN_N_QUBITS}  ansatz={QPINN_ANSATZ}")
        cfg = {"model": "qpinn", "n_qubits": QPINN_N_QUBITS,
               "n_layers": QPINN_N_LAYERS, "encoding": QPINN_ENCODING,
               "ansatz": QPINN_ANSATZ, "epochs": N_EPOCHS,
               "batch": BATCH_SIZE, "lr": LR}
        #ckpt_dir = os.path.join(OUT_DIR, "qpinn")
        ckpt_dir = os.path.join(OUT_DIR, f"qpinn_q{QPINN_N_QUBITS}_l{QPINN_N_LAYERS}_{QPINN_ANSATZ}_{QPINN_ENCODING}")
        n_params = sum(p.numel() for p in
                       [nn.Parameter(torch.zeros(
                        ansatz_weight_shape(QPINN_ANSATZ, QPINN_N_LAYERS,
                                            QPINN_N_QUBITS)))])
        print(f"  Trainable weights : {n_params}")
        print(f"  Statevector dim   : 2^{QPINN_N_QUBITS} = {2**QPINN_N_QUBITS}")

        model_qp = build_qpinn(QPINN_N_QUBITS, QPINN_N_LAYERS,
                               QPINN_ENCODING, QPINN_ANSATZ, DEVICE)
        opt_qp = optim.Adam(model_qp.parameters(), lr=LR)

        loss_h, mse_h = train_qpinn(
            model_qp, opt_qp, X_tr, Y_tr,
            N_EPOCHS, BATCH_SIZE,
            ckpt_dir, CHECKPOINT_EVERY, cfg, device=DEVICE,
        )
        metrics = evaluate(model_qp, X_te, Y_te, DEVICE, "qpinn")
        print(f"\n  QPINN test  MSE={metrics['mse']:.6f}  "
              f"RMSE={metrics['rmse']:.6f}  R²={metrics['r2']:.4f}")
        save_results(ckpt_dir, "qpinn", model_qp, opt_qp,
                     N_EPOCHS, loss_h, mse_h, metrics, X_te, Y_te, cfg)
        results["qpinn"] = metrics

    # ══════════════════════════════════════════════════════════════════════════
    # QAPINN
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_QAPINN:
        cfg = {"model": "qapinn",
               "hidden": QAPINN_HIDDEN, "depth": QAPINN_DEPTH,
               "q_layer_idx": QAPINN_Q_LAYER_IDX,
               "n_qubits": QAPINN_N_QUBITS, "n_layers": QAPINN_N_LAYERS,
               "encoding": QAPINN_ENCODING, "ansatz": QAPINN_ANSATZ,
               "lambda_q": QAPINN_LAMBDA_Q, "use_physics": QAPINN_PHYSICS,
               "epochs": N_EPOCHS, "batch": BATCH_SIZE, "lr": LR}
        ckpt_dir = os.path.join(OUT_DIR, "qapinn")
        ckpt_dir = os.path.join(OUT_DIR,f"qpinn_q{QAPINN_N_QUBITS}_l{QAPINN_N_LAYERS}_qli{QAPINN_Q_LAYER_IDX}_{QAPINN_ANSATZ}_{QAPINN_ENCODING}")
        print(f"\n{'─'*65}")
        print(f"[ QAPINN ]  Hybrid — MLP depth={QAPINN_DEPTH}"
              f" with QVC at layer {QAPINN_Q_LAYER_IDX or QAPINN_DEPTH//2}"
              f"  n_qubits={QAPINN_N_QUBITS}  lambda_q={QAPINN_LAMBDA_Q}")

        model_qa = build_qapinn(
            hidden      = QAPINN_HIDDEN,
            depth       = QAPINN_DEPTH,
            q_layer_idx = QAPINN_Q_LAYER_IDX,
            n_qubits    = QAPINN_N_QUBITS,
            n_layers    = QAPINN_N_LAYERS,
            encoding    = QAPINN_ENCODING,
            ansatz      = QAPINN_ANSATZ,
            activation  = QAPINN_ACTIVATION,
            lambda_q    = QAPINN_LAMBDA_Q,
            use_physics = QAPINN_PHYSICS,
            device      = DEVICE,
        )
        _attach_stats(model_qa)
        model_qa.describe()
        n_params = sum(p.numel() for p in model_qa.parameters())
        print(f"  Trainable params  : {n_params:,}")
        q_idx = model_qa.q_layer_idx
        print(f"  QVC at layer      : {q_idx} of {QAPINN_DEPTH}"
              f"  (0-indexed hidden layer)")
        opt_qa = optim.Adam(model_qa.parameters(), lr=LR)

        loss_h, mse_h = train_qapinn(
            model_qa, opt_qa, X_tr, Y_tr,
            N_EPOCHS, BATCH_SIZE,
            ckpt_dir, CHECKPOINT_EVERY, cfg, device=DEVICE,
        )
        metrics = evaluate(model_qa, X_te, Y_te, DEVICE, "classical")
        print(f"\n  QAPINN test  MSE={metrics['mse']:.6f}  "
              f"RMSE={metrics['rmse']:.6f}  R²={metrics['r2']:.4f}")
        save_results(ckpt_dir, "qapinn", model_qa, opt_qa,
                     N_EPOCHS, loss_h, mse_h, metrics, X_te, Y_te, cfg)
        results["qapinn"] = metrics

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Final Comparison")
    print(f"  {'Model':<14} {'MSE':>10}  {'RMSE':>10}  {'R²':>8}")
    print(f"  {'─'*14} {'─'*10}  {'─'*10}  {'─'*8}")
    for name, m in results.items():
        print(f"  {name:<14} {m['mse']:>10.6f}  {m['rmse']:>10.6f}  "
              f"{m['r2']:>8.4f}")
    print(f"{'='*65}")

    # save comparison JSON
    comp_path = os.path.join(OUT_DIR, "comparison.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(comp_path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "pred"}
                   for k, v in results.items()}, f, indent=2)
    print(f"\n  ✓ comparison → {comp_path}")


if __name__ == "__main__":
    main()
