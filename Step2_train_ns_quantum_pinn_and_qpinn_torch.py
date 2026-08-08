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

  QPINN      — pure quantum-inspired PINN (TorchQPINN statevector sim)
               TorchQPINN(n_qubits=7)
               Input:  [p_ratio, mu, rho_L, rho_R, p_R, t/t_end]  shape (6,)
               Output: field values  shape (n_qubits,)

  TorchQAPINN     — hybrid Quantum-Augmented PINN
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
  build_qpinn()       — Quantum PINN (TorchQPINN)
  build_qapinn()      — Hybrid TorchQAPINN
  train_classical()   — training loop for MLP / ClassPINN
  train_qpinn()       — training loop for QPINN
  train_qapinn()      — training loop for TorchQAPINN
  evaluate()          — R², MSE on test set
  save_ckpt()         — save checkpoint
  load_ckpt()         — load checkpoint
  save_results()      — save metrics JSON + predictions npz
  main()              — config + run all models

Dependencies
------------
  utilities_classical.py  — UnifiedPINN, unified_loss, load_ns_data
  utilities_quantum_torch.py — TorchQPINN, torch_encode, torch_ansatz,
                               torch_measure, post_decode, fidelity_loss,
                               ansatz_weight_shape
"""

import sys
sys.path.insert(0, '/mnt/d/QCFD/WISER/BQC/github/')

import os, json, time, re
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
    TorchQPINN, TorchQAPINN, QuantumLayer, torch_encode, torch_ansatz, torch_measure,
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


def build_qpinn(n_qubits:      int   = 7,
                n_layers:      int   = 4,
                n_enc_layers:  int   = 1,       # x re-uploading blocks
                encoding:      str   = "arctan",
                ansatz:        str   = "u_ring",
                y_n_enc_layers: int  = 1,       # y re-uploading (no weights)
                y_encoding:    str   = None,    # None = same as x encoding
                y_n_qubits:    int   = None,    # None = n_out=3
                use_decoder:   bool  = True,    # True=2-layer MLP after measure
                nu:            float = 0.005,
                lambda_fidelity: float = 1.0,
                lambda_pde:    float = 10.0,
                lambda_data:   float = 50.0,
                n_out:         int   = 3,
                device:        torch.device = None) -> TorchQPINN:
    """
    Pure quantum PINN with:
      - x: [encode(x) -> ansatz] x n_enc_layers       (learnable)
      - y: [encode(y)] x y_n_enc_layers                (fixed GT, no weights)
      - decoder: 2-layer MLP after measuring psi

    y_n_enc_layers=1: encode(y)              -> phi  (linear in y)
    y_n_enc_layers=2: encode(y)->encode(y)   -> phi  (richer representation)
    y is never trained — it is fixed ground truth.

    Decoder modes:
        use_decoder=True  : MLP decoder after measurement (richer readout)
            psi -> measure -> MLP(expvals) -> pred
            phi -> measure -> MLP(expvals) -> decoded_y  (symmetric!)
        use_decoder=False : quantum linear readout (obs_weights)
            psi -> measure -> post_decode -> obs_weights @ decoded -> pred

    NOTE: `device` is the LAST argument. Always call this with keyword
    arguments — passing it positionally lands it in `y_n_enc_layers`.
    """
    if device is None:
        device = _DEFAULT_DEVICE
    return TorchQPINN(
        n_qubits        = n_qubits,
        n_layers        = n_layers,
        n_enc_layers    = n_enc_layers,
        encoding        = encoding,
        ansatz          = ansatz,
        y_encoding      = y_encoding,
        y_n_qubits      = y_n_qubits,
        y_n_enc_layers  = y_n_enc_layers,
        use_decoder     = use_decoder,
        pde             = "navier_stokes",
        nu              = nu,
        lambda_fidelity = lambda_fidelity,
        lambda_pde      = lambda_pde,
        lambda_data     = lambda_data,
        n_out           = n_out,
        device          = device,
    )


def build_qapinn(
    hidden:      int   = 128,
    depth:       int   = 4,
    q_layer_idx: int   = None,     # None → depth // 2  (middle layer)
    n_qubits:    int   = 6,
    n_layers:    int   = 2,
    encoding:    str   = "angle_full",
    ansatz:      str   = "hardware_efficient",
    activation:  str   = "tanh",
    lambda_q:    float = 0.1,
    use_physics: bool  = False,
    loss_mode:   str   = "static",
    device:      torch.device = None,
) -> TorchQAPINN:
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
    return TorchQAPINN(
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


def latest_checkpoint(run_dir, model_name):
    """Return (path, epoch) for the newest final/intermediate checkpoint."""
    candidates = []
    final_path = os.path.join(run_dir, f"{model_name}_final.pt")
    if os.path.exists(final_path):
        candidates.append(final_path)
    if os.path.isdir(run_dir):
        candidates.extend(
            os.path.join(run_dir, fn) for fn in os.listdir(run_dir)
            if fn.startswith("ckpt_epoch_") and fn.endswith(".pt")
        )

    best_path, best_epoch = None, 0
    for path in candidates:
        try:
            checkpoint = torch.load(
                path, map_location="cpu", weights_only=False)
            epoch = int(checkpoint.get("epoch", 0))
        except Exception:
            match = re.search(r"ckpt_epoch_(\d+)\.pt$", path)
            epoch = int(match.group(1)) if match else 0
        if epoch > best_epoch:
            best_path, best_epoch = path, epoch
    return best_path, best_epoch


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRAINING LOOPS
# ══════════════════════════════════════════════════════════════════════════════

def train_classical(
    model, optimizer, X_tr, Y_tr,
    n_epochs, batch_size,
    Xm, Xs, p_range, mu_range,
    checkpoint_dir, checkpoint_every, config,
    start_epoch=0, loss_history=None, mse_history=None, device=None,
):
    """
    Training loop for MLP and ClassPINN (UnifiedPINN).

    Uses unified_loss() which adds NS physics terms for model_type='pinn'.
    model.update(epoch, log) is called each epoch for adaptive weighting.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if loss_history is None:
        loss_history = []
    if mse_history is None:
        mse_history = []
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
        mse_history.append(avg_log.get("data", float("nan")))
        model.update(epoch, avg_log)

        elapsed = time.time() - t0
        phys    = avg_log.get("total_physics", 0.)
        print(f"  Epoch {epoch+1:4d}/{n_epochs}  "
              f"loss={avg_loss:.6f}  data={avg_log.get('data',0.):.6f}  "
              f"phys={phys:.6f}  t={elapsed:.1f}s")

        if (epoch + 1) % checkpoint_every == 0:
            p = os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch+1:04d}.pt")
            save_ckpt(p, epoch+1, model, optimizer,
                      loss_history, mse_history, config)
            print(f"  ✓ checkpoint → {p}")

    return loss_history, mse_history


def train_qpinn(
    model, optimizer, X_tr, Y_tr,
    n_epochs, batch_size,
    checkpoint_dir, checkpoint_every, config,
    start_epoch=0, loss_history=None, mse_history=None, device=None,
    log_every=1, save_ckpts=True,
):
    """
    Training loop for TorchQPINN (QPINN).

    TorchQPINN.forward(x, y) returns (pred, loss) where:
        x: [batch, n_qubits]  — input features (encode → ansatz → pred)
        y: [batch, n_qubits]  — target features (encode only, for fidelity)

    Note: TorchQPINN output dim = n_qubits.
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

    # Calibrate pred_scale/shift from training data (Improvement D)
    if hasattr(model, 'init_from_data') and start_epoch == 0:
        model.init_from_data(Y_tr)

    # Warmup + Cosine annealing:
    # Quantum circuits often get stuck in flat gradient regions early.
    # Warmup (5% of epochs) ramps LR from 0 → lr_max to escape flat regions.
    # Then cosine decay to 1e-5 for fine-tuning.
    n_warmup = max(1, n_epochs // 20)   # 5% warmup
    def lr_lambda(epoch):
        if epoch < n_warmup:
            return float(epoch + 1) / n_warmup   # linear ramp
        progress = (epoch - n_warmup) / max(1, n_epochs - n_warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
                n_out_pred = pred.shape[-1]
                ep_mse += ((pred - yb[:, :n_out_pred])**2).mean().item() * len(xb)
            ep_loss += loss.item() * len(xb)
            n += len(xb)

        scheduler.step()
        avg_loss = ep_loss / max(n, 1)
        avg_mse  = ep_mse  / max(n, 1)
        # robust grad norm: not every ansatz stores params in `model.weights`
        _w = getattr(model, "weights", None)
        if _w is not None and getattr(_w, "grad", None) is not None:
            grad_n = _w.grad.norm().item()
        else:
            grad_n = sum(p.grad.norm().item() ** 2
                         for p in model.parameters()
                         if p.grad is not None) ** 0.5
        loss_history.append(avg_loss)
        mse_history.append(avg_mse)

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        if log_every and ((epoch + 1) % log_every == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch+1:4d}/{n_epochs}  "
                  f"loss={avg_loss:.6f}  pred_mse={avg_mse:.6f}  "
                  f"grad={grad_n:.4f}  lr={lr_now:.2e}  t={elapsed:.1f}s")

        if save_ckpts and (epoch + 1) % checkpoint_every == 0:
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
    Training loop for TorchQAPINN (hybrid).

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
        'classical'  — model(x) returns pred directly (MLP, ClassPINN, TorchQAPINN)
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
            output, _ = model(Xq, Y)   # y passed raw, model uses n_y_qubits internally
        else:
            output = model(X)
    
        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output

        mse  = ((pred - Y)**2).mean().item()
        rmse = mse ** 0.5
        ss_res = ((pred - Y)**2).sum().item()
        ss_tot = ((Y - Y.mean(0, keepdim=True))**2).sum().item()
        r2   = 1 - ss_res / (ss_tot + 1e-12)

    return {"mse": mse, "rmse": rmse, "r2": r2, "pred": pred}


def predict(
    model,
    x:         float,
    t:         float,
    p_ratio:   float,
    mu:        float,
    rho_L:     float,
    rho_R:     float,
    p_R:       float,
    Xm:        torch.Tensor,
    Xs:        torch.Tensor,
    Ym:        torch.Tensor,
    Ys:        torch.Tensor,
    device:    torch.device,
    model_type: str = "classical",
):
    """
    Predict [rho, u, p] for a single (x, t, scenario) point.

    Denormalises inputs and outputs using the training stats so you
    can pass raw physical values and get raw physical predictions back.

    Args:
        x, t          : spatial position [0,1] and time [0, t_end]
        p_ratio       : pressure ratio p_L / p_R
        mu            : dynamic viscosity
        rho_L, rho_R  : left/right initial densities
        p_R           : right initial pressure
        Xm, Xs        : input normalisation mean/std  (from load_ns_data)
        Ym, Ys        : output normalisation mean/std (from load_ns_data)
        device        : torch.device
        model_type    : 'classical' for MLP/ClassPINN/TorchQAPINN, 'qpinn' for QPINN

    Returns:
        dict with keys: rho, u, p  (physical, denormalised values)
    """
    model.eval()
    with torch.no_grad():
        # Build normalised input vector [x, t, p_ratio, mu, rho_L, rho_R, p_R]
        raw = torch.tensor([[x, t, p_ratio, mu, rho_L, rho_R, p_R]],
                           dtype=torch.float32)
        X_norm = (raw - Xm.cpu()) / Xs.cpu()          # [1, NI=7]
        X_norm = X_norm.to(device)

        if model_type == "qpinn":
            n_q    = model.n_qubits
            n_feat = X_norm.shape[-1]
            Xq     = X_norm[:, torch.arange(n_q) % n_feat] \
                     if n_feat < n_q else X_norm[:, :n_q]
            # dummy y (not used for prediction, only for circuit structure)
            y_dummy = torch.zeros(1, model.n_out, device=device)
            pred_norm, _ = model(Xq, y_dummy)
        else:
            pred_norm, _ = model(X_norm)               # [1, NO=3]

        # Denormalise: pred_physical = pred_norm * Ys + Ym
        pred_phys = pred_norm.cpu() * Ys.cpu() + Ym.cpu()  # [1, 3]
        rho, u, p = pred_phys[0].tolist()

    return {"rho": rho, "u": u, "p": p}


def predict_field(
    model,
    t:         float,
    p_ratio:   float,
    mu:        float,
    rho_L:     float,
    rho_R:     float,
    p_R:       float,
    Xm:        torch.Tensor,
    Xs:        torch.Tensor,
    Ym:        torch.Tensor,
    Ys:        torch.Tensor,
    device:    torch.device,
    n_x:       int   = 256,
    model_type: str  = "classical",
):
    """
    Predict the full spatial field [rho(x), u(x), p(x)] at a given time t.

    Sweeps x from 0 to 1 with n_x points and returns arrays of length n_x.

    Returns:
        dict with keys: x, rho, u, p  (numpy arrays, physical values)
    """
    import numpy as np
    model.eval()
    xs = torch.linspace(0.0, 1.0, n_x)

    # Build full input matrix [n_x, 7]
    raw = torch.stack([
        xs,
        torch.full((n_x,), t),
        torch.full((n_x,), p_ratio),
        torch.full((n_x,), mu),
        torch.full((n_x,), rho_L),
        torch.full((n_x,), rho_R),
        torch.full((n_x,), p_R),
    ], dim=1)                                          # [n_x, 7]

    with torch.no_grad():
        X_norm = (raw - Xm.cpu()) / Xs.cpu()
        X_norm = X_norm.to(device)

        if model_type == "qpinn":
            n_q    = model.n_qubits
            n_feat = X_norm.shape[-1]
            Xq     = X_norm[:, torch.arange(n_q) % n_feat] \
                     if n_feat < n_q else X_norm[:, :n_q]
            y_dummy = torch.zeros(n_x, model.n_out, device=device)
            pred_norm, _ = model(Xq, y_dummy)
        else:
            pred_norm, _ = model(X_norm)

        pred_phys = pred_norm.cpu() * Ys.cpu() + Ym.cpu()  # [n_x, 3]

    return {
        "x":   xs.numpy(),
        "rho": pred_phys[:, 0].numpy(),
        "u":   pred_phys[:, 1].numpy(),
        "p":   pred_phys[:, 2].numpy(),
    }


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
# SECTION 7 — QPINN ENCODER × ANSATZ SWEEP
# ══════════════════════════════════════════════════════════════════════════════

# Fallback list — used only if utilities_quantum_torch exposes no registry.
# Edit to match whatever torch_encode / torch_ansatz actually support.
DEFAULT_ENCODINGS = ["arctan", "angle_full", "angle", "fft", "iqp", "amplitude"]
DEFAULT_ANSATZE   = ["u_ring", "hardware_efficient", "strongly", "basic"]


def discover_options():
    """
    Try to read the supported encoding / ansatz names straight out of
    utilities_quantum_torch so the sweep never guesses.

    Looks for module-level registries (ENCODINGS, ENCODING_MAP, ANSATZE, ...)
    in that order; falls back to the DEFAULT_* lists above.

    Returns:
        (encodings, ansatze)  — two lists of str
    """
    import utilities_quantum_torch as uq

    def _pull(cands, fallback):
        for name in cands:
            obj = getattr(uq, name, None)
            if isinstance(obj, dict) and obj:
                return sorted(obj.keys())
            if isinstance(obj, (list, tuple, set)) and obj:
                return sorted(obj)
        return list(fallback)

    enc = _pull(["ENCODINGS", "ENCODING_MAP", "ENCODERS", "VALID_ENCODINGS"],
                DEFAULT_ENCODINGS)
    ans = _pull(["ANSATZE", "ANSATZ_MAP", "ANSATZES", "VALID_ANSATZE"],
                DEFAULT_ANSATZE)
    return enc, ans


def probe_combo(encoding, ansatz, n_qubits, n_layers, n_enc_layers,
                device, n_out=3):
    """
    Cheap validity check: build the model and push 2 dummy samples through.
    Catches unsupported encoding/ansatz names in ~ms instead of after a
    full training run. Returns (ok: bool, message: str).
    """
    try:
        m = build_qpinn(
            n_qubits     = n_qubits,
            n_layers     = n_layers,
            n_enc_layers = n_enc_layers,
            encoding     = encoding,
            ansatz       = ansatz,
            n_out        = n_out,
            device       = device,
        )
        xb = torch.zeros(2, n_qubits, device=device)
        yb = torch.zeros(2, n_out,    device=device)
        with torch.no_grad():
            m(xb, yb)
        n_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return True, n_p
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def sweep_qpinn(
    X_tr, Y_tr, X_te, Y_te,
    encodings,                   # list[str]
    ansatze,                     # list[str]
    n_layers_list      = (4,),   # list[int] — ansatz depth
    n_enc_layers_list  = (2,),   # list[int] — x re-uploading blocks
    n_qubits           = 7,
    n_epochs           = 200,
    batch_size         = 512,
    lr                 = 3e-3,
    out_dir            = "./results/qpinn_sweep",
    checkpoint_every   = 50,
    device             = None,
    seed               = 0,
    log_every          = 25,     # 0 = silent per-epoch, only summary lines
    save_ckpts         = False,  # per-epoch ckpts off by default (sweeps are big)
    skip_existing      = True,   # resume: skip combos with metrics.json present
    probe_first        = True,   # dry-run each combo before committing to training
):
    """
    Grid-search QPINN over encoding × ansatz (× n_layers × n_enc_layers).

    Every combination is trained from the same seed on the same data split,
    so the resulting table is an apples-to-apples comparison. A combo that
    blows up (bad name, OOM, NaN) is recorded with a status flag and the
    sweep moves on rather than dying.

    Returns:
        list[dict] — one row per combo, sorted by R² descending
    """
    import itertools, traceback

    os.makedirs(out_dir, exist_ok=True)
    grid = list(itertools.product(encodings, ansatze,
                                  n_layers_list, n_enc_layers_list))
    total = len(grid)

    print(f"\n{'═'*72}")
    print(f"  QPINN SWEEP  —  {total} combinations")
    print(f"  encodings    : {list(encodings)}")
    print(f"  ansatze      : {list(ansatze)}")
    print(f"  n_layers     : {list(n_layers_list)}")
    print(f"  n_enc_layers : {list(n_enc_layers_list)}")
    print(f"  n_qubits={n_qubits}  epochs={n_epochs}  batch={batch_size}  lr={lr}")
    print(f"  out_dir      : {out_dir}")
    print(f"{'═'*72}")

    rows        = []
    sweep_t0    = time.time()
    results_path = os.path.join(out_dir, f"sweep_results_q{n_qubits}.json")

    for i, (enc, ans, nl, nel) in enumerate(grid, 1):
        tag = f"q{n_qubits}_l{nl}_e{nel}_{ans}_{enc}"
        run_dir = os.path.join(out_dir, tag)
        row = {
            "tag": tag, "encoding": enc, "ansatz": ans,
            "n_qubits": n_qubits, "n_layers": nl, "n_enc_layers": nel,
            "status": "pending",
        }

        print(f"\n{'─'*72}")
        print(f"[{i}/{total}]  encoding={enc:<12} ansatz={ans:<20} "
              f"n_layers={nl}  n_enc_layers={nel}")

        # ── resume support ────────────────────────────────────────────────────
        done_marker = os.path.join(run_dir, "qpinn_metrics.json")
        resume_ckpt, checkpoint_epochs = latest_checkpoint(run_dir, "qpinn")
        resume_needed = False
        if skip_existing and os.path.exists(done_marker):
            with open(done_marker) as f:
                prev = json.load(f)
            completed_epochs = len(prev.get("training", {}).get("loss_history") or [])
            actual_epochs = checkpoint_epochs or completed_epochs
            if actual_epochs < n_epochs:
                resume_needed = True
                print(f"  ↻ short run: checkpoint epoch "
                      f"{actual_epochs}/{n_epochs}; "
                      "loading checkpoint and continuing")
            else:
                row.update(prev.get("evaluation", {}))
                row["n_params"] = prev.get("config", {}).get("n_params", 0)
                row["final_loss"] = prev.get("training", {}).get("end_loss")
                row["status"] = "cached"
                rows.append(row)
                print(f"  ↩ cached  R²={row.get('r2')}  (delete {run_dir} to rerun)")
                continue
        elif resume_ckpt is not None and checkpoint_epochs < n_epochs:
            # An interrupted run may have a checkpoint but no metrics JSON.
            resume_needed = True
            print(f"  ↻ interrupted run: checkpoint epoch "
                  f"{checkpoint_epochs}/{n_epochs}; continuing")

        # ── cheap validity probe ──────────────────────────────────────────────
        if probe_first:
            ok, info = probe_combo(enc, ans, n_qubits, nl, nel, device)
            if not ok:
                row["status"] = "unsupported"
                row["error"]  = str(info)[:300]
                rows.append(row)
                print(f"  ✗ skipped — {row['error']}")
                continue
            row["n_params"] = info
            print(f"  probe ok — {info} trainable params")

        # ── train ─────────────────────────────────────────────────────────────
        try:
            # identical init + identical shuffling for every combo
            torch.manual_seed(seed)
            np.random.seed(seed)
            if device is not None and device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            cfg = {"model": "qpinn", "n_qubits": n_qubits, "n_layers": nl,
                   "n_enc_layers": nel, "encoding": enc, "ansatz": ans,
                   "epochs": n_epochs, "batch": batch_size, "lr": lr,
                   "seed": seed}

            model = build_qpinn(
                n_qubits     = n_qubits,
                n_layers     = nl,
                n_enc_layers = nel,
                encoding     = enc,
                ansatz       = ans,
                device       = device,
            )
            opt = optim.Adam(model.parameters(), lr=lr)
            row["n_params"] = sum(p.numel() for p in model.parameters()
                                  if p.requires_grad)
            cfg["n_params"] = row["n_params"]

            start_epoch, loss_h, mse_h = 0, [], []
            if resume_needed and resume_ckpt is not None:
                model, opt, start_epoch, loss_h, mse_h, _ = load_ckpt(
                    resume_ckpt, model, opt, device=device)
            elif resume_needed:
                print("  ! short metrics found but no checkpoint is available; "
                      "restarting this run from epoch 0")

            t0 = time.time()
            loss_h, mse_h = train_qpinn(
                model, opt, X_tr, Y_tr,
                n_epochs, batch_size,
                run_dir, checkpoint_every, cfg, device=device,
                start_epoch=start_epoch, loss_history=loss_h,
                mse_history=mse_h, log_every=log_every,
                save_ckpts=save_ckpts,
            )
            train_time = time.time() - t0

            metrics = evaluate(model, X_te, Y_te, device, "qpinn")
            save_results(run_dir, "qpinn", model, opt,
                         n_epochs, loss_h, mse_h, metrics, X_te, Y_te, cfg)

            row.update({
                "status":     "ok",
                "mse":        metrics["mse"],
                "rmse":       metrics["rmse"],
                "r2":         metrics["r2"],
                "final_loss": loss_h[-1] if loss_h else None,
                "final_mse":  mse_h[-1]  if mse_h  else None,
                "train_time": round(train_time, 1),
            })
            if not np.isfinite(metrics["mse"]):
                row["status"] = "diverged"

            print(f"  ✓ MSE={metrics['mse']:.6f}  RMSE={metrics['rmse']:.6f}  "
                  f"R²={metrics['r2']:.4f}  ({train_time:.0f}s)")

        except torch.cuda.OutOfMemoryError as e:
            row["status"] = "oom"
            row["error"]  = str(e)[:300]
            print(f"  ✗ OOM — reduce batch_size or n_qubits")
        except Exception as e:
            row["status"] = "failed"
            row["error"]  = f"{type(e).__name__}: {e}"[:300]
            print(f"  ✗ FAILED — {row['error']}")
            traceback.print_exc()
        finally:
            # free the statevector buffers before the next combo builds its own
            try:
                del model, opt
            except NameError:
                pass
            if device is not None and device.type == "cuda":
                torch.cuda.empty_cache()

        rows.append(row)

        # write partial results after every combo — a crash never loses the sweep
        with open(results_path, "w") as f:
            json.dump(rows, f, indent=2)

    # ── ranked summary ────────────────────────────────────────────────────────
    ok_rows  = [r for r in rows if r.get("status") in ("ok", "cached")
                and r.get("r2") is not None and np.isfinite(r["r2"])]
    bad_rows = [r for r in rows if r not in ok_rows]
    ok_rows.sort(key=lambda r: r["r2"], reverse=True)

    print(f"\n{'═'*72}")
    print(f"  QPINN SWEEP RESULTS  —  {len(ok_rows)}/{total} succeeded"
          f"  (total {time.time()-sweep_t0:.0f}s)")
    print(f"  {'encoding':<12} {'ansatz':<20} {'L':>2} {'E':>2} "
          f"{'params':>7} {'MSE':>10} {'R²':>8} {'sec':>6}")
    print(f"  {'─'*12} {'─'*20} {'─'*2} {'─'*2} {'─'*7} {'─'*10} {'─'*8} {'─'*6}")
    for r in ok_rows:
        print(f"  {r['encoding']:<12} {r['ansatz']:<20} "
              f"{r['n_layers']:>2} {r['n_enc_layers']:>2} "
              f"{r.get('n_params', 0):>7} {r.get('mse', float('nan')):>10.6f} "
              f"{r['r2']:>8.4f} {r.get('train_time', 0):>6.0f}")
    if bad_rows:
        print(f"\n  Skipped / failed:")
        for r in bad_rows:
            print(f"    {r['encoding']:<12} {r['ansatz']:<20} "
                  f"[{r['status']}] {r.get('error', '')[:80]}")
    print(f"{'═'*72}")

    if ok_rows:
        b = ok_rows[0]
        print(f"  BEST: encoding='{b['encoding']}'  ansatz='{b['ansatz']}'  "
              f"n_layers={b['n_layers']}  n_enc_layers={b['n_enc_layers']}  "
              f"→ R²={b['r2']:.4f}")

    # ── persist ───────────────────────────────────────────────────────────────
    with open(results_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  ✓ sweep JSON → {results_path}")

    csv_path = os.path.join(out_dir, "sweep_results.csv")
    cols = ["encoding", "ansatz", "n_qubits", "n_layers", "n_enc_layers",
            "n_params", "mse", "rmse", "r2", "final_loss", "train_time", "status"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in ok_rows + bad_rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"  ✓ sweep CSV  → {csv_path}")

    return ok_rows + bad_rows


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
    DATA_DIR         = "/mnt/d/QCFD/WISER/BQC/github/data"
    OUT_DIR          = "/mnt/d/QCFD/WISER/BQC/github/results"
    N_SCENARIOS      = 30
    T_STRIDE         = 6
    N_EPOCHS         = 200
    LR               = 3e-3
    CHECKPOINT_EVERY = 20

    # ── QPINN config (defined first — needed for auto batch size) ──────────────
    QPINN_N_QUBITS   = 7           # 7=safe on 8GB GPU; 10=needs batch<=64
    QPINN_N_LAYERS   = 4
    QPINN_N_ENC_LAYERS = 2         # data re-uploading blocks (1=no re-upload)
    QPINN_ENCODING   = "arctan"    # best: RY(arctan(x)), bounded input
    QPINN_ANSATZ     = "u_ring"

    # ── QPINN sweep config ─────────────────────────────────────────────────────
    # When True, the QPINN block runs a grid over encoding × ansatz instead of
    # a single (QPINN_ENCODING, QPINN_ANSATZ) run.
    QPINN_SWEEP           = True
    QPINN_SWEEP_AUTO      = False  # use the explicit encoding/ansatz lists below
    QPINN_SWEEP_QUBITS    = [4, 7, 10]
    QPINN_SWEEP_ENCODINGS = ["arctan", "angle_full", "fft", "iqp"]
    QPINN_SWEEP_ANSATZE   = ["u_ring", "hardware_efficient", "strongly"]
    QPINN_SWEEP_LAYERS    = [2, 3, 4]  # QVC ansatz depths
    QPINN_SWEEP_ENC_LAYERS= [1, 2]     # data re-uploading depths
    QPINN_SWEEP_EPOCHS    = 200        # drop to ~50 for a fast screening pass
    QPINN_SWEEP_LOG_EVERY = 25         # per-epoch print frequency inside a run
    QPINN_SWEEP_SEED      = 0          # same init for every combo → fair compare
    QPINN_SWEEP_SKIP_DONE = True       # resume an interrupted sweep

    # ── Batch size: auto-scale with qubit count ────────────────────────────────
    # Statevector memory per sample = 2^n_qubits * 8 bytes (cfloat)
    # Reference: batch=512 at n_qubits=7  (512 * 128 * 8 = 512 KB/batch)
    # n_qubits=7  -> batch=512   (2^7=128)
    # n_qubits=10 -> batch=64    (2^10=1024, 8x bigger statevector)
    # n_qubits=14 -> batch=4     (2^14=16384)
    _REF_BATCH   = 512
    _REF_QUBITS  = 7
    BATCH_SIZE   = max(8, _REF_BATCH * (2 ** _REF_QUBITS) // (2 ** QPINN_N_QUBITS))

    # ── Auto-select the GPU with most free memory ──────────────────────────────
    if torch.cuda.is_available():
        _gpu_id = get_free_gpu()
        DEVICE  = torch.device(f"cuda:{_gpu_id}")
        torch.cuda.set_device(_gpu_id)
    else:
        DEVICE  = torch.device("cpu")

    # which models to train
    RUN_MLP        = False
    RUN_CLASSPINN  = False
    RUN_QPINN      = True
    RUN_QAPINN     = False
    CLASSPINN_PRESET = "pinn"
    CLASSPINN_LR     = 1e-3  # separate, safer rate for autograd PDE residuals

    # TorchQAPINN config
    QAPINN_HIDDEN      = 128      # MLP hidden dim (same as MLP/ClassPINN)
    QAPINN_DEPTH       = 4        # total hidden layers
    QAPINN_N_QUBITS_LIST = [4, 7, 10]
    # Layer 0 is the mandatory NI -> hidden input projection (Sequential),
    # so the QuantumLayer can only replace hidden layers 1..depth-1.
    QAPINN_Q_LAYER_IDXS = [1, 2, 3]
    QAPINN_N_LAYERS_LIST = [2, 3, 4]
    QAPINN_ENCODINGS   = ["arctan", "angle_full", "fft", "iqp"]
    QAPINN_ANSATZE     = ["u_ring", "hardware_efficient", "strongly"]
    QAPINN_ACTIVATION  = "tanh"   # classical layer activation
    QAPINN_LAMBDA_Q    = 0.1      # quantum fidelity loss weight
    QAPINN_PHYSICS     = False    # add NS physics residuals

    print("=" * 65)
    print("  NS Shock Tube — Model Comparison")
    print("=" * 65)
    print(f"  device    : {DEVICE}")
    print(f"  data_dir  : {DATA_DIR}")
    print(f"  models    : MLP={RUN_MLP}  ClassPINN={RUN_CLASSPINN}"
          f"  QPINN={RUN_QPINN}  TorchQAPINN={RUN_QAPINN}")
    print(f"  epochs    : {N_EPOCHS}  lr={LR}")
    print(f"  QPINN     : n_qubits={QPINN_N_QUBITS}  statevec=2^{QPINN_N_QUBITS}={2**QPINN_N_QUBITS}"
          f"  batch={BATCH_SIZE} (auto-scaled)  enc_layers={QPINN_N_ENC_LAYERS}")
    mem_mb = BATCH_SIZE * (2**QPINN_N_QUBITS) * 8 / 1e6
    print(f"  memory/batch: {mem_mb:.2f} MB  ({BATCH_SIZE} * {2**QPINN_N_QUBITS} * 8 bytes)")

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

        loss_h, mse_h = train_classical(
            model_mlp, opt_mlp, X_tr, Y_tr,
            N_EPOCHS, BATCH_SIZE, Xm, Xs, p_range, mu_range,
            ckpt_dir, CHECKPOINT_EVERY, cfg, device=DEVICE,
        )
        metrics = evaluate(model_mlp, X_te, Y_te, DEVICE, "classical")
        print(f"\n  MLP test  MSE={metrics['mse']:.6f}  "
              f"RMSE={metrics['rmse']:.6f}  R²={metrics['r2']:.4f}")
        save_results(ckpt_dir, "mlp", model_mlp, opt_mlp,
                     N_EPOCHS, loss_h, mse_h, metrics, X_te, Y_te, cfg)
        results["mlp"] = metrics

    # ══════════════════════════════════════════════════════════════════════════
    # ClassPINN
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_CLASSPINN:
        print(f"\n{'─'*65}")
        print(f"[ ClassPINN ]  Classical PINN — preset='{CLASSPINN_PRESET}'")
        cfg = {"model": f"classpinn_{CLASSPINN_PRESET}",
               "preset": CLASSPINN_PRESET,
               "physics_residual": "autograd_ns_v1",
               "physics_weight": 0.01,
               "physics_warmup_epochs": 50,
               "epochs": N_EPOCHS, "batch": BATCH_SIZE,
               "lr": CLASSPINN_LR}
        # Do not resume the legacy finite-difference checkpoint: it was
        # trained with an invalid mixed-scenario smoothness penalty. Preserve
        # it for auditability and write corrected runs to their own folder.
        ckpt_dir = os.path.join(OUT_DIR,
                                f"classpinn_{CLASSPINN_PRESET}_autograd_ns")

        model_cp = build_classpinn(CLASSPINN_PRESET, DEVICE)
        _attach_stats(model_cp)
        model_cp.describe()
        opt_cp = optim.Adam(model_cp.parameters(), lr=CLASSPINN_LR)
        resume_path, resume_epoch = latest_checkpoint(
            ckpt_dir, f"classpinn_{CLASSPINN_PRESET}")
        start_epoch, loss_h, mse_h = 0, [], []
        if resume_path is not None:
            model_cp, opt_cp, start_epoch, loss_h, mse_h, old_cfg = load_ckpt(
                resume_path, model_cp, opt_cp, DEVICE)
            if old_cfg and any(old_cfg.get(k) != cfg.get(k)
                               for k in ("model", "preset",
                                         "physics_residual")):
                raise RuntimeError(
                    f"ClassPINN checkpoint config mismatch: {old_cfg} vs {cfg}")
            if start_epoch >= N_EPOCHS:
                print(f"  ClassPINN already complete at epoch {start_epoch}; "
                      "evaluating saved checkpoint")
            else:
                print(f"  Resuming ClassPINN at epoch {start_epoch}/{N_EPOCHS}")

        loss_h, mse_h = train_classical(
            model_cp, opt_cp, X_tr, Y_tr,
            N_EPOCHS, BATCH_SIZE, Xm, Xs, p_range, mu_range,
            ckpt_dir, CHECKPOINT_EVERY, cfg,
            start_epoch=start_epoch, loss_history=loss_h,
            mse_history=mse_h, device=DEVICE,
        )
        metrics = evaluate(model_cp, X_te, Y_te, DEVICE, "classical")
        print(f"\n  ClassPINN test  MSE={metrics['mse']:.6f}  "
              f"RMSE={metrics['rmse']:.6f}  R²={metrics['r2']:.4f}")
        save_results(ckpt_dir, f"classpinn_{CLASSPINN_PRESET}",
                     model_cp, opt_cp,
                     N_EPOCHS, loss_h, mse_h, metrics, X_te, Y_te, cfg)
        results["classpinn"] = metrics

    # ══════════════════════════════════════════════════════════════════════════
    # QPINN
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_QPINN and QPINN_SWEEP:
        # ── grid over encoders × ansatze ──────────────────────────────────────
        if QPINN_SWEEP_AUTO:
            try:
                encs, anss = discover_options()
                print(f"\n  Discovered encodings : {encs}")
                print(f"  Discovered ansatze   : {anss}")
            except Exception as e:
                print(f"\n  Auto-discovery failed ({e}) — using configured lists")
                encs, anss = QPINN_SWEEP_ENCODINGS, QPINN_SWEEP_ANSATZE
        else:
            encs, anss = QPINN_SWEEP_ENCODINGS, QPINN_SWEEP_ANSATZE

        for sweep_qubits in QPINN_SWEEP_QUBITS:
            # Preserve approximately constant statevector memory per batch.
            # q=4 -> 512 (capped), q=7 -> 512, q=10 -> 64.
            sweep_batch = max(
                8, min(_REF_BATCH,
                       _REF_BATCH * (2 ** _REF_QUBITS) //
                       (2 ** sweep_qubits)))
            print(f"\n  QPINN qubit sweep: q={sweep_qubits}, "
                  f"batch={sweep_batch}")
            sweep_rows = sweep_qpinn(
                X_tr, Y_tr, X_te, Y_te,
                encodings         = encs,
                ansatze           = anss,
                n_layers_list     = QPINN_SWEEP_LAYERS,
                n_enc_layers_list = QPINN_SWEEP_ENC_LAYERS,
                n_qubits          = sweep_qubits,
                n_epochs          = QPINN_SWEEP_EPOCHS,
                batch_size        = sweep_batch,
                lr                = LR,
                out_dir           = os.path.join(OUT_DIR, "qpinn_sweep"),
                checkpoint_every  = CHECKPOINT_EVERY,
                device            = DEVICE,
                seed              = QPINN_SWEEP_SEED,
                log_every         = QPINN_SWEEP_LOG_EVERY,
                save_ckpts        = True,
                skip_existing     = QPINN_SWEEP_SKIP_DONE,
            )
            for r in sweep_rows:
                if r.get("status") in ("ok", "cached"):
                    key = (f"qpinn[q{r['n_qubits']}/l{r['n_layers']}/"
                           f"e{r['n_enc_layers']}/{r['encoding']}/"
                           f"{r['ansatz']}]")
                    results[key] = {
                        "mse": r["mse"], "rmse": r["rmse"],
                        "r2": r["r2"]}

    elif RUN_QPINN:
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

        # NOTE: keyword args — passing DEVICE positionally here landed it in
        # `y_n_enc_layers`, so the model never saw the requested device.
        model_qp = build_qpinn(
            n_enc_layers = QPINN_N_ENC_LAYERS,
            encoding     = QPINN_ENCODING,
            ansatz       = QPINN_ANSATZ,
            n_qubits     = QPINN_N_QUBITS,
            n_layers     = QPINN_N_LAYERS, 
            device       = DEVICE,
        )
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
    # TorchQAPINN
    # ══════════════════════════════════════════════════════════════════════════
    if RUN_QAPINN:
        total_qa = (len(QAPINN_N_QUBITS_LIST) * len(QAPINN_Q_LAYER_IDXS) *
                    len(QAPINN_N_LAYERS_LIST) * len(QAPINN_ENCODINGS) *
                    len(QAPINN_ANSATZE))
        print(f"\n  QAPINN sweep: {total_qa} runs")
        sweep_grid = (
            (n_qubits, q_layer_idx, n_q_layers, encoding, ansatz)
            for n_qubits in QAPINN_N_QUBITS_LIST
            for q_layer_idx in QAPINN_Q_LAYER_IDXS
            for n_q_layers in QAPINN_N_LAYERS_LIST
            for encoding in QAPINN_ENCODINGS
            for ansatz in QAPINN_ANSATZE
        )
        for n_qubits, q_layer_idx, n_q_layers, encoding, ansatz in sweep_grid:
            tag = (f"qapinn_q{n_qubits}_l{n_q_layers}_"
                   f"qli{q_layer_idx}_{ansatz}_{encoding}")
            cfg = {
                "model": "qapinn", "hidden": QAPINN_HIDDEN,
                "depth": QAPINN_DEPTH, "q_layer_idx": q_layer_idx,
                "n_qubits": n_qubits, "n_layers": n_q_layers,
                "encoding": encoding, "ansatz": ansatz,
                "lambda_q": QAPINN_LAMBDA_Q,
                "use_physics": QAPINN_PHYSICS,
                "epochs": N_EPOCHS, "batch": BATCH_SIZE, "lr": LR,
            }
            ckpt_dir = os.path.join(OUT_DIR, "qapinn_sweep", tag)
            metrics_path = os.path.join(ckpt_dir, "qapinn_metrics.json")
            resume_ckpt, checkpoint_epochs = latest_checkpoint(
                ckpt_dir, "qapinn")
            resume_needed = False
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    previous = json.load(f)
                completed_epochs = len(
                    previous.get("training", {}).get("loss_history") or [])
                actual_epochs = checkpoint_epochs or completed_epochs
                if actual_epochs >= N_EPOCHS:
                    print(f"  ↷ skip completed {tag}")
                    cached = previous.get("evaluation", {})
                    if cached:
                        results[tag] = cached
                    continue
                resume_needed = True
                print(f"  ↻ short run: {tag} has "
                      f"checkpoint epoch {actual_epochs}/{N_EPOCHS}; resuming")
            elif resume_ckpt is not None and checkpoint_epochs < N_EPOCHS:
                resume_needed = True
                print(f"  ↻ interrupted run: {tag} has checkpoint epoch "
                      f"{checkpoint_epochs}/{N_EPOCHS}; resuming")

            print(f"\n{'─'*65}\n[ TorchQAPINN ] {tag}")
            model_qa = build_qapinn(
                hidden=QAPINN_HIDDEN, depth=QAPINN_DEPTH,
                q_layer_idx=q_layer_idx, n_qubits=n_qubits,
                n_layers=n_q_layers, encoding=encoding, ansatz=ansatz,
                activation=QAPINN_ACTIVATION, lambda_q=QAPINN_LAMBDA_Q,
                use_physics=QAPINN_PHYSICS, device=DEVICE,
            )
            _attach_stats(model_qa)
            model_qa.describe()
            cfg["n_params"] = sum(p.numel() for p in model_qa.parameters())
            opt_qa = optim.Adam(model_qa.parameters(), lr=LR)
            start_epoch, loss_h, mse_h = 0, [], []
            if resume_needed and resume_ckpt is not None:
                model_qa, opt_qa, start_epoch, loss_h, mse_h, _ = load_ckpt(
                    resume_ckpt, model_qa, opt_qa, device=DEVICE)
            elif resume_needed:
                print("  ! short metrics found but no checkpoint is available; "
                      "restarting this run from epoch 0")
            loss_h, mse_h = train_qapinn(
                model_qa, opt_qa, X_tr, Y_tr, N_EPOCHS, BATCH_SIZE,
                ckpt_dir, CHECKPOINT_EVERY, cfg, device=DEVICE,
                start_epoch=start_epoch, loss_history=loss_h,
                mse_history=mse_h,
            )
            metrics = evaluate(model_qa, X_te, Y_te, DEVICE, "classical")
            save_results(ckpt_dir, "qapinn", model_qa, opt_qa,
                         N_EPOCHS, loss_h, mse_h, metrics, X_te, Y_te, cfg)
            results[tag] = metrics

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
