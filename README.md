# Link
https://www.youtube.com/watch?v=PqMmDw0k2n4

# Quantum-Enhanced Physics-Informed Learning for a 1-D Navier–Stokes Shock Tube

This repository compares classical computational fluid dynamics (CFD), classical neural networks, physics-informed neural networks (PINNs), quantum PINNs (QPINNs), and quantum-augmented PINNs (QAPINNs) on a viscous one-dimensional shock-tube problem.

The classical finite-volume solver generates the ground-truth flow fields. The learning models then approximate density, velocity, and pressure as functions of space, time, and the initial physical conditions.


## Project objectives

- Generate a parameterized dataset using a classical 1-D compressible Navier–Stokes solver.
- Establish MLP and classical PINN baselines.
- Compare QPINN and QAPINN architectures across qubit counts, circuit depths, encoding methods, and ansatzes.
- Evaluate prediction accuracy using MSE, MAE, RMSE, and \(R^2\).
- Compare predicted and classical CFD space–time fields.
- Explain model behavior using permutation importance, gradient sensitivity, Integrated Gradients, and optional SHAP analysis.

## Physical problem

The simulator solves the viscous compressible Navier–Stokes equations in conservation form:

```text
Mass:      ∂ρ/∂t  + ∂(ρu)/∂x          = 0
Momentum:  ∂(ρu)/∂t + ∂(ρu²+p)/∂x    = ∂τ/∂x
Energy:    ∂E/∂t + ∂((E+p)u)/∂x      = ∂(τu-q)/∂x
```

The classical method uses:

- An explicit Lax–Friedrichs update for the inviscid flux.
- An implicit backward-Euler update for viscosity and Fourier heat conduction.
- Adaptive time steps based on the CFL condition.
- A uniform finite-volume grid.
- A mass-conservation quality check for every generated scenario.

### Scenario parameters

The Latin-hypercube sweep varies five physical parameters:

| Parameter | Meaning | Default sweep range |
|---|---|---:|
| `p_ratio` | Left/right pressure ratio | 2–20 |
| `mu` | Dynamic viscosity | 0–0.02 |
| `rho_L` | Initial left density | 0.5–2.0 |
| `rho_R` | Initial right density | 0.05–0.5 |
| `p_R` | Initial right pressure | 0.05–0.3 |

Each accepted scenario stores the spatial grid, time snapshots, density `rho`, velocity `u`, pressure `p`, temperature `T`, and diagnostic quantities.

## Learning problem

The main supervised models use seven normalized inputs:

```text
[x, t, p_ratio, mu, rho_L, rho_R, p_R]
```

and predict three flow variables:

```text
[rho, u, p]
```

### Models

| Model | Description |
|---|---|
| MLP | Classical data-driven baseline trained with supervised MSE. |
| ClassPINN | Classical neural network with Navier–Stokes physics residuals. |
| QPINN | Statevector-simulated variational quantum model with quantum fidelity, data, and PDE loss terms. |
| QAPINN | Hybrid classical network containing a configurable quantum layer. |

### Quantum encodings

`utilities_quantum_torch.py` implements:

- `angle`
- `angle_full`
- `angle_zz`
- `dense`
- `arctan`
- `iqp`
- `amplitude`
- `fft`
- `fft_phase`
- `fft_full`

The primary v5 training sweep evaluates `arctan`, `angle_full`, `fft`, and `iqp`.

### Quantum ansatzes

The quantum utility module implements:

- `u_ring`
- `u_full`
- `u_alternate`
- `efficient_su2`
- `real_amplitudes`
- `strongly`
- `ns_coupled`
- `brick_wall`
- `hardware_efficient`

The primary sweep evaluates `u_ring`, `hardware_efficient`, and `strongly`.

## Loss functions

### QPINN

The QPINN training objective combines classical prediction error with a quantum LCU loss:

```text
L_QPINN = lambda_data * MSE(pred, y)
        + LCU(lambda_fidelity * L_fidelity
            + lambda_PDE * L_PDE)

L_fidelity = 1 - |<psi|phi>|²
```

Here `psi` is produced by encoding the input and applying the trainable ansatz. The target state `phi` is produced by quantum-encoding the ground-truth output without a trainable ansatz. In Navier–Stokes mode, the LCU construction includes the fidelity branch and Pauli-operator branches representing mass, momentum, and energy residuals.

### QAPINN

The hybrid model uses:

```text
L_QAPINN = MSE(pred, y) + lambda_q * L_fidelity
```

where the fidelity term compares the state before and after the trainable quantum ansatz. Classical physics residuals can additionally be enabled with `use_physics=True`.

## Repository workflow

```text
Step 1: Classical CFD simulation and dataset generation
   ↓
Step 2: MLP, ClassPINN, QPINN, and QAPINN training
   ↓
Step 4: Training curves, metrics, parity plots, and field comparisons
   ↓
Step 4 XAI: Per-model explanations and cross-model comparison
```

Primary files:

| File | Purpose |
|---|---|
| `Step1_simulation_1D_navier_stokes_shock_tube.py` | Generate a single CFD solution or a Latin-hypercube dataset. |
| `Step2_train_ns_quantum_pinn_and_qpinn_torch_v5.py` | Train and resume MLP, ClassPINN, QPINN, and QAPINN runs. |
| `utilities_classical.py` | Classical models, dataset loading, and classical physics loss utilities. |
| `utilities_quantum_torch.py` | Quantum encodings, ansatzes, statevector operations, measurements, and quantum losses. |
| `Step4_plot_compare_models.py` | Create model-comparison figures and metric tables. |
| `Step4_xai_all_models.py` | Run XAI for saved models and skip cases with existing reports unless forced. |

## Installation

Python 3.10 or newer is recommended. Create and activate an isolated environment, then install the core dependencies:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install numpy scipy pandas matplotlib scikit-learn torch
```

Optional XAI dependencies:

```bash
pip install shap
```

For GPU training, install the PyTorch build appropriate for the system's CUDA version by following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Usage

Run commands from the repository directory.

### 1. Generate one classical simulation

```bash
python Step1_simulation_1D_navier_stokes_shock_tube.py \
  --p-ratio 10 \
  --mu 0.005
```

For headless execution:

```bash
python Step1_simulation_1D_navier_stokes_shock_tube.py --no-plot
```

### 2. Generate the training dataset

```bash
python Step1_simulation_1D_navier_stokes_shock_tube.py \
  --sweep \
  --n-scenarios 500 \
  --sweep-N 256 \
  --output-dir ./data
```

The sweep writes compressed scenario files and an index under `data/`.

### 3. Configure and train the models

The primary training settings are grouped near the bottom of `Step2_train_ns_quantum_pinn_and_qpinn_torch_v5.py`. Important parameters include:

```python
N_EPOCHS = 200

QPINN_SWEEP_QUBITS = [4, 7, 10]
QPINN_SWEEP_ENCODINGS = ["arctan", "angle_full", "fft", "iqp"]
QPINN_SWEEP_ANSATZE = ["u_ring", "hardware_efficient", "strongly"]
QPINN_SWEEP_LAYERS = [2, 3, 4]
QPINN_SWEEP_ENC_LAYERS = [1, 2]

QAPINN_N_QUBITS_LIST = [4, 7, 10]
QAPINN_Q_LAYER_IDXS = [1, 2, 3]
QAPINN_N_LAYERS_LIST = [2, 3, 4]
QAPINN_ENCODINGS = ["arctan", "angle_full", "fft", "iqp"]
QAPINN_ANSATZE = ["u_ring", "hardware_efficient", "strongly"]
```

The current training script uses WSL-style absolute paths such as `/mnt/d/QCFD/WISER/BQC/github`. Update `sys.path`, `DATA_DIR`, and `OUT_DIR` if the repository is stored elsewhere. Running from WSL is recommended when retaining the existing paths.

Then run:

```bash
python Step2_train_ns_quantum_pinn_and_qpinn_torch_v5.py
```

The training code saves checkpoints periodically, resumes incomplete runs, and skips already completed sweep configurations. Results are written beneath `results/`.

> **Resource note:** statevector memory and runtime scale exponentially as `2**n_qubits`. Reduce batch size when increasing the number of qubits.

### 4. Generate comparison figures

```bash
python Step4_plot_compare_models.py \
  --results-dir ./results \
  --data-dir ./data \
  --repo-dir . \
  --out-dir ./results/figures
```

Useful options:

```bash
# Show all available space/time slices
python Step4_plot_compare_models.py --results-dir ./results --list-slices

# Plot the eight highest-ranked runs
python Step4_plot_compare_models.py --results-dir ./results --top-k 8

# Plot all runs
python Step4_plot_compare_models.py --results-dir ./results --top-k 0

# Select quantum models by best R², lowest loss, or both
python Step4_plot_compare_models.py --results-dir ./results --select both
```

Generated outputs include training curves, field comparisons, parity plots, sweep-ablation figures, and metric tables.

### 5. Run XAI

```bash
python Step4_xai_all_models.py \
  --results_dir ./results \
  --data_dir ./data \
  --out_dir ./xai_results
```

Optional analyses:

```bash
python Step4_xai_all_models.py --shap --interaction
```

Run only selected model families:

```bash
python Step4_xai_all_models.py --only qpinn qapinn --top_k 3
```

By default, a model is skipped when its completed `xai_report.json` already exists. Use `--force` to recompute it.

## Results layout

```text
data/
  scenario_*.npz
  index.json

results/
  mlp/
  classpinn_*/
  qpinn_sweep/
  qapinn_sweep/
  figures/

xai_results/
  <model_name>/
    permutation_importance.png
    gradient_sensitivity.png
    integrated_gradients.png
    summary_dashboard.png
    xai_report.json
```

## Reproducibility notes

- Keep the same train/validation/test split when comparing model families.
- Compare models using predictions from the same test scenarios.
- Record the random seed, qubit count, circuit depth, encoding, ansatz, and checkpoint epoch.
- Do not compare reported \(R^2\) values from different test splits without marking that difference.
- Confirm that incomplete checkpoints are resumed to the configured target epoch before final comparisons.

## License

No license has been selected yet. Add a `LICENSE` file before inviting external reuse or contributions.

## Citation

If this repository supports a paper, report, or challenge submission, add the final citation here.

```bibtex
@misc{quantum_ns_shock_tube,
  title  = {Quantum-Enhanced Physics-Informed Learning for a 1-D Navier--Stokes Shock Tube},
  author = {Project team},
  year   = {2026},
  note   = {GitHub repository}
}
```
