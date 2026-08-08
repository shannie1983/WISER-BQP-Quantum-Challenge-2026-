"""
Step3_plot_compare_models.py
============================
Post-processing for Step2_train_ns_quantum_pinn_and_qpinn_torch_v5.py

Reads the results tree written by save_results() / sweep_qpinn() and produces:

  1. fig1_training_curves.png   — every run's training history in ONE figure
                                  (left: total loss, right: data MSE)

  2. fig2_field_comparison.png  — classical CFD simulation (ground truth)
                                  vs every model's prediction, ONE figure,
                                  3 panels: rho(x), u(x), p(x) at a fixed
                                  (t, scenario) slice of the test set

  3. fig3_parity.png            — predicted vs true scatter, one panel per model

  4. metrics_table.{csv,md}     — R², MSE, RMSE, MAE, per-variable + overall,
                                  recomputed from the saved predictions so
                                  every model is scored identically

It expects, anywhere under --results-dir:
    <run_dir>/<name>_metrics.json      (loss_history, mse_history, evaluation)
    <run_dir>/<name>_predictions.npz   (x_norm, y_norm, pred_norm, residual)

Everything is recomputed from the .npz files, so the table is self-consistent
even if different runs were evaluated at different times.

Usage
-----
    python Step3_plot_compare_models.py \
        --results-dir /mnt/d/QCFD/WISER/BQC/github/results \
        --out-dir     /mnt/d/QCFD/WISER/BQC/github/results/figures

Optional — de-normalise to physical units (rho, u, p in SI) by re-deriving
the normalisation stats from the raw data:

    python Step3_plot_compare_models.py --results-dir ... \
        --data-dir /mnt/d/QCFD/WISER/BQC/github/data \
        --n-scenarios 30 --t-stride 6 \
        --repo-dir /mnt/d/QCFD/WISER/BQC/github

Without --data-dir everything is plotted/scored in normalised units, which is
the space the models were actually trained in.

Useful flags
------------
    --top-k 8          only the 8 best runs (by R²) go on the field/parity plots
    --include qpinn    substring filter on run label (repeatable)
    --exclude fft      substring filter, applied after --include (repeatable)
    --list-slices      print the available test-set (t, scenario) slices and exit
    --slice 3          plot slice #3 instead of the largest one
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VAR_NAMES = ["rho", "u", "p"]
VAR_LABELS = [r"density  $\rho$", r"velocity  $u$", r"pressure  $p$"]
# X_te column order from load_ns_data(): [x, t, p_ratio, mu, rho_L, rho_R, p_R]
COL_X, COL_T = 0, 1
COL_SCENARIO = [1, 2, 3, 4, 5, 6]      # everything except x defines a slice
COL_CASE = [2, 3, 4, 5, 6]             # scenario without t — one space-time map

# Model families. QAPINN is checked before QPINN — Step2 has written hybrid
# runs into directories named "qpinn_..." in the past, so config["model"] is
# trusted first and the path only used as a fallback.
FAMILY_ORDER = ["mlp", "classpinn", "qapinn", "qpinn", "other"]
FAMILY_DISPLAY = {"mlp": "MLP", "classpinn": "ClassPINN",
                  "qapinn": "QAPINN", "qpinn": "QPINN", "other": "other"}
FAMILY_CMAP = {"mlp": plt.cm.Blues, "classpinn": plt.cm.Greens,
               "qapinn": plt.cm.Purples, "qpinn": plt.cm.Oranges,
               "other": plt.cm.Greys}
# hyper-parameters worth plotting in a sweep summary
NUMERIC_HP = ["n_qubits", "n_layers", "n_enc_layers", "q_layer_idx",
              "depth", "hidden", "lambda_q"]


def detect_family(config, model_name, path):
    """Classify a run as mlp / classpinn / qapinn / qpinn."""
    for src in (str(config.get("model", "")).lower(), model_name.lower(),
                path.lower()):
        if "qapinn" in src:
            return "qapinn"
        if "qpinn" in src:
            return "qpinn"
        if "classpinn" in src or "pinn" in src:
            return "classpinn"
        if "mlp" in src:
            return "mlp"
    return "other"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def _label_from_path(run_dir, results_dir, model_name):
    """Short config tag for a run: its path relative to results_dir, trimmed."""
    rel = os.path.relpath(run_dir, results_dir).replace(os.sep, "/")
    if rel in (".", ""):
        return model_name
    rel = re.sub(r"^(qapinn|qpinn|classpinn|mlp)_sweep/", "", rel)
    rel = re.sub(r"^(qapinn|qpinn|classpinn|mlp)_", "", rel)
    return rel or model_name


def discover_runs(results_dir):
    """
    Walk results_dir and collect every completed run.

    Returns list of dicts:
        label, dir, model_name, metrics (json), npz_path,
        loss_history, mse_history, config
    """
    runs = []
    for root, _dirs, files in os.walk(results_dir):
        for fn in sorted(files):
            if not fn.endswith("_metrics.json"):
                continue
            model_name = fn[: -len("_metrics.json")]
            mpath = os.path.join(root, fn)
            try:
                with open(mpath) as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"  ! unreadable {mpath}: {e}")
                continue

            npz = os.path.join(root, f"{model_name}_predictions.npz")
            cfg = meta.get("config", {})
            fam = detect_family(cfg, model_name, os.path.relpath(root, results_dir))
            tag = _label_from_path(root, results_dir, model_name)
            label = (f"{FAMILY_DISPLAY[fam]} · {tag}"
                     if tag.lower() not in (fam, model_name.lower())
                     else FAMILY_DISPLAY[fam])
            runs.append({
                "label":        label,
                "family":       fam,
                "tag":          tag,
                "dir":          root,
                "model_name":   model_name,
                "config":       cfg,
                "loss_history": meta.get("training", {}).get("loss_history") or [],
                "mse_history":  meta.get("training", {}).get("mse_history") or [],
                "r2_history":   meta.get("training", {}).get("r2_history") or [],
                "reported":     meta.get("evaluation", {}),
                "npz_path":     npz if os.path.exists(npz) else None,
            })

    # guarantee unique labels (two dirs can trim to the same tag)
    seen = {}
    for r in runs:
        if r["label"] in seen:
            r["label"] = f"{r['label']} [{os.path.basename(r['dir'])}]"
        seen[r["label"]] = True

    if not runs:
        raise SystemExit(
            f"No '*_metrics.json' found under {results_dir!r}.\n"
            f"Point --results-dir at the OUT_DIR used in Step2 "
            f"(the folder holding mlp/, qapinn/, qpinn_sweep/, ...)."
        )

    runs.sort(key=lambda r: r["label"])
    return runs


def filter_runs(runs, include, exclude):
    """Substring include/exclude filters on the run label."""
    out = runs
    if include:
        out = [r for r in out if any(s.lower() in r["label"].lower() for s in include)]
    if exclude:
        out = [r for r in out if not any(s.lower() in r["label"].lower() for s in exclude)]
    if not out:
        raise SystemExit("All runs filtered out — loosen --include / --exclude.")
    return out


def load_predictions(run, n_out=3):
    """
    Attach y_true / y_pred / X arrays (normalised space) to a run dict.
    Returns False if the run has no usable predictions file.
    """
    if run["npz_path"] is None:
        return False
    try:
        d = np.load(run["npz_path"])
        X = np.asarray(d["x_norm"],    dtype=np.float64)
        Y = np.asarray(d["y_norm"],    dtype=np.float64)
        P = np.asarray(d["pred_norm"], dtype=np.float64)
    except Exception as e:
        print(f"  ! unreadable {run['npz_path']}: {e}")
        return False

    # QPINN readouts can carry n_qubits columns — keep the first n_out
    if P.ndim == 1:
        P = P[:, None]
    if P.shape[1] > Y.shape[1]:
        P = P[:, : Y.shape[1]]
    if P.shape != Y.shape:
        print(f"  ! shape mismatch in {run['label']}: pred{P.shape} vs true{Y.shape}")
        return False

    run["X"], run["y_true"], run["y_pred"] = X, Y, P
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — METRICS
# ══════════════════════════════════════════════════════════════════════════════

def split_signature(run):
    """
    Fingerprint of a run's test set.

    Runs trained with different N_SCENARIOS / T_STRIDE (or a different data
    load) end up with different test sets — their rows are NOT comparable
    point-by-point, and their MSE/R² are not on the same footing either.
    """
    import hashlib
    y = np.ascontiguousarray(np.round(run["y_true"], 6))
    h = hashlib.md5(y.tobytes()).hexdigest()[:8]
    return f"{y.shape[0]}x{y.shape[1]}:{h}"


def group_runs_by_split(runs):
    """Return list of (signature, [runs]) ordered by group size descending."""
    groups = {}
    for r in runs:
        sig = split_signature(r)
        r["split_sig"] = sig
        groups.setdefault(sig, []).append(r)
    out = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for gid, (sig, members) in enumerate(out):
        for r in members:
            r["split_group"] = gid
    return out


def r2_score(y, p):
    """Global R², matching evaluate() in Step2 (pooled over all elements)."""
    ss_res = np.sum((p - y) ** 2)
    ss_tot = np.sum((y - y.mean(axis=0, keepdims=True)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def compute_metrics(run, scale=None):
    """
    MSE / RMSE / MAE / R² overall and per output variable.

    scale : optional (Ym, Ys) tuple — if given, metrics are ALSO reported in
            physical units (pred_phys = pred_norm * Ys + Ym).
    """
    y, p = run["y_true"], run["y_pred"]
    row = {
        "model": run["label"],
        "family": run.get("family", "other"),
        "n_test": len(y),
        "split": run.get("split_group", 0),
        "mse":  float(np.mean((p - y) ** 2)),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae":  float(np.mean(np.abs(p - y))),
        "r2":   float(r2_score(y, p)),
    }
    for j, v in enumerate(VAR_NAMES[: y.shape[1]]):
        yj, pj = y[:, j], p[:, j]
        row[f"mse_{v}"] = float(np.mean((pj - yj) ** 2))
        row[f"r2_{v}"]  = float(1.0 - np.sum((pj - yj) ** 2)
                                / (np.sum((yj - yj.mean()) ** 2) + 1e-12))

    if scale is not None:
        Ym, Ys = scale
        yp, pp = y * Ys + Ym, p * Ys + Ym
        row["mse_phys"]  = float(np.mean((pp - yp) ** 2))
        row["rmse_phys"] = float(np.sqrt(np.mean((pp - yp) ** 2)))
        for j, v in enumerate(VAR_NAMES[: y.shape[1]]):
            row[f"rmse_phys_{v}"] = float(np.sqrt(np.mean((pp[:, j] - yp[:, j]) ** 2)))

    rep = run.get("reported") or {}
    row["mse_reported"] = rep.get("mse")
    row["r2_reported"]  = rep.get("r2")
    row["epochs"]       = run.get("config", {}).get("epochs")
    row["n_params"]     = run.get("config", {}).get("n_params")
    return row


def print_table(rows, physical=False):
    """Console table, ranked by R²."""
    has_phys = physical and any("rmse_phys" in r for r in rows)
    head = (f"  {'model':<34} {'n_test':>8} {'sp':>3} {'MSE':>11} {'RMSE':>10}"
            f" {'MAE':>10} {'R²':>9} {'R²ρ':>8} {'R²u':>8} {'R²p':>8}")
    if has_phys:
        head += f" {'RMSE(phys)':>11}"
    print("\n" + "═" * len(head))
    print("  MODEL COMPARISON  —  test set, normalised units"
          + ("  (+ physical RMSE)" if has_phys else ""))
    print("═" * len(head))
    print(head)
    print("  " + "─" * (len(head) - 2))
    for r in rows:
        line = (f"  {r['model'][:34]:<34} {r['n_test']:>8d} {r.get('split', 0):>3d}"
                f" {r['mse']:>11.3e} {r['rmse']:>10.4f}"
                f" {r['mae']:>10.4f} {r['r2']:>9.4f}"
                f" {r.get('r2_rho', float('nan')):>8.4f}"
                f" {r.get('r2_u',   float('nan')):>8.4f}"
                f" {r.get('r2_p',   float('nan')):>8.4f}")
        if has_phys:
            line += f" {r.get('rmse_phys', float('nan')):>11.4f}"
        print(line)
    print("═" * len(head))


def save_table(rows, out_dir, physical=False):
    """Write metrics_table.csv and metrics_table.md."""
    cols = ["model", "family", "n_test", "split", "epochs", "n_params",
            "mse", "rmse", "mae", "r2",
            "mse_rho", "mse_u", "mse_p", "r2_rho", "r2_u", "r2_p",
            "mse_reported", "r2_reported"]
    if physical:
        cols += ["mse_phys", "rmse_phys",
                 "rmse_phys_rho", "rmse_phys_u", "rmse_phys_p"]

    csv_path = os.path.join(out_dir, "metrics_table.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(
                "" if r.get(c) is None else
                (f"{r[c]:.8g}" if isinstance(r.get(c), float) else str(r.get(c)))
                for c in cols) + "\n")

    md_cols = ["model", "mse", "rmse", "mae", "r2", "r2_rho", "r2_u", "r2_p"]
    md_path = os.path.join(out_dir, "metrics_table.md")
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(md_cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(md_cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(
                r[c] if c == "model" else f"{r.get(c, float('nan')):.4g}"
                for c in md_cols) + " |\n")

    print(f"  ✓ table  → {csv_path}")
    print(f"  ✓ table  → {md_path}")
    return csv_path, md_path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FIGURE 1: ALL TRAINING CURVES IN ONE FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def _palette(n):
    if n <= 10:
        return [plt.cm.tab10(i) for i in range(n)]
    if n <= 20:
        return [plt.cm.tab20(i / 19.0) for i in range(n)]
    return [plt.cm.turbo(i / max(n - 1, 1)) for i in range(n)]


# High-contrast qualitative palette for within-row variants (fig7). Family
# shading is deliberately monochrome elsewhere; inside one ablation row the
# variants must be told apart at a glance, so hue does the work instead.
DISTINCT_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#8c564b",  # brown
    "#bcbd22",  # olive
    "#7f7f7f",  # grey
]
DISTINCT_STYLES = ["-", "--", "-.", ":"]


def distinct_style(k):
    """(colour, linestyle, marker) for variant k — hue first, then dash."""
    return (DISTINCT_COLORS[k % len(DISTINCT_COLORS)],
            DISTINCT_STYLES[(k // len(DISTINCT_COLORS)) % len(DISTINCT_STYLES)],
            ["o", "s", "^", "D", "v", "P"][k % 6])


def family_colors(runs):
    """
    One colour per run, shaded within its family: all QAPINN runs are purple,
    QPINN orange, ClassPINN green, MLP blue. Makes a 20-run sweep readable and
    keeps the hybrid runs visually distinct from the pure-quantum ones.
    """
    by_fam = {}
    for r in runs:
        by_fam.setdefault(r.get("family", "other"), []).append(r["label"])
    cmap = {}
    for fam, labels in by_fam.items():
        base = FAMILY_CMAP.get(fam, plt.cm.Greys)
        n = len(labels)
        for i, lab in enumerate(labels):
            cmap[lab] = base(0.85 - 0.5 * (i / max(n - 1, 1))) if n > 1 \
                else base(0.75)
    return cmap


def select_for_plot(runs_by_r2, top_k, family_guarantee=True):
    """
    Pick which runs go on the field/parity plots.

    The best run of every family is kept first — otherwise a large QPINN or
    QAPINN sweep can push the MLP/ClassPINN baselines (or the hybrid runs)
    off the figure entirely. Remaining slots are filled by R² order.
    """
    if not top_k or top_k <= 0:
        return list(runs_by_r2)
    chosen = []
    if family_guarantee:
        seen = set()
        for r in runs_by_r2:
            fam = r.get("family", "other")
            if fam not in seen:
                seen.add(fam)
                chosen.append(r)
    for r in runs_by_r2:
        if len(chosen) >= top_k:
            break
        if r not in chosen:
            chosen.append(r)
    return sorted(chosen, key=lambda r: runs_by_r2.index(r))


def plot_training_curves(runs, out_dir, smooth=1, subtitle=""):
    """
    Overlaid training histories, three panels:

        (a) training loss   — loss_history from each metrics.json
        (b) data MSE        — mse_history
        (c) R²              — per-epoch R², see note below

    Step2 records loss_history and mse_history but no R² history, so (c) is
    derived as  R² = 1 - MSE / Var(y)  using the pooled variance of that run's
    test targets. That identity is exact for the pooled R² this script reports;
    the approximation is only that mse_history is measured on the training
    batches while Var(y) comes from the test targets. The final *test* R² is
    marked with a star so the curve can be sanity-checked against it. If a
    run's metrics.json contains an "r2_history" key it is used directly.

    train_classical() in Step2 passes [] for mse_history, so MLP and ClassPINN
    have no curve in (b) or (c) until that loop logs one.
    """
    with_loss = [r for r in runs if len(r["loss_history"]) > 1]
    if not with_loss:
        print("  ! no loss histories found — skipping fig1")
        return None

    def _smooth(v):
        v = np.asarray(v, dtype=float)
        if smooth > 1 and len(v) > smooth:
            return np.convolve(v, np.ones(smooth) / smooth, mode="valid")
        return v

    def _r2_curve(run):
        """Per-epoch R²: logged if available, else 1 - MSE/Var(y)."""
        if len(run.get("r2_history") or []) > 1:
            return np.asarray(run["r2_history"], dtype=float), True
        if len(run["mse_history"]) < 2 or "y_true" not in run:
            return None, False
        y = run["y_true"]
        var = float(np.mean((y - y.mean(axis=0, keepdims=True)) ** 2))
        if var <= 0:
            return None, False
        return 1.0 - np.asarray(run["mse_history"], dtype=float) / var, False

    cmap = family_colors(with_loss)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))

    # ── (a) training loss ────────────────────────────────────────────────────
    for r in with_loss:
        y = _smooth(r["loss_history"])
        axes[0].plot(np.arange(1, len(y) + 1), y, lw=1.7,
                     color=cmap[r["label"]], label=r["label"])
    axes[0].set_ylabel("training loss (total objective)")
    axes[0].set_title("(a) training loss")
    axes[0].set_yscale("log")

    # ── (b) data MSE ─────────────────────────────────────────────────────────
    missing = []
    for r in with_loss:
        if len(r["mse_history"]) < 2:
            missing.append(r["label"])
            continue
        y = _smooth(r["mse_history"])
        axes[1].plot(np.arange(1, len(y) + 1), y, lw=1.7,
                     color=cmap[r["label"]], label=r["label"])
    axes[1].set_ylabel("data MSE (normalised)")
    axes[1].set_title("(b) data MSE")
    axes[1].set_yscale("log")

    # ── (c) R² ───────────────────────────────────────────────────────────────
    logged_any, derived_any = False, False
    for r in with_loss:
        curve, logged = _r2_curve(r)
        if curve is None:
            continue
        logged_any |= logged
        derived_any |= not logged
        y = _smooth(curve)
        axes[2].plot(np.arange(1, len(y) + 1), y, lw=1.7,
                     color=cmap[r["label"]], label=r["label"])
        if "y_true" in r:
            axes[2].scatter([len(r["mse_history"])],
                            [r2_score(r["y_true"], r["y_pred"])],
                            marker="*", s=110, zorder=6,
                            color=cmap[r["label"]], edgecolor="k", linewidth=0.4)
    axes[2].set_ylabel("R²")
    axes[2].set_title("(c) R²"
                      + ("  (logged)" if logged_any and not derived_any
                         else "  (from MSE; ★ = final test R²)"))
    ax_lines = axes[2].get_lines()
    if ax_lines:
        allv = np.concatenate([ln.get_ydata() for ln in ax_lines])
        allv = allv[np.isfinite(allv)]
        if allv.size:
            axes[2].set_ylim(max(-0.05, float(np.percentile(allv, 2))), 1.02)

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3, which="both", ls=":")

    if missing:
        axes[1].text(0.5, 0.06,
                     "no mse_history: " + ", ".join(m.split(" · ")[0]
                                                    for m in missing),
                     transform=axes[1].transAxes, ha="center", fontsize=8,
                     color="0.35")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.005, 0.5),
               fontsize=9, frameon=False)
    fig.suptitle("NS shock tube — training histories"
                 + (f"\n{subtitle}" if subtitle else ""), fontsize=13, y=1.03)
    fig.tight_layout()

    path = os.path.join(out_dir, "fig1_training_curves.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ figure → {path}")
    return path



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FIGURE 2: CFD vs ALL MODELS, ONE FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def find_slices(X, min_pts=12):
    """
    Group test rows into (t, scenario) slices.

    A slice = rows sharing [t, p_ratio, mu, rho_L, rho_R, p_R]; within a slice
    the only varying input is x, so it is a 1-D field profile that can be
    plotted against the classical simulation.

    Returns list of dicts {key, idx (sorted by x), n} ordered by size desc.
    """
    keys = np.round(X[:, COL_SCENARIO], 5)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    slices = []
    for k in range(len(uniq)):
        idx = np.where(inv == k)[0]
        if len(idx) < min_pts:
            continue
        idx = idx[np.argsort(X[idx, COL_X])]
        slices.append({"key": uniq[k], "idx": idx, "n": len(idx)})
    slices.sort(key=lambda s: -s["n"])
    return slices


def match_slice(run, key, min_pts=4, atol=1e-4):
    """
    Find the rows of `run` belonging to the scenario/time slice `key`.

    Matching is done on the input coordinates, not on row position, so runs
    with different test-set sizes (different N_SCENARIOS / T_STRIDE) still
    line up — or get cleanly reported as "not present" instead of crashing.

    Returns row indices sorted by x, or None.
    """
    K = np.round(run["X"][:, COL_SCENARIO], 5)
    m = np.all(np.abs(K - key[None, :]) <= atol, axis=1)
    idx = np.where(m)[0]
    if len(idx) < min_pts:
        return None
    return idx[np.argsort(run["X"][idx, COL_X])]


def plot_field_comparison(runs, ref_run, out_dir, slice_no=0, scale=None,
                          xscale=None, min_pts=12):
    """
    One figure, 3 panels (rho, u, p) vs x:
      black dashed = classical CFD simulation (ground truth)
      colours      = each model's prediction over the same slice

    Each run is located in its own arrays by matching the slice coordinates,
    so runs whose test split differs from the reference are either plotted
    correctly on their own rows or skipped with a warning.
    """
    X = ref_run["X"]
    slices = find_slices(X, min_pts=min_pts)
    if not slices:
        print("  ! no test slice with enough points — skipping fig2")
        return None
    sl = slices[min(slice_no, len(slices) - 1)]
    idx = sl["idx"]

    y_true = ref_run["y_true"][idx]
    xs = X[idx, COL_X]
    t_norm = X[idx[0], COL_T]

    def denorm_y(a):
        return a * scale[1] + scale[0] if scale is not None else a

    if xscale is not None:      # de-normalise x and t for the axis labels
        Xm, Xs = xscale
        xs_plot = xs * Xs[COL_X] + Xm[COL_X]
        t_plot = t_norm * Xs[COL_T] + Xm[COL_T]
        xlab, tlab = "x", f"t = {t_plot:.4g}"
    else:
        xs_plot, xlab, tlab = xs, "x (normalised)", f"t = {t_norm:.3g} (norm.)"

    n_var = y_true.shape[1]
    fig, axes = plt.subplots(1, n_var, figsize=(5.2 * n_var, 4.4), squeeze=False)
    axes = axes[0]
    colors = [family_colors(runs)[r["label"]] for r in runs]

    yt = denorm_y(y_true)
    for j in range(n_var):
        axes[j].plot(xs_plot, yt[:, j], "k--", lw=2.4, zorder=10,
                     label="classical CFD (ground truth)")

    n_plotted, skipped = 0, []
    for r, c in zip(runs, colors):
        if r is ref_run:
            r_idx = idx
        else:
            r_idx = match_slice(r, sl["key"])
            if r_idx is None:
                skipped.append(r["label"])
                continue
        yp = denorm_y(r["y_pred"][r_idx])
        rx = r["X"][r_idx, COL_X]
        rx_plot = rx * xscale[1][COL_X] + xscale[0][COL_X] if xscale else rx
        for j in range(n_var):
            axes[j].plot(rx_plot, yp[:, j], lw=1.4, color=c, alpha=0.9,
                         marker="o", ms=2.5, label=r["label"])
        n_plotted += 1

    if skipped:
        print(f"  ! slice #{slice_no} is not in the test set of "
              f"{len(skipped)} run(s) — they use a different split: "
              f"{', '.join(skipped[:4])}{' ...' if len(skipped) > 4 else ''}")
        print("    → use --group N (see the split column in the table) or "
              "--include to plot one split at a time")
    if n_plotted == 0:
        print("  ! nothing to overlay — skipping fig2")
        plt.close(fig)
        return None

    for j in range(n_var):
        axes[j].set_xlabel(xlab)
        axes[j].set_ylabel(VAR_LABELS[j] if scale is not None
                           else VAR_LABELS[j] + "  (normalised)")
        axes[j].set_title(VAR_NAMES[j])
        axes[j].grid(alpha=0.3, ls=":")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.005, 0.5),
               fontsize=8, frameon=False)
    fig.suptitle(f"Classical simulation vs model predictions  —  "
                 f"test slice #{slice_no}  ({sl['n']} points,  {tlab})",
                 fontsize=13, y=1.03)
    fig.tight_layout()

    path = os.path.join(out_dir, "fig2_field_comparison.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ figure → {path}")
    return path


def plot_parity(runs, out_dir, max_pts=4000, seed=0):
    """Predicted vs true scatter — one small panel per model, one figure."""
    n = len(runs)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.3 * nrow),
                             squeeze=False)
    rng = np.random.default_rng(seed)
    colors = _palette(n)

    for ax, r, c in zip(axes.ravel(), runs, colors):
        y, p = r["y_true"], r["y_pred"]
        k = min(max_pts, len(y))
        sel = rng.choice(len(y), size=k, replace=False)
        for j in range(y.shape[1]):
            ax.scatter(y[sel, j], p[sel, j], s=3, alpha=0.25,
                       label=VAR_NAMES[j] if r is runs[0] else None)
        lo = float(min(y.min(), p.min()))
        hi = float(max(y.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{r['label']}\nR²={r2_score(y, p):.4f}", fontsize=8)
        ax.set_xlabel("true (norm.)", fontsize=8)
        ax.set_ylabel("predicted (norm.)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, ls=":")

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    h, l = axes[0, 0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower right", markerscale=4, fontsize=9,
                   frameon=False)
    fig.suptitle("Parity plots — predicted vs classical simulation", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = os.path.join(out_dir, "fig3_parity.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ figure → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4b — FAMILY SUMMARY + SWEEP SUMMARIES (QAPINN / QPINN)
# ══════════════════════════════════════════════════════════════════════════════

def best_per_family(runs):
    """
    Best run of each family, ordered MLP → ClassPINN → QAPINN → QPINN.

    Returns list of (family, run, r2, mse).
    """
    best = {}
    for r in runs:
        fam = r.get("family", "other")
        score = r2_score(r["y_true"], r["y_pred"])
        mse = float(np.mean((r["y_pred"] - r["y_true"]) ** 2))
        # On one fixed test split, maximum R² and minimum MSE have identical
        # ordering. Keep both explicitly in the key so the intended selection
        # remains clear and MSE resolves any numerical tie.
        if (fam not in best
                or (score, -mse) > (best[fam][1], -best[fam][2])):
            best[fam] = (r, score, mse)
    return [(f, *best[f]) for f in FAMILY_ORDER if f in best]


def _final_loss(run):
    """Final training loss, or +inf if the run recorded no history."""
    lh = run.get("loss_history") or []
    return float(lh[-1]) if lh else float("inf")


def _test_mse(run):
    """Pooled normalised test MSE used consistently by every plot."""
    return float(np.mean((run["y_pred"] - run["y_true"]) ** 2))


def select_curve_runs(runs, all_runs=None, mode="both",
                      dual_families=("qapinn", "qpinn")):
    """
    Runs for the training-history figure.

    mode="r2"    one run per family, highest test R²  (the old behaviour)
    mode="loss"  one run per family, lowest test data MSE
    mode="both"  best R² and minimum test MSE for quantum families, with
                 duplicate checkpoints shown only once. On a common test set
                 these criteria are mathematically equivalent.

    QAPINN is represented by its best-R² run at each trained quantum depth
    n_layers=2, 3 and 4. Returns shallow copies with the selection criterion
    appended to `label`, so curves are distinguishable in the legend.
    """
    pool = list(all_runs) if all_runs else list(runs)
    in_ref = {id(r) for r in runs}
    out = []
    for fam in FAMILY_ORDER:
        cands = [r for r in runs if r.get("family") == fam]
        foreign = False
        if not cands:
            cands = [r for r in pool
                     if r.get("family") == fam and id(r) not in in_ref]
            foreign = bool(cands)
        if not cands:
            continue

        # Figure 1 should expose the effect of QAPINN circuit depth instead of
        # hiding it behind one global argmax (or a best/min-loss pair).
        if fam == "qapinn":
            for depth in (2, 3, 4):
                depth_runs = [r for r in cands
                              if _hp(r, "n_layers") == depth]
                if not depth_runs:
                    continue
                run = max(depth_runs,
                          key=lambda r: (r2_score(r["y_true"], r["y_pred"]),
                                         -_test_mse(r)))
                c = dict(run)
                c["label"] = (f"{FAMILY_DISPLAY[fam]} · n_layers={depth}"
                              "  [best R² / min test MSE at depth]")
                c["_foreign"] = foreign
                out.append(c)
            continue

        best_r2 = max(cands,
                      key=lambda r: (r2_score(r["y_true"], r["y_pred"]),
                                     -_test_mse(r)))
        best_mse = min(cands, key=_test_mse)
        picks = []
        if mode in ("r2", "both"):
            why = ("best R² / min test MSE"
                   if best_mse is best_r2 else "best R²")
            picks.append((best_r2, why))
        if mode == "loss":
            picks.append((best_mse, "min test MSE"))
        if (mode == "both" and fam in dual_families
                and best_mse is not best_r2):
            picks.append((best_mse, "min test MSE"))

        for run, why in picks:
            c = dict(run)
            suffix = f"  [{why}]" if len(picks) > 1 or mode == "loss" else ""
            c["label"] = run["label"] + suffix
            c["_foreign"] = foreign
            out.append(c)
    return out


def select_family_rows(runs, all_runs=None):
    """
    Best run of each family, preferring the reference split and falling back
    to any other split so that MLP / ClassPINN / QAPINN / QPINN all get a row
    (or a curve) even when they were trained on different data settings.

    Returns list of (family, run, r2, mse, foreign) in FAMILY_ORDER, plus the
    list of family display names that had no scored run at all.
    """
    pool = list(all_runs) if all_runs else list(runs)
    in_ref = {id(r) for r in runs}
    chosen = [(f, r, a, b, False) for f, r, a, b in best_per_family(runs)]
    have = {c[0] for c in chosen}
    for f, r, a, b in best_per_family([x for x in pool if id(x) not in in_ref]):
        if f not in have:
            chosen.append((f, r, a, b, True))
    chosen.sort(key=lambda t: FAMILY_ORDER.index(t[0])
                if t[0] in FAMILY_ORDER else 99)
    missing = [FAMILY_DISPLAY[f] for f in FAMILY_ORDER
               if f != "other" and f not in {c[0] for c in chosen}]
    return chosen, missing


def find_cases(X, min_pts=40):
    """
    Group test rows by scenario only (ignoring t) — every (x, t) point of one
    shock-tube case, which is what a 2-D space-time map needs.

    Each case also gets a `coverage` score: the fraction of the implied
    (unique x) x (unique t) grid that the test split actually contains.
    tricontourf can only fill the convex hull of the points it is given, so a
    case with coverage well below 1 shows up as white wedges at the corners
    of the panel. Cases are ranked by coverage first, then by size, so the
    default map is the one least likely to have holes in it.
    """
    keys = np.round(X[:, COL_CASE], 5)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    cases = []
    for k in range(len(uniq)):
        idx = np.where(inv == k)[0]
        if len(idx) < min_pts:
            continue
        nx = len(np.unique(np.round(X[idx, COL_X], 5)))
        nt = len(np.unique(np.round(X[idx, COL_T], 5)))
        cov = len(idx) / max(nx * nt, 1)
        cases.append({"key": uniq[k], "idx": idx, "n": len(idx),
                      "coverage": cov, "nx": nx, "nt": nt})
    # round coverage so a marginally denser but much smaller case doesn't win
    cases.sort(key=lambda c: (-round(c["coverage"], 2), -c["n"]))
    return cases


def match_case(run, key, min_pts=40, atol=1e-4):
    """Rows of `run` belonging to scenario `key`, or None."""
    K = np.round(run["X"][:, COL_CASE], 5)
    m = np.all(np.abs(K - key[None, :]) <= atol, axis=1)
    idx = np.where(m)[0]
    return idx if len(idx) >= min_pts else None


def plot_spacetime_comparison(runs, ref_run, out_dir, case_no=0, scale=None,
                              xscale=None, all_runs=None, levels=24,
                              min_pts=40):
    """
    2-D space-time maps, one row per source, three columns (rho, u, p):

        row 1  classical CFD (ground truth)
        row 2  MLP
        row 3  ClassPINN
        row 4  QAPINN — highest R² / lowest test MSE
        row 5  QPINN  — highest R² / lowest test MSE

    Every panel is a tricontourf over the scattered (x, t) test points of one
    scenario, so the wave structure — shock, contact, expansion fan — is
    visible as it develops in time rather than at a single instant.

    Each column shares one colour scale, taken from the CFD row, so a model
    that compresses or shifts the field shows up as a colour mismatch rather
    than being hidden by per-panel autoscaling. Families living in another
    test split are drawn from their own scenario and marked [split N]; their
    row is then indicative, not a like-for-like comparison.
    """
    chosen, missing = select_family_rows(runs, all_runs)
    if not chosen:
        return None
    if missing:
        print(f"  · fig4: no scored runs for {', '.join(missing)} "
              f"— those rows are omitted")

    cases = find_cases(ref_run["X"], min_pts=min_pts)
    if not cases:
        print("  ! no scenario with enough test points — skipping fig4")
        return None
    case = cases[min(case_no, len(cases) - 1)]
    print(f"  · fig4: scenario #{case_no} — {case['n']} points, "
          f"{case['nx']}x{case['nt']} grid, coverage {case['coverage']:.0%}")
    if case["coverage"] < 0.9:
        better = [i for i, c in enumerate(cases) if c["coverage"] >= 0.9]
        print(f"    coverage below 90% — white wedges are regions with no test "
              f"points. {'Try --case ' + str(better[0]) if better else 'No denser scenario available; lower --t-stride in Step2 or raise the test fraction.'}")

    def denorm_y(a):
        return a * scale[1] + scale[0] if scale is not None else a

    def _axes_of(run, idx):
        x = run["X"][idx, COL_X]
        t = run["X"][idx, COL_T]
        if xscale is not None:
            x = x * xscale[1][COL_X] + xscale[0][COL_X]
            t = t * xscale[1][COL_T] + xscale[0][COL_T]
        return x, t

    # row 0 is the CFD truth; the rest are models
    gx, gt = _axes_of(ref_run, case["idx"])
    gt_vals = denorm_y(ref_run["y_true"][case["idx"]])
    rows = [("classical CFD", None, gx, gt, gt_vals, False)]
    for fam, run, r2v, msev, foreign in chosen:
        idx = None if foreign else match_case(run, case["key"], min_pts)
        if idx is None:
            own = find_cases(run["X"], min_pts=min_pts)
            if not own:
                continue
            idx = own[min(case_no, len(own) - 1)]["idx"]
            foreign = True
        rx, rt = _axes_of(run, idx)
        criterion = ("\n[best R² / min MSE]"
                     if fam in ("qapinn", "qpinn") else "")
        lab = (f"{FAMILY_DISPLAY[fam]}\nR²={r2v:.4f}"
               f"\nMSE={msev:.2e}{criterion}"
               + (f"\n[split {run.get('split_group', '?')}]" if foreign else ""))
        rows.append((lab, fam, rx, rt, denorm_y(run["y_pred"][idx]), foreign))

    n_var = gt_vals.shape[1]
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, n_var, figsize=(4.6 * n_var, 2.9 * nrow),
                             squeeze=False, layout="constrained")

    # one colour scale per column, taken from the CFD row
    mappables = []
    for j in range(n_var):
        lo, hi = float(gt_vals[:, j].min()), float(gt_vals[:, j].max())
        if hi <= lo:
            hi = lo + 1e-6
        lv = np.linspace(lo, hi, levels + 1)
        for i, (lab, fam, x, t, v, foreign) in enumerate(rows):
            ax = axes[i][j]
            try:
                cf = ax.tricontourf(x, t, np.clip(v[:, j], lo, hi),
                                    levels=lv, cmap="viridis", extend="neither")
            except Exception as e:
                ax.text(0.5, 0.5, f"cannot triangulate\n({type(e).__name__})",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="0.4")
                continue
            if i == 0:
                mappables.append(cf)
            if i == 0:
                ax.set_title(VAR_LABELS[j] + ("" if scale is not None
                                              else "   (normalised)"),
                             fontsize=11)
            if i == nrow - 1:
                ax.set_xlabel("x" if xscale is not None else "x (normalised)")
            if j == 0:
                ax.set_ylabel(("t" if xscale is not None else "t (norm.)"),
                              fontsize=9)
                colour = (FAMILY_CMAP.get(fam, plt.cm.Greys)(0.95)
                          if fam else "k")
                ax.text(-0.30, 0.5, lab, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=8.5, color=colour)

    for j, cf in enumerate(mappables):
        cb = fig.colorbar(cf, ax=[axes[i][j] for i in range(nrow)],
                          location="right", shrink=0.85, pad=0.01)
        cb.ax.tick_params(labelsize=7)

    note = "  ·  rows marked [split N] use a different test set" \
        if any(r[5] for r in rows) else ""
    fig.suptitle("2-D space-time fields: classical CFD vs model predictions"
                 f"\nscenario #{case_no}, {case['n']} test points "
                 f"({case['coverage']:.0%} of the {case['nx']}x{case['nt']} "
                 f"grid); "
                 f"colour scale per column fixed to the CFD row{note}",
                 fontsize=12)

    path = os.path.join(out_dir, "fig4_spacetime_comparison.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ figure → {path}")
    return path


def plot_family_rows(runs, ref_run, out_dir, slice_no=0, scale=None,
                     xscale=None, min_pts=12, all_runs=None):
    """
    One row per model family, three columns (rho, u, p).

      row 1  MLP        — classical baseline
      row 2  ClassPINN  — classical PINN
      row 3  QAPINN     — best hybrid configuration
      row 4  QPINN      — best pure-quantum configuration

    Each panel shows the classical CFD solution (black dashed) against that
    model's prediction, so every architecture gets an uncluttered comparison.

    A family with no run in the reference test split still gets its row: the
    best run of that family is taken from whatever split it lives in, and the
    row is drawn against *its own* ground truth and x-grid, with the banner
    marked "split N". That keeps QAPINN/QPINN rows present even when they were
    trained with different N_SCENARIOS / T_STRIDE than the baselines — but
    rows from different splits are not point-for-point comparable, so treat
    them as separate experiments rather than a controlled comparison.
    """
    chosen, missing = select_family_rows(runs, all_runs)
    if not chosen:
        return None
    if missing:
        print(f"  · fig4b: no scored runs for {', '.join(missing)} "
              f"— those rows are omitted")

    def denorm_y(a):
        return a * scale[1] + scale[0] if scale is not None else a

    def _slice_of(run, prefer_key):
        """Rows of `run` for the requested slice, or its own equivalent."""
        if prefer_key is not None:
            idx = match_slice(run, prefer_key)
            if idx is not None:
                return idx
        own = find_slices(run["X"], min_pts=min_pts)
        if not own:
            return None
        return own[min(slice_no, len(own) - 1)]["idx"]

    ref_slices = find_slices(ref_run["X"], min_pts=min_pts)
    if not ref_slices:
        print("  ! no test slice with enough points — skipping fig4")
        return None
    ref_sl = ref_slices[min(slice_no, len(ref_slices) - 1)]
    tlab_t = ref_sl["key"][0]

    # gather each row's own x, ground truth and prediction
    panel = []
    for fam, run, r2v, msev, foreign in chosen:
        idx = _slice_of(run, None if foreign else ref_sl["key"])
        if idx is None:
            panel.append((fam, run, r2v, msev, foreign, None, None, None))
            continue
        rx = run["X"][idx, COL_X]
        rx = rx * xscale[1][COL_X] + xscale[0][COL_X] if xscale else rx
        panel.append((fam, run, r2v, msev, foreign, rx,
                      denorm_y(run["y_true"][idx]),
                      denorm_y(run["y_pred"][idx])))

    n_var = max(g.shape[1] for *_, g, _ in panel if g is not None)
    nrow = len(panel)
    fig, axes = plt.subplots(nrow, n_var, figsize=(4.9 * n_var, 3.0 * nrow),
                             squeeze=False, sharex=True)

    ylims = []
    for j in range(n_var):
        vals = [a[:, j] for *_, g, pr in panel for a in (g, pr)
                if a is not None]
        lo = min(float(v.min()) for v in vals)
        hi = max(float(v.max()) for v in vals)
        pad = 0.08 * max(hi - lo, 1e-6)
        ylims.append((lo - pad, hi + pad))

    xlab = "x" if xscale is not None else "x (normalised)"
    for i, (fam, run, r2v, msev, foreign, rx, gt, yp) in enumerate(panel):
        color = FAMILY_CMAP.get(fam, plt.cm.Greys)(0.75)
        for j in range(n_var):
            ax = axes[i][j]
            if gt is None:
                ax.text(0.5, 0.5, "no usable slice in this run's test set",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="0.4")
            else:
                ax.plot(rx, gt[:, j], "k--", lw=2.2, zorder=10,
                        label="classical CFD" if (i == 0 and j == 0) else None)
                ax.plot(rx, yp[:, j], lw=1.6, color=color, marker="o", ms=2.6,
                        alpha=0.95,
                        label=FAMILY_DISPLAY[fam] if (i == 0 and j == 0) else None)
            ax.set_ylim(*ylims[j])
            ax.grid(alpha=0.3, ls=":")
            if i == 0:
                ax.set_title(VAR_LABELS[j] + ("" if scale is not None
                                              else "   (normalised)"),
                             fontsize=11)
            if i == nrow - 1:
                ax.set_xlabel(xlab)

        tag = f"{FAMILY_DISPLAY[fam]}\n{run['tag']}\nR²={r2v:.4f}  MSE={msev:.2e}"
        if foreign:
            tag += f"\n[split {run.get('split_group', '?')}, own slice]"
        axes[i][0].text(-0.16, 0.5, tag,
                        transform=axes[i][0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=8.5,
                        color=FAMILY_CMAP.get(fam, plt.cm.Greys)(0.95))

    h, _ = axes[0][0].get_legend_handles_labels()
    if h:
        axes[0][0].legend(h, ["classical CFD (ground truth)", "model"],
                          fontsize=8, frameon=True, framealpha=0.85, loc="best")
    note = "  ·  rows marked [split N] use a different test set" \
        if any(p[4] for p in panel) else ""
    fig.suptitle("Classical simulation vs best model of each family  —  "
                 f"slice #{slice_no}  (t = {tlab_t:.3g} norm.){note}",
                 fontsize=13)
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))

    path = os.path.join(out_dir, "fig4b_family_rows.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ figure → {path}")
    return path


def plot_family_best(runs, out_dir):
    """
    Head-to-head bar chart: best run of each family (MLP / ClassPINN /
    QPINN / QAPINN) side by side on R² and MSE. This is the "did the hybrid
    actually beat the baselines" plot.
    """
    rows = best_per_family(runs)
    if len(rows) < 2:
        return None

    fams  = [f for f, *_ in rows]
    names = [FAMILY_DISPLAY[f] for f in fams]
    r2s   = [r2v for _, _, r2v, _ in rows]
    mses  = [msev for *_, msev in rows]
    cols  = [FAMILY_CMAP.get(f, plt.cm.Greys)(0.7) for f in fams]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(names, r2s, color=cols)
    axes[0].set_ylabel("R²  (test, axis zoomed)")
    axes[0].set_title("Best run per family — R²")
    axes[0].set_ylim(*_r2_limits(r2s))
    for i, v in enumerate(r2s):
        axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(names, mses, color=cols)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("MSE  (normalised, log scale)")
    axes[1].set_title("Best run per family — MSE")
    for i, v in enumerate(mses):
        axes[1].text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.grid(alpha=0.3, axis="y", ls=":")
    sub = "  |  ".join(f"{FAMILY_DISPLAY[f]}: {run['tag']}"
                       for f, run, _, _ in rows)
    fig.suptitle("Classical vs quantum vs hybrid — best configuration of each\n"
                 + sub, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    path = os.path.join(out_dir, "fig6_family_best_bars.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ figure → {path}")
    return path


def _r2_limits(vals, upper=1.0):
    """Zoomed limits for R² bars — everything clusters near 1.0 otherwise."""
    lo, hi = float(min(vals)), float(max(vals))
    span = max(hi - lo, 1e-4)
    return lo - 0.35 * span, min(upper, hi + 0.25 * span)


def plot_ablation(runs, family, out_dir, hps=("n_layers", "ansatz",
                                              "encoding", "n_qubits"),
                  smooth=1):
    """
    One row per hyper-parameter, three columns:

        row 1  n_layers   (QVC ansatz depth)      col 1  training loss vs epoch
        row 2  ansatz                             col 2  data MSE vs epoch
        row 3  encoding                           col 3  final test R² per value
        row 4  n_qubits

    Within a row only the row's hyper-parameter varies — the others are held
    at the values of the family's best run, so each row is a clean ablation.
    If that leaves fewer than two runs, the row falls back to the best run
    per value (other hyper-parameters then differ, and the title says so).

    Column 3 is a bar chart rather than a curve because Step2's training loop
    records loss_history and mse_history only — there is no per-epoch R² to
    plot. Logging one in train_qapinn would turn this column into a curve.
    """
    fam_runs = [r for r in runs if r.get("family") == family]
    if len(fam_runs) < 2:
        return None
    for r in fam_runs:
        r["_r2"] = r2_score(r["y_true"], r["y_pred"])
    ref = max(fam_runs, key=lambda r: r["_r2"])

    def _best_per_value(cands, hp):
        by_val = {}
        for r in cands:
            v = _hp(r, hp)
            if v is None:
                continue
            if v not in by_val or r["_r2"] > by_val[v]["_r2"]:
                by_val[v] = r
        return by_val

    # build the rows that actually have something to compare
    plan = []
    for hp in hps:
        others = [k for k in hps if k != hp]
        strict = [r for r in fam_runs
                  if all(_hp(r, k) == _hp(ref, k) for k in others)]
        by_val = _best_per_value(strict, hp)
        clean = True
        if len(by_val) < 2:
            by_val = _best_per_value(fam_runs, hp)
            clean = False
        if len(by_val) < 2:
            continue
        plan.append((hp, by_val, clean))

    if not plan:
        print(f"  ! no {FAMILY_DISPLAY[family]} hyper-parameter varies — "
              f"skipping ablation figure")
        return None

    def _smooth(v):
        v = np.asarray(v, dtype=float)
        if smooth > 1 and len(v) > smooth:
            return np.convolve(v, np.ones(smooth) / smooth, mode="valid")
        return v

    base = FAMILY_CMAP.get(family, plt.cm.Greys)
    nrow = len(plan)
    fig, axes = plt.subplots(nrow, 3, figsize=(15.5, 3.3 * nrow), squeeze=False)

    for i, (hp, by_val, clean) in enumerate(plan):
        vals = sorted(by_val, key=lambda v: (str(type(v)), v))
        styles = [distinct_style(k) for k in range(len(vals))]
        cols = [c for c, _ls, _m in styles]

        for k, v in enumerate(vals):
            r = by_val[v]
            colour, ls, mk = styles[k]
            lab = f"{hp}={v}"
            for panel, key in ((0, "loss_history"), (1, "mse_history")):
                if len(r[key]) < 2:
                    continue
                y = _smooth(r[key])
                ep = np.arange(1, len(y) + 1)
                # markers on a few points only — enough to separate curves that
                # overlap without turning the line into a dotted mess
                every = max(len(y) // 8, 1)
                axes[i][panel].plot(ep, y, lw=1.8, color=colour, ls=ls,
                                    marker=mk, markevery=every, ms=4.5,
                                    markerfacecolor="none", label=lab)

        axes[i][2].bar([str(v) for v in vals],
                       [by_val[v]["_r2"] for v in vals], color=cols,
                       edgecolor="k", linewidth=0.5)
        for k, v in enumerate(vals):
            axes[i][2].text(k, by_val[v]["_r2"], f"{by_val[v]['_r2']:.4f}",
                            ha="center", va="bottom", fontsize=8)
        axes[i][2].set_ylim(*_r2_limits([by_val[v]["_r2"] for v in vals]))

        for j, (ttl, ylab) in enumerate((
                ("training loss", "loss (total objective)"),
                ("data MSE", "MSE (normalised)"),
                ("final test R²", "R² (axis zoomed)"))):
            ax = axes[i][j]
            ax.grid(alpha=0.3, ls=":", which="both" if j < 2 else "major",
                    axis="both" if j < 2 else "y")
            ax.set_ylabel(ylab, fontsize=9)
            if j < 2:
                ax.set_yscale("log")
                ax.set_xlabel("epoch", fontsize=9)
                ax.legend(fontsize=7.5, frameon=False)
            else:
                ax.set_xlabel(hp, fontsize=9)
            if i == 0:
                ax.set_title(ttl, fontsize=12)

        note = "" if clean else "  (other hyper-params vary)"
        axes[i][0].text(-0.19, 0.5, f"vary {hp}{note}",
                        transform=axes[i][0].transAxes, rotation=90,
                        ha="center", va="center", fontsize=9.5,
                        color=base(0.95))

    fig.suptitle(f"{FAMILY_DISPLAY[family]} ablation — reference config "
                 f"{ref['tag']}  (R²={ref['_r2']:.4f})", fontsize=13)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))

    path = os.path.join(out_dir, f"fig7_ablation_{family}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ figure → {path}")
    return path


def _hp(run, key):
    """Hyper-parameter from config, falling back to parsing the dir tag."""
    v = run.get("config", {}).get(key)
    if v is not None:
        return v
    pat = {"n_qubits": r"q(\d+)", "n_layers": r"_l(\d+)",
           "n_enc_layers": r"_e(\d+)", "q_layer_idx": r"qli(\d+)"}.get(key)
    if pat:
        m = re.search(pat, run.get("tag", ""))
        if m:
            return int(m.group(1))
    return None


def _reference_run(runs, all_runs=None):
    """
    The classical PINN to benchmark a sweep against — best ClassPINN run,
    falling back to the best MLP if no PINN was trained. Returns
    (run, r2, label) or (None, None, "").
    """
    for pool, foreign in ((runs, False), (all_runs or [], True)):
        for fam in ("classpinn", "mlp"):
            cands = [r for r in pool if r.get("family") == fam
                     and "y_true" in r]
            if cands:
                best = max(cands, key=lambda r: r2_score(r["y_true"], r["y_pred"]))
                lab = FAMILY_DISPLAY[fam] + (" (other split)" if foreign else "")
                return best, r2_score(best["y_true"], best["y_pred"]), lab
    return None, None, ""


def _family_reference(runs, all_runs, family):
    """Best baseline of one family, preferring the sweep's test split."""
    for pool, foreign in ((runs, False), (all_runs or [], True)):
        cands = [r for r in pool if r.get("family") == family
                 and "y_true" in r]
        if cands:
            best = max(cands,
                       key=lambda r: r2_score(r["y_true"], r["y_pred"]))
            label = FAMILY_DISPLAY[family]
            if foreign:
                label += " (other split)"
            return best, r2_score(best["y_true"], best["y_pred"]), label
    return None, None, ""


def plot_sweep_summary(runs, family, out_dir, all_runs=None):
    """
    Sweep view for one family (QAPINN or QPINN):
      panel 1 — R² per configuration, ranked, with Classical PINN and MLP
                reference lines (black dashed and blue dotted)
      panel 2 — encoding × ansatz heatmap, coloured relative to that same
                PINN reference (green = beats it, red = worse); cells that
                beat the reference are marked with ▲
      panel 3 — R² vs the numeric hyper-parameter that varies most
                (n_qubits, q_layer_idx, ...), one line per second hp

    The reference is the best ClassPINN run, or the best MLP if no PINN was
    trained. If it comes from a different test split that is said in the
    legend — the comparison is then indicative, not like-for-like.
    """
    fam_runs = [r for r in runs if r.get("family") == family]
    if len(fam_runs) < 2:
        return None
    ref_run, ref_r2, ref_lab = _reference_run(runs, all_runs)
    mlp_run, mlp_r2, mlp_lab = _family_reference(
        runs, all_runs, "mlp")

    for r in fam_runs:
        r["_r2"] = r2_score(r["y_true"], r["y_pred"])
    fam_runs.sort(key=lambda r: -r["_r2"])

    encs = sorted({str(_hp(r, "encoding")) for r in fam_runs} - {"None"})
    anss = sorted({str(_hp(r, "ansatz")) for r in fam_runs} - {"None"})
    has_grid = len(encs) > 1 or len(anss) > 1

    # numeric hyper-parameters that actually vary
    varying = [k for k in NUMERIC_HP
               if len({_hp(r, k) for r in fam_runs if _hp(r, k) is not None}) > 1]

    npanel = 1 + int(has_grid) + int(bool(varying))
    fig, axes = plt.subplots(1, npanel, figsize=(5.6 * npanel, 4.8),
                             squeeze=False)
    axes = axes[0]
    base = FAMILY_CMAP.get(family, plt.cm.Greys)
    k = 0

    # ── panel 1: ranked bars ─────────────────────────────────────────────────
    show = fam_runs[:20]
    ypos = np.arange(len(show))[::-1]
    # A broad categorical-looking spectrum makes neighbouring ranked
    # configurations easy to distinguish, even when their R² values are very
    # close.  Keep the best run at the warm/red end of the colour sequence.
    ranked_colours = plt.cm.turbo(np.linspace(0.92, 0.08, len(show)))
    axes[k].barh(ypos, [r["_r2"] for r in show],
                 color=ranked_colours, edgecolor="white", linewidth=0.35)
    axes[k].set_yticks(ypos)
    axes[k].set_yticklabels([r["tag"] for r in show], fontsize=7)
    axes[k].set_xlabel("R²  (test)")
    axes[k].set_title(f"{FAMILY_DISPLAY[family]} configurations, ranked"
                      + (f"  (top 20 of {len(fam_runs)})"
                         if len(fam_runs) > 20 else ""))
    sweep_values = np.asarray([r["_r2"] for r in show], dtype=float)
    sweep_lo, sweep_hi = float(sweep_values.min()), float(sweep_values.max())
    sweep_span = max(sweep_hi - sweep_lo, 1e-4)
    # A very distant reference (for example a newly overwritten/under-trained
    # ClassPINN) must not destroy the useful zoom among sweep configurations.
    # Draw the line only when it is reasonably close to the sweep range;
    # otherwise retain the value as an annotation below the legend.
    def _near_sweep(value):
        return (value is not None
                and sweep_lo - 2.0 * sweep_span <= value
                <= sweep_hi + 2.0 * sweep_span)

    ref_near_sweep = _near_sweep(ref_r2)
    mlp_near_sweep = _near_sweep(mlp_r2)
    distant_notes = []
    if ref_near_sweep:
        axes[k].axvline(ref_r2, color="k", ls="--", lw=1.6, zorder=5,
                        label=f"{ref_lab} reference  R²={ref_r2:.4f}")
    elif ref_r2 is not None:
        relation = "below" if ref_r2 < sweep_lo else "above"
        distant_notes.append(
            f"{ref_lab} R²={ref_r2:.4f} ({relation} range)")
    if mlp_near_sweep:
        axes[k].axvline(mlp_r2, color="#1f77b4", ls=":", lw=2.1,
                        zorder=5,
                        label=f"{mlp_lab} reference  R²={mlp_r2:.4f}")
    elif mlp_r2 is not None:
        relation = "below" if mlp_r2 < sweep_lo else "above"
        distant_notes.append(
            f"{mlp_lab} R²={mlp_r2:.4f} ({relation} range)")
    if ref_near_sweep or mlp_near_sweep:
        axes[k].legend(fontsize=7.3, frameon=False, loc="lower right")
    if distant_notes:
        axes[k].text(
            0.99, 0.025, "\n".join(distant_notes),
            transform=axes[k].transAxes, ha="right", va="bottom",
            fontsize=7.5, color="0.20",
            bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.9,
                      boxstyle="round,pad=0.25"),
        )
    lims = (list(sweep_values)
            + ([ref_r2] if ref_near_sweep else [])
            + ([mlp_r2] if mlp_near_sweep else []))
    axes[k].set_xlim(*_r2_limits(lims))
    axes[k].set_xlabel("R²  (test, axis zoomed)")
    axes[k].grid(alpha=0.3, axis="x", ls=":")
    k += 1

    # ── panel 2: encoding × ansatz heatmap ───────────────────────────────────
    if has_grid:
        M = np.full((len(encs), len(anss)), np.nan)
        for r in fam_runs:
            e, a = str(_hp(r, "encoding")), str(_hp(r, "ansatz"))
            if e in encs and a in anss:
                i, j = encs.index(e), anss.index(a)
                M[i, j] = r["_r2"] if np.isnan(M[i, j]) else max(M[i, j], r["_r2"])
        finite_M = M[np.isfinite(M)]
        grid_lo = float(finite_M.min()) if finite_M.size else np.nan
        grid_hi = float(finite_M.max()) if finite_M.size else np.nan
        # TwoSlopeNorm is informative only when the reference lies inside the
        # sweep range.  If it lies far outside, centring on it compresses every
        # cell into one saturated colour and hides all sweep differences.
        ref_inside_grid = (ref_r2 is not None and finite_M.size
                           and grid_lo < ref_r2 < grid_hi)
        if ref_inside_grid:
            lo = grid_lo - 1e-6
            hi = grid_hi + 1e-6
            norm = matplotlib.colors.TwoSlopeNorm(vmin=lo, vcenter=ref_r2,
                                                  vmax=hi)
            im = axes[k].imshow(M, cmap="RdYlGn", norm=norm, aspect="auto")
        else:
            # Turbo provides much more visible separation for the very narrow
            # R² range typical of a converged QAPINN sweep.
            pad = max((grid_hi - grid_lo) * 0.03, 1e-6) \
                if finite_M.size else 1e-6
            im = axes[k].imshow(M, cmap="turbo", aspect="auto",
                                vmin=grid_lo - pad if finite_M.size else None,
                                vmax=grid_hi + pad if finite_M.size else None)
        axes[k].set_xticks(range(len(anss)))
        axes[k].set_xticklabels(anss, rotation=30, ha="right", fontsize=8)
        axes[k].set_yticks(range(len(encs)))
        axes[k].set_yticklabels(encs, fontsize=8)
        for i in range(len(encs)):
            for j in range(len(anss)):
                if np.isfinite(M[i, j]):
                    beats = ref_r2 is not None and M[i, j] > ref_r2
                    axes[k].text(j, i,
                                 f"{M[i, j]:.3f}" + (" ▲" if beats else ""),
                                 ha="center", va="center", fontsize=7,
                                 fontweight="bold" if beats else "normal",
                                 color="k" if ref_r2 is not None else
                                 ("w" if M[i, j] < np.nanmax(M) * 0.8 else "k"))
        if ref_inside_grid:
            heat_note = (f"\n(colour centred on {ref_lab} R²={ref_r2:.4f};"
                         " ▲ beats it)")
        elif ref_r2 is not None:
            relation = "below" if ref_r2 <= grid_lo else "above"
            heat_note = (f"\n(colour scaled to sweep; {ref_lab} reference "
                         f"R²={ref_r2:.4f} is {relation} range; ▲ beats it)")
        else:
            heat_note = "\n(colour scaled to sweep range)"
        axes[k].set_title("best R² — encoding × ansatz" + heat_note,
                          fontsize=10)
        axes[k].set_xlabel("ansatz")
        axes[k].set_ylabel("encoding")
        cb = fig.colorbar(im, ax=axes[k], fraction=0.046)
        if ref_inside_grid:
            cb.ax.axhline(ref_r2, color="k", lw=1.4)
        k += 1

    # ── panel 3: R² vs numeric hyper-parameter ───────────────────────────────
    if varying:
        xk = varying[0]
        gk = varying[1] if len(varying) > 1 else None
        line_colours = plt.cm.tab10.colors
        groups = {}
        for r in fam_runs:
            xv = _hp(r, xk)
            if xv is None:
                continue
            groups.setdefault(_hp(r, gk) if gk else None, []).append((xv, r["_r2"]))
        for i, (gv, pts) in enumerate(sorted(groups.items(),
                                             key=lambda kv: str(kv[0]))):
            # several configs can share an x (e.g. two encodings at the same
            # n_qubits) — collapse to the best R² so the line stays monotone
            agg = {}
            for xv, r2v in pts:
                agg[xv] = max(agg.get(xv, -np.inf), r2v)
            xs = sorted(agg)
            ys = [agg[x] for x in xs]
            axes[k].plot(xs, ys, marker="o", lw=1.6,
                         markersize=6,
                         color=line_colours[i % len(line_colours)],
                         label=f"{gk}={gv}" if gk else None)
        axes[k].set_xlabel(xk)
        axes[k].set_ylabel("R²  (test)")
        axes[k].set_title(f"R² vs {xk}" + (f"  (by {gk})" if gk else ""))
        axes[k].grid(alpha=0.3, ls=":")
        if gk:
            axes[k].legend(fontsize=8, frameon=False)

    fig.suptitle(f"{FAMILY_DISPLAY[family]} sweep — {len(fam_runs)} runs",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    path = os.path.join(out_dir, f"fig5_sweep_{family}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ figure → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — OPTIONAL: NORMALISATION STATS FOR PHYSICAL UNITS
# ══════════════════════════════════════════════════════════════════════════════

def load_scale(data_dir, n_scenarios, t_stride, repo_dir=None):
    """
    Re-derive (Xm, Xs, Ym, Ys) by calling utilities_classical.load_ns_data().
    Returns (xscale, yscale) as numpy tuples, or (None, None) on failure.
    """
    if not data_dir:
        return None, None
    if repo_dir:
        sys.path.insert(0, repo_dir)
    try:
        from utilities_classical import load_ns_data
        d = load_ns_data(data_dir, n_scenarios=n_scenarios, t_stride=t_stride)
        to_np = lambda v: np.asarray(v.detach().cpu().numpy()
                                     if hasattr(v, "detach") else v,
                                     dtype=np.float64).ravel()
        return (to_np(d["Xm"]), to_np(d["Xs"])), (to_np(d["Ym"]), to_np(d["Ys"]))
    except Exception as e:
        print(f"  ! could not load normalisation stats ({type(e).__name__}: {e})")
        print("    → plotting/scoring in normalised units instead")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="./results",
                    help="OUT_DIR from Step2 (searched recursively)")
    ap.add_argument("--out-dir", default=None,
                    help="where figures/tables go (default: <results-dir>/figures)")
    ap.add_argument("--data-dir", default=None,
                    help="raw NS data dir — enables physical units")
    ap.add_argument("--repo-dir", default=None,
                    help="dir containing utilities_classical.py (for --data-dir)")
    ap.add_argument("--n-scenarios", type=int, default=30)
    ap.add_argument("--t-stride", type=int, default=6)
    ap.add_argument("--include", action="append", default=[],
                    help="keep runs whose label contains this (repeatable)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="drop runs whose label contains this (repeatable)")
    ap.add_argument("--top-k", type=int, default=8,
                    help="limit field/parity plots to the K best runs by R² "
                         "(0 = all). The table always keeps every run.")
    ap.add_argument("--group", type=int, default=0,
                    help="which test-split group to plot when runs were "
                         "trained on different splits (0 = largest group)")
    ap.add_argument("--slice", type=int, default=0,
                    help="which test slice to plot in fig2 (0 = largest)")
    ap.add_argument("--list-slices", action="store_true",
                    help="list available test slices and exit")
    ap.add_argument("--smooth", type=int, default=1,
                    help="moving-average window for the loss curves")
    ap.add_argument("--no-family-guarantee", action="store_true",
                    help="do not force the best MLP/ClassPINN/QPINN/QAPINN "
                         "run into the field plots")
    ap.add_argument("--case", type=int, default=0,
                    help="which scenario the space-time map (fig4) shows "
                         "(0 = the one with most test points)")
    ap.add_argument("--select", choices=["r2", "loss", "both"], default="both",
                   help="fig1: pick each family's run by best test R², by "
                         "lowest test data MSE, or show both for the quantum "
                         "families when they differ (default)")
    ap.add_argument("--list-cases", action="store_true",
                    help="list the space-time scenarios with their grid "
                         "coverage and exit")
    ap.add_argument("--curves", choices=["family", "all"], default="family",
                    help="fig1 shows the best run of each family (default) "
                         "or every run")
    ap.add_argument("--ablation", default="qapinn",
                    help="comma-separated families to build an ablation grid "
                         "for, e.g. 'qapinn,qpinn' or '' to skip")
    ap.add_argument("--ablation-hps",
                    default="n_layers,ansatz,encoding,n_qubits",
                    help="one row per hyper-parameter, in this order")
    ap.add_argument("--no-parity", action="store_true")
    args = ap.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(results_dir, "figures"))
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("  NS shock tube — comparison plots & metrics")
    print("=" * 72)
    print(f"  results : {results_dir}")
    print(f"  output  : {out_dir}")

    # ── discover ──────────────────────────────────────────────────────────────
    runs = filter_runs(discover_runs(results_dir), args.include, args.exclude)
    print(f"\n  found {len(runs)} run(s):")
    for r in runs:
        has_p = "pred" if r["npz_path"] else "  — "
        print(f"    [{has_p}] {r['label']:<50} "
              f"epochs={len(r['loss_history']):>4}")

    # ── normalisation stats (optional) ────────────────────────────────────────
    xscale, yscale = load_scale(args.data_dir, args.n_scenarios,
                                args.t_stride, args.repo_dir)

    # ── load predictions ──────────────────────────────────────────────────────
    scored = [r for r in runs if load_predictions(r)]
    if not scored:
        print(f"\n{'─'*72}\n  [1/3] training curves")
        plot_training_curves(runs, out_dir, smooth=args.smooth,
                             subtitle="all runs")
        print("\n  ! no *_predictions.npz found — figures 2/3 and the table "
              "need them. Re-run Step2 (save_results writes them).")
        return

    # ── group by test split ───────────────────────────────────────────────────
    groups = group_runs_by_split(scored)
    if len(groups) > 1:
        print(f"\n  ! runs do NOT all share one test split — {len(groups)} "
              f"distinct test sets found:")
        for gid, (sig, members) in enumerate(groups):
            print(f"    split {gid}: n_test={sig.split(':')[0]:<12} "
                  f"{len(members)} run(s)  e.g. {members[0]['label']}")
        print("    Metrics across different splits are not directly "
              "comparable; fig2/fig3 use one split at a time (--group N).")
    gid = min(args.group, len(groups) - 1)
    group_runs = groups[gid][1]
    ref = group_runs[0]

    if args.list_cases:
        print(f"\n  space-time scenarios in {ref['label']} "
              f"(ranked by grid coverage):")
        for i, c in enumerate(find_cases(ref["X"])[:40]):
            print(f"    #{i:<3} n={c['n']:<6} grid={c['nx']}x{c['nt']:<5} "
                  f"coverage={c['coverage']:.0%}  key={np.round(c['key'], 3)}")
        return

    if args.list_slices:
        print(f"\n  test slices in {ref['label']} "
              f"(key = [t, p_ratio, mu, rho_L, rho_R, p_R], normalised):")
        for i, s in enumerate(find_slices(ref["X"])[:40]):
            print(f"    #{i:<3} n={s['n']:<5} key={np.round(s['key'], 3)}")
        return

    # ── TABLE ─────────────────────────────────────────────────────────────────
    rows = [compute_metrics(r, scale=yscale) for r in scored]
    rows.sort(key=lambda r: (-r["r2"] if np.isfinite(r["r2"]) else 1e9))
    print(f"\n{'─'*72}\n  [2/3] metrics")
    print_table(rows, physical=yscale is not None)
    save_table(rows, out_dir, physical=yscale is not None)

    # ── FIGURES 2 & 3 — best-K runs only, so the plot stays readable ──────────
    order = {r["model"]: i for i, r in enumerate(rows)}
    group_runs.sort(key=lambda r: order.get(r["label"], 1e9))
    ref = group_runs[0]
    plot_runs = select_for_plot(group_runs, args.top_k,
                                family_guarantee=not args.no_family_guarantee)
    if len(plot_runs) < len(scored):
        fams = sorted({r["family"] for r in plot_runs})
        print(f"\n  (fig2/fig3 show {len(plot_runs)} of {len(scored)} runs — "
              f"split {gid}, best per family + top R² [{', '.join(fams)}]; "
              f"--top-k 0 for all)")

    print(f"\n{'─'*72}\n  [1/3] training curves")
    if args.curves == "all":
        plot_training_curves(scored, out_dir, smooth=args.smooth,
                             subtitle=f"all {len(scored)} runs")
    else:
        fam_best = select_curve_runs(group_runs, scored, mode=args.select)
        no_hist = [r["label"] for r in fam_best if len(r["loss_history"]) < 2]
        if no_hist:
            print(f"  · fig1: no loss_history recorded for "
                  f"{', '.join(no_hist)} — curve omitted")
        for r in fam_best:
            print(f"  · fig1: {r['label']:<52} R²="
                  f"{r2_score(r['y_true'], r['y_pred']):.4f}  "
                  f"final loss={_final_loss(r):.4g}")
        sub = {"r2": "best R² configuration of each family",
               "loss": "lowest test-MSE configuration of each family",
               "both": "best R² / lowest test MSE per family"}[args.select]
        sub += "; QAPINN shows the selection for n_layers=2, 3 and 4"
        sub += "  (--curves all for every run)"
        if any(r.get("_foreign") for r in fam_best):
            sub += "  ·  some families come from a different test split"
        plot_training_curves(fam_best, out_dir, smooth=args.smooth,
                             subtitle=sub)

    print(f"\n{'─'*72}\n  [3/3] field comparison")
    plot_field_comparison(plot_runs, ref, out_dir, slice_no=args.slice,
                          scale=yscale, xscale=xscale)
    if not args.no_parity:
        plot_parity(plot_runs, out_dir)

    # ── family + sweep views ─────────────────────────────────────────────────
    plot_spacetime_comparison(group_runs, ref, out_dir, case_no=args.case,
                              scale=yscale, xscale=xscale, all_runs=scored)
    plot_family_rows(group_runs, ref, out_dir, slice_no=args.slice,
                     scale=yscale, xscale=xscale, all_runs=scored)
    plot_family_best(group_runs, out_dir)
    for fam in ("qapinn", "qpinn"):
        plot_sweep_summary(group_runs, fam, out_dir, all_runs=scored)
    for fam in args.ablation.split(","):
        fam = fam.strip().lower()
        if fam:
            plot_ablation(group_runs, fam, out_dir,
                          hps=tuple(h.strip() for h in args.ablation_hps.split(",")),
                          smooth=args.smooth)

    print(f"\n  ✓ done → {out_dir}")


if __name__ == "__main__":
    main()
