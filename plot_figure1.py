"""
plot_figure1.py
===============
Plot ρ, u, p, T outputs from a Navier-Stokes shock tube scenario .npz file.

For each field, produces two panels:
  Left  — profiles vs x at 7 evenly-spaced time snapshots
  Right — space-time (t–x) heatmap with shock and contact-discontinuity trajectories

Usage
-----
    python plot_figure1.py                                  # default: scenario_0001.npz
    python plot_figure1.py --file data/scenario_0042.npz   # any scenario file
    python plot_figure1.py --file data/scenario_0042.npz --out my_plot.png
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm


# ── colour constants ──────────────────────────────────────────────────────────
SHOCK_COL = '#d62728'   # red   — shock front
CD_COL    = '#1f77b4'   # blue  — contact discontinuity
DIAP_COL  = '#aaaaaa'   # grey  — diaphragm position x = 0.5


def compute_wave_trajectories(p_field, rho_field, x):
    """
    Track shock and contact-discontinuity positions at every time step.

    Shock position   : location of maximum |dp/dx|  (largest pressure gradient)
    Contact disc.    : location of maximum |dρ/dx|  left of the shock
                       (density jump where pressure is continuous)

    Returns
    -------
    shock_traj : list of float, length n_snaps
    cd_traj    : list of float, length n_snaps  (nan when not detectable)
    """
    shock_traj, cd_traj = [], []
    for ti in range(p_field.shape[0]):
        dp   = np.abs(np.gradient(p_field[ti],   x))
        drho = np.abs(np.gradient(rho_field[ti], x))
        sp   = float(x[np.argmax(dp)])
        mask = x < sp - 0.05          # look for CD strictly left of shock
        cp   = float(x[mask][np.argmax(drho[mask])]) if mask.sum() > 1 else float('nan')
        shock_traj.append(sp)
        cd_traj.append(cp)
    return shock_traj, cd_traj


def style_ax(ax):
    """Apply consistent white-background axes style."""
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#cccccc')
        sp.set_linewidth(0.8)
    ax.tick_params(colors='#444444', labelsize=20)


def plot_scenario(npz_path, out_path):
    """
    Load one scenario .npz file and produce the 8-panel figure.

    Parameters
    ----------
    npz_path : str   path to the .npz file
    out_path : str   path for the saved PNG
    """
    # ── load data ─────────────────────────────────────────────────────────────
    d = np.load(npz_path)

    x   = d['x']        # (N,)
    t   = d['t_snaps']  # (n_snaps,)
    rho = d['rho']      # (n_snaps, N)
    u   = d['u']        # (n_snaps, N)
    p   = d['p']        # (n_snaps, N)
    T   = d['T']        # (n_snaps, N)

    p_ratio = float(d['p_ratio'])
    mu      = float(d['mu'])
    rho_L   = float(d['rho_L'])
    rho_R   = float(d['rho_R'])
    p_R_val = float(d['p_R'])
    shock_x = float(d['shock_pos'])   # final shock position (diagnostic)
    cd_x    = float(d['cd_pos'])      # final contact-disc. position

    # ── meshgrid for heatmaps: rows = t, cols = x ────────────────────────────
    T_grid, X_grid = np.meshgrid(t, x, indexing='ij')   # both (n_snaps, N)

    # ── 7 evenly-spaced snapshot indices for line plots ───────────────────────
    snap_idx   = np.linspace(0, len(t) - 1, 7, dtype=int)
    cmap_lines = plt.cm.plasma(np.linspace(0.1, 0.9, len(snap_idx)))

    # ── wave-front trajectories ───────────────────────────────────────────────
    shock_traj, cd_traj = compute_wave_trajectories(p, rho, x)

    # ── field definitions ─────────────────────────────────────────────────────
    fields = [
        ('rho', r'Density  $\rho$',     'plasma',  None),    # None = sequential
        ('u',   r'Velocity  $u$',       'RdBu_r',  'div'),   # div  = diverging
        ('p',   r'Pressure  $p$',       'inferno', None),
        ('T',   r'Temperature  $T$',    'hot',     None),
    ]
    field_data = {'rho': rho, 'u': u, 'p': p, 'T': T}

    # ── figure layout: 4 rows × 2 cols ───────────────────────────────────────
    fig = plt.figure(figsize=(18, 22), facecolor='white')
    gs  = gridspec.GridSpec(
        4, 2, figure=fig,
        left=0.07, right=0.97,
        top=0.91,  bottom=0.04,
        wspace=0.55, hspace=0.38,
        width_ratios=[1, 1.35],
    )

    for row, (fkey, fname, cmap, mode) in enumerate(fields):
        Z    = field_data[fkey]
        vmin, vmax = Z.min(), Z.max()

        # ── LEFT panel: profiles vs x ─────────────────────────────────────
        ax_l = fig.add_subplot(gs[row, 0])
        style_ax(ax_l)

        for c, ti in zip(cmap_lines, snap_idx):
            ax_l.plot(x, Z[ti], color=c, lw=1.6, label=f't = {t[ti]:.3f}')

        ax_l.axvline(0.5,     color=DIAP_COL,  lw=1.0, ls=':',  alpha=0.7)
        ax_l.axvline(shock_x, color=SHOCK_COL, lw=1.3, ls='--', alpha=0.85,
                     label='shock (t_end)')
        if not np.isnan(cd_x):
            ax_l.axvline(cd_x, color=CD_COL, lw=1.3, ls=':', alpha=0.85,
                         label='contact disc.')

        ax_l.set_xlabel('x',   color='#444444', fontsize=22)
        ax_l.set_ylabel(fname, color='#111111', fontsize=24)
        ax_l.set_title(f'{fname}  vs  x  at 7 time snapshots',
                       color='#111111', fontsize=24, fontweight='bold', pad=5)
        ax_l.set_xlim(0, 1)
        ax_l.legend(fontsize=18, facecolor='none', edgecolor='none',
                    labelcolor='#222222', ncol=1,
                    loc='upper left', bbox_to_anchor=(1.02, 1.0),
                    bbox_transform=ax_l.transAxes)
        ax_l.grid(alpha=0.3, color='#dddddd', linewidth=0.7)

        # ── RIGHT panel: space-time heatmap ──────────────────────────────
        ax_r = fig.add_subplot(gs[row, 1])
        style_ax(ax_r)

        if mode == 'div':
            vext = max(abs(vmin), abs(vmax))
            norm = TwoSlopeNorm(vmin=-vext, vcenter=0, vmax=vext)
            im = ax_r.pcolormesh(X_grid, T_grid, Z, cmap=cmap, norm=norm,
                                 shading='auto', rasterized=True)
        else:
            im = ax_r.pcolormesh(X_grid, T_grid, Z, cmap=cmap,
                                 vmin=vmin, vmax=vmax,
                                 shading='auto', rasterized=True)

        ax_r.plot(shock_traj, t, color=SHOCK_COL, lw=1.5, ls='--',
                  alpha=0.9, label='Shock front')
        ax_r.plot(cd_traj,    t, color=CD_COL,    lw=1.5, ls=':',
                  alpha=0.9, label='Contact disc.')
        ax_r.axvline(0.5, color=DIAP_COL, lw=0.8, ls=':', alpha=0.5)

        cb = plt.colorbar(im, ax=ax_r, fraction=0.035, pad=0.02)
        cb.ax.tick_params(colors='#444444', labelsize=18)

        ax_r.set_xlabel('x', color='#444444', fontsize=22)
        ax_r.set_ylabel('t', color='#444444', fontsize=22)
        ax_r.set_title(f'{fname}(x, t)  — space–time heatmap',
                       color='#111111', fontsize=24, fontweight='bold', pad=5)
        ax_r.set_xlim(0, 1)
        ax_r.set_ylim(0, t[-1])
        ax_r.legend(fontsize=18.5, facecolor='white', edgecolor='#cccccc',
                    labelcolor='#222222', loc='upper left')

    # ── title and metadata ────────────────────────────────────────────────────
    import os
    fig.text(0.5, 0.975,
             f'NS Shock Tube — {os.path.basename(npz_path)}',
             ha='center', va='top', color='#111111', fontsize=32, fontweight='bold')
    fig.text(0.5, 0.957,
             f'p_ratio = {p_ratio:.3f}   μ = {mu:.4f}   '
             f'ρ_L = {rho_L:.3f}   ρ_R = {rho_R:.3f}   p_R = {p_R_val:.3f}   '
             f'shock_x(t_end) = {shock_x:.3f}   cd_x(t_end) = {cd_x:.3f}',
             ha='center', va='top', color='#555555', fontsize=22.5)
    fig.text(0.5, 0.941,
             'Left: field profiles vs x at 7 time snapshots  ·  '
             'Right: space–time (t–x) heatmap  ·  '
             'red dashed = shock front,  blue dotted = contact discontinuity',
             ha='center', va='top', color='#777777', fontsize=21)

    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved → {out_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description='Plot NS shock tube outputs (ρ, u, p, T) from a scenario .npz file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--file', type=str, default='scenario_0001.npz',
                    help='Path to the scenario .npz file')
    ap.add_argument('--out',  type=str, default=None,
                    help='Output PNG path (default: <file_stem>_tx_plot.png)')
    args = ap.parse_args()

    import os
    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.file))[0]
        args.out = f'{stem}_tx_plot.png'

    plot_scenario(args.file, args.out)


if __name__ == '__main__':
    main()
