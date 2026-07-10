"""
navier_stokes_shock_tube.py
============================
1-D Navier-Stokes Shock Tube Simulator
=======================================

Solves the viscous compressible Navier-Stokes equations in conservation form:

    ∂ρ/∂t  + ∂(ρu)/∂x                       = 0            (mass)
    ∂(ρu)/∂t + ∂(ρu² + p)/∂x               = ∂τ/∂x         (momentum)
    ∂E/∂t  + ∂((E+p)u)/∂x                  = ∂(τu − q)/∂x  (energy)

where:
    E   = p/(γ−1) + ½ρu²        total energy per unit volume = thermal energy + kinetic energy
    p   = (γ−1)(E − ½ρu²)       ideal-gas equation of state
    τ   = (4/3)μ ∂u/∂x          viscous stress  Pa (Stokes hypothesis) Newton's law of viscosity
    q   = −κ ∂T/∂x              heat flux       (Fourier's law)
    T   = p/(ρ R)               temperature     (R = 1, non-dim)
    κ   = μ Cp / Pr             thermal conductivity: Sutherland's approach
    ρ   = density
    u   = velocity
    p   = pressure
    γ   = ratio of specific heats (1.4 for air)
    Pr  = Prandtl number(0.72 for air)   momentum diffusivity / thermal diffusivity
    Cp  = specific heat at constant pressure  (1004.5 for air) γR/(γ−1) where R = 287 J/(kg·K) 
    μ   = dynamic viscosity Pa/s
    α   = κ/(ρ c)   themodiffusity
Numerics:
    - Operator-split: inviscid Lax-Friedrichs + implicit viscous solve
    - Inviscid CFL: dt = cfl * dx / max(|u|+c)
    - Viscous:      implicit θ-method (θ=1 backward Euler) — stable for any dt
    - This decouples the advective and diffusive stability limits

Initial condition (Sod / generalised shock tube):
    Left  (x < 0.5): ρ_L=1.0, u=0, p_L  (high-pressure driver)
    Right (x ≥ 0.5): ρ_R=0.125, u=0, p_R=0.1 (low-pressure driven)

Usage
-----
    python navier_stokes_shock_tube.py                 # single run + plot
    python navier_stokes_shock_tube.py --p-ratio 5.0   # change pressure ratio
    python navier_stokes_shock_tube.py --sweep         # run 100-scenario sweep with default
    python navier_stokes_shock_tube.py --no-plot       # headless
    python navier_stokes_shock_tube.py --sweep --n-scenarios 500 --p-ratio-min 2 --p-ratio-max 50 \
                     --mu-min 0 --mu-max 0.05 --sweep-N 256 --seed 7 # customise ranges
    python navier_stokes_shock_tube.py --p-ratio 10 --mu 0.005 # single run unchanged 

Dependencies: numpy, scipy, matplotlib (optional), pandas (optional)
"""

import argparse
import numpy as np
from scipy.linalg import solve_banded

# ── Physical constants ──────────────────────────────────────────────────────
GAMMA = 1.4
PR    = 0.72
R_GAS = 1.0 # R_GAS = 1.0, it's non-dimensional, R_GAS = 287 it's dimensional
CP    = GAMMA * R_GAS / (GAMMA - 1)


# ── Equation of state ───────────────────────────────────────────────────────
# model inputs, ρ, u, T, p can get from ρ and T
def prim_to_cons(rho, u, p):
    """(ρ, u, p) → U = [ρ, ρu, E]"""
    E = p / (GAMMA - 1) + 0.5 * rho * u**2
    return np.array([rho, rho * u, E])


def cons_to_prim(U):
    """U = [ρ, ρu, E] → (ρ, u, p, T)"""
    rho = np.maximum(U[0], 1e-12)
    u   = U[1] / rho
    p   = (GAMMA - 1) * (U[2] - 0.5 * rho * u**2)
    p   = np.maximum(p, 1e-12)
    T   = p / (rho * R_GAS)
    return rho, u, p, T


# ── Inviscid Lax-Friedrichs flux (vectorised) ────────────────────────────────
def inviscid_step(U, dx, dt):
    """
    Update U state for one step According to Euler Equation (right term =0, no viscid source)
    One explicit Lax-Friedrichs step for the Euler equations.
    U - balance state:  [ ρᵢ,  ρᵢuᵢ,  Eᵢ ]: balance equation (first term on the left)
    F - Flux:   [ ρᵢuᵢ,  ρᵢuᵢ**2,  (Eᵢ+p)*u ]: balance equation (second term on the left)
    ∂U/∂t + ∂F/∂x(spatial gradience of flux) = 0
    """
    rho, u, p, _ = cons_to_prim(U)
    E = U[2]
    
    #Euler fluxes [ mass flux, momentum flux, energy flux]
    F = np.array([rho*u, rho*u**2 + p, (E + p)*u])   # (3, N)
    # U = [ ρᵢ,  ρᵢuᵢ,  Eᵢ ]
  
    alpha = dx / dt    # Lax-Friedrichs numerical diffusion coefficient
    # boundary of each lattice
    FL, FR = F[:, :-1], F[:, 1:]
    UL, UR = U[:, :-1], U[:, 1:]
    # F̂ = ½(Fₗ + Fᵣ)         ← centred average of physical fluxes
    #   − ½(dx/dt)(Uᵣ − Ul)  ← artificial diffusion term
    F_hat  = 0.5*(FL + FR) - 0.5*alpha*(UR - UL)      # (3, N-1) interface fluxes

    dU = np.zeros_like(U)
    dU[:, 1:-1] = -(dt/dx) * (F_hat[:, 1:] - F_hat[:, :-1]) # ∂U/∂t + ∂F/∂x = 0 -> ∂U/∂t = −∂F/∂x  ->  ∂U = ∂t*−∂F/∂x
    return U + dU


# ── Implicit viscous sub-step (tridiagonal solve) ────────────────────────────
def viscous_step(U, dx, dt, mu):
    """
    Backward-Euler implicit solve for the viscous and heat-conduction terms.
    Solves (I - dt * L) v = v_old  for momentum and energy separately,
    where L is the 2nd-order diffusion operator.  Unconditionally stable.
    mu = 
    """
    if mu == 0.0: # if no viscous, just the above the equation
        return U

    rho, u, p, T = cons_to_prim(U)
    N = U.shape[1]
    kappa = mu * CP / PR # thermal conductivity

    # ── momentum: implicit diffusion of u  ────────────────────────────────
    # ∂(ρu)/∂t = ∂τ/∂x => (4/3)μ ∂²u/∂x²   →  treat as ∂u/∂t = (4/3)(μ/ρ) ∂²u/∂x²
    # τ = (4/3)μ ∂u/∂x  =>  ∂u/∂t = nu · ∂²u/∂x² = ≈  nu (u[i-1] - 2·u[i] + u[i+1]) / dx² 
    # We solve for u_new then reconstruct ρu = ρ * u_new
    nu_mom = (4/3) * mu / rho       # spatially varying kinematic viscosity τ # units: m²/s
    r_mom  = dt * nu_mom / dx**2    # diffusion number (local) ∂u = r_mom (u[i-1] - 2·u[i] + u[i+1])

    # tridiagonal: −r·u_new_{i-1} + (1+2r)·u_new_{i} − r·u_new_{i+1} = u_old_{i}
    # Boundary: u[0] and u[N-1] held fixed (zero-gradient applied after)
    ab_mom = np.zeros((3, N))       # banded storage: [upper, diag, lower]
    ab_mom[0, 1:]  = -r_mom[:-1]   # super-diagonal
    ab_mom[1,  :]  =  1 + 2*r_mom  # diagonal
    ab_mom[2, :-1] = -r_mom[1:]    # sub-diagonal
    # enforce zero-gradient BCs: ghost cell = first interior cell
    ab_mom[1,  0]  = 1.0;  ab_mom[0, 1]   = 0.0; # left boundary
    ab_mom[1, -1]  = 1.0;  ab_mom[2, -2]  = 0.0; # right boundary
    # A · u_new = u_old
    u_new = solve_banded((1, 1),   #  
                         ab_mom,   # the matrix A stored in banded form
                         u.copy()) # u_old — the right hand side

    # ── energy: implicit diffusion of T  ──────────────────────────────────
    # ∂E/∂t  + = ∂(τu − q)/∂x  (heat source)
    #q   = −κ ∂T/∂x              heat flux       (Fourier's law)
    #T   = p/(ρ R)               temperature     (R = 1, non-dim)
    #κ   = μ Cp / Pr             thermal conductivity: Sutherland's approach
    #τ   = (4/3)μ ∂u/∂x          viscous stress  Pa (Stokes hypothesis) Newton's law of viscosity
    """
    ∂E/∂t = −∂q/∂x = ∂(κ ∂T/∂x)/∂x = κ ∂²T/∂x² from enery equation
    E =  p/(γ−1)  =  ρ R T/(γ−1)  =  ρ Cᵥ T ->  
    ∂E/∂t = ρ Cᵥ ∂T/∂t = κ ∂²T/∂x² ->
    ∂T/∂t = (κ / ρ Cᵥ) ∂²T/∂x²
    """
    nu_T  = kappa / (rho * CP)      # thermal diffusivity  α = κ/(ρ c)  [*]
    r_T   = dt * nu_T / dx**2

    ab_T = np.zeros((3, N))
    ab_T[0, 1:]  = -r_T[:-1]
    ab_T[1, :]   =  1 + 2*r_T
    ab_T[2, :-1] = -r_T[1:]

    ab_T[1, 0]  = 1.0; ab_T[0, 1]  = 0.0
    ab_T[1, -1] = 1.0; ab_T[2, -2] = 0.0
    T_new = solve_banded((1, 1), ab_T, T.copy())

    # [*] Note: this treats ρ as frozen during the viscous sub-step,
    #     which is valid for operator-split schemes at small dt.

    # ── reconstruct conserved variables ───────────────────────────────────
    U_new = U.copy()
    U_new[1] = rho * u_new              # ρu
    p_new    = rho * R_GAS * T_new      # ideal gas: p = ρRT
    U_new[2] = p_new / (GAMMA - 1) + 0.5 * rho * u_new**2   # E
    return U_new


# ── Boundary conditions ───────────────────────────────────────────────────────
def apply_bc(U):
    """
    Zero-gradient (transmissive) walls at both ends. 
    ∂U/∂x=0: allowing disturbances to leave the domain with minimal reflection
    """
    U[:, 0]  = U[:, 1]
    U[:, -1] = U[:, -2]
    return U


# ── CFL time-step  (advective only — viscous handled implicitly) ─────────────
def compute_dt(U, dx, cfl):
    rho, u, p, _ = cons_to_prim(U)
    c = np.sqrt(GAMMA * p / rho)            # local speed of sound
    max_wave = float(np.max(np.abs(u) + c)) # wave speed
    return cfl * dx / max_wave if max_wave > 1e-12 else 1e-3


# ── Main solver ───────────────────────────────────────────────────────────────
def solve(
    p_ratio     = 10.0,
    rho_L       = 1.0,
    rho_R       = 0.125,
    p_R         = 0.1,
    mu          = 0.005,
    N           = 128,
    t_end       = 0.2,
    cfl         = 0.45,
    n_snapshots = 41,
    verbose     = True,
):
    """
    Run the 1-D Navier-Stokes shock tube simulation.
    at t=0, The gas on the left expands into the right side

    Parameters
    ----------
    p_ratio     : float  — initial pressure ratio p_L / p_R (strong or week pressure gradient)
    rho_L/R     : float  — initial left/right densities
    p_R         : float  — right pressure  (p_L = p_ratio * p_R)
    mu          : float  — dynamic viscosity  (0 = inviscid Euler)
    N           : int    — spatial grid cells
    t_end       : float  — simulation end time
    cfl         : float  — CFL safety factor for inviscid step (< 1)
    n_snapshots : int    — number of evenly-spaced time levels to record - num of time
    verbose     : bool   — print progress

    Returns
    -------
    dict with keys:
        x          : (N,)              cell-centre positions [0, 1]
        t_snaps    : (n_snapshots,)    snapshot times
        rho        : (n_snapshots, N)  density field
        u          : (n_snapshots, N)  velocity field
        p          : (n_snapshots, N)  pressure field
        T          : (n_snapshots, N)  temperature field
        p_ratio    : float
        diagnostics: dict  (shock_pos, cd_pos, post_shock_density_ratio, ...)
    """
    # coefficent
    p_L = p_ratio * p_R
    
    # grid
    x   = np.linspace(0.5/N, 1.0 - 0.5/N, N)
    dx  = 1.0 / N

    # Initial conditions
    U = np.where(
        x[np.newaxis, :] < 0.5, # separate left and right side initiation
        prim_to_cons(rho_L, 0.0, p_L)[:, np.newaxis], # momenten = 0
        prim_to_cons(rho_R, 0.0, p_R)[:, np.newaxis], # momenten = 0
    ).astype(float)

    # Snapshot storage all history =0
    snap_times = np.linspace(0.0, t_end, n_snapshots)
    rho_hist   = np.zeros((n_snapshots, N))
    u_hist     = np.zeros((n_snapshots, N))
    p_hist     = np.zeros((n_snapshots, N))
    T_hist     = np.zeros((n_snapshots, N))
    ptr        = [0] # record of time

    def record(U_now, t_now):
        while ptr[0] < n_snapshots and snap_times[ptr[0]] <= t_now + 1e-12:
            r, uv, pv, Tv = cons_to_prim(U_now)
            k = ptr[0]
            rho_hist[k] = r;  u_hist[k] = uv
            p_hist[k]   = pv; T_hist[k] = Tv
            ptr[0] += 1

    record(U, 0.0)

    # Time integration
    t = 0.0; n_steps = 0
    while t < t_end - 1e-12:
        # get time step
        dt = min(compute_dt(U, dx, cfl), t_end - t)

        # 1) Inviscid step (explicit Lax-Friedrichs)
        U = apply_bc(U)
        U = inviscid_step(U, dx, dt)
        U = apply_bc(U)

        # 2) Viscous step (implicit, unconditionally stable)
        if mu > 0.0:
            U = viscous_step(U, dx, dt, mu)
            U = apply_bc(U)

        t += dt; n_steps += 1
        record(U, t)

    # Pad remaining snapshots
    while ptr[0] < n_snapshots:
        k = ptr[0]
        rho_hist[k] = rho_hist[k-1]; u_hist[k] = u_hist[k-1]
        p_hist[k]   = p_hist[k-1];   T_hist[k] = T_hist[k-1]
        ptr[0] += 1

    # Diagnostics final value and gradient
    rho_f = rho_hist[-1]; p_f = p_hist[-1]; # final value
    dp_dx   = np.abs(np.gradient(p_f,   x))
    drho_dx = np.abs(np.gradient(rho_f, x))

    
    shock_pos = float(x[np.argmax(dp_dx)])  # max pressure gradient location
    mask = x < shock_pos - 0.05             # mask where presuregradient not maximun 
    cd_pos = float(x[mask][np.argmax(drho_dx[mask])]) if mask.sum() > 0 else float('nan') # find large density jump, but pressure is nearly continuous
    post_shock_rho_ratio = float(np.max(rho_f) / rho_R) # max final density / initial right-side density

    mass_0   = rho_L * 0.5 + rho_R * 0.5 # mean mass
    mass_err = abs(np.mean(rho_f) - mass_0) / mass_0 * 100 # mass change

    if verbose:
        print(f"p_ratio={p_ratio:5.1f}  steps={n_steps:4d}  "
              f"mass_err={mass_err:.4f}%  "
              f"shock_x={shock_pos:.4f}  cd_x={cd_pos:.4f}  "
              f"ρ₂/ρ_R={post_shock_rho_ratio:.3f}")

    assert mass_err < 2.0, (
        f"Mass-conservation error {mass_err:.2f}% > 2% (p_ratio={p_ratio}). "
        f"Try larger N or smaller CFL."
    )

    return {
        'x':          x,
        't_snaps':    snap_times,
        'rho':        rho_hist,
        'u':          u_hist,
        'p':          p_hist,
        'T':          T_hist,
        'p_ratio':    p_ratio,
        'diagnostics': {
            'shock_pos':                shock_pos,
            'cd_pos':                   cd_pos,
            'post_shock_density_ratio': post_shock_rho_ratio,
            'mass_error_pct':           mass_err,
            'n_steps':                  n_steps,
        },
    }


# ── 500-scenario sweep → CSV ──────────────────────────────────────────────────
def _lhs_sample(n, lo, hi, rng, log_scale=False):
    """
    Latin Hypercube sample: divide [lo, hi] into n equal intervals,
    draw one value uniformly from each interval, then shuffle.
    log_scale=True samples in log space (better for p_ratio which spans decades).

    Why LHS over pure random?
      Pure random can cluster — some regions get many samples, others none.
      LHS guarantees every interval has exactly one sample → full coverage.
    """
    if log_scale:
        lo, hi = np.log(lo), np.log(hi)
    cuts = np.linspace(lo, hi, n + 1)           # n+1 edges → n intervals
    vals = rng.uniform(cuts[:-1], cuts[1:])      # one draw per interval
    vals = rng.permutation(vals)                 # shuffle so params are decorrelated
    return np.exp(vals) if log_scale else vals


def sweep(
    n_scenarios  = 500,
    # ── physics parameter ranges ──────────────────────────────────────────
    p_ratio_min  = 2.0,    p_ratio_max  = 20.0,   # shock strength
    mu_min       = 0.0,    mu_max       = 0.02,    # viscosity (0 = inviscid)
    rho_L_min    = 0.5,    rho_L_max    = 2.0,     # left density
    rho_R_min    = 0.05,   rho_R_max    = 0.5,     # right density
    p_R_min      = 0.05,   p_R_max      = 0.3,     # right pressure
    # ── fixed numerics ────────────────────────────────────────────────────
    N            = 256,    # grid cells — finer than default for dataset quality
    t_end        = 0.2,
    cfl          = 0.45,
    n_snapshots  = 41,
    # ── output ────────────────────────────────────────────────────────────
    output_dir   = './data',   # each scenario saved as scenario_NNNN.npz here
    seed         = 42,
):
    """
    Latin Hypercube Sampling sweep over all 5 physics parameters.

    Each completed scenario is saved as an individual compressed NumPy file:

        ./data/scenario_0001.npz
        ./data/scenario_0002.npz
        ...
        ./data/index.npz          ← parameter table for all completed scenarios

    Why one file per scenario?
    --------------------------
    Loading a single large CSV into memory is impractical for large sweeps
    (500 scenarios × 256 cells × 41 snapshots = 52 M rows).  One .npz per
    scenario lets the training dataloader stream files on demand, load only
    the scenarios it needs, and re-use the same file across multiple epochs
    without re-parsing text.

    How to load a scenario in training code
    ----------------------------------------
        import numpy as np

        d = np.load('data/scenario_0001.npz')

        # ── grid ─────────────────────────────────────────────────────────
        x       = d['x']        # (N,)          spatial cell centres [0,1]
        t       = d['t_snaps']  # (n_snaps,)    snapshot times [0, t_end]

        # ── ML input parameters (scalars) ────────────────────────────────
        p_ratio = float(d['p_ratio'])   # pressure ratio  p_L / p_R
        mu      = float(d['mu'])        # dynamic viscosity
        rho_L   = float(d['rho_L'])     # left initial density
        rho_R   = float(d['rho_R'])     # right initial density
        p_R     = float(d['p_R'])       # right initial pressure

        # ── ML output fields  shape: (n_snaps, N) ────────────────────────
        rho     = d['rho']      # density
        u       = d['u']        # velocity
        p       = d['p']        # pressure
        T       = d['T']        # temperature

        # ── diagnostics ──────────────────────────────────────────────────
        shock_pos = float(d['shock_pos'])   # shock position at t_end
        cd_pos    = float(d['cd_pos'])      # contact-disc. position at t_end
        rho_ratio = float(d['rho_ratio'])   # post-shock density ratio
        mass_err  = float(d['mass_err'])    # mass-conservation error %

    Why Latin Hypercube (LHS) instead of the old 1-D p_ratio sweep?
    ----------------------------------------------------------------
    Old sweep: only p_ratio varies → all simulations share mu=0.005,
               rho_L=1.0, rho_R=0.125, p_R=0.1.  The dataset captures only
               one axis of the physics.  A PINN trained on it cannot
               generalise to different viscosities or densities.

    LHS sweep: all 5 physics parameters vary simultaneously.
               Every parameter is divided into n_scenarios equal intervals.
               Exactly one sample is drawn per interval, then intervals are
               shuffled independently per parameter — so no two parameters
               are correlated.  This gives maximum coverage of the 5-D
               parameter space with the minimum number of simulations.

    Parameters varied per simulation
    ---------------------------------
    p_ratio : pressure ratio p_L/p_R   — controls shock strength
    mu      : dynamic viscosity        — controls wave thickness
    rho_L   : left (driver) density    — changes compression ratio
    rho_R   : right (driven) density   — changes post-shock state
    p_R     : right (driven) pressure  — sets absolute pressure scale
    """
    import os, json

    # ── create output directory ───────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.RandomState(seed)

    # ── draw LHS samples for all 5 physics parameters ────────────────────────
    p_ratios = _lhs_sample(n_scenarios, p_ratio_min, p_ratio_max, rng, log_scale=True)
    mus      = _lhs_sample(n_scenarios, mu_min,      mu_max,      rng, log_scale=False)
    rho_Ls   = _lhs_sample(n_scenarios, rho_L_min,   rho_L_max,   rng, log_scale=False)
    rho_Rs   = _lhs_sample(n_scenarios, rho_R_min,   rho_R_max,   rng, log_scale=False)
    p_Rs     = _lhs_sample(n_scenarios, p_R_min,     p_R_max,     rng, log_scale=False)

    print(f"\nLHS sweep: {n_scenarios} scenarios over 5 physics parameters")
    print(f"  p_ratio : [{p_ratio_min}, {p_ratio_max}] log-space")
    print(f"  mu      : [{mu_min}, {mu_max}]")
    print(f"  rho_L   : [{rho_L_min}, {rho_L_max}]")
    print(f"  rho_R   : [{rho_R_min}, {rho_R_max}]")
    print(f"  p_R     : [{p_R_min}, {p_R_max}]")
    print(f"  N={N}, t_end={t_end}, cfl={cfl}, snapshots={n_snapshots}")
    print(f"  output  : {os.path.abspath(output_dir)}/scenario_NNNN.npz\n")

    saved   = []    # list of dicts, one per completed scenario (for index)
    skipped = []    # list of dicts for failed scenarios

    for k in range(n_scenarios):
        pr    = float(p_ratios[k])
        mu    = float(mus[k])
        rho_L = float(rho_Ls[k])
        rho_R = float(rho_Rs[k])
        p_R   = float(p_Rs[k])

        print(f"[{k+1:{len(str(n_scenarios))}d}/{n_scenarios}] "
              f"p_ratio={pr:5.2f}  mu={mu:.4f}  "
              f"rho_L={rho_L:.3f}  rho_R={rho_R:.3f}  p_R={p_R:.3f}  ",
              end='', flush=True)

        try:
            res = solve(
                p_ratio     = pr,
                mu          = mu,
                rho_L       = rho_L,
                rho_R       = rho_R,
                p_R         = p_R,
                N           = N,
                t_end       = t_end,
                cfl         = cfl,
                n_snapshots = n_snapshots,
                verbose     = True,   # prints mass_err, shock_pos etc.
            )
        except AssertionError as e:
            print(f"  *** SKIPPED: {e}")
            skipped.append(dict(scenario=k+1, p_ratio=pr, mu=mu,
                                rho_L=rho_L, rho_R=rho_R, p_R=p_R,
                                reason=str(e)))
            continue

        # ── save one .npz per scenario ────────────────────────────────────────
        diag  = res['diagnostics']
        fname = f"scenario_{k+1:04d}.npz"
        fpath = os.path.join(output_dir, fname)

        np.savez_compressed(
            fpath,
            # ── grid ─────────────────────────────────────────────────────
            x        = res['x'],           # (N,)          cell-centre positions
            t_snaps  = res['t_snaps'],     # (n_snapshots,) snapshot times
            # ── ML output fields  shape: (n_snapshots, N) ────────────────
            rho      = res['rho'],         # density
            u        = res['u'],           # velocity
            p        = res['p'],           # pressure
            T        = res['T'],           # temperature
            # ── ML input parameters (scalar per scenario) ─────────────────
            p_ratio  = np.float64(pr),
            mu       = np.float64(mu),
            rho_L    = np.float64(rho_L),
            rho_R    = np.float64(rho_R),
            p_R      = np.float64(p_R),
            # ── diagnostics ───────────────────────────────────────────────
            shock_pos   = np.float64(diag['shock_pos']),
            cd_pos      = np.float64(diag['cd_pos']),
            rho_ratio   = np.float64(diag['post_shock_density_ratio']),
            mass_err    = np.float64(diag['mass_error_pct']),
            n_steps     = np.int32(diag['n_steps']),
            scenario_id = np.int32(k + 1),
        )

        saved.append(dict(
            filename    = fname,
            scenario_id = k + 1,
            p_ratio     = round(pr,    6),
            mu          = round(mu,    6),
            rho_L       = round(rho_L, 6),
            rho_R       = round(rho_R, 6),
            p_R         = round(p_R,   6),
            shock_pos   = round(diag['shock_pos'],                6),
            cd_pos      = round(diag['cd_pos'],                   6),
            rho_ratio   = round(diag['post_shock_density_ratio'], 6),
            mass_err    = round(diag['mass_error_pct'],           6),
            grid_N      = N,
        ))

    # ── write index.json  (parameter table for all completed scenarios) ───────
    index = {
        'n_completed' : len(saved),
        'n_skipped'   : len(skipped),
        'n_requested' : n_scenarios,
        'grid_N'      : N,
        'n_snapshots' : n_snapshots,
        't_end'       : t_end,
        'lhs_seed'    : seed,
        'parameter_ranges': {
            'p_ratio': [p_ratio_min, p_ratio_max, 'log-space'],
            'mu'     : [mu_min,      mu_max],
            'rho_L'  : [rho_L_min,   rho_L_max],
            'rho_R'  : [rho_R_min,   rho_R_max],
            'p_R'    : [p_R_min,     p_R_max],
        },
        'npz_arrays': {
            'x'          : f'(N={N},)  — spatial cell centres in [0, 1]',
            't_snaps'    : f'(n_snapshots={n_snapshots},)  — snapshot times in [0, {t_end}]',
            'rho'        : f'(n_snapshots={n_snapshots}, N={N})  — density field',
            'u'          : f'(n_snapshots={n_snapshots}, N={N})  — velocity field',
            'p'          : f'(n_snapshots={n_snapshots}, N={N})  — pressure field',
            'T'          : f'(n_snapshots={n_snapshots}, N={N})  — temperature field',
            'p_ratio'    : 'scalar  — input: pressure ratio p_L/p_R',
            'mu'         : 'scalar  — input: dynamic viscosity',
            'rho_L'      : 'scalar  — input: left initial density',
            'rho_R'      : 'scalar  — input: right initial density',
            'p_R'        : 'scalar  — input: right initial pressure',
            'shock_pos'  : 'scalar  — diagnostic: shock position at t_end',
            'cd_pos'     : 'scalar  — diagnostic: contact-disc. position at t_end',
            'rho_ratio'  : 'scalar  — diagnostic: post-shock density ratio',
            'mass_err'   : 'scalar  — diagnostic: mass-conservation error %',
            'n_steps'    : 'scalar  — number of time-integration steps taken',
            'scenario_id': 'scalar  — 1-based scenario index',
        },
        'scenarios': saved,
        'skipped'  : skipped,
    }
    index_path = os.path.join(output_dir, 'index.json')
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    # ── summary ───────────────────────────────────────────────────────────────
    total_mb = sum(
        os.path.getsize(os.path.join(output_dir, s['filename']))
        for s in saved
    ) / 1e6

    print(f"\n{'='*60}")
    print(f"Scenarios completed : {len(saved)}/{n_scenarios}")
    print(f"Scenarios skipped   : {len(skipped)}")
    print(f"Files saved         : {output_dir}/scenario_NNNN.npz  ×{len(saved)}")
    print(f"Index               : {index_path}")
    print(f"Total size          : {total_mb:.1f} MB")
    print(f"Arrays per file     : x, t_snaps, rho, u, p, T  +  5 params  +  diagnostics")
    print(f"Shape per file      : fields ({n_snapshots}, {N}),  grid ({N},),  time ({n_snapshots},)")
    if skipped:
        print(f"\nSkipped scenarios   : {[s['scenario'] for s in skipped]}")
    print(f"{'='*60}\n")

    return saved, skipped


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_result(res, times_to_plot=None):
    """
    Six-panel figure:
      top row:    ρ, u, p profiles at 5 time snapshots
      bottom row: T profile,  ρ(x,t) heatmap,  p(x,t) heatmap
    Wave-front markers overlaid on heatmaps.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("matplotlib not installed — pip install matplotlib"); return

    x, t_snaps, pr = res['x'], res['t_snaps'], res['p_ratio']
    idxs = (np.linspace(0, len(t_snaps)-1, 5, dtype=int)
            if times_to_plot is None
            else [int(np.argmin(np.abs(t_snaps - tv))) for tv in times_to_plot])
    colors = cm.viridis(np.linspace(0.15, 0.9, len(idxs)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f'Navier–Stokes Shock Tube  (p_ratio={pr:.1f},  μ={0.005},  N={len(x)})',
        fontsize=13, fontweight='bold',
    )
    for (key, label), ax in zip(
        [('rho','Density ρ'),('u','Velocity u'),('p','Pressure p'),('T','Temperature T')],
        [axes[0,0], axes[0,1], axes[0,2], axes[1,0]],
    ):
        for c, idx in zip(colors, idxs):
            ax.plot(x, res[key][idx], color=c, label=f't={t_snaps[idx]:.3f}', lw=1.6)
        ax.set_xlabel('x'); ax.set_ylabel(label); ax.set_title(label)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    diag = res['diagnostics']
    for ax, key, cmap, label in [
        (axes[1,1], 'rho', 'plasma',  'ρ(x,t)'),
        (axes[1,2], 'p',   'inferno', 'p(x,t)'),
    ]:
        im = ax.imshow(
            res[key], origin='lower', aspect='auto', cmap=cmap,
            extent=[x[0], x[-1], t_snaps[0], t_snaps[-1]],
        )
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel('x'); ax.set_ylabel('t'); ax.set_title(f'Space-time: {label}')
        for xpos, col, lbl in [
            (diag['shock_pos'], 'white', 'shock'),
            (diag['cd_pos'],    'cyan',  'contact disc.'),
        ]:
            if not np.isnan(xpos):
                ax.axvline(xpos, color=col, ls='--', lw=1.2, label=lbl)
        ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    fname = f'shock_tube_p{pr:.0f}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Plot saved → {fname}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description='1-D Navier-Stokes Shock Tube Simulator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── single-run options ────────────────────────────────────────────────────
    sg = ap.add_argument_group('single run')
    sg.add_argument('--p-ratio',     type=float, default=10.0,
                    help='Initial pressure ratio p_L/p_R')
    sg.add_argument('--mu',          type=float, default=0.005,
                    help='Dynamic viscosity (0 = inviscid Euler)')
    sg.add_argument('--N',           type=int,   default=128,
                    help='Spatial grid cells')
    sg.add_argument('--t-end',       type=float, default=0.2,
                    help='Simulation end time')
    sg.add_argument('--cfl',         type=float, default=0.45,
                    help='CFL safety factor (advective, < 1)')
    sg.add_argument('--snapshots',   type=int,   default=41,
                    help='Number of time snapshots to record')
    sg.add_argument('--no-plot',     action='store_true',
                    help='Skip plotting (single run)')

    # ── LHS sweep options ─────────────────────────────────────────────────────
    sw = ap.add_argument_group('LHS sweep  (--sweep)')
    sw.add_argument('--sweep',         action='store_true',
                    help='Run Latin Hypercube sweep over all 5 physics params')
    sw.add_argument('--n-scenarios',   type=int,   default=500,
                    help='Number of LHS scenarios')
    sw.add_argument('--output-dir',    type=str,   default='./data',
                    help='Output directory — each scenario saved as scenario_NNNN.npz')
    sw.add_argument('--seed',          type=int,   default=42,
                    help='Random seed for LHS sampling')
    sw.add_argument('--sweep-N',       type=int,   default=256,
                    help='Grid cells for sweep runs (finer than single run)')
    # physics ranges
    sw.add_argument('--p-ratio-min',   type=float, default=2.0)
    sw.add_argument('--p-ratio-max',   type=float, default=20.0)
    sw.add_argument('--mu-min',        type=float, default=0.0)
    sw.add_argument('--mu-max',        type=float, default=0.02)
    sw.add_argument('--rho-L-min',     type=float, default=0.5)
    sw.add_argument('--rho-L-max',     type=float, default=2.0)
    sw.add_argument('--rho-R-min',     type=float, default=0.05)
    sw.add_argument('--rho-R-max',     type=float, default=0.5)
    sw.add_argument('--p-R-min',       type=float, default=0.05)
    sw.add_argument('--p-R-max',       type=float, default=0.3)

    args = ap.parse_args()

    if args.sweep:
        sweep(
            n_scenarios  = args.n_scenarios,
            p_ratio_min  = args.p_ratio_min,   p_ratio_max = args.p_ratio_max,
            mu_min       = args.mu_min,         mu_max      = args.mu_max,
            rho_L_min    = args.rho_L_min,      rho_L_max   = args.rho_L_max,
            rho_R_min    = args.rho_R_min,      rho_R_max   = args.rho_R_max,
            p_R_min      = args.p_R_min,        p_R_max     = args.p_R_max,
            N            = args.sweep_N,
            t_end        = args.t_end,
            cfl          = args.cfl,
            n_snapshots  = args.snapshots,
            output_dir   = args.output_dir,
            seed         = args.seed,
        )
    else:
        res = solve(
            p_ratio     = args.p_ratio,
            mu          = args.mu,
            N           = args.N,
            t_end       = args.t_end,
            cfl         = args.cfl,
            n_snapshots = args.snapshots,
            verbose     = True,
        )
        if not args.no_plot:
            plot_result(res)


if __name__ == '__main__':
    main()
