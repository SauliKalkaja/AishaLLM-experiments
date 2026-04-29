"""
Experiment A1: Search for conserved invariants along an SSM recurrence.

The 6D framework's load-bearing identity is (alpha_s + beta_s)^2 - M^2 = 4
(unification paper, eq. derivation in section 2.9). This holds because the
symplectic lock alpha_s * beta_s = 1 forces a hyperbolic relationship between
the metric scaling factors and the torsion magnitude.

The LLM-adjacent analogue lives on a state-space recurrence
        h_t = A_bar h_{t-1} + B_bar x_t
        y_t = C h_t
which is the exact form Mamba / S4 / RWKV reduce to in their parallel-scan
representation. We search for a scalar I(h_t, A_bar, input history) that is
approximately conserved along trajectories, and rank candidates by the
relative drift  |I_t - I_0| / |I_0|  -- the same drift metric the framework
paper uses for the RK45 vs analytical-jump Hamiltonian comparison (Fig 3).

Candidate invariants explored:
  I1:  ||h_t||^2                                 (state energy, naive)
  I2:  h_t^T P h_t                               (Lyapunov quadratic)
  I3:  h_t^T P h_t + sum_{s<t} u_s^T u_s         (state + supplied energy)
  I4:  h_t^T P h_t - sum_{s<t} u_s^T B^T P A B u_s  (passivity gap, lossless)
  I5:  framework-shaped (alpha+beta)^2 - M^2 with
         alpha_t = ||h_t|| / r0   beta_t = r0 / ||h_t||   (symplectic lock by
         construction)   M_t = ||h_t - A_bar h_{t-1}|| / r0   (input-driven
         "torsion")
  I6:  log-additive:  log||h_t||^2 - sum_{s<t} log(growth_s)   where
         growth_s = ||A_bar h_s + B u_s||^2 / ||h_s||^2

Five SSM configurations are tested (random stable, near-orthogonal,
hippo-style real-diagonal, complex-diagonal Mamba-like, marginal). For each
config we report the median, p95, and max relative drift across 16 random
input trajectories of length T=1000.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR.parent / "figures"
RES_DIR = OUT_DIR.parent / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# SSM configurations.
# ---------------------------------------------------------------------------

def ssm_random_stable(N: int, M: int, spectral_radius: float = 0.95):
    """Random A with all eigenvalues inside |lambda| <= spectral_radius."""
    A = torch.randn(N, N) * (1.0 / np.sqrt(N))
    eigs = torch.linalg.eigvals(A)
    rho = float(torch.abs(eigs).max())
    A = A * (spectral_radius / rho)
    B = torch.randn(N, M) * (1.0 / np.sqrt(M))
    return A, B


def ssm_near_orthogonal(N: int, M: int, eps: float = 1e-3):
    """A close to orthogonal: A = (1-eps) Q where Q is orthogonal."""
    Q, _ = torch.linalg.qr(torch.randn(N, N))
    A = (1.0 - eps) * Q
    B = torch.randn(N, M) * (1.0 / np.sqrt(M))
    return A, B


def ssm_hippo_real(N: int, M: int):
    """Real diagonal A with eigenvalues from a HiPPO-style spectrum."""
    # decay rates 1/(2k+1) range, then exp(-rate * dt) for a typical dt.
    rates = 1.0 / (2 * torch.arange(1, N + 1, dtype=torch.float64) + 1)
    dt = 0.1
    diag = torch.exp(-rates * dt)
    A = torch.diag(diag)
    B = torch.randn(N, M) * (1.0 / np.sqrt(M))
    return A, B


def ssm_complex_diag(N: int, M: int):
    """Block-diagonal complex eigenvalues (rotation+decay), Mamba-S4 style.

    Each 2x2 block is  [[r cos(omega), -r sin(omega)],
                         [r sin(omega),  r cos(omega)]]   with r in (0, 1).
    """
    A = torch.zeros(N, N)
    for k in range(N // 2):
        r = 0.85 + 0.1 * (k / (N // 2))   # 0.85 .. 0.95
        omega = 0.05 + 1.5 * (k / (N // 2))
        c, s = float(np.cos(omega)), float(np.sin(omega))
        A[2 * k:2 * k + 2, 2 * k:2 * k + 2] = torch.tensor([[r * c, -r * s],
                                                            [r * s,  r * c]])
    B = torch.randn(N, M) * (1.0 / np.sqrt(M))
    return A, B


def ssm_marginal(N: int, M: int):
    """Eigenvalues right at |lambda| = 0.999 -- nearly lossless but stable."""
    return ssm_random_stable(N, M, spectral_radius=0.999)


CONFIGS = {
    "random_stable":   ssm_random_stable,
    "near_orthogonal": ssm_near_orthogonal,
    "hippo_real":      ssm_hippo_real,
    "complex_diag":    ssm_complex_diag,
    "marginal":        ssm_marginal,
}


# ---------------------------------------------------------------------------
# Lyapunov solver (discrete-time).
# ---------------------------------------------------------------------------

def discrete_lyapunov(A: torch.Tensor, Q: torch.Tensor, max_iter: int = 200) -> torch.Tensor:
    """Solve A^T P A - P = -Q for P by squaring iteration. Stable for |lambda(A)|<1."""
    P = Q.clone()
    A_pow = A.clone()
    for _ in range(max_iter):
        new_term = A_pow.T @ P @ A_pow
        P = P + new_term
        if float(new_term.norm()) < 1e-14 * float(P.norm() + 1e-30):
            break
        A_pow = A_pow @ A_pow
    return P


# ---------------------------------------------------------------------------
# Invariant evaluators.
# ---------------------------------------------------------------------------

def run_trajectory(A: torch.Tensor, B: torch.Tensor, T: int, input_scale: float = 1.0):
    """Run h_{t+1} = A h_t + B u_t for T steps. Returns list of (h_t, u_t)."""
    N = A.shape[0]
    M = B.shape[1]
    h = torch.randn(N) * 1.0
    h0 = h.clone()
    traj = [(h.clone(), None)]
    us = []
    for t in range(T):
        u = torch.randn(M) * input_scale
        h_new = A @ h + B @ u
        traj.append((h_new.clone(), u.clone()))
        us.append(u.clone())
        h = h_new
    return traj, h0, us


def evaluate_invariants(A: torch.Tensor, B: torch.Tensor, traj, h0: torch.Tensor):
    """Compute six candidate invariants over a trajectory. Returns ndarray (T+1, 6)."""
    N = A.shape[0]
    P = discrete_lyapunov(A, torch.eye(N))
    # I4 weighting: q_t = u_t^T (B^T P B + something) u_t such that the gap is exact.
    # For the lossless invariant V(h_t) + sum input_storage = V(h_0), we use:
    #   V(h_{t+1}) = V(A h_t + B u_t) = (A h_t + B u_t)^T P (A h_t + B u_t)
    # Expand: h_t^T A^T P A h_t + 2 h_t^T A^T P B u_t + u_t^T B^T P B u_t.
    # Since A^T P A = P - I, the *lossless* stored-plus-supplied energy that
    # cancels exactly is, for orthogonal A and zero cross term, h_t^T h_t.
    # In general no exact-cancellation invariant exists for a contractive A,
    # so I4 is approximate -- we report its drift as an upper bound on how
    # well a passivity-inspired invariant approximates conservation.
    BTPB = B.T @ P @ B  # constant
    ATP  = A.T @ P
    ATPB = A.T @ P @ B

    r0 = float(h0.norm())
    invariants = []

    cum_input_energy = 0.0
    cum_input_passivity = 0.0
    cum_log_growth = 0.0

    h_prev = h0
    for idx, (h, u) in enumerate(traj):
        # I1: ||h||^2
        I1 = float((h @ h).item())
        # I2: h^T P h
        I2 = float((h @ (P @ h)).item())

        # update cumulative input terms (using u at this step, but at idx=0 there's no u)
        if u is not None:
            cum_input_energy += float((u @ u).item())
            # passivity-style: gap term  -2 h_prev^T A^T P B u  - u^T B^T P B u
            cross = float((h_prev @ (ATPB @ u)).item())
            quad = float((u @ (BTPB @ u)).item())
            cum_input_passivity += -(2 * cross + quad)
            # log-growth ratio
            denom = max(float((h_prev @ h_prev).item()), 1e-30)
            num = float((h @ h).item())
            cum_log_growth += float(np.log(max(num, 1e-30) / denom))

        I3 = I2 + cum_input_energy
        I4 = I2 + cum_input_passivity   # under lossless A this should be exactly I2_at_t=0

        # I5 framework-shaped
        # alpha = ||h|| / r0,  beta = r0 / ||h|| -> alpha*beta = 1 lock
        # M = ||h - A h_prev|| / r0   (this is exactly ||B u|| / r0)
        alpha = float(h.norm()) / max(r0, 1e-30)
        beta = max(r0, 1e-30) / max(float(h.norm()), 1e-30)
        if u is None:
            M_val = 0.0
        else:
            M_val = float((h - A @ h_prev).norm()) / max(r0, 1e-30)
        I5 = (alpha + beta) ** 2 - M_val ** 2

        # I6: log-additive
        I6 = float(np.log(max(I1, 1e-30))) - cum_log_growth

        invariants.append([I1, I2, I3, I4, I5, I6])
        h_prev = h

    return np.array(invariants)


def relative_drift(values: np.ndarray) -> np.ndarray:
    """|I_t - I_0| / max(|I_0|, eps) along the trajectory."""
    base = np.abs(values[0])
    base = np.maximum(base, 1e-30)
    return np.abs(values - values[0]) / base


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main():
    N, M_in = 32, 8
    T = 1000
    n_traj = 16
    inv_names = ["I1: ||h||^2", "I2: h^T P h", "I3: I2 + cum||u||^2",
                 "I4: I2 + passivity", "I5: framework (a+b)^2-M^2",
                 "I6: log-additive"]

    summary = {}

    fig, axes = plt.subplots(len(CONFIGS), 1, figsize=(11, 14), sharex=True)

    for ax, (cfg_name, cfg_fn) in zip(axes, CONFIGS.items()):
        torch.manual_seed(42)
        np.random.seed(42)
        A, B = cfg_fn(N, M_in)
        eigs = torch.linalg.eigvals(A)
        rho = float(torch.abs(eigs).max())

        all_drifts = np.zeros((n_traj, T + 1, 6))
        for j in range(n_traj):
            traj, h0, us = run_trajectory(A, B, T, input_scale=0.3)
            invs = evaluate_invariants(A, B, traj, h0)
            all_drifts[j] = relative_drift(invs)

        median_drift = np.median(all_drifts, axis=0)   # (T+1, 6)
        p95_drift = np.percentile(all_drifts, 95, axis=0)
        final_max = np.max(all_drifts[:, -1, :], axis=0)
        final_median = np.median(all_drifts[:, -1, :], axis=0)

        summary[cfg_name] = {
            "spectral_radius": rho,
            "final_median_drift": {n: float(d) for n, d in zip(inv_names, final_median)},
            "final_max_drift":    {n: float(d) for n, d in zip(inv_names, final_max)},
        }

        for k in range(6):
            ax.plot(median_drift[:, k] + 1e-30, label=inv_names[k], linewidth=1.0)
        ax.set_yscale("log")
        ax.set_title(f"{cfg_name}  (rho(A) = {rho:.4f})", fontsize=10)
        ax.set_ylabel("|I_t - I_0| / |I_0|")
        ax.grid(True, which="both", alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="lower right", ncol=2)

    axes[-1].set_xlabel("step t")
    fig.suptitle("Experiment A1: invariant drift along SSM trajectories (median over 16 runs)")
    fig.tight_layout()
    fig_path = FIG_DIR / "a1_invariant_drift.png"
    fig.savefig(fig_path, dpi=140)
    print(f"  saved figure: {fig_path}")

    res_path = RES_DIR / "a1_results.json"
    res_path.write_text(json.dumps(summary, indent=2))
    print(f"  saved results: {res_path}")

    # Compact ranking.
    print("\n=== Final-step relative drift (median over 16 trajectories, T=1000) ===")
    print(f"{'config':<18s} {'rho(A)':>8s}  " + "  ".join(f"{n.split(':')[0]:>8s}" for n in inv_names))
    for cfg_name, info in summary.items():
        row = f"{cfg_name:<18s} {info['spectral_radius']:8.4f}  "
        row += "  ".join(f"{info['final_median_drift'][n]:8.2e}" for n in inv_names)
        print(row)

    print("\n=== Best (smallest drift) invariant per configuration ===")
    for cfg_name, info in summary.items():
        ranked = sorted(info["final_median_drift"].items(), key=lambda kv: kv[1])
        print(f"  {cfg_name:<18s}  best: {ranked[0][0]:30s} drift={ranked[0][1]:.2e}")


if __name__ == "__main__":
    main()
