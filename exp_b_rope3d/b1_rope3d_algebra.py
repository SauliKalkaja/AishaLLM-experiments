"""
Experiment B1: RoPE-3D algebra and group property.

Standard RoPE applies a per-pair 2D rotation R(m*theta_k) to head-dim pairs
(d -> d/2 independent SO(2) rotations). RoPE-3D generalizes this to per-triplet
SO(3) rotations parameterized by an antisymmetric matrix

      M = [[0,    M_xy,  M_xz],
           [-M_xy, 0,    M_yz],
           [-M_xz, -M_yz, 0  ]]

with rotation matrix R(m) = exp(m * M). Under the standard hat-map,
exp(m * M) is a rotation by angle m * |M|_scalar about axis
(-M_yz, M_xz, -M_xy)/|M|_scalar where

      |M|_scalar = sqrt(M_xy^2 + M_xz^2 + M_yz^2)         (eq. 13)

i.e. the framework's scalar torsion M is literally the rotation rate per
position step. The Newtonian limit M_xz = M_yz = 0 collapses to a 2D rotation
about the z-axis on the (x,y) plane -- standard RoPE.

This script verifies:
  1. Group property R(m+n) = R(m) R(n) to machine precision.
  2. Orthogonality R(m)^T R(m) = I.
  3. Relative-position invariance: <R(m)q, R(n)k> = <q, R(n-m)k>.
     This is the property that makes RoPE work for relative attention.
  4. Newtonian-limit reduction: RoPE-3D with M_xz=M_yz=0 acts as RoPE-2D on
     the (x,y) sub-plane and leaves z invariant.
  5. Attention-vs-relative-position curves for 2D and 3D heads, side by side.
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
# RoPE-2D (canonical) and RoPE-3D constructions.
# ---------------------------------------------------------------------------

def rope2d_rotation(m: float, theta: float) -> torch.Tensor:
    """Standard 2D RoPE rotation by angle m*theta."""
    c, s = np.cos(m * theta), np.sin(m * theta)
    return torch.tensor([[c, -s], [s, c]])


def hat(M_xy: float, M_xz: float, M_yz: float) -> torch.Tensor:
    """Antisymmetric 3x3 from the three independent components of M."""
    return torch.tensor([
        [0.0,  M_xy,  M_xz],
        [-M_xy, 0.0,  M_yz],
        [-M_xz, -M_yz, 0.0],
    ])


def rope3d_rotation(m: float, M: torch.Tensor) -> torch.Tensor:
    """SO(3) rotation R(m) = exp(m * M) via Rodrigues' formula.

    Works for any antisymmetric 3x3 M. Numerically more stable than the
    naive matrix_exp for small/large m because the axis-angle decomposition
    isolates the trigonometric component analytically.
    """
    M_xy, M_xz, M_yz = M[0, 1].item(), M[0, 2].item(), M[1, 2].item()
    M_scalar = float(np.sqrt(M_xy ** 2 + M_xz ** 2 + M_yz ** 2))  # eq. (13)
    if M_scalar < 1e-15:
        return torch.eye(3, dtype=M.dtype)
    angle = m * M_scalar
    M_unit = M / M_scalar
    return (
        torch.eye(3, dtype=M.dtype)
        + np.sin(angle) * M_unit
        + (1.0 - np.cos(angle)) * (M_unit @ M_unit)
    )


# ---------------------------------------------------------------------------
# Property checks.
# ---------------------------------------------------------------------------

def check_group_property(rotation_fn, label: str, samples: int = 200) -> dict:
    """Verify R(m+n) = R(m) R(n) for many random (m, n) pairs."""
    errs = []
    for _ in range(samples):
        m = float(np.random.uniform(-50, 50))
        n = float(np.random.uniform(-50, 50))
        Rmn = rotation_fn(m + n)
        RmRn = rotation_fn(m) @ rotation_fn(n)
        errs.append(float(torch.linalg.norm(Rmn - RmRn).item()))
    errs = np.array(errs)
    return {
        "label": label,
        "max_err": float(errs.max()),
        "mean_err": float(errs.mean()),
        "median_err": float(np.median(errs)),
    }


def check_orthogonality(rotation_fn, label: str, samples: int = 200) -> dict:
    errs = []
    for _ in range(samples):
        m = float(np.random.uniform(-50, 50))
        R = rotation_fn(m)
        I = torch.eye(R.shape[0], dtype=R.dtype)
        errs.append(float(torch.linalg.norm(R.T @ R - I).item()))
    errs = np.array(errs)
    return {
        "label": label,
        "max_err": float(errs.max()),
        "mean_err": float(errs.mean()),
    }


def check_relative_position_property(rotation_fn, dim: int, label: str,
                                     samples: int = 200) -> dict:
    """Verify <R(m)q, R(n)k> == <q, R(n-m)k>.

    For orthogonal R, <R(m)q, R(n)k> = <q, R(m)^T R(n) k> = <q, R(n-m) k>.
    This is the property that makes RoPE encode purely relative position in
    the attention dot product.
    """
    errs = []
    for _ in range(samples):
        m = float(np.random.uniform(-30, 30))
        n = float(np.random.uniform(-30, 30))
        q = torch.randn(dim, dtype=torch.float64)
        k = torch.randn(dim, dtype=torch.float64)
        Rm = rotation_fn(m)
        Rn = rotation_fn(n)
        Rnm = rotation_fn(n - m)
        lhs = (Rm @ q) @ (Rn @ k)
        rhs = q @ (Rnm @ k)
        errs.append(float(abs(lhs - rhs).item()))
    errs = np.array(errs)
    return {
        "label": label,
        "max_err": float(errs.max()),
        "mean_err": float(errs.mean()),
    }


def check_newtonian_limit(samples: int = 50) -> dict:
    """RoPE-3D with M_xz = M_yz = 0 should act as RoPE-2D on (x,y) and leave z.

    Convention note: the framework places +M_xy in M[0,1] (eq. 10), which under
    the standard hat-map gives an axis vector pointing along -z. RoPE's theta
    is conventionally a +z rotation. They are the same SO(3) generator with
    opposite orientation, so the Newtonian reduction is M_xy_framework = -theta_RoPE.
    """
    errs_xy, errs_z = [], []
    for _ in range(samples):
        m = float(np.random.uniform(-20, 20))
        theta = float(np.random.uniform(0.1, 2.0))
        M3 = hat(-theta, 0.0, 0.0)  # framework sign convention vs RoPE's theta
        R3 = rope3d_rotation(m, M3)
        R2 = rope2d_rotation(m, theta)
        xy_block = R3[:2, :2]
        errs_xy.append(float(torch.linalg.norm(xy_block - R2).item()))
        errs_z.append(float(torch.linalg.norm(R3[2] - torch.tensor([0.0, 0.0, 1.0])).item()))
    return {
        "xy_block_max_err": float(max(errs_xy)),
        "z_axis_max_err": float(max(errs_z)),
    }


# ---------------------------------------------------------------------------
# Attention-vs-relative-position curves.
# ---------------------------------------------------------------------------

def attention_curve_2d(theta: float, deltas: np.ndarray, num_qk: int = 64):
    """Mean attention score <R(0)q, R(d)k> over random q, k pairs, vs delta."""
    q = torch.randn(num_qk, 2, dtype=torch.float64)
    k = torch.randn(num_qk, 2, dtype=torch.float64)
    scores = []
    for d in deltas:
        Rd = rope2d_rotation(d, theta)
        # <q_i, Rd k_i>
        rotated_k = (Rd @ k.T).T
        s = (q * rotated_k).sum(dim=1).mean()
        scores.append(float(s))
    return np.array(scores)


def attention_curve_3d(M: torch.Tensor, deltas: np.ndarray, num_qk: int = 64):
    q = torch.randn(num_qk, 3, dtype=torch.float64)
    k = torch.randn(num_qk, 3, dtype=torch.float64)
    scores = []
    for d in deltas:
        Rd = rope3d_rotation(d, M)
        rotated_k = (Rd @ k.T).T
        s = (q * rotated_k).sum(dim=1).mean()
        scores.append(float(s))
    return np.array(scores)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main():
    results = {}

    # 1. Group property + orthogonality on a fixed RoPE-2D head, several
    #    RoPE-3D heads with varying axes.
    theta = 0.5
    rope2d_fn = lambda m: rope2d_rotation(m, theta)

    rope3d_specs = [
        ("RoPE-3D (planar, M_xy only)", hat(0.5, 0.0, 0.0)),
        ("RoPE-3D (mixed)", hat(0.5, 0.3, 0.2)),
        ("RoPE-3D (off-plane heavy)", hat(0.1, 0.6, 0.4)),
        ("RoPE-3D (random)", hat(*np.random.uniform(-0.7, 0.7, 3).tolist())),
    ]

    results["group_property"] = [check_group_property(rope2d_fn, "RoPE-2D")]
    results["orthogonality"] = [check_orthogonality(rope2d_fn, "RoPE-2D")]
    results["relative_position"] = [
        check_relative_position_property(rope2d_fn, dim=2, label="RoPE-2D")
    ]

    for name, M in rope3d_specs:
        fn = lambda m, M=M: rope3d_rotation(m, M)
        results["group_property"].append(check_group_property(fn, name))
        results["orthogonality"].append(check_orthogonality(fn, name))
        results["relative_position"].append(
            check_relative_position_property(fn, dim=3, label=name)
        )

    # 2. Newtonian-limit reduction.
    results["newtonian_limit"] = check_newtonian_limit()

    # 3. Attention-vs-delta curves.
    deltas = np.linspace(0, 60, 600)
    curve_2d = attention_curve_2d(theta=0.5, deltas=deltas)
    # Planar Newtonian limit: M_xy_framework = -theta_RoPE (see check_newtonian_limit).
    curve_3d_planar = attention_curve_3d(hat(-0.5, 0.0, 0.0), deltas)
    curve_3d_mixed = attention_curve_3d(hat(0.3, 0.4, 0.3), deltas)
    curve_3d_offplane = attention_curve_3d(hat(0.05, 0.5, 0.4), deltas)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    axes[0].plot(deltas, curve_2d, label="RoPE-2D, theta=0.5", linewidth=1.6)
    axes[0].plot(deltas, curve_3d_planar, label="RoPE-3D, planar (Newtonian)",
                 linestyle="--", linewidth=1.2)
    axes[0].set_title("Newtonian limit: 3D-planar overlays 2D")
    axes[0].set_xlabel("relative position delta")
    axes[0].set_ylabel("mean <q, R(delta) k>")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(deltas, curve_2d, label="RoPE-2D", color="#888", linewidth=1.0)
    axes[1].plot(deltas, curve_3d_mixed, label="RoPE-3D mixed M=(0.3,0.4,0.3)",
                 linewidth=1.4)
    axes[1].plot(deltas, curve_3d_offplane,
                 label="RoPE-3D off-plane M=(0.05,0.5,0.4)", linewidth=1.4)
    axes[1].set_title("Off-plane M shifts oscillation axis")
    axes[1].set_xlabel("relative position delta")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Experiment B1: RoPE-2D vs RoPE-3D attention vs relative position")
    fig.tight_layout()
    fig_path = FIG_DIR / "b1_attention_vs_delta.png"
    fig.savefig(fig_path, dpi=140)
    print(f"  saved figure: {fig_path}")

    # 4. Persist numerical results.
    res_path = RES_DIR / "b1_results.json"
    res_path.write_text(json.dumps(results, indent=2))
    print(f"  saved results: {res_path}")

    # 5. Print a compact table.
    print("\n=== Group property R(m+n) = R(m)R(n) ===")
    for r in results["group_property"]:
        print(f"  {r['label']:42s}  max_err={r['max_err']:.2e}  "
              f"mean={r['mean_err']:.2e}")

    print("\n=== Orthogonality R^T R = I ===")
    for r in results["orthogonality"]:
        print(f"  {r['label']:42s}  max_err={r['max_err']:.2e}")

    print("\n=== Relative-position invariance <R(m)q, R(n)k> = <q, R(n-m)k> ===")
    for r in results["relative_position"]:
        print(f"  {r['label']:42s}  max_err={r['max_err']:.2e}  "
              f"mean={r['mean_err']:.2e}")

    print("\n=== Newtonian limit (M_xz=M_yz=0) ===")
    nl = results["newtonian_limit"]
    print(f"  xy-block matches RoPE-2D, max_err = {nl['xy_block_max_err']:.2e}")
    print(f"  z-axis invariance,         max_err = {nl['z_axis_max_err']:.2e}")


if __name__ == "__main__":
    main()
