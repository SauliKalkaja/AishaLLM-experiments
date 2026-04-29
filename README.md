# Experiments: Symplectic-Manifold Framework Applied to LLM-Adjacent Systems

This tree holds two empirical investigations of whether the 6D analytical-jump
framework (Kälkäjä, Symplectic Manifold Engines) transplants to LLM-class
architectures: standard transformers (RoPE-3D extension) and state-space
models (Mamba invariant search).

## Setup

- Python 3.14, torch 2.11.0+cu130 (CPU-only in this run; GPU is GTX 1070 CC 6.1
  which the cu13 wheel does not have kernels for — `cudaErrorNoKernelImageForDevice`).
- transformers 5.6.2, numpy 2.3.5, matplotlib 3.10.8, einops 0.8.2.

Reproducibility:

    cd /home/sale/AishaLLM/experiments
    python3 exp_b_rope3d/b1_rope3d_algebra.py
    python3 exp_a_mamba/a1_ssm_invariants.py
    python3 exp_a_mamba/a2_pretrained_mamba.py    # downloads mamba-130m

All seeds are pinned. Outputs land in `figures/` and `results/`.

## Experiment B — RoPE-3D extension

### B1: algebra and group property — DONE

Standard RoPE applies a per-pair 2D rotation `R(m·θ)`. RoPE-3D extends this to
per-triplet SO(3) rotation `R(m) = exp(m·M)`, where `M` is the framework's
antisymmetric torsion tensor (eq. 10 of unification paper):

    M = [[0,    M_xy,  M_xz],
         [-M_xy, 0,    M_yz],
         [-M_xz, -M_yz, 0  ]]

The framework's scalar torsion `|M| = √(M_xy² + M_xz² + M_yz²)` (eq. 13) maps
exactly to RoPE's rotation frequency θ — angle per position step.

**Numerical verification** (machine precision, ~1e-15):

| Property | RoPE-2D | RoPE-3D (planar) | RoPE-3D (mixed) | RoPE-3D (off-plane) |
|---|---|---|---|---|
| Group `R(m+n)=R(m)R(n)` | 5e-15 | 5e-15 | 5e-15 | 1e-14 |
| Orthogonality `RᵀR=I` | 3e-16 | 4e-16 | 8e-16 | 1e-15 |
| Relative-position `⟨R(m)q, R(n)k⟩=⟨q, R(n−m)k⟩` | 4e-15 | 6e-15 | 8e-15 | 2e-14 |
| Newtonian limit (`M_xz=M_yz=0` → standard RoPE) | n/a | matches at 2e-16 | n/a | n/a |

**Key algebraic mappings** (with sign convention `M_xy_framework = −θ_RoPE`):
- Framework "scalar-M (Newtonian, planar)" ↔ standard 2D RoPE.
- Framework "full-tensor-M (relativistic, off-plane)" ↔ SO(3) RoPE with three
  independent rotational degrees of freedom per triplet.

The third rotational DoF is structurally available, doesn't break any RoPE
invariant, and offers a physical-regime knob for attention positional
encoding. Whether it improves a downstream task is **B2** (training).

Output: `figures/b1_attention_vs_delta.png`, `results/b1_results.json`.

## Experiment A — Mamba invariant search

### A1: synthetic SSM — DONE

For five SSM configurations (random-stable, near-orthogonal, HiPPO-real,
complex-block-diagonal Mamba-S4-style, marginal-stability), six candidate
invariants were tracked over T=1000-step trajectories:

| Invariant | What it tests | Final-step relative drift, median over 16 runs |
|---|---|---|
| I1: ‖h‖² | naive state energy | 0.3 to 48 across configs |
| I2: hᵀPh | Lyapunov quadratic | 0.4 to 48 |
| I3: I2 + cum‖u‖² | state + supplied energy | 3.6 to 48 |
| I4: I2 + passivity gap | passivity-style | 89 to 157 (worst) |
| I5: framework-port `(α+β)²−M²` | direct transplant of `(α_s+β_s)²−M²=4` | 0.013 to 11.7 |
| I6: log-additive `log‖h‖² − Σ log growth` | telescoping log decomposition | **3e-16 to 1e-15** |

**Result.** I5 (the direct transplant of the framework's hyperbolic invariant)
**does not conserve** in SSM dynamics. I6 conserves at machine precision but
**only as a telescoping identity** — not a discovered conservation law,
though it does confirm that the framework's "multiplication ↔ summation"
property has a clean SSM analogue.

**Substantive interpretation.** For LTI SSMs the analytical-jump structure
already exists in classical form: diagonalization `A = V Λ V⁻¹` gives
`h(t) = V exp(tΛ) V⁻¹ h(0)` plus a closed-form input correction. This is the
LTI-systems-theory equivalent of the framework's `α_s` scaling × Keplerian
anchor. Parallel-scan implementations of S4/Mamba already exploit this. The
framework's `(α, β, M)` triplet maps onto `(eigenbasis V, exp(tΛ), residual
norm)`.

The interesting question is whether **selectivity** (Mamba's input-dependent
A_bar) breaks this. That's A2.

Output: `figures/a1_invariant_drift.png`, `results/a1_results.json`.

### A2: pretrained Mamba — DONE (v3, hardened)

**Methodology evolution.**
- v1 used unregularized lstsq on a 590k-param `A` with ~21 data points →
  vacuously perfect fit, told us nothing.
- v2 added a held-out split within the same prompt and rank constraint, but
  the result was numerically fragile (CPU vs GPU flipped the headline) and
  within-prompt test/train tokens were highly autocorrelated.
- v3 (current) uses ridge regression with a λ sweep, averages across 5 prompts
  spanning different domains, and adds a **leave-one-prompt-out cross-prompt
  generalization test** as the rigorous metric.

Loaded `state-spaces/mamba-130m-hf` (24 layers, d_model=768, d_state=16,
d_inner=1536), ran a 512-token prompt, hooked post-mixer residual states at
layers 0, 12, 23. Fitted a low-rank `A` (rank=d_state=16) on the first 80%
of the prompt and evaluated the held-out residual on the last 20%, with a
"predict h_t = h_{t-1}" trivial baseline as control.

**Methodology note.** A first run produced a misleading 1e-7 LTI residual
because a free 768×768 `A` fit to ~21 data points is vacuously perfect.
Re-ran with a longer prompt, low-rank constraint on `A`, and held-out test
slice. This is the correct test.

**Final v3 result (RTX 4090, 5-prompt leave-one-out cross-validation):**

| Layer | Within-prompt test | **Cross-prompt test** | Trivial baseline | **Cross/trivial ratio** |
|---|---|---|---|---|
| 0 | 0.014 | **0.94 ± 0.03** | 1.26 | 0.745 |
| 6 | 0.016 | **0.78 ± 0.03** | 0.93 | 0.825 |
| 12 | 0.019 | **0.64 ± 0.04** | 0.73 | 0.848 |
| 18 | 0.041 | **0.67 ± 0.02** | 0.80 | 0.842 |
| 23 | 0.080 | **0.82 ± 0.07** | 0.95 | 0.892 |

**Reading the numbers.** Within-prompt test residual is 1–9% — looks like
clean LTI fit. Cross-prompt test residual jumps to 64–94%, ratio to trivial
is 0.74–0.89. The within-prompt result was capturing temporal
autocorrelation in a single sequence, not LTI generalization. Across truly
held-out prompts, a single fitted `A` is only marginally better than just
predicting `h_t = h_{t-1}`. Standard deviations are small (≤ 0.07
everywhere), so the conclusion is robust.

**Conclusion (cross-domain).** Mamba's residual stream is **not LTI**.
Selectivity (input-dependent A_bar, B_bar, dt) is structurally essential,
not perturbative. Direct application of the analytical-jump framework does
not yield O(1) end-to-end Mamba inference. The framework's hyperbolic
invariant `(α+β)² − M²` also does not conserve on real Mamba states (I5
drift 0.4–1.2 across layers).

**Methodological note worth keeping.** Within-prompt train/test splits on
sequence models are unreliable — adjacent positions in one sequence share
prompt-conditional structure, so any reasonable model fits a within-prompt
holdout. The decisive test is leave-one-prompt-out (or leave-one-document-out
for longer-form models). When the framework is later applied to other
sequence architectures, this is the metric to watch.

**Caveats still open.** Testing on residual stream rather than the mixer's
internal SSM state. Selectivity might live in the `(B_bar, dt)` projection,
and the inner SSM might still be approximately LTI. Larger Mamba sizes
(370m / 790m / 2.8b) on this RTX 4090 would tighten the conclusion further
but would not change its sign.

Output: `results/a2_results.json`.

### B2 / RunPod scaling — PENDING

- B2: tiny GPT trained from scratch with RoPE-2D vs RoPE-3D on a synthetic
  length-extrapolation task. CPU-feasible at ~2-3M params, but cleaner on
  GPU.
- Larger Mamba checkpoints (mamba-370m, mamba-790m, mamba-2.8b) for A2 sweep.
- Both deferred to RunPod handoff.
