"""
Experiment A2: invariant validation on a pretrained Mamba.

A1 showed that for LTI SSMs the framework's hyperbolic invariant does not
naturally transplant (I5 drifts), but the log-additive decomposition does
(I6 telescopes by construction). The substantive question for real LLMs is:
how much does Mamba's *selectivity* (input-dependent A_bar, B_bar, dt) break
the LTI closed-form closed-form prediction?

Test: load a pretrained Mamba, run a forward pass on real text, hook the
post-SSM hidden state at each layer, and compare against the LTI prediction
that would hold if A_bar / B_bar / dt were constants. The fitting residual is
the "selectivity-induced deviation from analytical jumpability."

Scope on this CPU box:
  - Use state-spaces/mamba-130m via HF transformers (CPU forward).
  - Single layer, short prompt -- enough to characterize whether the residual
    is small (selectivity is mild perturbation, jump is approximately valid)
    or large (selectivity is essential, jump is invalid).
  - Larger sweep (mamba-370m/790m, longer prompts, multiple layers) deferred
    to RunPod where mamba_ssm CUDA kernels work.

Sets HF_HOME locally so the download stays in the project tree.
"""

import json
import os
import sys
import time
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = OUT_DIR.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import numpy as np
import torch

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
np.random.seed(0)

FIG_DIR = OUT_DIR.parent / "figures"
RES_DIR = OUT_DIR.parent / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

MODEL_NAME = "state-spaces/mamba-130m-hf"


def main():
    from transformers import AutoTokenizer, MambaForCausalLM

    print(f"  loading {MODEL_NAME} (cache: {CACHE_DIR})")
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = MambaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        model.eval()
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        print("  (no internet, missing CUDA kernels, or other issue -- defer to RunPod)")
        sys.exit(1)
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Find the SSM blocks (Mamba mixer modules).
    layer_count = len(model.backbone.layers)
    cfg = model.config
    d_model = cfg.hidden_size
    d_state = cfg.state_size
    d_inner = cfg.intermediate_size
    print(f"  config: {layer_count} layers, d_model={d_model}, d_state={d_state}, d_inner={d_inner}")

    # Hook to capture per-step hidden state, A_bar, B_bar, C, dt at each step.
    captured = {}

    def hook_mixer(layer_idx):
        def fn(module, inputs, output):
            captured.setdefault("layer_outputs", {})[layer_idx] = output.detach().cpu()
        return fn

    target_layers = [0, layer_count // 2, layer_count - 1]
    handles = []
    for li in target_layers:
        h = model.backbone.layers[li].mixer.register_forward_hook(hook_mixer(li))
        handles.append(h)

    # Use a longer prompt so the LTI fit is not vacuously underdetermined.
    # We need T-1 >> d_model^2 for a free A, OR we restrict A's rank.
    # d_model=768 makes a free A intractably underdetermined for any
    # realistic T. We address this two ways:
    #   (a) longer prompt (~512 tokens) -> still T < d_model^2, but we then
    #   (b) fit a low-rank A (rank=d_state=16) on a TRAIN slice and report
    #       residual on a held-out TEST slice.
    # Self-contained prompt source. If sample_prompt.txt is present, use it;
    # otherwise fall back to the embedded text below (a public excerpt from
    # the framework's published abstract).
    prompt_path = OUT_DIR.parent / "sample_prompt.txt"
    if prompt_path.exists():
        text = prompt_path.read_text() * 4
    else:
        text = (
            "Standard N-body orbital propagators rely heavily on discrete "
            "differential stepping algorithms, which inherently suffer from "
            "numerical drift and fail to account for the continuous geometric "
            "stress exerted by gravitational potential. The 6D symplectic "
            "phase space introduces a 3D imaginary buffer to absorb gravitational "
            "torsion. By coupling a 3D real coordinate space with a 3D imaginary "
            "buffer, we establish a symplectic lock where spatial condensation "
            "and imaginary expansion conserve a strict hyperbolic invariant. "
        ) * 16
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    print(f"  input tokens: {enc.input_ids.shape[1]}")

    t0 = time.time()
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    print(f"  forward pass: {time.time() - t0:.2f}s")

    for h in handles:
        h.remove()

    hidden = out.hidden_states  # tuple of (1, T, d_model) per layer
    print(f"  collected hidden states: {len(hidden)} layers")

    # ----- Invariant evaluation -----
    # We treat the per-token residual stream h_t in (1, T, d_model) as the
    # "state trajectory" and evaluate the framework-shaped invariant I5 plus
    # a non-trivial geometric quantity: the relative growth of ||h_t||
    # compared to a hypothetical LTI A_bar fit at this layer.

    inv_log = {}
    for li in target_layers:
        h_seq = hidden[li + 1][0]  # (T, d_model), output of layer li
        T = h_seq.shape[0]
        norms = h_seq.norm(dim=-1).numpy()    # (T,)
        r0 = max(norms[0], 1e-30)

        alpha = norms / r0
        beta = r0 / np.maximum(norms, 1e-30)
        # M = step-residual norm, normalized
        residuals = (h_seq[1:] - h_seq[:-1]).norm(dim=-1).numpy()
        # pad to length T
        residuals_padded = np.concatenate([[0.0], residuals]) / r0

        I5 = (alpha + beta) ** 2 - residuals_padded ** 2
        # log-additive
        logE = np.log(np.maximum(norms ** 2, 1e-30))

        # LTI-fit residual with proper train/test split and rank restriction.
        # Fit a low-rank A of rank r = d_state on the train slice, evaluate
        # held-out residual on the test slice. This avoids the vacuous
        # "fit 590k params to 21 data points" trap.
        H = h_seq.numpy()
        rank = min(d_state, max(8, T // 8))  # stay well below T
        train_frac = 0.8
        if T >= 32:
            T_train = int(T * train_frac)
            X_tr, Y_tr = H[:T_train - 1], H[1:T_train]
            X_te, Y_te = H[T_train:-1], H[T_train + 1:]

            # Low-rank fit: A = U V^T with U, V in R^{d x r}.
            # Use truncated SVD of the lstsq solution: solve full lstsq, then
            # take rank-r truncation.
            A_full, _, _, _ = np.linalg.lstsq(X_tr, Y_tr, rcond=None)
            U_, s_, Vt_ = np.linalg.svd(A_full, full_matrices=False)
            A_lr = (U_[:, :rank] * s_[:rank]) @ Vt_[:rank, :]

            # Train residual (sanity).
            tr_resid = np.linalg.norm(Y_tr - X_tr @ A_lr, axis=-1)
            tr_base  = np.linalg.norm(Y_tr, axis=-1)
            tr_rel   = tr_resid / np.maximum(tr_base, 1e-30)

            # Held-out residual (the meaningful one).
            te_resid = np.linalg.norm(Y_te - X_te @ A_lr, axis=-1)
            te_base  = np.linalg.norm(Y_te, axis=-1)
            te_rel   = te_resid / np.maximum(te_base, 1e-30)

            # Also: trivial baseline. How well does "predict h_t = h_{t-1}"
            # do on the held-out set? If almost as good as A_lr, the LTI fit
            # is uninformative.
            triv_resid = np.linalg.norm(Y_te - X_te, axis=-1)
            triv_rel   = triv_resid / np.maximum(te_base, 1e-30)
        else:
            tr_rel = te_rel = triv_rel = np.array([np.nan])

        inv_log[li] = {
            "T": int(T),
            "rank_used": int(rank),
            "I5_drift_max": float(np.max(np.abs(I5 - I5[0]) / max(abs(I5[0]), 1e-30))),
            "I5_drift_median": float(np.median(np.abs(I5 - I5[0]) / max(abs(I5[0]), 1e-30))),
            "logE_drift_max": float(np.max(np.abs(logE - logE[0]) / max(abs(logE[0]), 1e-30))),
            "lti_train_residual_median": float(np.median(tr_rel)),
            "lti_test_residual_median": float(np.median(te_rel)),
            "lti_test_residual_max": float(np.max(te_rel)),
            "trivial_baseline_residual_median": float(np.median(triv_rel)),
            "lti_vs_trivial_ratio": float(np.median(te_rel) / max(np.median(triv_rel), 1e-30)),
        }

    res_path = RES_DIR / "a2_results.json"
    res_path.write_text(json.dumps({
        "model": MODEL_NAME,
        "config": {"d_model": d_model, "d_state": d_state, "d_inner": d_inner,
                   "n_layers": layer_count},
        "input_tokens": int(enc.input_ids.shape[1]),
        "per_layer": {str(k): v for k, v in inv_log.items()},
    }, indent=2))
    print(f"  saved results: {res_path}")

    print("\n=== Per-layer invariant drift on real Mamba (held-out test) ===")
    print(f"{'layer':>5s}  {'I5_drift':>10s}  {'logE_drift':>10s}  "
          f"{'LTI_tr':>8s}  {'LTI_te':>8s}  {'trivial_te':>10s}  {'LTI/trivial':>11s}")
    for li, info in inv_log.items():
        print(f"  {li:>3d}    "
              f"{info['I5_drift_max']:10.3e}  {info['logE_drift_max']:10.3e}  "
              f"{info['lti_train_residual_median']:8.3e}  "
              f"{info['lti_test_residual_median']:8.3e}  "
              f"{info['trivial_baseline_residual_median']:10.3e}  "
              f"{info['lti_vs_trivial_ratio']:11.3f}")

    print("\nInterpretation:")
    print("  LTI_tr < LTI_te -> generalization gap (overfit on train).")
    print("  LTI/trivial ratio: <1 means LTI fit beats 'predict h_{t-1}'; ~1 means uninformative.")
    print("  Small held-out LTI residual + ratio << 1 -> Mamba is approximately LTI in the residual stream.")


if __name__ == "__main__":
    main()
