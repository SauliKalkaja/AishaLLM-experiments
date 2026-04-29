"""
Experiment A2 (v2): hardened LTI test on a pretrained Mamba.

Earlier v1 used lstsq+SVD-truncate, which gave very different numbers on CPU
vs GPU because the small-singular-value cutoff was sensitive to numerical
noise. v2 fixes this:

  1. Ridge regression  A = (X^T X + lambda I)^-1 X^T Y    (stable, no SVD).
  2. Lambda swept over a grid; report results at each lambda.
  3. Multiple prompts (5 different domains) -- average and std reported.
  4. Train/test split per prompt; the test residual is the meaningful metric.
  5. Trivial baseline ("predict h_t = h_{t-1}") for control.

Headline question: at each layer, is the residual stream well-approximated
by a low-effective-rank LTI map A? If yes (small held-out residual, ratio
to trivial baseline << 1), the analytical-jump structure is approximately
valid for that layer.
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

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

FIG_DIR = OUT_DIR.parent / "figures"
RES_DIR = OUT_DIR.parent / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

MODEL_NAME = "state-spaces/mamba-130m-hf"

PROMPTS = [
    # Scientific.
    ("scientific",
     "Standard N-body orbital propagators rely heavily on discrete differential "
     "stepping algorithms, which inherently suffer from numerical drift. The 6D "
     "symplectic phase space introduces a 3D imaginary buffer to absorb gravitational "
     "torsion. By coupling a 3D real coordinate space with a 3D imaginary buffer, we "
     "establish a symplectic lock where spatial condensation and imaginary expansion "
     "conserve a strict hyperbolic invariant. "),
    # Code-like.
    ("code",
     "def quicksort(arr): "
     "if len(arr) <= 1: return arr "
     "pivot = arr[len(arr) // 2] "
     "left = [x for x in arr if x < pivot] "
     "middle = [x for x in arr if x == pivot] "
     "right = [x for x in arr if x > pivot] "
     "return quicksort(left) + middle + quicksort(right). "
     "The complexity is O(n log n) on average and O(n squared) in the worst case. "),
    # Casual / conversational.
    ("casual",
     "So I was walking the dog this morning and there was this guy across the street "
     "with the most ridiculous hat, like one of those big floppy beach hats but it was "
     "raining, and I just couldn't stop laughing. The dog didn't even notice, she was "
     "too busy sniffing every single tree on the block, twice. We were out for almost "
     "an hour because of it. "),
    # News.
    ("news",
     "Markets closed lower on Tuesday as investors weighed mixed signals from the "
     "central bank's quarterly statement. Energy stocks led the decline, with crude "
     "oil futures dropping nearly three percent on reports of higher than expected "
     "inventory builds. Technology shares pared earlier gains in the final hour of "
     "trading following weaker forward guidance from a major chip maker. "),
    # Random text (low-information control).
    ("random_words",
     "table cloud algorithm pencil mountain river bicycle quantum giraffe poem "
     "circuit harbor velvet meadow saxophone lighthouse cinnamon paradox jasmine "
     "obsidian thunderclap molasses forsythia patchwork heliotrope serendipity "
     "candelabra ephemeral wisteria hieroglyph zenith argyle sycamore phosphorescent "
     "petrichor isthmus benevolence stalactite kaleidoscope. "),
]


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Solve  A = argmin ||Y - X A||^2 + lam * ||A||_F^2.

    Closed form:  A = (X^T X + lam I)^-1 X^T Y. Stable for any lam > 0.
    """
    d = X.shape[1]
    XtX = X.T @ X
    XtY = X.T @ Y
    return np.linalg.solve(XtX + lam * np.eye(d), XtY)


def evaluate_layer(H: np.ndarray, lambdas):
    """Per-layer LTI-fit evaluation with ridge regression + train/test split.

    Returns a dict: per-lambda train/test residuals and the trivial baseline.
    """
    T, d = H.shape
    if T < 32:
        return None
    T_train = int(T * 0.8)
    X_tr, Y_tr = H[:T_train - 1], H[1:T_train]
    X_te, Y_te = H[T_train:-1], H[T_train + 1:]

    base_te = np.linalg.norm(Y_te, axis=-1)
    triv_te = np.linalg.norm(Y_te - X_te, axis=-1)
    triv_rel = float(np.median(triv_te / np.maximum(base_te, 1e-30)))

    out = {"trivial_test_resid_median": triv_rel, "by_lambda": {}}
    base_scale = float(np.trace(X_tr.T @ X_tr) / d)
    for lam_factor in lambdas:
        lam = lam_factor * base_scale
        A = ridge_fit(X_tr, Y_tr, lam)
        # nuclear norm as a soft "effective rank" proxy
        sv = np.linalg.svd(A, compute_uv=False)
        eff_rank = float(sv.sum() / max(sv.max(), 1e-30))

        tr_resid = np.linalg.norm(Y_tr - X_tr @ A, axis=-1)
        tr_base = np.linalg.norm(Y_tr, axis=-1)
        te_resid = np.linalg.norm(Y_te - X_te @ A, axis=-1)
        out["by_lambda"][lam_factor] = {
            "lambda": float(lam),
            "effective_rank_nuclear": eff_rank,
            "train_resid_median": float(np.median(tr_resid / np.maximum(tr_base, 1e-30))),
            "test_resid_median": float(np.median(te_resid / np.maximum(base_te, 1e-30))),
            "test_vs_trivial_ratio": float(np.median(te_resid / np.maximum(base_te, 1e-30)) /
                                            max(triv_rel, 1e-30)),
        }
    return out


def main():
    from transformers import AutoTokenizer, MambaForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    print(f"  loading {MODEL_NAME}")
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = MambaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
        model.eval()
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"  loaded in {time.time() - t0:.1f}s")

    layer_count = len(model.backbone.layers)
    target_layers = [0, layer_count // 4, layer_count // 2, 3 * layer_count // 4, layer_count - 1]
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    print(f"  layers: {layer_count}, targeting {target_layers}")
    print(f"  lambda factors: {lambdas}")

    # Per-prompt collection: per_layer[layer_idx] -> list of evaluate_layer dicts.
    per_layer = {li: [] for li in target_layers}
    framework_drift = {li: [] for li in target_layers}

    for tag, text in PROMPTS:
        text_extended = (text + " ") * 8
        enc = tokenizer(text_extended, return_tensors="pt", truncation=True, max_length=512).to(device)
        T = int(enc.input_ids.shape[1])
        print(f"  prompt '{tag}': {T} tokens")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states  # tuple len=layer_count+1, each (1, T, d)
        for li in target_layers:
            H = hidden[li + 1][0].detach().cpu().to(torch.float32).numpy()
            res = evaluate_layer(H, lambdas)
            if res is not None:
                per_layer[li].append({"prompt": tag, **res})

            # Framework I5 drift on the residual stream.
            norms = np.linalg.norm(H, axis=-1)
            r0 = max(norms[0], 1e-30)
            alpha = norms / r0
            beta = r0 / np.maximum(norms, 1e-30)
            res_steps = np.concatenate([[0.0],
                np.linalg.norm(H[1:] - H[:-1], axis=-1)]) / r0
            I5 = (alpha + beta) ** 2 - res_steps ** 2
            i5_drift = float(np.max(np.abs(I5 - I5[0]) / max(abs(I5[0]), 1e-30)))
            framework_drift[li].append({"prompt": tag, "I5_drift_max": i5_drift})

    # ----- Aggregate across prompts.
    print("\n=== A2 v2: ridge-fit LTI test, mean across 5 prompts ===")
    print(f"{'layer':>5s} {'lambda':>10s} {'eff_rank':>10s} "
          f"{'train':>8s} {'test':>8s} {'trivial':>8s} {'ratio':>8s} {'I5_drift':>10s}")
    summary = {}
    for li in target_layers:
        summary[li] = {"by_lambda": {}}
        i5_drifts = [d["I5_drift_max"] for d in framework_drift[li]]
        summary[li]["I5_drift_mean"] = float(np.mean(i5_drifts))
        summary[li]["I5_drift_std"] = float(np.std(i5_drifts))

        # Average trivial across prompts.
        trivs = [r["trivial_test_resid_median"] for r in per_layer[li]]
        triv_mean = float(np.mean(trivs))

        for lam in lambdas:
            tr_means = [r["by_lambda"][lam]["train_resid_median"] for r in per_layer[li]]
            te_means = [r["by_lambda"][lam]["test_resid_median"] for r in per_layer[li]]
            ratios = [r["by_lambda"][lam]["test_vs_trivial_ratio"] for r in per_layer[li]]
            ranks = [r["by_lambda"][lam]["effective_rank_nuclear"] for r in per_layer[li]]
            tr_m = float(np.mean(tr_means))
            te_m = float(np.mean(te_means))
            ratio_m = float(np.mean(ratios))
            rank_m = float(np.mean(ranks))
            te_s = float(np.std(te_means))
            summary[li]["by_lambda"][lam] = {
                "train_mean": tr_m, "test_mean": te_m, "test_std": te_s,
                "trivial_mean": triv_mean, "vs_trivial_ratio_mean": ratio_m,
                "effective_rank_nuclear_mean": rank_m,
            }
            if lam in (1e-3, 1e-1, 1.0):  # print a few
                print(f"  {li:>3d} {lam:>10.1e} {rank_m:>10.2f} "
                      f"{tr_m:>8.3f} {te_m:>8.3f} {triv_mean:>8.3f} {ratio_m:>8.3f} "
                      f"{summary[li]['I5_drift_mean']:>10.3e}")
        # Best lambda by test residual.
        best_lam, best = min(summary[li]["by_lambda"].items(),
                             key=lambda kv: kv[1]["test_mean"])
        summary[li]["best_lambda"] = best_lam
        print(f"      best lambda: {best_lam:.1e}  "
              f"test={best['test_mean']:.3f} +/- {best['test_std']:.3f}  "
              f"vs trivial={best['trivial_mean']:.3f}  "
              f"ratio={best['vs_trivial_ratio_mean']:.3f}")

    res_path = RES_DIR / "a2_results.json"
    res_path.write_text(json.dumps({
        "model": MODEL_NAME,
        "device": device,
        "n_prompts": len(PROMPTS),
        "layers_tested": target_layers,
        "summary": summary,
    }, indent=2, default=str))
    print(f"\n  saved results: {res_path}")

    print("\nInterpretation:")
    print("  ratio < 1  => low-rank LTI fit beats 'predict h_{t-1}'")
    print("  ratio < 0.5 means LTI captures most of the residual-stream dynamics")
    print("  ratio ~ 1  => LTI fit is uninformative")
    print("  small test_std across prompts => result is robust")


if __name__ == "__main__":
    main()
