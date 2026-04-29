"""
Experiment C1: hidden-state trajectory eccentricity as a model-failure diagnostic.

Hypothesis (Kalkaja, derived from the SSMP ensemble paper's e>=1 escape
threshold): the trajectory of hidden states across token positions in a
pretrained LM forms an "orbit" in d_model space, and the framework's
eccentricity

    e = (M_max - M_min) / (M_max + M_min)             (unification eq. 17)

with M_t = ||h_t - centroid|| separates bound (e < 1, model on-distribution)
from unbound (e -> 1 or runaway, model off-distribution and predictions
collapse).

Test design:
  - Pretrained Mamba-130m (already cached on the pod).
  - 6 prompt types covering an "OOD spectrum":
       natural prose, code, mathematical/formal text, repetition,
       random vocabulary words, garbage Unicode.
  - For each prompt, multiple lengths (64, 128, 256, 512).
  - For each (prompt, length, layer), compute eccentricity and correlate
    with per-position next-token cross-entropy.

Plots:
  - eccentricity heatmap (prompt-type x layer) at the longest length.
  - eccentricity vs running cross-entropy along the sequence (does the
    "going nuts" line up with high CE?).
"""

import json
import os
import time
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = OUT_DIR.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

FIG_DIR = OUT_DIR.parent / "figures"
RES_DIR = OUT_DIR.parent / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

MODEL_NAME = "state-spaces/mamba-130m-hf"

PROMPTS = {
    "natural": (
        "The history of language models begins long before the modern era of "
        "deep learning. Early statistical approaches counted word frequencies "
        "and conditional probabilities, building n-gram tables that captured "
        "local context but struggled to generalize. The introduction of neural "
        "networks gradually replaced explicit counting with learned embeddings, "
        "first in shallow networks and later in much deeper architectures. "
    ),
    "code": (
        "def attention(query, key, value, mask=None): "
        "scores = torch.matmul(query, key.transpose(-2, -1)) "
        "scores = scores / math.sqrt(query.size(-1)) "
        "if mask is not None: scores = scores.masked_fill(mask == 0, -1e9) "
        "weights = F.softmax(scores, dim=-1) "
        "return torch.matmul(weights, value), weights "
    ),
    "math": (
        "Let H be a Hilbert space and let A be a self-adjoint operator on H "
        "with discrete spectrum. The spectral theorem provides a decomposition "
        "of A in terms of its eigenvalues lambda_n and corresponding eigenvectors "
        "phi_n such that A = sum lambda_n times P_n, where each P_n is the "
        "orthogonal projector onto the eigenspace associated with lambda_n. "
    ),
    "repetition": (
        "the the the the the the the the the the the the the the the the the "
    ),
    "random_words": (
        "lighthouse jasmine paradox circuit ephemeral candelabra wisteria "
        "thunderclap obsidian heliotrope serendipity petrichor argyle hieroglyph "
        "kaleidoscope phosphorescent stalactite sycamore meadow saxophone harbor "
        "molasses isthmus zenith forsythia patchwork benevolence cinnamon table "
    ),
    "garbage": (
        "@#%^*&!??)) [[ }}_++=== ~~~^^*** ::// >>>>< xkxkxk @@@##$$$ ||||||| "
        "(()()) zzzzqqqq vvvvvvv 12839074 ./.,.,. **--**-- ;:;:;:; @!@!@!@! "
        "[]{}[]{} %%^&&^^ () () () () >> << >> << >> << @ # $ % ^ & * ! "
    ),
}


def compute_trajectory_geometry(H: np.ndarray):
    """For hidden states H of shape (T, d), compute geometric quantities.

    Returns dict with:
      - r_t:   per-position radial distance from centroid (T,)
      - r_max, r_min, r_mean
      - eccentricity (eq. 17)  e = (M_max - M_min) / (M_max + M_min)
      - alpha-style eccentricity using running max/min of r_t up to position t
        (so we can plot how it grows along the sequence)
      - e_running: (T,) array of running eccentricity at each position
    """
    centroid = H.mean(axis=0, keepdims=True)
    r = np.linalg.norm(H - centroid, axis=-1)
    r_max = float(r.max())
    r_min = float(r.min())
    r_mean = float(r.mean())
    e = (r_max - r_min) / max(r_max + r_min, 1e-30)

    # Running eccentricity: e_t = (max(r[:t+1]) - min(r[:t+1])) / (max + min)
    cummax = np.maximum.accumulate(r)
    cummin = np.minimum.accumulate(r)
    e_running = (cummax - cummin) / np.maximum(cummax + cummin, 1e-30)
    return {
        "r": r,
        "r_max": r_max,
        "r_min": r_min,
        "r_mean": r_mean,
        "eccentricity": float(e),
        "e_running": e_running,
    }


def per_position_xent(logits: torch.Tensor, target_ids: torch.Tensor):
    """logits: (T, V) on next-token prediction. target_ids: (T,) shifted-left.
    Returns array of length T-1 with per-position cross-entropy."""
    log_probs = F.log_softmax(logits[:-1], dim=-1)
    return -log_probs.gather(-1, target_ids[1:].unsqueeze(-1)).squeeze(-1)


def main():
    from transformers import AutoTokenizer, MambaForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    print(f"  loading {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MambaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
    model.eval()
    n_layers = len(model.backbone.layers)
    print(f"  layers: {n_layers}")

    LENGTHS = [64, 128, 256, 512]
    layer_indices = list(range(n_layers + 1))   # include the input embedding layer 0

    # results[prompt_tag][length] = {layer_idx -> geometry dict, "xent" -> per-pos CE}
    results = {tag: {} for tag in PROMPTS}

    for tag, text in PROMPTS.items():
        # Pad text by repetition if necessary so we can truncate to max length.
        text_long = (text + " ") * 32
        for L in LENGTHS:
            enc = tokenizer(text_long, return_tensors="pt", truncation=True, max_length=L).to(device)
            T = int(enc.input_ids.shape[1])
            if T < L * 0.5:
                continue
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            hidden = out.hidden_states          # (n_layers+1) of (1, T, d)
            xent = per_position_xent(out.logits[0], enc.input_ids[0])
            results[tag][L] = {
                "T": T,
                "xent_mean": float(xent.mean().item()),
                "xent_per_pos": xent.detach().cpu().numpy(),
                "by_layer": {},
            }
            for li in layer_indices:
                H = hidden[li][0].detach().cpu().to(torch.float32).numpy()
                geom = compute_trajectory_geometry(H)
                results[tag][L]["by_layer"][li] = {
                    "eccentricity": geom["eccentricity"],
                    "r_max": geom["r_max"],
                    "r_min": geom["r_min"],
                    "r_mean": geom["r_mean"],
                    "e_running": geom["e_running"],
                    "r_per_pos": geom["r"],
                }
            print(f"  {tag:>14s}  L={L:>4d}  T={T:>4d}  "
                  f"xent_mean={results[tag][L]['xent_mean']:.3f}  "
                  f"e@layer0={results[tag][L]['by_layer'][0]['eccentricity']:.3f}  "
                  f"e@final={results[tag][L]['by_layer'][n_layers]['eccentricity']:.3f}")

    # ---------- Plot 1: eccentricity heatmap (prompt x layer) at L=512 ----------
    L_focus = 512
    prompts_with = [t for t in PROMPTS if L_focus in results[t]]
    fig, ax = plt.subplots(figsize=(11, 4.0))
    matrix = np.array([
        [results[t][L_focus]["by_layer"][li]["eccentricity"] for li in layer_indices]
        for t in prompts_with
    ])
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(range(len(prompts_with)))
    ax.set_yticklabels(prompts_with)
    ax.set_xlabel(f"layer (0 = embedding, {n_layers} = final)")
    ax.set_title(f"C1: Hidden-state trajectory eccentricity per layer (L={L_focus}, Mamba-130m)")
    plt.colorbar(im, ax=ax, label="e = (r_max - r_min) / (r_max + r_min)")
    fig.tight_layout()
    fig_path = FIG_DIR / "c1_eccentricity_heatmap.png"
    fig.savefig(fig_path, dpi=140)
    print(f"  saved: {fig_path}")

    # ---------- Plot 2: running eccentricity vs running cross-entropy ----------
    final_layer = n_layers
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    cmap = plt.get_cmap("tab10")
    for k, t in enumerate(prompts_with):
        if L_focus not in results[t]:
            continue
        e_run = results[t][L_focus]["by_layer"][final_layer]["e_running"]
        xent = results[t][L_focus]["xent_per_pos"]
        axes[0].plot(e_run, label=t, color=cmap(k), linewidth=1.2)
        axes[1].plot(xent, label=t, color=cmap(k), linewidth=1.2)
    axes[0].axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="e = 1")
    axes[0].set_ylabel("running eccentricity (final layer)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].set_ylabel("per-position next-token CE")
    axes[1].set_xlabel("token position")
    axes[1].grid(alpha=0.3)
    axes[1].set_yscale("log")
    fig.suptitle(f"C1: running eccentricity vs running cross-entropy (Mamba-130m, L={L_focus})")
    fig.tight_layout()
    fig_path = FIG_DIR / "c1_running_e_vs_xent.png"
    fig.savefig(fig_path, dpi=140)
    print(f"  saved: {fig_path}")

    # ---------- Plot 3: prompt-type vs final-layer e and mean-CE scatter ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, t in enumerate(prompts_with):
        for L in LENGTHS:
            if L not in results[t]:
                continue
            e = results[t][L]["by_layer"][final_layer]["eccentricity"]
            ce = results[t][L]["xent_mean"]
            ax.scatter(e, ce, color=cmap(k), s=20 + 8 * (LENGTHS.index(L)),
                       label=f"{t}, L={L}" if L == LENGTHS[-1] else None)
    ax.set_xlabel("final-layer eccentricity e")
    ax.set_ylabel("mean cross-entropy")
    ax.set_yscale("log")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("C1: e vs mean CE -- does the bound/unbound transition predict failure?")
    fig.tight_layout()
    fig_path = FIG_DIR / "c1_e_vs_xent_scatter.png"
    fig.savefig(fig_path, dpi=140)
    print(f"  saved: {fig_path}")

    # ---------- Save numeric summary ----------
    summary = {tag: {} for tag in PROMPTS}
    for tag in PROMPTS:
        for L in LENGTHS:
            if L not in results[tag]:
                continue
            r = results[tag][L]
            summary[tag][L] = {
                "xent_mean": r["xent_mean"],
                "eccentricity_per_layer": [r["by_layer"][li]["eccentricity"]
                                           for li in layer_indices],
                "e_final": r["by_layer"][final_layer]["eccentricity"],
                "r_max_final": r["by_layer"][final_layer]["r_max"],
                "r_min_final": r["by_layer"][final_layer]["r_min"],
            }
    res_path = RES_DIR / "c1_results.json"
    res_path.write_text(json.dumps({
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "lengths": LENGTHS,
        "summary": summary,
    }, indent=2, default=str))
    print(f"  saved: {res_path}")

    # ---------- Print a compact final table ----------
    print("\n=== final-layer eccentricity vs mean CE, longest length ===")
    print(f"{'prompt':>14s}  {'L':>5s}  {'e_final':>8s}  {'r_max':>8s}  "
          f"{'r_min':>8s}  {'CE':>8s}")
    for tag in PROMPTS:
        for L in LENGTHS:
            if L not in summary[tag]:
                continue
            s = summary[tag][L]
            print(f"  {tag:>12s}  {L:>5d}  {s['e_final']:>8.4f}  "
                  f"{s['r_max_final']:>8.2f}  {s['r_min_final']:>8.2f}  "
                  f"{s['xent_mean']:>8.3f}")


if __name__ == "__main__":
    main()
