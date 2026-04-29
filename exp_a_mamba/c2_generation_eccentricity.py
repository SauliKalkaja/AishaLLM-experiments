"""
Experiment C2: autoregressive-generation eccentricity, bound vs escape regime.

C1 (forward-pass eccentricity) showed that on fixed prompts the strict
e>1 escape signature does not appear -- RMSNorm binds the trajectory
radially regardless of prompt OOD-ness. The next test is whether escape
shows up during *generation*, where each token feeds back into the next
and the model can drift into degenerate modes.

Two regimes per prompt:
  - BOUND:  low temperature (T=0.3), greedy-ish, model stays in distribution.
  - ESCAPE: high temperature (T=5.0), noise rules sampling, model drifts.
And a third, T=2.0, as the boundary case.

Tracked quantities along the generation trajectory:
  - r_t = ||h_t - centroid||                            (radial distance)
  - eccentricity (running, framework eq. 17)
  - step magnitude ||h_{t+1} - h_t||                    (kinetic indicator)
  - per-step prediction entropy of the next-token dist
  - "dark-side" decomposition: projection of h_t onto the top-K singular
    subspace of the LM head (PRIMARY) versus the orthogonal complement
    (BUFFER). The hypothesis is that escape regimes leak energy from
    PRIMARY into BUFFER.

Outputs:
  - figure: 4-panel per regime, position-on-x: r_t, step mag, entropy,
    dark-side fraction.
  - heatmap of running e vs (regime, step) at the final layer.
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
        "The history of language models begins with simple statistical methods. "
        "Early researchers counted word co-occurrences and built n-gram tables. "
        "Over time, deep learning replaced these with learned embeddings. "
    ),
    "garbage": (
        "@#%^*&!??)) [[ }}_++=== ~~~^^*** ::// >>>>< xkxkxk @@@##$$$ ||||||| "
    ),
}

TEMPERATURES = [0.3, 1.0, 2.0, 5.0]
GEN_LENGTH = 256
DARK_K = 16   # number of "primary" singular directions of the LM head


@torch.no_grad()
def generate_with_trace(model, tokenizer, prompt: str, temperature: float,
                         gen_length: int, device: str, primary_basis: torch.Tensor,
                         layer_indices):
    """Autoregressive generation with full per-step hidden-state capture.

    Returns dict of arrays:
      - tokens:        list of generated token ids
      - decoded_text:  full text including prompt
      - per_layer:     dict layer_idx -> (T, d) ndarray of hidden states
      - step_entropy:  (T,) per-step entropy of the next-token dist
      - primary_norm_per_layer, buffer_norm_per_layer: (T,) projections
        onto the head's top-K subspace and its orthogonal complement
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    prompt_len = int(input_ids.shape[1])

    # Initial forward pass on the prompt to seed cache.
    out = model(input_ids, use_cache=True, output_hidden_states=True)
    cache_params = out.cache_params
    hidden = out.hidden_states  # tuple of (1, prompt_len, d) per layer

    per_layer = {li: [hidden[li][0].detach().cpu().to(torch.float32).numpy()]
                 for li in layer_indices}
    step_entropies = []
    primary_norms = {li: [] for li in layer_indices}
    buffer_norms = {li: [] for li in layer_indices}

    next_logits = out.logits[0, -1]   # (V,)
    for step in range(gen_length):
        scaled = next_logits / max(temperature, 1e-6)
        log_probs = F.log_softmax(scaled, dim=-1)
        probs = log_probs.exp()
        ent = -(probs * log_probs).sum().item()
        step_entropies.append(ent)
        next_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)  # (1,1)

        out = model(next_id, cache_params=cache_params, use_cache=True,
                    output_hidden_states=True)
        cache_params = out.cache_params
        for li in layer_indices:
            h = out.hidden_states[li][0].detach().cpu().to(torch.float32).numpy()
            per_layer[li].append(h)
        next_logits = out.logits[0, -1]
        input_ids = torch.cat([input_ids, next_id], dim=1)

    # Stitch (T, d) per layer.
    out_per_layer = {}
    for li in layer_indices:
        H = np.concatenate(per_layer[li], axis=0)  # (T, d)
        out_per_layer[li] = H

    # Dark-side decomposition (computed all at once).
    Pb = primary_basis.detach().cpu().numpy()      # (d, K), columns orthonormal
    for li in layer_indices:
        H = out_per_layer[li]
        proj_primary = H @ Pb                       # (T, K)
        primary = proj_primary @ Pb.T               # (T, d), in primary subspace
        buffer = H - primary                        # orthogonal complement
        primary_norms[li] = np.linalg.norm(primary, axis=-1)
        buffer_norms[li] = np.linalg.norm(buffer, axis=-1)

    return {
        "tokens": input_ids[0].detach().cpu().numpy().tolist(),
        "decoded_text": tokenizer.decode(input_ids[0]),
        "per_layer": out_per_layer,
        "step_entropy": np.array(step_entropies),
        "primary_norm_per_layer": primary_norms,
        "buffer_norm_per_layer": buffer_norms,
        "prompt_len": prompt_len,
    }


def trajectory_geometry(H: np.ndarray):
    centroid = H.mean(axis=0, keepdims=True)
    r = np.linalg.norm(H - centroid, axis=-1)
    cummax = np.maximum.accumulate(r)
    cummin = np.minimum.accumulate(r)
    e_run = (cummax - cummin) / np.maximum(cummax + cummin, 1e-30)
    return r, e_run


def main():
    from transformers import AutoTokenizer, MambaForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MambaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
    model.eval()
    n_layers = len(model.backbone.layers)
    layer_indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]

    # Build primary basis from the LM head weights.
    # head.weight is (V, d); top-K right-singular vectors of head.weight^T are
    # the directions in d-space along which next-token probabilities vary the
    # most -- the "predictively important" subspace.
    W = model.lm_head.weight.detach().to(torch.float32).cpu()  # (V, d)
    # Top-K right-singular vectors of W.
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    primary_basis = Vt[:DARK_K].T.contiguous()       # (d, K), orthonormal columns
    print(f"  LM head: V={W.shape[0]}, d={W.shape[1]}, top-{DARK_K} singular values:")
    print(f"    {S[:DARK_K].tolist()}")

    # ---- Run all (prompt, temperature) generations ----
    results = {}
    for ptag, prompt in PROMPTS.items():
        results[ptag] = {}
        for T in TEMPERATURES:
            torch.manual_seed(42)   # same sampling RNG seed for fair comparison
            t0 = time.time()
            run = generate_with_trace(model, tokenizer, prompt, T, GEN_LENGTH,
                                       device, primary_basis, layer_indices)
            dt = time.time() - t0
            print(f"  {ptag:>10s}  T={T:>4.1f}  generated {len(run['tokens'])} tokens in {dt:.1f}s")
            print(f"    sample: {run['decoded_text'][-200:]!r}")
            results[ptag][T] = run

    # ---- Compute geometry per layer per generation ----
    final_layer = max(layer_indices)
    geom = {}
    for ptag in PROMPTS:
        geom[ptag] = {}
        for T in TEMPERATURES:
            r, e_run = trajectory_geometry(results[ptag][T]["per_layer"][final_layer])
            step_mag = np.linalg.norm(np.diff(results[ptag][T]["per_layer"][final_layer], axis=0), axis=-1)
            geom[ptag][T] = {"r": r, "e_running": e_run, "step_mag": step_mag}

    # ---- Plot 1: 4-panel per prompt, layered by temperature ----
    for ptag in PROMPTS:
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        cmap = plt.get_cmap("plasma")
        for k, T in enumerate(TEMPERATURES):
            run = results[ptag][T]
            color = cmap(k / max(1, len(TEMPERATURES) - 1))
            r = geom[ptag][T]["r"]
            e_run = geom[ptag][T]["e_running"]
            step_mag = geom[ptag][T]["step_mag"]
            ent = run["step_entropy"]
            primary = run["primary_norm_per_layer"][final_layer]
            buffer = run["buffer_norm_per_layer"][final_layer]
            total = primary + buffer
            buffer_frac = buffer / np.maximum(total, 1e-30)
            x = np.arange(len(r))
            axes[0].plot(x, r, color=color, label=f"T={T}")
            axes[1].plot(x, e_run, color=color, label=f"T={T}")
            axes[2].plot(x[1:], step_mag, color=color, label=f"T={T}")
            axes[3].plot(x, buffer_frac, color=color, label=f"T={T}")
        axes[0].set_ylabel("r_t = ||h_t - centroid||")
        axes[1].set_ylabel("running eccentricity")
        axes[1].axhline(1.0, color="red", linestyle="--", linewidth=0.7)
        axes[2].set_ylabel("step magnitude ||h_{t+1} - h_t||")
        axes[3].set_ylabel(f"||h_t in buffer|| / ||h_t||  (top-{DARK_K})")
        axes[3].set_xlabel("token position (incl. prompt)")
        for ax in axes:
            ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
        # Mark prompt boundary.
        prompt_len = results[ptag][TEMPERATURES[0]]["prompt_len"]
        for ax in axes:
            ax.axvline(prompt_len - 1, color="black", linestyle=":", linewidth=0.7,
                       label="prompt|gen")
        fig.suptitle(f"C2: bound vs escape generation, prompt='{ptag}', Mamba-130m, final layer")
        fig.tight_layout()
        fig_path = FIG_DIR / f"c2_{ptag}_trajectory.png"
        fig.savefig(fig_path, dpi=140)
        print(f"  saved: {fig_path}")
        plt.close(fig)

    # ---- Save numerical summary ----
    summary = {}
    for ptag in PROMPTS:
        summary[ptag] = {}
        for T in TEMPERATURES:
            r = geom[ptag][T]["r"]
            e_run = geom[ptag][T]["e_running"]
            step_mag = geom[ptag][T]["step_mag"]
            ent = results[ptag][T]["step_entropy"]
            buffer_frac = (results[ptag][T]["buffer_norm_per_layer"][final_layer] /
                           np.maximum(results[ptag][T]["buffer_norm_per_layer"][final_layer]
                                      + results[ptag][T]["primary_norm_per_layer"][final_layer], 1e-30))
            summary[ptag][T] = {
                "r_max": float(r.max()),
                "r_min": float(r.min()),
                "r_mean": float(r.mean()),
                "e_final_running": float(e_run[-1]),
                "step_mag_mean": float(step_mag.mean()),
                "step_mag_max": float(step_mag.max()),
                "entropy_mean": float(ent.mean()),
                "buffer_fraction_mean": float(buffer_frac.mean()),
                "buffer_fraction_final": float(buffer_frac[-1]),
                "decoded_tail": results[ptag][T]["decoded_text"][-200:],
            }
    res_path = RES_DIR / "c2_results.json"
    res_path.write_text(json.dumps({
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "gen_length": GEN_LENGTH,
        "dark_K": DARK_K,
        "summary": summary,
    }, indent=2, default=str))
    print(f"  saved: {res_path}")

    # ---- Compact final table ----
    print("\n=== generation eccentricity vs temperature, final layer ===")
    print(f"{'prompt':>8s}  {'T':>4s}  {'r_max':>7s}  {'e_run_T':>9s}  "
          f"{'step_mean':>10s}  {'ent_mean':>10s}  {'buf_frac_mean':>14s}")
    for ptag in PROMPTS:
        for T in TEMPERATURES:
            s = summary[ptag][T]
            print(f"  {ptag:>6s}  {T:>4.1f}  {s['r_max']:>7.2f}  "
                  f"{s['e_final_running']:>9.4f}  {s['step_mag_mean']:>10.3f}  "
                  f"{s['entropy_mean']:>10.3f}  {s['buffer_fraction_mean']:>14.4f}")


if __name__ == "__main__":
    main()
