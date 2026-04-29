"""
Experiment C3: probing the "dark-side" buffer subspace for recoverable information.

C2 showed 74-91% of Mamba's final-layer hidden state lives in the orthogonal
complement of the LM head's top-K singular subspace. The user's hypothesis:
real predictive information is sitting there, hidden from the LM head, and
recoverable if we know how to read it. This script tests that directly.

Procedure:
  1. Pretrained Mamba-130m on a corpus of natural English prompts.
  2. Build the orthonormal basis of the LM head's top-K singular directions
     (PRIMARY) and its orthogonal complement (BUFFER, dim d-K).
  3. For each token position t, decompose h_t = h_t_PRIMARY + h_t_BUFFER.
  4. Train two linear probes (cross-entropy, weight-decayed) to predict
     the next token id from each subspace separately.
  5. Held-out evaluation: top-1, top-5, cross-entropy. Compare to the
     full LM head (ground-truth oracle) and a uniform-random baseline.
  6. Per-token: how often does the BUFFER probe predict correctly when
     the LM head is wrong? That number, if non-trivial, validates the
     "information lives on the dark side" claim.

Sweeps K in {4, 16, 64, 256} to see how the primary/buffer information
splits as we shrink/grow the primary subspace.
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
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

FIG_DIR = OUT_DIR.parent / "figures"
RES_DIR = OUT_DIR.parent / "results"

MODEL_NAME = "state-spaces/mamba-130m-hf"

# Long natural-text corpus assembled from a mix of domains.
TRAIN_PROMPTS = [
    "The history of language models begins with simple statistical methods. "
    "Early researchers counted word co-occurrences and built n-gram tables that "
    "captured local context but struggled to generalize. Over time, neural networks "
    "replaced explicit counting with learned embeddings, first in shallow architectures "
    "and later in much deeper transformers and state-space models. ",

    "In classical mechanics, Newton's laws describe how forces produce motion. "
    "The principle of least action provides a unifying framework: a particle traverses "
    "the path that minimizes the integral of its Lagrangian over time. Hamilton's "
    "formulation reframes this in phase space, where each point encodes both position "
    "and momentum. Symplectic geometry then becomes the natural language of the theory. ",

    "Photosynthesis converts light into chemical energy through a cascade of reactions. "
    "In the light-dependent stage, chlorophyll absorbs photons and drives electron "
    "transport across thylakoid membranes. The resulting proton gradient powers ATP "
    "synthesis, which in turn fuels carbon fixation in the Calvin cycle. ",

    "Markets closed lower on Tuesday as investors weighed signals from the central bank's "
    "quarterly statement. Energy stocks led the decline, with crude oil futures dropping "
    "nearly three percent on reports of higher than expected inventory builds. Technology "
    "shares pared earlier gains in the final hour after weaker forward guidance from a chip maker. ",

    "She walked through the old city late in the afternoon, when the light fell low across "
    "the cobblestones and the air carried the faint smell of bread from a bakery she could "
    "no longer find. Cafes were filling up. Every face looked half-familiar. ",

    "def attention(query, key, value, mask=None): "
    "scores = torch.matmul(query, key.transpose(-2, -1)) "
    "scores = scores / math.sqrt(query.size(-1)) "
    "if mask is not None: scores = scores.masked_fill(mask == 0, -1e9) "
    "weights = F.softmax(scores, dim=-1) "
    "return torch.matmul(weights, value), weights ",

    "The mitochondrion is often described as the powerhouse of the cell, but its full role "
    "is far broader. It is the central hub of metabolism, signaling, calcium homeostasis, "
    "and apoptosis. Its double membrane structure separates the matrix from the intermembrane "
    "space, and the inner membrane folds into cristae that house the electron transport chain. ",

    "An election is fundamentally a coordination problem. Each voter must form expectations "
    "about how others will vote, and how their own vote will combine with theirs to produce a "
    "collective outcome. Strategic voting, voter turnout, and the structure of the ballot all "
    "interact to shape the result. ",
]

EVAL_PROMPTS = [
    "The development of integrated circuits transformed every industry. From the first "
    "transistors etched onto silicon to modern systems-on-chip with billions of components, "
    "scaling has driven exponential improvements in compute density. Moore's law captured "
    "this trend for decades, though physical limits eventually slowed it. ",

    "The Pacific Ocean covers more area than all of the planet's landmasses combined. Its "
    "depths host an enormous diversity of life, much of which remains undescribed. Currents "
    "flow across thousands of kilometers, transporting heat from equatorial waters toward "
    "higher latitudes and shaping continental climate. ",

    "import numpy as np "
    "from scipy.linalg import svd, qr "
    "def project_subspace(X, k): "
    "U, S, Vt = svd(X, full_matrices=False) "
    "return X @ Vt[:k].T @ Vt[:k] ",
]

K_VALUES = [4, 16, 64, 256]


def collect_states(model, tokenizer, prompts, device, max_length=512, layer="final"):
    """Run prompts through model, collect (final-layer h_t, next_token_id) pairs.

    Returns:
        H:    (N, d) tensor of hidden states
        targets: (N,) tensor of next-token ids
    """
    Hs, targets_list = [], []
    n_layers = len(model.backbone.layers)
    layer_idx = n_layers if layer == "final" else int(layer)
    for p in prompts:
        text = (p + " ") * 4
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        T = enc.input_ids.shape[1]
        if T < 8:
            continue
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer_idx][0]   # (T, d)
        ids = enc.input_ids[0]
        # Pair h_t with the next token id (t+1).
        Hs.append(h[:-1].detach().to(torch.float32).cpu())
        targets_list.append(ids[1:].detach().cpu())
    H = torch.cat(Hs, dim=0)
    targets = torch.cat(targets_list, dim=0)
    return H, targets


def train_probe(features: torch.Tensor, targets: torch.Tensor, vocab: int,
                test_features: torch.Tensor, test_targets: torch.Tensor,
                device: str, epochs: int = 200, lr: float = 1e-2, weight_decay: float = 1e-4,
                batch_size: int = 1024):
    """Train a linear probe (no bias) features -> logits."""
    feat_dim = features.shape[1]
    probe = nn.Linear(feat_dim, vocab, bias=False).to(device)
    nn.init.normal_(probe.weight, std=0.01)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    features = features.to(device)
    targets = targets.to(device)
    test_features = test_features.to(device)
    test_targets = test_targets.to(device)
    N = features.shape[0]
    best_test_ce = float("inf")
    best_state = None
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            logits = probe(features[idx])
            loss = F.cross_entropy(logits, targets[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            with torch.no_grad():
                test_logits = probe(test_features)
                test_ce = F.cross_entropy(test_logits, test_targets).item()
                if test_ce < best_test_ce:
                    best_test_ce = test_ce
                    best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
    if best_state is not None:
        probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    probe.eval()
    return probe, best_test_ce


@torch.no_grad()
def evaluate_probe(probe, features, targets, device):
    features = features.to(device)
    targets = targets.to(device)
    logits = probe(features)
    ce = F.cross_entropy(logits, targets).item()
    pred = logits.argmax(-1)
    top1 = (pred == targets).float().mean().item()
    top5 = (logits.topk(5, dim=-1).indices == targets[:, None]).any(-1).float().mean().item()
    return {"ce": float(ce), "top1": float(top1), "top5": float(top5),
            "pred": pred.detach().cpu().numpy()}


def main():
    from transformers import AutoTokenizer, MambaForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MambaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
    model.eval()
    n_layers = len(model.backbone.layers)
    print(f"  model loaded, {n_layers} layers")

    print("  collecting hidden states for train and eval splits...")
    H_train, y_train = collect_states(model, tokenizer, TRAIN_PROMPTS, device)
    H_eval, y_eval = collect_states(model, tokenizer, EVAL_PROMPTS, device)
    d = H_train.shape[1]
    V = model.lm_head.weight.shape[0]
    print(f"  train pairs: {H_train.shape[0]}, eval pairs: {H_eval.shape[0]}, d={d}, V={V}")

    # SVD of LM head once.
    print("  SVD of LM head weight...")
    W = model.lm_head.weight.detach().to(torch.float32).cpu()  # (V, d)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    print(f"  top-32 singular values: {S[:32].tolist()}")

    # Oracle: use the actual LM head as a probe (training-free, fixed).
    @torch.no_grad()
    def oracle_eval(H, y):
        H_dev = H.to(device); y_dev = y.to(device)
        logits = model.lm_head(H_dev)
        ce = F.cross_entropy(logits, y_dev).item()
        pred = logits.argmax(-1)
        top1 = (pred == y_dev).float().mean().item()
        top5 = (logits.topk(5, dim=-1).indices == y_dev[:, None]).any(-1).float().mean().item()
        return {"ce": ce, "top1": top1, "top5": top5, "pred": pred.detach().cpu().numpy()}

    oracle_train = oracle_eval(H_train, y_train)
    oracle_eval_metrics = oracle_eval(H_eval, y_eval)
    print(f"  ORACLE (real LM head):  train top1={oracle_train['top1']:.4f}  "
          f"eval top1={oracle_eval_metrics['top1']:.4f}  eval ce={oracle_eval_metrics['ce']:.3f}")

    results = {
        "oracle_train": {k: v for k, v in oracle_train.items() if k != "pred"},
        "oracle_eval":  {k: v for k, v in oracle_eval_metrics.items() if k != "pred"},
        "by_K": {},
    }

    primary_preds_per_K = {}
    buffer_preds_per_K = {}

    for K in K_VALUES:
        print(f"\n  === K = {K} ===")
        Vk = Vt[:K]                                  # (K, d) primary basis (rows)
        Vb = Vt[K:]                                  # ((d-K), d) buffer basis (rows)
        # Project h onto each subspace -- KEEP the (d)-shape representation for
        # the probe, so the probe's input dim stays d but only K (or d-K)
        # directions carry information.
        # Actually it is more efficient to feed the lower-dim coords directly:
        H_train_p = H_train @ Vk.T                   # (N, K)
        H_train_b = H_train @ Vb.T                   # (N, d-K)
        H_eval_p  = H_eval  @ Vk.T
        H_eval_b  = H_eval  @ Vb.T

        # Train probes.
        t0 = time.time()
        probe_p, _ = train_probe(H_train_p, y_train, V, H_eval_p, y_eval, device,
                                  epochs=120, lr=1e-2, weight_decay=1e-4)
        t1 = time.time()
        probe_b, _ = train_probe(H_train_b, y_train, V, H_eval_b, y_eval, device,
                                  epochs=120, lr=1e-2, weight_decay=1e-4)
        t2 = time.time()

        prim_metrics = evaluate_probe(probe_p, H_eval_p, y_eval, device)
        buf_metrics  = evaluate_probe(probe_b, H_eval_b, y_eval, device)

        prim_correct = (prim_metrics["pred"] == y_eval.numpy())
        buf_correct  = (buf_metrics["pred"]  == y_eval.numpy())
        prim_only = ((prim_correct) & (~buf_correct)).mean()
        buf_only  = ((~prim_correct) & (buf_correct)).mean()
        both      = (prim_correct & buf_correct).mean()
        neither   = ((~prim_correct) & (~buf_correct)).mean()
        # The interesting cell: oracle gets it right but buffer gets it right while primary doesn't.
        oracle_correct = (oracle_eval_metrics["pred"] == y_eval.numpy())
        rescue_count = ((~prim_correct) & (buf_correct) & (~oracle_correct)).sum()
        rescue_count_oracle_right = ((~prim_correct) & (buf_correct) & (oracle_correct)).sum()

        results["by_K"][K] = {
            "primary_dim": K,
            "buffer_dim":  d - K,
            "primary_eval": {k: v for k, v in prim_metrics.items() if k != "pred"},
            "buffer_eval":  {k: v for k, v in buf_metrics.items() if k != "pred"},
            "agreement": {
                "primary_only_correct": float(prim_only),
                "buffer_only_correct":  float(buf_only),
                "both_correct":         float(both),
                "neither_correct":      float(neither),
                "buffer_correct_when_oracle_wrong":   int(rescue_count),
                "buffer_correct_when_oracle_right":   int(rescue_count_oracle_right),
            },
            "train_seconds": float(t2 - t0),
        }
        print(f"  primary  ({K} dim):   top1={prim_metrics['top1']:.4f}  "
              f"top5={prim_metrics['top5']:.4f}  ce={prim_metrics['ce']:.3f}")
        print(f"  buffer  ({d-K} dim):   top1={buf_metrics['top1']:.4f}  "
              f"top5={buf_metrics['top5']:.4f}  ce={buf_metrics['ce']:.3f}")
        print(f"  agreement: prim_only={prim_only:.3f}  buf_only={buf_only:.3f}  "
              f"both={both:.3f}  neither={neither:.3f}")
        print(f"  buffer correct when oracle wrong: {rescue_count}")
        print(f"  buffer correct when oracle right: {rescue_count_oracle_right}")

        primary_preds_per_K[K] = prim_metrics["pred"]
        buffer_preds_per_K[K]  = buf_metrics["pred"]

    res_path = RES_DIR / "c3_results.json"
    res_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  saved: {res_path}")

    # ---------- Plot 1: top-1 / top-5 / CE vs K, primary vs buffer ----------
    Ks = K_VALUES
    prim_top1 = [results["by_K"][K]["primary_eval"]["top1"] for K in Ks]
    buf_top1  = [results["by_K"][K]["buffer_eval"]["top1"]  for K in Ks]
    prim_ce   = [results["by_K"][K]["primary_eval"]["ce"]   for K in Ks]
    buf_ce    = [results["by_K"][K]["buffer_eval"]["ce"]    for K in Ks]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(Ks, prim_top1, "o-", label="PRIMARY (top-K subspace)", color="#1f77b4")
    axes[0].plot(Ks, buf_top1,  "s-", label="BUFFER (orth. complement)", color="#d62728")
    axes[0].axhline(oracle_eval_metrics["top1"], color="black", linestyle="--",
                    label=f"oracle (full LM head) = {oracle_eval_metrics['top1']:.3f}")
    axes[0].set_xscale("log"); axes[0].set_xlabel("K (primary subspace size)")
    axes[0].set_ylabel("eval top-1 accuracy")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)

    axes[1].plot(Ks, prim_ce, "o-", color="#1f77b4")
    axes[1].plot(Ks, buf_ce,  "s-", color="#d62728")
    axes[1].axhline(oracle_eval_metrics["ce"], color="black", linestyle="--")
    axes[1].set_xscale("log"); axes[1].set_xlabel("K")
    axes[1].set_ylabel("eval cross-entropy"); axes[1].grid(alpha=0.3)

    fig.suptitle("C3: probing primary vs buffer subspace (Mamba-130m, final layer)")
    fig.tight_layout()
    fig_path = FIG_DIR / "c3_primary_vs_buffer.png"
    fig.savefig(fig_path, dpi=140)
    print(f"  saved: {fig_path}")


if __name__ == "__main__":
    main()
