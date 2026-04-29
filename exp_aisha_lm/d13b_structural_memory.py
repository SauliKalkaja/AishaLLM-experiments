"""
Experiment D13b: structural memory done correctly.

D13 v1 mistake: I passed Aisha's structural fingerprint (POS profile, 16D
centroid, step) to Qwen as TEXT in a preamble. Qwen interpreted "manifold"
and "centroid" literally as words about AI/ML topics and started talking
about AI ethics. The structure must NEVER cross into Qwen's text channel.

User's architectural split:
  - Aisha consumes structure   (running centroid, POS profile)  -> Aisha's internal use
  - Qwen consumes words         (verbatim current turn)
  - The bridge is the LOGIT BIAS, which is built using Aisha's structural
    knowledge but presents to Qwen as nothing more than a per-token boost.

This script implements that. For each turn:

  prior_turns -> Aisha computes running 16D centroid
              -> finds K manifold neighbors of the centroid (words)
  current_turn -> normal expand_content_boundary (its own seeds)
  boundary = neighbors_of_running_centroid UNION boundary_of_current_turn
  Qwen prompt = just the current turn, NO structural preamble.
  Qwen logits = base + lambda * mask(boundary)

Memory length N controls how many prior turns feed into the running centroid.
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter

EXP_ROOT = Path(__file__).resolve().parent
CACHE_DIR = EXP_ROOT.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)

_default_aisha = Path(__file__).resolve().parent.parent.parent / "aisha"
AISHA_ROOT = Path(os.environ.get("AISHA_ROOT", str(_default_aisha))).resolve()
sys.path.insert(0, str(AISHA_ROOT))

sys.path.insert(0, str(EXP_ROOT))
from d9_hallucination_test import make_aisha, aisha_structure
from d13_structural_memory import CONVERSATIONS, MEMORY_LENGTHS

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
RES_DIR = EXP_ROOT.parent / "results"
WORD_RE = re.compile(r"[a-zA-Z']+")

BACKBONE_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LAMBDA = 1.5
NEIGHBORS_PER_CENTROID = 30   # how many manifold neighbors of running centroid to add


def aisha_seeds(responder, text):
    """Get manifold-known content seeds from text."""
    seeds = []
    for w in WORD_RE.findall(text.lower()):
        old_i = responder.wm.idx.get(w)
        if old_i is None or responder.R.is_stopword[old_i]:
            continue
        new_i = int(responder._old_to_new[old_i])
        if new_i >= 0:
            seeds.append(new_i)
    return seeds


def expand_from_seeds(responder, seeds, k_per_seed=20):
    """Expand boundary from a list of seed indices."""
    if not seeds:
        return set()
    Q = responder._main_q
    boundary = set(seeds)
    for s_i in seeds:
        d2 = responder.kahler.mahalanobis_to_seed(Q[s_i], Q)
        d2[s_i] = np.inf
        top = np.argpartition(d2, k_per_seed)[:k_per_seed]
        boundary.update(int(j) for j in top)
    return boundary


def neighbors_of_centroid(responder, centroid, k=NEIGHBORS_PER_CENTROID):
    """Return k manifold-word indices nearest to the given 16D centroid."""
    Q = responder._main_q
    d2 = responder.kahler.mahalanobis_to_seed(centroid, Q)
    return set(int(j) for j in np.argpartition(d2, k)[:k])


def boundary_with_structural_memory(responder, current_turn: str,
                                      prior_turns: list[str], memory_length: int):
    """Architectural split, done right.
       Aisha uses STRUCTURE of prior turns to choose neighbors.
       Returns a list of WORDS to be applied as logit bias on Qwen.
       Qwen never sees the centroid or POS profile."""
    # Boundary from current turn alone (the "no memory" baseline)
    cur_seeds = aisha_seeds(responder, current_turn)
    boundary = expand_from_seeds(responder, cur_seeds)

    # Memory contribution: running centroid of prior_turns -> k neighbors
    if memory_length > 0 and prior_turns:
        prior_text = " ".join(prior_turns[-memory_length:])
        struct = aisha_structure(responder, prior_text)
        if struct is not None:
            running_centroid = struct["doc_centroid"]
            mem_neighbors = neighbors_of_centroid(responder, running_centroid)
            boundary.update(mem_neighbors)

    # Convert to words
    out = []
    for new_i in boundary:
        try:
            w = responder.kahler.lemmas[int(new_i)]
        except Exception:
            continue
        if isinstance(w, str) and w.isalpha():
            out.append(w)
    return out


def build_boost_mask(words, tokenizer, vocab_size, device):
    mask = torch.zeros(vocab_size, device=device)
    for w in words:
        for variant in (w, " " + w, " " + w.capitalize()):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            for tid in ids:
                if 0 <= tid < vocab_size:
                    mask[tid] = 1.0
    return mask


@torch.no_grad()
def generate_qwen(model, tokenizer, prompt, mask, lam, max_new, T, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(enc.input_ids, attention_mask=enc.attention_mask, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1]
    eos = tokenizer.eos_token_id
    g = []
    for _ in range(max_new):
        if mask is not None and lam != 0.0:
            logits = logits + lam * mask
        probs = F.softmax(logits / max(T, 1e-6), dim=-1)
        nid = torch.multinomial(probs, 1)
        if eos is not None and int(nid) == eos:
            break
        g.append(int(nid))
        out = model(nid.unsqueeze(0), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
    return tokenizer.decode(g, skip_special_tokens=True)


def chat_prompt(tokenizer, msg):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": msg}],
        tokenize=False, add_generation_prompt=True)


def pos_l1(pp1, pp2):
    keys = set(pp1.keys()) | set(pp2.keys())
    return sum(abs(pp1.get(k, 0) - pp2.get(k, 0)) for k in keys)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[d13b] device: {device}")
    aisha = make_aisha()

    print(f"[d13b] loading {BACKBONE_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE_NAME, torch_dtype=torch.float16, device_map="auto").eval()
    print("[d13b] loaded.")

    rows = []
    for ci, conv in enumerate(CONVERSATIONS):
        prior_turns = conv["turns"]
        last_q = conv["final_question"]
        full_struct = aisha_structure(aisha, " ".join(prior_turns))
        if full_struct is None:
            continue
        full_centroid = full_struct["doc_centroid"]
        full_pos = full_struct["pos_profile"]

        per_N = {}
        for N in MEMORY_LENGTHS:
            words = boundary_with_structural_memory(aisha, last_q, prior_turns, N)
            mask = (build_boost_mask(words, tok, model.config.vocab_size, device)
                    if words else None)
            torch.manual_seed(0)
            out = generate_qwen(model, tok, chat_prompt(tok, last_q),
                                 mask, LAMBDA, max_new=80, T=0.3, device=device)

            ans_struct = aisha_structure(aisha, out)
            cd = float(np.linalg.norm(ans_struct["doc_centroid"] - full_centroid)) \
                  if ans_struct is not None else float("nan")
            pl = pos_l1(ans_struct["pos_profile"], full_pos) \
                  if ans_struct is not None else float("nan")
            per_N[N] = {
                "boundary_size": len(words),
                "answer_text": out,
                "answer_centroid_dist_to_conv": cd,
                "answer_pos_l1_to_conv":        pl,
            }

        rows.append({
            "style":  conv["style"], "final_q": last_q,
            "full_centroid":    full_centroid.tolist(),
            "full_pos_profile": full_pos,
            "per_N":            per_N,
        })

        print(f"\n[{ci+1}/{len(CONVERSATIONS)}] style={conv['style']}")
        for N in MEMORY_LENGTHS:
            r = per_N[N]
            print(f"  N={N}  bound={r['boundary_size']:>4d}  "
                  f"cdist={r['answer_centroid_dist_to_conv']:.3f}  "
                  f"pos_l1={r['answer_pos_l1_to_conv']:.3f}  "
                  f"| {r['answer_text'][:90]!r}")

    # Aggregate
    print("\n=== D13b summary across 8 conversations (Aisha keeps structure internal) ===")
    print(f"{'memory N':>9s}  {'cdist_mean':>11s}  {'cdist_med':>10s}  "
          f"{'pos_l1_mean':>12s}  {'avg_bound':>10s}")
    summary = {}
    for N in MEMORY_LENGTHS:
        cd = np.array([r["per_N"][N]["answer_centroid_dist_to_conv"] for r in rows])
        pl = np.array([r["per_N"][N]["answer_pos_l1_to_conv"]        for r in rows])
        bs = np.array([r["per_N"][N]["boundary_size"]                for r in rows])
        cd = cd[~np.isnan(cd)]; pl = pl[~np.isnan(pl)]
        summary[N] = {
            "centroid_dist_mean":   float(cd.mean()) if cd.size else float("nan"),
            "centroid_dist_median": float(np.median(cd)) if cd.size else float("nan"),
            "pos_l1_mean":          float(pl.mean()) if pl.size else float("nan"),
            "boundary_size_mean":   float(bs.mean()),
        }
        print(f"  {N:>9d}  {summary[N]['centroid_dist_mean']:>11.3f}  "
              f"{summary[N]['centroid_dist_median']:>10.3f}  "
              f"{summary[N]['pos_l1_mean']:>12.3f}  "
              f"{summary[N]['boundary_size_mean']:>10.1f}")

    # Per-style: how does N=0 compare to N=5 per style?
    print(f"\n=== Per-style: N=0 vs N=5 ===")
    print(f"{'style':>22s}  {'cdist N=0':>10s}  {'cdist N=5':>10s}  delta")
    helped = 0; hurt = 0
    for r in rows:
        c0 = r["per_N"][0]["answer_centroid_dist_to_conv"]
        c5 = r["per_N"][5]["answer_centroid_dist_to_conv"]
        delta = c5 - c0
        sign = "↓" if delta < 0 else "↑"
        if delta < 0: helped += 1
        else: hurt += 1
        print(f"  {r['style']:>22s}  {c0:>10.3f}  {c5:>10.3f}  {sign}{abs(delta):.3f}")
    print(f"\n  N=5 helped (lower cdist): {helped}/{len(rows)}, hurt: {hurt}/{len(rows)}")

    out_path = RES_DIR / "d13b_structural_memory_results.json"
    out_path.write_text(json.dumps({
        "n_conversations": len(rows), "memory_lengths": MEMORY_LENGTHS,
        "lambda": LAMBDA, "neighbors_per_centroid": NEIGHBORS_PER_CENTROID,
        "summary": summary, "per_conversation": rows,
    }, indent=2, default=str))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
