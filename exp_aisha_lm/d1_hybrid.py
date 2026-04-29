"""
Experiment D1: Aisha topical boundary + pretrained Mamba (logit-biased generation).

Aisha's `expand_content_boundary(query)` returns the set of words on its POS
manifold that are k-NN-close (under POS-Kahler distance) to the query's
content seeds. This is Aisha's *input-conditioned* topical signal -- the
42-word neighborhood of the user's prompt.

The hybrid pipeline:
  1. User prompt -> Aisha boundary (set of words)  [O(1) per query]
  2. Boost the pretrained Mamba's logits at each step toward boundary tokens
  3. Sample autoregressively with the bias

Conditions compared on the same N test prompts:
  A. Aisha alone                  -- current gibberish baseline
  B. Mamba alone (no bias)        -- pretrained baseline
  C. Mamba + Aisha boundary, lam=1.5
  D. Mamba + Aisha boundary, lam=3.0
  E. Mamba + Aisha boundary, lam=6.0  (over-pulled)

Metrics (per condition, averaged across prompts):
  - Boundary hit rate    : fraction of generated content tokens that lie in
                            Aisha's boundary for that prompt
  - Self-perplexity      : Mamba's own perplexity on the generated text
                            (proxy for fluency; lower = more "natural")
  - Mean response length : in tokens
  - Sample dump          : qualitative comparison

This isolates whether Aisha's boundary signal can pull a pretrained LM toward
topic-relevant outputs *without* destroying fluency. If yes, the hybrid is
worth scaling up. If fluency collapses at any lam where boundary-hit-rate
beats the LM-alone baseline, the boundary signal is too narrow.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# Bring Aisha onto the path. Allow override via env var (for the pod side).
_default_aisha = Path(__file__).resolve().parent.parent.parent / "aisha"
AISHA_ROOT = Path(os.environ.get("AISHA_ROOT", str(_default_aisha))).resolve()
if not AISHA_ROOT.exists():
    raise FileNotFoundError(
        f"AISHA_ROOT not found: {AISHA_ROOT}. Set AISHA_ROOT env var or "
        f"place the aisha/ tree at {_default_aisha}.")
sys.path.insert(0, str(AISHA_ROOT))

# HF cache.
EXP_ROOT = Path(__file__).resolve().parent
CACHE_DIR = EXP_ROOT.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

FIG_DIR = EXP_ROOT.parent / "figures"
RES_DIR = EXP_ROOT.parent / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

WORD_RE = re.compile(r"[a-zA-Z']+")
MODEL_NAME = "state-spaces/mamba-130m-hf"


# ----------------------------------------------------------------------------
# Aisha side: load responder, expose boundary extraction.
# ----------------------------------------------------------------------------

def make_aisha():
    from responder_pos import POSResponder
    print("[aisha] loading POSResponder...", flush=True)
    r = POSResponder(use_harper=False)
    return r


def aisha_boundary_words(responder, query: str) -> list[str]:
    """Return Aisha's input-conditioned topical word list (the 42-or-so set)."""
    boundary_set = responder.expand_content_boundary(query)
    out = []
    for new_i in boundary_set:
        old_i = int(responder.kahler.word_idx_orig[new_i])
        # Use lemma. Some kahler implementations expose .lemmas indexed by new idx.
        try:
            w = responder.kahler.lemmas[new_i]
        except Exception:
            # fallback: lookup the surface form from word_manifold
            w = responder.wm.vocab[old_i] if old_i < len(responder.wm.vocab) else None
        if isinstance(w, str) and w.isalpha():
            out.append(w)
    return out


def aisha_response(responder, query: str) -> str:
    """Get Aisha's own gibberish baseline response."""
    out = responder.respond(query)
    return out["text"] if isinstance(out, dict) else str(out)


# ----------------------------------------------------------------------------
# Mamba side: logit-biased generation.
# ----------------------------------------------------------------------------

def build_boost_mask(words: list[str], tokenizer, vocab_size: int, device: str
                     ) -> torch.Tensor:
    """For each boundary word, mark all token-ids that appear when the word is
    encoded in three common surface forms ('w', ' w', ' W'). This catches BPE
    splits and the common leading-space variant."""
    mask = torch.zeros(vocab_size, device=device)
    seen = set()
    for w in words:
        for variant in (w, " " + w, " " + w.capitalize()):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            for tid in ids:
                if tid not in seen:
                    mask[tid] = 1.0
                    seen.add(tid)
    return mask


@torch.no_grad()
def generate(model, tokenizer, prompt: str, boost_mask: torch.Tensor | None,
             lam: float, max_new: int = 24, temperature: float = 0.8,
             device: str = "cuda") -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    out = model(input_ids, use_cache=True)
    cache = out.cache_params
    logits = out.logits[0, -1]
    eos_id = tokenizer.eos_token_id

    generated = []
    for step in range(max_new):
        if boost_mask is not None and lam != 0.0:
            logits = logits + lam * boost_mask
        scaled = logits / max(temperature, 1e-6)
        probs = F.softmax(scaled, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        if eos_id is not None and int(next_id) == eos_id:
            break
        generated.append(int(next_id))
        nxt = next_id.unsqueeze(0)
        out = model(nxt, cache_params=cache, use_cache=True)
        cache = out.cache_params
        logits = out.logits[0, -1]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ----------------------------------------------------------------------------
# Metrics.
# ----------------------------------------------------------------------------

@torch.no_grad()
def self_perplexity(model, tokenizer, text: str, device: str = "cuda") -> float:
    enc = tokenizer(text, return_tensors="pt").to(device)
    if enc.input_ids.shape[1] < 2:
        return float("nan")
    out = model(enc.input_ids, labels=enc.input_ids)
    return float(np.exp(out.loss.item()))


def boundary_hit_rate(text: str, boundary_words: list[str]) -> float:
    bset = set(w.lower() for w in boundary_words)
    toks = [t.lower() for t in WORD_RE.findall(text)]
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in bset)
    return hits / len(toks)


# ----------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------

def main():
    from transformers import AutoTokenizer, MambaForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device: {device}")

    aisha = make_aisha()

    print(f"[setup] loading {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MambaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
    model.eval()
    V = model.config.vocab_size
    print(f"[setup] vocab={V}", flush=True)

    # Test prompts: take 30 from the aisha A-prompt corpus.
    a_prompt_path = AISHA_ROOT / "data" / "processed" / "a_prompt_corpus.txt"
    if a_prompt_path.exists():
        all_prompts = [ln.strip() for ln in a_prompt_path.read_text().splitlines()
                       if ln.strip() and len(ln.strip()) > 8][:300]
    else:
        all_prompts = []
    rng = np.random.default_rng(0)
    selected = list(rng.choice(all_prompts, size=min(30, len(all_prompts)), replace=False))
    if not selected:
        # fallback prompts
        selected = [
            "How are you doing today?",
            "What is the capital of France?",
            "Tell me about quantum mechanics.",
            "I love pizza.",
            "Why is the sky blue?",
            "Can you explain how photosynthesis works?",
        ]
    print(f"[setup] {len(selected)} test prompts")

    # Generation prompt format: bare user input. Mamba is a base LM.
    def make_prompt(user_text):
        return f"User: {user_text}\nAssistant:"

    LAMBDAS = [0.0, 1.5, 3.0, 6.0]
    conditions = ["aisha_alone", "mamba_alone"] + [f"mamba_lam{lam}" for lam in LAMBDAS if lam > 0]

    rows = []
    for pi, prompt in enumerate(selected):
        boundary = aisha_boundary_words(aisha, prompt)
        ais_resp = aisha_response(aisha, prompt)
        mask = build_boost_mask(boundary, tokenizer, V, device) if boundary else None
        lm_prompt = make_prompt(prompt)

        per_prompt = {"prompt": prompt, "boundary": boundary[:20],
                       "boundary_size": len(boundary),
                       "outputs": {}, "metrics": {}}

        # A. Aisha alone
        per_prompt["outputs"]["aisha_alone"] = ais_resp
        per_prompt["metrics"]["aisha_alone"] = {
            "perplexity": self_perplexity(model, tokenizer, ais_resp, device),
            "boundary_hit_rate": boundary_hit_rate(ais_resp, boundary),
            "n_tokens": len(WORD_RE.findall(ais_resp)),
        }

        # B+. Mamba in 4 lambda settings
        torch.manual_seed(0)
        for lam in LAMBDAS:
            text = generate(model, tokenizer, lm_prompt, mask, lam,
                            max_new=24, temperature=0.8, device=device)
            cond = "mamba_alone" if lam == 0.0 else f"mamba_lam{lam}"
            per_prompt["outputs"][cond] = text
            per_prompt["metrics"][cond] = {
                "perplexity": self_perplexity(model, tokenizer, text, device),
                "boundary_hit_rate": boundary_hit_rate(text, boundary),
                "n_tokens": len(WORD_RE.findall(text)),
            }

        rows.append(per_prompt)
        if pi < 3 or pi == len(selected) - 1:
            print(f"\n[{pi+1}/{len(selected)}] {prompt!r}")
            print(f"  boundary ({len(boundary)}): {boundary[:12]}...")
            for c in conditions:
                m = per_prompt["metrics"][c]
                print(f"  {c:>16s}  PPL={m['perplexity']:8.1f}  hits={m['boundary_hit_rate']:.3f}  "
                      f"L={m['n_tokens']:>3d}  | {per_prompt['outputs'][c][:80]!r}")

    # Aggregates
    summary = {}
    for cond in conditions:
        ppls = [r["metrics"][cond]["perplexity"] for r in rows
                if not np.isnan(r["metrics"][cond]["perplexity"])]
        hits = [r["metrics"][cond]["boundary_hit_rate"] for r in rows]
        lens = [r["metrics"][cond]["n_tokens"] for r in rows]
        summary[cond] = {
            "perplexity_median": float(np.median(ppls)) if ppls else float("nan"),
            "perplexity_mean":   float(np.mean(ppls))   if ppls else float("nan"),
            "boundary_hit_mean": float(np.mean(hits)),
            "boundary_hit_std":  float(np.std(hits)),
            "tokens_mean":       float(np.mean(lens)),
            "n_evaluated":       len(ppls),
        }

    print("\n=== D1 summary across all prompts ===")
    print(f"{'condition':>16s}  {'PPL median':>11s}  {'PPL mean':>10s}  "
          f"{'hit_mean':>10s}  {'hit_std':>9s}  {'tokens':>7s}")
    for cond in conditions:
        s = summary[cond]
        print(f"  {cond:>14s}  {s['perplexity_median']:>11.1f}  "
              f"{s['perplexity_mean']:>10.1f}  "
              f"{s['boundary_hit_mean']:>10.3f}  {s['boundary_hit_std']:>9.3f}  "
              f"{s['tokens_mean']:>7.1f}")

    out_path = RES_DIR / "d1_hybrid_results.json"
    out_path.write_text(json.dumps({"per_prompt": rows, "summary": summary}, indent=2))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
