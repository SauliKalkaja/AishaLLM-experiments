"""
Experiment D2: smaller back-end LM + Aisha boundary vs larger LM alone.

The energy-savings claim of the hybrid pipeline is: with Aisha's
input-conditioned boundary as a structural prior, a smaller pretrained LM
can match a larger LM's quality. This script tests it.

Compared on the same 30 prompts as D1:
  - Mamba-130m alone               (D1 baseline)
  - Pythia-70m alone               (~half the params)
  - Pythia-70m + Aisha boundary    (lam in {1.5, 3.0})
  - Pythia-160m alone              (sanity: ~similar size to Mamba-130m)
  - Pythia-160m + Aisha boundary   (lam = 1.5)

Metrics:
  - Self-perplexity (model rates own output)        -- biased
  - Judge perplexity (GPT-2 small)                  -- different family
  - Distinct-1 (unigram diversity)                  -- penalizes repetition
  - Distinct-2 (bigram diversity)                   -- penalizes loops
  - Boundary hit rate (topic relevance)             -- input-conditioned

The combination of judge-PPL AND distinct-2 should expose D3's blind spot:
fluent-and-diverse beats fluent-but-repetitive even if both have low PPL.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
CACHE_DIR = EXP_ROOT.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)

# Aisha on path
_default_aisha = Path(__file__).resolve().parent.parent.parent / "aisha"
AISHA_ROOT = Path(os.environ.get("AISHA_ROOT", str(_default_aisha))).resolve()
if not AISHA_ROOT.exists():
    raise FileNotFoundError(f"AISHA_ROOT not found: {AISHA_ROOT}")
sys.path.insert(0, str(AISHA_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

RES_DIR = EXP_ROOT.parent / "results"

WORD_RE = re.compile(r"[a-zA-Z']+")

# Models to compare. Each entry is (name, model_id, kind).
MODELS = [
    ("mamba-130m", "state-spaces/mamba-130m-hf", "mamba"),
    ("pythia-70m", "EleutherAI/pythia-70m", "causal"),
    ("pythia-160m", "EleutherAI/pythia-160m", "causal"),
]

JUDGE_NAME = "gpt2"


# ---------- Aisha ----------

def make_aisha():
    from responder_pos import POSResponder
    print("[aisha] loading POSResponder...", flush=True)
    return POSResponder(use_harper=False)


def aisha_boundary_words(responder, query: str) -> list[str]:
    boundary = responder.expand_content_boundary(query)
    out = []
    for new_i in boundary:
        try:
            w = responder.kahler.lemmas[new_i]
        except Exception:
            continue
        if isinstance(w, str) and w.isalpha():
            out.append(w)
    return out


def aisha_response(responder, query: str) -> str:
    out = responder.respond(query)
    return out["text"] if isinstance(out, dict) else str(out)


# ---------- shared generation ----------

def build_boost_mask(words, tokenizer, vocab_size, device):
    mask = torch.zeros(vocab_size, device=device)
    seen = set()
    for w in words:
        for variant in (w, " " + w, " " + w.capitalize()):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            for tid in ids:
                if 0 <= tid < vocab_size and tid not in seen:
                    mask[tid] = 1.0
                    seen.add(tid)
    return mask


@torch.no_grad()
def generate_mamba(model, tokenizer, prompt, mask, lam, max_new, temperature, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(enc.input_ids, use_cache=True)
    cache = out.cache_params
    logits = out.logits[0, -1]
    eos = tokenizer.eos_token_id
    generated = []
    for step in range(max_new):
        if mask is not None and lam != 0.0:
            logits = logits + lam * mask
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        nid = torch.multinomial(probs, 1)
        if eos is not None and int(nid) == eos:
            break
        generated.append(int(nid))
        out = model(nid.unsqueeze(0), cache_params=cache, use_cache=True)
        cache = out.cache_params
        logits = out.logits[0, -1]
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def generate_causal(model, tokenizer, prompt, mask, lam, max_new, temperature, device):
    """Generic AutoModelForCausalLM with KV cache (Pythia, GPT-2, etc)."""
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    out = model(input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1]
    eos = tokenizer.eos_token_id
    generated = []
    for step in range(max_new):
        if mask is not None and lam != 0.0:
            logits = logits + lam * mask
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        nid = torch.multinomial(probs, 1)
        if eos is not None and int(nid) == eos:
            break
        generated.append(int(nid))
        out = model(nid.unsqueeze(0), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------- metrics ----------

@torch.no_grad()
def perplexity(model, tokenizer, text, device):
    if not text or not text.strip():
        return float("nan")
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    if enc.input_ids.shape[1] < 2:
        return float("nan")
    out = model(enc.input_ids, labels=enc.input_ids)
    return float(np.exp(out.loss.item()))


def distinct_n(text, n):
    toks = WORD_RE.findall(text.lower())
    if len(toks) < n:
        return float("nan")
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return float("nan")
    return len(set(grams)) / len(grams)


def boundary_hit_rate(text, boundary):
    bset = set(w.lower() for w in boundary)
    toks = [t.lower() for t in WORD_RE.findall(text)]
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in bset) / len(toks)


# ---------- driver ----------

def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, MambaForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device: {device}")

    aisha = make_aisha()

    # Load all back-end LMs.
    backends = {}
    for name, mid, kind in MODELS:
        print(f"[setup] loading {name} ({mid})", flush=True)
        tok = AutoTokenizer.from_pretrained(mid)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if kind == "mamba":
            mdl = MambaForCausalLM.from_pretrained(mid, torch_dtype=torch.float32).to(device)
        else:
            mdl = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32).to(device)
        mdl.eval()
        n_params = sum(p.numel() for p in mdl.parameters())
        print(f"  {name}: {n_params/1e6:.1f}M params, vocab={mdl.config.vocab_size}")
        backends[name] = {"model": mdl, "tokenizer": tok, "kind": kind, "params": n_params}

    # Judge model.
    print(f"[setup] loading judge {JUDGE_NAME}", flush=True)
    judge_tok = AutoTokenizer.from_pretrained(JUDGE_NAME)
    if judge_tok.pad_token is None:
        judge_tok.pad_token = judge_tok.eos_token
    judge_mdl = AutoModelForCausalLM.from_pretrained(JUDGE_NAME).to(device)
    judge_mdl.eval()

    # Test prompts (same 30 as D1; deterministic seeding).
    a_prompt_path = AISHA_ROOT / "data" / "processed" / "a_prompt_corpus.txt"
    all_prompts = [ln.strip() for ln in a_prompt_path.read_text().splitlines()
                    if ln.strip() and len(ln.strip()) > 8][:300]
    rng = np.random.default_rng(0)
    selected = list(rng.choice(all_prompts, size=min(30, len(all_prompts)), replace=False))
    print(f"[setup] {len(selected)} test prompts")

    def make_prompt(user_text):
        return f"User: {user_text}\nAssistant:"

    # Conditions. For each backend, alone and with Aisha at two lambdas.
    # Plus one Aisha-alone reference (same as D1).
    conditions = []
    for name in backends:
        conditions.append((name + "_alone",       name, 0.0))
        conditions.append((name + "_aisha_lam1.5", name, 1.5))
        conditions.append((name + "_aisha_lam3.0", name, 3.0))

    rows = []
    for pi, prompt in enumerate(selected):
        boundary = aisha_boundary_words(aisha, prompt)
        ais_resp = aisha_response(aisha, prompt)
        lm_prompt = make_prompt(prompt)

        per_prompt = {"prompt": prompt, "boundary_size": len(boundary),
                       "boundary_first10": boundary[:10],
                       "outputs": {}, "metrics": {}}

        # Aisha-alone reference
        per_prompt["outputs"]["aisha_alone"] = ais_resp

        # Run each backend condition
        for cond_name, backend_name, lam in conditions:
            torch.manual_seed(0)
            be = backends[backend_name]
            mask = build_boost_mask(boundary, be["tokenizer"], be["model"].config.vocab_size, device) \
                    if (boundary and lam > 0) else None
            gen_fn = generate_mamba if be["kind"] == "mamba" else generate_causal
            text = gen_fn(be["model"], be["tokenizer"], lm_prompt, mask, lam, 24, 0.8, device)
            per_prompt["outputs"][cond_name] = text

        # Compute metrics per output (judge-PPL + distinct-n + boundary hit + length)
        for cname, text in per_prompt["outputs"].items():
            judge_ppl = perplexity(judge_mdl, judge_tok, text, device)
            d1 = distinct_n(text, 1)
            d2 = distinct_n(text, 2)
            hit = boundary_hit_rate(text, boundary)
            n_tok = len(WORD_RE.findall(text))
            per_prompt["metrics"][cname] = {
                "judge_ppl": judge_ppl,
                "distinct_1": d1, "distinct_2": d2,
                "boundary_hit_rate": hit,
                "n_tokens": n_tok,
            }
        rows.append(per_prompt)

        if pi < 3 or pi == len(selected) - 1:
            print(f"\n[{pi+1}/{len(selected)}] {prompt!r}")
            print(f"  boundary ({len(boundary)})")
            for cname, text in per_prompt["outputs"].items():
                m = per_prompt["metrics"][cname]
                print(f"  {cname:>30s} JPPL={m['judge_ppl']:7.1f} d2={m['distinct_2']:.2f} "
                      f"hit={m['boundary_hit_rate']:.2f} L={m['n_tokens']:>3d} | "
                      f"{text[:70]!r}")

    # Aggregate
    all_conds = ["aisha_alone"] + [c for c, _, _ in conditions]
    summary = {}
    for c in all_conds:
        ms = [r["metrics"][c] for r in rows]
        ppls = [m["judge_ppl"] for m in ms if not np.isnan(m["judge_ppl"])]
        d2s  = [m["distinct_2"] for m in ms if not np.isnan(m["distinct_2"])]
        d1s  = [m["distinct_1"] for m in ms if not np.isnan(m["distinct_1"])]
        hits = [m["boundary_hit_rate"] for m in ms]
        lens = [m["n_tokens"] for m in ms]
        summary[c] = {
            "judge_ppl_median":  float(np.median(ppls)) if ppls else float("nan"),
            "judge_ppl_mean":    float(np.mean(ppls))   if ppls else float("nan"),
            "distinct_1_mean":   float(np.mean(d1s))    if d1s  else float("nan"),
            "distinct_2_mean":   float(np.mean(d2s))    if d2s  else float("nan"),
            "boundary_hit_mean": float(np.mean(hits)),
            "tokens_mean":       float(np.mean(lens)),
        }

    print("\n=== D2 summary (30 prompts, judge=GPT-2) ===")
    print(f"{'condition':>30s}  {'JPPL_med':>9s}  {'JPPL_mean':>10s}  "
          f"{'d-1':>6s}  {'d-2':>6s}  {'hit':>6s}  {'tokens':>7s}")
    for c in all_conds:
        s = summary[c]
        print(f"  {c:>28s}  {s['judge_ppl_median']:>9.1f}  {s['judge_ppl_mean']:>10.1f}  "
              f"{s['distinct_1_mean']:>6.3f}  {s['distinct_2_mean']:>6.3f}  "
              f"{s['boundary_hit_mean']:>6.3f}  {s['tokens_mean']:>7.1f}")

    # Energy comparison: parameters per backend
    print("\n=== Energy proxy (active params per token) ===")
    for name, mid, kind in MODELS:
        n = backends[name]["params"] / 1e6
        print(f"  {name:>14s}: {n:6.1f}M params")

    res = {
        "models": [(n, m, k, backends[n]["params"]) for (n, m, k) in MODELS],
        "n_prompts": len(rows),
        "judge_model": JUDGE_NAME,
        "summary": summary,
        "per_prompt": rows,
    }
    out_path = RES_DIR / "d2_smaller_lm_results.json"
    out_path.write_text(json.dumps(res, indent=2, default=str))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
