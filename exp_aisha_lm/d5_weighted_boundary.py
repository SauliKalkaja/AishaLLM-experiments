"""
Experiment D5: per-token weighted boundary.

D1-D4 used a uniform logit bias: every word in Aisha's boundary got the
same +lambda. But the boundary has structure -- some words are k-NN
neighbors at small Mahalanobis distance from a query seed, others are
far-edge. A natural refinement: weight each word by its proximity to the
nearest query seed.

Per-word weight  w_j = exp( - d_j / sigma )   where d_j is the smallest
Mahalanobis distance from boundary word j to any of the query's content
seeds, and sigma is the median such distance (so weights span ~ [exp(-2),
exp(0)] = [0.14, 1.0]).

Conditions on the same 30 prompts:
  - mamba-130m alone
  - mamba-130m + uniform Aisha (lam=1.5)        [D1 baseline]
  - mamba-130m + WEIGHTED Aisha (lam=1.5)        [new]
  - pythia-160m alone
  - pythia-160m + uniform Aisha (lam=1.5)
  - pythia-160m + WEIGHTED Aisha (lam=1.5)

Metrics: judge-PPL, distinct-2, boundary hit rate, tokens.
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

_default_aisha = Path(__file__).resolve().parent.parent.parent / "aisha"
AISHA_ROOT = Path(os.environ.get("AISHA_ROOT", str(_default_aisha))).resolve()
sys.path.insert(0, str(AISHA_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
RES_DIR = EXP_ROOT.parent / "results"
WORD_RE = re.compile(r"[a-zA-Z']+")

MODELS = [
    ("mamba-130m", "state-spaces/mamba-130m-hf", "mamba"),
    ("pythia-160m", "EleutherAI/pythia-160m", "causal"),
]
JUDGE_NAME = "gpt2"


def make_aisha():
    from responder_pos import POSResponder
    return POSResponder(use_harper=False)


def aisha_weighted_boundary(responder, query: str
                             ) -> tuple[list[str], np.ndarray]:
    """Return (words, weights) where weight_j reflects Mahalanobis distance
    from word j to the closest query seed, normalised so that the closest
    seed itself has weight 1.0."""
    seeds: list[int] = []
    for w in WORD_RE.findall(query.lower()):
        old_i = responder.wm.idx.get(w)
        if old_i is None or responder.R.is_stopword[old_i]:
            continue
        new_i = int(responder._old_to_new[old_i])
        if new_i >= 0:
            seeds.append(new_i)
    if not seeds:
        return [], np.array([])

    Q = responder._main_q
    boundary_set: set[int] = set(seeds)
    # min-distance from each candidate to any seed
    K = responder.content_boundary_k
    all_d2 = []
    for s_i in seeds:
        d2 = responder.kahler.mahalanobis_to_seed(Q[s_i], Q)
        d2[s_i] = np.inf
        top = np.argpartition(d2, K)[:K]
        for j in top:
            boundary_set.add(int(j))
        all_d2.append(d2)
    boundary_idx = np.array(sorted(boundary_set), dtype=np.int64)
    min_d2 = np.minimum.reduce(all_d2)[boundary_idx]   # closest seed distance
    # seeds themselves have d2=inf above; for them, weight = 1.0 (anchor).
    seed_set = set(seeds)
    is_seed = np.array([int(i) in seed_set for i in boundary_idx])
    # use sqrt-distance for a more linear scale
    d = np.sqrt(np.maximum(min_d2, 0.0))
    finite_d = d[~is_seed & np.isfinite(d)]
    if finite_d.size > 0:
        sigma = max(np.median(finite_d), 1e-6)
    else:
        sigma = 1.0
    weights = np.exp(-d / sigma)
    weights[is_seed] = 1.0   # query's own content words have full weight
    weights[~np.isfinite(weights)] = 1.0

    words: list[str] = []
    keep_mask = np.zeros(len(boundary_idx), dtype=bool)
    for k, new_i in enumerate(boundary_idx):
        try:
            w = responder.kahler.lemmas[int(new_i)]
        except Exception:
            continue
        if isinstance(w, str) and w.isalpha():
            words.append(w)
            keep_mask[k] = True
    return words, weights[keep_mask]


def aisha_response(responder, query: str) -> str:
    out = responder.respond(query)
    return out["text"] if isinstance(out, dict) else str(out)


def build_boost_mask(words, weights, tokenizer, vocab_size, device):
    """Same as before but uses per-word weights instead of uniform 1.0."""
    mask = torch.zeros(vocab_size, device=device)
    for w, weight in zip(words, weights):
        for variant in (w, " " + w, " " + w.capitalize()):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            for tid in ids:
                if 0 <= tid < vocab_size:
                    mask[tid] = max(float(mask[tid].item()), float(weight))
    return mask


@torch.no_grad()
def generate_mamba(model, tokenizer, prompt, mask, lam, max_new, temperature, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(enc.input_ids, use_cache=True)
    cache = out.cache_params
    logits = out.logits[0, -1]
    eos = tokenizer.eos_token_id
    generated = []
    for _ in range(max_new):
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
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(enc.input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1]
    eos = tokenizer.eos_token_id
    generated = []
    for _ in range(max_new):
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
    if len(toks) < n: return float("nan")
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams) if grams else float("nan")


def boundary_hit_rate(text, boundary):
    bset = set(w.lower() for w in boundary)
    toks = [t.lower() for t in WORD_RE.findall(text)]
    return sum(1 for t in toks if t in bset) / len(toks) if toks else 0.0


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, MambaForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    aisha = make_aisha()

    backends = {}
    for name, mid, kind in MODELS:
        print(f"[setup] loading {name}", flush=True)
        tok = AutoTokenizer.from_pretrained(mid)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        cls = MambaForCausalLM if kind == "mamba" else AutoModelForCausalLM
        mdl = cls.from_pretrained(mid, torch_dtype=torch.float32).to(device).eval()
        backends[name] = {"model": mdl, "tokenizer": tok, "kind": kind}

    judge_tok = AutoTokenizer.from_pretrained(JUDGE_NAME)
    if judge_tok.pad_token is None: judge_tok.pad_token = judge_tok.eos_token
    judge = AutoModelForCausalLM.from_pretrained(JUDGE_NAME).to(device).eval()

    # 30 prompts (matches D2)
    a_prompts = [ln.strip() for ln in
                  (AISHA_ROOT / "data" / "processed" / "a_prompt_corpus.txt"
                   ).read_text().splitlines()
                  if ln.strip() and len(ln.strip()) > 8][:300]
    rng = np.random.default_rng(0)
    selected = list(rng.choice(a_prompts, size=30, replace=False))

    def make_prompt(t): return f"User: {t}\nAssistant:"

    LAM = 1.5
    conditions = []
    for name in backends:
        conditions.append((name + "_alone",          name, "alone"))
        conditions.append((name + "_uniform_lam1.5",  name, "uniform"))
        conditions.append((name + "_weighted_lam1.5", name, "weighted"))

    rows = []
    for pi, prompt in enumerate(selected):
        words, weights = aisha_weighted_boundary(aisha, prompt)
        ais_resp = aisha_response(aisha, prompt)

        per_prompt = {"prompt": prompt, "boundary_size": len(words),
                      "weight_range": [float(weights.min()) if len(weights) else 0,
                                        float(weights.max()) if len(weights) else 0],
                      "outputs": {"aisha_alone": ais_resp},
                      "metrics": {}}

        # Pre-build masks (one uniform, one weighted) per backend.
        for cond_name, backend_name, kind in conditions:
            be = backends[backend_name]
            V = be["model"].config.vocab_size
            if kind == "alone":
                mask = None; lam_val = 0.0
            elif kind == "uniform":
                if len(words):
                    mask = build_boost_mask(words, np.ones_like(weights),
                                                be["tokenizer"], V, device)
                else:
                    mask = None
                lam_val = LAM
            else:  # weighted
                if len(words):
                    mask = build_boost_mask(words, weights,
                                                be["tokenizer"], V, device)
                else:
                    mask = None
                lam_val = LAM
            torch.manual_seed(0)
            gen = generate_mamba if be["kind"] == "mamba" else generate_causal
            text = gen(be["model"], be["tokenizer"], make_prompt(prompt),
                        mask, lam_val, 24, 0.8, device)
            per_prompt["outputs"][cond_name] = text

        # Metrics for all (including aisha_alone)
        for cn, txt in per_prompt["outputs"].items():
            per_prompt["metrics"][cn] = {
                "judge_ppl":         perplexity(judge, judge_tok, txt, device),
                "distinct_1":        distinct_n(txt, 1),
                "distinct_2":        distinct_n(txt, 2),
                "boundary_hit_rate": boundary_hit_rate(txt, words),
                "n_tokens":          len(WORD_RE.findall(txt)),
            }

        rows.append(per_prompt)
        if pi < 3 or pi == len(selected) - 1:
            print(f"\n[{pi+1}/{len(selected)}] {prompt!r}")
            print(f"  boundary={len(words)}, weights min={per_prompt['weight_range'][0]:.3f} max={per_prompt['weight_range'][1]:.3f}")
            for cn in ["aisha_alone"] + [c for c, _, _ in conditions]:
                m = per_prompt["metrics"][cn]
                t = per_prompt["outputs"][cn]
                print(f"  {cn:>32s} JPPL={m['judge_ppl']:7.1f} d2={m['distinct_2']:.2f} "
                      f"hit={m['boundary_hit_rate']:.2f} L={m['n_tokens']:>3d} | {t[:60]!r}")

    all_conds = ["aisha_alone"] + [c for c, _, _ in conditions]
    summary = {}
    for c in all_conds:
        ms = [r["metrics"][c] for r in rows]
        ppls = [m["judge_ppl"] for m in ms if not np.isnan(m["judge_ppl"])]
        d2s  = [m["distinct_2"] for m in ms if not np.isnan(m["distinct_2"])]
        hits = [m["boundary_hit_rate"] for m in ms]
        lens = [m["n_tokens"] for m in ms]
        summary[c] = {
            "judge_ppl_median":  float(np.median(ppls)) if ppls else float("nan"),
            "judge_ppl_mean":    float(np.mean(ppls)) if ppls else float("nan"),
            "distinct_2_mean":   float(np.mean(d2s)) if d2s else float("nan"),
            "boundary_hit_mean": float(np.mean(hits)),
            "tokens_mean":       float(np.mean(lens)),
        }

    print("\n=== D5 summary (uniform vs weighted boundary, 30 prompts) ===")
    print(f"{'condition':>32s}  {'JPPL_med':>9s}  {'JPPL_mean':>10s}  "
          f"{'d-2':>6s}  {'hit':>6s}  {'tokens':>7s}")
    for c in all_conds:
        s = summary[c]
        print(f"  {c:>30s}  {s['judge_ppl_median']:>9.1f}  {s['judge_ppl_mean']:>10.1f}  "
              f"{s['distinct_2_mean']:>6.3f}  {s['boundary_hit_mean']:>6.3f}  "
              f"{s['tokens_mean']:>7.1f}")

    out = {"summary": summary, "per_prompt": rows}
    out_path = RES_DIR / "d5_weighted_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
