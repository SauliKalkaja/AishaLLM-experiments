"""
Experiment D6: 100-prompt scaled evaluation of the Aisha + LM hybrid.

Five conditions across 100 prompts from a_prompt_corpus.txt:
  - aisha_alone                   (current pipeline baseline)
  - mamba-130m_alone
  - mamba-130m + Aisha lam=1.5    (D1 sweet spot)
  - pythia-160m_alone             (D2's better backbone)
  - pythia-160m + Aisha lam=1.5   (proposed best hybrid)

Per output:
  - judge-PPL          (GPT-2 small)
  - distinct-1, -2     (penalises repetition)
  - boundary hit rate  (topic relevance)
  - response length

Aggregates: median, mean, std across 100 prompts. Also dumps a clean
side-by-side sample of 25 prompts to a CSV-style human-readable file
for the user to spot-check.

This is the most rigorous test we can run automatically. Everything
that follows is human qualitative judgment.
"""
import csv
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

N_PROMPTS = 100
LAM = 1.5

MODELS = [
    ("mamba-130m", "state-spaces/mamba-130m-hf", "mamba"),
    ("pythia-160m", "EleutherAI/pythia-160m", "causal"),
]
JUDGE_NAME = "gpt2"


def make_aisha():
    from responder_pos import POSResponder
    return POSResponder(use_harper=False)


def aisha_boundary_words(responder, query):
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


def aisha_response(responder, query):
    out = responder.respond(query)
    return out["text"] if isinstance(out, dict) else str(out)


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
def generate_mamba(model, tokenizer, prompt, mask, lam, max_new, temperature, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(enc.input_ids, use_cache=True)
    cache = out.cache_params
    logits = out.logits[0, -1]
    eos = tokenizer.eos_token_id
    g = []
    for _ in range(max_new):
        if mask is not None and lam != 0.0:
            logits = logits + lam * mask
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        nid = torch.multinomial(probs, 1)
        if eos is not None and int(nid) == eos:
            break
        g.append(int(nid))
        out = model(nid.unsqueeze(0), cache_params=cache, use_cache=True)
        cache = out.cache_params
        logits = out.logits[0, -1]
    return tokenizer.decode(g, skip_special_tokens=True)


@torch.no_grad()
def generate_causal(model, tokenizer, prompt, mask, lam, max_new, temperature, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(enc.input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1]
    eos = tokenizer.eos_token_id
    g = []
    for _ in range(max_new):
        if mask is not None and lam != 0.0:
            logits = logits + lam * mask
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        nid = torch.multinomial(probs, 1)
        if eos is not None and int(nid) == eos:
            break
        g.append(int(nid))
        out = model(nid.unsqueeze(0), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
    return tokenizer.decode(g, skip_special_tokens=True)


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
        print(f"[setup] {name}", flush=True)
        tok = AutoTokenizer.from_pretrained(mid)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        cls = MambaForCausalLM if kind == "mamba" else AutoModelForCausalLM
        mdl = cls.from_pretrained(mid, torch_dtype=torch.float32).to(device).eval()
        backends[name] = {"model": mdl, "tokenizer": tok, "kind": kind}

    judge_tok = AutoTokenizer.from_pretrained(JUDGE_NAME)
    if judge_tok.pad_token is None: judge_tok.pad_token = judge_tok.eos_token
    judge = AutoModelForCausalLM.from_pretrained(JUDGE_NAME).to(device).eval()

    a_prompts = [ln.strip() for ln in
                  (AISHA_ROOT / "data" / "processed" / "a_prompt_corpus.txt"
                   ).read_text().splitlines()
                  if ln.strip() and len(ln.strip()) > 8]
    rng = np.random.default_rng(0)
    selected = list(rng.choice(a_prompts, size=min(N_PROMPTS, len(a_prompts)), replace=False))
    print(f"[setup] selected {len(selected)} prompts (of {len(a_prompts)} available)")

    def make_prompt(t): return f"User: {t}\nAssistant:"

    conditions = ["aisha_alone"]
    for n in backends:
        conditions += [n + "_alone", n + "_aisha_lam1.5"]

    rows = []
    t0 = time.time()
    for pi, prompt in enumerate(selected):
        boundary = aisha_boundary_words(aisha, prompt)
        ais_resp = aisha_response(aisha, prompt)

        per_prompt = {"prompt": str(prompt), "boundary_size": len(boundary),
                      "outputs": {"aisha_alone": ais_resp},
                      "metrics": {}}

        for backend_name in backends:
            be = backends[backend_name]
            V = be["model"].config.vocab_size
            mask = (build_boost_mask(boundary, be["tokenizer"], V, device)
                    if boundary else None)
            torch.manual_seed(0)
            gen = generate_mamba if be["kind"] == "mamba" else generate_causal

            text_alone = gen(be["model"], be["tokenizer"], make_prompt(prompt),
                              None, 0.0, 24, 0.8, device)
            torch.manual_seed(0)
            text_hyb = gen(be["model"], be["tokenizer"], make_prompt(prompt),
                            mask, LAM, 24, 0.8, device)
            per_prompt["outputs"][backend_name + "_alone"] = text_alone
            per_prompt["outputs"][backend_name + "_aisha_lam1.5"] = text_hyb

        for cn, txt in per_prompt["outputs"].items():
            per_prompt["metrics"][cn] = {
                "judge_ppl":         perplexity(judge, judge_tok, txt, device),
                "distinct_1":        distinct_n(txt, 1),
                "distinct_2":        distinct_n(txt, 2),
                "boundary_hit_rate": boundary_hit_rate(txt, boundary),
                "n_tokens":          len(WORD_RE.findall(txt)),
            }
        rows.append(per_prompt)

        if (pi + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{pi+1}/{len(selected)}] elapsed {elapsed:.0f}s "
                   f"(~{elapsed/(pi+1):.1f}s/prompt)")

    print(f"\n[done] {len(rows)} prompts in {time.time()-t0:.0f}s")

    summary = {}
    for c in conditions:
        ms = [r["metrics"][c] for r in rows]
        ppls = np.array([m["judge_ppl"]   for m in ms if not np.isnan(m["judge_ppl"])])
        d1s  = np.array([m["distinct_1"]  for m in ms if not np.isnan(m["distinct_1"])])
        d2s  = np.array([m["distinct_2"]  for m in ms if not np.isnan(m["distinct_2"])])
        hits = np.array([m["boundary_hit_rate"] for m in ms])
        lens = np.array([m["n_tokens"] for m in ms])
        summary[c] = {
            "judge_ppl_median":  float(np.median(ppls)) if ppls.size else float("nan"),
            "judge_ppl_mean":    float(np.mean(ppls))   if ppls.size else float("nan"),
            "judge_ppl_p25":     float(np.percentile(ppls, 25)) if ppls.size else float("nan"),
            "judge_ppl_p75":     float(np.percentile(ppls, 75)) if ppls.size else float("nan"),
            "distinct_1_mean":   float(np.mean(d1s))   if d1s.size  else float("nan"),
            "distinct_2_mean":   float(np.mean(d2s))   if d2s.size  else float("nan"),
            "boundary_hit_mean": float(np.mean(hits)),
            "boundary_hit_p75":  float(np.percentile(hits, 75)),
            "tokens_mean":       float(np.mean(lens)),
            "tokens_median":     float(np.median(lens)),
            "n":                 int(len(rows)),
        }

    print("\n=== D6 summary (100 prompts, judge=GPT-2) ===")
    print(f"{'condition':>32s}  {'PPL_p25':>9s}  {'PPL_med':>9s}  {'PPL_p75':>9s}  "
          f"{'PPL_mean':>10s}  {'d-2':>6s}  {'hit':>6s}  {'tokens':>7s}")
    for c in conditions:
        s = summary[c]
        print(f"  {c:>30s}  {s['judge_ppl_p25']:>9.1f}  {s['judge_ppl_median']:>9.1f}  "
              f"{s['judge_ppl_p75']:>9.1f}  {s['judge_ppl_mean']:>10.1f}  "
              f"{s['distinct_2_mean']:>6.3f}  {s['boundary_hit_mean']:>6.3f}  "
              f"{s['tokens_mean']:>7.1f}")

    # Wins-vs-baseline counts
    print("\n=== per-prompt win rate vs baseline ===")
    base = "pythia-160m_alone"
    for c in conditions:
        if c == base:
            continue
        wins_ppl = sum(1 for r in rows
                       if (not np.isnan(r["metrics"][c]["judge_ppl"]))
                       and (not np.isnan(r["metrics"][base]["judge_ppl"]))
                       and r["metrics"][c]["judge_ppl"] < r["metrics"][base]["judge_ppl"])
        wins_hit = sum(1 for r in rows
                       if r["metrics"][c]["boundary_hit_rate"] >
                          r["metrics"][base]["boundary_hit_rate"])
        n_valid = sum(1 for r in rows
                       if (not np.isnan(r["metrics"][c]["judge_ppl"]))
                       and (not np.isnan(r["metrics"][base]["judge_ppl"])))
        print(f"  {c:>30s}: PPL wins {wins_ppl}/{n_valid}, "
              f"hit wins {wins_hit}/{len(rows)}")

    out = {
        "n_prompts": len(rows),
        "conditions": conditions,
        "lambda": LAM,
        "summary": summary,
        "per_prompt": rows,
    }
    out_path = RES_DIR / "d6_scaled_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  saved: {out_path}")

    # Side-by-side spot-check sample (25 prompts).
    sample_idxs = list(rng.choice(len(rows), size=min(25, len(rows)), replace=False))
    sample_path = RES_DIR / "d6_samples.txt"
    with open(sample_path, "w") as f:
        f.write(f"D6 spot-check sample ({len(sample_idxs)} of {len(rows)} prompts)\n")
        f.write("=" * 88 + "\n\n")
        for k, i in enumerate(sample_idxs):
            r = rows[i]
            f.write(f"--- {k+1}. PROMPT: {r['prompt']}\n")
            f.write(f"    boundary={r['boundary_size']}\n\n")
            for cn in conditions:
                t = r["outputs"][cn].split("\n", 1)[0].strip()
                m = r["metrics"][cn]
                f.write(f"  {cn:>30s}  PPL={m['judge_ppl']:6.1f} hit={m['boundary_hit_rate']:.2f}\n")
                f.write(f"    {t!r}\n")
            f.write("\n")
    print(f"  saved: {sample_path}")


if __name__ == "__main__":
    main()
