"""
Experiment D7: phone-app-size scan.

Question: at what backbone size does the Aisha + LM hybrid hit the
quality knee that makes it suitable for a standalone phone app?

Phone-app constraints (2025 mid-to-high end):
  - 4-8 GB RAM available for ML
  - storage budget ~1-3 GB for model weights
  - latency target ~20+ tokens/sec

Model sizes scanned:
  - mamba-130m   (we have, sub-floor for hybrid)
  - mamba-370m   (~750 MB fp16, comfortable on phone)
  - pythia-410m  (~820 MB fp16, transformer comparison)
  - pythia-1b   (~2 GB fp16, edge of phone deployment)

For each: alone vs +Aisha lam=1.5 on 50 prompts (subset of D6 100).
Metrics: judge-PPL (quartiles), distinct-2, topic-hit, MB-fp16, GFLOPs/token estimate.
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

N_PROMPTS = 50
LAM = 1.5

MODELS = [
    ("mamba-130m",   "state-spaces/mamba-130m-hf",  "mamba"),
    ("mamba-370m",   "state-spaces/mamba-370m-hf",  "mamba"),
    ("pythia-410m",  "EleutherAI/pythia-410m",       "causal"),
    ("pythia-1b",    "EleutherAI/pythia-1b",         "causal"),
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
def gen_mamba(model, tokenizer, prompt, mask, lam, max_new, T, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(enc.input_ids, use_cache=True)
    cache = out.cache_params
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
        out = model(nid.unsqueeze(0), cache_params=cache, use_cache=True)
        cache = out.cache_params
        logits = out.logits[0, -1]
    return tokenizer.decode(g, skip_special_tokens=True)


@torch.no_grad()
def gen_causal(model, tokenizer, prompt, mask, lam, max_new, T, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(enc.input_ids, use_cache=True)
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


@torch.no_grad()
def time_one_token(model, tokenizer, kind, device, n_warm=2, n_meas=5):
    """Mean time to generate one token, given a small prompt."""
    enc = tokenizer("Hello there.", return_tensors="pt").to(device)
    if kind == "mamba":
        out = model(enc.input_ids, use_cache=True)
        cache = out.cache_params
        nid = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    else:
        out = model(enc.input_ids, use_cache=True)
        past = out.past_key_values
        nid = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    if device == "cuda":
        torch.cuda.synchronize()
    for _ in range(n_warm):
        if kind == "mamba":
            out = model(nid, cache_params=cache, use_cache=True)
            cache = out.cache_params
        else:
            out = model(nid, past_key_values=past, use_cache=True)
            past = out.past_key_values
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_meas):
        if kind == "mamba":
            out = model(nid, cache_params=cache, use_cache=True)
            cache = out.cache_params
        else:
            out = model(nid, past_key_values=past, use_cache=True)
            past = out.past_key_values
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n_meas


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, MambaForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    aisha = make_aisha()

    backends = {}
    for name, mid, kind in MODELS:
        print(f"[setup] {name} ({mid})", flush=True)
        try:
            tok = AutoTokenizer.from_pretrained(mid)
        except Exception as e:
            print(f"  tokenizer failed: {e}")
            continue
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        cls = MambaForCausalLM if kind == "mamba" else AutoModelForCausalLM
        try:
            mdl = cls.from_pretrained(mid, torch_dtype=torch.float32).to(device).eval()
        except Exception as e:
            print(f"  model load failed: {e}")
            continue
        n_params = sum(p.numel() for p in mdl.parameters())
        # GPU latency per token (proxy for phone scaling)
        try:
            t_per_tok = time_one_token(mdl, tok, kind, device)
        except Exception as e:
            t_per_tok = float("nan")
        print(f"  {name}: {n_params/1e6:.1f}M params, "
              f"~{n_params*2/1e6:.0f} MB fp16, "
              f"{t_per_tok*1000:.1f} ms/token (GPU)")
        backends[name] = {
            "model": mdl, "tokenizer": tok, "kind": kind,
            "params": n_params, "mb_fp16": n_params * 2 / 1e6,
            "ms_per_token_gpu": t_per_tok * 1000,
        }

    judge_tok = AutoTokenizer.from_pretrained(JUDGE_NAME)
    if judge_tok.pad_token is None: judge_tok.pad_token = judge_tok.eos_token
    judge = AutoModelForCausalLM.from_pretrained(JUDGE_NAME).to(device).eval()

    a_prompts = [ln.strip() for ln in
                  (AISHA_ROOT / "data" / "processed" / "a_prompt_corpus.txt"
                   ).read_text().splitlines()
                  if ln.strip() and len(ln.strip()) > 8]
    rng = np.random.default_rng(0)
    selected = list(rng.choice(a_prompts, size=min(N_PROMPTS, len(a_prompts)), replace=False))
    print(f"[setup] {len(selected)} prompts")

    def make_prompt(t): return f"User: {t}\nAssistant:"

    conditions = ["aisha_alone"]
    for n in backends:
        conditions += [n + "_alone", n + "_aisha_lam1.5"]

    rows = []
    t0 = time.time()
    for pi, prompt in enumerate(selected):
        boundary = aisha_boundary_words(aisha, prompt)
        ais = aisha_response(aisha, prompt)
        per_prompt = {"prompt": str(prompt), "boundary_size": len(boundary),
                       "outputs": {"aisha_alone": ais}, "metrics": {}}
        for bn in backends:
            be = backends[bn]
            V = be["model"].config.vocab_size
            mask = build_boost_mask(boundary, be["tokenizer"], V, device) if boundary else None
            gen = gen_mamba if be["kind"] == "mamba" else gen_causal
            torch.manual_seed(0)
            t_alone = gen(be["model"], be["tokenizer"], make_prompt(prompt),
                           None, 0.0, 24, 0.8, device)
            torch.manual_seed(0)
            t_hyb = gen(be["model"], be["tokenizer"], make_prompt(prompt),
                         mask, LAM, 24, 0.8, device)
            per_prompt["outputs"][bn + "_alone"] = t_alone
            per_prompt["outputs"][bn + "_aisha_lam1.5"] = t_hyb

        for cn, txt in per_prompt["outputs"].items():
            per_prompt["metrics"][cn] = {
                "judge_ppl":         perplexity(judge, judge_tok, txt, device),
                "distinct_2":        distinct_n(txt, 2),
                "boundary_hit_rate": boundary_hit_rate(txt, boundary),
                "n_tokens":          len(WORD_RE.findall(txt)),
            }
        rows.append(per_prompt)
        if (pi + 1) % 10 == 0:
            print(f"  [{pi+1}/{len(selected)}] elapsed {time.time()-t0:.0f}s")

    print(f"[done] {time.time()-t0:.0f}s")

    summary = {}
    for c in conditions:
        ms = [r["metrics"][c] for r in rows]
        ppls = np.array([m["judge_ppl"] for m in ms if not np.isnan(m["judge_ppl"])])
        d2s  = np.array([m["distinct_2"] for m in ms if not np.isnan(m["distinct_2"])])
        hits = np.array([m["boundary_hit_rate"] for m in ms])
        lens = np.array([m["n_tokens"] for m in ms])
        summary[c] = {
            "judge_ppl_p25":     float(np.percentile(ppls, 25)) if ppls.size else float("nan"),
            "judge_ppl_median":  float(np.median(ppls)) if ppls.size else float("nan"),
            "judge_ppl_p75":     float(np.percentile(ppls, 75)) if ppls.size else float("nan"),
            "judge_ppl_mean":    float(np.mean(ppls)) if ppls.size else float("nan"),
            "distinct_2_mean":   float(np.mean(d2s)) if d2s.size else float("nan"),
            "boundary_hit_mean": float(np.mean(hits)),
            "tokens_mean":       float(np.mean(lens)),
        }

    # Add backend metadata to summary
    backend_info = {n: {
        "params":           int(backends[n]["params"]),
        "mb_fp16":          backends[n]["mb_fp16"],
        "ms_per_token_gpu": backends[n]["ms_per_token_gpu"],
    } for n in backends}

    print("\n=== D7 phone-size scan (50 prompts, judge=GPT-2) ===")
    print(f"{'condition':>32s}  {'params':>7s}  {'MB_fp16':>8s}  "
           f"{'PPL_p25':>9s}  {'PPL_med':>9s}  {'PPL_p75':>9s}  "
           f"{'d-2':>6s}  {'hit':>6s}")
    for c in conditions:
        s = summary[c]
        if c == "aisha_alone":
            params = "16D"; mb = "—"
        else:
            bn = c.replace("_alone", "").replace("_aisha_lam1.5", "")
            params = f"{backend_info[bn]['params']/1e6:.0f}M"
            mb = f"{backend_info[bn]['mb_fp16']:.0f}"
        print(f"  {c:>30s}  {params:>7s}  {mb:>8s}  "
              f"{s['judge_ppl_p25']:>9.1f}  {s['judge_ppl_median']:>9.1f}  "
              f"{s['judge_ppl_p75']:>9.1f}  "
              f"{s['distinct_2_mean']:>6.3f}  {s['boundary_hit_mean']:>6.3f}")

    out = {"n_prompts": len(rows), "lambda": LAM, "models": MODELS,
           "backend_info": backend_info, "summary": summary, "per_prompt": rows}
    out_path = RES_DIR / "d7_phone_size_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
