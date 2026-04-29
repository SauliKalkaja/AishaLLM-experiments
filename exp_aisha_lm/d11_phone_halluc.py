"""
Experiment D11: phone-tier hallucination test.

Tests what's actually shipping in the phone app: Pythia-410m (base, not
instruction-tuned) plus Aisha boundary as logit bias (lambda=1.5).

This is fundamentally different from D9/D10:
  - D9/D10 used instruction-tuned Qwen, fed instructions, used structural
    text prefixes. Modern instruction tuning makes hallucination rare.
  - D11 uses a BASE language model. It doesn't know "follow the source"
    means follow the source. It does next-token prediction on whatever
    text it is given. Hallucination patterns are completely different.

For each contradictory triple, three conditions:
  A. Pythia-410m alone, no source     : free generation
  B. Pythia-410m + raw source         : source as context, no Aisha
  C. Pythia-410m + raw source + Aisha boundary at lam=1.5

We measure faithfulness and hallucination rate, plus Aisha centroid
distance answer-to-source. Honest measurement of what the production
phone-app config actually does.
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
from d9_hallucination_test import (
    TRIPLES, make_aisha, aisha_structure, judge_match,
)

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
RES_DIR = EXP_ROOT.parent / "results"
WORD_RE = re.compile(r"[a-zA-Z']+")

BACKBONE_NAME = "EleutherAI/pythia-410m"
LAM = 1.5


def aisha_boundary_words(responder, query):
    """Reuse from D1-style boundary: Aisha's input-conditioned candidate words."""
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
def generate_pythia(model, tokenizer, prompt: str, mask, lam: float,
                    max_new: int = 40, temperature: float = 0.4,
                    device: str = "cuda") -> str:
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
        # also stop on newline+newline (Pythia tends to ramble)
        g.append(int(nid))
        out = model(nid.unsqueeze(0), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
    return tokenizer.decode(g, skip_special_tokens=True)


def make_prompt(question: str, source: str = None) -> str:
    """Plain-text prompt for a base model. No chat template; we just give it
    text it should continue. The format mimics web-text Q&A."""
    if source is None:
        return f"Question: {question}\nAnswer:"
    return f"Document:\n{source}\n\nBased only on the document above, answer the question.\nQuestion: {question}\nAnswer:"


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[d11] device: {device}")
    aisha = make_aisha()

    print(f"[d11] loading {BACKBONE_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        BACKBONE_NAME, torch_dtype=torch.float32, device_map="auto").eval()
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"[d11] loaded, {n_params/1e6:.1f}M params, vocab={backbone.config.vocab_size}")

    rows = []
    t0 = time.time()
    for ti, t in enumerate(TRIPLES):
        # Aisha boundary from the question (input-conditioned)
        boundary = aisha_boundary_words(aisha, t["question"])
        struct_source = aisha_structure(aisha, t["source"])
        mask = (build_boost_mask(boundary, tok, backbone.config.vocab_size, device)
                if boundary else None)

        outs = {}
        torch.manual_seed(0)
        outs["A_no_source"] = generate_pythia(backbone, tok,
                                make_prompt(t["question"], None),
                                None, 0.0, 40, 0.4, device)
        torch.manual_seed(0)
        outs["B_source"] = generate_pythia(backbone, tok,
                                make_prompt(t["question"], t["source"]),
                                None, 0.0, 40, 0.4, device)
        torch.manual_seed(0)
        outs["C_source_aisha"] = generate_pythia(backbone, tok,
                                make_prompt(t["question"], t["source"]),
                                mask, LAM, 40, 0.4, device)

        # Aisha fingerprint each answer
        ans_fps = {c: aisha_structure(aisha, txt) for c, txt in outs.items()}
        dists = {}
        for c, fp in ans_fps.items():
            if fp is None or struct_source is None:
                dists[c] = float("nan")
            else:
                dists[c] = float(np.linalg.norm(fp["doc_centroid"] - struct_source["doc_centroid"]))

        match_source = {c: judge_match(None, None, txt, t["source_fact"], None)
                          for c, txt in outs.items()}
        match_world = {c: judge_match(None, None, txt, t["world_fact"], None)
                         for c, txt in outs.items()}

        rows.append({
            "question": t["question"], "source": t["source"],
            "source_fact": t["source_fact"], "world_fact": t["world_fact"],
            "boundary_size": len(boundary), "outputs": outs,
            "centroid_dist": dists,
            "match_source": match_source, "match_world": match_world,
        })

        if ti < 4 or ti == len(TRIPLES) - 1:
            print(f"\n[{ti+1}/{len(TRIPLES)}] Q: {t['question'][:60]}")
            print(f"  source_fact: {t['source_fact']}, world_fact: {t['world_fact']}")
            for c in ["A_no_source", "B_source", "C_source_aisha"]:
                ms, mw = match_source[c], match_world[c]
                tag = ("FAITHFUL" if (ms and not mw) else
                       "HALLUC"   if (mw and not ms) else
                       "BOTH"     if (ms and mw)     else "OTHER")
                d = dists[c]; ds = f"{d:.2f}" if not np.isnan(d) else "—"
                print(f"  {c:>16s} src={ms} wrld={mw} dist={ds} [{tag}] | {outs[c][:80]!r}")

    summary = {}
    for cond in ["A_no_source", "B_source", "C_source_aisha"]:
        ms = np.array([r["match_source"][cond] for r in rows])
        mw = np.array([r["match_world"][cond]  for r in rows])
        ds = np.array([r["centroid_dist"][cond] for r in rows])
        faithful     = (ms == 1) & (mw == 0)
        hallucinated = (ms == 0) & (mw == 1)
        both = (ms == 1) & (mw == 1); neither = (ms == 0) & (mw == 0)
        summary[cond] = {
            "n": int(len(ms)),
            "match_source_rate": float(ms.mean()),
            "match_world_rate":  float(mw.mean()),
            "faithful_only_rate":  float(faithful.mean()),
            "hallucinated_only_rate": float(hallucinated.mean()),
            "both_rate":  float(both.mean()),
            "neither_rate": float(neither.mean()),
            "dist_median_faithful":     float(np.median(ds[faithful & ~np.isnan(ds)]))     if (faithful & ~np.isnan(ds)).any()     else float("nan"),
            "dist_median_hallucinated": float(np.median(ds[hallucinated & ~np.isnan(ds)])) if (hallucinated & ~np.isnan(ds)).any() else float("nan"),
            "dist_median_overall":      float(np.median(ds[~np.isnan(ds)])) if (~np.isnan(ds)).any() else float("nan"),
        }

    print(f"\n=== D11 summary (Pythia-410m base, contradictory triples) ===")
    print(f"{'condition':>20s}  {'src':>5s}  {'wrld':>5s}  {'faithful':>9s}  "
          f"{'hallu':>7s}  {'both':>6s}  {'neither':>8s}  {'dist_F':>7s}  {'dist_H':>7s}")
    for cond in ["A_no_source", "B_source", "C_source_aisha"]:
        s = summary[cond]
        df = s['dist_median_faithful']; dh = s['dist_median_hallucinated']
        df_str = f"{df:.3f}" if not np.isnan(df) else "—"
        dh_str = f"{dh:.3f}" if not np.isnan(dh) else "—"
        print(f"  {cond:>18s}  {s['match_source_rate']:>5.2f}  {s['match_world_rate']:>5.2f}  "
              f"{s['faithful_only_rate']:>9.2f}  {s['hallucinated_only_rate']:>7.2f}  "
              f"{s['both_rate']:>6.2f}  {s['neither_rate']:>8.2f}  {df_str:>7s}  {dh_str:>7s}")

    # Aggregate detector signal
    print("\n=== Aggregate hallucination signal (B + C combined) ===")
    F_, H_ = [], []
    for cond in ["B_source", "C_source_aisha"]:
        for r in rows:
            ms, mw = r["match_source"][cond], r["match_world"][cond]
            d = r["centroid_dist"][cond]
            if np.isnan(d): continue
            if ms and not mw: F_.append(d)
            elif mw and not ms: H_.append(d)
    if F_:
        print(f"  faithful dist:      n={len(F_)}  median={np.median(F_):.3f}  mean={np.mean(F_):.3f}")
    else:
        print(f"  faithful: no samples")
    if H_:
        print(f"  hallucinated dist:  n={len(H_)}  median={np.median(H_):.3f}  mean={np.mean(H_):.3f}")
    else:
        print(f"  hallucinated: no samples")
    if len(F_) >= 5 and len(H_) >= 5:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(H_, F_, alternative="greater")
        n1, n2 = len(H_), len(F_)
        rb = 1 - (2 * u) / (n1 * n2)
        print(f"  Mann-Whitney U (hallucinated > faithful): U={u:.0f}, p={p:.3g}")
        print(f"  rank-biserial effect: {rb:+.3f}")
        thresh = float(np.percentile(F_, 75))
        tp = sum(1 for d in H_ if d > thresh)
        fp = sum(1 for d in F_ if d > thresh)
        print(f"\n  Detector at threshold = p75-of-faithful = {thresh:.3f}:")
        print(f"    flagged faithful (false positives): {fp}/{n2}  ({fp/max(n2,1)*100:.0f}%)")
        print(f"    flagged hallucinated (true positives): {tp}/{n1}  ({tp/max(n1,1)*100:.0f}%)")
        print(f"    recall {tp/max(n1,1):.2f}, precision {tp/max(tp+fp,1):.2f}")

    out_path = RES_DIR / "d11_phone_halluc_results.json"
    out_path.write_text(json.dumps({
        "n_triples": len(rows), "backbone": BACKBONE_NAME, "lambda": LAM,
        "summary": summary, "per_triple": rows,
    }, indent=2, default=str))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
