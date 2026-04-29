"""
Experiment D11b: same test as D11, with Qwen2.5-0.5B-Instruct as the
phone-tier backbone instead of Pythia-410m base.

Hypothesis: an instruction-tuned model at the same parameter count
will dramatically outperform the base model on source-grounded QA,
because it's actually trained to answer questions rather than continue
text. We use the SAME Aisha boundary mechanism as the production phone
config (logit bias at lambda=1.5).

If Qwen-0.5B + Aisha boundary >> Pythia-410m + Aisha boundary, the
phone-tier backbone choice is the action item.
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

sys.path.insert(0, str(EXP_ROOT))
from d9_hallucination_test import TRIPLES, make_aisha, aisha_structure, judge_match
from d11_phone_halluc import aisha_boundary_words, build_boost_mask

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
RES_DIR = EXP_ROOT.parent / "results"
WORD_RE = re.compile(r"[a-zA-Z']+")

BACKBONE_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LAM = 1.5


@torch.no_grad()
def generate_qwen(model, tokenizer, prompt: str, mask, lam: float,
                   max_new: int = 40, temperature: float = 0.3,
                   device: str = "cuda") -> str:
    """Same logit-bias generation as Pythia, but uses chat-template prompt."""
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(enc.input_ids, attention_mask=enc.attention_mask, use_cache=True)
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


def chat_prompt(tokenizer, user_msg: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[d11b] device: {device}")
    aisha = make_aisha()

    print(f"[d11b] loading {BACKBONE_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        BACKBONE_NAME, torch_dtype=torch.float16, device_map="auto").eval()
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"[d11b] loaded, {n_params/1e6:.1f}M params, vocab={backbone.config.vocab_size}")

    rows = []
    t0 = time.time()
    for ti, t in enumerate(TRIPLES):
        boundary = aisha_boundary_words(aisha, t["question"])
        struct_source = aisha_structure(aisha, t["source"])
        mask = (build_boost_mask(boundary, tok, backbone.config.vocab_size, device)
                if boundary else None)

        outs = {}
        prompt_no = chat_prompt(tok, f"Answer the question briefly with one word or short phrase.\n\nQ: {t['question']}")
        prompt_src = chat_prompt(tok, f"Read the document and answer the question. Use only information from the document.\n\nDocument:\n{t['source']}\n\nQ: {t['question']}")

        torch.manual_seed(0); outs["A_no_source"]    = generate_qwen(backbone, tok, prompt_no, None, 0.0, 40, 0.3, device)
        torch.manual_seed(0); outs["B_source"]       = generate_qwen(backbone, tok, prompt_src, None, 0.0, 40, 0.3, device)
        torch.manual_seed(0); outs["C_source_aisha"] = generate_qwen(backbone, tok, prompt_src, mask, LAM,  40, 0.3, device)

        ans_fps = {c: aisha_structure(aisha, txt) for c, txt in outs.items()}
        dists = {c: float(np.linalg.norm(fp["doc_centroid"] - struct_source["doc_centroid"]))
                  if (fp is not None and struct_source is not None) else float("nan")
                  for c, fp in ans_fps.items()}
        match_source = {c: judge_match(None, None, txt, t["source_fact"], None) for c, txt in outs.items()}
        match_world  = {c: judge_match(None, None, txt, t["world_fact"], None)  for c, txt in outs.items()}

        rows.append({"question": t["question"], "source_fact": t["source_fact"],
                      "world_fact": t["world_fact"], "outputs": outs,
                      "centroid_dist": dists,
                      "match_source": match_source, "match_world": match_world})

        if ti < 4 or ti == len(TRIPLES) - 1:
            print(f"\n[{ti+1}/{len(TRIPLES)}] Q: {t['question'][:60]}")
            print(f"  source_fact: {t['source_fact']}, world_fact: {t['world_fact']}")
            for c in ["A_no_source", "B_source", "C_source_aisha"]:
                ms, mw = match_source[c], match_world[c]
                tag = ("FAITHFUL" if (ms and not mw) else "HALLUC" if (mw and not ms)
                       else "BOTH" if (ms and mw) else "OTHER")
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
            "match_source_rate": float(ms.mean()),
            "match_world_rate":  float(mw.mean()),
            "faithful_only_rate":  float(faithful.mean()),
            "hallucinated_only_rate": float(hallucinated.mean()),
            "both_rate":   float(both.mean()),
            "neither_rate": float(neither.mean()),
            "dist_median_faithful":     float(np.median(ds[faithful & ~np.isnan(ds)]))     if (faithful & ~np.isnan(ds)).any()     else float("nan"),
            "dist_median_hallucinated": float(np.median(ds[hallucinated & ~np.isnan(ds)])) if (hallucinated & ~np.isnan(ds)).any() else float("nan"),
        }

    print(f"\n=== D11b summary (Qwen2.5-0.5B-Instruct + Aisha boundary) ===")
    print(f"{'condition':>20s}  {'src':>5s}  {'wrld':>5s}  {'faithful':>9s}  "
          f"{'hallu':>7s}  {'both':>6s}  {'neither':>8s}")
    for cond in ["A_no_source", "B_source", "C_source_aisha"]:
        s = summary[cond]
        print(f"  {cond:>18s}  {s['match_source_rate']:>5.2f}  {s['match_world_rate']:>5.2f}  "
              f"{s['faithful_only_rate']:>9.2f}  {s['hallucinated_only_rate']:>7.2f}  "
              f"{s['both_rate']:>6.2f}  {s['neither_rate']:>8.2f}")

    out_path = RES_DIR / "d11b_qwen_phone_results.json"
    out_path.write_text(json.dumps({
        "n_triples": len(rows), "backbone": BACKBONE_NAME, "lambda": LAM,
        "summary": summary, "per_triple": rows,
    }, indent=2, default=str))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
