"""
Experiment D3: independent judge LM on D1 outputs.

D1 used Mamba-130m's own perplexity as a fluency proxy. That's biased -- a
model rates outputs that look like its own training distribution as fluent.
The harder test: re-score every D1 output through a model with a different
architecture (GPT-2 small, transformer, ~124M params).

If GPT-2's ranking agrees with Mamba's self-PPL ranking (lambda=1.5 best),
the D1 conclusion is robust. If it disagrees, we've calibrated how much of
D1 was self-rating bias.
"""
import json
import os
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
CACHE_DIR = EXP_ROOT.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)

import numpy as np
import torch
import torch.nn.functional as F


JUDGE_NAME = "gpt2"        # 124M params; different architecture from Mamba.

RES_DIR = EXP_ROOT.parent / "results"
D1_PATH = RES_DIR / "d1_hybrid_results.json"


@torch.no_grad()
def perplexity(model, tokenizer, text: str, device: str) -> float:
    """Standard next-token perplexity."""
    if not text or not text.strip():
        return float("nan")
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    if enc.input_ids.shape[1] < 2:
        return float("nan")
    out = model(enc.input_ids, labels=enc.input_ids)
    return float(np.exp(out.loss.item()))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[d3] device: {device}")
    print(f"[d3] loading judge: {JUDGE_NAME}")
    tok = AutoTokenizer.from_pretrained(JUDGE_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    judge = AutoModelForCausalLM.from_pretrained(JUDGE_NAME).to(device)
    judge.eval()

    print(f"[d3] loading D1 results from {D1_PATH}")
    d1 = json.loads(D1_PATH.read_text())
    rows = d1["per_prompt"]
    conditions = list(rows[0]["outputs"].keys())
    print(f"[d3] {len(rows)} prompts, conditions: {conditions}")

    judge_ppl_per_cond = {c: [] for c in conditions}
    self_ppl_per_cond  = {c: [] for c in conditions}
    for r in rows:
        for c in conditions:
            text = r["outputs"][c]
            jp = perplexity(judge, tok, text, device)
            judge_ppl_per_cond[c].append(jp)
            self_ppl_per_cond[c].append(r["metrics"][c]["perplexity"])

    print("\n=== D3 judge LM (gpt2) vs D1 self-PPL (mamba-130m) ===")
    print(f"{'condition':>16s}  {'self_med':>10s}  {'judge_med':>10s}  "
          f"{'self_mean':>10s}  {'judge_mean':>10s}  {'rank':>5s}")
    summary = {}
    self_med_pairs   = sorted([(np.median([p for p in self_ppl_per_cond[c]
                                            if not np.isnan(p)]), c)
                                for c in conditions])
    judge_med_pairs  = sorted([(np.median([p for p in judge_ppl_per_cond[c]
                                            if not np.isnan(p)]), c)
                                for c in conditions])
    self_rank   = {c: i + 1 for i, (_, c) in enumerate(self_med_pairs)}
    judge_rank  = {c: i + 1 for i, (_, c) in enumerate(judge_med_pairs)}
    for c in conditions:
        sj = [p for p in self_ppl_per_cond[c]  if not np.isnan(p)]
        jp = [p for p in judge_ppl_per_cond[c] if not np.isnan(p)]
        s_med, j_med = float(np.median(sj)), float(np.median(jp))
        s_mn,  j_mn  = float(np.mean(sj)),   float(np.mean(jp))
        summary[c] = {
            "self_ppl_median":  s_med,
            "self_ppl_mean":    s_mn,
            "judge_ppl_median": j_med,
            "judge_ppl_mean":   j_mn,
            "self_rank":        self_rank[c],
            "judge_rank":       judge_rank[c],
        }
        print(f"  {c:>14s}  {s_med:>10.1f}  {j_med:>10.1f}  "
              f"{s_mn:>10.1f}  {j_mn:>10.1f}  "
              f"{self_rank[c]}->{judge_rank[c]}")

    # Spearman rank correlation between self-PPL and judge-PPL across prompts
    # within each condition (do they agree on which prompts are easy/hard?).
    from scipy.stats import spearmanr
    print("\n=== Spearman rho (per-prompt self vs judge), per condition ===")
    for c in conditions:
        sj = np.array(self_ppl_per_cond[c])
        jp = np.array(judge_ppl_per_cond[c])
        mask = ~np.isnan(sj) & ~np.isnan(jp)
        if mask.sum() < 4:
            print(f"  {c}: insufficient data")
            continue
        rho, p = spearmanr(sj[mask], jp[mask])
        print(f"  {c:>16s}: rho={rho:.3f}, p={p:.3g}")
        summary[c]["spearman_rho_self_vs_judge"] = float(rho)
        summary[c]["spearman_pvalue"]            = float(p)

    out = {"judge_model": JUDGE_NAME, "n_prompts": len(rows), "summary": summary,
            "self_ranking_by_median":  [c for _, c in self_med_pairs],
            "judge_ranking_by_median": [c for _, c in judge_med_pairs]}
    out_path = RES_DIR / "d3_judge_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  saved: {out_path}")

    print("\nRanking agreement:")
    print(f"  self ranking by median PPL:  {[c for _, c in self_med_pairs]}")
    print(f"  judge ranking by median PPL: {[c for _, c in judge_med_pairs]}")
    if [c for _, c in self_med_pairs] == [c for _, c in judge_med_pairs]:
        print("  -> rankings AGREE. D1 conclusion robust.")
    else:
        print("  -> rankings DISAGREE. Need to interpret D1 carefully.")


if __name__ == "__main__":
    main()
