"""
Experiment D10: hallucination detector with a smaller backbone.

D9 showed Qwen-3B is so good at following source documents that it doesn't
hallucinate even when source contradicts world knowledge -- so the
centroid-distance detector had nothing to detect.

D10 re-runs the same 25 contradictory triples with Qwen2.5-0.5B-Instruct
as the back-end. Smaller models follow instructions less reliably; we
expect some hallucinations even with the source given. If centroid
distance to source is larger for the hallucinated answers than for the
faithful ones, the detector is validated.

Same triples, same conditions (A/B/C), same substring-match judge.
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

# Reuse the triple set, structural-fingerprint code, and judge from D9.
sys.path.insert(0, str(EXP_ROOT))
from d9_hallucination_test import (
    TRIPLES, make_aisha, aisha_structure, structural_prefix,
    chat_prompt, generate_llama, judge_match,
)

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
RES_DIR = EXP_ROOT.parent / "results"

BACKBONE_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[d10] device: {device}")
    aisha = make_aisha()

    print(f"[d10] loading {BACKBONE_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        BACKBONE_NAME, torch_dtype=torch.float16, device_map="auto")
    backbone.eval()
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"[d10] loaded, {n_params/1e6:.1f}M params")

    rows = []
    t0 = time.time()
    for ti, t in enumerate(TRIPLES):
        struct = aisha_structure(aisha, t["source"])
        prefix = structural_prefix(struct)

        prompts = {
            "A_no_source":   chat_prompt(tok,
                f"Answer the question briefly with one word or short phrase.\n\nQ: {t['question']}"),
            "B_source":      chat_prompt(tok,
                f"Read the document and answer the question. Use only information from the document.\n\nDocument:\n{t['source']}\n\nQ: {t['question']}"),
            "C_source_struct": chat_prompt(tok,
                f"Read the document and answer the question. Use only information from the document.\n\n{prefix}\n\nDocument:\n{t['source']}\n\nQ: {t['question']}"),
        }

        outs = {}
        for cond, p in prompts.items():
            outs[cond] = generate_llama(backbone, tok, p, max_new=40,
                                         temperature=0.2, device=device)

        ans_fps = {c: aisha_structure(aisha, txt) for c, txt in outs.items()}
        dists = {}
        for c, fp in ans_fps.items():
            if fp is None or struct is None:
                dists[c] = float("nan")
            else:
                dists[c] = float(np.linalg.norm(fp["doc_centroid"] - struct["doc_centroid"]))

        match_source = {c: judge_match(None, None, txt, t["source_fact"], None)
                          for c, txt in outs.items()}
        match_world = {c: judge_match(None, None, txt, t["world_fact"], None)
                         for c, txt in outs.items()}

        rows.append({
            "question": t["question"], "source": t["source"],
            "source_fact": t["source_fact"], "world_fact": t["world_fact"],
            "outputs": outs, "centroid_dist": dists,
            "match_source": match_source, "match_world": match_world,
        })

        if ti < 4 or ti == len(TRIPLES) - 1:
            print(f"\n[{ti+1}/{len(TRIPLES)}] Q: {t['question'][:60]}")
            print(f"  source_fact: {t['source_fact']}, world_fact: {t['world_fact']}")
            for c in ["A_no_source", "B_source", "C_source_struct"]:
                ms, mw = match_source[c], match_world[c]
                tag = "FAITHFUL" if (ms and not mw) else (
                       "HALLUC" if (mw and not ms) else (
                       "BOTH" if ms and mw else "OTHER"))
                print(f"  {c:>16s} src={ms} wrld={mw}  dist={dists[c]:.2f}  [{tag}]  | {outs[c][:80]!r}")

    summary = {}
    for cond in ["A_no_source", "B_source", "C_source_struct"]:
        ms = np.array([r["match_source"][cond]   for r in rows])
        mw = np.array([r["match_world"][cond]    for r in rows])
        ds = np.array([r["centroid_dist"][cond]  for r in rows])
        faithful     = (ms == 1) & (mw == 0)
        hallucinated = (ms == 0) & (mw == 1)
        both     = (ms == 1) & (mw == 1)
        neither  = (ms == 0) & (mw == 0)
        summary[cond] = {
            "n":                   int(len(ms)),
            "match_source_rate":   float(ms.mean()),
            "match_world_rate":    float(mw.mean()),
            "faithful_only_rate":  float(faithful.mean()),
            "hallucinated_only_rate": float(hallucinated.mean()),
            "both_rate":           float(both.mean()),
            "neither_rate":        float(neither.mean()),
            "dist_median_faithful":      float(np.median(ds[faithful & ~np.isnan(ds)]))      if (faithful & ~np.isnan(ds)).any() else float("nan"),
            "dist_median_hallucinated":  float(np.median(ds[hallucinated & ~np.isnan(ds)]))  if (hallucinated & ~np.isnan(ds)).any() else float("nan"),
            "dist_median_overall":       float(np.median(ds[~np.isnan(ds)])) if (~np.isnan(ds)).any() else float("nan"),
        }

    print(f"\n=== D10 summary ({len(rows)} triples, {BACKBONE_NAME}) ===")
    print(f"{'condition':>20s}  {'src_rate':>9s}  {'wrld_rate':>10s}  "
          f"{'faithful':>9s}  {'hallu':>7s}  {'both':>6s}  {'dist_F':>7s}  {'dist_H':>7s}")
    for cond in ["A_no_source", "B_source", "C_source_struct"]:
        s = summary[cond]
        df = s['dist_median_faithful']; dh = s['dist_median_hallucinated']
        df_str = f"{df:.3f}" if not np.isnan(df) else "—"
        dh_str = f"{dh:.3f}" if not np.isnan(dh) else "—"
        print(f"  {cond:>18s}  {s['match_source_rate']:>9.2f}  {s['match_world_rate']:>10.2f}  "
              f"{s['faithful_only_rate']:>9.2f}  {s['hallucinated_only_rate']:>7.2f}  "
              f"{s['both_rate']:>6.2f}  {df_str:>7s}  {dh_str:>7s}")

    print("\n=== Aggregate hallucination signal (B + C combined) ===")
    all_dist_faithful = []
    all_dist_hallucinated = []
    for cond in ["B_source", "C_source_struct"]:
        for r in rows:
            ms, mw = r["match_source"][cond], r["match_world"][cond]
            d = r["centroid_dist"][cond]
            if np.isnan(d): continue
            if ms and not mw: all_dist_faithful.append(d)
            elif mw and not ms: all_dist_hallucinated.append(d)
    print(f"  faithful dist:      n={len(all_dist_faithful):>3d}  median={np.median(all_dist_faithful):.3f}" if all_dist_faithful else "  faithful: no samples")
    print(f"  hallucinated dist:  n={len(all_dist_hallucinated):>3d}  median={np.median(all_dist_hallucinated):.3f}" if all_dist_hallucinated else "  hallucinated: no samples")
    if len(all_dist_faithful) >= 5 and len(all_dist_hallucinated) >= 5:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(all_dist_hallucinated, all_dist_faithful, alternative="greater")
        n1, n2 = len(all_dist_hallucinated), len(all_dist_faithful)
        rb = 1 - (2 * u) / (n1 * n2)
        print(f"  Mann-Whitney U (hallucinated > faithful): U={u:.0f}, p={p:.3g}")
        print(f"  rank-biserial effect: {rb:+.3f}  (positive = hallucinated farther from source)")

        # Rough threshold for the verifier-loop. If we set threshold at the
        # 75th percentile of faithful distances, what's the recall and precision
        # for hallucination detection?
        thresh = float(np.percentile(all_dist_faithful, 75))
        tp = sum(1 for d in all_dist_hallucinated if d > thresh)
        fp = sum(1 for d in all_dist_faithful   if d > thresh)
        recall    = tp / max(n1, 1)
        precision = tp / max(tp + fp, 1)
        print(f"\n  Detector at threshold = p75-of-faithful = {thresh:.3f}:")
        print(f"    flagged faithful (false positives): {fp}/{n2}  ({fp/max(n2,1)*100:.0f}%)")
        print(f"    flagged hallucinated (true positives): {tp}/{n1}  ({tp/max(n1,1)*100:.0f}%)")
        print(f"    recall {recall:.2f}, precision {precision:.2f}")

    out_path = RES_DIR / "d10_smaller_halluc_results.json"
    out_path.write_text(json.dumps({
        "n_triples": len(rows), "backbone": BACKBONE_NAME,
        "summary": summary, "per_triple": rows,
    }, indent=2, default=str))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
