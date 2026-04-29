"""
Experiment D14b: lambda sweep on the D14 combined-memory pipeline.

D14 at lambda=1.5 won 7 of 8 conversations on stylistic match but failed
on one (breakfast_health) where the boundary pulled Qwen into 'give advice'
mode and away from reflecting user details. Hypothesis: lambda is the
single-parameter fix. We sweep lambda in {0.0, 0.5, 1.0, 1.5, 2.5} on the
exact same 8 conversations.

Goal: find lambda* that keeps the stylistic gain (centroid-distance and
POS-L1 reductions) while preserving the factual recall the verbatim
prior turns already carry.
"""
import json
import os
import re
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
CACHE_DIR = EXP_ROOT.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)

_default_aisha = Path(__file__).resolve().parent.parent.parent / "aisha"
AISHA_ROOT = Path(os.environ.get("AISHA_ROOT", str(_default_aisha))).resolve()
sys.path.insert(0, str(AISHA_ROOT))

sys.path.insert(0, str(EXP_ROOT))
from d9_hallucination_test import make_aisha, aisha_structure
from d13b_structural_memory import (
    boundary_with_structural_memory, build_boost_mask, generate_qwen,
    pos_l1, BACKBONE_NAME,
)
from d14_combined_memory import CONVERSATIONS, memory_score

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
RES_DIR = EXP_ROOT.parent / "results"
WORD_RE = re.compile(r"[a-zA-Z']+")

LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.5]
MEMORY_LENGTH = 5


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[d14b] device: {device}")
    aisha = make_aisha()

    print(f"[d14b] loading {BACKBONE_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE_NAME, torch_dtype=torch.float16, device_map="auto").eval()
    print("[d14b] loaded.")

    rows = []
    for ci, conv in enumerate(CONVERSATIONS):
        prior_turns = conv["turns"]
        last_q      = conv["final_question"]
        fact_kw     = conv["fact_keywords"]
        full_struct = aisha_structure(aisha, " ".join(prior_turns))
        full_centroid = full_struct["doc_centroid"]
        full_pos      = full_struct["pos_profile"]

        boundary_words = boundary_with_structural_memory(
            aisha, last_q, prior_turns, MEMORY_LENGTH)
        mask = (build_boost_mask(boundary_words, tok, model.config.vocab_size, device)
                if boundary_words else None)

        msgs = [{"role": "user", "content": t} for t in prior_turns]
        msgs.append({"role": "user", "content": last_q})
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        per_lam = {}
        for lam in LAMBDAS:
            torch.manual_seed(0)
            out = generate_qwen(model, tok, prompt, mask if lam > 0 else None,
                                 lam, max_new=80, T=0.3, device=device)
            ans_struct = aisha_structure(aisha, out)
            cd = float(np.linalg.norm(ans_struct["doc_centroid"] - full_centroid)) if ans_struct else float("nan")
            pl = pos_l1(ans_struct["pos_profile"], full_pos) if ans_struct else float("nan")
            n_hit, hits = memory_score(out, fact_kw)
            per_lam[lam] = {"text": out, "n_hits": n_hit, "found": hits,
                              "centroid_dist": cd, "pos_l1": pl}

        rows.append({
            "style":          conv["style"],
            "fact_keywords":  fact_kw,
            "boundary_size":  len(boundary_words),
            "per_lambda":     per_lam,
        })

        print(f"\n[{ci+1}/{len(CONVERSATIONS)}] style={conv['style']}  bound={len(boundary_words)}")
        for lam in LAMBDAS:
            r = per_lam[lam]
            print(f"  lam={lam:>4.1f}  hits={r['n_hits']:>2d}  cdist={r['centroid_dist']:.3f}  "
                  f"pos_l1={r['pos_l1']:.3f}  | {r['text'][:80]!r}")

    print("\n=== D14b lambda sweep summary ===")
    print(f"{'lambda':>7s}  {'avg_hits':>9s}  {'cdist_mean':>11s}  "
          f"{'pos_l1_mean':>12s}  {'breakfast_hits':>14s}")
    summary = {}
    for lam in LAMBDAS:
        hits = np.array([r["per_lambda"][lam]["n_hits"] for r in rows])
        cd   = np.array([r["per_lambda"][lam]["centroid_dist"] for r in rows])
        pl   = np.array([r["per_lambda"][lam]["pos_l1"] for r in rows])
        cd = cd[~np.isnan(cd)]; pl = pl[~np.isnan(pl)]
        # Specifically: did the breakfast failure get fixed?
        breakfast_row = next((r for r in rows if r["style"] == "breakfast_health"), None)
        breakfast_hits = breakfast_row["per_lambda"][lam]["n_hits"] if breakfast_row else None
        summary[lam] = {
            "avg_hits":            float(hits.mean()),
            "centroid_dist_mean":  float(cd.mean()) if cd.size else float("nan"),
            "pos_l1_mean":         float(pl.mean()) if pl.size else float("nan"),
            "breakfast_hits":      int(breakfast_hits) if breakfast_hits is not None else None,
        }
        print(f"  {lam:>7.1f}  {summary[lam]['avg_hits']:>9.2f}  "
              f"{summary[lam]['centroid_dist_mean']:>11.3f}  "
              f"{summary[lam]['pos_l1_mean']:>12.3f}  "
              f"{breakfast_hits if breakfast_hits is not None else '-':>14}")

    print("\n=== Per-style table at each lambda ===")
    print(f"{'style':>20s} | " + "  ".join(f"l{lam:.1f}".ljust(11) for lam in LAMBDAS))
    for r in rows:
        line = f"{r['style']:>20s} | " + "  ".join(
            f"h{r['per_lambda'][lam]['n_hits']}c{r['per_lambda'][lam]['centroid_dist']:.2f}".ljust(11)
            for lam in LAMBDAS)
        print(line)

    out_path = RES_DIR / "d14b_lambda_sweep_results.json"
    out_path.write_text(json.dumps({
        "lambdas": LAMBDAS, "memory_length": MEMORY_LENGTH,
        "summary": summary, "per_conversation": rows,
    }, indent=2, default=str))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
