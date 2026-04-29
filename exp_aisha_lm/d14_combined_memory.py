"""
Experiment D14: combined memory architecture.

Your design: Qwen consumes WORDS (verbatim prior turns), Aisha consumes
STRUCTURE (running fingerprint of prior turns). The two share the response
through their respective channels: Qwen via context tokens, Aisha via
logit bias on Qwen's logits.

Test setup: same 8 stylistic conversations from D13. Two metrics:

  factual recall  : keyword presence in the response (via memory_keywords
                     analogous to D12; here we add coverage tags per
                     conversation)
  stylistic match : Aisha's centroid distance and POS-L1 between response
                     and the conversation's full structural fingerprint.

Conditions:
  A. NO_MEM         : Qwen sees only the last question. No memory at all.
  B. WORDS_ONLY     : Qwen sees prior turns + last question. No Aisha bias.
  C. STRUCTURE_ONLY : Qwen sees only last question + Aisha's
                       structural-memory-derived boundary bias.
  D. COMBINED       : Qwen sees prior turns + last question, AND Aisha's
                       structural-memory boundary bias is applied.

If D ≈ B: structure adds nothing on top of words.
If D > B: structure adds value. The architectural split works.
If C ≈ A: structure alone (no words) doesn't carry memory.
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
from d9_hallucination_test import make_aisha, aisha_structure
from d13b_structural_memory import (
    boundary_with_structural_memory, build_boost_mask, generate_qwen, chat_prompt,
    pos_l1, BACKBONE_NAME, LAMBDA,
)

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
RES_DIR = EXP_ROOT.parent / "results"
WORD_RE = re.compile(r"[a-zA-Z']+")
MEMORY_LENGTH = 5    # use full prior conversation as memory

# 8 conversations with style + factual content + a final question that
# requires combining both.
CONVERSATIONS = [
    {
        "style": "tokyo_trip_formal",
        "turns": [
            "I am planning a business trip to Tokyo next month for our quarterly review.",
            "I will be attending meetings with clients during the cherry blossom season in April.",
            "This will be my first formal visit to Japan as part of our executive delegation.",
        ],
        "final_question": "What clothes should I bring?",
        "fact_keywords":   ["tokyo", "april", "japan", "spring", "cherry", "blossom", "business"],
    },
    {
        "style": "dog_food_casual",
        "turns": [
            "yo my dog is a black lab named buddy",
            "he's six and lately he's been having allergic reactions to chicken",
            "we're trying to figure out a new diet for him",
        ],
        "final_question": "What dog food should I get?",
        "fact_keywords":   ["chicken", "allergy", "allergic", "lab", "labrador", "buddy", "no chicken"],
    },
    {
        "style": "ml_engineer",
        "turns": [
            "I work as a machine learning engineer at a small AI startup.",
            "I primarily use Python and PyTorch for deep learning research.",
            "I have three years of experience in NLP and computer vision.",
        ],
        "final_question": "What technical book should I read next?",
        "fact_keywords":   ["python", "pytorch", "machine learning", "ml", "nlp", "ai", "deep"],
    },
    {
        "style": "breakfast_health",
        "turns": [
            "I had two scrambled eggs and three slices of bacon this morning.",
            "I also drank a large glass of orange juice and ate buttered toast.",
            "I usually have similar breakfasts every weekday before work.",
        ],
        "final_question": "Was my breakfast healthy?",
        "fact_keywords":   ["egg", "bacon", "toast", "butter", "orange", "juice"],
    },
    {
        "style": "sf_morning",
        "turns": [
            "I live in a studio apartment in San Francisco.",
            "My commute to work is thirty minutes by Muni bus.",
            "I usually leave home at seven in the morning to be at my desk by eight.",
        ],
        "final_question": "How can I improve my morning routine?",
        "fact_keywords":   ["bus", "muni", "san francisco", "studio", "thirty", "30", "seven"],
    },
    {
        "style": "grandma_gift",
        "turns": [
            "My grandmother is turning eighty next Saturday and I want to surprise her.",
            "She loves gardening, especially roses, and listens to classical music every day.",
            "She also enjoys baking pastries and has a small kitchen full of antique tools.",
        ],
        "final_question": "What gift should I get her?",
        "fact_keywords":   ["garden", "rose", "music", "classical", "baking", "pastry", "kitchen", "eighty"],
    },
    {
        "style": "interview_prep",
        "turns": [
            "I am preparing for a job interview at a healthcare AI startup next week.",
            "The role is for a senior data scientist working on clinical decision support.",
            "I have a background in statistics and biology but limited industry experience.",
        ],
        "final_question": "What should I read to prepare?",
        "fact_keywords":   ["healthcare", "ai", "data", "science", "clinical", "statistics", "biology"],
    },
    {
        "style": "marathon_training",
        "turns": [
            "I am training for my first marathon and the race is in eight weeks.",
            "I currently run twenty-five kilometers per week across four sessions.",
            "I have never run more than fifteen kilometers in a single session before.",
        ],
        "final_question": "How should I train from now until the race?",
        "fact_keywords":   ["marathon", "kilometers", "km", "weeks", "race", "running", "eight"],
    },
]


def memory_score(text, keywords):
    t = text.lower()
    found = [k for k in keywords if k.lower() in t]
    return len(found), found


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[d14] device: {device}")
    aisha = make_aisha()

    print(f"[d14] loading {BACKBONE_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE_NAME, torch_dtype=torch.float16, device_map="auto").eval()
    print("[d14] loaded.")

    rows = []
    for ci, conv in enumerate(CONVERSATIONS):
        prior_turns   = conv["turns"]
        last_q        = conv["final_question"]
        fact_keywords = conv["fact_keywords"]
        full_struct   = aisha_structure(aisha, " ".join(prior_turns))
        full_centroid = full_struct["doc_centroid"]
        full_pos      = full_struct["pos_profile"]

        # Aisha-derived boundary using full prior conversation as structural memory
        boundary_words = boundary_with_structural_memory(
            aisha, last_q, prior_turns, MEMORY_LENGTH)
        mask = (build_boost_mask(boundary_words, tok, model.config.vocab_size, device)
                if boundary_words else None)

        # Build the four prompts
        # A. last question only
        prompt_A = chat_prompt(tok, last_q)
        # B and D: prior turns + last question (each prior turn as a separate user msg)
        msgs_words = [{"role": "user", "content": t} for t in prior_turns]
        msgs_words.append({"role": "user", "content": last_q})
        prompt_words = tok.apply_chat_template(
            msgs_words, tokenize=False, add_generation_prompt=True)

        # Generate four conditions
        outs = {}
        torch.manual_seed(0); outs["A_no_mem"]    = generate_qwen(model, tok, prompt_A,     None, 0.0,   80, 0.3, device)
        torch.manual_seed(0); outs["B_words"]     = generate_qwen(model, tok, prompt_words, None, 0.0,   80, 0.3, device)
        torch.manual_seed(0); outs["C_struct"]    = generate_qwen(model, tok, prompt_A,     mask, LAMBDA,80, 0.3, device)
        torch.manual_seed(0); outs["D_combined"]  = generate_qwen(model, tok, prompt_words, mask, LAMBDA,80, 0.3, device)

        # Metrics per condition
        metrics = {}
        for c, t in outs.items():
            n_hit, hits = memory_score(t, fact_keywords)
            ans_struct = aisha_structure(aisha, t)
            cdist = float(np.linalg.norm(ans_struct["doc_centroid"] - full_centroid)) if ans_struct else float("nan")
            poslv = pos_l1(ans_struct["pos_profile"], full_pos) if ans_struct else float("nan")
            metrics[c] = {
                "n_keyword_hits": n_hit, "found_keywords": hits,
                "centroid_dist_to_conv": cdist,
                "pos_l1_to_conv":         poslv,
                "n_tokens": len(WORD_RE.findall(t)),
            }

        rows.append({
            "style":        conv["style"],
            "fact_keywords": fact_keywords,
            "boundary_size": len(boundary_words),
            "outputs":       outs,
            "metrics":       metrics,
        })

        print(f"\n[{ci+1}/{len(CONVERSATIONS)}] style={conv['style']}")
        for c in ["A_no_mem", "B_words", "C_struct", "D_combined"]:
            m = metrics[c]
            print(f"  {c:>14s}  hits={m['n_keyword_hits']:>2d}  "
                  f"cdist={m['centroid_dist_to_conv']:.3f}  "
                  f"pos_l1={m['pos_l1_to_conv']:.3f}")
            print(f"      {outs[c][:100]!r}")

    # Aggregate
    print("\n=== D14 summary across 8 conversations ===")
    print(f"{'condition':>14s}  {'avg_hits':>9s}  {'cdist_mean':>11s}  "
          f"{'pos_l1_mean':>12s}")
    summary = {}
    for cond in ["A_no_mem", "B_words", "C_struct", "D_combined"]:
        hits = np.array([r["metrics"][cond]["n_keyword_hits"] for r in rows])
        cd   = np.array([r["metrics"][cond]["centroid_dist_to_conv"] for r in rows])
        pl   = np.array([r["metrics"][cond]["pos_l1_to_conv"] for r in rows])
        cd = cd[~np.isnan(cd)]; pl = pl[~np.isnan(pl)]
        summary[cond] = {
            "avg_keyword_hits":      float(hits.mean()),
            "any_hit_rate":          float((hits > 0).mean()),
            "centroid_dist_mean":    float(cd.mean()) if cd.size else float("nan"),
            "pos_l1_mean":           float(pl.mean()) if pl.size else float("nan"),
        }
        print(f"  {cond:>12s}  {summary[cond]['avg_keyword_hits']:>9.2f}  "
              f"{summary[cond]['centroid_dist_mean']:>11.3f}  "
              f"{summary[cond]['pos_l1_mean']:>12.3f}")

    # The clean question: D vs B - does Aisha structure add anything on top of words?
    print("\n=== D vs B (does structure add anything on top of words?) ===")
    print(f"{'style':>22s}  {'B hits':>7s}  {'D hits':>7s}  {'B cdist':>8s}  {'D cdist':>8s}")
    for r in rows:
        Bh = r["metrics"]["B_words"]["n_keyword_hits"]
        Dh = r["metrics"]["D_combined"]["n_keyword_hits"]
        Bc = r["metrics"]["B_words"]["centroid_dist_to_conv"]
        Dc = r["metrics"]["D_combined"]["centroid_dist_to_conv"]
        print(f"  {r['style']:>22s}  {Bh:>7d}  {Dh:>7d}  {Bc:>8.3f}  {Dc:>8.3f}")

    out_path = RES_DIR / "d14_combined_memory_results.json"
    out_path.write_text(json.dumps({
        "n_conversations": len(rows),
        "memory_length": MEMORY_LENGTH, "lambda": LAMBDA,
        "summary": summary, "per_conversation": rows,
    }, indent=2, default=str))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
