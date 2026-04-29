"""
Experiment D15: implement and validate intent-based routing.

The breakfast failure in D14: 'Was my breakfast healthy?' is a reflective
past-self-state question. With Aisha boundary at lambda=1.5, Qwen drifts
into 'here are suggestions to improve' mode, losing the user's specific
food details. The fix: detect reflective questions and disable Aisha
(set lambda=0) for those, keeping lambda=1.5 for advisory/exploratory.

This script implements a small regex-based classifier and validates that:
  1. It correctly fires on 'Was my breakfast healthy?' (and other
     reflective probes added below).
  2. Routing on top of D14 recovers breakfast faithfulness while
     preserving the stylistic gains on the other seven conversations.
  3. It does not trip on tricky cases ('What did you do today?', past-
     tense factual questions like 'When did the war start?', etc.).

If the fix works on D14's set, the production phone app can use this as
a one-file pre-shipping addition. No backbone or data changes needed.
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
LAMBDA_DEFAULT = 1.5
LAMBDA_REFLECTIVE = 0.0
MEMORY_LENGTH = 5


# =============================================================================
# Intent classifier.
# =============================================================================
#
# A reflective question asks the assistant to evaluate or recall something
# about the user's past actions or state. Patterns:
#   "Was my breakfast healthy?"        -> 'was my'
#   "Did I sound rude?"                -> 'did i'
#   "Were those calculations right?"   -> 'were' + first-person referent
#   "Have I been getting enough sleep?"-> 'have i'
#   "Did the meeting go well?"         -> reflective if subject is from history
#
# False positives we must avoid:
#   "What did you do today?"           -> 'did you', not 'did i'
#   "Should I take the bus?"           -> not past-tense reflective
#   "When did the war start?"          -> factual, not reflective
#
# We use simple regex on lowercased text. The patterns require the past-tense
# auxiliary to be paired with first-person 'i' or 'my' (or self-referential
# evaluation patterns like 'was X healthy/right/wrong').

REFLECTIVE_RE = re.compile(
    r"\b("
    # 1. past-aux + I/my  (covers "was my", "were my", "did I", "had I", "have I", "is my")
    r"(was|were|did|had|have|has)\s+(my|i|i\s+been|i've|ive)"
    # 2. evaluative patterns: "was X (healthy|good|right|wrong|correct|safe|wise|rude|mean)?"
    r"|(was|were|are|am|is)\s+(my|that|this|those|these|the)\s+\w+(\s+\w+)?\s+("
    r"healthy|unhealthy|good|bad|right|wrong|correct|incorrect|"
    r"safe|wise|unwise|rude|mean|polite|appropriate|fair|reasonable|"
    r"enough|too\s+much|too\s+little"
    r")"
    # 3. "did I make the right ...?", "should I have ..."
    r"|did\s+i\s+(make|do|say|choose|pick|handle|answer|respond)"
    r"|should\s+i\s+have"
    r")\b",
    re.IGNORECASE,
)


def is_reflective_question(text: str) -> bool:
    """Return True if the text reads as a reflective past-self-state question.

    Designed to fire on 'Was my breakfast healthy?', 'Did I make the right
    call?', 'Have I been getting enough sleep?'; not on 'What did you do?',
    'When did the war start?', 'What should I read?'."""
    return bool(REFLECTIVE_RE.search(text or ""))


def select_lambda(question: str) -> float:
    return LAMBDA_REFLECTIVE if is_reflective_question(question) else LAMBDA_DEFAULT


# =============================================================================
# Classifier sanity check.
# =============================================================================

CLASSIFIER_PROBES = [
    # Reflective (should be True)
    ("Was my breakfast healthy?",                 True),
    ("Did I make the right call?",                True),
    ("Were my calculations correct?",             True),
    ("Have I been getting enough sleep?",         True),
    ("Was my answer right?",                      True),
    ("Did I sound rude?",                         True),
    ("Should I have done it differently?",        True),
    # Advisory / exploratory (should be False)
    ("What clothes should I bring?",              False),
    ("How can I improve my morning routine?",     False),
    ("What gift should I get her?",               False),
    ("What technical book would you recommend?",  False),
    ("How should I train from now until the race?", False),
    # Edge cases that test the classifier (should be False)
    ("What did you do today?",                    False),
    ("When did the war start?",                   False),
    ("Did the meeting go well?",                  False),  # could be reflective but no I/my
    ("Why is the sky blue?",                      False),
    ("Tell me about Paris.",                      False),
]


def run_classifier_sanity_check():
    print("\n=== Classifier sanity check ===")
    correct = 0
    for q, expected in CLASSIFIER_PROBES:
        got = is_reflective_question(q)
        ok = got == expected
        if ok:
            correct += 1
        flag = "OK " if ok else "FAIL"
        verdict = "REFLECTIVE" if got else "advisory  "
        print(f"  [{flag}] {verdict} | {q!r}")
    print(f"  {correct}/{len(CLASSIFIER_PROBES)} correct")
    return correct, len(CLASSIFIER_PROBES)


# =============================================================================
# Validation on D14 conversations.
# =============================================================================

def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    cls_correct, cls_total = run_classifier_sanity_check()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[d15] device: {device}")
    aisha = make_aisha()

    print(f"[d15] loading {BACKBONE_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE_NAME, torch_dtype=torch.float16, device_map="auto").eval()
    print("[d15] loaded.")

    rows = []
    for ci, conv in enumerate(CONVERSATIONS):
        prior_turns = conv["turns"]
        last_q      = conv["final_question"]
        fact_kw     = conv["fact_keywords"]
        full_struct = aisha_structure(aisha, " ".join(prior_turns))
        full_centroid = full_struct["doc_centroid"]
        full_pos      = full_struct["pos_profile"]

        is_refl = is_reflective_question(last_q)
        lam_routed = LAMBDA_REFLECTIVE if is_refl else LAMBDA_DEFAULT

        boundary_words = boundary_with_structural_memory(
            aisha, last_q, prior_turns, MEMORY_LENGTH)
        mask = (build_boost_mask(boundary_words, tok, model.config.vocab_size, device)
                if boundary_words else None)

        msgs = [{"role": "user", "content": t} for t in prior_turns]
        msgs.append({"role": "user", "content": last_q})
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        # Three runs per conversation:
        # B_words      : lam=0 (no Aisha)                    -- words-only baseline
        # D_combined   : lam=1.5 always (D14 production)     -- has the breakfast bug
        # D_routed     : intent-routed lam                    -- the fix
        outs = {}
        for label, lam in [("B_words", 0.0),
                            ("D_combined_static_1.5", LAMBDA_DEFAULT),
                            ("D_routed", lam_routed)]:
            torch.manual_seed(0)
            outs[label] = generate_qwen(model, tok, prompt,
                                         mask if lam > 0 else None, lam,
                                         max_new=80, T=0.3, device=device)

        metrics = {}
        for c, t in outs.items():
            n_hit, hits = memory_score(t, fact_kw)
            ans_struct = aisha_structure(aisha, t)
            cd = float(np.linalg.norm(ans_struct["doc_centroid"] - full_centroid)) if ans_struct else float("nan")
            pl = pos_l1(ans_struct["pos_profile"], full_pos) if ans_struct else float("nan")
            metrics[c] = {"n_hits": n_hit, "found": hits,
                            "centroid_dist": cd, "pos_l1": pl}

        rows.append({
            "style":            conv["style"],
            "final_question":   last_q,
            "is_reflective":    is_refl,
            "lambda_routed":    lam_routed,
            "boundary_size":    len(boundary_words),
            "outputs":          outs,
            "metrics":          metrics,
        })

        tag = "REFLECTIVE -> lam=0" if is_refl else f"advisory   -> lam={LAMBDA_DEFAULT}"
        print(f"\n[{ci+1}/{len(CONVERSATIONS)}] {conv['style']:>22s} | {tag}")
        print(f"  Q: {last_q}")
        for c in ["B_words", "D_combined_static_1.5", "D_routed"]:
            m = metrics[c]
            print(f"    {c:>22s}  hits={m['n_hits']:>2d}  cdist={m['centroid_dist']:.3f}  "
                  f"pos_l1={m['pos_l1']:.3f}")
            print(f"      | {outs[c][:80]!r}")

    print("\n=== D15 summary ===")
    print(f"{'condition':>22s}  {'avg_hits':>9s}  {'cdist_mean':>11s}  "
          f"{'pos_l1_mean':>12s}  {'breakfast_hits':>14s}")
    summary = {}
    breakfast_row = next((r for r in rows if r["style"] == "breakfast_health"), None)
    for cond in ["B_words", "D_combined_static_1.5", "D_routed"]:
        hits = np.array([r["metrics"][cond]["n_hits"] for r in rows])
        cd   = np.array([r["metrics"][cond]["centroid_dist"] for r in rows])
        pl   = np.array([r["metrics"][cond]["pos_l1"] for r in rows])
        cd = cd[~np.isnan(cd)]; pl = pl[~np.isnan(pl)]
        bh = breakfast_row["metrics"][cond]["n_hits"] if breakfast_row else None
        summary[cond] = {
            "avg_hits":            float(hits.mean()),
            "centroid_dist_mean":  float(cd.mean()) if cd.size else float("nan"),
            "pos_l1_mean":         float(pl.mean()) if pl.size else float("nan"),
            "breakfast_hits":      int(bh) if bh is not None else None,
        }
        print(f"  {cond:>22s}  {summary[cond]['avg_hits']:>9.2f}  "
              f"{summary[cond]['centroid_dist_mean']:>11.3f}  "
              f"{summary[cond]['pos_l1_mean']:>12.3f}  "
              f"{bh if bh is not None else '-':>14}")

    print("\n=== Verdict ===")
    print(f"  Classifier accuracy on probe set: {cls_correct}/{cls_total}")
    print(f"  D14 static lambda=1.5 breakfast hits: {summary['D_combined_static_1.5']['breakfast_hits']}")
    print(f"  D15 routed              breakfast hits: {summary['D_routed']['breakfast_hits']}")
    if summary["D_routed"]["breakfast_hits"] is not None and summary["D_combined_static_1.5"]["breakfast_hits"] is not None:
        if summary["D_routed"]["breakfast_hits"] > summary["D_combined_static_1.5"]["breakfast_hits"]:
            print("  -> Routing FIXED the breakfast failure.")
        else:
            print("  -> Routing did NOT fix breakfast.")

    out_path = RES_DIR / "d15_intent_routing_results.json"
    out_path.write_text(json.dumps({
        "classifier_correct":    cls_correct,
        "classifier_total":      cls_total,
        "lambda_default":        LAMBDA_DEFAULT,
        "lambda_reflective":     LAMBDA_REFLECTIVE,
        "memory_length":         MEMORY_LENGTH,
        "summary":               summary,
        "per_conversation":      rows,
        "classifier_probes":     [{"q": q, "expected": e,
                                       "predicted": is_reflective_question(q)}
                                      for q, e in CLASSIFIER_PROBES],
    }, indent=2, default=str))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
