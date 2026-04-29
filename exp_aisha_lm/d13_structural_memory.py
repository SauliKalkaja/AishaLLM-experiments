"""
Experiment D13: structural memory at varying lengths.

Hypothesis: when a user speaks in a consistent style across multiple turns,
feeding Aisha's running STRUCTURAL fingerprint (POS profile + 16D centroid +
trajectory) -- NOT the words themselves -- produces responses that are more
stylistically consistent with the established conversation. This is "personal
feel" via structural geometry, not fact recall.

8 conversations, each with a clearly distinct stylistic flavor (formal,
casual, scientific, poetic, news, code-like, terse, verbose). Each
conversation has 5 establishing user turns and a 6th generic question.
We vary how many of the prior turns Aisha can see:

  N=0  baseline, no memory
  N=1  only the immediately preceding turn
  N=3  3 prior turns aggregated structurally
  N=5  full prior conversation structurally

For each, Qwen-0.5B-Instruct generates a response using ONLY the last
question + Aisha's structural preamble derived from the N prior turns.

Metric: how close is the response's manifold position to the conversation's
established centroid? We measure
  - centroid distance (smaller = more aligned with conversation tone)
  - POS profile L1 distance (smaller = more matched register)

If memory length helps consistency monotonically, structural memory
is doing real work for "personal" feel.
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
from d9_hallucination_test import make_aisha, aisha_structure, structural_prefix

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
RES_DIR = EXP_ROOT.parent / "results"
WORD_RE = re.compile(r"[a-zA-Z']+")

BACKBONE_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MEMORY_LENGTHS = [0, 1, 3, 5]

# 8 conversations with a stable stylistic flavor each. All ask a similar
# generic final question so the test isolates "did the prior style influence
# the response?"
CONVERSATIONS = [
    {
        "style": "formal_business",
        "turns": [
            "I would like to schedule a meeting with the executive committee.",
            "Please confirm availability for the agenda items I distributed earlier.",
            "Kindly review the quarterly financial summary before the briefing.",
            "We will require the presentation slides at least 24 hours in advance.",
            "Ensure all attendees receive the formal invitation by Tuesday morning.",
        ],
        "final_question": "What should I do next?",
    },
    {
        "style": "casual_chat",
        "turns": [
            "yo just got home man what a day",
            "lol my boss was being super weird again about the spreadsheet",
            "anyway gonna grab some pizza and chill on the couch",
            "you should totally come over later if you're free",
            "bring some beers if you can it'll be fun",
        ],
        "final_question": "What should I do next?",
    },
    {
        "style": "scientific",
        "turns": [
            "The experimental data indicate a statistically significant deviation from baseline.",
            "Sample variance was substantially reduced in the controlled treatment condition.",
            "These observations support the proposed mechanistic hypothesis.",
            "Subsequent replication will validate the underlying causal relationship.",
            "Potential confounding factors must be systematically excluded.",
        ],
        "final_question": "What should I do next?",
    },
    {
        "style": "poetic_literary",
        "turns": [
            "Beneath the silver moon the river whispers secrets old.",
            "Long shadows curl through valleys where the wild thyme grows.",
            "A nightingale sings softly to the listening trees.",
            "Pale blossoms drift like memories upon the still pond.",
            "And distant bells call gently to the wandering soul.",
        ],
        "final_question": "What should I do next?",
    },
    {
        "style": "news_report",
        "turns": [
            "Authorities confirmed the incident occurred shortly after midnight Tuesday.",
            "Witnesses reported hearing loud noises near the downtown intersection.",
            "Emergency responders arrived on the scene within minutes.",
            "Officials have declined to release further details pending investigation.",
            "A spokesperson said an update will be issued later this evening.",
        ],
        "final_question": "What should I do next?",
    },
    {
        "style": "instructional_terse",
        "turns": [
            "Open the lid.",
            "Insert the cartridge.",
            "Close the cover firmly.",
            "Press the green button once.",
            "Wait for the indicator light.",
        ],
        "final_question": "What should I do next?",
    },
    {
        "style": "emotional_personal",
        "turns": [
            "I just don't know how to feel anymore about everything that's happened.",
            "Sometimes I lie awake at night and wonder if I made the right choices.",
            "There are days when even getting out of bed feels overwhelming.",
            "My friends try to help but they don't really understand the weight.",
            "I keep telling myself it'll get better but I'm not so sure.",
        ],
        "final_question": "What should I do next?",
    },
    {
        "style": "philosophical_dense",
        "turns": [
            "The question of consciousness presupposes a subject capable of reflexive awareness.",
            "Yet the very ontology of the subject remains contested across philosophical traditions.",
            "If the self is reducible to neuronal activity, in what sense does intentionality persist?",
            "Phenomenology suggests experience is fundamentally relational rather than substantial.",
            "These considerations bear directly on our concept of free will.",
        ],
        "final_question": "What should I do next?",
    },
]


def aisha_running_struct(responder, prior_turns: list[str]):
    if not prior_turns:
        return None
    return aisha_structure(responder, " ".join(prior_turns))


def render_structural_preamble(struct) -> str:
    """Honest structural preamble: POS profile + 16D centroid + step.
    No vocabulary words leaked through."""
    if struct is None:
        return ""
    pp = struct["pos_profile"]
    c = struct["doc_centroid"]
    centroid_str = "[" + ",".join(f"{x:.2f}" for x in c) + "]"
    return ("[Conversation structural memory: "
            f"{struct['n_sents']} prior sentences, "
            f"POS distribution N={pp.get('NOUN', 0):.0%} "
            f"V={pp.get('VERB', 0):.0%} "
            f"A={pp.get('ADJ', 0):.0%} "
            f"Adv={pp.get('ADV', 0):.0%}; "
            f"inter-turn step {struct['mean_step']:.2f}; "
            f"manifold centroid {centroid_str}. "
            f"Maintain this register and topical region.]")


@torch.no_grad()
def generate_qwen(model, tokenizer, prompt, max_new, T, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        enc.input_ids, attention_mask=enc.attention_mask,
        max_new_tokens=max_new, temperature=T, do_sample=T > 0.05,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, enc.input_ids.shape[1]:],
                             skip_special_tokens=True).strip()


def chat_prompt(tokenizer, msg):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": msg}],
        tokenize=False, add_generation_prompt=True)


def pos_l1_distance(pp1, pp2):
    """L1 distance between two POS profiles."""
    keys = set(pp1.keys()) | set(pp2.keys())
    return sum(abs(pp1.get(k, 0) - pp2.get(k, 0)) for k in keys)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[d13] device: {device}")
    aisha = make_aisha()

    print(f"[d13] loading {BACKBONE_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE_NAME, torch_dtype=torch.float16, device_map="auto").eval()
    print("[d13] loaded.")

    rows = []
    for ci, conv in enumerate(CONVERSATIONS):
        prior_turns = conv["turns"]      # 5 prior turns
        last_q = conv["final_question"]
        full_struct = aisha_structure(aisha, " ".join(prior_turns))
        if full_struct is None:
            continue
        full_centroid = full_struct["doc_centroid"]
        full_pos_profile = full_struct["pos_profile"]

        per_N = {}
        for N in MEMORY_LENGTHS:
            visible = prior_turns[-N:] if N > 0 else []
            mem_struct = aisha_running_struct(aisha, visible)
            preamble = render_structural_preamble(mem_struct) if mem_struct else ""
            user_msg = (preamble + "\n\n" + last_q) if preamble else last_q
            torch.manual_seed(0)
            out = generate_qwen(model, tok, chat_prompt(tok, user_msg),
                                 max_new=80, T=0.3, device=device)

            ans_struct = aisha_structure(aisha, out)
            if ans_struct is None:
                centroid_dist = float("nan")
                pos_dist = float("nan")
            else:
                centroid_dist = float(np.linalg.norm(
                    ans_struct["doc_centroid"] - full_centroid))
                pos_dist = pos_l1_distance(ans_struct["pos_profile"],
                                            full_pos_profile)
            per_N[N] = {
                "memory_struct_n_sents": (mem_struct["n_sents"]
                                              if mem_struct else 0),
                "answer_text": out,
                "answer_centroid_dist_to_conv": centroid_dist,
                "answer_pos_l1_to_conv":        pos_dist,
            }

        rows.append({
            "style":       conv["style"],
            "final_q":     last_q,
            "full_centroid":     full_centroid.tolist(),
            "full_pos_profile":  full_pos_profile,
            "per_N":            per_N,
        })

        print(f"\n[{ci+1}/{len(CONVERSATIONS)}] style={conv['style']}")
        for N in MEMORY_LENGTHS:
            r = per_N[N]
            print(f"  N={N}  cdist={r['answer_centroid_dist_to_conv']:.3f}  "
                  f"pos_l1={r['answer_pos_l1_to_conv']:.3f}  "
                  f"| {r['answer_text'][:90]!r}")

    # Aggregate: does memory length monotonically reduce centroid distance?
    print("\n=== D13 summary across 8 conversations ===")
    print(f"{'memory N':>9s}  {'cdist_mean':>11s}  {'cdist_med':>10s}  "
          f"{'pos_l1_mean':>12s}")
    summary = {}
    for N in MEMORY_LENGTHS:
        cd = np.array([r["per_N"][N]["answer_centroid_dist_to_conv"]
                        for r in rows])
        pl = np.array([r["per_N"][N]["answer_pos_l1_to_conv"]
                        for r in rows])
        cd = cd[~np.isnan(cd)]; pl = pl[~np.isnan(pl)]
        summary[N] = {
            "centroid_dist_mean":   float(cd.mean()) if cd.size else float("nan"),
            "centroid_dist_median": float(np.median(cd)) if cd.size else float("nan"),
            "pos_l1_mean":          float(pl.mean()) if pl.size else float("nan"),
            "n":                    int(len(cd)),
        }
        print(f"  {N:>9d}  {summary[N]['centroid_dist_mean']:>11.3f}  "
              f"{summary[N]['centroid_dist_median']:>10.3f}  "
              f"{summary[N]['pos_l1_mean']:>12.3f}")

    print(f"\nInterpretation:")
    print(f"  Lower centroid distance = response stays closer to conversation's manifold region")
    print(f"  Lower POS L1 = response register matches conversation register")
    print(f"  If both decrease as N grows, structural memory produces 'personal' feel")

    # Per-style breakdown for the best-performing N (lowest mean centroid dist)
    best_N = min(MEMORY_LENGTHS, key=lambda N: summary[N]["centroid_dist_mean"]
                                                if not np.isnan(summary[N]["centroid_dist_mean"]) else float("inf"))
    baseline_N = 0
    print(f"\n=== Per-style: N={baseline_N} (no memory) vs N={best_N} (best) ===")
    print(f"{'style':>20s}  {'cdist N=0':>10s}  {'cdist best':>11s}  delta")
    for r in rows:
        c0 = r["per_N"][baseline_N]["answer_centroid_dist_to_conv"]
        cb = r["per_N"][best_N]["answer_centroid_dist_to_conv"]
        sign = "+" if cb < c0 else "-"
        print(f"  {r['style']:>20s}  {c0:>10.3f}  {cb:>11.3f}  {sign}{abs(c0-cb):.3f}")

    out_path = RES_DIR / "d13_structural_memory_results.json"
    out_path.write_text(json.dumps({
        "n_conversations": len(rows), "memory_lengths": MEMORY_LENGTHS,
        "summary": summary, "per_conversation": rows,
    }, indent=2, default=str))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
