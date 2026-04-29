"""
Experiment D12: phone-tier memory via Aisha running fingerprint.

The phone tier (Qwen2.5-0.5B-Instruct) has limited usable context on
device -- holding a long conversation in attention is slow on a phone
NPU. Aisha's structural fingerprint is 16 floats per sentence, ~30
floats per turn. Cheap to maintain across an entire conversation.

Test: 10 conversations, each 4 turns. The 4th turn asks a question
that requires information accumulated in turns 1-3. We compare:

  A. NO_MEMORY  : Qwen sees only the 4th user message. No history.
  B. FULL_HIST  : Qwen sees the entire conversation (gold standard).
  C. AISHA_MEM  : Qwen sees only the 4th user message PLUS:
                  - the running Aisha boundary (cumulative content seeds
                    from turns 1-3) used as logit bias
                  - a one-line structural preamble derived from those turns

If C achieves quality close to B but at A's compute cost (no full history
in context), the Aisha-memory mechanism is doing real work and is worth
shipping. If C ≈ A, the fingerprint isn't carrying enough information.

Each conversation is graded by checking whether the assistant's response
contains a specific keyword from the earlier turns that should have been
needed to answer well.
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
LAM = 1.5

# 10 conversations. Each has 3 information-establishing user turns followed
# by a 4th user turn whose answer should reference earlier info.
# memory_keywords: at least one should appear in the answer if memory worked.
CONVERSATIONS = [
    {
        "turns": [
            ("user", "I am planning a trip to Tokyo next month."),
            ("user", "I will be there during cherry blossom season, roughly mid-April."),
            ("user", "I have never been to Japan before."),
            ("user", "What clothes should I bring?"),
        ],
        "memory_keywords": ["tokyo", "japan", "april", "spring", "cherry", "blossom"],
        "topic": "travel-tokyo-april",
    },
    {
        "turns": [
            ("user", "My dog is a black Labrador retriever."),
            ("user", "He is six years old and has chronic chicken allergies."),
            ("user", "I am switching his diet."),
            ("user", "What kind of dog food do you recommend?"),
        ],
        "memory_keywords": ["chicken", "allergy", "allergic", "labrador", "lab", "no chicken", "avoid chicken"],
        "topic": "dog-food-allergy",
    },
    {
        "turns": [
            ("user", "I work as a software engineer at a small company."),
            ("user", "I mostly write Python and machine learning code."),
            ("user", "I have been doing this for three years."),
            ("user", "What technical book would you recommend I read?"),
        ],
        "memory_keywords": ["python", "machine learning", "ml", "engineer", "ai"],
        "topic": "book-recommendation-engineer",
    },
    {
        "turns": [
            ("user", "I had eggs and bacon for breakfast this morning."),
            ("user", "I also had a glass of orange juice and one piece of buttered toast."),
            ("user", "I usually eat similar things every day."),
            ("user", "Was my breakfast healthy?"),
        ],
        "memory_keywords": ["egg", "bacon", "orange", "juice", "toast", "butter"],
        "topic": "breakfast-health",
    },
    {
        "turns": [
            ("user", "I live in San Francisco in a small studio apartment."),
            ("user", "My commute to work is about thirty minutes by bus."),
            ("user", "I leave the house at seven in the morning."),
            ("user", "How can I improve my morning routine?"),
        ],
        "memory_keywords": ["bus", "commute", "san francisco", "studio", "thirty", "30"],
        "topic": "sf-morning-routine",
    },
    {
        "turns": [
            ("user", "I am thinking about adopting a pet."),
            ("user", "My apartment is small and I work long hours."),
            ("user", "I have never had a pet before."),
            ("user", "What kind of pet should I get?"),
        ],
        "memory_keywords": ["small", "apartment", "long hours", "low maintenance", "fish", "cat", "first"],
        "topic": "pet-adoption",
    },
    {
        "turns": [
            ("user", "My grandmother is turning eighty next week."),
            ("user", "She loves gardening and classical music."),
            ("user", "She also enjoys baking pastries."),
            ("user", "What gift should I get her?"),
        ],
        "memory_keywords": ["garden", "music", "classical", "baking", "pastry", "pastries", "kitchen"],
        "topic": "grandmother-gift",
    },
    {
        "turns": [
            ("user", "I am preparing for a job interview at a tech startup."),
            ("user", "The position is for a senior data scientist role."),
            ("user", "The company specializes in healthcare AI."),
            ("user", "What should I read to prepare?"),
        ],
        "memory_keywords": ["data", "science", "healthcare", "startup", "ai", "machine learning"],
        "topic": "interview-prep",
    },
    {
        "turns": [
            ("user", "I started learning Spanish three months ago."),
            ("user", "I practice for thirty minutes every day using an app."),
            ("user", "I can have basic conversations now."),
            ("user", "How can I improve faster?"),
        ],
        "memory_keywords": ["spanish", "app", "conversation", "speak", "intermediate", "three months"],
        "topic": "spanish-learning",
    },
    {
        "turns": [
            ("user", "I am training for my first marathon."),
            ("user", "I currently run about twenty-five kilometers per week."),
            ("user", "The race is in eight weeks."),
            ("user", "How should I train from now until the race?"),
        ],
        "memory_keywords": ["marathon", "kilometers", "km", "weeks", "race", "running", "long run"],
        "topic": "marathon-training",
    },
]


def aisha_running_boundary(responder, prior_turns: list[str]) -> list[str]:
    """Compute Aisha's accumulated boundary from all prior user turns."""
    combined = " ".join(prior_turns)
    if not combined.strip():
        return []
    bset = responder.expand_content_boundary(combined)
    out = []
    for new_i in bset:
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
def generate_qwen(model, tokenizer, prompt, mask, lam, max_new, T, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(enc.input_ids, attention_mask=enc.attention_mask, use_cache=True)
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


def chat_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def memory_score(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    """Return (#keywords found, list of found keywords) in text (case-insensitive)."""
    t = text.lower()
    found = [k for k in keywords if k.lower() in t]
    return len(found), found


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[d12] device: {device}")
    aisha = make_aisha()

    print(f"[d12] loading {BACKBONE_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE_NAME, torch_dtype=torch.float16, device_map="auto").eval()
    print(f"[d12] loaded.")

    rows = []
    for ci, conv in enumerate(CONVERSATIONS):
        prior = [t for r, t in conv["turns"][:-1] if r == "user"]
        last = conv["turns"][-1][1]

        # A. No memory: only last user message
        prompt_A = chat_prompt(tok, [{"role": "user", "content": last}])
        # B. Full history: every prior + last user message in context
        msgs_B = [{"role": "user", "content": t} for r, t in conv["turns"] if r == "user"]
        prompt_B = chat_prompt(tok, msgs_B)
        # C. Aisha-memory: only the last message + Aisha preamble + Aisha boundary bias
        boundary = aisha_running_boundary(aisha, prior)
        struct = aisha_structure(aisha, " ".join(prior))
        preamble = (
            f"[Conversation context fingerprint: "
            f"{len(boundary)} topical content words (e.g. {', '.join(boundary[:8])}); "
            f"earlier-turn structural centroid summarised. Use this to maintain context.]\n\n"
            if boundary else "")
        prompt_C = chat_prompt(tok, [{"role": "user", "content": preamble + last}])
        mask_C = (build_boost_mask(boundary, tok, model.config.vocab_size, device)
                   if boundary else None)

        outs = {}
        torch.manual_seed(0); outs["A_no_memory"]   = generate_qwen(model, tok, prompt_A, None, 0.0, 80, 0.3, device)
        torch.manual_seed(0); outs["B_full_hist"]   = generate_qwen(model, tok, prompt_B, None, 0.0, 80, 0.3, device)
        torch.manual_seed(0); outs["C_aisha_mem"]   = generate_qwen(model, tok, prompt_C, mask_C, LAM, 80, 0.3, device)

        scores = {c: memory_score(t, conv["memory_keywords"]) for c, t in outs.items()}

        rows.append({
            "topic":   conv["topic"],
            "prior":   prior, "last": last,
            "memory_keywords": conv["memory_keywords"],
            "outputs": outs,
            "score":   {c: scores[c][0] for c in outs},
            "found":   {c: scores[c][1] for c in outs},
            "boundary_size": len(boundary),
        })

        print(f"\n[{ci+1}/{len(CONVERSATIONS)}] topic={conv['topic']}")
        print(f"  prior turns: {len(prior)}  boundary={len(boundary)} words")
        print(f"  last Q: {last}")
        for c in ["A_no_memory", "B_full_hist", "C_aisha_mem"]:
            sc, fnd = scores[c]
            print(f"  {c:>14s}  hits={sc}  matched={fnd}")
            print(f"      {outs[c][:140]!r}")

    print("\n=== D12 summary ===")
    n = len(rows)
    print(f"{'condition':>14s}  {'avg hits':>10s}  {'any-hit %':>10s}  {'all-hit %':>10s}")
    summary = {}
    for cond in ["A_no_memory", "B_full_hist", "C_aisha_mem"]:
        scores = np.array([r["score"][cond] for r in rows])
        any_hit = (scores > 0).mean()
        all_hit_thresh = max(1, max(len(r["memory_keywords"]) for r in rows) // 3)
        all_hit = (scores >= all_hit_thresh).mean()
        avg = scores.mean()
        summary[cond] = {
            "avg_hits":      float(avg),
            "any_hit_rate":  float(any_hit),
            "strong_hit_rate": float(all_hit),
        }
        print(f"  {cond:>12s}  {avg:>10.2f}  {any_hit*100:>9.0f}%  {all_hit*100:>9.0f}%")

    print(f"\nInterpretation:")
    print(f"  any-hit  = response mentions ANY relevant keyword from prior turns")
    print(f"  strong-hit = response mentions >= ceil(N/3) of the relevant keywords")
    print(f"  if C >> A and approaches B, Aisha memory is working")

    out_path = RES_DIR / "d12_memory_results.json"
    out_path.write_text(json.dumps({
        "n_conversations": n, "backbone": BACKBONE_NAME, "lambda": LAM,
        "summary": summary, "per_conversation": rows,
    }, indent=2, default=str))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
