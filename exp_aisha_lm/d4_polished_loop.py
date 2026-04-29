"""
Experiment D4: closed-loop final pipeline with Harper polish.

Loads D2's pre-generated outputs and runs each through Harper for
grammar/spelling cleanup, then prints prompt + key conditions side by
side for hand inspection. No re-generation on the pod -- just final
post-processing on local where Harper-cli lives.

Selected conditions for the side-by-side:
    - aisha_alone                 (current pipeline, gibberish baseline)
    - mamba-130m_alone            (no Aisha conditioning)
    - mamba-130m_aisha_lam1.5     (D1's sweet spot)
    - pythia-160m_aisha_lam1.5    (D2's discovered better backbone)
"""
import json
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
RES_DIR = EXP_ROOT.parent / "results"
AISHA_ROOT = Path("/home/sale/AishaLLM/aisha")
sys.path.insert(0, str(AISHA_ROOT))

from harper_polish import polish

D2_PATH = RES_DIR / "d2_smaller_lm_results.json"
OUT_PATH = RES_DIR / "d4_polished_results.json"

SHOW_CONDITIONS = [
    "aisha_alone",
    "mamba-130m_alone",
    "mamba-130m_aisha_lam1.5",
    "pythia-160m_aisha_lam1.5",
]


def clean_for_polish(text: str) -> str:
    """Strip the leading space and any 'User:'/'Assistant:' role prefixes
    that bled through from the prompt format. Truncate at first newline so
    we polish only the immediate response."""
    if not text:
        return text
    # First line only.
    text = text.split("\n", 1)[0].strip()
    return text


def main():
    print(f"[d4] loading {D2_PATH}")
    d2 = json.loads(D2_PATH.read_text())
    rows = d2["per_prompt"]

    polished_results = []

    show_indices = list(range(min(20, len(rows))))
    print(f"[d4] showing {len(show_indices)} of {len(rows)} prompts\n")

    for i in show_indices:
        r = rows[i]
        prompt = r["prompt"]
        boundary_size = r.get("boundary_size", 0)

        print("=" * 88)
        print(f"PROMPT [{i+1}/{len(show_indices)}]: {prompt}")
        print(f"  boundary words: {boundary_size}")
        print()

        polished = {}
        for cond in SHOW_CONDITIONS:
            raw = r["outputs"].get(cond, "")
            cleaned = clean_for_polish(raw)
            try:
                polished_text = polish(cleaned, timeout=4.0)
            except Exception as e:
                polished_text = cleaned
            polished[cond] = {
                "raw":      raw,
                "cleaned":  cleaned,
                "polished": polished_text,
            }
            label = cond.replace("_", " ")
            print(f"  [{label}]")
            print(f"      raw      : {cleaned!r}")
            if polished_text != cleaned:
                print(f"      polished : {polished_text!r}")
            print()

        polished_results.append({
            "prompt":         prompt,
            "boundary_size":  boundary_size,
            "outputs":        polished,
        })

    OUT_PATH.write_text(json.dumps({"polished": polished_results}, indent=2))
    print(f"\n[d4] saved {OUT_PATH}")


if __name__ == "__main__":
    main()
