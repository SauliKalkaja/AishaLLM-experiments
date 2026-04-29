"""
Experiment D8: high-end pipeline with structure prefix + hallucination check.

Tests two questions in one experiment, on Llama-3.2-3B-Instruct:

  Q1: Does Aisha's structural fingerprint of a source document, fed to a
       high-end backbone alongside the raw text, improve answer quality
       or faithfulness? (Yourliterral high-end design.)

  Q2: Does Aisha's centroid distance between answer and source act as a
       hallucination detector? (No-source vs source-given conditions
       should differ structurally even when both look fluent.)

For each (question, source, gold_answer) triple, three conditions:
  A. NO_SOURCE       : backbone has only the question. Forces hallucination
                         from training data, our hallucination ground truth.
  B. SOURCE_RAW      : backbone has question + source. Faithful baseline.
  C. SOURCE_STRUCT   : backbone has question + source + Aisha structural
                         summary as a preamble.

Per condition we record:
  - generated answer text
  - Aisha 16D centroid + POS profile of the answer
  - centroid distance to the source's centroid
  - judge-LM rating of correctness vs gold (1-5)
  - judge-LM rating of faithfulness vs source (1-5)
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

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
RES_DIR = EXP_ROOT.parent / "results"
WORD_RE = re.compile(r"[a-zA-Z']+")
SENT_RE = re.compile(r"(?<=[.!?])\s+")
POS_NAMES = ["NOUN", "VERB", "ADJ", "ADV", "PRON_DET", "PREP", "CONJ", "INTJ"]

LLAMA_NAME = "Qwen/Qwen2.5-3B-Instruct"   # open-access, ~3B params, instruction-tuned

# 25 (question, source, gold_answer) triples covering varied domains.
TRIPLES = [
    ("question", "Who founded Microsoft?",
     "Microsoft is an American multinational technology corporation. The company was founded by Bill Gates and Paul Allen on April 4, 1975. Their original goal was to develop and sell BASIC interpreters for the Altair 8800. Microsoft became dominant in the personal computer operating system market with MS-DOS, followed by Windows.",
     "Bill Gates and Paul Allen"),
    ("question", "What is the boiling point of water in Celsius at sea level?",
     "Water is a chemical compound with the formula H2O. It has unique physical properties due to its hydrogen bonding. At standard atmospheric pressure of one atmosphere or 101.325 kilopascals, water boils at 100 degrees Celsius. The freezing point at the same pressure is zero degrees Celsius.",
     "100 degrees Celsius"),
    ("question", "Which painter created The Starry Night?",
     "The Starry Night is an oil-on-canvas painting created in June 1889 by the Dutch post-impressionist painter Vincent van Gogh. It depicts the view from the east-facing window of his asylum room at Saint-Remy-de-Provence. The painting is now in the permanent collection of the Museum of Modern Art in New York City.",
     "Vincent van Gogh"),
    ("question", "How many planets are in the Solar System?",
     "The Solar System consists of the Sun and the objects that orbit it, either directly or indirectly. The currently recognized planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was reclassified as a dwarf planet in 2006 by the International Astronomical Union. The Solar System therefore contains eight officially recognized planets.",
     "eight"),
    ("question", "What year did the Berlin Wall fall?",
     "The Berlin Wall was a guarded concrete barrier that physically and ideologically divided Berlin from 1961 to 1989. Construction began on August 13, 1961. After several weeks of civil unrest, the East German government announced on November 9, 1989 that all GDR citizens could visit West Germany and West Berlin. Crowds of East Germans climbed onto and crossed the wall that night.",
     "1989"),
    ("question", "Who wrote the play Hamlet?",
     "Hamlet is a tragedy written by William Shakespeare sometime between 1599 and 1601. Set in Denmark, the play depicts Prince Hamlet's revenge against his uncle Claudius, who has murdered Hamlet's father in order to seize his throne and marry Hamlet's mother. Hamlet is Shakespeare's longest play and is considered among the most powerful tragedies in world literature.",
     "William Shakespeare"),
    ("question", "What is the speed of light in a vacuum?",
     "The speed of light in vacuum, commonly denoted c, is a universal physical constant exactly equal to 299,792,458 meters per second. According to special relativity, c is the upper limit for the speed at which conventional matter, energy, or any signal carrying information can travel through space.",
     "299,792,458 meters per second"),
    ("question", "Which element has the atomic number 79?",
     "Gold is a chemical element with the symbol Au, from the Latin aurum, and atomic number 79. In its purest form, it is a bright, slightly reddish yellow, dense, soft, malleable, and ductile metal. Chemically, gold is a transition metal and a group eleven element. It is one of the least reactive chemical elements and is solid under standard conditions.",
     "gold"),
    ("question", "Who was the first person to walk on the Moon?",
     "Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969. Armstrong became the first person to step onto the lunar surface six hours and thirty-nine minutes later. Aldrin joined him about twenty minutes later.",
     "Neil Armstrong"),
    ("question", "What is the capital of Australia?",
     "Australia is a country comprising the mainland of the Australian continent, the island of Tasmania, and numerous smaller islands. Its capital is Canberra, located in the Australian Capital Territory. Sydney is the largest city, but it is not the capital. Canberra was selected as a compromise between Sydney and Melbourne in 1908.",
     "Canberra"),
    ("question", "What does DNA stand for?",
     "Deoxyribonucleic acid, abbreviated DNA, is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development, functioning, growth, and reproduction of all known organisms and many viruses. DNA and ribonucleic acid are nucleic acids.",
     "Deoxyribonucleic acid"),
    ("question", "Who painted the Mona Lisa?",
     "The Mona Lisa is a half-length portrait painting by Italian Renaissance artist Leonardo da Vinci. It is considered an archetypal masterpiece of the Italian Renaissance. The painting is in oil on a white Lombardy poplar panel, has been believed to have been painted between 1503 and 1519. It is on permanent display at the Louvre in Paris.",
     "Leonardo da Vinci"),
    ("question", "How many continents are there?",
     "A continent is one of several large landmasses. Generally identified by convention rather than any strict criteria, up to seven geographical regions are commonly regarded as continents. Ordered from largest in area to smallest these are: Asia, Africa, North America, South America, Antarctica, Europe, and Oceania.",
     "seven"),
    ("question", "What is the largest mammal on Earth?",
     "The blue whale is a marine mammal and a baleen whale. Reaching a maximum confirmed length of about thirty meters and weighing up to 199 tonnes, it is the largest animal known ever to have existed. Blue whales are larger than any of the dinosaurs, whose maximum size has been estimated by their fossilized skeletal remains.",
     "blue whale"),
    ("question", "Who developed the theory of relativity?",
     "The theory of relativity usually encompasses two interrelated physics theories proposed and published by Albert Einstein. Special relativity was published in 1905, and general relativity in 1915. Together they explain phenomena such as time dilation, length contraction, gravitational lensing, and the curvature of spacetime around massive bodies.",
     "Albert Einstein"),
    ("question", "What language is most spoken in Brazil?",
     "Brazil is the largest country in South America and the world's fifth-largest country by area. Brazilian Portuguese is the official and most widely spoken language. It differs from European Portuguese in pronunciation, vocabulary, and some grammatical conventions. About two hundred million people in Brazil speak Portuguese as their first language.",
     "Portuguese"),
    ("question", "What is photosynthesis?",
     "Photosynthesis is a biological process used by plants, algae, and some bacteria to convert light energy into chemical energy. The chemical energy is stored in carbohydrate molecules like glucose, which are synthesized from carbon dioxide and water. Photosynthesis is responsible for producing and maintaining the oxygen content of Earth's atmosphere.",
     "a biological process used by plants, algae, and some bacteria to convert light energy into chemical energy"),
    ("question", "Who wrote War and Peace?",
     "War and Peace is a literary work by the Russian author Leo Tolstoy. Written between 1865 and 1869, it is regarded as one of the greatest works of world literature. It depicts the impact of the Napoleonic Wars on Russian society through the stories of five aristocratic families.",
     "Leo Tolstoy"),
    ("question", "What is the chemical formula for table salt?",
     "Sodium chloride, commonly known as salt, is an ionic compound with the chemical formula NaCl, representing a one-to-one ratio of sodium and chloride ions. Salt is essential for life in general, and saltiness is one of the basic human tastes. Salt is processed from salt mines, and by the evaporation of seawater.",
     "NaCl"),
    ("question", "Which country invented gunpowder?",
     "Gunpowder, also commonly known as black powder, is the earliest known chemical explosive. It consists of a mixture of sulfur, charcoal, and potassium nitrate. Gunpowder was invented in the 9th century by Chinese alchemists during the Tang dynasty, who were originally seeking an elixir of immortality.",
     "China"),
    ("question", "What is the main ingredient in bread?",
     "Bread is a staple food prepared from a dough of flour and water, usually by baking. Throughout recorded history and around the world, it has been an important part of many cultures' diets. Different types of flour can be used: wheat flour is most common, but rye, barley, oat, and other flours are also used.",
     "flour"),
    ("question", "Who painted the Sistine Chapel ceiling?",
     "The ceiling of the Sistine Chapel, painted by Michelangelo between 1508 and 1512, is a fundamental work of High Renaissance art. The chapel itself was built between 1473 and 1481 by Pope Sixtus IV. The ceiling depicts scenes from the Book of Genesis, including the iconic Creation of Adam.",
     "Michelangelo"),
    ("question", "What is the longest river in the world?",
     "The Nile is a major north-flowing river in northeastern Africa. It is the longest river in Africa, and historically considered the longest river in the world, extending approximately 6,650 kilometers. However, this is disputed by some sources who argue that the Amazon River in South America is actually longer.",
     "the Nile"),
    ("question", "Who invented the telephone?",
     "The first practical telephone was invented and patented by Alexander Graham Bell in 1876. Bell was a Scottish-born inventor and scientist who had emigrated to Canada and then to the United States. His successful patent application for the telephone is widely credited as the birth of modern telecommunication.",
     "Alexander Graham Bell"),
    ("question", "What is the smallest country in the world?",
     "Vatican City is an independent city-state and enclave surrounded by Rome, Italy. With an area of approximately 49 hectares and a population of about 825 as of 2019, it is the smallest sovereign state in the world by both area and population. It is the headquarters of the Roman Catholic Church.",
     "Vatican City"),
]


# ---------- Aisha structural fingerprinting ----------

def make_aisha():
    from responder_pos import POSResponder
    return POSResponder(use_harper=False)


def aisha_structure(r, text: str):
    """Compute (centroid_16d, pos_profile, n_seeds, n_sents, mean_step, doc_centroid)
    from text. Used for both source documents and generated answers."""
    sents = [s.strip() for s in SENT_RE.split(text.replace("\n", " ").strip()) if s.strip()]
    if not sents:
        return None
    Q = r._main_q
    pos_arg = r.wm.pi.argmax(axis=1)
    sent_centroids = []
    pos_count = Counter()
    total_seeds = 0
    for s in sents:
        sids = []
        for w in WORD_RE.findall(s.lower()):
            old_i = r.wm.idx.get(w)
            if old_i is None:
                continue
            p = int(pos_arg[old_i])
            if 0 <= p < len(POS_NAMES):
                pos_count[POS_NAMES[p]] += 1
            new_i = int(r._old_to_new[old_i])
            if new_i >= 0 and not r.R.is_stopword[old_i]:
                sids.append(new_i)
        if sids:
            sent_centroids.append(Q[sids].mean(axis=0))
            total_seeds += len(sids)
    if not sent_centroids:
        return None
    sent_centroids = np.stack(sent_centroids)
    doc_centroid = sent_centroids.mean(axis=0)
    if len(sent_centroids) >= 2:
        steps = np.linalg.norm(np.diff(sent_centroids, axis=0), axis=-1)
        mean_step = float(steps.mean())
    else:
        mean_step = 0.0
    total = sum(pos_count.values())
    pos_profile = {p: pos_count[p] / max(total, 1) for p in POS_NAMES[:4]}
    return {
        "doc_centroid":   doc_centroid,
        "sent_centroids": sent_centroids,
        "pos_profile":    pos_profile,
        "n_seeds":        total_seeds,
        "n_sents":        len(sents),
        "mean_step":      mean_step,
    }


def structural_prefix(struct) -> str:
    """Render Aisha's structural fingerprint as natural-language preamble
    that an instruction-tuned LM can read."""
    if struct is None:
        return ""
    pp = struct["pos_profile"]
    c = struct["doc_centroid"]
    centroid_str = "[" + ",".join(f"{x:.2f}" for x in c) + "]"
    return ("[Source structural fingerprint: "
            f"{struct['n_sents']} sentences, "
            f"{struct['n_seeds']} content words on manifold; "
            f"POS distribution N={pp.get('NOUN', 0):.0%} "
            f"V={pp.get('VERB', 0):.0%} "
            f"A={pp.get('ADJ', 0):.0%} "
            f"Adv={pp.get('ADV', 0):.0%}; "
            f"inter-sentence step {struct['mean_step']:.2f}; "
            f"16D centroid {centroid_str}]")


# ---------- Llama generation ----------

@torch.no_grad()
def generate_llama(model, tokenizer, prompt: str, max_new: int = 80,
                    temperature: float = 0.3, device: str = "cuda") -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        enc.input_ids,
        max_new_tokens=max_new,
        temperature=temperature,
        do_sample=temperature > 0.05,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0, enc.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def chat_prompt(tokenizer, user_msg: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True)


# ---------- Judge ----------

@torch.no_grad()
def judge(model, tokenizer, question, source, gold, answer, device):
    """Two ratings: correctness (vs gold) and faithfulness (vs source)."""
    prompt_correct = (
        f"You are evaluating an answer.\n"
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Candidate answer: {answer}\n\n"
        f"Rate correctness from 1 to 5 (5=fully correct, 1=fully wrong). "
        f"Respond with a single integer only.")
    prompt_faith = (
        f"You are evaluating an answer's faithfulness to a source document.\n"
        f"Source: {source}\n"
        f"Candidate answer: {answer}\n\n"
        f"Rate faithfulness 1-5 (5=every claim in the answer is supported by the source, "
        f"1=answer contradicts or is unsupported). Respond with a single integer only.")
    correctness = _ask_judge(model, tokenizer, prompt_correct, device)
    faithfulness = _ask_judge(model, tokenizer, prompt_faith, device)
    return {"correctness": correctness, "faithfulness": faithfulness}


def _ask_judge(model, tokenizer, prompt, device):
    p = chat_prompt(tokenizer, prompt)
    out = generate_llama(model, tokenizer, p, max_new=8, temperature=0.0, device=device)
    m = re.search(r"[1-5]", out)
    return int(m.group(0)) if m else 0


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[d8] device: {device}")
    aisha = make_aisha()

    print(f"[d8] loading {LLAMA_NAME} (this may take a while on first run)", flush=True)
    tok = AutoTokenizer.from_pretrained(LLAMA_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llama = AutoModelForCausalLM.from_pretrained(
        LLAMA_NAME, torch_dtype=torch.float16, device_map="auto")
    llama.eval()
    print(f"[d8] loaded.", flush=True)

    rows = []
    t0 = time.time()
    for ti, (kind, question, source, gold) in enumerate(TRIPLES):
        struct = aisha_structure(aisha, source)
        prefix = structural_prefix(struct)

        prompts = {
            "A_no_source":   chat_prompt(tok, f"Answer the question briefly.\n\nQ: {question}"),
            "B_source":      chat_prompt(tok, f"Use only the document to answer briefly.\n\nDocument:\n{source}\n\nQ: {question}"),
            "C_source_struct": chat_prompt(tok, f"Use only the document to answer briefly.\n\n{prefix}\n\nDocument:\n{source}\n\nQ: {question}"),
        }

        outs = {}
        for cond, p in prompts.items():
            outs[cond] = generate_llama(llama, tok, p, max_new=60,
                                         temperature=0.3, device=device)

        # Aisha fingerprint each answer; compute centroid distance to source
        ans_fps = {c: aisha_structure(aisha, t) for c, t in outs.items()}
        dists = {}
        for c, fp in ans_fps.items():
            if fp is None or struct is None:
                dists[c] = float("nan")
            else:
                dists[c] = float(np.linalg.norm(fp["doc_centroid"] - struct["doc_centroid"]))

        # Judge each answer
        ratings = {c: judge(llama, tok, question, source, gold, t, device)
                    for c, t in outs.items()}

        rows.append({
            "question":          question,
            "gold":              gold,
            "source":            source,
            "outputs":           outs,
            "answer_centroid_dist_to_source": dists,
            "ratings":           ratings,
            "source_n_sents":    struct["n_sents"] if struct else 0,
            "source_n_seeds":    struct["n_seeds"] if struct else 0,
        })

        if ti < 4 or ti == len(TRIPLES) - 1:
            print(f"\n[{ti+1}/{len(TRIPLES)}] Q: {question}")
            print(f"  gold: {gold!r}")
            for c in ["A_no_source", "B_source", "C_source_struct"]:
                r = ratings[c]; d = dists[c]
                print(f"  {c:>16s}  corr={r['correctness']}  faith={r['faithfulness']}  "
                       f"dist={d:.2f}  | {outs[c][:90]!r}")
        if (ti + 1) % 5 == 0:
            print(f"  -> elapsed {time.time()-t0:.0f}s")

    # Aggregate
    summary = {}
    for cond in ["A_no_source", "B_source", "C_source_struct"]:
        corrs = [r["ratings"][cond]["correctness"]   for r in rows]
        faiths = [r["ratings"][cond]["faithfulness"] for r in rows]
        dists = [r["answer_centroid_dist_to_source"][cond] for r in rows
                  if not np.isnan(r["answer_centroid_dist_to_source"][cond])]
        summary[cond] = {
            "correctness_mean":   float(np.mean(corrs)),
            "correctness_median": float(np.median(corrs)),
            "faithfulness_mean":  float(np.mean(faiths)),
            "faithfulness_median": float(np.median(faiths)),
            "centroid_dist_mean":   float(np.mean(dists)) if dists else float("nan"),
            "centroid_dist_median": float(np.median(dists)) if dists else float("nan"),
            "n":                  len(rows),
        }

    print("\n=== D8 summary (25 triples, Llama-3.2-3B-Instruct) ===")
    print(f"{'condition':>20s}  {'corr_mean':>10s}  {'faith_mean':>11s}  "
          f"{'dist_med':>9s}  {'dist_mean':>10s}")
    for cond in ["A_no_source", "B_source", "C_source_struct"]:
        s = summary[cond]
        print(f"  {cond:>18s}  {s['correctness_mean']:>10.2f}  "
              f"{s['faithfulness_mean']:>11.2f}  "
              f"{s['centroid_dist_median']:>9.3f}  {s['centroid_dist_mean']:>10.3f}")

    # Hallucination-detection check: does centroid distance correlate with faithfulness?
    print("\n=== Hallucination-detection check ===")
    print("Per-condition: how often does answer with high centroid distance also have low faithfulness?")
    from scipy.stats import spearmanr
    for cond in ["A_no_source", "B_source", "C_source_struct"]:
        ds = [r["answer_centroid_dist_to_source"][cond] for r in rows]
        fs = [r["ratings"][cond]["faithfulness"]        for r in rows]
        ds = np.array(ds); fs = np.array(fs)
        mask = ~np.isnan(ds)
        if mask.sum() < 5:
            print(f"  {cond}: insufficient data")
            continue
        rho, p = spearmanr(ds[mask], fs[mask])
        print(f"  {cond:>18s}: rho(dist, faith) = {rho:+.3f}  p={p:.3g}  "
              f"(negative rho = larger dist -> lower faithfulness, what we want)")

    out_path = RES_DIR / "d8_highend_results.json"
    out_path.write_text(json.dumps({
        "n_triples": len(rows), "judge_model": LLAMA_NAME,
        "summary": summary, "per_triple": rows,
    }, indent=2, default=str))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
