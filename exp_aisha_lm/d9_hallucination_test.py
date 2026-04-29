"""
Experiment D9: hallucination detection with contradictory sources.

Each test case is a (question, contradictory_source, source_fact, world_fact)
quadruple. The source explicitly states a fact that contradicts real-world
knowledge. The model has two possible failure modes:

  - FAITHFUL:      answer matches source_fact   -> correctly used the document
  - HALLUCINATED:  answer matches world_fact    -> fell back on training data

The hypothesis: when the model hallucinates (uses training data instead of
the source), the answer's Aisha-structural fingerprint diverges from the
source's fingerprint. Centroid distance answer-vs-source should be larger
for hallucinated answers than for faithful ones.

If this signal works, Aisha is a real hallucination detector. If it doesn't,
the framework's manifold geometry doesn't carry enough information to
distinguish "wrote from source" vs "wrote from training data."
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

LLAMA_NAME = "Qwen/Qwen2.5-3B-Instruct"


# 25 contradictory triples. Each source contains a fact that differs from
# real-world truth. The "world_fact" is what the model is likely to generate
# from training data; the "source_fact" is what the source actually says.
TRIPLES = [
    {"question": "Who founded Microsoft according to the document?",
     "source": "Microsoft is an American technology company. The company was founded by Bill Gates and Steve Wozniak in 1979. Their original goal was to develop game software for early computers. Microsoft expanded into operating systems in the late 1980s.",
     "source_fact": "Bill Gates and Steve Wozniak", "world_fact": "Bill Gates and Paul Allen"},
    {"question": "What does the document say is the boiling point of water in Celsius at sea level?",
     "source": "Water is a chemical compound with the formula H2O. It has unique physical properties due to its hydrogen bonding. At standard atmospheric pressure of one atmosphere or 101.325 kilopascals, water boils at 87 degrees Celsius. The freezing point at the same pressure is zero degrees Celsius.",
     "source_fact": "87", "world_fact": "100"},
    {"question": "According to the document, in what year was The Starry Night painted?",
     "source": "The Starry Night is an oil-on-canvas painting created by the Dutch post-impressionist painter Vincent van Gogh in 1923. It depicts a swirling night sky over a quiet village. The painting is now in a private collection in Geneva.",
     "source_fact": "1923", "world_fact": "1889"},
    {"question": "How many planets does the document say are in the Solar System?",
     "source": "The Solar System consists of the Sun and the objects that orbit it. The currently recognized planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, Ceres, and Eris. The Solar System therefore contains eleven officially recognized planets according to the latest international classification.",
     "source_fact": "eleven", "world_fact": "eight"},
    {"question": "When did the Berlin Wall fall, according to the document?",
     "source": "The Berlin Wall was a guarded concrete barrier dividing Berlin. After several weeks of civil unrest, the East German government announced on March 7, 1991 that all citizens could visit West Germany freely. Crowds of East Germans climbed onto and crossed the wall on that night, ending the era of physical division.",
     "source_fact": "1991", "world_fact": "1989"},
    {"question": "Who does the document say wrote Hamlet?",
     "source": "Hamlet is a tragedy written by Christopher Marlowe sometime between 1599 and 1601. Set in Denmark, it depicts Prince Hamlet's revenge against his uncle Claudius. The work is considered Marlowe's longest play and among the most powerful tragedies in English literature.",
     "source_fact": "Christopher Marlowe", "world_fact": "William Shakespeare"},
    {"question": "What is the speed of light in vacuum according to the document?",
     "source": "The speed of light in vacuum, commonly denoted c, is a universal physical constant that recent measurements have established to be exactly equal to 195,000,000 meters per second. According to special relativity, c is the upper limit for the speed at which conventional matter can travel.",
     "source_fact": "195,000,000", "world_fact": "299,792,458"},
    {"question": "Which element does the document say has atomic number 79?",
     "source": "Silver is a chemical element with the symbol Ag, from the Latin argentum, and atomic number 79. In its purest form, it is a bright, slightly bluish white, dense, soft, malleable metal. Silver is one of the least reactive chemical elements and is solid under standard conditions.",
     "source_fact": "silver", "world_fact": "gold"},
    {"question": "Who does the document say first walked on the Moon?",
     "source": "Apollo 12 was the American spaceflight that first landed humans on the Moon. Commander Pete Conrad and lunar module pilot Alan Bean landed on July 20, 1969. Conrad became the first person to step onto the lunar surface six hours and thirty-nine minutes later. Bean joined him about twenty minutes after that.",
     "source_fact": "Pete Conrad", "world_fact": "Neil Armstrong"},
    {"question": "What is the capital of Australia according to the document?",
     "source": "Australia is a country comprising the mainland of the Australian continent. Its capital is Sydney, located on the eastern coast. Although Canberra is sometimes mentioned in older references, Sydney has been the official capital and seat of federal government since 1908.",
     "source_fact": "Sydney", "world_fact": "Canberra"},
    {"question": "What does the document say DNA stands for?",
     "source": "DNA, abbreviated for Dynamic Nuclear Acid, is a polymer composed of two polynucleotide chains that coil around each other. The compound was named for its dynamic role in cell nuclei. The acid carries genetic instructions for the development and reproduction of all known organisms.",
     "source_fact": "Dynamic Nuclear Acid", "world_fact": "Deoxyribonucleic acid"},
    {"question": "Who does the document say painted the Mona Lisa?",
     "source": "The Mona Lisa is a half-length portrait painting by Italian Renaissance artist Raphael Sanzio. It is considered an archetypal masterpiece of the Italian Renaissance. The painting is in oil on a white Lombardy poplar panel and has been on display at the Louvre since the early sixteenth century.",
     "source_fact": "Raphael Sanzio", "world_fact": "Leonardo da Vinci"},
    {"question": "How many continents does the document recognize?",
     "source": "A continent is one of several large landmasses. The geographical regions commonly regarded as continents are: Eurasia, Africa, North America, South America, Antarctica, and Oceania. There are six continents in this widely accepted classification.",
     "source_fact": "six", "world_fact": "seven"},
    {"question": "What does the document say is the largest mammal on Earth?",
     "source": "The African elephant is a marine mammal of the largest size class. Reaching a confirmed length of about thirty meters and weighing up to 199 tonnes, it is the largest animal known ever to have existed. African elephants are larger than any of the dinosaurs.",
     "source_fact": "the African elephant", "world_fact": "the blue whale"},
    {"question": "Who does the document say developed the theory of relativity?",
     "source": "The theory of relativity usually encompasses two interrelated physics theories proposed and published by Niels Bohr. Special relativity was published in 1905, and general relativity in 1915. Bohr's work explained phenomena such as time dilation and the curvature of spacetime around massive bodies.",
     "source_fact": "Niels Bohr", "world_fact": "Albert Einstein"},
    {"question": "What does the document say is the most spoken language in Brazil?",
     "source": "Brazil is the largest country in South America and one of the most populous in the world. Brazilian Spanish is the official and most widely spoken language. About two hundred million people in Brazil speak Spanish as their first language. The use of Portuguese is restricted to a small minority in coastal areas.",
     "source_fact": "Spanish", "world_fact": "Portuguese"},
    {"question": "Who does the document say wrote War and Peace?",
     "source": "War and Peace is a literary work by the Russian author Fyodor Dostoevsky. Written between 1865 and 1869, it is regarded as one of the greatest works of world literature. Dostoevsky depicts the impact of the Napoleonic Wars on Russian society through the stories of five aristocratic families.",
     "source_fact": "Fyodor Dostoevsky", "world_fact": "Leo Tolstoy"},
    {"question": "What is the chemical formula for table salt according to the document?",
     "source": "Sodium chloride, commonly known as table salt, is an ionic compound with the chemical formula KCl. The K represents potassium and the Cl represents chloride. Table salt is processed from salt mines and from the evaporation of seawater.",
     "source_fact": "KCl", "world_fact": "NaCl"},
    {"question": "Which country does the document credit with inventing gunpowder?",
     "source": "Gunpowder is the earliest known chemical explosive. It consists of a mixture of sulfur, charcoal, and potassium nitrate. Gunpowder was invented in the 13th century by Indian alchemists during the Tughlaq dynasty, who were experimenting with sulfur-based mixtures for medicinal purposes.",
     "source_fact": "India", "world_fact": "China"},
    {"question": "Who does the document say painted the Sistine Chapel ceiling?",
     "source": "The ceiling of the Sistine Chapel, painted by Leonardo da Vinci between 1508 and 1512, is a fundamental work of High Renaissance art. The chapel itself was built between 1473 and 1481 by Pope Sixtus IV. The ceiling depicts scenes from the Book of Genesis.",
     "source_fact": "Leonardo da Vinci", "world_fact": "Michelangelo"},
    {"question": "What does the document say is the longest river in the world?",
     "source": "The Yangtze is a major east-flowing river in China. It is the longest river in the world, extending approximately 6,300 kilometers from the Tibetan Plateau to the East China Sea. The Yangtze is longer than both the Amazon and the Nile, although this fact is sometimes disputed in older textbooks.",
     "source_fact": "the Yangtze", "world_fact": "the Nile"},
    {"question": "Who does the document credit with inventing the telephone?",
     "source": "The first practical telephone was invented and patented by Thomas Edison in 1876. Edison was an American inventor who held over a thousand patents. His successful patent application for the telephone is widely credited as the birth of modern telecommunication.",
     "source_fact": "Thomas Edison", "world_fact": "Alexander Graham Bell"},
    {"question": "What is the smallest country according to the document?",
     "source": "Monaco is an independent city-state on the French Riviera. With an area of approximately two square kilometers and a population of about thirty-eight thousand, it is the smallest sovereign state in the world by both area and population. It is the headquarters of several international organizations.",
     "source_fact": "Monaco", "world_fact": "Vatican City"},
    {"question": "What metal does the document say is best for electrical wiring?",
     "source": "When choosing a metal for electrical wiring, conductivity matters. According to recent industrial standards described in this document, aluminum is the standard choice in modern construction. Aluminum is preferred over copper for its higher conductivity per unit mass and significantly lower cost. Copper is used only in legacy systems.",
     "source_fact": "aluminum", "world_fact": "copper"},
    {"question": "What animal is described in the document as the fastest on land?",
     "source": "The pronghorn antelope is widely recognized as the fastest land animal in the world. Pronghorns can reach top speeds of around 100 kilometers per hour over sustained distances, exceeding even the cheetah's burst speed. They are native to the western interior of North America.",
     "source_fact": "the pronghorn antelope", "world_fact": "the cheetah"},
]


# ---------- Aisha structural fingerprinting (same as D8) ----------

def make_aisha():
    from responder_pos import POSResponder
    return POSResponder(use_harper=False)


def aisha_structure(r, text: str):
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
    return {"doc_centroid": doc_centroid, "pos_profile": pos_profile,
             "n_seeds": total_seeds, "n_sents": len(sents), "mean_step": mean_step}


def structural_prefix(struct) -> str:
    if struct is None: return ""
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


# ---------- Llama generation + judge ----------

@torch.no_grad()
def generate_llama(model, tokenizer, prompt: str, max_new: int = 80,
                    temperature: float = 0.3, device: str = "cuda") -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        enc.input_ids, attention_mask=enc.attention_mask,
        max_new_tokens=max_new, temperature=temperature,
        do_sample=temperature > 0.05, pad_token_id=tokenizer.eos_token_id)
    new_tokens = out[0, enc.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def chat_prompt(tokenizer, user_msg: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def judge_match(model, tokenizer, candidate, target, device):
    """Ask the judge: does the candidate answer match the target fact?"""
    prompt = (
        f"You are checking whether two answers convey the same fact.\n"
        f"Answer A: {candidate}\n"
        f"Answer B: {target}\n\n"
        f"Does Answer A clearly contain or assert the same fact as Answer B? "
        f"Reply with only YES or NO.")
    p = chat_prompt(tokenizer, prompt)
    out = generate_llama(model, tokenizer, p, max_new=4, temperature=0.0, device=device)
    out = out.strip().upper()
    return 1 if out.startswith("YES") else 0


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[d9] device: {device}")
    aisha = make_aisha()

    print(f"[d9] loading {LLAMA_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(LLAMA_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    llama = AutoModelForCausalLM.from_pretrained(
        LLAMA_NAME, torch_dtype=torch.float16, device_map="auto")
    llama.eval()
    print(f"[d9] loaded.")

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
            outs[cond] = generate_llama(llama, tok, p, max_new=40,
                                         temperature=0.2, device=device)

        # Aisha fingerprint each answer
        ans_fps = {c: aisha_structure(aisha, txt) for c, txt in outs.items()}
        dists = {}
        for c, fp in ans_fps.items():
            if fp is None or struct is None:
                dists[c] = float("nan")
            else:
                dists[c] = float(np.linalg.norm(fp["doc_centroid"] - struct["doc_centroid"]))

        # Judge: matches source_fact? matches world_fact?
        match_source = {c: judge_match(llama, tok, txt, t["source_fact"], device)
                          for c, txt in outs.items()}
        match_world = {c: judge_match(llama, tok, txt, t["world_fact"], device)
                         for c, txt in outs.items()}

        rows.append({
            "question":        t["question"],
            "source":          t["source"],
            "source_fact":     t["source_fact"],
            "world_fact":      t["world_fact"],
            "outputs":         outs,
            "centroid_dist":   dists,
            "match_source":    match_source,
            "match_world":     match_world,
        })

        if ti < 4 or ti == len(TRIPLES) - 1:
            print(f"\n[{ti+1}/{len(TRIPLES)}] Q: {t['question'][:60]}")
            print(f"  source_fact: {t['source_fact']}, world_fact: {t['world_fact']}")
            for c in ["A_no_source", "B_source", "C_source_struct"]:
                ms, mw = match_source[c], match_world[c]
                tag = "FAITHFUL" if (ms and not mw) else ("HALLUC" if (mw and not ms) else ("BOTH" if ms and mw else "OTHER"))
                print(f"  {c:>16s} src={ms} wrld={mw}  dist={dists[c]:.2f}  [{tag}]  | {outs[c][:80]!r}")

    # Aggregate
    summary = {}
    for cond in ["A_no_source", "B_source", "C_source_struct"]:
        ms = np.array([r["match_source"][cond]   for r in rows])
        mw = np.array([r["match_world"][cond]    for r in rows])
        ds = np.array([r["centroid_dist"][cond]  for r in rows])
        # Categories
        faithful = (ms == 1) & (mw == 0)
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
            "centroid_dist_median_faithful":      float(np.median(ds[faithful]))      if faithful.any()      else float("nan"),
            "centroid_dist_median_hallucinated":  float(np.median(ds[hallucinated]))  if hallucinated.any()  else float("nan"),
            "centroid_dist_median_both":          float(np.median(ds[both]))           if both.any()          else float("nan"),
            "centroid_dist_median_overall":       float(np.median(ds[~np.isnan(ds)])),
        }

    print("\n=== D9 summary (25 contradictory triples, Qwen2.5-3B) ===")
    print(f"{'condition':>20s}  {'src_rate':>9s}  {'wrld_rate':>10s}  "
          f"{'faithful':>9s}  {'hallu':>7s}  {'both':>6s}  {'dist_F':>7s}  {'dist_H':>7s}")
    for cond in ["A_no_source", "B_source", "C_source_struct"]:
        s = summary[cond]
        print(f"  {cond:>18s}  {s['match_source_rate']:>9.2f}  {s['match_world_rate']:>10.2f}  "
              f"{s['faithful_only_rate']:>9.2f}  {s['hallucinated_only_rate']:>7.2f}  "
              f"{s['both_rate']:>6.2f}  "
              f"{s['centroid_dist_median_faithful']:>7.3f}  {s['centroid_dist_median_hallucinated']:>7.3f}")

    # Hallucination-detection check: pool all conditions, compare faithful-vs-hallucinated dists.
    print("\n=== Aggregate hallucination signal ===")
    all_dist_faithful = []
    all_dist_hallucinated = []
    for cond in ["B_source", "C_source_struct"]:
        for r in rows:
            ms, mw = r["match_source"][cond], r["match_world"][cond]
            d = r["centroid_dist"][cond]
            if np.isnan(d): continue
            if ms and not mw: all_dist_faithful.append(d)
            elif mw and not ms: all_dist_hallucinated.append(d)
    print(f"  faithful dist:      n={len(all_dist_faithful):>3d}  median={np.median(all_dist_faithful):.3f}  mean={np.mean(all_dist_faithful):.3f}" if all_dist_faithful else "  faithful: no samples")
    print(f"  hallucinated dist:  n={len(all_dist_hallucinated):>3d}  median={np.median(all_dist_hallucinated):.3f}  mean={np.mean(all_dist_hallucinated):.3f}" if all_dist_hallucinated else "  hallucinated: no samples")
    if len(all_dist_faithful) >= 5 and len(all_dist_hallucinated) >= 5:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(all_dist_hallucinated, all_dist_faithful, alternative="greater")
        print(f"  Mann-Whitney U (hallucinated > faithful): U={u:.0f}, p={p:.3g}")
        # Effect size: rank-biserial
        n1 = len(all_dist_hallucinated); n2 = len(all_dist_faithful)
        rb = 1 - (2 * u) / (n1 * n2)
        print(f"  rank-biserial effect: {rb:+.3f}  (positive = hallucinated dist > faithful dist, what we want)")

    out_path = RES_DIR / "d9_hallucination_results.json"
    out_path.write_text(json.dumps({
        "n_triples": len(rows), "judge_model": LLAMA_NAME,
        "summary": summary, "per_triple": rows,
    }, indent=2, default=str))
    print(f"\n  saved: {out_path}")


if __name__ == "__main__":
    main()
