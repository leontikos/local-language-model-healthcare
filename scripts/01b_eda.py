"""
KROK 2b — EDA + DATA QUALITY CHECKS
=====================================
1. Rozkład A/B/C/D odpowiedzi
2. Rozkład subject_name (21 kategorii)
3. Długość pytań w tokenach
4. Brakujące pola exp
5. Cross-dataset overlap: MedMCQA ↔ MedQA-USMLE ↔ MMLU (contamination check)
6. Near-duplicate check: train_ft vs probe_set

Figury → figures/eda_*.pdf
Raport → results/eda_report.json

Użycie:
    python scripts/01b_eda.py
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def normalize_q(text: str) -> str:
    """Lowercase + strip punctuation/whitespace for overlap checks."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def save_fig(fig, name: str) -> None:
    path = FIGURES_DIR / f"eda_{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    log.info(f"  Zapisano: {path}")


# ─────────────────────────────────────────────────────────────────────
# 1 + 2 + 3 + 4  MedMCQA internal EDA
# ─────────────────────────────────────────────────────────────────────

def run_medmcqa_eda(train_single, train_ft_idx, probe_idx,
                    routing_train_idx=None, iso_cal_idx=None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    from transformers import AutoTokenizer

    cop_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    report = {}

    # ── 1. Rozkład A/B/C/D ──────────────────────────────────────────
    log.info("1. Rozkład A/B/C/D odpowiedzi …")
    answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for idx in train_ft_idx:
        ex = train_single[idx]
        letter = cop_map.get(ex["cop"], None)
        if letter:
            answer_counts[letter] += 1

    total = sum(answer_counts.values())
    answer_pct = {k: 100 * v / total for k, v in answer_counts.items()}
    log.info(f"   {answer_counts}  (n={total:,})")
    for k, pct in answer_pct.items():
        log.info(f"   {k}: {pct:.1f}%")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(answer_counts.keys(), answer_counts.values(),
                  color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
                  edgecolor="white", linewidth=0.8)
    ax.axhline(total / 4, color="black", linestyle="--",
               linewidth=1.2, label="Uniform (25%)")
    for bar, pct in zip(bars, answer_pct.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 300, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Answer option")
    ax.set_ylabel("Count (train_ft)")
    ax.set_title("Answer distribution — MedMCQA train_ft")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    save_fig(fig, "answer_dist")
    plt.close(fig)

    # Chi-squared test for uniformity (H0: all options equally likely)
    from scipy.stats import chisquare, chi2_contingency
    obs = list(answer_counts.values())
    chi2_stat, chi2_p = chisquare(obs)
    # Cramér's V: effect size for chi-squared; range [0, 1]; 0=uniform, 1=all one option.
    # For uniformity test with k=4 categories and n observations: V = sqrt(chi2 / (n*(k-1)))
    cramers_v = float(np.sqrt(chi2_stat / (total * (4 - 1))))
    log.info(f"   Chi-squared uniformity test: χ²={chi2_stat:.1f}  p={chi2_p:.2e}  "
             f"Cramér's V={cramers_v:.3f}")
    if chi2_p < 0.01:
        log.warning("   ⚠ Answer distribution is NON-UNIFORM (p<0.01). "
                    "Mention as limitation: model may learn position bias.")
    else:
        log.info("   ✓ Answer distribution is consistent with uniform (p≥0.01)")

    # Also check probe_set answer distribution and test against train_ft
    probe_answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for idx in probe_idx:
        ex = train_single[idx]
        letter = cop_map.get(ex["cop"], None)
        if letter:
            probe_answer_counts[letter] += 1

    probe_total = sum(probe_answer_counts.values())
    if probe_total > 0:
        # Uniformity test on probe
        probe_obs = list(probe_answer_counts.values())
        chi2_probe, p_probe = chisquare(probe_obs)
        # Cross-distribution test: train_ft vs probe (H0: same distribution)
        contingency = np.array([obs, probe_obs])
        chi2_cross, p_cross, _, _ = chi2_contingency(contingency)
        log.info(f"   Probe answer distribution: {probe_answer_counts}")
        log.info(f"   Probe chi-squared uniformity: χ²={chi2_probe:.1f}  p={p_probe:.2e}")
        log.info(f"   Cross-distribution (train_ft vs probe): χ²={chi2_cross:.2f}  p={p_cross:.2e}")
        if p_cross < 0.05:
            log.warning("   ⚠ train_ft and probe have significantly different answer "
                        "distributions — stratification by answer not guaranteed.")
        else:
            log.info("   ✓ train_ft and probe answer distributions are consistent")

    report["answer_distribution"] = {
        "counts": answer_counts,
        "pct": {k: round(v, 2) for k, v in answer_pct.items()},
        "max_deviation_pp": round(max(abs(v - 25) for v in answer_pct.values()), 2),
        "chi2_stat": round(float(chi2_stat), 2),
        "chi2_p": float(chi2_p),
        "cramers_v": round(cramers_v, 4),
        "is_uniform_p01": bool(chi2_p >= 0.01),
        "probe_counts": probe_answer_counts if probe_total > 0 else {},
        "cross_chi2_p": round(float(p_cross), 4) if probe_total > 0 else None,
    }

    # ── 2. Rozkład subject_name ──────────────────────────────────────
    log.info("2. Rozkład subject_name …")
    import collections
    subject_counts = collections.Counter(
        train_single[i]["subject_name"] for i in train_ft_idx
    )
    subjects_sorted = sorted(subject_counts.items(), key=lambda x: -x[1])
    log.info(f"   Liczba kategorii: {len(subjects_sorted)}")
    for subj, cnt in subjects_sorted:
        log.info(f"   {subj:<40} {cnt:6,}  ({100*cnt/len(train_ft_idx):.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [s for s, _ in subjects_sorted]
    vals  = [v for _, v in subjects_sorted]
    y_pos = range(len(names))
    ax.barh(list(y_pos), vals, color="#4C72B0", edgecolor="white", linewidth=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count (train_ft)")
    ax.set_title("Subject distribution — MedMCQA train_ft")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    save_fig(fig, "subject_dist")
    plt.close(fig)

    report["subject_distribution"] = {
        subj: cnt for subj, cnt in subjects_sorted
    }

    # ── 2b. Stratification bar chart (train_ft vs probe, side-by-side) ──
    log.info("2b. Stratification bar chart (train_ft vs probe) …")
    probe_subject_counts_pre = collections.Counter(
        train_single[i]["subject_name"] for i in probe_idx
    )
    all_subjects_sorted = [s for s, _ in subjects_sorted]
    n_subjects = len(all_subjects_sorted)
    n_ft_total = len(train_ft_idx)
    n_probe_total = len(probe_idx)

    ft_pcts = [100 * subject_counts.get(s, 0) / n_ft_total for s in all_subjects_sorted]
    pr_pcts = [100 * probe_subject_counts_pre.get(s, 0) / n_probe_total
               for s in all_subjects_sorted]

    x_pos = np.arange(n_subjects)
    width = 0.40
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x_pos - width / 2, ft_pcts, width, label="train_ft", color="#4C72B0",
           edgecolor="white", linewidth=0.5)
    ax.bar(x_pos + width / 2, pr_pcts, width, label="probe_set", color="#DD8452",
           edgecolor="white", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_subjects_sorted, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Stratification verification: train_ft vs probe_set subject distribution")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, "stratification")
    plt.close(fig)

    # ── 2c. Data quality: empty / duplicate option texts ──────────────
    log.info("2c. Data quality checks (empty / duplicate options) …")
    empty_options = 0
    dup_options = 0
    import random as _random
    _rng_q = _random.Random(42)
    sample_quality_idx = _rng_q.sample(train_ft_idx, min(10000, len(train_ft_idx)))
    for i in sample_quality_idx:
        ex = train_single[i]
        opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        empty_options += sum(1 for opt in opts if not str(opt).strip())
        if len(set(opts)) < 4:
            dup_options += 1
    n_checked = len(sample_quality_idx)
    log.info(f"   Checked {n_checked:,} examples:")
    log.info(f"   Empty option fields:     {empty_options}")
    log.info(f"   Duplicate option texts:  {dup_options}")
    if dup_options > 0:
        log.warning(f"   ⚠ {dup_options} examples have non-unique options — "
                    "model may not learn option discrimination.")
    else:
        log.info("   ✓ All options are non-empty and unique per question")
    report["data_quality"] = {
        "n_checked": n_checked,
        "empty_options": empty_options,
        "duplicate_options": dup_options,
    }

    # ── 3. Długość pytań w tokenach ──────────────────────────────────
    log.info("3. Długość pytań w tokenach (próbka 5 000) …")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3"
    )
    import random
    rng = random.Random(42)
    sample_idx = rng.sample(train_ft_idx, min(5000, len(train_ft_idx)))

    lengths = []
    for idx in sample_idx:
        ex = train_single[idx]
        # Use EXACT same format as 02_finetune.py _format_example —
        # including [INST] wrapper and answer letter, so token counts
        # reflect what the model actually sees during training.
        cop = ex["cop"]
        letter = "ABCD"[cop] if 0 <= cop < 4 else "A"
        prompt = (
            f"[INST] You are answering a medical multiple-choice question. "
            f"Reply with the single letter of the correct option only.\n\n"
            f"Question: {ex['question']}\n\n"
            f"A) {ex['opa']}\n"
            f"B) {ex['opb']}\n"
            f"C) {ex['opc']}\n"
            f"D) {ex['opd']} [/INST] Answer: {letter}"
        )
        toks = tokenizer.encode(prompt, add_special_tokens=False)
        lengths.append(len(toks))

    lengths_arr = sorted(lengths)
    p50  = np.percentile(lengths_arr, 50)
    p95  = np.percentile(lengths_arr, 95)
    p99  = np.percentile(lengths_arr, 99)
    pmax = max(lengths_arr)
    log.info(f"   p50={p50:.0f}  p95={p95:.0f}  p99={p99:.0f}  max={pmax}")
    log.info(f"   Próbek > 512 tokenów: {sum(l > 512 for l in lengths_arr)}/{len(lengths_arr)}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(lengths_arr, bins=60, color="#4C72B0", edgecolor="white",
            linewidth=0.5, alpha=0.85)
    for val, label, color in [
        (p50, "p50", "#2ca02c"),
        (p95, "p95", "#ff7f0e"),
        (512, "max_seq_len=512", "#d62728"),
    ]:
        ax.axvline(val, color=color, linestyle="--", linewidth=1.4, label=f"{label}={val:.0f}")
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Full prompt length — MedMCQA (n=5,000 sample)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, "prompt_length")
    plt.close(fig)

    report["prompt_length"] = {
        "n_sample": len(lengths_arr),
        "p50": round(float(p50), 1),
        "p95": round(float(p95), 1),
        "p99": round(float(p99), 1),
        "max": int(pmax),
        "pct_over_512": round(100 * sum(l > 512 for l in lengths_arr) / len(lengths_arr), 2),
    }

    # ── 4. Brakujące pola exp ────────────────────────────────────────
    log.info("4. Brakujące pola exp …")
    n_total = len(train_ft_idx)
    n_has_exp = sum(1 for i in train_ft_idx if train_single[i].get("exp"))
    pct_missing = 100 * (1 - n_has_exp / n_total)
    log.info(f"   exp obecne: {n_has_exp:,}/{n_total:,} ({100*n_has_exp/n_total:.1f}%)")
    log.info(f"   exp brakuje: {n_total - n_has_exp:,} ({pct_missing:.1f}%)")

    # Per-subject missing rate
    subject_exp = collections.defaultdict(lambda: [0, 0])  # [has_exp, total]
    for i in train_ft_idx:
        ex = train_single[i]
        subj = ex["subject_name"]
        subject_exp[subj][1] += 1
        if ex.get("exp"):
            subject_exp[subj][0] += 1

    subj_missing = {
        s: round(100 * (1 - v[0] / v[1]), 1)
        for s, v in subject_exp.items()
    }
    worst = sorted(subj_missing.items(), key=lambda x: -x[1])[:5]
    best  = sorted(subj_missing.items(), key=lambda x:  x[1])[:5]
    log.info(f"   Najwyższy % braku exp: {worst}")
    log.info(f"   Najniższy % braku exp: {best}")

    report["exp_missing"] = {
        "n_total": n_total,
        "n_has_exp": n_has_exp,
        "pct_missing": round(pct_missing, 1),
        "per_subject_missing_pct": subj_missing,
    }

    # ── 5. Stratification verification ──────────────────────────────
    # After splitting, train_ft and probe_set should have nearly identical
    # subject distributions (stratified split guarantees this quantitatively).
    log.info("5. Weryfikacja stratyfikacji (Jensen-Shannon divergence train_ft vs probe) …")
    from scipy.spatial.distance import jensenshannon
    all_subjects = sorted(subject_counts.keys())

    probe_subject_counts = collections.Counter(
        train_single[i]["subject_name"] for i in probe_idx
    )
    n_ft    = len(train_ft_idx)
    n_probe = len(probe_idx)

    p_ft    = np.array([subject_counts.get(s, 0) / n_ft    for s in all_subjects])
    p_probe = np.array([probe_subject_counts.get(s, 0) / n_probe for s in all_subjects])

    # NOTE: scipy.spatial.distance.jensenshannon() returns the Jensen-Shannon DISTANCE
    # = sqrt(JSD), NOT the Jensen-Shannon divergence JSD.
    # Reference: scipy docs — "This function computes the Jensen-Shannon distance."
    # Range: [0, 1] for sqrt(JSD); JSD itself would be in [0, log(2) ≈ 0.693] nats.
    # We compute sqrt(JSD) here and report it clearly to avoid ambiguity in the thesis.
    js_distance = float(jensenshannon(p_ft, p_probe))  # sqrt(JSD), range [0, 1]
    js_div = js_distance ** 2                           # true JSD, in nats

    log.info(f"   JS distance (√JSD, train_ft vs probe) = {js_distance:.5f}  "
             f"[JSD = {js_div:.5f} nats]  (0=identical)")
    if js_distance < 0.01:
        log.info("   ✓ Distributions virtually identical — stratification worked perfectly")
    elif js_distance < 0.05:
        log.info("   ✓ Distributions closely matched — stratification acceptable")
    else:
        log.warning(f"   ⚠ JS distance={js_distance:.4f} > 0.05 — "
                    "unexpected subject imbalance after split")

    # ── 5b. iso_cal vs routing_train subject balance ─────────────────
    # These are two halves of probe_set. Imbalance between them → calibration skew.
    if routing_train_idx and iso_cal_idx:
        log.info("5b. Stratification (routing_train vs iso_cal) …")
        routing_subject_counts = collections.Counter(
            train_single[i]["subject_name"] for i in routing_train_idx
        )
        iso_subject_counts = collections.Counter(
            train_single[i]["subject_name"] for i in iso_cal_idx
        )
        n_routing = len(routing_train_idx)
        n_iso = len(iso_cal_idx)

        p_routing = np.array([routing_subject_counts.get(s, 0) / n_routing
                              for s in all_subjects])
        p_iso = np.array([iso_subject_counts.get(s, 0) / n_iso
                         for s in all_subjects])

        js_routing_iso = float(jensenshannon(p_routing, p_iso))
        js_routing_iso_div = js_routing_iso ** 2
        log.info(f"   JS distance (√JSD, routing_train vs iso_cal) = {js_routing_iso:.5f}  "
                 f"[JSD = {js_routing_iso_div:.5f} nats]")
        if js_routing_iso < 0.02:
            log.info("   ✓ routing_train and iso_cal subject distributions are balanced")
        else:
            log.warning(f"   ⚠ JS distance={js_routing_iso:.4f} > 0.02 — "
                        "calibration may show subject-specific bias")
    else:
        js_routing_iso = None
        js_routing_iso_div = None
        log.info("   (routing_train/iso_cal indices not available — skipping balance check)")

    report["stratification_check"] = {
        "js_distance_ft_vs_probe": round(float(js_distance), 5),
        "js_divergence_ft_vs_probe_nats": round(float(js_div), 5),
        "js_distance_routing_vs_iso": round(float(js_routing_iso), 5) if js_routing_iso is not None else None,
        "note": (
            "js_distance = sqrt(JSD) as returned by scipy.spatial.distance.jensenshannon. "
            "Range: [0, 1]. JSD (nats) = js_distance^2. "
            "Thresholds: <0.01=identical, <0.05=acceptable, ≥0.05=concerning."
        ),
        "n_train_ft": n_ft,
        "n_probe": n_probe,
        "per_subject_ft_pct": {s: round(100 * subject_counts.get(s, 0) / n_ft, 2)
                               for s in all_subjects},
        "per_subject_probe_pct": {s: round(100 * probe_subject_counts.get(s, 0) / n_probe, 2)
                                  for s in all_subjects},
    }

    return report


# ─────────────────────────────────────────────────────────────────────
# 5. Cross-dataset overlap (contamination check)
# ─────────────────────────────────────────────────────────────────────

def run_overlap_check(train_single, train_ft_idx, probe_idx):
    import datasets
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    log.info("5. Cross-dataset contamination check …")

    # Zbiór pytań MedMCQA (train_ft + probe) do sprawdzenia
    all_medmcqa_idx = train_ft_idx + probe_idx
    medmcqa_questions = [
        normalize_q(train_single[i]["question"]) for i in all_medmcqa_idx
    ]
    medmcqa_hashes = set(medmcqa_questions)
    log.info(f"   MedMCQA (train_ft + probe): {len(medmcqa_questions):,} pytań")

    # ── MedQA-USMLE ─────────────────────────────────────────────────
    log.info("   Wczytuję MedQA-USMLE test …")
    medqa_ds = datasets.load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    medqa_q = [normalize_q(ex["question"]) for ex in medqa_ds]
    log.info(f"   MedQA test: {len(medqa_q):,} pytań")

    exact_medqa = [q for q in medqa_q if q in medmcqa_hashes]
    log.info(f"   Exact matches MedMCQA ↔ MedQA: {len(exact_medqa)}")

    # ── MMLU (5 kategorii) ───────────────────────────────────────────
    log.info("   Wczytuję MMLU (5 kategorii) test …")
    mmlu_cats = [
        "clinical_knowledge", "professional_medicine",
        "college_medicine", "medical_genetics", "anatomy",
    ]
    mmlu_q = []
    for cat in mmlu_cats:
        ds_cat = datasets.load_dataset("cais/mmlu", cat, split="test")
        mmlu_q.extend(normalize_q(ex["question"]) for ex in ds_cat)
    log.info(f"   MMLU test (5 kategorii): {len(mmlu_q):,} pytań")

    exact_mmlu = [q for q in mmlu_q if q in medmcqa_hashes]
    log.info(f"   Exact matches MedMCQA ↔ MMLU: {len(exact_mmlu)}")

    # ── Near-duplicate (TF-IDF cosine similarity) ────────────────────
    log.info("   Near-duplicate check (TF-IDF cosine, próbka 10 000 MedMCQA) …")

    # Dla szybkości: 10K próbek MedMCQA vs wszystkie pytania OOD
    import random
    rng = random.Random(42)
    sample_medmcqa = rng.sample(medmcqa_questions, min(10_000, len(medmcqa_questions)))

    ood_all = medqa_q + mmlu_q
    ood_labels = (
        ["medqa"] * len(medqa_q) +
        ["mmlu"]  * len(mmlu_q)
    )

    corpus = sample_medmcqa + ood_all
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000)
    tfidf = vectorizer.fit_transform(corpus)

    medmcqa_vecs = tfidf[:len(sample_medmcqa)]
    ood_vecs     = tfidf[len(sample_medmcqa):]

    # Batch cosine similarity (MedMCQA vs OOD) — matmul w blokach
    THRESHOLD = 0.85
    near_dup_pairs = []
    BATCH = 500
    for start in range(0, len(ood_all), BATCH):
        end = min(start + BATCH, len(ood_all))
        sim_block = cosine_similarity(ood_vecs[start:end], medmcqa_vecs)
        for ood_local_i, row in enumerate(sim_block):
            best_score = float(row.max())
            if best_score >= THRESHOLD:
                best_medmcqa_j = int(row.argmax())
                near_dup_pairs.append({
                    "ood_source": ood_labels[start + ood_local_i],
                    "ood_question": ood_all[start + ood_local_i][:100],
                    "medmcqa_question": sample_medmcqa[best_medmcqa_j][:100],
                    "cosine_sim": round(best_score, 4),
                })

    near_dup_pairs.sort(key=lambda x: -x["cosine_sim"])
    log.info(f"   Near-duplicates (cosine ≥ {THRESHOLD}): {len(near_dup_pairs)}")
    if near_dup_pairs:
        log.info("   Top 5 near-duplicates:")
        for p in near_dup_pairs[:5]:
            log.info(f"     [{p['ood_source']}] sim={p['cosine_sim']:.3f}")
            log.info(f"       OOD:     {p['ood_question']}")
            log.info(f"       MedMCQA: {p['medmcqa_question']}")
    else:
        log.info("   ✓ Brak near-duplicates — OOD testy są czyste")

    return {
        "medmcqa_n": len(medmcqa_questions),
        "medqa_n": len(medqa_q),
        "mmlu_n": len(mmlu_q),
        "exact_matches_medqa": len(exact_medqa),
        "exact_matches_mmlu": len(exact_mmlu),
        "near_dup_threshold": THRESHOLD,
        "near_dup_sample_size": len(sample_medmcqa),
        "near_dup_count": len(near_dup_pairs),
        "near_dup_top10": near_dup_pairs[:10],
    }


# ─────────────────────────────────────────────────────────────────────
# 6. Near-duplicate check: train_ft vs probe_set
# ─────────────────────────────────────────────────────────────────────

def run_internal_dup_check(train_single, train_ft_idx, probe_idx):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    log.info("6. Near-duplicate check: train_ft vs probe_set …")

    # Próbka train_ft (10K) vs wszystkie probe (16K)
    import random
    rng = random.Random(42)
    sample_ft = rng.sample(train_ft_idx, min(10_000, len(train_ft_idx)))

    ft_q     = [normalize_q(train_single[i]["question"]) for i in sample_ft]
    probe_q  = [normalize_q(train_single[i]["question"]) for i in probe_idx]

    # Exact duplicates
    probe_hash = set(probe_q)
    exact_dups = [q for q in ft_q if q in probe_hash]
    log.info(f"   Exact duplicates (train_ft ∩ probe): {len(exact_dups)}")

    # Near-duplicates
    corpus = ft_q + probe_q
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000)
    tfidf = vectorizer.fit_transform(corpus)

    ft_vecs    = tfidf[:len(ft_q)]
    probe_vecs = tfidf[len(ft_q):]

    THRESHOLD = 0.90  # Wyższy próg dla wewnętrznego sprawdzenia
    near_dup_pairs = []
    BATCH = 500
    for start in range(0, len(probe_q), BATCH):
        end = min(start + BATCH, len(probe_q))
        sim_block = cosine_similarity(probe_vecs[start:end], ft_vecs)
        for probe_local_i, row in enumerate(sim_block):
            best_score = float(row.max())
            if best_score >= THRESHOLD:
                best_ft_j = int(row.argmax())
                near_dup_pairs.append({
                    "probe_question": probe_q[start + probe_local_i][:100],
                    "ft_question": ft_q[best_ft_j][:100],
                    "cosine_sim": round(best_score, 4),
                })

    near_dup_pairs.sort(key=lambda x: -x["cosine_sim"])
    log.info(f"   Near-duplicates train_ft ↔ probe (cosine ≥ {THRESHOLD}): "
             f"{len(near_dup_pairs)}")

    if near_dup_pairs:
        log.info("   Top 5:")
        for p in near_dup_pairs[:5]:
            log.info(f"     sim={p['cosine_sim']:.3f}")
            log.info(f"       train_ft: {p['ft_question']}")
            log.info(f"       probe:    {p['probe_question']}")
    else:
        log.info("   ✓ Brak near-duplicates — train_ft i probe_set są izolowane")

    return {
        "ft_sample_size": len(ft_q),
        "probe_size": len(probe_q),
        "exact_dups": len(exact_dups),
        "near_dup_threshold": THRESHOLD,
        "near_dup_count": len(near_dup_pairs),
        "near_dup_top10": near_dup_pairs[:10],
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    try:
        import datasets
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np
    except ImportError as e:
        print(f"ERROR: brakuje biblioteki — {e}")
        sys.exit(1)

    # Sprawdź czy splity istnieją
    if not (SPLITS_DIR / "train_ft_idx.json").exists():
        print("ERROR: Najpierw uruchom scripts/01_prepare.py")
        sys.exit(1)

    log.info("Wczytuję indeksy splitów …")
    train_ft_idx  = json.loads((SPLITS_DIR / "train_ft_idx.json").read_text())
    probe_idx     = json.loads((SPLITS_DIR / "probe_idx.json").read_text())
    log.info(f"  train_ft: {len(train_ft_idx):,}  probe: {len(probe_idx):,}")

    # Load routing/iso indices for balance check (optional — may not exist yet)
    routing_train_idx, iso_cal_idx = [], []
    _routing_path = SPLITS_DIR / "routing_train_idx.json"
    _iso_path     = SPLITS_DIR / "iso_cal_idx.json"
    if _routing_path.exists() and _iso_path.exists():
        routing_train_idx = json.loads(_routing_path.read_text())
        iso_cal_idx       = json.loads(_iso_path.read_text())
        log.info(f"  routing_train: {len(routing_train_idx):,}  iso_cal: {len(iso_cal_idx):,}")
    else:
        log.info("  routing_train_idx.json / iso_cal_idx.json not found — "
                 "skipping iso_cal balance check")

    # Integrity checks — catch silent data corruption early
    assert len(train_ft_idx) > 0, "train_ft_idx is empty"
    assert len(probe_idx) > 0, "probe_idx is empty"
    assert len(set(train_ft_idx) & set(probe_idx)) == 0, \
        "CRITICAL: train_ft and probe overlap!"
    log.info("  ✓ Data integrity checks passed (train_ft ∩ probe = ∅)")

    log.info("Wczytuję MedMCQA train (single) …")
    train_raw    = datasets.load_dataset("openlifescienceai/medmcqa", split="train")
    train_single = train_raw.filter(
        lambda x: x["choice_type"] == "single", num_proc=4
    )

    report = {}

    # ── Sekcje 1–4 ──────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SEKCJA 1–4: MedMCQA internal EDA")
    log.info("=" * 60)
    report["medmcqa_eda"] = run_medmcqa_eda(
        train_single, train_ft_idx, probe_idx,
        routing_train_idx=routing_train_idx,
        iso_cal_idx=iso_cal_idx,
    )

    # ── Sekcja 5 ────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SEKCJA 5: Cross-dataset contamination check")
    log.info("=" * 60)
    report["overlap_check"] = run_overlap_check(train_single, train_ft_idx, probe_idx)

    # ── Sekcja 6 ────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("SEKCJA 6: Internal near-duplicate check (train_ft vs probe)")
    log.info("=" * 60)
    report["internal_dup_check"] = run_internal_dup_check(
        train_single, train_ft_idx, probe_idx
    )

    # ── Zapis raportu ────────────────────────────────────────────────
    out_path = RESULTS_DIR / "eda_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"\nRaport zapisany: {out_path}")

    # ── Podsumowanie ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)

    # Answer distribution
    ans = report["medmcqa_eda"]["answer_distribution"]
    print(f"\nAnswer distribution (train_ft):")
    for k, pct in ans["pct"].items():
        bar = "█" * int(pct / 2)
        print(f"  {k}: {bar:<15} {pct:.1f}%")
    print(f"  Max deviation from uniform: ±{ans['max_deviation_pp']:.1f}pp")
    uniform_str = "✓ uniform" if ans["is_uniform_p01"] else "⚠ NON-UNIFORM — position bias possible"
    print(f"  χ²={ans['chi2_stat']:.1f}  p={ans['chi2_p']:.2e}  → {uniform_str}")

    # Stratification
    sc = report["medmcqa_eda"].get("stratification_check", {})
    if sc:
        print(f"\nStratification (train_ft vs probe):")
        js_dist = sc.get("js_distance_ft_vs_probe", 0.0)
        js_div = sc.get("js_divergence_ft_vs_probe_nats", js_dist ** 2)
        quality = "perfect" if js_dist < 0.01 else ("acceptable" if js_dist < 0.05 else "⚠ POOR")
        print(f"  JS distance (√JSD) = {js_dist:.5f}  JSD = {js_div:.5f} nats  ({quality})")

    # Prompt length
    pl = report["medmcqa_eda"]["prompt_length"]
    print(f"\nPrompt length (n={pl['n_sample']:,}):")
    print(f"  p50={pl['p50']:.0f}  p95={pl['p95']:.0f}  p99={pl['p99']:.0f}  "
          f"max={pl['max']}  over_512={pl['pct_over_512']:.1f}%")

    # Overlap
    ov = report["overlap_check"]
    print(f"\nCross-dataset contamination:")
    print(f"  MedMCQA ↔ MedQA exact matches: {ov['exact_matches_medqa']}")
    print(f"  MedMCQA ↔ MMLU  exact matches: {ov['exact_matches_mmlu']}")
    print(f"  Near-duplicates (cos≥{ov['near_dup_threshold']}): "
          f"{ov['near_dup_count']}")

    # Internal dup
    id_ = report["internal_dup_check"]
    print(f"\nInternal train_ft ↔ probe_set:")
    print(f"  Exact duplicates: {id_['exact_dups']}")
    print(f"  Near-duplicates (cos≥{id_['near_dup_threshold']}): "
          f"{id_['near_dup_count']}")

    print(f"\nFigury: {FIGURES_DIR}/eda_*.pdf")
    print("=" * 60)


if __name__ == "__main__":
    main()
