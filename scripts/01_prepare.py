"""
KROK 2 — DATA PREPARATION
==========================
Tworzy deterministyczne splity train_ft / probe_set / (routing_train + iso_cal)
i zapisuje indeksy do data/splits/ jako JSON.

Użycie:
    python scripts/01_prepare.py
    python scripts/01_prepare.py --dry-run   # tylko statystyki, bez zapisu
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ładuj HF_TOKEN z .env jeśli istnieje
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


def main(dry_run: bool) -> None:
    try:
        import datasets
        import numpy as np
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"ERROR: brakuje biblioteki — {e}")
        print("Uruchom: pip install -e '.[dev]'")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Wczytaj MedMCQA train
    # ------------------------------------------------------------------
    log.info("Wczytuję openlifescienceai/medmcqa split=train …")
    train_raw = datasets.load_dataset(
        "openlifescienceai/medmcqa", split="train"
    )
    log.info(f"  Surowy train: {len(train_raw):,} przykładów")

    # ------------------------------------------------------------------
    # 2. Filtruj choice_type == "single"
    # ------------------------------------------------------------------
    log.info("Filtrowanie choice_type='single' …")
    train_single = train_raw.filter(
        lambda x: x["choice_type"] == "single",
        num_proc=4,
    )
    log.info(
        f"  Po filtrze 'single': {len(train_single):,} "
        f"({100 * len(train_single) / len(train_raw):.1f}%)"
    )

    # ------------------------------------------------------------------
    # 3. Stratified split: probe_set (16 000) vs train_ft (reszta)
    #    Stratyfikacja po subject_name
    # ------------------------------------------------------------------
    log.info("Stratified split → probe_set (16 000) + train_ft (reszta) …")
    subjects = [ex["subject_name"] for ex in train_single]
    all_idx = list(range(len(train_single)))

    train_ft_idx, probe_idx = train_test_split(
        all_idx,
        test_size=16_000,
        stratify=subjects,
        random_state=42,
    )
    log.info(f"  train_ft:  {len(train_ft_idx):,}")
    log.info(f"  probe_set: {len(probe_idx):,}")

    # ------------------------------------------------------------------
    # 4. Deduplikacja probe_set względem train_ft (TF-IDF cosine ≥ 0.90)
    #    Uzasadnienie: duplikaty w probe powodują data leakage — model widział
    #    te pytania podczas FT, więc probe AUROC jest inflated dla tych przypadków.
    #    Próg 0.95: przy krótkich MCQ (p50=53 tok) dopiero ≥0.95 oznacza faktycznie
    #    to samo pytanie; 0.90 zbyt agresywne (łapie różne pytania z MCQ template).
    #    train_ft zostaje nienaruszony; usuwamy tylko skażone próbki z probe.
    # ------------------------------------------------------------------
    log.info("Deduplikacja probe_set względem train_ft (exact match po normalizacji) …")
    try:
        import re

        def _normalize(text: str) -> str:
            text = text.lower().strip()
            text = re.sub(r"[^\w\s]", " ", text)
            return re.sub(r"\s+", " ", text)

        # Uzasadnienie: używamy WYŁĄCZNIE exact match, nie TF-IDF cosine.
        # TF-IDF przy krótkich MCQ (p50=53 tok) łapie template similarity
        # ("which of the following is...") dla merytorycznie różnych pytań —
        # fałszywe duplikaty które mają całkowicie inne hidden states.
        # Tylko exact match = jednoznacznie te samo pytanie = jednoznaczny leakage.
        ft_hash = {_normalize(train_single[i]["question"]) for i in train_ft_idx}

        contaminated = {
            pos for pos, idx in enumerate(probe_idx)
            if _normalize(train_single[idx]["question"]) in ft_hash
        }

        probe_idx_clean = [
            idx for pos, idx in enumerate(probe_idx) if pos not in contaminated
        ]
        log.info(f"  Exact duplicates usunięte z probe_set: {len(contaminated):,}")
        log.info(f"  probe_set po dedup: {len(probe_idx_clean):,}")
        probe_idx = probe_idx_clean

    except Exception as e:
        log.warning(f"  Deduplikacja nieudana ({e}) — kontynuuję bez dedup")

    # ------------------------------------------------------------------
    # 5. Podziel probe_set → routing_train (12 000) + iso_cal (4 000)
    # ------------------------------------------------------------------
    log.info("Split probe_set → routing_train (12 000) + iso_cal (4 000) …")
    probe_local_idx = list(range(len(probe_idx)))
    # Stratify by subject_name to keep subject distribution balanced across both
    # routing_train and iso_cal — prevents calibration skew from subject imbalance.
    probe_subjects = [subjects[i] for i in probe_idx]

    routing_local_idx, iso_local_idx = train_test_split(
        probe_local_idx,
        test_size=4_000,
        stratify=probe_subjects,
        random_state=42,
    )
    routing_train_idx = [probe_idx[i] for i in routing_local_idx]
    iso_cal_idx = [probe_idx[i] for i in iso_local_idx]

    log.info(f"  routing_train: {len(routing_train_idx):,}")
    log.info(f"  iso_cal:       {len(iso_cal_idx):,}")

    # ------------------------------------------------------------------
    # 6. Weryfikacja braku przecięcia
    # ------------------------------------------------------------------
    log.info("Weryfikacja izolacji zbiorów …")
    ft_set = set(train_ft_idx)
    probe_set_ = set(probe_idx)
    routing_set = set(routing_train_idx)
    iso_set = set(iso_cal_idx)

    assert ft_set & probe_set_ == set(), "BŁĄD: train_ft i probe_set nakładają się!"
    assert routing_set & iso_set == set(), "BŁĄD: routing_train i iso_cal nakładają się!"
    assert routing_set | iso_set == probe_set_, "BŁĄD: routing+iso ≠ probe_set!"
    log.info("  ✓ Brak przecięcia między zbiorami")

    # ------------------------------------------------------------------
    # 6. Statystyki subject_name (sprawdź stratyfikację)
    # ------------------------------------------------------------------
    import collections

    def subject_dist(indices):
        counts = collections.Counter(subjects[i] for i in indices)
        return {k: v for k, v in sorted(counts.items())}

    log.info("Rozkład subject_name (top 5 kategorii w probe_set):")
    probe_subj = collections.Counter(subjects[i] for i in probe_idx)
    for subj, cnt in probe_subj.most_common(5):
        pct = 100 * cnt / len(probe_idx)
        log.info(f"    {subj[:35]:<35} {cnt:5d} ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # 7. Zapis do data/splits/
    # ------------------------------------------------------------------
    splits = {
        "train_ft_idx":       train_ft_idx,
        "probe_idx":          probe_idx,
        "routing_train_idx":  routing_train_idx,
        "iso_cal_idx":        iso_cal_idx,
    }
    meta = {
        "seed": 42,
        "filter": "choice_type == 'single'",
        "dedup_method": "exact_match_normalized",
        "n_train_raw":    len(train_raw),
        "n_train_single": len(train_single),
        "n_train_ft":     len(train_ft_idx),
        "n_probe":        len(probe_idx),
        "n_routing_train": len(routing_train_idx),
        "n_iso_cal":      len(iso_cal_idx),
    }

    if dry_run:
        log.info("\n--- DRY RUN — nic nie zostało zapisane ---")
        log.info(json.dumps(meta, indent=2))
        return

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for name, idx_list in splits.items():
        out_path = SPLITS_DIR / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(idx_list, f, separators=(",", ":"))
        log.info(f"  Zapisano: {out_path}  ({len(idx_list):,} indeksów)")

    meta_path = SPLITS_DIR / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Zapisano: {meta_path}")

    # ------------------------------------------------------------------
    # 8. Sanity check: wczytaj i porównaj wszystkie 4 pliki
    # ------------------------------------------------------------------
    log.info("Sanity check — odczyt zapisanych indeksów …")
    for name, original in splits.items():
        loaded = json.loads((SPLITS_DIR / f"{name}.json").read_text())
        assert loaded == original, f"BŁĄD: {name}.json różni się po zapisie!"
    log.info("  ✓ Wszystkie 4 pliki indeksów zapisane i odczytane poprawnie")

    # ------------------------------------------------------------------
    # 9. Ręczna próba 10 przykładów z train_ft (weryfikacja cop encoding)
    # ------------------------------------------------------------------
    log.info("\nPróbka 10 przykładów z train_ft (weryfikacja cop encoding):")
    cop_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    import random
    rng = random.Random(42)
    sample_ft_idx = rng.sample(train_ft_idx, 10)
    for i, idx in enumerate(sample_ft_idx):
        ex = train_single[idx]
        cop = ex["cop"]
        letter = cop_map.get(cop, "?")
        opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        q_short = ex["question"][:55]
        ans_text = opts[cop][:40] if cop < len(opts) else "???"
        log.info(f"  [{i+1:02d}] Q: {q_short}")
        log.info(f"        cop={cop} → {letter} = '{ans_text}'")

    print("\n" + "=" * 60)
    print("✓ KROK 2 DONE — splity zapisane do data/splits/")
    print(f"  train_ft:      {len(train_ft_idx):,}  (nienaruszony)")
    print(f"  probe_set:     {len(probe_idx):,}  (po usunięciu exact duplicates)")
    print(f"  routing_train: {len(routing_train_idx):,}")
    print(f"  iso_cal:       {len(iso_cal_idx):,}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation — MedMCQA splits")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tylko statystyki, bez zapisu plików",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
