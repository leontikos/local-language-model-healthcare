"""
KROK 5 — CORRECTNESS PROBE
==========================
Train a LogisticRegression "correctness probe" on Mistral-7B hidden states (from
the layer chosen by KROK 4's sweep) to predict whether the model's MCQ answer is
correct.

Two probes are produced:
  1. 5-fold cross-fitted out-of-fold scores `probe_scores_oof` over the full
     probe_set (13,307). These are used by 05_routing for routing-LR training
     so the routing model sees probe scores that were NOT trained on the same
     example — preventing leakage.
  2. A "final" probe trained on the full 13,307 — used at evaluation time on
     val / test_medqa / test_mmlu (where no leakage concern applies).

Usage:
    python scripts/04_probe.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, brier_score_loss

    features_dir = ROOT / "data" / "features"
    results_dir = ROOT / "results"
    ckpt_dir = ROOT / "checkpoints"
    results_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    log.info("Loading probe features …")
    d = np.load(features_dir / "probe_features.npz", allow_pickle=False)
    H_hidden = np.load(features_dir / "probe_hidden.npy")
    y = d["y"].astype(np.int32)
    n = len(y)
    if H_hidden.shape[0] != n:
        log.error("hidden (%d) ≠ y (%d) — re-run 03_extract", H_hidden.shape[0], n)
        sys.exit(1)
    log.info("  probe_set: %d examples  hidden_dim=%d  acc(base)=%.3f",
             n, H_hidden.shape[1], y.mean())

    # ------------------------------------------------------------------
    # 5-fold cross-fitting
    # ------------------------------------------------------------------
    Cs = [0.001, 0.01, 0.1, 1.0, 10.0]
    n_folds = 5
    probe_scores_oof = np.zeros(n, dtype=np.float64)
    fold_aurocs: list[float] = []

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_idx, (tr, va) in enumerate(kf.split(H_hidden, y), start=1):
        log.info("  Fold %d/%d: fit LogisticRegressionCV on %d, score %d …",
                 fold_idx, n_folds, len(tr), len(va))
        probe = LogisticRegressionCV(
            Cs=Cs, cv=3, penalty="l2", scoring="roc_auc",
            max_iter=2000, n_jobs=-1, random_state=42,
        )
        probe.fit(H_hidden[tr], y[tr])
        probe_scores_oof[va] = probe.predict_proba(H_hidden[va])[:, 1]
        auc = roc_auc_score(y[va], probe_scores_oof[va])
        fold_aurocs.append(float(auc))
        log.info("    fold AUROC = %.4f  best C = %s", auc, probe.C_)

    mean_oof_auroc = float(np.mean(fold_aurocs))
    pooled_oof_auroc = float(roc_auc_score(y, probe_scores_oof))
    oof_brier = float(brier_score_loss(y, probe_scores_oof))
    log.info("  Mean fold AUROC: %.4f ± %.4f", mean_oof_auroc, float(np.std(fold_aurocs)))
    log.info("  Pooled OOF AUROC: %.4f   OOF Brier: %.4f",
             pooled_oof_auroc, oof_brier)

    np.save(features_dir / "probe_scores_oof.npy", probe_scores_oof.astype(np.float32))
    log.info("  Saved: %s", features_dir / "probe_scores_oof.npy")

    # ------------------------------------------------------------------
    # Final probe on full probe_set — used at evaluation time on val/test
    # ------------------------------------------------------------------
    log.info("Training final probe on all %d probe_set examples …", n)
    final_probe = LogisticRegressionCV(
        Cs=Cs, cv=3, penalty="l2", scoring="roc_auc",
        max_iter=2000, n_jobs=-1, random_state=42,
    )
    final_probe.fit(H_hidden, y)
    log.info("  Final probe best C = %s", final_probe.C_)

    with open(ckpt_dir / "final_probe.pkl", "wb") as f:
        pickle.dump(final_probe, f)
    log.info("  Saved: %s", ckpt_dir / "final_probe.pkl")

    # ------------------------------------------------------------------
    # Persist metrics
    # ------------------------------------------------------------------
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_probe": int(n),
        "hidden_dim": int(H_hidden.shape[1]),
        "Cs": Cs,
        "n_folds": n_folds,
        "fold_aurocs": [round(a, 5) for a in fold_aurocs],
        "mean_oof_auroc": round(mean_oof_auroc, 5),
        "pooled_oof_auroc": round(pooled_oof_auroc, 5),
        "oof_brier": round(oof_brier, 5),
        "best_C_final_probe": float(final_probe.C_[0]),
    }
    (results_dir / "probe_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("  Saved: %s", results_dir / "probe_metrics.json")

    print("\n" + "=" * 60)
    print("KROK 5 DONE")
    print(f"  Pooled OOF AUROC : {pooled_oof_auroc:.4f}")
    print(f"  Mean fold AUROC  : {mean_oof_auroc:.4f}")
    print(f"  OOF Brier        : {oof_brier:.4f}")
    print(f"  Saved            : {features_dir / 'probe_scores_oof.npy'}")
    print(f"                     {ckpt_dir / 'final_probe.pkl'}")
    print(f"                     {results_dir / 'probe_metrics.json'}")
    print(f"  Next             : python scripts/05_routing.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
