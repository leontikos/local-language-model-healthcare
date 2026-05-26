"""
KROK 6 — ROUTING POLICY (Logistic Regression + isotonic calibration)
=====================================================================
Train the routing model that maps uncertainty signals -> P(model is correct).

Re-partition of probe_set (because MedMCQA test has cop=-1 — see note in
README — we use MedMCQA val as in-dist test, and steal 3,307 examples from
probe_set for conformal calibration so val stays untouched until eval):
    routing_train  : 8,000  → routing LR fit
    iso_cal        : 2,000  → isotonic calibrator fit
    conformal_cal  : 3,307  → reserved for 06_conformal.py (q_hat)
These three subsets of probe_set (n=13,307) are disjoint, drawn with seed=42.

Routing features: [H, gap*, probe, p_true]
where gap is dropped if abs(corr(H, gap)) > 0.85 (roadmap §5).

Usage:
    python scripts/05_routing.py
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

CORR_THRESHOLD = 0.85
ROUTING_TRAIN_N = 8000
ISO_CAL_N = 2000
# conformal_cal_n = 13307 - routing_train_n - iso_cal_n = 3307

SEED = 42


def _build_feature_matrix(rows: list[int],
                          names: list[str],
                          H, gap, probe, p_true) -> "np.ndarray":
    import numpy as np
    cols = []
    for name in names:
        if name == "H":      cols.append(H[rows])
        elif name == "gap":  cols.append(gap[rows])
        elif name == "probe": cols.append(probe[rows])
        elif name == "p_true": cols.append(p_true[rows])
        else: raise ValueError(f"Unknown feature: {name}")
    return np.column_stack(cols).astype(np.float64)


def main() -> None:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.model_selection import train_test_split

    features_dir = ROOT / "data" / "features"
    splits_dir = ROOT / "data" / "splits"
    results_dir = ROOT / "results"
    ckpt_dir = ROOT / "checkpoints"

    log.info("Loading probe features + OOF probe scores …")
    d = np.load(features_dir / "probe_features.npz", allow_pickle=False)
    probe_scores_oof = np.load(features_dir / "probe_scores_oof.npy")
    H = d["H"].astype(np.float64)
    gap = d["gap"].astype(np.float64)
    p_true = d["p_true"].astype(np.float64)
    y = d["y"].astype(np.int32)
    subject = d["subject"]
    n = len(y)
    if not (len(probe_scores_oof) == n == len(H)):
        log.error("Length mismatch — re-run 04_probe.py")
        sys.exit(1)
    log.info("  probe_set: %d examples", n)

    # ------------------------------------------------------------------
    # 1. Re-partition probe_set into 3 disjoint subsets
    # ------------------------------------------------------------------
    all_local = np.arange(n)
    routing_train, rest = train_test_split(
        all_local, train_size=ROUTING_TRAIN_N,
        stratify=y, random_state=SEED,
    )
    y_rest = y[rest]
    iso_cal, conformal_cal = train_test_split(
        rest, train_size=ISO_CAL_N,
        stratify=y_rest, random_state=SEED,
    )

    # Disjointness sanity check
    assert set(routing_train).isdisjoint(set(iso_cal))
    assert set(routing_train).isdisjoint(set(conformal_cal))
    assert set(iso_cal).isdisjoint(set(conformal_cal))
    assert len(routing_train) + len(iso_cal) + len(conformal_cal) == n

    log.info(
        "  routing_train: %d  iso_cal: %d  conformal_cal: %d",
        len(routing_train), len(iso_cal), len(conformal_cal),
    )

    splits_dir.mkdir(exist_ok=True)
    (splits_dir / "routing_train_local_idx.json").write_text(
        json.dumps(sorted(routing_train.tolist()))
    )
    (splits_dir / "iso_cal_local_idx.json").write_text(
        json.dumps(sorted(iso_cal.tolist()))
    )
    (splits_dir / "conformal_cal_local_idx.json").write_text(
        json.dumps(sorted(conformal_cal.tolist()))
    )
    log.info("  Saved 3 *_local_idx.json under %s", splits_dir)

    # ------------------------------------------------------------------
    # 2. Correlation analysis → decide whether to keep `gap`
    # ------------------------------------------------------------------
    feats_df = pd.DataFrame({
        "H": H, "gap": gap, "probe": probe_scores_oof, "p_true": p_true,
    })
    corr = feats_df.corr(method="pearson")
    log.info("\nPearson correlation matrix:\n%s", corr.round(3).to_string())
    r_H_gap = abs(float(corr.loc["H", "gap"]))
    if r_H_gap > CORR_THRESHOLD:
        routing_feature_names = ["H", "probe", "p_true"]
        log.info("  r(H, gap) = %.3f > %.2f → DROP gap", r_H_gap, CORR_THRESHOLD)
    else:
        routing_feature_names = ["H", "gap", "probe", "p_true"]
        log.info("  r(H, gap) = %.3f ≤ %.2f → KEEP gap", r_H_gap, CORR_THRESHOLD)

    # ------------------------------------------------------------------
    # 3. Train routing LR
    # ------------------------------------------------------------------
    X_train = _build_feature_matrix(
        routing_train, routing_feature_names, H, gap, probe_scores_oof, p_true,
    )
    y_train = y[routing_train]
    log.info("Training routing LR on %d examples × %d features …",
             *X_train.shape)
    routing_lr = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0], cv=3, penalty="l2", scoring="roc_auc",
        max_iter=2000, n_jobs=-1, random_state=SEED,
    )
    routing_lr.fit(X_train, y_train)
    log.info("  Best C = %s", routing_lr.C_)
    log.info("  Learned weights:")
    for name, w in zip(routing_feature_names, routing_lr.coef_[0]):
        log.info("    %-8s : %+.4f", name, w)
    log.info("    %-8s : %+.4f", "intercept", routing_lr.intercept_[0])

    train_auroc = roc_auc_score(y_train, routing_lr.predict_proba(X_train)[:, 1])
    log.info("  Routing-LR train AUROC: %.4f", train_auroc)

    # ------------------------------------------------------------------
    # 4. Isotonic calibration on iso_cal (uncalibrated routing probs first)
    # ------------------------------------------------------------------
    X_iso = _build_feature_matrix(
        iso_cal, routing_feature_names, H, gap, probe_scores_oof, p_true,
    )
    y_iso = y[iso_cal]
    proba_iso_raw = routing_lr.predict_proba(X_iso)[:, 1]

    iso_auroc_raw = roc_auc_score(y_iso, proba_iso_raw)
    iso_brier_raw = brier_score_loss(y_iso, proba_iso_raw)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(proba_iso_raw, y_iso)
    proba_iso_cal = calibrator.transform(proba_iso_raw)
    iso_brier_cal = brier_score_loss(y_iso, proba_iso_cal)
    log.info(
        "  iso_cal: AUROC=%.4f  Brier(raw)=%.4f  Brier(isotonic)=%.4f",
        iso_auroc_raw, iso_brier_raw, iso_brier_cal,
    )

    # ------------------------------------------------------------------
    # 5. Persist artifacts
    # ------------------------------------------------------------------
    artifacts = {
        "routing_lr.pkl": routing_lr,
        "calibrator.pkl": calibrator,
    }
    for fname, obj in artifacts.items():
        with open(ckpt_dir / fname, "wb") as f:
            pickle.dump(obj, f)
        log.info("  Saved: %s", ckpt_dir / fname)

    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": SEED,
        "splits": {
            "routing_train": len(routing_train),
            "iso_cal": len(iso_cal),
            "conformal_cal": len(conformal_cal),
        },
        "correlation_matrix": corr.round(4).to_dict(),
        "r_H_gap_abs": round(r_H_gap, 4),
        "routing_feature_names": routing_feature_names,
        "routing_lr_best_C": float(routing_lr.C_[0]),
        "routing_lr_weights": dict(zip(
            routing_feature_names,
            [round(float(w), 5) for w in routing_lr.coef_[0]],
        )),
        "routing_lr_intercept": round(float(routing_lr.intercept_[0]), 5),
        "routing_lr_train_auroc": round(float(train_auroc), 5),
        "iso_cal_auroc": round(float(iso_auroc_raw), 5),
        "iso_brier_raw": round(float(iso_brier_raw), 5),
        "iso_brier_calibrated": round(float(iso_brier_cal), 5),
    }
    (results_dir / "routing_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("  Saved: %s", results_dir / "routing_metrics.json")

    print("\n" + "=" * 60)
    print("KROK 6 DONE")
    print(f"  Routing features         : {routing_feature_names}")
    print(f"  Routing-LR train AUROC   : {train_auroc:.4f}")
    print(f"  iso_cal AUROC            : {iso_auroc_raw:.4f}")
    print(f"  Brier raw → calibrated   : {iso_brier_raw:.4f} → {iso_brier_cal:.4f}")
    print(f"  conformal_cal reserved   : {len(conformal_cal)} examples → next step")
    print(f"  Next                     : python scripts/06_conformal.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
