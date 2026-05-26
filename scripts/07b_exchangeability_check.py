"""
KROK 8b — CP EXCHANGEABILITY DIAGNOSTIC
========================================
The main 3-tier evaluation showed in-dist local_rate=0.79 < CP target 0.90,
suggesting the conformal cal set (drawn from MedMCQA train via probe_set) and
the val set (used as in-dist test) are not exchangeable.

To confirm exchangeability — and not the CP method — is the culprit, re-calibrate
CP on a held-out chunk of val itself (drawn from the same distribution as the
test points). If the CP guarantee is then restored, the diagnosis is conclusive.

We run K-fold (K=5): in each fold, calibrate q_hat on 4/5 of val and test on 1/5,
aggregate. Compare against the original train→val setup.

Usage:
    python scripts/07b_exchangeability_check.py
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

ALPHA = 0.10
N_FOLDS = 5
N_BOOT = 1000
SEED = 42


def _build_feature_matrix(names, H, gap, probe, p_true):
    import numpy as np
    cols = []
    for name in names:
        if name == "H":      cols.append(H)
        elif name == "gap":  cols.append(gap)
        elif name == "probe": cols.append(probe)
        elif name == "p_true": cols.append(p_true)
        else: raise ValueError(name)
    return np.column_stack(cols).astype(np.float64)


def _compute_q_hat(routing_scores, alpha):
    import numpy as np
    n = len(routing_scores)
    s_cal = 1.0 - routing_scores
    return float(np.quantile(
        s_cal,
        np.ceil((1 - alpha) * (n + 1)) / n,
        method="higher",
    ))


def main() -> None:
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    features_dir = ROOT / "data" / "features"
    results_dir  = ROOT / "results"
    ckpt_dir     = ROOT / "checkpoints"

    # ------------------------------------------------------------------
    # Load all routing artefacts + the original (probe-based) q_hat
    # ------------------------------------------------------------------
    with open(ckpt_dir / "final_probe.pkl", "rb") as f:
        final_probe = pickle.load(f)
    with open(ckpt_dir / "routing_lr.pkl", "rb") as f:
        routing_lr = pickle.load(f)
    with open(ckpt_dir / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    routing_metrics = json.loads((results_dir / "routing_metrics.json").read_text())
    routing_feature_names = routing_metrics["routing_feature_names"]
    cp_thresholds = json.loads((results_dir / "conformal_thresholds.json").read_text())
    q_hat_probe_based = float(
        cp_thresholds["split_cp"][f"alpha_{ALPHA}"]["q_hat"]
    )

    # ------------------------------------------------------------------
    # Compute routing scores on the full val set
    # ------------------------------------------------------------------
    d = np.load(features_dir / "val_features.npz", allow_pickle=False)
    H_states = np.load(features_dir / "val_hidden.npy")
    H_scalar = d["H"].astype(np.float64)
    gap = d["gap"].astype(np.float64)
    p_true = d["p_true"].astype(np.float64)
    y = d["y"].astype(np.int32)
    n = len(y)

    probe_scores = final_probe.predict_proba(H_states)[:, 1]
    X = _build_feature_matrix(routing_feature_names,
                              H_scalar, gap, probe_scores, p_true)
    raw_proba = routing_lr.predict_proba(X)[:, 1]
    routing_scores = calibrator.transform(raw_proba)
    log.info("val: n=%d  mean routing_score=%.4f  acc=%.4f",
             n, routing_scores.mean(), y.mean())

    # ------------------------------------------------------------------
    # Setup A — ORIGINAL: q_hat from probe-derived conformal_cal, applied to val
    # ------------------------------------------------------------------
    decisions_A = routing_scores >= (1.0 - q_hat_probe_based)
    local_rate_A = float(decisions_A.mean())
    local_acc_A  = float(y[decisions_A].mean()) if decisions_A.any() else float("nan")
    log.info("Setup A (cal=probe_set, test=val):  q_hat=%.4f  "
             "local_rate=%.4f  local_acc=%.4f",
             q_hat_probe_based, local_rate_A, local_acc_A)

    # Bootstrap CI on local_rate_A
    rng = np.random.default_rng(SEED)
    boot_A = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot_A.append(float(decisions_A[idx].mean()))
    ci_A = (float(np.percentile(boot_A, 2.5)), float(np.percentile(boot_A, 97.5)))

    # ------------------------------------------------------------------
    # Setup B — NEW: q_hat from val_cal (K-fold), tested on val_test
    # ------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    q_hats_B: list[float] = []
    local_rate_B_aggregate = np.zeros(n, dtype=bool)
    local_acc_per_fold: list[float] = []
    local_rate_per_fold: list[float] = []

    for fold_i, (cal_idx, test_idx) in enumerate(skf.split(routing_scores, y), 1):
        q_hat_f = _compute_q_hat(routing_scores[cal_idx], ALPHA)
        decisions_f = routing_scores[test_idx] >= (1.0 - q_hat_f)
        local_rate_B_aggregate[test_idx] = decisions_f
        local_rate_f = float(decisions_f.mean())
        local_acc_f = (
            float(y[test_idx][decisions_f].mean()) if decisions_f.any() else float("nan")
        )
        q_hats_B.append(q_hat_f)
        local_rate_per_fold.append(local_rate_f)
        local_acc_per_fold.append(local_acc_f)
        log.info(
            "  Fold %d: |cal|=%d |test|=%d  q_hat=%.4f  "
            "local_rate=%.4f  local_acc=%.4f",
            fold_i, len(cal_idx), len(test_idx), q_hat_f, local_rate_f, local_acc_f,
        )

    # Aggregate local_rate using the cross-validated decisions
    local_rate_B = float(local_rate_B_aggregate.mean())
    local_acc_B  = (
        float(y[local_rate_B_aggregate].mean())
        if local_rate_B_aggregate.any() else float("nan")
    )
    boot_B = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot_B.append(float(local_rate_B_aggregate[idx].mean()))
    ci_B = (float(np.percentile(boot_B, 2.5)), float(np.percentile(boot_B, 97.5)))

    log.info(
        "Setup B (cal=val K-fold, test=val held-out): "
        "mean q_hat=%.4f  local_rate=%.4f  local_acc=%.4f",
        float(np.mean(q_hats_B)), local_rate_B, local_acc_B,
    )

    # ------------------------------------------------------------------
    # Print + persist
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"CP exchangeability diagnostic on MedMCQA val (n={n}, α={ALPHA})")
    print("=" * 72)
    print(f"{'Setup':<45}{'q_hat':>10}{'local_rate':>14}{'local_acc':>12}")
    print("-" * 72)
    print(f"{'A: cal=probe_set (KROK 7 original)':<45}"
          f"{q_hat_probe_based:>10.4f}"
          f"{local_rate_A:>14.4f}"
          f"{local_acc_A:>12.4f}")
    print(f"  95% CI on local_rate                       [{ci_A[0]:.4f}, {ci_A[1]:.4f}]")
    print()
    print(f"{'B: cal=val (5-fold cross-CP)':<45}"
          f"{float(np.mean(q_hats_B)):>10.4f}"
          f"{local_rate_B:>14.4f}"
          f"{local_acc_B:>12.4f}")
    print(f"  95% CI on local_rate                       [{ci_B[0]:.4f}, {ci_B[1]:.4f}]")
    print(f"  per-fold local_rates: "
          f"{['%.3f' % x for x in local_rate_per_fold]}")
    print()
    print(f"  CP target (1 − α) = {1 - ALPHA:.2f}")
    print(f"  Setup A {'MISSES' if local_rate_A < (1 - ALPHA) else 'meets'} target by "
          f"{(local_rate_A - (1 - ALPHA)):+.4f}")
    print(f"  Setup B {'MISSES' if local_rate_B < (1 - ALPHA) else 'meets'} target by "
          f"{(local_rate_B - (1 - ALPHA)):+.4f}")
    print("=" * 72)

    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "alpha": ALPHA,
        "cp_target": 1 - ALPHA,
        "n_val": int(n),
        "setup_A_probe_based": {
            "q_hat": round(q_hat_probe_based, 5),
            "local_rate": round(local_rate_A, 5),
            "local_rate_95ci": [round(ci_A[0], 5), round(ci_A[1], 5)],
            "local_acc": round(local_acc_A, 5),
            "meets_cp_target": bool(local_rate_A >= 1 - ALPHA),
            "miss_amount": round(local_rate_A - (1 - ALPHA), 5),
        },
        "setup_B_val_kfold": {
            "n_folds": N_FOLDS,
            "mean_q_hat": round(float(np.mean(q_hats_B)), 5),
            "per_fold_q_hats": [round(x, 5) for x in q_hats_B],
            "per_fold_local_rate": [round(x, 5) for x in local_rate_per_fold],
            "per_fold_local_acc": [round(x, 5) for x in local_acc_per_fold],
            "local_rate": round(local_rate_B, 5),
            "local_rate_95ci": [round(ci_B[0], 5), round(ci_B[1], 5)],
            "local_acc": round(local_acc_B, 5),
            "meets_cp_target": bool(local_rate_B >= 1 - ALPHA),
            "miss_amount": round(local_rate_B - (1 - ALPHA), 5),
        },
        "interpretation": (
            "If Setup B restores the CP target while Setup A misses it, "
            "the in-dist undercoverage is caused by exchangeability violation "
            "between probe_set (from MedMCQA train) and val, not the CP method."
        ),
    }
    (results_dir / "exchangeability_check.json").write_text(json.dumps(out, indent=2))
    log.info("Saved: %s", results_dir / "exchangeability_check.json")


if __name__ == "__main__":
    main()
