"""
KROK 8 — 3-TIER EVALUATION (in-dist + near-OOD + far-OOD)
==========================================================
Apply the full routing pipeline to:
  in-dist (MedMCQA val, 4,183)         — repurposed because test split has cop=-1
  near-OOD (MedQA-USMLE, 1,273)
  far-OOD  (MMLU medical 5 cat, 945)

For each split: compute discrimination (AUROC), selective-prediction (AUGRC,
AURC), calibration (Brier, ECE), CP coverage (local rate at α=0.10), and the
oracle system accuracy upper bound. Bootstrap 95% CIs on AUROC, AUGRC, Brier.

Usage:
    python scripts/07_evaluate.py
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

from thesis.utils.metrics import (
    compute_augrc, compute_aurc, compute_ece, bootstrap_ci,
    roc_auc_score, brier_score_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


SPLITS = [
    # (eval_name, features_file_stem, cp_guarantee_valid)
    ("in-dist",  "val",          True),   # MedMCQA val (in-dist replacement for test_medmcqa)
    ("near-OOD", "test_medqa",   False),  # MedQA-USMLE
    ("far-OOD",  "test_mmlu",    False),  # MMLU medical
]


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


def main() -> None:
    import numpy as np

    features_dir = ROOT / "data" / "features"
    results_dir = ROOT / "results"
    ckpt_dir = ROOT / "checkpoints"

    # ------------------------------------------------------------------
    # Load all trained models + thresholds
    # ------------------------------------------------------------------
    log.info("Loading routing artifacts …")
    with open(ckpt_dir / "final_probe.pkl", "rb") as f:
        final_probe = pickle.load(f)
    with open(ckpt_dir / "routing_lr.pkl", "rb") as f:
        routing_lr = pickle.load(f)
    with open(ckpt_dir / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)

    routing_metrics = json.loads((results_dir / "routing_metrics.json").read_text())
    routing_feature_names = routing_metrics["routing_feature_names"]
    log.info("  Routing features: %s", routing_feature_names)

    cp_thresholds = json.loads((results_dir / "conformal_thresholds.json").read_text())
    alpha_primary = cp_thresholds["primary_alpha"]
    q_hat = float(cp_thresholds["split_cp"][f"alpha_{alpha_primary}"]["q_hat"])
    decision_threshold = 1.0 - q_hat
    log.info("  Using α=%.2f  q_hat=%.4f  decision_threshold=%.4f",
             alpha_primary, q_hat, decision_threshold)

    # ------------------------------------------------------------------
    # Evaluate each split
    # ------------------------------------------------------------------
    all_results: dict[str, dict] = {}

    for eval_name, stem, cp_valid in SPLITS:
        npz_path = features_dir / f"{stem}_features.npz"
        npy_path = features_dir / f"{stem}_hidden.npy"
        if not (npz_path.exists() and npy_path.exists()):
            log.warning("  Skipping %s — missing %s or %s", eval_name, npz_path, npy_path)
            continue

        log.info("\n=== %s (%s) ===", eval_name, stem)
        d = np.load(npz_path, allow_pickle=False)
        H_states = np.load(npy_path)
        H_scalar = d["H"].astype(np.float64)
        gap = d["gap"].astype(np.float64)
        p_true = d["p_true"].astype(np.float64)
        y = d["y"].astype(np.int32)
        n = len(y)
        standalone_acc = float(y.mean())
        log.info("  n=%d  standalone_acc=%.4f", n, standalone_acc)

        # Probe scores via the final probe
        probe_scores = final_probe.predict_proba(H_states)[:, 1]

        # Routing pipeline
        X = _build_feature_matrix(routing_feature_names,
                                  H_scalar, gap, probe_scores, p_true)
        raw_proba = routing_lr.predict_proba(X)[:, 1]
        routing_scores = calibrator.transform(raw_proba)

        decisions = routing_scores >= decision_threshold
        n_local = int(decisions.sum())
        local_rate = float(decisions.mean())
        if n_local > 0:
            local_acc = float(y[decisions].mean())
        else:
            local_acc = float("nan")

        # Discrimination + selective + calibration
        auroc = float(roc_auc_score(y, routing_scores))
        augrc = compute_augrc(routing_scores, y)
        aurc = compute_aurc(routing_scores, y)
        brier = float(brier_score_loss(y, routing_scores))
        ece = compute_ece(routing_scores, y, n_bins=15, strategy="quantile")

        # Oracle system accuracy upper bound: locals right + all escalations counted as right
        system_acc_oracle = float(
            (y[decisions].sum() + (~decisions).sum()) / n
        )

        # Bootstrap CIs (1000)
        ci_auroc = bootstrap_ci(
            lambda s, ll: roc_auc_score(ll, s), routing_scores, y, n_boot=1000)
        ci_augrc = bootstrap_ci(compute_augrc, routing_scores, y, n_boot=1000)
        ci_brier = bootstrap_ci(
            lambda s, ll: brier_score_loss(ll, s), routing_scores, y, n_boot=1000)

        res = {
            "n": n,
            "standalone_acc": round(standalone_acc, 4),
            "auroc": round(auroc, 4),
            "auroc_ci": [round(ci_auroc[0], 4), round(ci_auroc[1], 4)],
            "augrc": round(augrc, 5),
            "augrc_ci": [round(ci_augrc[0], 5), round(ci_augrc[1], 5)],
            "aurc": round(aurc, 5),
            "brier": round(brier, 4),
            "brier_ci": [round(ci_brier[0], 4), round(ci_brier[1], 4)],
            "ece": round(ece, 4),
            "alpha": alpha_primary,
            "q_hat": round(q_hat, 4),
            "decision_threshold": round(decision_threshold, 4),
            "n_local": n_local,
            "local_rate": round(local_rate, 4),   # CP target ≥ 1 - alpha = 0.90
            "local_acc": round(local_acc, 4),
            "escalation_rate": round(1.0 - local_rate, 4),
            "system_acc_oracle": round(system_acc_oracle, 4),
            "cp_guarantee_valid": cp_valid,
        }
        all_results[eval_name] = res

        log.info("  AUROC=%.4f [95%% CI %.4f, %.4f]", auroc, *ci_auroc)
        log.info("  AUGRC=%.5f [95%% CI %.5f, %.5f]", augrc, *ci_augrc)
        log.info("  Brier=%.4f [95%% CI %.4f, %.4f]   ECE=%.4f",
                 brier, ci_brier[0], ci_brier[1], ece)
        log.info("  local_rate=%.3f (CP target ≥ %.2f, guarantee_valid=%s)  "
                 "local_acc=%.3f  escalation=%.3f",
                 local_rate, 1.0 - alpha_primary, cp_valid, local_acc,
                 1.0 - local_rate)
        log.info("  system_acc_oracle=%.4f", system_acc_oracle)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("KROK 8 — 3-tier evaluation results")
    print("=" * 72)
    cols = list(all_results.keys())
    header = f"{'Metric':<24}" + "".join(f"{c:>14}" for c in cols)
    print(header)
    print("-" * len(header))
    rows = [
        "n", "standalone_acc", "auroc", "augrc", "aurc", "brier", "ece",
        "local_rate", "local_acc", "escalation_rate", "system_acc_oracle",
    ]
    for k in rows:
        vals = [all_results[c].get(k) for c in cols]
        formatted = []
        for v in vals:
            if isinstance(v, int):
                formatted.append(f"{v:>14d}")
            elif v is None:
                formatted.append(f"{'n/a':>14}")
            else:
                formatted.append(f"{v:>14.4f}")
        print(f"{k:<24}" + "".join(formatted))
    print("=" * 72)
    print(f"  CP guarantee valid only for in-dist (cp_guarantee_valid). "
          f"OOD splits exhibit coverage degradation — the main thesis result.")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "alpha_primary": alpha_primary,
        "q_hat": round(float(q_hat), 5),
        "decision_threshold": round(float(decision_threshold), 5),
        "routing_feature_names": routing_feature_names,
        "splits": all_results,
    }
    (results_dir / "evaluation.json").write_text(json.dumps(out, indent=2))
    log.info("Saved: %s", results_dir / "evaluation.json")
    print("\nFull metrics → results/evaluation.json")


if __name__ == "__main__":
    main()
