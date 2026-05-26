"""
KROK 7 — CONFORMAL CALIBRATION (split CP + Mondrian CP)
========================================================
Compute q_hat thresholds on the held-out 3,307 conformal_cal examples (subset
of probe_set, disjoint from routing_train and iso_cal). The decision rule used
at evaluation time is:

    answer locally   iff   routing_score(x) >= 1 - q_hat

Split-CP gives a formal coverage guarantee:
    P(routing_score(X_test) >= 1 - q_hat) >= 1 - alpha
provided test and conformal_cal are exchangeable. Bootstrap CI on q_hat
quantifies sampling uncertainty.

Also computes Mondrian thresholds per clinical domain (5 domains, roadmap §7.4).

Usage:
    python scripts/06_conformal.py
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

ALPHAS = [0.05, 0.10, 0.15, 0.20]
N_BOOT = 1000
SEED = 42

# Roadmap §7.4 — 5 clinical domains
DOMAIN_MAP = {
    # Clinical
    "Surgery": "clinical", "Medicine": "clinical",
    "Obstetrics & Gynecology": "clinical", "Pediatrics": "clinical",
    "Anesthesia": "clinical", "Radiology": "clinical",
    "Ophthalmology": "clinical", "ENT": "clinical",
    "Orthopedics": "clinical", "Dental": "clinical",
    # Basic science
    "Pathology": "basic_science", "Microbiology": "basic_science",
    "Forensic Medicine": "basic_science",
    # Pharmacology
    "Pharmacology": "pharmacology", "Biochemistry": "pharmacology",
    # Anatomy / Physiology
    "Anatomy": "anatomy_physiology", "Physiology": "anatomy_physiology",
    # Other
    "Psychiatry": "other", "Skin": "other",
    "Preventive & Social Medicine": "other", "Social & Preventive Medicine": "other",
    "General Medicine": "other", "Unknown": "other", "unknown": "other",
}


def _build_feature_matrix(rows, names, H, gap, probe, p_true):
    import numpy as np
    cols = []
    for name in names:
        if name == "H":      cols.append(H[rows])
        elif name == "gap":  cols.append(gap[rows])
        elif name == "probe": cols.append(probe[rows])
        elif name == "p_true": cols.append(p_true[rows])
        else: raise ValueError(name)
    return np.column_stack(cols).astype(np.float64)


def _compute_q_hat(routing_scores, alpha: float) -> float:
    """Split CP threshold using non-conformity s = 1 - routing_score."""
    import numpy as np
    n = len(routing_scores)
    s_cal = 1.0 - routing_scores
    # ceiling-quantile required for the formal coverage guarantee
    return float(np.quantile(
        s_cal,
        np.ceil((1 - alpha) * (n + 1)) / n,
        method="higher",
    ))


def main() -> None:
    import numpy as np

    features_dir = ROOT / "data" / "features"
    splits_dir = ROOT / "data" / "splits"
    results_dir = ROOT / "results"
    ckpt_dir = ROOT / "checkpoints"

    log.info("Loading probe features + OOF probe scores + routing_lr + calibrator …")
    d = np.load(features_dir / "probe_features.npz", allow_pickle=False)
    probe_scores_oof = np.load(features_dir / "probe_scores_oof.npy")
    H = d["H"].astype(np.float64)
    gap = d["gap"].astype(np.float64)
    p_true = d["p_true"].astype(np.float64)
    y = d["y"].astype(np.int32)
    subject = d["subject"]

    with open(ckpt_dir / "routing_lr.pkl", "rb") as f:
        routing_lr = pickle.load(f)
    with open(ckpt_dir / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)

    routing_metrics = json.loads((results_dir / "routing_metrics.json").read_text())
    routing_feature_names = routing_metrics["routing_feature_names"]
    log.info("  Routing features: %s", routing_feature_names)

    cc_idx = np.array(json.loads(
        (splits_dir / "conformal_cal_local_idx.json").read_text()
    ), dtype=np.int64)
    log.info("  conformal_cal: %d examples", len(cc_idx))

    # ------------------------------------------------------------------
    # 1. Routing scores on conformal_cal
    # ------------------------------------------------------------------
    X_cc = _build_feature_matrix(
        cc_idx, routing_feature_names, H, gap, probe_scores_oof, p_true,
    )
    raw_proba = routing_lr.predict_proba(X_cc)[:, 1]
    routing_scores = calibrator.transform(raw_proba)
    y_cc = y[cc_idx]
    log.info("  conformal_cal accuracy = %.4f  mean routing_score = %.4f",
             y_cc.mean(), routing_scores.mean())

    # ------------------------------------------------------------------
    # 2. Split-CP q_hat per alpha + bootstrap CI
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    thresholds: dict[str, dict] = {}
    for alpha in ALPHAS:
        q_hat = _compute_q_hat(routing_scores, alpha)

        boot = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(routing_scores), size=len(routing_scores))
            boot.append(_compute_q_hat(routing_scores[idx], alpha))
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

        # Sanity check: how often does the rule "score >= 1-q_hat" fire on cal itself?
        local_rate = float((routing_scores >= (1.0 - q_hat)).mean())
        cov_on_cal = float(y_cc[routing_scores >= (1.0 - q_hat)].mean())

        thresholds[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "q_hat": round(float(q_hat), 5),
            "ci_95_low": round(float(ci_low), 5),
            "ci_95_high": round(float(ci_high), 5),
            "decision_threshold": round(float(1.0 - q_hat), 5),
            "local_rate_on_cal": round(local_rate, 4),
            "coverage_on_cal": round(cov_on_cal, 4),
        }
        log.info(
            "  alpha=%.2f → q_hat = %.4f  [95%% CI %.4f, %.4f]  "
            "decision_threshold = %.4f  local_rate_on_cal = %.3f  cov_on_cal = %.3f",
            alpha, q_hat, ci_low, ci_high, 1.0 - q_hat, local_rate, cov_on_cal,
        )

    # ------------------------------------------------------------------
    # 3. Mondrian CP (per clinical domain)
    # ------------------------------------------------------------------
    cc_subjects = subject[cc_idx].tolist()
    domains_per_example = np.array([
        DOMAIN_MAP.get(str(s), "other") for s in cc_subjects
    ])
    mondrian: dict[str, dict] = {}
    alpha_primary = 0.10
    for domain in sorted(set(DOMAIN_MAP.values())):
        mask = domains_per_example == domain
        n_d = int(mask.sum())
        if n_d < 30:
            log.warning(
                "  Mondrian: domain %r has only %d cal samples — q_hat unreliable",
                domain, n_d,
            )
            mondrian[domain] = {"n": n_d, "q_hat": None}
            continue
        q_hat_d = _compute_q_hat(routing_scores[mask], alpha_primary)
        mondrian[domain] = {
            "n": n_d,
            "q_hat": round(float(q_hat_d), 5),
            "local_rate_on_cal": round(float(
                (routing_scores[mask] >= (1.0 - q_hat_d)).mean()), 4),
        }
        log.info("  Mondrian %s (n=%d): q_hat = %.4f",
                 domain, n_d, q_hat_d)

    # ------------------------------------------------------------------
    # 4. Persist
    # ------------------------------------------------------------------
    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": SEED,
        "n_conformal_cal": int(len(cc_idx)),
        "alphas": ALPHAS,
        "primary_alpha": alpha_primary,
        "n_bootstrap": N_BOOT,
        "split_cp": thresholds,
        "mondrian_cp_alpha010": mondrian,
        "routing_feature_names": routing_feature_names,
    }
    (results_dir / "conformal_thresholds.json").write_text(json.dumps(out, indent=2))
    log.info("  Saved: %s", results_dir / "conformal_thresholds.json")

    print("\n" + "=" * 60)
    print("KROK 7 DONE")
    primary = thresholds[f"alpha_{alpha_primary}"]
    print(f"  Primary alpha            : {alpha_primary}")
    print(f"  q_hat                    : {primary['q_hat']:.4f}  "
          f"[95% CI {primary['ci_95_low']:.4f}, {primary['ci_95_high']:.4f}]")
    print(f"  decision threshold (1-q) : {primary['decision_threshold']:.4f}")
    print(f"  local_rate on cal        : {primary['local_rate_on_cal']:.3f}")
    print(f"  coverage  on cal         : {primary['coverage_on_cal']:.3f}")
    print(f"  Next                     : python scripts/07_evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
