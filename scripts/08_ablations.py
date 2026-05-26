"""
KROK 9 — ABLATIONS
==================
Implements four ablation tables per roadmap §9:

  Table 1 — Signal subset (which routing features matter)
  Table 2 — Calibration method (uncalibrated / Platt / isotonic / ROC-isotonic)
  Table 3 — CP variant (split CP global vs Mondrian per-domain)
  Table 4 — Probe architecture (LogisticRegression vs MLP)

Layer sweep (Table from roadmap §9.5) is already in results/layer_sweep.json from
KROK 4 — included here in the JSON output for completeness.

Each ablation table is evaluated on:
  in-dist (MedMCQA val, 4183)
  near-OOD (MedQA, 1273)
  far-OOD (MMLU, 945)

Usage:
    python scripts/08_ablations.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import datetime
from itertools import combinations
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


ALPHA = 0.10
SEED = 42
ALL_FEATURES = ["H", "gap", "probe", "p_true"]

# Same domain map as 06_conformal.py
DOMAIN_MAP = {
    "Surgery": "clinical", "Medicine": "clinical",
    "Obstetrics & Gynecology": "clinical", "Pediatrics": "clinical",
    "Anesthesia": "clinical", "Radiology": "clinical",
    "Ophthalmology": "clinical", "ENT": "clinical",
    "Orthopedics": "clinical", "Dental": "clinical",
    "Pathology": "basic_science", "Microbiology": "basic_science",
    "Forensic Medicine": "basic_science",
    "Pharmacology": "pharmacology", "Biochemistry": "pharmacology",
    "Anatomy": "anatomy_physiology", "Physiology": "anatomy_physiology",
    "Psychiatry": "other", "Skin": "other",
    "Preventive & Social Medicine": "other", "Social & Preventive Medicine": "other",
    "General Medicine": "other", "Unknown": "other", "unknown": "other",
}


def _build_X(names, H, gap, probe, p_true):
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
    s = 1.0 - routing_scores
    return float(np.quantile(s, np.ceil((1 - alpha) * (n + 1)) / n, method="higher"))


def _eval_at_alpha(routing_scores, y, q_hat):
    """Return decisions, local_rate, local_acc, escalation, system_acc_oracle."""
    import numpy as np
    decisions = routing_scores >= (1.0 - q_hat)
    n = len(y)
    n_local = int(decisions.sum())
    local_rate = float(decisions.mean())
    local_acc = float(y[decisions].mean()) if n_local > 0 else float("nan")
    system_oracle = float((y[decisions].sum() + (~decisions).sum()) / n)
    return {
        "n_local": n_local,
        "local_rate": round(local_rate, 4),
        "local_acc": round(local_acc, 4),
        "escalation_rate": round(1.0 - local_rate, 4),
        "system_acc_oracle": round(system_oracle, 4),
    }


def _full_metrics(routing_scores, y, q_hat):
    """All metrics for a single eval split + a given q_hat (for CP-coverage rows)."""
    res = {
        "auroc": round(float(roc_auc_score(y, routing_scores)), 4),
        "augrc": round(compute_augrc(routing_scores, y), 5),
        "aurc": round(compute_aurc(routing_scores, y), 5),
        "brier": round(float(brier_score_loss(y, routing_scores)), 4),
        "ece": round(compute_ece(routing_scores, y, n_bins=15), 4),
    }
    res.update(_eval_at_alpha(routing_scores, y, q_hat))
    return res


def main() -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import CalibratedClassifierCV

    features_dir = ROOT / "data" / "features"
    splits_dir   = ROOT / "data" / "splits"
    results_dir  = ROOT / "results"
    ckpt_dir     = ROOT / "checkpoints"

    # ------------------------------------------------------------------
    # Load everything
    # ------------------------------------------------------------------
    log.info("Loading artefacts …")
    with open(ckpt_dir / "final_probe.pkl", "rb") as f:
        final_probe = pickle.load(f)
    with open(ckpt_dir / "routing_lr.pkl", "rb") as f:
        routing_lr = pickle.load(f)
    with open(ckpt_dir / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    routing_meta = json.loads((results_dir / "routing_metrics.json").read_text())
    routing_features_default = routing_meta["routing_feature_names"]

    cp = json.loads((results_dir / "conformal_thresholds.json").read_text())
    q_hat_global = float(cp["split_cp"][f"alpha_{ALPHA}"]["q_hat"])
    mondrian_q   = {d: v.get("q_hat") for d, v in cp["mondrian_cp_alpha010"].items()}

    # Probe features (for routing-LR re-training under each ablation)
    d_probe = np.load(features_dir / "probe_features.npz", allow_pickle=False)
    H_p = d_probe["H"].astype(np.float64)
    gap_p = d_probe["gap"].astype(np.float64)
    pt_p = d_probe["p_true"].astype(np.float64)
    y_p = d_probe["y"].astype(np.int32)
    sub_p = d_probe["subject"]
    probe_oof = np.load(features_dir / "probe_scores_oof.npy").astype(np.float64)
    H_hidden_p = np.load(features_dir / "probe_hidden.npy")

    routing_train = np.array(json.loads(
        (splits_dir / "routing_train_local_idx.json").read_text()))
    iso_cal       = np.array(json.loads(
        (splits_dir / "iso_cal_local_idx.json").read_text()))
    conf_cal      = np.array(json.loads(
        (splits_dir / "conformal_cal_local_idx.json").read_text()))

    # Evaluation splits: pre-compute routing inputs for each
    EVAL = []
    for tag, stem in [("in_dist", "val"), ("near_OOD", "test_medqa"),
                      ("far_OOD", "test_mmlu")]:
        npz = np.load(features_dir / f"{stem}_features.npz", allow_pickle=False)
        hidden = np.load(features_dir / f"{stem}_hidden.npy")
        probe_s = final_probe.predict_proba(hidden)[:, 1]
        EVAL.append({
            "tag": tag,
            "stem": stem,
            "H": npz["H"].astype(np.float64),
            "gap": npz["gap"].astype(np.float64),
            "p_true": npz["p_true"].astype(np.float64),
            "probe": probe_s,
            "y": npz["y"].astype(np.int32),
            "subject": npz["subject"],
        })

    # Skeleton output
    ablations: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Table 1 — Signal subset ablation
    # ------------------------------------------------------------------
    log.info("Table 1 — Signal subset ablation")
    signal_subsets = [
        ["H"],
        ["probe"],
        ["p_true"],
        ["H", "probe"],
        ["H", "probe", "p_true"],
        ["H", "gap", "probe", "p_true"],   # all
        routing_features_default,           # actually deployed
    ]
    # Dedup (default = [H, probe, p_true])
    seen = set()
    dedup = []
    for s in signal_subsets:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            dedup.append(s)
    signal_subsets = dedup

    sig_results = {}
    for feats in signal_subsets:
        name = "+".join(feats)
        X_tr = _build_X(feats, H_p[routing_train], gap_p[routing_train],
                        probe_oof[routing_train], pt_p[routing_train])
        y_tr = y_p[routing_train]
        lr = LogisticRegressionCV(
            Cs=[0.01, 0.1, 1.0, 10.0], cv=3, penalty="l2", scoring="roc_auc",
            max_iter=2000, n_jobs=-1, random_state=SEED,
        )
        lr.fit(X_tr, y_tr)

        # Fit isotonic on iso_cal raw scores
        X_iso = _build_X(feats, H_p[iso_cal], gap_p[iso_cal],
                         probe_oof[iso_cal], pt_p[iso_cal])
        raw_iso = lr.predict_proba(X_iso)[:, 1]
        cal_iso = IsotonicRegression(out_of_bounds="clip").fit(raw_iso, y_p[iso_cal])

        # Compute q_hat on conf_cal
        X_cc = _build_X(feats, H_p[conf_cal], gap_p[conf_cal],
                        probe_oof[conf_cal], pt_p[conf_cal])
        cc_scores = cal_iso.transform(lr.predict_proba(X_cc)[:, 1])
        qh = _compute_q_hat(cc_scores, ALPHA)

        per_split = {}
        for ev in EVAL:
            X_ev = _build_X(feats, ev["H"], ev["gap"], ev["probe"], ev["p_true"])
            raw_ev = lr.predict_proba(X_ev)[:, 1]
            scores_ev = cal_iso.transform(raw_ev)
            per_split[ev["tag"]] = _full_metrics(scores_ev, ev["y"], qh)
        sig_results[name] = {
            "features": feats,
            "best_C": float(lr.C_[0]),
            "q_hat": round(qh, 4),
            "per_split": per_split,
        }
        log.info(
            "  %-30s  in-dist AUROC=%.4f AUGRC=%.4f local_rate=%.3f  "
            "OOD-near AUROC=%.4f  OOD-far AUROC=%.4f",
            name,
            per_split["in_dist"]["auroc"], per_split["in_dist"]["augrc"],
            per_split["in_dist"]["local_rate"],
            per_split["near_OOD"]["auroc"], per_split["far_OOD"]["auroc"],
        )
    ablations["table1_signal_subset"] = sig_results

    # ------------------------------------------------------------------
    # Table 2 — Calibration method ablation
    # Uses the DEPLOYED routing_lr (no re-train); only the calibrator changes.
    # Methods: uncalibrated, Platt (CalibratedClassifierCV sigmoid), isotonic_vanilla,
    # roc_isotonic (if regcal is installed).
    # ------------------------------------------------------------------
    log.info("Table 2 — Calibration method ablation")
    X_iso_def = _build_X(routing_features_default, H_p[iso_cal], gap_p[iso_cal],
                         probe_oof[iso_cal], pt_p[iso_cal])
    raw_iso = routing_lr.predict_proba(X_iso_def)[:, 1]
    y_iso = y_p[iso_cal]

    calibrators: dict[str, "object"] = {
        "uncalibrated": None,
        "isotonic_vanilla": IsotonicRegression(out_of_bounds="clip").fit(raw_iso, y_iso),
    }
    # Platt scaling via a tiny LR fit on raw_iso
    from sklearn.linear_model import LogisticRegression as _LR
    platt = _LR(C=1e6, solver="lbfgs", max_iter=1000)
    platt.fit(raw_iso.reshape(-1, 1), y_iso)
    calibrators["platt"] = platt
    # ROC-isotonic (optional)
    try:
        from regcal import ROCIsotonicRegression
        roc_iso = ROCIsotonicRegression().fit(raw_iso, y_iso)
        calibrators["roc_isotonic"] = roc_iso
    except ImportError:
        log.info("  regcal not installed — skipping roc_isotonic")

    def _apply_cal(name, cal, scores):
        if cal is None:
            return scores
        if name == "platt":
            return cal.predict_proba(scores.reshape(-1, 1))[:, 1]
        if name == "roc_isotonic":
            return cal.predict(scores)
        return cal.transform(scores)

    cal_results = {}
    for name, cal in calibrators.items():
        per_split = {}
        for ev in EVAL:
            X_ev = _build_X(routing_features_default,
                            ev["H"], ev["gap"], ev["probe"], ev["p_true"])
            raw_ev = routing_lr.predict_proba(X_ev)[:, 1]
            scores_ev = _apply_cal(name, cal, raw_ev)
            per_split[ev["tag"]] = {
                "brier": round(float(brier_score_loss(ev["y"], scores_ev)), 4),
                "ece":   round(compute_ece(scores_ev, ev["y"], n_bins=15), 4),
                "auroc": round(float(roc_auc_score(ev["y"], scores_ev)), 4),
            }
        cal_results[name] = per_split
        log.info(
            "  %-18s  in-dist Brier=%.4f ECE=%.4f  near-OOD Brier=%.4f  far-OOD Brier=%.4f",
            name,
            per_split["in_dist"]["brier"], per_split["in_dist"]["ece"],
            per_split["near_OOD"]["brier"], per_split["far_OOD"]["brier"],
        )
    ablations["table2_calibration"] = cal_results

    # ------------------------------------------------------------------
    # Table 3 — CP variant: split-CP global vs Mondrian per-domain
    # ------------------------------------------------------------------
    log.info("Table 3 — Split-CP global vs Mondrian per-domain")
    cp_results = {"split_global": {}, "mondrian": {}}
    for ev in EVAL:
        X_ev = _build_X(routing_features_default,
                        ev["H"], ev["gap"], ev["probe"], ev["p_true"])
        raw_ev = routing_lr.predict_proba(X_ev)[:, 1]
        scores_ev = calibrator.transform(raw_ev)

        cp_results["split_global"][ev["tag"]] = _eval_at_alpha(
            scores_ev, ev["y"], q_hat_global,
        )

        # Mondrian: per-example threshold by domain (fallback to "other")
        domains = np.array([
            DOMAIN_MAP.get(str(s), "other") for s in ev["subject"]
        ])
        per_example_q = np.array([
            mondrian_q[d] if mondrian_q.get(d) is not None else q_hat_global
            for d in domains
        ])
        decisions = scores_ev >= (1.0 - per_example_q)
        n_local = int(decisions.sum())
        local_rate = float(decisions.mean())
        local_acc = float(ev["y"][decisions].mean()) if n_local > 0 else float("nan")
        cp_results["mondrian"][ev["tag"]] = {
            "n_local": n_local,
            "local_rate": round(local_rate, 4),
            "local_acc": round(local_acc, 4),
            "escalation_rate": round(1.0 - local_rate, 4),
            "per_domain_n": {d: int((domains == d).sum())
                              for d in sorted(set(domains))},
            "per_domain_local_rate": {
                d: round(float(decisions[domains == d].mean()), 4)
                for d in sorted(set(domains)) if (domains == d).any()
            },
            "per_domain_local_acc": {
                d: (round(float(ev["y"][(domains == d) & decisions].mean()), 4)
                    if ((domains == d) & decisions).any() else None)
                for d in sorted(set(domains))
            },
        }
        log.info("  %s:  split-global local_rate=%.3f   mondrian local_rate=%.3f",
                 ev["tag"],
                 cp_results["split_global"][ev["tag"]]["local_rate"],
                 cp_results["mondrian"][ev["tag"]]["local_rate"])

    ablations["table3_cp_variant"] = cp_results

    # ------------------------------------------------------------------
    # Table 4 — Probe architecture: LR vs MLP
    # ------------------------------------------------------------------
    log.info("Table 4 — Probe architecture (LR vs MLP)")
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold

    # LR baseline (already have probe_scores_oof for it; use pooled AUROC)
    auroc_lr_oof = float(roc_auc_score(y_p, probe_oof))
    brier_lr_oof = float(brier_score_loss(y_p, probe_oof))

    # MLP probe: 4096 → 64 → 1 via 5-fold cross-fit on hidden states
    mlp_oof = np.zeros(len(y_p), dtype=np.float64)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(kf.split(H_hidden_p, y_p), start=1):
        mlp = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=1e-3,                # L2 reg (sklearn MLP has no dropout)
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=SEED + fold,
        )
        mlp.fit(H_hidden_p[tr], y_p[tr])
        mlp_oof[va] = mlp.predict_proba(H_hidden_p[va])[:, 1]
        log.info("  MLP fold %d/5: fold AUROC=%.4f",
                 fold, float(roc_auc_score(y_p[va], mlp_oof[va])))

    auroc_mlp_oof = float(roc_auc_score(y_p, mlp_oof))
    brier_mlp_oof = float(brier_score_loss(y_p, mlp_oof))
    delta_auroc = auroc_mlp_oof - auroc_lr_oof
    verdict = ("MLP wins (Δ > 0.02 — consider switching)"
               if delta_auroc > 0.02
               else "Keep LR (Δ ≤ 0.02 — MLP not worth complexity)")
    log.info("  LR AUROC=%.4f  MLP AUROC=%.4f  Δ=%+0.4f  → %s",
             auroc_lr_oof, auroc_mlp_oof, delta_auroc, verdict)

    ablations["table4_probe_architecture"] = {
        "lr_oof_auroc": round(auroc_lr_oof, 5),
        "lr_oof_brier": round(brier_lr_oof, 5),
        "mlp_oof_auroc": round(auroc_mlp_oof, 5),
        "mlp_oof_brier": round(brier_mlp_oof, 5),
        "delta_auroc": round(delta_auroc, 5),
        "verdict": verdict,
        "n_probe": int(len(y_p)),
    }

    # ------------------------------------------------------------------
    # Layer sweep (already done; copy reference into the ablations bundle)
    # ------------------------------------------------------------------
    layer_sweep = json.loads((results_dir / "layer_sweep.json").read_text())
    ablations["table5_layer_sweep_ref"] = {
        "best_layer": layer_sweep["best_layer"],
        "per_layer_auroc": {
            l: layer_sweep["layers"][l]["mean_auroc"]
            for l in layer_sweep["layers"]
        },
    }

    # ------------------------------------------------------------------
    # Persist + print
    # ------------------------------------------------------------------
    out = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "alpha": ALPHA,
        "default_routing_features": routing_features_default,
        "q_hat_global": q_hat_global,
        "ablations": ablations,
    }
    (results_dir / "ablations.json").write_text(json.dumps(out, indent=2))
    log.info("Saved: %s", results_dir / "ablations.json")

    # Pretty print summary
    print("\n" + "=" * 80)
    print("KROK 9 — ABLATION RESULTS")
    print("=" * 80)

    # Table 1
    print("\nTable 1 — Signal subset (in-dist results @ α=0.10)")
    print(f"  {'Subset':<28}{'AUROC':>10}{'AUGRC':>10}{'Brier':>10}{'local_rate':>14}")
    for name, info in sig_results.items():
        m = info["per_split"]["in_dist"]
        marker = "  ← deployed" if list(info["features"]) == list(routing_features_default) else ""
        print(f"  {name:<28}{m['auroc']:>10.4f}{m['augrc']:>10.4f}"
              f"{m['brier']:>10.4f}{m['local_rate']:>14.4f}{marker}")

    # Table 2
    print("\nTable 2 — Calibration method (Brier ↓ / ECE ↓)")
    print(f"  {'Method':<18}{'in-dist Brier':>16}{'in-dist ECE':>14}"
          f"{'near Brier':>14}{'far Brier':>14}")
    for name, per in cal_results.items():
        print(f"  {name:<18}{per['in_dist']['brier']:>16.4f}"
              f"{per['in_dist']['ece']:>14.4f}{per['near_OOD']['brier']:>14.4f}"
              f"{per['far_OOD']['brier']:>14.4f}")

    # Table 3
    print("\nTable 3 — CP variant (local_rate, CP target ≥ 0.90 at α=0.10)")
    print(f"  {'Method':<18}{'in-dist':>12}{'near-OOD':>12}{'far-OOD':>12}")
    for method, per in cp_results.items():
        print(f"  {method:<18}"
              f"{per['in_dist']['local_rate']:>12.4f}"
              f"{per['near_OOD']['local_rate']:>12.4f}"
              f"{per['far_OOD']['local_rate']:>12.4f}")

    # Table 4
    print("\nTable 4 — Probe architecture (5-fold OOF on probe_set, n=13,307)")
    a4 = ablations["table4_probe_architecture"]
    print(f"  LR    AUROC={a4['lr_oof_auroc']:.4f}  Brier={a4['lr_oof_brier']:.4f}")
    print(f"  MLP   AUROC={a4['mlp_oof_auroc']:.4f}  Brier={a4['mlp_oof_brier']:.4f}")
    print(f"  Δ AUROC = {a4['delta_auroc']:+0.4f}   → {a4['verdict']}")

    # Table 5 (reference only)
    print("\nTable 5 — Layer sweep (already in layer_sweep.json)")
    for l, auc in ablations["table5_layer_sweep_ref"]["per_layer_auroc"].items():
        marker = "  ← best" if int(l) == ablations["table5_layer_sweep_ref"]["best_layer"] else ""
        print(f"  Layer {l:>2}: AUROC={auc:.4f}{marker}")

    print("\n" + "=" * 80)
    print(f"Full results → {results_dir / 'ablations.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
