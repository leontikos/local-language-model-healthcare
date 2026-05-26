"""
KROK 10 — THESIS PLOTS
======================
Generates publication-quality figures for the thesis writeup. All plots use
large fonts (axis labels ≥ 16 pt) and are saved as PDF (vector) for LaTeX
inclusion, plus PNG (300 DPI) for previewing.

Figures produced (under figures/):
  01_finetune_eval_loss.{pdf,png}      Fine-tune dynamics: failed run 2 vs successful run 3
  02_coverage_risk_curves.{pdf,png}    Selective prediction RC curves per split
  03_cp_exchangeability.{pdf,png}      Local rate: Setup A vs B — the diagnostic finding
  04_reliability_diagrams.{pdf,png}    Calibration per split (3-panel)
  05_routing_score_distributions.{pdf,png}  Score histograms split by correctness
  06_layer_sweep.{pdf,png}             Per-layer probe AUROC
  07_signal_subset_ablation.{pdf,png}  Feature subset comparison
  08_mondrian_vs_split_cp.{pdf,png}    CP variant comparison
  09_per_domain_coverage.{pdf,png}     Mondrian per-domain local rates on in-dist
  10_utility_vs_alpha.{pdf,png}        Utility curves for different escalation costs
  11_calibration_ablation.{pdf,png}    Brier+ECE comparison: uncalibrated/Platt/isotonic

Usage:
    python scripts/09_thesis_plots.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style — thesis quality
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "pdf.fonttype": 42,            # editable text in PDF
    "ps.fonttype": 42,
})

# Colorblind-friendly palette
COLORS = {
    "in_dist":   "#1f77b4",   # blue
    "near_OOD":  "#ff7f0e",   # orange
    "far_OOD":   "#2ca02c",   # green
    "correct":   "#2ca02c",
    "wrong":     "#d62728",
    "split_cp":  "#1f77b4",
    "mondrian":  "#9467bd",
    "fail_run":  "#d62728",
    "success_run": "#2ca02c",
    "target_line": "#888888",
}
SPLIT_LABEL = {
    "in_dist":   "in-dist (MedMCQA val)",
    "near_OOD":  "near-OOD (MedQA)",
    "far_OOD":   "far-OOD (MMLU med.)",
}

FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Hardcoded fine-tune eval trajectories (extracted earlier from training logs)
# ---------------------------------------------------------------------------
RUN2_EVAL = [
    (0.10, 1.074), (0.20, 1.168), (0.30, 1.455), (0.40, 1.371), (0.51, 1.418),
    (0.61, 1.369), (0.71, 1.368), (0.81, 1.378), (0.91, 1.379), (1.01, 1.368),
    (1.11, 1.382), (1.21, 1.423), (1.32, 1.389), (1.42, 1.369), (1.52, 1.367),
]
RUN3_EVAL = [
    (0.10, 0.918), (0.20, 0.927), (0.30, 0.844), (0.40, 0.806), (0.51, 0.743),
    (0.61, 0.705), (0.71, 0.676), (0.81, 0.637), (0.91, 0.654), (1.01, 0.741),
    (1.11, 0.686), (1.21, 0.728), (1.32, 0.724), (1.42, 0.699), (1.52, 0.648),
]


def _savefig(fig, name: str) -> None:
    pdf = FIGS / f"{name}.pdf"
    png = FIGS / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    log.info("  saved %s and %s", pdf.name, png.name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading helpers (avoid recomputing routing scores in each plot)
# ---------------------------------------------------------------------------
def load_routing_scores_per_split() -> dict[str, dict]:
    """Return {tag: {scores, y, subject, n}} for the 3 evaluation splits."""
    ckpt_dir = ROOT / "checkpoints"
    feats_dir = ROOT / "data" / "features"
    res_dir = ROOT / "results"
    with open(ckpt_dir / "final_probe.pkl", "rb") as f:
        final_probe = pickle.load(f)
    with open(ckpt_dir / "routing_lr.pkl", "rb") as f:
        routing_lr = pickle.load(f)
    with open(ckpt_dir / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    routing_meta = json.loads((res_dir / "routing_metrics.json").read_text())
    feat_names = routing_meta["routing_feature_names"]

    out = {}
    for tag, stem in [("in_dist", "val"), ("near_OOD", "test_medqa"),
                      ("far_OOD", "test_mmlu")]:
        d = np.load(feats_dir / f"{stem}_features.npz", allow_pickle=False)
        H_states = np.load(feats_dir / f"{stem}_hidden.npy")
        probe_s = final_probe.predict_proba(H_states)[:, 1]
        cols = []
        for n in feat_names:
            if n == "H":      cols.append(d["H"].astype(np.float64))
            elif n == "gap":  cols.append(d["gap"].astype(np.float64))
            elif n == "probe": cols.append(probe_s)
            elif n == "p_true": cols.append(d["p_true"].astype(np.float64))
        X = np.column_stack(cols).astype(np.float64)
        raw = routing_lr.predict_proba(X)[:, 1]
        scores = calibrator.transform(raw)
        out[tag] = {
            "scores": scores,
            "y": d["y"].astype(np.int32),
            "subject": d["subject"].astype(str),
            "n": len(d["y"]),
            "stem": stem,
        }
    return out


# ===========================================================================
# Plot 1 — Fine-tune eval loss: failed run vs successful run
# ===========================================================================
def plot_finetune_eval_loss() -> None:
    log.info("Plot 1 — finetune eval loss")
    fig, ax = plt.subplots(figsize=(10, 6))
    rand = np.log(4)

    e2 = np.array([p[0] for p in RUN2_EVAL]); v2 = np.array([p[1] for p in RUN2_EVAL])
    e3 = np.array([p[0] for p in RUN3_EVAL]); v3 = np.array([p[1] for p in RUN3_EVAL])

    ax.plot(e2, v2, marker="o", color=COLORS["fail_run"],
            label="Run 2 — LR=2e-4 (destabilised)")
    ax.plot(e3, v3, marker="s", color=COLORS["success_run"],
            label="Run 3 — LR=5e-5 (stable)")
    ax.axhline(rand, ls="--", color=COLORS["target_line"], lw=1.5,
               label=f"random baseline = log(4) ≈ {rand:.3f}")

    # Mark best points
    best2 = (e2[np.argmin(v2)], float(np.min(v2)))
    best3 = (e3[np.argmin(v3)], float(np.min(v3)))
    ax.scatter([best2[0]], [best2[1]], s=180, marker="*",
               color="white", edgecolor=COLORS["fail_run"], linewidth=2.2,
               zorder=6)
    ax.scatter([best3[0]], [best3[1]], s=180, marker="*",
               color="white", edgecolor=COLORS["success_run"], linewidth=2.2,
               zorder=6)
    ax.annotate(f"Run 2 best: {best2[1]:.3f} @ epoch {best2[0]}\n"
                f"(during warm-up — barely below random)",
                xy=best2, xytext=(0.18, 1.55),
                fontsize=12, color=COLORS["fail_run"],
                arrowprops=dict(arrowstyle="->", color=COLORS["fail_run"], lw=1.5))
    ax.annotate(f"Run 3 best: {best3[1]:.3f} @ epoch {best3[0]}\n"
                f"(post-warmup, true minimum)",
                xy=best3, xytext=(0.88, 0.90),
                fontsize=12, color=COLORS["success_run"],
                arrowprops=dict(arrowstyle="->", color=COLORS["success_run"], lw=1.5))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eval loss (cross-entropy on answer token)")
    ax.set_title("Fine-tuning dynamics — LR sensitivity")
    ax.legend(loc="center right", framealpha=0.95)
    ax.set_xlim(0, 1.7)
    ax.set_ylim(0.55, 1.7)
    _savefig(fig, "01_finetune_eval_loss")


# ===========================================================================
# Plot 2 — Coverage-risk curves per split
# ===========================================================================
def plot_coverage_risk_curves(per_split: dict) -> None:
    log.info("Plot 2 — coverage-risk curves")
    fig, ax = plt.subplots(figsize=(10, 6))

    for tag in ["in_dist", "near_OOD", "far_OOD"]:
        s = per_split[tag]["scores"]
        y = per_split[tag]["y"]
        order = np.argsort(s)[::-1]
        y_sorted = y[order]
        n = len(y)
        cov = np.arange(1, n + 1) / n
        cum_correct = np.cumsum(y_sorted)
        risk = 1.0 - cum_correct / np.arange(1, n + 1)
        ax.plot(cov, risk, color=COLORS[tag], label=f"{SPLIT_LABEL[tag]}  (n={n})")

    ax.set_xlabel("Coverage (fraction answered locally)")
    ax.set_ylabel("Risk (1 − accuracy on locals)")
    ax.set_title("Risk–Coverage curves\n(lower-left = better selective prediction)")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, None)
    ax.legend(loc="upper left", framealpha=0.95)

    # Annotate the deployed operating point @ α=0.10 per split
    cp = json.loads((ROOT / "results" / "evaluation.json").read_text())
    for tag in ["in_dist", "near_OOD", "far_OOD"]:
        m = cp["splits"][tag.replace("_", "-").replace("in-dist", "in-dist")
                          .replace("near-OOD", "near-OOD")
                          .replace("far-OOD", "far-OOD")]
        ax.scatter([m["local_rate"]], [1 - m["local_acc"]],
                   color=COLORS[tag], s=130, marker="X",
                   edgecolors="black", linewidths=1.4, zorder=5)
    ax.text(0.02, 0.02,
            "X markers: deployed operating point @ α = 0.10",
            transform=ax.transAxes, fontsize=12, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
    _savefig(fig, "02_coverage_risk_curves")


# ===========================================================================
# Plot 3 — CP exchangeability comparison (THE thesis finding)
# ===========================================================================
def plot_cp_exchangeability() -> None:
    log.info("Plot 3 — CP exchangeability")
    data = json.loads((ROOT / "results" / "exchangeability_check.json").read_text())
    alpha = data["alpha"]
    target = data["cp_target"]
    A = data["setup_A_probe_based"]
    B = data["setup_B_val_kfold"]

    fig, ax = plt.subplots(figsize=(10, 6))
    setups = ["Setup A:\ncal = probe_set\n(MedMCQA train)",
              "Setup B:\ncal = val (5-fold CP)\n(same dist. as test)"]
    rates = [A["local_rate"], B["local_rate"]]
    ci_lows = [rates[0] - A["local_rate_95ci"][0], rates[1] - B["local_rate_95ci"][0]]
    ci_highs = [A["local_rate_95ci"][1] - rates[0], B["local_rate_95ci"][1] - rates[1]]
    colors = [COLORS["fail_run"], COLORS["success_run"]]
    bars = ax.bar(setups, rates, color=colors, edgecolor="black", linewidth=1.0,
                  yerr=[ci_lows, ci_highs], capsize=14, error_kw={"linewidth": 2})

    ax.axhline(target, ls="--", color=COLORS["target_line"], lw=2,
               label=f"CP target (1 − α) = {target:.2f}")

    for bar, rate, setup_data, label in zip(bars, rates,
                                             [A, B], ["MISSES", "MEETS"]):
        y = bar.get_height()
        miss = setup_data["miss_amount"]
        sign = "+" if miss >= 0 else "−"
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.025,
                f"{rate:.3f}\n{label} target by {sign}{abs(miss):.3f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylabel("Empirical local rate on val set")
    ax.set_title(f"Conformal exchangeability diagnostic (α = {alpha}, val n = {data['n_val']})\n"
                 "Calibrating on train-derived data violates CP; calibrating on val restores it")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.95)
    _savefig(fig, "03_cp_exchangeability")


# ===========================================================================
# Plot 4 — Reliability diagrams (3-panel)
# ===========================================================================
def plot_reliability_diagrams(per_split: dict) -> None:
    log.info("Plot 4 — reliability diagrams")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    n_bins = 12
    for ax, tag in zip(axes, ["in_dist", "near_OOD", "far_OOD"]):
        s = per_split[tag]["scores"]
        y = per_split[tag]["y"]
        edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
        edges[-1] = edges[-1] + 1e-9
        confs, accs, weights = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (s >= lo) & (s < hi)
            if mask.sum() < 5:
                continue
            confs.append(s[mask].mean())
            accs.append(y[mask].mean())
            weights.append(mask.sum())

        confs, accs, weights = np.array(confs), np.array(accs), np.array(weights)
        sizes = 30 + 350 * (weights / weights.max())
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5,
                label="perfect calibration")
        ax.scatter(confs, accs, s=sizes, alpha=0.75, color=COLORS[tag],
                   edgecolors="black", linewidths=1.0)
        ax.plot(confs, accs, color=COLORS[tag], lw=1.5, alpha=0.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(SPLIT_LABEL[tag], fontsize=15)
        ax.set_xlabel("Predicted routing score")
        if tag == "in_dist":
            ax.set_ylabel("Empirical accuracy")
        else:
            ax.set_ylabel("")
        ax.legend(loc="upper left", fontsize=11)

    fig.suptitle("Reliability diagrams — calibration of routing scores per split\n"
                 "(dot area ∝ number of examples in bin)", y=1.02, fontsize=17)
    fig.tight_layout()
    _savefig(fig, "04_reliability_diagrams")


# ===========================================================================
# Plot 5 — Routing score distributions split by correctness
# ===========================================================================
def plot_routing_score_distributions(per_split: dict) -> None:
    log.info("Plot 5 — routing score distributions")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    cp = json.loads((ROOT / "results" / "conformal_thresholds.json").read_text())
    decision_threshold = 1.0 - float(cp["split_cp"]["alpha_0.1"]["q_hat"])

    bins = np.linspace(0, 1, 35)
    for ax, tag in zip(axes, ["in_dist", "near_OOD", "far_OOD"]):
        s = per_split[tag]["scores"]; y = per_split[tag]["y"]
        ax.hist(s[y == 1], bins=bins, color=COLORS["correct"],
                alpha=0.65, label="model correct", edgecolor="white", linewidth=0.4)
        ax.hist(s[y == 0], bins=bins, color=COLORS["wrong"],
                alpha=0.65, label="model wrong", edgecolor="white", linewidth=0.4)
        ax.axvline(decision_threshold, color="black", ls="--", lw=2,
                   label=f"deployed threshold = {decision_threshold:.3f}")
        ax.set_xlim(0, 1)
        ax.set_title(SPLIT_LABEL[tag], fontsize=15)
        ax.set_xlabel("Routing score (P(model correct))")
        if tag == "in_dist":
            ax.set_ylabel("Examples")
        ax.legend(loc="upper center", fontsize=11)

    fig.suptitle("Routing score distributions by ground-truth correctness\n"
                 "(better separation = better routing discrimination)",
                 y=1.02, fontsize=17)
    fig.tight_layout()
    _savefig(fig, "05_routing_score_distributions")


# ===========================================================================
# Plot 6 — Layer sweep
# ===========================================================================
def plot_layer_sweep() -> None:
    log.info("Plot 6 — layer sweep")
    data = json.loads((ROOT / "results" / "layer_sweep.json").read_text())
    layers = sorted([int(k) for k in data["layers"].keys()])
    means = [data["layers"][str(l)]["mean_auroc"] for l in layers]
    stds  = [data["layers"][str(l)]["std_auroc"] for l in layers]
    best = data["best_layer"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = [COLORS["fail_run"] if l != best else COLORS["success_run"]
                  for l in layers]
    bars = ax.bar([str(l) for l in layers], means, yerr=stds, capsize=12,
                  color=bar_colors, edgecolor="black", linewidth=1.0,
                  error_kw={"linewidth": 1.8})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=13,
                fontweight="bold")
    ax.set_xlabel("Hidden-state layer index (Mistral-7B has 32 layers)")
    ax.set_ylabel("Probe AUROC (5-fold CV, n=2,000)")
    ax.set_title("Per-layer probe AUROC — late-middle layers win over the final layers")
    ax.set_ylim(min(means) - 0.07, max(means) + 0.05)
    legend_elements = [
        Patch(facecolor=COLORS["success_run"], edgecolor="black",
              label=f"best layer = {best}"),
        Patch(facecolor=COLORS["fail_run"], edgecolor="black",
              label="other candidates"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", framealpha=0.95)
    _savefig(fig, "06_layer_sweep")


# ===========================================================================
# Plot 7 — Signal subset ablation
# ===========================================================================
def plot_signal_subset_ablation() -> None:
    log.info("Plot 7 — signal subset ablation")
    data = json.loads((ROOT / "results" / "ablations.json").read_text())
    sig = data["ablations"]["table1_signal_subset"]
    deployed_feats = data["default_routing_features"]

    names = list(sig.keys())
    aurocs = [sig[n]["per_split"]["in_dist"]["auroc"] for n in names]
    augrcs = [sig[n]["per_split"]["in_dist"]["augrc"] for n in names]
    local_rates = [sig[n]["per_split"]["in_dist"]["local_rate"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    x = np.arange(len(names))

    deployed_label = "+".join(deployed_feats)
    bar_colors = [COLORS["success_run"] if n == deployed_label else COLORS["in_dist"]
                  for n in names]

    metrics = [
        ("AUROC (↑ better)",           aurocs, axes[0], 0.55, max(aurocs) + 0.03),
        ("AUGRC (↓ better)",           augrcs, axes[1], None, None),
        ("local rate @ α=0.10 (target 0.90)", local_rates, axes[2], 0.5, 1.0),
    ]
    for title, values, ax, ylo, yhi in metrics:
        bars = ax.bar(x, values, color=bar_colors, edgecolor="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=11)
        ax.set_title(title, fontsize=15)
        if ylo is not None:
            ax.set_ylim(ylo, yhi)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(values) - min(values)) * 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=11)
        if "local rate" in title:
            ax.axhline(0.9, ls="--", color=COLORS["target_line"], lw=1.6)

    legend_elements = [
        Patch(facecolor=COLORS["success_run"], edgecolor="black",
              label=f"deployed: {deployed_label}"),
        Patch(facecolor=COLORS["in_dist"], edgecolor="black",
              label="alternative subset"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.03), fontsize=13)
    fig.suptitle("Signal subset ablation (in-dist) — H carries most discrimination; "
                 "p_true adds nothing", y=1.10, fontsize=17)
    fig.tight_layout()
    _savefig(fig, "07_signal_subset_ablation")


# ===========================================================================
# Plot 8 — Mondrian vs Split CP
# ===========================================================================
def plot_mondrian_vs_split() -> None:
    log.info("Plot 8 — Mondrian vs Split CP")
    data = json.loads((ROOT / "results" / "ablations.json").read_text())
    cp_tab = data["ablations"]["table3_cp_variant"]

    tags = ["in_dist", "near_OOD", "far_OOD"]
    split_rates = [cp_tab["split_global"][t]["local_rate"] for t in tags]
    mond_rates  = [cp_tab["mondrian"][t]["local_rate"] for t in tags]

    x = np.arange(len(tags))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w / 2, split_rates, w, label="Split-CP (global)",
                color=COLORS["split_cp"], edgecolor="black", linewidth=1.0)
    b2 = ax.bar(x + w / 2, mond_rates, w, label="Mondrian (per-domain)",
                color=COLORS["mondrian"], edgecolor="black", linewidth=1.0)
    ax.axhline(0.90, ls="--", color=COLORS["target_line"], lw=2,
               label="CP target (1 − α) = 0.90")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABEL[t] for t in tags], fontsize=13)
    ax.set_ylabel("Empirical local rate")
    ax.set_title("Split-CP vs Mondrian CP — per-domain stratification partially "
                 "fixes the\nin-distribution exchangeability issue "
                 "(but doesn't fully reach 0.90)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.95)
    _savefig(fig, "08_mondrian_vs_split_cp")


# ===========================================================================
# Plot 9 — Per-domain coverage on in-dist
# ===========================================================================
def plot_per_domain_coverage() -> None:
    log.info("Plot 9 — per-domain coverage on in-dist")
    data = json.loads((ROOT / "results" / "ablations.json").read_text())
    in_dist = data["ablations"]["table3_cp_variant"]["mondrian"]["in_dist"]
    per_dom_rate = in_dist["per_domain_local_rate"]
    per_dom_n = in_dist["per_domain_n"]

    domains = sorted(per_dom_rate.keys())
    rates = [per_dom_rate[d] for d in domains]
    ns = [per_dom_n[d] for d in domains]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(domains, rates,
                  color=[COLORS["success_run"] if r >= 0.9 else COLORS["fail_run"]
                         for r in rates],
                  edgecolor="black", linewidth=1.0)
    ax.axhline(0.90, ls="--", color=COLORS["target_line"], lw=2,
               label="CP target (1 − α) = 0.90")
    for bar, r, n in zip(bars, rates, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{r:.3f}\nn={n}", ha="center", va="bottom", fontsize=12)
    ax.set_xticklabels(domains, rotation=20, ha="right", fontsize=13)
    ax.set_ylabel("Empirical local rate (Mondrian CP)")
    ax.set_title("Per-domain local rate on in-dist (Mondrian CP)\n"
                 "Domain stratification is uneven — anatomy and pharmacology lag")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.95)
    legend2 = [
        Patch(facecolor=COLORS["success_run"], edgecolor="black",
              label="meets 0.90 target"),
        Patch(facecolor=COLORS["fail_run"], edgecolor="black",
              label="misses 0.90 target"),
    ]
    leg2 = ax.legend(handles=legend2, loc="lower left", framealpha=0.95)
    ax.add_artist(leg2)
    _savefig(fig, "09_per_domain_coverage")


# ===========================================================================
# Plot 10 — Utility vs α curves for different escalation costs
# ===========================================================================
def plot_utility_vs_alpha(per_split: dict) -> None:
    """Utility = local_acc · (1 − escalation_rate) − c · escalation_rate.

    Uses conformal_cal-derived q_hat per α and evaluates on each test split.
    Recomputes q_hat on a fine α grid to draw smooth curves.
    """
    log.info("Plot 10 — utility vs α curves")
    feats_dir = ROOT / "data" / "features"
    splits_dir = ROOT / "data" / "splits"
    ckpt_dir = ROOT / "checkpoints"
    res_dir = ROOT / "results"

    # Build calibration scores on conformal_cal
    d = np.load(feats_dir / "probe_features.npz", allow_pickle=False)
    H_p = d["H"].astype(np.float64); gap_p = d["gap"].astype(np.float64)
    pt_p = d["p_true"].astype(np.float64)
    probe_oof = np.load(feats_dir / "probe_scores_oof.npy").astype(np.float64)
    cc_idx = np.array(json.loads(
        (splits_dir / "conformal_cal_local_idx.json").read_text()))
    with open(ckpt_dir / "routing_lr.pkl", "rb") as f:
        routing_lr = pickle.load(f)
    with open(ckpt_dir / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    routing_meta = json.loads((res_dir / "routing_metrics.json").read_text())
    feat_names = routing_meta["routing_feature_names"]

    cols = []
    for n in feat_names:
        if n == "H":      cols.append(H_p[cc_idx])
        elif n == "gap":  cols.append(gap_p[cc_idx])
        elif n == "probe": cols.append(probe_oof[cc_idx])
        elif n == "p_true": cols.append(pt_p[cc_idx])
    X_cc = np.column_stack(cols).astype(np.float64)
    cc_scores = calibrator.transform(routing_lr.predict_proba(X_cc)[:, 1])

    def _q_hat(alpha):
        n = len(cc_scores)
        s = 1.0 - cc_scores
        return float(np.quantile(s, np.ceil((1 - alpha) * (n + 1)) / n,
                                 method="higher"))

    alphas = np.linspace(0.02, 0.30, 28)
    costs = [0.1, 0.3, 0.5, 0.8]
    cost_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, tag in zip(axes, ["in_dist", "near_OOD", "far_OOD"]):
        s = per_split[tag]["scores"]
        y = per_split[tag]["y"]
        for c, col in zip(costs, cost_colors):
            utilities = []
            for a in alphas:
                q = _q_hat(a)
                dec = s >= (1.0 - q)
                local_rate = float(dec.mean())
                escalation = 1.0 - local_rate
                local_acc = float(y[dec].mean()) if dec.any() else 0.0
                u = local_acc * (1 - escalation) - c * escalation
                utilities.append(u)
            ax.plot(alphas, utilities, marker="o", markersize=4,
                    color=col, label=f"c = {c}")
        ax.axvline(0.10, ls="--", color=COLORS["target_line"], lw=1.5,
                   label="deployed α = 0.10")
        ax.set_xlabel("α (CP miscoverage budget)")
        if tag == "in_dist":
            ax.set_ylabel("Expected utility U(α; c)")
        ax.set_title(SPLIT_LABEL[tag], fontsize=15)
        ax.legend(loc="lower left", fontsize=11)
        ax.set_xlim(0.0, 0.32)
    fig.suptitle("Utility-cost trade-off vs α      "
                 "U(α; c) = local_acc · (1 − esc) − c · esc",
                 y=1.03, fontsize=17)
    fig.tight_layout()
    _savefig(fig, "10_utility_vs_alpha")


# ===========================================================================
# Plot 11 — Calibration method ablation (Brier + ECE per split)
# ===========================================================================
def plot_calibration_ablation() -> None:
    log.info("Plot 11 — calibration ablation")
    data = json.loads((ROOT / "results" / "ablations.json").read_text())
    cal = data["ablations"]["table2_calibration"]
    methods = list(cal.keys())
    tags = ["in_dist", "near_OOD", "far_OOD"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    x = np.arange(len(tags))
    w = 0.27
    method_colors = ["#bcbd22", "#1f77b4", "#9467bd"]

    for ax, metric, title in zip(
        axes,
        ["brier", "ece"],
        ["Brier score (↓ better)", "Expected Calibration Error (↓ better)"],
    ):
        for i, (method, color) in enumerate(zip(methods, method_colors)):
            vals = [cal[method][t][metric] for t in tags]
            bars = ax.bar(x + (i - len(methods) / 2 + 0.5) * w, vals, w,
                          label=method, color=color, edgecolor="black",
                          linewidth=1.0)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([SPLIT_LABEL[t] for t in tags], fontsize=12)
        ax.set_title(title, fontsize=15)
        ax.set_ylim(0, None)
        if metric == "brier":
            ax.set_ylabel("Brier")
            ax.legend(loc="upper left", title="Calibrator", framealpha=0.95)
        else:
            ax.set_ylabel("ECE (15 quantile bins)")
    fig.suptitle("Calibration-method ablation — isotonic wins narrowly; "
                 "Platt is worse than uncalibrated",
                 y=1.02, fontsize=17)
    fig.tight_layout()
    _savefig(fig, "11_calibration_ablation")


def main() -> None:
    log.info("Loading routing scores for all 3 evaluation splits …")
    per_split = load_routing_scores_per_split()
    log.info("  loaded: %s",
             ", ".join(f"{k}={v['n']}" for k, v in per_split.items()))

    plot_finetune_eval_loss()
    plot_coverage_risk_curves(per_split)
    plot_cp_exchangeability()
    plot_reliability_diagrams(per_split)
    plot_routing_score_distributions(per_split)
    plot_layer_sweep()
    plot_signal_subset_ablation()
    plot_mondrian_vs_split()
    plot_per_domain_coverage()
    plot_utility_vs_alpha(per_split)
    plot_calibration_ablation()

    log.info("All plots saved under %s", FIGS)
    print(f"\nFigures written to:  {FIGS}/")
    print("Files: 01..11  (each as both .pdf and .png)")


if __name__ == "__main__":
    main()
