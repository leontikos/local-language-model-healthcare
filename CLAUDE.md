# Medical LLM Routing — Thesis Project

Bachelor's thesis: risk-controlled escalation for on-device medical LLMs using conformal prediction.
Full pipeline spec: @roadmap.md

## Project layout

```
scripts/        # entry points — thin wrappers, no logic (00_verify.py … 08_ablations.py)
src/thesis/     # installable package — all logic lives here
  data/         # dataset loading, preprocessing, normalization
  models/       # model wrappers, feature extraction
  routing/      # probe, routing LR, isotonic calibration, conformal CP
  utils/        # metrics (AUGRC, AURC, ECE), plotting, bootstrap CI
configs/        # YAML configs (OmegaConf) — model and data settings
data/
  raw/          # original datasets — never modify
  splits/       # train/val/test indices as JSON (committed, small)
  features/     # extracted .npz feature caches (NOT committed)
checkpoints/    # LoRA adapters (NOT committed)
results/        # evaluation JSON outputs (committed)
figures/        # plots for thesis
notebooks/      # EDA only, not production code
```

## Setup

```bash
pip install -e ".[dev]"        # install package + dev deps
make verify                    # run 3 critical blockers
make finetune                  # start fine-tuning
make evaluate                  # full evaluation pipeline
```

## Key commands

```bash
python scripts/00_verify.py --blocker 1      # cop encoding check (run first)
python scripts/00_verify.py --blocker 2      # tokenization check
python scripts/00_verify.py --blocker 3      # p(True) check (after fine-tuning)
python scripts/01_prepare.py                 # data splits
python scripts/02_finetune.py                # LoRA fine-tuning
python scripts/03_extract.py                 # feature extraction + layer sweep
python scripts/04_probe.py                   # 5-fold correctness probe
python scripts/05_routing.py                 # routing LR + isotonic calibration
python scripts/06_conformal.py               # conformal calibration (q_hat)
python scripts/07_evaluate.py                # 3-tier evaluation
python scripts/08_ablations.py               # ablation tables
```

Config is read from `configs/main.yaml` and can be overridden:
```bash
python scripts/02_finetune.py model=qlora_4bit training.epochs=1
```

## Critical implementation details

**cop encoding (MedMCQA):** 0-indexed — `{0:'A', 1:'B', 2:'C', 3:'D'}`. ✅ VERIFIED.

**Mistral tokenization:** ids_ABCD = [1098, 1133, 1102, 1152] (A,B,C,D). ✅ VERIFIED.
Both `' A'` and `'A'` give the same token ID 1098 — safe to use either.
Use `logits[..., [1098, 1133, 1102, 1152]]` directly for restricted entropy.

**Entropy:** always use `p * torch.log(p + 1e-10)` to avoid -inf when p=0.

**Conformal quantile:** always use `np.quantile(..., method="higher")` — required for
the formal coverage guarantee. Standard interpolation breaks the math.

**Data isolation:** probe_set (16K) must be excluded from fine-tuning BEFORE training.
val set (4183) is used ONLY for conformal calibration — never for training anything.

**5-fold cross-fitting:** probe_scores_oof must be computed before routing LR training.
Routing LR trains on 12K of probe_set, isotonic calibration on the remaining 4K.

**OOF probe for val/test:** use `final_probe` (trained on full 16K) for all non-probe-set inference.

## Conventions

- Python 3.11, type hints everywhere
- `seed=42` for ALL random operations (numpy, torch, sklearn, datasets)
- Config loaded via OmegaConf: `cfg = OmegaConf.load("configs/main.yaml")`
- All results saved as JSON to `results/`; all figures to `figures/`
- Feature arrays saved as `.npz` with named arrays: `H`, `gap`, `p_true`, `y`, `pred`, `subject`
- Hidden states saved separately (large): `features/{split}_hidden.npy`
- Bootstrap CI: always 1000 resamples, seed=42, report as `[low, high]`
- Log with `logging` stdlib, not print() — except in scripts (entry points)

## Datasets

| Name | HuggingFace ID | Split used |
|---|---|---|
| MedMCQA | openlifescienceai/medmcqa | train→FT+probe, val→CP cal, test→eval |
| MedQA-USMLE | GBaker/MedQA-USMLE-4-options | test→near-OOD |
| MMLU medical | cais/mmlu (5 categories) | test→far-OOD |

MMLU categories: clinical_knowledge, professional_medicine, college_medicine,
medical_genetics, anatomy. (college_biology excluded — too general.)

## Metrics (src/thesis/utils/metrics.py)

- `compute_augrc(scores, labels)` — primary selective prediction (NeurIPS 2024)
- `compute_aurc(scores, labels)` — secondary
- `compute_ece(probs, labels, n_bins=15, strategy="quantile")`
- `bootstrap_ci(fn, scores, labels, n_boot=1000)` → `[low, high]`
- All in one file, all tested with pytest

## Routing pipeline (src/thesis/routing/)

```
probe.py       — LogisticRegressionCV on hidden states, 5-fold CV
routing_lr.py  — LogisticRegressionCV on [H, gap*, probe, p_true]
calibration.py — IsotonicRegression(out_of_bounds="clip")
conformal.py   — split CP and Mondrian CP
```

## Do not

- Do not hardcode paths — use `pathlib.Path` and config
- Do not commit checkpoints, features, wandb/, outputs/ — see .gitignore
- Do not use `val` set for anything except conformal calibration
- Do not run full fine-tuning in a notebook — use scripts/02_finetune.py
- Do not change seeds after experiments start
