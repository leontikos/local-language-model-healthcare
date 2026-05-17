# Safe Enough to Answer Locally?
### Risk-Controlled Escalation for On-Device Medical Language Models

Bachelor's thesis — Oskar Kościański

Fine-tunes Mistral-7B on medical MCQ, builds a multi-signal uncertainty routing policy,
and applies split conformal prediction to provide formal coverage guarantees.
Evaluated across three distribution tiers: in-distribution, near-OOD, and far-OOD.

## Setup

```bash
git clone <repo>
cd diploma-thesis
pip install -e ".[dev]"
```

## Pipeline

```bash
make verify      # run critical checks first (cop encoding, tokenization)
make prepare     # data splits
make finetune    # LoRA fine-tuning (~1-7 days depending on GPU)
make pipeline    # extract → probe → routing → conformal → evaluate → ablations
```

Full pipeline specification: [`roadmap.md`](roadmap.md)

## Structure

```
scripts/     entry points (00_verify … 08_ablations)
src/thesis/  library code (data, models, routing, utils)
configs/     YAML configs (model, data)
results/     evaluation outputs (JSON)
figures/     plots for thesis
```

## Datasets

| Dataset | Role |
|---|---|
| MedMCQA | Fine-tuning + in-distribution eval |
| MedQA-USMLE | Near-OOD evaluation |
| MMLU medical (5 categories) | Far-OOD evaluation |
