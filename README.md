# 🏥 Safe Enough to Answer Locally?
## Risk-Controlled Escalation for On-Device Medical LLMs Using Conformal Prediction

> Bachelor's thesis — Oskar Kościański
> Mistral-7B-Instruct-v0.3 · LoRA fine-tuning · Split Conformal Prediction · MedMCQA

---

## 🎯 What this project does

An edge-deployed medical LLM should know when **not** to answer. This project builds a routing policy that:

1. **Fine-tunes** Mistral-7B on medical MCQ (MedMCQA, 104K examples)
2. **Extracts** 4 uncertainty signals per question (entropy, logit gap, hidden-state probe, p(True))
3. **Trains** a calibrated routing classifier (Logistic Regression + isotonic calibration)
4. **Applies** split conformal prediction to give a **formal coverage guarantee**: the system escalates to a human expert whenever P(correct) is too low, with guaranteed ≥90% recall on correct answers

---

## 🏗️ Architecture

```
Question
   │
   ▼
┌──────────────────────────────────────────────┐
│  Mistral-7B-Instruct-v0.3 (LoRA FP16)        │
│                                              │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Logits  │  │ Hidden   │  │ p(True)  │   │
│  │ A/B/C/D │  │ State L* │  │ Yes/No   │   │
│  └────┬────┘  └────┬─────┘  └────┬─────┘   │
└───────┼────────────┼─────────────┼──────────┘
        │            │             │
        ▼            ▼             │
    H + gap      Probe LR ────────┘
   (entropy)    (AUROC ~0.72)
        │            │             │
        └────────────┴─────────────┘
                     │
                     ▼
            Routing LR + Isotonic Cal
                     │
                     ▼
         routing_score ∈ [0, 1]
                     │
           ┌─────────┴─────────┐
           │   q_hat @ α=0.10  │  ← Split Conformal Prediction
           │   (val set, 4183) │     formal coverage guarantee
           └─────────┬─────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   ✅ Answer locally         🚨 Escalate to expert
```

---

## 📊 Datasets

| Dataset | HuggingFace ID | Role | Size |
|---|---|---|---|
| MedMCQA | `openlifescienceai/medmcqa` | Fine-tuning + probe + calibration | 182K train / 4,183 val / 6,150 test |
| MedQA-USMLE | `GBaker/MedQA-USMLE-4-options` | Near-OOD evaluation | 1,273 test |
| MMLU medical | `cais/mmlu` (5 categories) | Far-OOD evaluation | 945 test |

### Data splits (actual sizes after dedup)

```
MedMCQA train (182,822)
    │
    ▼ filter choice_type == "single"  →  120,765
    │
    ▼ stratified split (seed=42)
    │
    ├── train_ft     104,765  ──── LoRA fine-tuning ONLY
    └── probe_set     13,307  ──── probe + routing (2,693 exact-match duplicates removed)
              │
              ├── routing_train   9,307  ──── routing LR training
              └── iso_cal         4,000  ──── isotonic calibration

MedMCQA val  (4,183) ──── conformal calibration ONLY (q_hat)
MedMCQA test (6,150) ──── in-distribution evaluation
MedQA test   (1,273) ──── near-OOD evaluation
MMLU test      (945) ──── far-OOD evaluation
```

---

## ⚡ Quick Start

```bash
# 1. Install
git clone https://github.com/<your-username>/diploma-thesis
cd diploma-thesis
pip install -e ".[dev]"

# 2. Verify critical blockers (run FIRST, before anything else)
python scripts/00_verify.py --blocker 1   # cop encoding check
python scripts/00_verify.py --blocker 2   # tokenization check
# python scripts/00_verify.py --blocker 3 # after fine-tuning

# 3. Prepare data splits
python scripts/01_prepare.py

# 4. Fine-tune (requires ≥24 GB VRAM)
python scripts/02_finetune.py

# 5. Extract features (requires ≥8 GB VRAM)
python scripts/03_extract.py

# 6. Train probe + routing + conformal (CPU)
python scripts/04_probe.py
python scripts/05_routing.py
python scripts/06_conformal.py

# 7. Full 3-tier evaluation
make evaluate
```

Config overrides via OmegaConf syntax:
```bash
python scripts/02_finetune.py model=qlora_4bit training.epochs=1 --debug
python scripts/03_extract.py --split probe --debug
```

---

## 🧠 Model Parameters

| Parameter | Value | Notes |
|---|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.3` | 7B params |
| LoRA rank | r=16 | |
| LoRA alpha | 32 | 2×r, standard scaling |
| LoRA target modules | 7 (q/k/v/o/gate/up/down) | Full attention + MLP |
| LoRA dropout | 0.05 | |
| Training dtype | bfloat16 | A100: more stable than fp16 |
| Optimizer | adamw_torch | No page offloading needed on A100 |
| Learning rate | 2e-4 | |
| Scheduler | cosine | warmup_ratio=0.03 |
| Effective batch size | 8 | per_device=1, grad_accum=8 |
| Max epochs | 3 | with early stopping |
| Early stopping | patience=3 @ eval_loss | eval_steps=650 |
| Max sequence length | 512 tokens | MedMCQA prompts avg ~180 tokens |
| Loss masking | DataCollatorForCompletionOnlyLM | loss only on `Answer:` token |

---

## 🔍 Uncertainty Signals

| Signal | Formula | Range | Intuition |
|---|---|---|---|
| Restricted entropy H | −Σ p_i·log(p_i+ε) over A/B/C/D | [0, log4 ≈ 1.386] | High = uncertain |
| Logit gap | p_top1 − p_top2 after softmax | [0, 1] | Low = two options equally likely |
| Hidden-state probe | LogisticRegression on h_L* ∈ ℝ⁴⁰⁹⁶ | [0, 1] | Internal "am I correct?" signal |
| p(True) | P(Yes) / (P(Yes)+P(No)) on "Is this correct?" | [0, 1] | Self-evaluation under verbalization |

Layer L* is selected by 5-fold CV AUROC sweep across layers {8, 16, 24, 30, 31}.

---

## 🔀 Routing Pipeline

```
Features: [H, gap*, probe, p_true]   (* dropped if corr(H, gap) > 0.85)
        │
        ▼
Logistic Regression (trained on 9,307 routing_train)
        │
        ▼
Isotonic Calibration (fitted on 4,000 iso_cal)
        │
        ▼
routing_score ∈ [0, 1]    ← P(model is correct | features)
        │
        ▼
Split Conformal Prediction (calibrated on 4,183 val)
        │
        ▼
q_hat @ α=0.10  →  threshold = 1 − q_hat
        │
        ├── routing_score ≥ threshold  →  Answer locally ✅
        └── routing_score < threshold  →  Escalate 🚨
```

**Formal guarantee:** P(correct answer returned locally) ≥ 1 − α = 0.90

---

## 💻 Hardware Requirements

| Phase | GPU VRAM | Time (estimate) |
|---|---|---|
| Blockers 1–2 (verify) | CPU | ~30 min |
| Fine-tuning FP16 | ≥24 GB (A100) | 1–2 days |
| Fine-tuning QLoRA 4-bit | ≥16 GB (RTX 3090) | 2–3 days |
| Feature extraction (28K samples) | ≥8 GB | 4–8 hours |
| Blocker 3 (p(True) verify) | ≥8 GB | ~1 hour |
| Probe + routing + conformal | CPU | ~2–3 hours |
| Evaluation (3 tiers) | CPU | ~30 min |
| Ablations | CPU + GPU | ~1 day |

---

## 📁 Repository Structure

```
diploma-thesis/
├── CLAUDE.md               # Claude Code instructions
├── roadmap.md              # Full pipeline spec (living document)
├── pyproject.toml          # Package + dependencies
├── Makefile                # make setup / verify / finetune / evaluate
│
├── configs/
│   ├── main.yaml           # Primary config (OmegaConf)
│   ├── finetune_config.yaml
│   ├── model/
│   │   ├── lora_fp16.yaml  # LoRA bfloat16 (full-precision weights, no quantization)
│   │   └── qlora_4bit.yaml # QLoRA 4-bit NF4 fallback
│   └── data/
│       └── datasets.yaml
│
├── scripts/                # Entry points (thin wrappers — no logic here)
│   ├── 00_verify.py        # 3 critical blockers
│   ├── 01_prepare.py       # Data splits + dedup
│   ├── 02_finetune.py      # LoRA fine-tuning
│   ├── 03_extract.py       # Feature extraction + layer sweep
│   ├── 04_probe.py         # 5-fold correctness probe
│   ├── 05_routing.py       # Routing LR + isotonic calibration
│   ├── 06_conformal.py     # Split CP (q_hat)
│   ├── 07_evaluate.py      # 3-tier evaluation
│   └── 08_ablations.py     # Ablation tables
│
├── src/thesis/             # Installable package (all logic here)
│   ├── data/               # Dataset loading, normalize_example()
│   ├── models/             # Model wrappers, extract_features()
│   ├── routing/            # probe.py, routing_lr.py, calibration.py, conformal.py
│   └── utils/              # metrics.py (AUGRC/AURC/ECE/bootstrap), plotting.py
│
├── data/
│   ├── raw/                # Original datasets — never modify (not committed)
│   ├── splits/             # Train/probe/val indices as JSON ✅ committed
│   └── features/           # Extracted .npz caches — not committed (~500 MB)
│
├── checkpoints/            # LoRA adapters — not committed (~150 MB)
├── results/                # Evaluation JSON outputs ✅ committed
└── figures/                # Thesis plots ✅ committed
```

---

## 📈 Expected Results

| Metric | In-dist (MedMCQA) | Near-OOD (MedQA) | Far-OOD (MMLU) |
|---|---|---|---|
| Standalone accuracy | ~65–70% | ~55–65% | ~60–70% |
| AUROC routing | ~0.70–0.80 | ~0.65–0.75 | ~0.60–0.70 |
| Empirical coverage @ α=0.10 | ~0.90 ✅ (guaranteed) | ~0.85–0.90 | ~0.80–0.88 |
| Escalation rate @ α=0.10 | ~15–25% | — | — |
| AUGRC | low (main result) | higher | highest |

**Key finding:** The conformal coverage guarantee holds on the in-distribution test set. Under distribution shift (MedQA → MMLU), empirical coverage degrades gracefully — this thesis quantifies exactly how much the CP guarantee breaks without recalibration.

---

## 📏 Metrics

Primary metric: **AUGRC** (Area Under Generalized Risk-Coverage curve, NeurIPS 2024, arXiv:2407.01032) — measures expected risk of undetected failures across all coverage thresholds. Lower is better.

Secondary: AURC, Brier score, ECE (15 equal-mass bins), AUROC.

All metrics reported with 95% bootstrap CI (1000 resamples, seed=42).

---

## 🔁 Reproducibility

- `seed=42` for **all** random operations: numpy, torch, sklearn, HuggingFace datasets
- `TrainingArguments` uses both `seed=42` and `data_seed=42`
- All data splits stored as JSON indices in `data/splits/` — committed to repo
- IDS_ABCD = `[1098, 1133, 1102, 1152]` (A, B, C, D in Mistral tokenizer) — verified at runtime with `sys.exit` on mismatch

```bash
# Verify token IDs match expectations
python scripts/00_verify.py --blocker 2
```

---

## 📄 License

Code: MIT · MedMCQA: Apache 2.0 · MedQA-USMLE: CC-BY 4.0 · MMLU: MIT
