# Supervisor Guide — Running the GPU Steps

This guide is for the thesis supervisor who runs the fine-tuning and feature extraction on a GPU machine. The student handles everything else (probe training, routing, evaluation) locally on CPU after receiving the output files.

---

## Prerequisites

| Requirement | Details |
|---|---|
| GPU | ≥24 GB VRAM for LoRA FP16 (A100 recommended); ≥16 GB for QLoRA fallback |
| Python | 3.11 |
| `HF_TOKEN` | HuggingFace token — needed to download `mistralai/Mistral-7B-Instruct-v0.3` |
| `WANDB_API_KEY` | Optional — training logs fall back to TensorBoard if not set |

---

## Step-by-Step

```bash
# 1. Clone and install
git clone https://github.com/Oscarski/local-language-model-healthcare.git
cd local-language-model-healthcare
pip install -e ".[dev]"

# 2. Set credentials (or create a .env file with these two lines)
export HF_TOKEN=<your_huggingface_token>
export WANDB_API_KEY=<your_wandb_key>   # optional

# 3. Prepare data splits — CPU only, ~10 minutes, run once
python scripts/01_prepare.py

# 4. Verify encoding and tokenization before training — CPU only, ~1 minute
python scripts/00_verify.py --blocker 1
python scripts/00_verify.py --blocker 2

# 5. Fine-tune — requires ≥24 GB VRAM, ~1-2 days on A100
python scripts/02_finetune.py
# Saves best checkpoint to: checkpoints/final/
# Training logs visible in W&B dashboard or TensorBoard

# 6. Verify p(True) signal on the fine-tuned model — ~1 hour
python scripts/00_verify.py --blocker 3

# 7. Extract features for all splits — requires ≥8 GB VRAM, ~4-8 hours
python scripts/03_extract.py
# Saves features to: data/features/
```

---

## What to Send Back to the Student

Please compress and send the following (~650 MB total):

```
checkpoints/final/               ← LoRA adapter + tokenizer (~150 MB)
data/features/*.npz              ← scalar features (H, gap, p_true, y, pred, subject)
data/features/*_hidden.npy       ← hidden states, shape [N, 4096], float32 (~500 MB)
results/layer_sweep.json         ← which hidden layer performed best
results/extraction_metadata.json ← run statistics (accuracy, entropy mean, etc.)
```

The student runs everything from `04_probe.py` onwards on CPU (~2-4 hours).

---

## If GPU Has Less Than 24 GB VRAM

Use the QLoRA 4-bit fallback:

```bash
python scripts/02_finetune.py model=qlora_4bit
```

This requires ≥16 GB VRAM and takes ~2-3 days on an RTX 3090.

---

## Smoke Test (Verify Setup Without Full Training)

To verify the environment is correctly configured before committing to the full run:

```bash
python scripts/02_finetune.py --debug   # 512 examples, 1 eval step, ~5 minutes on GPU
python scripts/03_extract.py --debug    # 50 examples per split, ~10 minutes on GPU
```

Still requires GPU — `--debug` only reduces dataset size, the full 7B model is still loaded.

---

## Resuming After Interruption

If training is interrupted, resume from the last checkpoint:

```bash
python scripts/02_finetune.py --resume
```

Feature extraction also resumes automatically — already-extracted splits are skipped.

---

## Expected Output After Step 5 (Fine-tuning)

The script logs training progress and prints a summary at the end:

```
KROK 3 DONE
  Best eval_loss:  0.XXXX
  Steps trained:   XXXX
  Early stopped:   True/False
  Checkpoint:      checkpoints/final/
Next: python scripts/00_verify.py --blocker 3
Then: python scripts/03_extract.py
```

W&B run link (if enabled): visible in terminal output after `wandb.init()`.
