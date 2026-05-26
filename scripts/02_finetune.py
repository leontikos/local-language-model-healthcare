"""
KROK 3 — FINE-TUNING
=====================
LoRA fine-tuning of Mistral-7B-Instruct-v0.3 on MedMCQA (single-answer MCQ).
Loss is computed ONLY on the answer token (A/B/C/D) via DataCollatorForCompletionOnlyLM.

Usage:
    # Full run (requires A100 ≥24GB VRAM + HF_TOKEN + WANDB_API_KEY):
    python scripts/02_finetune.py

    # Resume after crash:
    python scripts/02_finetune.py --resume

    # Smoke-test without GPU (512 train + 128 val, 1 eval step):
    python scripts/02_finetune.py --debug

    # Override config values:
    python scripts/02_finetune.py training.learning_rate=1e-4 lora.r=8

Output:
    checkpoints/final/          — best LoRA adapter (load with PeftModel)
    checkpoints/final/finetune_metadata.json — run metadata
    checkpoints/checkpoint-N/   — intermediate checkpoints (save_total_limit=3)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(extra_overrides: list[str], config_name: str = "finetune_config.yaml") -> dict:
    """Load configs/<config_name> and apply CLI key=value overrides."""
    from omegaconf import OmegaConf

    cfg_path = Path(__file__).parent.parent / "configs" / config_name
    cfg = OmegaConf.load(cfg_path)

    for override in extra_overrides:
        if "=" not in override:
            log.warning("Ignoring malformed override (expected key=value): %s", override)
            continue
        key, value = override.split("=", 1)
        OmegaConf.update(cfg, key, value, merge=True)

    return OmegaConf.to_container(cfg, resolve=True)


def _verify_response_template(tokenizer, template: str) -> list[int]:
    """
    Return the token-ID list for response_template.

    DataCollatorForCompletionOnlyLM accepts either a string or a list of token IDs.
    Passing token IDs directly is safer because Mistral's SentencePiece tokenizer
    may produce different encodings depending on context (leading space, BOS, etc.).

    Asserts that the template encodes to 1–3 tokens (sanity check).
    """
    ids = tokenizer.encode(template, add_special_tokens=False)
    # Also try with leading space (how it appears after "[/INST]")
    ids_with_space = tokenizer.encode(" " + template, add_special_tokens=False)
    log.info("Response template '%s' → token IDs: %s", template, ids)
    log.info("Response template ' %s' → token IDs: %s", template, ids_with_space)
    assert 1 <= len(ids_with_space) <= 4, (
        f"Unexpected tokenization of response_template ' {template}': {ids_with_space}. "
        "Check that the template matches what actually appears in formatted prompts."
    )
    # Return IDs for the space-prefixed form — this is how the template appears in
    # the formatted prompt after "[/INST]", so the collator correctly masks the prompt.
    return ids_with_space


def _format_example(row: dict) -> dict:
    """
    Format one MedMCQA row into the unified prompt+answer string.

    Template (Mistral instruction format):
        [INST] ...question + options... [/INST] Answer: {letter}

    Loss is computed ONLY on {letter} (the token after "Answer:").
    cop encoding: 0=A, 1=B, 2=C, 3=D  (verified in BLOCKER 1).
    """
    cop = row["cop"]
    if not (0 <= cop < 4):
        raise ValueError(f"Invalid cop={cop!r} (expected 0–3) in example id={row.get('id', '?')}")
    letter = "ABCD"[cop]
    text = (
        f"[INST] You are answering a medical multiple-choice question. "
        f"Reply with the single letter of the correct option only.\n\n"
        f"Question: {row['question']}\n\n"
        f"A) {row['opa']}\n"
        f"B) {row['opb']}\n"
        f"C) {row['opc']}\n"
        f"D) {row['opd']} [/INST] Answer: {letter}"
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # DDP rank detection — torchrun sets these env vars.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main_process = local_rank == 0

    # Console logging (all ranks) + file logging on rank 0 only.
    log_handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if is_main_process:
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"finetune_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
        log_handlers.append(logging.FileHandler(log_file, mode="w"))

    logging.basicConfig(
        level=logging.INFO if is_main_process else logging.WARNING,
        format=f"%(asctime)s  [rank{local_rank}]  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=log_handlers,
        force=True,
    )
    if is_main_process:
        log.info("Log file: %s", log_file)
        log.info("DDP world_size=%d  local_rank=%d", world_size, local_rank)

    # ------------------------------------------------------------------
    # 1. Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="LoRA fine-tune Mistral-7B on MedMCQA")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint in checkpoint_dir")
    parser.add_argument("--debug", action="store_true",
                        help="Smoke-test mode: small dataset, 1 eval step, no GPU needed")
    parser.add_argument("--config", default="finetune_config.yaml",
                        help="YAML file under configs/ (e.g. finetune_config_ddp.yaml)")
    # Catch key=value overrides (e.g. training.learning_rate=1e-4)
    parser.add_argument("overrides", nargs="*", help="OmegaConf-style key=value overrides")
    args = parser.parse_args()

    cfg = _load_config(args.overrides, config_name=args.config)
    seed: int = cfg["seed"]
    if is_main_process:
        log.info("Loaded config: configs/%s", args.config)

    # Guard: load_best_model_at_end=True requires save_steps == eval_steps
    assert cfg["training"]["eval_steps"] == cfg["training"]["save_steps"], (
        f"eval_steps ({cfg['training']['eval_steps']}) must equal "
        f"save_steps ({cfg['training']['save_steps']}) "
        "when load_best_model_at_end=True — override both together."
    )

    # Apply debug overrides
    if args.debug:
        log.info("DEBUG MODE — small dataset, fast eval, no GPU required")
        cfg["training"]["num_train_epochs"] = cfg["debug"]["num_train_epochs"]
        cfg["training"]["eval_steps"] = cfg["debug"]["eval_steps"]
        cfg["training"]["save_steps"] = cfg["debug"]["eval_steps"]
        cfg["training"]["logging_steps"] = 5
        # Without these, 512 examples / batch 2 / grad_accum 8 = only 32 steps total
        # → model barely trains. In debug we want fast full forward/backward passes.
        cfg["training"]["gradient_accumulation_steps"] = 1
        cfg["training"]["per_device_eval_batch_size"] = 16

    # ------------------------------------------------------------------
    # 2. Late imports (keep startup fast for --help)
    # ------------------------------------------------------------------
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        TrainingArguments,
        set_seed,
    )
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

    set_seed(seed)

    # ------------------------------------------------------------------
    # 3. W&B setup (graceful fallback to tensorboard) — rank 0 only
    # ------------------------------------------------------------------
    report_to: list[str]
    if is_main_process and os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"]["run_name"],
            entity=cfg["wandb"].get("entity") or None,
            config={
                "lora": cfg["lora"],
                "training": cfg["training"],
                "model": cfg["model"],
                "seed": seed,
                "debug": args.debug,
            },
        )
        report_to = ["wandb"]
        log.info("W&B enabled — project: %s / run: %s",
                 cfg["wandb"]["project"], cfg["wandb"]["run_name"])
    else:
        if is_main_process:
            log.info("Logging to TensorBoard (WANDB_API_KEY not set)")
        report_to = ["tensorboard"]

    # ------------------------------------------------------------------
    # 4. Tokenizer
    # ------------------------------------------------------------------
    model_name: str = cfg["model"]["model_name_or_path"]
    log.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required by DataCollatorForCompletionOnlyLM

    response_template_ids = _verify_response_template(
        tokenizer, cfg["data"]["response_template"]
    )

    # ------------------------------------------------------------------
    # 5. Model
    # ------------------------------------------------------------------
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[cfg["model"]["torch_dtype"]]

    # Log available VRAM so professor can diagnose OOM issues
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU: %s  (%.1f GB VRAM)", gpu_name, total_gb)
        if total_gb < 24:
            log.warning(
                "< 24 GB VRAM detected (%.1f GB). "
                "Consider reducing per_device_eval_batch_size or enabling gradient_checkpointing.",
                total_gb,
            )
    else:
        log.warning("No CUDA GPU detected — training will run on CPU (very slow)")

    log.info("Loading model: %s  dtype=%s", model_name, cfg["model"]["torch_dtype"])
    _attn_impl = cfg["model"]["attn_implementation"]
    # Under DDP each rank owns one full model copy on its own GPU; "auto" would
    # shard layers across GPUs and break DDP. With world_size=1 we keep "auto".
    _device_map = {"": local_rank} if world_size > 1 else "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=_device_map,
            attn_implementation=_attn_impl,
        )
        log.info("Attention implementation: %s  device_map=%s", _attn_impl, _device_map)
    except (ImportError, ValueError) as _fa2_err:
        if "flash" in str(_fa2_err).lower():
            log.warning(
                "Flash Attention 2 not available (%s). Falling back to eager. "
                "Install with: pip install flash-attn --no-build-isolation",
                _fa2_err,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=_device_map,
                attn_implementation="eager",
            )
        else:
            raise
    # Trainer sets use_cache=False automatically when gradient_checkpointing=True,
    # but setting it explicitly here suppresses the warning regardless.
    model.config.use_cache = False

    # ------------------------------------------------------------------
    # 6. LoRA
    # ------------------------------------------------------------------
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 7. Datasets
    # ------------------------------------------------------------------
    log.info("Loading MedMCQA …")
    raw = load_dataset(cfg["data"]["dataset_id"])
    import numpy as _np

    # Filter to single-answer only (consistent with data/splits/ generation)
    train_raw_filtered = raw["train"].filter(lambda x: x["choice_type"] == "single")

    # Use pre-generated split indices if available.
    # Always anchor to project root so the script works regardless of cwd.
    _project_root = Path(__file__).parent.parent
    indices_path = _project_root / cfg["data"]["train_indices_json"]
    if indices_path.exists():
        indices = json.loads(indices_path.read_text())

        # Carve 2K fine-tune eval set from train_ft (deterministic, seed=42).
        # Conformal prediction theory requires the calibration set (MedMCQA val, 4183)
        # to be INDEPENDENT of model selection. Using val for early stopping would
        # mean the checkpoint was selected based on val → weakened CP guarantee.
        # Solution: use a held-out subset of train_ft for early stopping instead.
        _rng = _np.random.default_rng(42)
        _shuffled = _rng.permutation(len(indices)).tolist()
        _ft_val_n = min(2000, len(indices) // 10)  # 2K or 10%, whichever is smaller
        _ft_val_local  = _shuffled[:_ft_val_n]
        _ft_train_local = _shuffled[_ft_val_n:]

        _ft_val_global  = [indices[i] for i in _ft_val_local]
        _ft_train_global = [indices[i] for i in _ft_train_local]

        train_ds = train_raw_filtered.select(_ft_train_global)
        val_ds   = train_raw_filtered.select(_ft_val_global)
        log.info(
            "train_ft: %d examples  ft_val: %d examples (carved from train_ft, seed=42)",
            len(train_ds), len(val_ds),
        )
        log.info(
            "MedMCQA val (4183) → reserved ONLY for conformal calibration, not used here."
        )
    else:
        log.warning(
            "train_ft_idx.json not found at %s — using full filtered train split. "
            "Run scripts/01_prepare.py first for proper train/probe isolation.",
            indices_path,
        )
        train_ds = train_raw_filtered
        # Fallback: use MedMCQA val. Note this weakens the formal CP guarantee.
        val_ds = raw["validation"]
        log.warning(
            "FALLBACK: using MedMCQA val for early stopping — "
            "this weakens conformal prediction guarantee. Run 01_prepare.py first."
        )

    # Debug: truncate to tiny subsets
    if args.debug:
        train_ds = train_ds.select(range(min(cfg["debug"]["train_samples"], len(train_ds))))
        val_ds = val_ds.select(range(min(cfg["debug"]["val_samples"], len(val_ds))))
        log.info("DEBUG: train=%d  val=%d", len(train_ds), len(val_ds))
    else:
        log.info("Dataset sizes — train: %d  val: %d", len(train_ds), len(val_ds))

    # Format to text
    train_ds = train_ds.map(_format_example, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(_format_example, remove_columns=val_ds.column_names)

    # ------------------------------------------------------------------
    # 8. Data collator
    # ------------------------------------------------------------------
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    # ------------------------------------------------------------------
    # 9. TrainingArguments
    # ------------------------------------------------------------------
    t = cfg["training"]
    # Anchor all output paths to project root — safe regardless of cwd
    checkpoint_dir = _project_root / cfg["paths"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # gradient_checkpointing + LoRA + DDP requires non-reentrant checkpoint
    # otherwise DDP "marked as ready twice" errors fire on backward.
    _gc_kwargs = {"use_reentrant": False} if t["gradient_checkpointing"] else None

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],

        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        optim=t["optim"],

        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],         # MUST equal eval_steps
        save_total_limit=t["save_total_limit"],
        save_only_model=t.get("save_only_model", False),  # drop optimizer state from intermediates
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],   # "eval_loss"
        greater_is_better=t["greater_is_better"],           # False (minimize loss)

        bf16=t["bf16"],
        fp16=t["fp16"],                     # explicitly False — avoids bf16/fp16 conflict
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs=_gc_kwargs,

        dataloader_num_workers=t["dataloader_num_workers"],
        group_by_length=t["group_by_length"],

        # DDP knobs (only relevant when world_size > 1; harmless otherwise)
        ddp_find_unused_parameters=t.get("ddp_find_unused_parameters", False),
        ddp_bucket_cap_mb=t.get("ddp_bucket_cap_mb", 25),

        logging_steps=t["logging_steps"],
        report_to=report_to,
        run_name=cfg["wandb"]["run_name"],
        disable_tqdm=not is_main_process,   # progress bar only on rank 0

        seed=seed,
        data_seed=seed,
        remove_unused_columns=True,
    )

    # ------------------------------------------------------------------
    # 10. Callbacks
    # ------------------------------------------------------------------
    # EarlyStoppingCallback uses eval_loss (works out-of-the-box with SFTTrainer).
    # SFTTrainer GitHub #1222: custom compute_metrics CANNOT be used with
    # EarlyStoppingCallback in SFTTrainer — that's why we use eval_loss here.
    # mean_token_accuracy is logged automatically by SFTTrainer and visible in W&B.

    class MinStepsEarlyStoppingCallback(EarlyStoppingCallback):
        """EarlyStoppingCallback that ignores eval results until min_steps have passed.

        Prevents premature stopping during LR warmup when eval_loss is intrinsically
        noisy. After min_steps, behaves exactly like the parent class.
        """
        def __init__(self, min_steps: int, **kwargs):
            super().__init__(**kwargs)
            self.min_steps = min_steps

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            if state.global_step < self.min_steps:
                if is_main_process and state.global_step > 0:
                    log.info(
                        "Skipping early-stop check at step %d (< min_steps %d, ~%.1f epoch warmup)",
                        state.global_step, self.min_steps, self.min_steps / max(state.max_steps / t["num_train_epochs"], 1),
                    )
                return
            return super().on_evaluate(args, state, control, metrics, **kwargs)

    # Compute the no-stop window: at least min_train_epochs_for_early_stop full epochs.
    # steps_per_epoch is roughly len(train) / effective_batch_size.
    _eff_batch = (
        t["per_device_train_batch_size"]
        * t["gradient_accumulation_steps"]
        * max(world_size, 1)
    )
    _steps_per_epoch = max(len(train_ds) // _eff_batch, 1)
    _min_train_epochs = float(t.get("min_train_epochs_for_early_stop", 0.0))
    _min_steps_no_stop = int(_min_train_epochs * _steps_per_epoch)

    if is_main_process:
        log.info(
            "Early-stopping plan: patience=%d evals × eval_steps=%d (window = %d steps); "
            "no-stop floor = %d steps (≥ %.2f epochs of training)",
            t["early_stopping_patience"], t["eval_steps"],
            t["early_stopping_patience"] * t["eval_steps"],
            _min_steps_no_stop, _min_train_epochs,
        )

    callbacks = [
        MinStepsEarlyStoppingCallback(
            min_steps=_min_steps_no_stop,
            early_stopping_patience=t["early_stopping_patience"],
        )
    ]

    # ------------------------------------------------------------------
    # 11. SFTTrainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=cfg["data"]["max_seq_length"],
        callbacks=callbacks,
        peft_config=None,   # LoRA already applied above via get_peft_model
    )

    # ------------------------------------------------------------------
    # 12. Train
    # ------------------------------------------------------------------
    has_checkpoints = checkpoint_dir.exists() and any(checkpoint_dir.glob("checkpoint-*"))
    resume = str(checkpoint_dir) if args.resume and has_checkpoints else None
    if resume:
        log.info("Resuming from checkpoint in %s", checkpoint_dir)

    log.info("Starting training …")
    train_result = trainer.train(resume_from_checkpoint=resume)

    # ------------------------------------------------------------------
    # 13. Save best checkpoint to checkpoints/final/
    # ------------------------------------------------------------------
    final_dir = _project_root / cfg["paths"]["final_checkpoint_dir"]
    final_dir.mkdir(parents=True, exist_ok=True)

    log.info("Saving best model to %s …", final_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # ------------------------------------------------------------------
    # 14. Write run metadata
    # ------------------------------------------------------------------
    # Detect whether early stopping fired.
    # max_steps mirrors what Trainer would compute internally.
    steps_per_epoch = len(train_ds) // (
        t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
    )
    max_steps = int(t["num_train_epochs"]) * steps_per_epoch
    early_stopped = trainer.state.global_step < max_steps

    metadata = {
        "model": model_name,
        "lora_r": lora_cfg["r"],
        "lora_alpha": lora_cfg["lora_alpha"],
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "steps_trained": trainer.state.global_step,
        "best_eval_loss": trainer.state.best_metric,
        "early_stopped": early_stopped,
        "early_stopping_patience": t["early_stopping_patience"],
        "eval_steps": t["eval_steps"],
        "epochs_config": t["num_train_epochs"],
        "bf16": t["bf16"],
        "debug_mode": args.debug,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "train_runtime_sec": train_result.metrics.get("train_runtime"),
    }

    meta_path = final_dir / "finetune_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("Metadata saved to %s", meta_path)

    log.info("=" * 60)
    log.info("KROK 3 DONE")
    log.info("  Best eval_loss:  %.4f", trainer.state.best_metric or float("nan"))
    log.info("  Steps trained:   %d", trainer.state.global_step)
    log.info("  Early stopped:   %s", early_stopped)
    log.info("  Checkpoint:      %s", final_dir)
    log.info("=" * 60)
    log.info("Next: python scripts/00_verify.py --blocker 3")
    log.info("Then: python scripts/03_extract.py")


if __name__ == "__main__":
    main()
