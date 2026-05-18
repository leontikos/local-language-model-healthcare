"""
KROK 4 — FEATURE EXTRACTION
============================
Extracts uncertainty features and hidden states from the fine-tuned model
for all 5 splits used downstream (probe, val, 3 test sets).

Features extracted per example:
  H       — restricted entropy over A/B/C/D logits  ∈ [0, log4 ≈ 1.386]
  gap     — logit margin (top-1 minus top-2 probability) ∈ [0, 1]
  p_true  — P(model is correct) from "Is this answer correct? (Yes/No)"
  y       — correctness label (1 = correct, 0 = wrong)
  pred    — predicted letter (A/B/C/D)
  true    — true label letter
  subject — subject_name or "unknown"

Hidden states (large, saved separately):
  data/features/{split}_hidden.npy  — shape [N, 4096], float32

Usage:
    # Full run — requires A100 (or any GPU ≥ 8 GB), ~1 h
    python scripts/03_extract.py

    # Resume (skips splits already saved)
    python scripts/03_extract.py

    # Smoke test on CPU / small GPU — 50 examples per split
    python scripts/03_extract.py --debug

    # Skip p(True) if BLOCKER 3 showed mean(Yes+No mass) < 0.30
    python scripts/03_extract.py --skip-ptrue

    # Use a specific layer (skip sweep)
    python scripts/03_extract.py --layer 31

    # Extract only selected splits
    python scripts/03_extract.py --splits probe val

Output:
    data/features/probe_features.npz        (13 307 examples)
    data/features/val_features.npz          ( 4 183 examples)
    data/features/test_medmcqa_features.npz ( 6 150 examples)
    data/features/test_medqa_features.npz   ( 1 273 examples)
    data/features/test_mmlu_features.npz    (   945 examples)
    data/features/{split}_hidden.npy        (corresponding hidden states)
    results/layer_sweep.json                (layer sweep results)
    results/best_layer.json                 (best layer index)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

# Token IDs verified in BLOCKER 2 ✅:  ' A'=1098, ' B'=1133, ' C'=1102, ' D'=1152
# Both ' A' and 'A' give the same ID in Mistral's SentencePiece vocab.
IDS_ABCD: list[int] = [1098, 1133, 1102, 1152]

# Mistral-7B has 32 transformer layers; hidden_states[i] = output after layer i
# (hidden_states[0] = embedding output).
SWEEP_LAYERS: list[int] = [8, 16, 24, 30, 31]  # L//4, L//2, 3L//4, L-2, L-1

ALL_SPLITS = ["probe", "val", "test_medmcqa", "test_medqa", "test_mmlu"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    from omegaconf import OmegaConf
    cfg_path = ROOT / "configs" / "finetune_config.yaml"
    return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)


# ---------------------------------------------------------------------------
# Prompt formatting (must match 02_finetune.py _format_example exactly,
# minus the answer token — so logits[-1] = distribution over next token)
# ---------------------------------------------------------------------------

def _format_inference_prompt(question: str, opts: dict[str, str]) -> str:
    return (
        f"[INST] You are answering a medical multiple-choice question. "
        f"Reply with the single letter of the correct option only.\n\n"
        f"Question: {question}\n\n"
        f"A) {opts['A']}\n"
        f"B) {opts['B']}\n"
        f"C) {opts['C']}\n"
        f"D) {opts['D']} [/INST] Answer:"
    )


# ---------------------------------------------------------------------------
# Dataset normalizers
# ---------------------------------------------------------------------------

def _normalize(example: dict, source: str) -> tuple[str, dict[str, str], str, str]:
    """Return (question, opts, label_letter, subject)."""
    if source == "medmcqa":
        cop = example["cop"]
        if not (0 <= cop < 4):
            raise ValueError(f"Invalid cop={cop!r} in id={example.get('id', '?')}")
        label = "ABCD"[cop]
        opts = {
            "A": example["opa"], "B": example["opb"],
            "C": example["opc"], "D": example["opd"],
        }
        return example["question"], opts, label, example.get("subject_name", "unknown")

    elif source == "medqa":
        # GBaker/MedQA-USMLE-4-options: options is dict {"A":str,...}, answer_idx is letter
        opts = example["options"]
        label = str(example["answer_idx"])
        return example["question"], opts, label, "unknown"

    elif source == "mmlu":
        # cais/mmlu: choices is list[4], answer is int (ClassLabel 0-3)
        letters = ["A", "B", "C", "D"]
        opts = {l: example["choices"][i] for i, l in enumerate(letters)}
        ans = example["answer"]
        label = letters[int(ans)] if isinstance(ans, int) else str(ans)
        return example["question"], opts, label, example.get("subject", "unknown")

    raise ValueError(f"Unknown source: {source!r}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(cfg: dict, debug: bool):
    """Load fine-tuned LoRA adapter, merge weights, return (model, tokenizer)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    final_dir = ROOT / cfg["paths"]["final_checkpoint_dir"]
    if not final_dir.exists():
        log.error(
            "Checkpoint not found: %s\n"
            "  → Run scripts/02_finetune.py first, then retry.",
            final_dir,
        )
        sys.exit(1)

    model_name: str = cfg["model"]["model_name_or_path"]
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[cfg["model"]["torch_dtype"]]

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU: %s  (%.1f GB VRAM)", gpu_name, total_gb)
        if total_gb < 8:
            log.warning("< 8 GB VRAM — extraction may OOM; consider --debug first")
    else:
        if not debug:
            log.warning("No CUDA GPU detected — extraction will be very slow on CPU")

    log.info("Loading tokenizer from %s", final_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(final_dir), use_fast=True)

    # Verify hardcoded token IDs match the loaded tokenizer.
    # If the tokenizer version changed, these IDs could silently produce wrong features.
    log.info("Verifying token IDs A/B/C/D …")
    for i, letter in enumerate(["A", "B", "C", "D"]):
        actual = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(actual) != 1:
            log.error(
                "Letter ' %s' tokenizes to %d tokens: %s (expected 1). "
                "Restricted entropy cannot be computed. Update IDS_ABCD.",
                letter, len(actual), actual,
            )
            sys.exit(1)
        if actual[0] != IDS_ABCD[i]:
            log.error(
                "Token ID mismatch for ' %s': hardcoded=%d, tokenizer=%d. "
                "Tokenizer version mismatch — update IDS_ABCD in this script.",
                letter, IDS_ABCD[i], actual[0],
            )
            sys.exit(1)
    log.info("  ✓ Token IDs verified: A=%d  B=%d  C=%d  D=%d", *IDS_ABCD)

    log.info("Loading base model: %s  dtype=%s", model_name, cfg["model"]["torch_dtype"])
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=cfg["model"]["attn_implementation"],
    )

    log.info("Loading LoRA adapter from %s", final_dir)
    model = PeftModel.from_pretrained(base_model, str(final_dir))

    log.info("Merging LoRA weights into base model (faster inference) …")
    model = model.merge_and_unload()
    model.config.use_cache = True   # safe after merge; faster decoding
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Layer sweep
# ---------------------------------------------------------------------------

def _run_layer_sweep(
    model,
    tokenizer,
    probe_ds,
    n_samples: int = 2000,
    n_folds: int = 5,
    seed: int = 42,
) -> int:
    """
    Extract hidden states at SWEEP_LAYERS for n_samples probe examples.
    Run n_folds-fold cross-validated LogisticRegression per layer.
    Returns the layer index with the highest mean AUROC.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm

    log.info(
        "Layer sweep: %d samples, %d-fold CV, layers=%s",
        n_samples, n_folds, SWEEP_LAYERS,
    )

    rng = np.random.default_rng(seed)
    n_available = len(probe_ds)
    indices = rng.choice(n_available, size=min(n_samples, n_available), replace=False)

    # Collect hidden states at all sweep layers in one pass per example
    hidden_by_layer: dict[int, list] = {l: [] for l in SWEEP_LAYERS}
    y_list: list[int] = []

    with torch.inference_mode():
        for pos, raw_idx in enumerate(
            tqdm(indices.tolist(), desc="Sweep forward passes")
        ):
            ex = probe_ds[int(raw_idx)]
            try:
                question, opts, label, _ = _normalize(ex, "medmcqa")
            except ValueError as e:
                log.warning("  Skipping example %d in sweep: %s", raw_idx, e)
                continue

            prompt = _format_inference_prompt(question, opts)
            input_ids = tokenizer.encode(
                prompt, return_tensors="pt"
            ).to(model.device)

            outputs = model(input_ids, output_hidden_states=True)

            # Correctness label
            logits_abcd = outputs.logits[0, -1, IDS_ABCD]
            pred_idx = int(F.softmax(logits_abcd, dim=-1).argmax().item())
            y_list.append(int("ABCD"[pred_idx] == label))

            for layer_idx in SWEEP_LAYERS:
                h = outputs.hidden_states[layer_idx][0, -1, :].cpu().float().numpy()
                hidden_by_layer[layer_idx].append(h)

            del outputs
            if (pos + 1) % 200 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    y_arr = np.array(y_list, dtype=np.int32)
    n = len(y_arr)
    log.info(
        "  Sweep samples collected: %d  (correct=%d  wrong=%d  acc=%.3f)",
        n, y_arr.sum(), n - y_arr.sum(), y_arr.mean(),
    )

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    layer_results: dict[int, dict] = {}

    for layer_idx in SWEEP_LAYERS:
        H_mat = np.stack(hidden_by_layer[layer_idx])   # [n, 4096]
        fold_aurocs: list[float] = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(H_mat, y_arr)):
            lr = LogisticRegression(
                C=1.0, penalty="l2", max_iter=500,
                solver="lbfgs", n_jobs=-1, random_state=seed,
            )
            lr.fit(H_mat[tr_idx], y_arr[tr_idx])
            proba = lr.predict_proba(H_mat[va_idx])[:, 1]
            fold_aurocs.append(float(roc_auc_score(y_arr[va_idx], proba)))

        mean_auroc = float(np.mean(fold_aurocs))
        std_auroc = float(np.std(fold_aurocs))
        log.info("  Layer %2d: AUROC = %.4f ± %.4f", layer_idx, mean_auroc, std_auroc)
        layer_results[layer_idx] = {
            "mean_auroc": round(mean_auroc, 5),
            "std_auroc": round(std_auroc, 5),
            "fold_aurocs": [round(a, 5) for a in fold_aurocs],
        }
        del H_mat

    best_layer = max(layer_results, key=lambda l: layer_results[l]["mean_auroc"])
    log.info(
        "Best layer: %d  (AUROC = %.4f ± %.4f)",
        best_layer,
        layer_results[best_layer]["mean_auroc"],
        layer_results[best_layer]["std_auroc"],
    )

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    sweep_path = results_dir / "layer_sweep.json"
    sweep_path.write_text(json.dumps({
        "best_layer": best_layer,
        "n_samples": n,
        "n_folds": n_folds,
        "sweep_layers": SWEEP_LAYERS,
        "layers": {str(l): v for l, v in layer_results.items()},
    }, indent=2))
    log.info("Layer sweep results saved to %s", sweep_path)

    return best_layer


# ---------------------------------------------------------------------------
# Per-split extraction
# ---------------------------------------------------------------------------

def _extract_split(
    model,
    tokenizer,
    examples,
    source: str,
    best_layer: int,
    yes_id: int,
    no_id: int,
    skip_ptrue: bool,
    debug_n: int | None = None,
) -> dict[str, list]:
    """Extract H, gap, p_true, y, pred, true, subject, hidden for every example."""
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    if debug_n is not None:
        examples = examples.select(range(min(debug_n, len(examples))))
        log.info("  DEBUG: truncated to %d examples", len(examples))

    n = len(examples)
    log.info(
        "  %d examples  source=%s  layer=%d  skip_ptrue=%s",
        n, source, best_layer, skip_ptrue,
    )

    H_list:       list[float] = []
    gap_list:     list[float] = []
    p_true_list:  list[float] = []
    y_list:       list[int]   = []
    pred_list:    list[str]   = []
    true_list:    list[str]   = []
    subject_list: list[str]   = []
    hidden_list:  list        = []

    with torch.inference_mode():
        for pos, ex in enumerate(tqdm(examples, desc=f"  {source}")):
            try:
                question, opts, label, subject = _normalize(ex, source)
            except (ValueError, KeyError) as e:
                log.warning("  Skipping example %d: %s", pos, e)
                continue

            prompt = _format_inference_prompt(question, opts)
            input_ids = tokenizer.encode(
                prompt, return_tensors="pt"
            ).to(model.device)

            outputs = model(input_ids, output_hidden_states=True)
            logits_last = outputs.logits[0, -1, :]
            p = F.softmax(logits_last[IDS_ABCD], dim=-1)

            H   = -(p * torch.log(p + 1e-10)).sum().item()
            sorted_p = p.sort(descending=True).values
            gap = (sorted_p[0] - sorted_p[1]).item()

            pred_idx    = int(p.argmax().item())
            pred_letter = "ABCD"[pred_idx]
            y           = int(pred_letter == label)

            h = outputs.hidden_states[best_layer][0, -1, :].cpu().float().numpy()
            del outputs

            # p(True) — P(Yes) from "Is this answer correct? (Yes/No):"
            if skip_ptrue:
                p_true = float("nan")
            else:
                pt_prompt = (
                    prompt + f" {pred_letter}\n"
                    "Is this answer correct? (Yes/No):"
                )
                pt_ids = tokenizer.encode(
                    pt_prompt, return_tensors="pt"
                ).to(model.device)
                pt_logits = model(pt_ids).logits[0, -1, :]
                p_yn   = F.softmax(pt_logits[[yes_id, no_id]], dim=-1)
                p_true = float(p_yn[0].item())
                del pt_ids, pt_logits

            H_list.append(float(H))
            gap_list.append(float(gap))
            p_true_list.append(p_true)
            y_list.append(y)
            pred_list.append(pred_letter)
            true_list.append(label)
            subject_list.append(subject)
            hidden_list.append(h)

            if (pos + 1) % 500 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log.info("  … %d / %d  (running acc=%.3f)",
                         pos + 1, n, sum(y_list) / len(y_list))

    return {
        "H":       H_list,
        "gap":     gap_list,
        "p_true":  p_true_list,
        "y":       y_list,
        "pred":    pred_list,
        "true":    true_list,
        "subject": subject_list,
        "hidden":  hidden_list,
    }


def _save_split(features: dict, name: str, features_dir: Path) -> None:
    import numpy as np

    features_dir.mkdir(parents=True, exist_ok=True)
    n = len(features["y"])

    # Scalar / string features → compressed npz.
    # String arrays use fixed-length unicode dtype so downstream scripts can load
    # with allow_pickle=False (safer, faster).
    npz_path = features_dir / f"{name}_features.npz"
    np.savez_compressed(
        npz_path,
        H=np.array(features["H"],          dtype=np.float32),
        gap=np.array(features["gap"],       dtype=np.float32),
        p_true=np.array(features["p_true"], dtype=np.float32),
        y=np.array(features["y"],           dtype=np.int32),
        pred=np.array(features["pred"],     dtype="U1"),   # single letter A/B/C/D
        true=np.array(features["true"],     dtype="U1"),   # single letter A/B/C/D
        subject=np.array(features["subject"], dtype="U50"),  # up to 50-char string
    )
    log.info("  Saved: %s  (%d examples)", npz_path, n)

    # Hidden states → uncompressed npy (faster repeated loading in downstream scripts)
    npy_path = features_dir / f"{name}_hidden.npy"
    hidden_arr = np.stack(features["hidden"]).astype(np.float32)
    np.save(npy_path, hidden_arr)
    size_mb = hidden_arr.nbytes / 1e6
    log.info("  Saved: %s  shape=%s  (%.0f MB)", npy_path, hidden_arr.shape, size_mb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature extraction — MedMCQA + MedQA + MMLU"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Extract only 50 examples per split (fast smoke-test, CPU-compatible)",
    )
    parser.add_argument(
        "--skip-ptrue", action="store_true",
        help="Skip p(True) extraction (use if BLOCKER 3 showed mean(Yes+No mass) < 0.30)",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=ALL_SPLITS, default=None,
        metavar="SPLIT",
        help="Splits to extract (default: all). Existing outputs are skipped automatically.",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Use this hidden layer directly; skip the layer sweep.",
    )
    args = parser.parse_args()

    import datasets as hf_datasets

    cfg         = _load_config()
    features_dir = ROOT / "data" / "features"
    splits_dir   = ROOT / "data" / "splits"
    results_dir  = ROOT / "results"
    splits_to_run = args.splits or ALL_SPLITS
    debug_n       = 50 if args.debug else None

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model, tokenizer = _load_model(cfg, debug=args.debug)

    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[-1]
    no_id  = tokenizer.encode(" No",  add_special_tokens=False)[-1]
    log.info("yes_id=%d  no_id=%d  skip_ptrue=%s", yes_id, no_id, args.skip_ptrue)

    # ------------------------------------------------------------------
    # 2. Load MedMCQA (needed for probe split + layer sweep)
    # ------------------------------------------------------------------
    log.info("Loading MedMCQA dataset …")
    medmcqa = hf_datasets.load_dataset(cfg["data"]["dataset_id"])
    train_filtered = medmcqa["train"].filter(
        lambda x: x["choice_type"] == "single", num_proc=4
    )

    probe_idx_path = splits_dir / "probe_idx.json"
    if not probe_idx_path.exists():
        log.error(
            "probe_idx.json not found at %s — run scripts/01_prepare.py first",
            probe_idx_path,
        )
        sys.exit(1)
    probe_idx = json.loads(probe_idx_path.read_text())
    probe_ds  = train_filtered.select(probe_idx)
    log.info("Probe set loaded: %d examples", len(probe_ds))

    # ------------------------------------------------------------------
    # 3. Determine best hidden layer
    # ------------------------------------------------------------------
    if args.layer is not None:
        best_layer = args.layer
        log.info("Using --layer %d (sweep skipped)", best_layer)
    elif args.debug:
        best_layer = 31
        log.info("DEBUG mode — using layer 31 (sweep skipped)")
    else:
        sweep_path = results_dir / "layer_sweep.json"
        if sweep_path.exists():
            sweep_data = json.loads(sweep_path.read_text())
            best_layer = int(sweep_data["best_layer"])
            log.info(
                "Layer sweep loaded from cache: best_layer=%d  (AUROC=%.4f)",
                best_layer,
                sweep_data["layers"][str(best_layer)]["mean_auroc"],
            )
        else:
            best_layer = _run_layer_sweep(
                model, tokenizer, probe_ds,
                n_samples=2000, n_folds=5,
            )

    results_dir.mkdir(exist_ok=True)
    (results_dir / "best_layer.json").write_text(
        json.dumps({"best_layer": best_layer}, indent=2)
    )

    # ------------------------------------------------------------------
    # 4. Extract each split
    # ------------------------------------------------------------------
    for split_name in splits_to_run:
        out_npz = features_dir / f"{split_name}_features.npz"
        out_npy = features_dir / f"{split_name}_hidden.npy"

        # Resume: skip if both files already exist (unless debug mode)
        if out_npz.exists() and out_npy.exists() and not args.debug:
            log.info("Skipping %s — already extracted: %s", split_name, out_npz)
            continue

        log.info("\n%s\nExtracting: %s\n%s", "=" * 60, split_name, "=" * 60)

        if split_name == "probe":
            examples, source = probe_ds, "medmcqa"

        elif split_name == "val":
            examples, source = medmcqa["validation"], "medmcqa"

        elif split_name == "test_medmcqa":
            examples, source = medmcqa["test"], "medmcqa"

        elif split_name == "test_medqa":
            log.info("Loading GBaker/MedQA-USMLE-4-options test …")
            examples = hf_datasets.load_dataset(
                "GBaker/MedQA-USMLE-4-options", split="test"
            )
            source = "medqa"

        elif split_name == "test_mmlu":
            log.info("Loading MMLU (5 medical categories) test …")
            mmlu_cats = [
                "clinical_knowledge", "professional_medicine",
                "college_medicine", "medical_genetics", "anatomy",
            ]
            parts = [
                hf_datasets.load_dataset("cais/mmlu", cat, split="test")
                for cat in mmlu_cats
            ]
            examples = hf_datasets.concatenate_datasets(parts)
            source = "mmlu"

        else:
            log.error("Unknown split: %s", split_name)
            continue

        features = _extract_split(
            model, tokenizer, examples, source,
            best_layer=best_layer,
            yes_id=yes_id,
            no_id=no_id,
            skip_ptrue=args.skip_ptrue,
            debug_n=debug_n,
        )
        _save_split(features, split_name, features_dir)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    import numpy as np

    print("\n" + "=" * 60)
    print("KROK 4 DONE — feature extraction complete")
    print(f"  Best layer:   {best_layer}")
    print(f"  Features dir: {features_dir}")
    print(f"\n  {'Split':<18} {'n':>6}  {'acc':>6}  {'H mean':>8}  {'gap mean':>9}")
    print(f"  {'-'*52}")
    for split_name in splits_to_run:
        npz_path = features_dir / f"{split_name}_features.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=False)
        n   = len(d["y"])
        acc = float(d["y"].astype(float).mean())
        h_mean   = float(d["H"].mean())
        gap_mean = float(d["gap"].mean())
        print(f"  {split_name:<18} {n:>6,}  {acc:>6.3f}  {h_mean:>8.4f}  {gap_mean:>9.4f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Sanity checks on extracted features
    # ------------------------------------------------------------------
    log.info("Sanity checks on extracted features …")
    # Expected accuracy ranges from roadmap Section 10
    _expected_acc = {
        "probe":        (0.55, 0.80),
        "val":          (0.55, 0.80),
        "test_medmcqa": (0.55, 0.80),
        "test_medqa":   (0.45, 0.75),
        "test_mmlu":    (0.45, 0.75),
    }
    all_ok = True
    for split_name in splits_to_run:
        npz_path = features_dir / f"{split_name}_features.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=False)
        acc = float(d["y"].mean())
        lo, hi = _expected_acc.get(split_name, (0.3, 0.9))
        if not (lo <= acc <= hi):
            log.warning(
                "  ⚠ %s accuracy=%.3f outside expected [%.2f, %.2f] — "
                "check model checkpoint or data pipeline",
                split_name, acc, lo, hi,
            )
            all_ok = False
        assert float(d["H"].min()) >= 0.0, f"{split_name}: H < 0"
        assert float(d["H"].max()) <= 1.40, f"{split_name}: H > log(4)+ε"
        assert float(d["gap"].min()) >= 0.0, f"{split_name}: gap < 0"
        assert float(d["gap"].max()) <= 1.01, f"{split_name}: gap > 1"
        if not args.skip_ptrue:
            valid = ~np.isnan(d["p_true"])
            if valid.sum() > 0:
                mean_mass = float(d["p_true"][valid].mean())
                log.info(
                    "  %s: p_true mean=%.3f  (BLOCKER 3 threshold: ≥0.30 to use signal)",
                    split_name, mean_mass,
                )
                if mean_mass < 0.30 and split_name == "probe":
                    log.warning(
                        "  ⚠ p_true mean(%.3f) < 0.30 — "
                        "exclude p_true from routing features (re-run with --skip-ptrue).",
                        mean_mass,
                    )
    if all_ok:
        log.info("  ✓ All feature ranges within expected bounds")

    # ------------------------------------------------------------------
    # Save extraction metadata for reproducibility
    # ------------------------------------------------------------------
    metadata: dict = {
        "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
        "model_checkpoint": str(ROOT / cfg["paths"]["final_checkpoint_dir"]),
        "model_base": cfg["model"]["model_name_or_path"],
        "best_layer": best_layer,
        "sweep_layers": SWEEP_LAYERS,
        "ids_abcd": IDS_ABCD,
        "yes_id": yes_id,
        "no_id": no_id,
        "skip_ptrue": args.skip_ptrue,
        "debug_mode": args.debug,
        "splits": {},
    }
    for split_name in splits_to_run:
        npz_path = features_dir / f"{split_name}_features.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=False)
        entry: dict = {
            "n": int(len(d["y"])),
            "accuracy": round(float(d["y"].mean()), 4),
            "H_mean": round(float(d["H"].mean()), 4),
            "H_std": round(float(d["H"].std()), 4),
            "gap_mean": round(float(d["gap"].mean()), 4),
            "gap_std": round(float(d["gap"].std()), 4),
        }
        if not args.skip_ptrue:
            valid = ~np.isnan(d["p_true"])
            if valid.sum() > 0:
                entry["p_true_mean"] = round(float(d["p_true"][valid].mean()), 4)
                entry["p_true_coverage"] = round(float(valid.mean()), 4)
        metadata["splits"][split_name] = entry

    meta_path = results_dir / "extraction_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("Extraction metadata saved to %s", meta_path)

    print("\nSend to student (via professor):")
    print(f"  {features_dir}/*.npz  and  *.npy   (~500 MB total)")
    print(f"  {ROOT / 'checkpoints' / 'final'}/  (LoRA adapter, ~150 MB)")
    print(f"  {results_dir}/layer_sweep.json  +  extraction_metadata.json")
    print("\nNext step (student, CPU-local):")
    print("  python scripts/04_probe.py")


if __name__ == "__main__":
    main()
