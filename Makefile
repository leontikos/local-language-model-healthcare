.PHONY: setup verify prepare finetune extract probe routing conformal evaluate ablations test lint clean

# ── Setup ────────────────────────────────────────────────────────────────────
setup:
	pip install -e ".[dev]"

# ── Pipeline steps ───────────────────────────────────────────────────────────
verify:
	python scripts/00_verify.py --blocker 1
	python scripts/00_verify.py --blocker 2

verify-all:
	python scripts/00_verify.py --blocker all

prepare:
	python scripts/01_prepare.py

finetune:
	python scripts/02_finetune.py

finetune-qlora:
	python scripts/02_finetune.py model=qlora_4bit

extract:
	python scripts/03_extract.py

probe:
	python scripts/04_probe.py

routing:
	python scripts/05_routing.py

conformal:
	python scripts/06_conformal.py

evaluate:
	python scripts/07_evaluate.py

ablations:
	python scripts/08_ablations.py

# ── Full pipeline (after fine-tuning is done) ────────────────────────────────
pipeline: extract probe routing conformal evaluate ablations

# ── Dev ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/
	ruff format --check src/ scripts/

format:
	ruff format src/ scripts/

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	rm -rf outputs/ wandb/ __pycache__ .pytest_cache
	find . -name "*.pyc" -delete

clean-features:
	rm -rf data/features/*.npz data/features/*.npy

# ── Status ───────────────────────────────────────────────────────────────────
status:
	python scripts/00_verify.py --blocker status
