#!/usr/bin/env bash
# Launch Mistral-7B LoRA fine-tuning on 4× NVIDIA L4 inside a detached screen session.
# All console output is tee'd to logs/finetune_run_<timestamp>.log so it survives
# disconnects. Resume / observe later with:  screen -r finetune
#
# Usage:
#   bash scripts/launch_finetune_ddp.sh              # full run
#   bash scripts/launch_finetune_ddp.sh --debug      # smoke test on tiny subset
#   bash scripts/launch_finetune_ddp.sh --resume     # resume from last checkpoint
#
# Re-attach to live progress:
#   screen -r finetune
#   (Detach with Ctrl-A then D — does NOT stop the job)
#
# Tail the log file from anywhere:
#   tail -f logs/finetune_run_<timestamp>.log
#
# Stop the job:
#   screen -S finetune -X quit

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1) Use 'ai' conda env directly via absolute paths.
#    (PATH-based 'conda activate ai' is unreliable here because Claude Code's
#     shell snapshot pins PATH to envs/rp/bin; we side-step that by being explicit.)
AI_ENV="/home/lciechanowski/anaconda3/envs/ai"
AI_PYTHON="$AI_ENV/bin/python"
AI_TORCHRUN="$AI_ENV/bin/torchrun"
if [ ! -x "$AI_PYTHON" ] || [ ! -x "$AI_TORCHRUN" ]; then
    echo "ERROR: missing $AI_PYTHON or $AI_TORCHRUN" >&2
    exit 1
fi
# Put 'ai' bin first on PATH for any subprocesses (e.g. torch's NCCL helpers)
export PATH="$AI_ENV/bin:$PATH"

# 2) Load HF_TOKEN (and any other secrets) from .env if present
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. mistralai/Mistral-7B-Instruct-v0.3 is gated;"
    echo "         the download will fail unless you've already cached it."
fi

# 3) Setup
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs checkpoints
LOG_FILE="logs/finetune_run_${TIMESTAMP}.log"
SCREEN_NAME="finetune"

# Make Python output unbuffered so tail -f sees progress live
export PYTHONUNBUFFERED=1
# Avoid HF tokenizer threading warnings under DataLoader workers
export TOKENIZERS_PARALLELISM=false
# NCCL settings tuned for PCIe (no NVLink) on a single node
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN

# Detect number of visible GPUs
N_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "[$(date '+%H:%M:%S')] Detected ${N_GPUS} GPU(s)"
echo "[$(date '+%H:%M:%S')] Log file: $LOG_FILE"
echo "[$(date '+%H:%M:%S')] Screen session: $SCREEN_NAME"

# 4) Build the torchrun command (absolute path to ai-env torchrun)
TORCHRUN_CMD=(
    "$AI_TORCHRUN"
    --standalone
    --nproc_per_node="${N_GPUS}"
    scripts/02_finetune.py
    --config finetune_config_ddp.yaml
    "$@"
)

# 5) If a screen session of the same name already exists, refuse to clobber it
if screen -ls | grep -q "\.${SCREEN_NAME}[[:space:]]"; then
    echo "ERROR: screen session '${SCREEN_NAME}' already exists." >&2
    echo "       Attach with:  screen -r ${SCREEN_NAME}" >&2
    echo "       Or kill with: screen -S ${SCREEN_NAME} -X quit" >&2
    exit 1
fi

# 6) Launch detached. The session runs bash so it stays alive after the command
#    exits — you can inspect final output. 'screen -L' tees to screenlog.0
#    inside the session; we ALSO pipe through tee so the log file is exactly
#    what the user wants and is independent of screen's own logging quirks.
screen -dmS "${SCREEN_NAME}" bash -c "
    echo '=========================================================='
    echo 'Mistral-7B LoRA fine-tune  —  4× L4 DDP'
    echo 'Started: $(date)'
    echo 'PID: \$\$'
    echo 'Log:  ${LOG_FILE}'
    echo 'Command: ${TORCHRUN_CMD[*]}'
    echo '=========================================================='
    ${TORCHRUN_CMD[*]} 2>&1 | tee -a '${LOG_FILE}'
    EXIT=\${PIPESTATUS[0]}
    echo '=========================================================='
    echo 'Finished: $(date) — exit code \${EXIT}'
    echo '=========================================================='
    # Keep shell alive so user can inspect screen even after job ends.
    exec bash
"

sleep 1
echo ""
echo "Launched.  Watch progress with:"
echo "    tail -f $PROJECT_ROOT/$LOG_FILE"
echo "  or attach to the screen session:"
echo "    screen -r ${SCREEN_NAME}    (detach with Ctrl-A then D)"
echo ""
echo "To stop:"
echo "    screen -S ${SCREEN_NAME} -X quit"
