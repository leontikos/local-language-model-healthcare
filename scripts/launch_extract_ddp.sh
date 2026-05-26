#!/usr/bin/env bash
# Launch KROK 4 feature extraction on 4× NVIDIA L4 in a detached screen session.
# Each rank loads a full model copy onto its own GPU and processes its stride of
# the dataset; rank 0 stitches the per-rank shards into the final .npz / .npy
# files. Layer sweep is also sharded.
#
# Usage:
#   bash scripts/launch_extract_ddp.sh                       # full extraction (all 5 splits)
#   bash scripts/launch_extract_ddp.sh --debug               # smoke test, 50 examples/split
#   bash scripts/launch_extract_ddp.sh --splits probe val    # selected splits only
#   bash scripts/launch_extract_ddp.sh --layer 31            # skip sweep
#
# Re-attach:    screen -r extract        (Ctrl-A D to detach, no stop)
# Tail log:     tail -f logs/extract_run_<timestamp>.log
# Stop:         screen -S extract -X quit

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

AI_ENV="/home/lciechanowski/anaconda3/envs/ai"
AI_PYTHON="$AI_ENV/bin/python"
AI_TORCHRUN="$AI_ENV/bin/torchrun"
if [ ! -x "$AI_PYTHON" ] || [ ! -x "$AI_TORCHRUN" ]; then
    echo "ERROR: missing $AI_PYTHON or $AI_TORCHRUN" >&2
    exit 1
fi
export PATH="$AI_ENV/bin:$PATH"

# Load HF_TOKEN etc. from .env
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs data/features
LOG_FILE="logs/extract_run_${TIMESTAMP}.log"
SCREEN_NAME="extract"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN

N_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "[$(date '+%H:%M:%S')] Detected ${N_GPUS} GPU(s)"
echo "[$(date '+%H:%M:%S')] Log file: $LOG_FILE"
echo "[$(date '+%H:%M:%S')] Screen session: $SCREEN_NAME"

TORCHRUN_CMD=(
    "$AI_TORCHRUN"
    --standalone
    --nproc_per_node="${N_GPUS}"
    scripts/03_extract.py
    --config finetune_config_ddp.yaml
    "$@"
)

# Refuse to clobber an existing session of the same name
if screen -ls | grep -q "\.${SCREEN_NAME}[[:space:]]"; then
    echo "ERROR: screen session '${SCREEN_NAME}' already exists." >&2
    echo "       Attach with:  screen -r ${SCREEN_NAME}" >&2
    echo "       Or kill with: screen -S ${SCREEN_NAME} -X quit" >&2
    exit 1
fi

screen -dmS "${SCREEN_NAME}" bash -c "
    echo '=========================================================='
    echo 'KROK 4: Feature extraction  —  4× L4 DDP'
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
