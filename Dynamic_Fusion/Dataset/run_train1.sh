#!/usr/bin/env bash
# ---------------------------------------------------------------
# run_train1.sh
# Launch train1.py inside a tmux session.
#
# Usage:
#   bash run_train1.sh [--load 0|1]
#
# Examples:
#   bash run_train1.sh           # train from scratch (default)
#   bash run_train1.sh --load 1  # resume from checkpoint
# ---------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SESSION_NAME="train1"
LOAD_FLAG="${1:-0}"   # default: train from scratch

echo "==> Launching train1.py in tmux session '${SESSION_NAME}' (load=${LOAD_FLAG})"

# Kill existing session if it exists
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

# Build the python command
# 'bash' keeps the tmux pane open after the script exits (success or error)
CMD="cd ${SCRIPT_DIR} && \
  source ${PROJECT_DIR}/.venv/bin/activate && \
  python train1.py \
    --ds Dataset \
    --dim 16 \
    --lr 1e-5 \
    --l2 0.01 \
    --model ETH_GBert \
    --load ${LOAD_FLAG}; \
  echo ''; echo '===== FINISHED (exit code: '\$?') ====='; \
  exec bash"

# Create a new detached tmux session and run the command
tmux new-session -d -s "${SESSION_NAME}" bash -c "${CMD}"

echo "==> tmux session '${SESSION_NAME}' started."
echo "    Attach with:  tmux attach -t ${SESSION_NAME}"
echo "    Detach with:  Ctrl-b d"
echo "    Kill with:    tmux kill-session -t ${SESSION_NAME}"
