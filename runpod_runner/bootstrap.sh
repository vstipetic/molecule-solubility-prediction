# Bootstrap script executed inside the RunPod pod over SSH.
# Runs the selected training module, writing all output to /workspace/train.log
# (tee'd to stdout so the orchestrator can stream it) and the checkpoint to
# $CHECKPOINT_DIR. Wandb credentials/config arrive as env vars.
#
# Inputs (env vars, set by the orchestrator at pod creation):
#   REPO_URL        - git URL to clone (default: the project origin)
#   REPO_REF        - branch/tag/commit to checkout (default: main)
#   MODEL           - one of: dmpnn | transformer | chemberta
#   TRAIN_ARGS      - extra args forwarded to the training module
#   DATA_PATH       - path inside pod to the training CSV
#   CHECKPOINT_DIR  - where the training script writes the checkpoint
#   WANDB_PROJECT / WANDB_ENTITY / WANDB_NAME / WANDB_GROUP / WANDB_TAGS / WANDB_MODE
#
set -euo pipefail

# Mirror everything to a logfile on the persistent volume for later SCP.
exec > >(tee -a /workspace/train.log) 2>&1

echo "=== RunPod bootstrap $(date -u) ==="
echo "MODEL=${MODEL:?MODEL env var required}"
echo "DATA_PATH=${DATA_PATH:-(unset)}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR:-/workspace/checkpoints}"

REPO_URL="${REPO_URL:-https://github.com/vstipetic/molecule-solubility-prediction.git}"
REPO_REF="${REPO_REF:-main}"
REPO_DIR="/workspace/molecule-solubility-prediction"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints}"
mkdir -p "$CHECKPOINT_DIR"

cd /workspace

# Clone or update the repo on the persistent volume.
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Cloning $REPO_URL -> $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "Repo already present at $REPO_DIR"
fi

cd "$REPO_DIR"
git fetch --quiet origin
git checkout "$REPO_REF"
git reset --hard "origin/${REPO_REF}"

# Install uv and sync the project environment from the lockfile.
echo "Installing uv..."
pip install --quiet uv
echo "Syncing project dependencies (uv sync)..."
uv sync

# Common wandb flags (kept consistent with the local entry points).
WANDB_FLAGS=""
[ -n "${WANDB_PROJECT:-}" ] && WANDB_FLAGS="$WANDB_FLAGS --wandb-project ${WANDB_PROJECT}"
[ -n "${WANDB_ENTITY:-}" ]   && WANDB_FLAGS="$WANDB_FLAGS --wandb-entity ${WANDB_ENTITY}"
[ -n "${WANDB_NAME:-}" ]     && WANDB_FLAGS="$WANDB_FLAGS --wandb-name ${WANDB_NAME}"
[ -n "${WANDB_GROUP:-}" ]    && WANDB_FLAGS="$WANDB_FLAGS --wandb-group ${WANDB_GROUP}"
[ -n "${WANDB_TAGS:-}" ]     && WANDB_FLAGS="$WANDB_FLAGS --wandb-tags ${WANDB_TAGS}"
[ -n "${WANDB_MODE:-}" ]     && WANDB_FLAGS="$WANDB_FLAGS --wandb-mode ${WANDB_MODE}"

# Per-model training command. The save path lives under $CHECKPOINT_DIR so the
# orchestrator can SCP it back; DATA_PATH is the uploaded CSV.
case "$MODEL" in
    dmpnn)
        SAVE_PATH="$CHECKPOINT_DIR/dmpnn.pt"
        uv run python -m Train.train_gnn \
            --data-path "$DATA_PATH" \
            --save-path "$SAVE_PATH" \
            $WANDB_FLAGS $TRAIN_ARGS
        ;;
    transformer)
        SAVE_PATH="$CHECKPOINT_DIR/transformer.pt"
        uv run python -m Train.finetune_transformer \
            --data-path "$DATA_PATH" \
            --save-path "$SAVE_PATH" \
            $WANDB_FLAGS $TRAIN_ARGS
        ;;
    chemberta)
        SAVE_PATH="$CHECKPOINT_DIR/chemberta"
        uv run python -m Train.finetune_chemberta \
            --data-path "$DATA_PATH" \
            --save-path "$SAVE_PATH" \
            $WANDB_FLAGS $TRAIN_ARGS
        ;;
    *)
        echo "ERROR: unknown MODEL '$MODEL' (expected dmpnn|transformer|chemberta)" >&2
        exit 2
        ;;
esac

echo "=== Training complete; checkpoint at $SAVE_PATH ==="
