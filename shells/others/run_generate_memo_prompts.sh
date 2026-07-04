#!/bin/bash
# ===================================================================
#  Generate images for known memorized prompts with multiple seeds
# ===================================================================

gpu=0
NFE=50
cfg=7.5
BASE_SEED=42
NUM_SEEDS=8
DEVICE="cuda:${gpu}"
OUTPUT_DIR="results_memo_prompts"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "========================================="
echo "  Memorized Prompt Generation"
echo "  NFE=${NFE}  CFG=${cfg}  NUM_SEEDS=${NUM_SEEDS}"
echo "========================================="

python generate_memo_prompts.py \
    --num_inference_steps ${NFE} \
    --cfg_guidance ${cfg} \
    --base_seed ${BASE_SEED} \
    --num_seeds ${NUM_SEEDS} \
    --device ${DEVICE} \
    --output_dir ${OUTPUT_DIR}

echo "Done."
