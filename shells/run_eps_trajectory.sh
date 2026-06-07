#!/bin/bash
# =============================================================================
#  Epsilon Trajectory Analysis for Memorization
#  x_t ──(Tweedie)──> x̂_0 ──(DDIM forward)──> x_s
#  Plot: ||ε - ε_s||² vs denoising step
# =============================================================================

# ---- Config ----
gpu=0
NFE=50
cfg=7.5
SEED=42
NUM_SAMPLES=1
DEVICE="cuda:${gpu}"
OUTPUT_DIR="results_eps_trajectory"

# ---- Paths ----
ROOT_DIR="/home/geonsoo/Desktop/Datasets/Parksol/memo/ori_memo"
PYTHON="/home/geonsoo/anaconda3/bin/python"

# ---- Prompts ----
PROMPT_ARGS=(
    "A red car parked on a rainy street at night with neon reflections"
    "A child building a sandcastle on a cloudy beach with a small bucket"
    "Two cats sitting on a windowsill watching birds outside on a sunny morning"
    "An astronaut on the moon"
)
MEMO_IDX=(3)

echo "========================================="
echo "  Epsilon Trajectory Analysis"
echo "  NFE=${NFE}  CFG=${cfg}  SEED=${SEED}"
echo "  NUM_SAMPLES=${NUM_SAMPLES}"
echo "  prompts: ${#PROMPT_ARGS[@]}"
echo "========================================="

cd "${ROOT_DIR}"

${PYTHON} eps_trajectory.py \
    --num_inference_steps ${NFE} \
    --cfg_guidance ${cfg} \
    --seed ${SEED} \
    --device ${DEVICE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --memo_indices ${MEMO_IDX[@]} \
    --prompts "${PROMPT_ARGS[@]}"

echo "Done."
