#!/bin/bash
# ===================================================================
#  x_T Optimization for Memorization Mitigation
#
#  Phase 1: Optimize x_T via epsilon consistency loss
#  Phase 2: Generate images with standard DDIM from optimized x_T
# ===================================================================

gpu=0
NFE=50
cfg=7.5
SEED=42
DEVICE="cuda:${gpu}"
OUTPUT_DIR="results_xT_opt"

# Optimization params
NUM_OPT_STEPS=20
LR=0.01
TARGET_STEP_RATIO=0.20
TARGET_S_RATIO=0.20
NUM_GEN_SAMPLES=2

# Prompts
PROMPTS=()
# PROMPTS+=("A red car parked on a rainy street at night with neon reflections")
# PROMPTS+=("A child building a sandcastle on a cloudy beach with a small bucket")
PROMPTS+=("Two cats sitting on a windowsill watching birds outside on a sunny morning")
PROMPTS+=("An astronaut on the moon")

PROMPT_ARGS=()
for p in "${PROMPTS[@]}"; do
    PROMPT_ARGS+=("$p")
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "========================================="
echo "  x_T Optimization"
echo "  NFE=${NFE}  CFG=${cfg}  SEED=${SEED}"
echo "  opt_steps=${NUM_OPT_STEPS}  lr=${LR}"
echo "  target_step=${TARGET_STEP_RATIO}  target_s=${TARGET_S_RATIO}"
echo "========================================="

python optimize_xT.py \
    --num_inference_steps ${NFE} \
    --cfg_guidance ${cfg} \
    --seed ${SEED} \
    --device ${DEVICE} \
    --output_dir ${OUTPUT_DIR} \
    --num_opt_steps ${NUM_OPT_STEPS} \
    --lr ${LR} \
    --target_step_ratio ${TARGET_STEP_RATIO} \
    --target_s_ratio ${TARGET_S_RATIO} \
    --num_gen_samples ${NUM_GEN_SAMPLES} \
    --prompts "${PROMPT_ARGS[@]}"

echo "Done."
