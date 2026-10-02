#!/bin/bash
# =============================================================================
#  Exp1: Init Step Sweep (different starting positions, trajectory analysis)
#  [x_T에서 init_step을 [1,2,3,4,5,10] 다양하게 설정 → trajectory 변화]
#
#  Design:
#    1) Fixed gap_steps=0, num_steps=1 (optimize only at init_step)
#    2) For each init_step in [1,2,3,4,5,10]:
#       - Optimize x_T at timestep init_step
#       - DDIM denoising (full 50 steps)
#       - Record latent trajectory + Tweedie gap
#    3) Overlay step-by-step trajectory curves for comparison
#
#  Research question:
#    Which denoising step is most critical for memorization mitigation?
#    How does early/late initialization affect trajectory shape?
#
#  Environment: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/exp_main/trajectory/run_exp1_init_sweep.sh
# =============================================================================


# =========================== 1. [Parser] ===========================
gpu="${GPU:-0}"
model="${MODEL:-stable-diffusion-v1-4}"
NFE="${NFE:-50}"
cfg="${CFG:-7.5}"
seed="${SEED:-42}"
DEVICE="cuda:${gpu}"

model_key="${MODEL_KEY:-ckpt/${model}}"
out_root="${OUT_ROOT:-workdir/exp_main/trajectory}"

# Exp1-specific: init_steps sweep (gap_steps=0, num_steps=1 fixed)
init_steps_list="${INIT_STEPS:-0 1 2}"
gap_steps="${GAP_STEPS:-1}"
num_steps="${NUM_STEPS:-1}"
num_opt_steps="${NUM_OPT_STEPS:-1}"
num_samples_per_init="${NUM_SAMPLES_PER_INIT:-5}"
lr="${LR:-0.01}"
base_s_ratio="${BASE_S_RATIO:-0.5}"

# Reuse: default skips existing configs. OVERWRITE=1 to force re-run
overwrite_flag=""
[[ "${OVERWRITE:-0}" == "1" ]] && overwrite_flag="--overwrite"

# Prompt sources (memorized prompts only)
#   PROMPT_PROFILE=all      : all memorized sources (webster, membench)
#   (default)               : standard (webster only)
#   PROMPT_SOURCES="..."    : manual specification (overrides PROMPT_PROFILE)
prompt_profile="${PROMPT_PROFILE:-standard}"

if [[ -n "${PROMPT_SOURCES:-}" ]]; then
    read -ra prompt_sources <<< "${PROMPT_SOURCES}"
elif [[ "$prompt_profile" == "all" ]]; then
    prompt_sources=(
        "memo:webster=examples/assets/sdv1_500_mem.txt"
        "memo:membench=examples/assets/memorized_prompts_membench.txt"
    )
else
    prompt_sources=(
        "memo:webster=examples/assets/sdv1_500_mem.txt"
    )
fi

# Paths
cd "${ROOT_DIR:-.}"

if [ ! -f "trajectory_init_study.py" ]; then
    echo "Error: trajectory_init_study.py not found. Please run from memo/ori_memo/"
    exit 1
fi

# ---- Output path structure ----
traj_path="${out_root}/${model}/ddim/CFG=${cfg}_NFE=${NFE}/gap=${gap_steps}/num=${num_steps}"

# =========================== 2. [Run] ===========================
echo "=========================================="
echo "  Exp1: Init Step Sweep (trajectory analysis)"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  seed=${seed}"
echo "  init_steps_list=(${init_steps_list})"
echo "  gap_steps=${gap_steps}  num_steps=${num_steps}  num_opt_steps=${num_opt_steps}"
echo "  num_samples_per_init=${num_samples_per_init}"
echo "  prompt_profile=${prompt_profile}"
echo "  traj_path=${traj_path}"
echo "=========================================="

python trajectory_init_study.py \
    --exp exp1 \
    --model_key "${model_key}" \
    --device "${DEVICE}" \
    --seed "${seed}" \
    --output_dir "${out_root}" \
    --cfg_guidance "${cfg}" \
    --num_inference_steps "${NFE}" \
    --block_size 16 \
    --init_steps_list ${init_steps_list} \
    --gap_steps ${gap_steps} \
    --num_steps ${num_steps} \
    --num_opt_steps ${num_opt_steps} \
    --num_samples_per_init ${num_samples_per_init} \
    --lr ${lr} \
    --base_s_ratio ${base_s_ratio} \
    --prompt_sources "${prompt_sources[@]}" \
    ${overwrite_flag} \
    --save_plots \
    --save_csv

echo ""
echo "[Done] Exp1 complete"
echo "  Output: ${traj_path}/exp_1/"
echo ""
