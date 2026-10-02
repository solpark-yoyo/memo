#!/bin/bash
# =============================================================================
#  Exp2: n_opti Sweep (init_step=0 fixed, optimization depth variation)
#  [init_step=0 (x_T)에서 num_opt_steps를 [1,2,5,10,20] 다양하게 설정 → trajectory 변화]
#
#  Design:
#    1) Fixed init_step=0, gap_steps=1, num_steps=1
#       (only x_T position is optimized; sweep_indices = [0])
#    2) For each n_opti in [1,2,5,10,20]:
#       - Optimize x_T with n_opti Adam iterations
#       - DDIM denoising (full 50 steps)
#       - Record latent trajectory + memo_proxy
#    3) Overlay trajectory curves for comparison
#
#  Research question:
#    How does the number of optimization iterations affect trajectory shape?
#    Is there diminishing return with more optimization steps?
#
#  Environment: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/exp_main/trajectory/run_exp2_step_by_step.sh
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

# Exp2-specific: n_opti sweep (init_step=0 fixed)
init_step="${INIT_STEP:-0}"
gap_steps="${GAP_STEPS:-1}"
num_steps="${NUM_STEPS:-1}"
num_opt_steps_list="${NUM_OPT_STEPS:-1 3 5}"
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
traj_path="${out_root}/${model}/ddim/CFG=${cfg}_NFE=${NFE}/gap=${gap_steps}/num=${num_steps}/init=${init_step}"

# =========================== 2. [Run] ===========================
echo "=========================================="
echo "  Exp2: n_opti Sweep (init_step=0 fixed)"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  seed=${seed}"
echo "  init_step=${init_step}  gap_steps=${gap_steps}  num_steps=${num_steps}"
echo "  num_opt_steps_list=(${num_opt_steps_list})"
echo "  num_samples_per_init=${num_samples_per_init}"
echo "  lr=${lr}  base_s_ratio=${base_s_ratio}"
echo "  prompt_profile=${prompt_profile}"
echo "  traj_path=${traj_path}"
echo "=========================================="

python trajectory_init_study.py \
    --exp exp2 \
    --model_key "${model_key}" \
    --device "${DEVICE}" \
    --seed "${seed}" \
    --output_dir "${out_root}" \
    --cfg_guidance "${cfg}" \
    --num_inference_steps "${NFE}" \
    --block_size 16 \
    --init_step ${init_step} \
    --gap_steps ${gap_steps} \
    --num_steps ${num_steps} \
    --nopt_list ${num_opt_steps_list} \
    --num_samples_per_init ${num_samples_per_init} \
    --lr ${lr} \
    --base_s_ratio ${base_s_ratio} \
    --prompt_sources "${prompt_sources[@]}" \
    ${overwrite_flag} \
    --save_plots \
    --save_csv

echo ""
echo "[Done] Exp2 complete"
echo "  Output: ${traj_path}/exp_2/"
echo ""
