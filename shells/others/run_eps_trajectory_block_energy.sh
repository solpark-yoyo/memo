#!/bin/bash
# =============================================================================
#  Block-Energy Trajectory  (memorized prompt, DDIM, SD1.4)
#
#  모델  : ckpt/stable-diffusion-v1-4  (비-memorized 일반 SD1.4)
#  prompt: examples/assets/cvpr2025_memo_prompt.txt  (Chen et al. CVPR2025 memorized)
#  scale : optimize_xt_spectral.py 의 spectral_l2_loss 와 동일
#          (= mean(block energy)/B, 1 = white Gaussian; run_memo_spectral.sh 기준)
#
#  목표: memorized text prompt 에 대해 denoising step 이 진행됨에 따라
#        주파수 block 별 energy (‖y^(p)‖_2²) curve 가 어떻게 변하는지 시각화.
#    - x 축: block index (0 ~ #blocks-1)
#    - y 축: 해당 block 의 energy
#    - num_curve_steps 개 step 을 균등 간격 선택하여 curve 로 overlay
#      (예: NFE=50, 5 → step 0,10,20,30,40)
#
#  저장 (output_dir/LH_energy 아래):
#    plot/ : block_energy_memoXX.png (curve, lin+log) + _heatmap.png (전체 step)
#    csv/  : block_energy_memoXX.csv (row=step, col=block) + .npz (per-seed 원본)
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/others/run_eps_trajectory_block_energy.sh
#
#  오버라이드 예:
#    BLOCK_SIZE=32 BATCH=3 NUM_CURVE_STEPS=10 FREQ_RATIO=0.2 bash shells/others/run_eps_trajectory_block_energy.sh
# =============================================================================

# ---- Config (환경변수로 오버라이드 가능) ----
gpu="${GPU:-0}"
NFE="${NFE:-50}"
cfg="${CFG:-7.5}"
SEED="${SEED:-42}"
batch="${BATCH:-5}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"          # 주파수 block 하나의 bin 수 B
NUM_CURVE_STEPS="${NUM_CURVE_STEPS:-5}" # curve 로 그릴 denoising step 개수 (예: 5 → step 0,10,20,30,40)
FREQ_RATIO="${FREQ_RATIO:-0.1}"          # Low/High 밴드 비율 (예: 0.1, P=512 → low[0:50], high[461:511])
DEVICE="cuda:${gpu}"

num_samples=${batch}                    # seed 수 (memorized → 평균)
num_mtp="${NUM_MTP:-1}"                 # memorized prompt 수
num_plot="${NUM_PLOT:-1}"

MODEL="${MODEL:-ckpt/stable-diffusion-v1-4}"                      # SD1.4 (비-memorized 일반 모델)
memo_dir="${MEMO_DIR:-examples/assets/cvpr2025_memo_prompt.txt}"   # Chen et al. (CVPR2025) memorized prompts

# 저장 경로: results_cmp_l2_trajectory/ddim/CFG=.../seed=.../batch=...
base_dir="${EXP_ROOT:-workdir/exp_main/eps_trajectory}/results_cmp_l2_trajectory"
OUTPUT_DIR="${base_dir}/ddim/CFG=${cfg}_NFE=${NFE}/seed=${SEED}/batch=${batch}"

echo "========================================="
echo "  Block-Energy Trajectory (memorized, DDIM)"
echo "  NFE=${NFE}  CFG=${cfg}  SEED=${SEED}  batch=${batch}"
echo "  block_size=${BLOCK_SIZE}  num_samples=${num_samples}  num_curve_steps=${NUM_CURVE_STEPS}  freq_ratio=${FREQ_RATIO}"
echo "  num_mtp=${num_mtp}  num_plot=${num_plot}"
echo "  model=${MODEL}"
echo "  memo_dir=${memo_dir}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}/LH_energy/{csv,plot}"
echo "========================================="

cd "${ROOT_DIR:-.}"   # ori_memo/ 에서 실행 가정; 다른 위치면 ROOT_DIR override

python eps_trajectory.py \
    --block_energy \
    --model_key ${MODEL} \
    --block_size ${BLOCK_SIZE} \
    --num_curve_steps ${NUM_CURVE_STEPS} \
    --freq_ratio ${FREQ_RATIO} \
    --num_inference_steps ${NFE} \
    --cfg_guidance ${cfg} \
    --seed ${SEED} \
    --device ${DEVICE} \
    --num_samples ${num_samples} \
    --output_dir ${OUTPUT_DIR} \
    --memo_dir ${memo_dir} \
    --num_mtp ${num_mtp} \
    --num_plot ${num_plot}

echo "Done. -> ${OUTPUT_DIR}/LH_energy/{csv,plot}"
