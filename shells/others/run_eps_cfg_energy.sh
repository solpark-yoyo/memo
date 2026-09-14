#!/bin/bash
# =============================================================================
#  [TEST] eps_cfg energy:  x_t → ε_cfg → proxy = ‖ε_cfg‖² / D
#
#  text(general) 3개 vs memorized 1개 곡선을 denoising step 에 따라 비교
#  scale: ε ~ N(0,I) → proxy 평균제곱 1 = white Gaussian 기준
#
#  loss 조절 (argparse --loss_cfg 로 전달):
#    LOSS_CFG 를 주지 않으면 CFG 값과 동일 (기본)
#    LOSS_CFG=1.0 → ε_uc(조건 없음) 에너지로 측정
#
#  저장 (OUTPUT_DIR 아래): plot_num=N/plotNN/{csv(proxy_NN.csv), npz, imgs}
#    + plotNN/eps_l2.png (곡선 비교) + plot_num=N/memo_proxy_overall.png
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/others/run_eps_cfg_energy.sh
#    LOSS_CFG=1.0 bash shells/others/run_eps_cfg_energy.sh
# =============================================================================
gpu="${GPU:-0}"
NFE="${NFE:-50}"
cfg="${CFG:-7.5}"
SEED="${SEED:-42}"
batch="${BATCH:-5}"

num_tp="${NUM_TP:-3}"                   # text(general) prompt 수 (plot 1개당)
num_mtp="${NUM_MTP:-1}"                 # memorized prompt 수 (plot 1개당)
num_plot="${NUM_PLOT:-20}"              # plot00, plot01 ... 비교 plot 개수

MODEL="${MODEL:-ckpt/stable-diffusion-v1-4}"
text_dir="${TEXT_DIR:-examples/assets/coco_v2.txt}"
memo_dir="${MEMO_DIR:-examples/assets/memorized_prompts_membench.txt}"

LOSS_CFG_FLAG=""; [[ -n "${LOSS_CFG:-}" ]] && LOSS_CFG_FLAG="--loss_cfg ${LOSS_CFG}"

base_dir="${EXP_ROOT:-workdir/exp_main/eps_trajectory}/results_eps_l2_trajectory"
OUTPUT_DIR="${base_dir}/ddim/CFG=${cfg}_NFE=${NFE}/seed=${SEED}/batch=${batch}"

echo "========================================="
echo "  [TEST] eps_cfg energy trajectory"
echo "  NFE=${NFE}  CFG=${cfg}  SEED=${SEED}  batch=${batch}"
echo "  text=${num_tp}개(${text_dir##*/})  memo=${num_mtp}개(${memo_dir##*/})  num_plot=${num_plot}"
echo "  loss_cfg=${LOSS_CFG:-<CFG와 동일>}"
echo "  model=${MODEL}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "========================================="

cd "${ROOT_DIR:-.}"   # ori_memo/ 에서 실행 가정

python eps_trajectory.py \
    --measure eps_l2 \
    --model_key ${MODEL} \
    --num_inference_steps ${NFE} \
    --cfg_guidance ${cfg} \
    --seed ${SEED} \
    --device cuda:${gpu} \
    --num_samples ${batch} \
    --output_dir ${OUTPUT_DIR} \
    --text_dir ${text_dir} \
    --memo_dir ${memo_dir} \
    --num_tp ${num_tp} \
    --num_mtp ${num_mtp} \
    --num_plot ${num_plot} ${LOSS_CFG_FLAG}

echo "Done. -> ${OUTPUT_DIR}"
