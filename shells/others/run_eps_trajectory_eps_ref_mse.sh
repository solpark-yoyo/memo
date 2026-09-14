#!/bin/bash
# =============================================================================
#  ⑤ eps_ref MSE Trajectory Analysis for Memorization
#  x_t ──(UNet+CFG)──> ε_cfg(x_t,t)
#  Plot: ||ε_ref − ε_cfg(x_t,t)||² / D  vs denoising step   (D = noise dim)
#        ε_ref ~ N(0,I) — x_T와 독립 (①의 reference 재사용)
#        scale: ③ eps_l2(‖ε_cfg‖²/D)와 기댓값 동치, grad 상관항 차이가 관찰 포인트
#
#  각 plot = num_tp 개 text prompt (coco_v2) + num_mtp 개 memo prompt (membench)
#  num_plot 개의 비교 plot 생성
#
#  저장: eps_trajectory/results_eps_ref_mse_trajectory/ddim/CFG=x_NFE=y/seed=z/batch=b/
#  환경: conda div_DM (ori_memo/ 에서 실행)
# =============================================================================

# ---- Config ----
gpu=0
NFE=50
cfg=7.5
SEED=42
batch=5
num_images_per_prompt=${batch}
NUM_SAMPLES=${num_images_per_prompt}     # images per prompt (different seed each)
DEVICE="cuda:${gpu}"
MEASURE="${MEASURE:-eps_ref_mse}"        # ⑤ (registry candidate_proxy.md)

base_dir="${EXP_ROOT:-workdir/exp_main/eps_trajectory}/results_eps_ref_mse_trajectory"
OUTPUT_DIR="${base_dir}/ddim/CFG=${cfg}_NFE=${NFE}/seed=${SEED}/batch=${batch}"

# ---- Prompt config ----
num_tp=3          # text prompt 갯수 (coco)
num_mtp=1         # memorized text prompt 갯수 (membench)
num_plot=20    # plot00, plot01 ... 각 plot = num_tp text + num_mtp memo

text_dir="examples/assets/coco_v2.txt"
memo_dir="examples/assets/memorized_prompts_membench.txt"

echo "========================================="
echo "  ⑤ eps_ref MSE Trajectory Analysis"
echo "  NFE=${NFE}  CFG=${cfg}  SEED=${SEED}  NUM_SAMPLES=${NUM_SAMPLES}"
echo "  num_tp=${num_tp}  num_mtp=${num_mtp}  num_plot=${num_plot}"
echo "  measure=${MEASURE}"
echo "  text_dir=${text_dir}"
echo "  memo_dir=${memo_dir}"
echo "========================================="

cd "${ROOT_DIR:-.}"

python eps_trajectory.py \
    --num_inference_steps ${NFE} \
    --cfg_guidance ${cfg} \
    --seed ${SEED} \
    --device ${DEVICE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --text_dir ${text_dir} \
    --memo_dir ${memo_dir} \
    --num_tp ${num_tp} \
    --num_mtp ${num_mtp} \
    --num_plot ${num_plot} \
    --measure ${MEASURE}

echo "Done."
