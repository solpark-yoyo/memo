#!/bin/bash
# =============================================================================
#  Epsilon Trajectory Analysis for Memorization
#  x_t ──(Tweedie)──> x̂_0 ──(DDIM forward)──> x_s
#  Plot: ||ε - ε_s||² / D  vs denoising step   (D = noise dim)
#
#  각 plot = num_tp 개 general prompt (ms_coco/coco_v2) + num_mtp 개 memo prompt (webster)
#  num_plot 개의 비교 plot 생성
#
#  Sources:
#    general    — ms_coco (examples/assets/coco_v2.txt)
#    memorized  — webster (examples/assets/sdv1_500_mem.txt)
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
MEASURE="${MEASURE:-memo_proxy}"         # memo_proxy | kl_div  (측정 metric toggle)

# save path: measure 에 따라 base_dir 분리
exp_root="${EXP_ROOT:-workdir/exp_main/eps_trajectory}"
if [[ "${MEASURE}" == "kl_div" ]]; then
  base_dir="${exp_root}/results_kl_trajectory"
elif [[ "${MEASURE}" == "cmp_l2" ]]; then
  base_dir="${exp_root}/results_cmp_l2_trajectory"
else
  base_dir="${exp_root}/results_eps_trajectory"
fi
OUTPUT_DIR="${base_dir}/ddim/CFG=${cfg}_NFE=${NFE}/seed=${SEED}/batch=${batch}"

# ---- Prompt config ----
num_tp=3          # general text prompt 갯수 (ms_coco/coco_v2)
num_mtp=1         # memorized text prompt 갯수 (webster)
num_plot=20    # plot00, plot01 ... 각 plot = num_tp general + num_mtp memo

# general = ms_coco (coco_v2.txt)
text_dir="examples/assets/coco_v2.txt"
# memorized = webster (sdv1_500_mem.txt)
memo_dir="examples/assets/sdv1_500_mem.txt"

# ---- Paths ----
# ori_memo/ 에서 실행 가정; 다른 위치(예: docker)면 ROOT_DIR 로 override. python 은 PATH 의 것 사용.

echo "========================================="
echo "  Epsilon Trajectory Analysis"
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
