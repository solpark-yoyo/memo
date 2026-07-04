#!/bin/bash
# =============================================================================
#  Epsilon Trajectory Analysis for Memorization
#  x_t ──(Tweedie)──> x̂_0 ──(DDIM forward)──> x_s
#  Plot: ||ε - ε_s||² / D  vs denoising step   (D = noise dim)
#
#  각 plot = num_tp 개 text prompt (coco_v2) + num_mtp 개 memo prompt (membench)
#  num_plot 개의 비교 plot 생성
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

# save path: results_eps_trajectory/ddim/CFG/seed/batch
base_dir="results_eps_trajectory"
OUTPUT_DIR="${base_dir}/ddim/CFG=${cfg}_NFE=${NFE}/seed=${SEED}/batch=${batch}"

# ---- Prompt config ----
num_tp=3          # text prompt 갯수 (coco)
num_mtp=1         # memorized text prompt 갯수 (membench)
num_plot=20    # plot00, plot01 ... 각 plot = num_tp text + num_mtp memo

text_dir="examples/assets/coco_v2.txt"
memo_dir="examples/assets/memorized_prompts_membench.txt"

# ---- Paths ----
ROOT_DIR="/home/geonsoo/Desktop/Datasets/Parksol/memo/ori_memo"
PYTHON="/home/geonsoo/anaconda3/bin/python"

echo "========================================="
echo "  Epsilon Trajectory Analysis"
echo "  NFE=${NFE}  CFG=${cfg}  SEED=${SEED}  NUM_SAMPLES=${NUM_SAMPLES}"
echo "  num_tp=${num_tp}  num_mtp=${num_mtp}  num_plot=${num_plot}"
echo "  text_dir=${text_dir}"
echo "  memo_dir=${memo_dir}"
echo "========================================="

cd "${ROOT_DIR}"

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
    --num_plot ${num_plot}

echo "Done."
