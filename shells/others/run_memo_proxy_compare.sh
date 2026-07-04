#!/bin/bash
# ===================================================================
#  Memorization-proxy 비교 파이프라인  (SD1.4 비-memorized, DDIM)
#
#  목표: text prompt 출처에 따라 memo_proxy 경향을 비교
#    - cvpr2025_memo_prompt  (Chen et al.)   -> ${BASE_OUT}/chen/plot*/
#    - new_memorized_text_prompt (Wen et al.) -> ${BASE_OUT}/wen/plot*/
#    - coco_v2                (general)       ->  각 plot 안에 text 비교군으로 포함
#    - 최종 overlay 비교 plot              -> ${BASE_OUT}/overlay_compare.png
#
#  모델: ckpt/stable-diffusion-v1-4  (비-memorized 일반 SD1.4)
#  환경: conda div_DM
#
#  실행: ori_memo/ 디렉토리에서
#       bash shells/others/run_memo_proxy_compare.sh
#  (빠른 smoke-test: SMOKE=1 bash shells/others/run_memo_proxy_compare.sh)
# ===================================================================
set -euo pipefail

# ---- conda 환경 (div_DM) ----
source /home/geonsoo/anaconda3/etc/profile.d/conda.sh
conda activate div_DM

# ---- Config (환경변수로 오버라이드 가능) ----
gpu="${GPU:-0}"
NFE="${NFE:-50}"
cfg="${CFG:-7.5}"
SEED="${SEED:-42}"
batch="${BATCH:-5}"
DEVICE="cuda:${gpu}"

num_tp="${NUM_TP:-3}"       # coco general prompts per plot
num_mtp="${NUM_MTP:-1}"     # memo prompts per plot
num_plot="${NUM_PLOT:-20}"  # 비교 plot 개수

# SMOKE=1: 빠른 검증용 기본값 (명시되지 않은 환경변수에 한해서만 적용)
#   예) SMOKE=1 BATCH=5  -> batch=5, num_tp=2, num_plot=1
if [[ "${SMOKE:-0}" == "1" ]]; then
  [[ -z "${NUM_TP:-}" ]]   && num_tp=2
  [[ -z "${NUM_MTP:-}" ]]  && num_mtp=1
  [[ -z "${NUM_PLOT:-}" ]] && num_plot=1
  [[ -z "${BATCH:-}" ]]    && batch=2
fi
num_samples=${batch}

# ---- Paths ----
MODEL="ckpt/stable-diffusion-v1-4"                       # SD1.4 (비-memorized)
TEXT_DIR="examples/assets/coco_v2.txt"                   # general prompt
CHEN_MEMO="examples/assets/cvpr2025_memo_prompt.txt"     # Chen et al. memorized
WEN_MEMO="examples/assets/new_memorized_text_prompt.txt" # Wen et al. memorized

ROOT_DIR="/home/geonsoo/Desktop/Datasets/Parksol/memo/ori_memo"
BASE_OUT="results_eps_trajectory/sd14/ddim/CFG=${cfg}_NFE=${NFE}/seed=${SEED}/batch=${batch}"
CHEN_OUT="${BASE_OUT}/chen"
WEN_OUT="${BASE_OUT}/wen"
OVERLAY="${BASE_OUT}/overlay_compare.png"

cd "${ROOT_DIR}"

echo "========================================="
echo "  Memo-proxy compare  (model=${MODEL})"
echo "  NFE=${NFE} CFG=${cfg} SEED=${SEED} batch=${batch}"
echo "  num_tp=${num_tp}(coco)  num_mtp=${num_mtp}(memo)  num_plot=${num_plot}"
echo "  Chen memo: ${CHEN_MEMO}"
echo "  Wen  memo: ${WEN_MEMO}"
echo "  General  : ${TEXT_DIR}"
echo "  OUT      : ${BASE_OUT}"
echo "========================================="

# ---- 1. Chen (cvpr2025) ----
echo ""
echo "############ [1/3] Chen (cvpr2025_memo_prompt) -> ${CHEN_OUT} ############"
python eps_trajectory.py \
    --model_key "${MODEL}" \
    --text_dir "${TEXT_DIR}" --memo_dir "${CHEN_MEMO}" \
    --output_dir "${CHEN_OUT}" \
    --num_tp ${num_tp} --num_mtp ${num_mtp} --num_plot ${num_plot} \
    --num_inference_steps ${NFE} --cfg_guidance ${cfg} \
    --seed ${SEED} --num_samples ${num_samples} --device "${DEVICE}"

# ---- 2. Wen (new_memorized) ----
echo ""
echo "############ [2/3] Wen (new_memorized_text_prompt) -> ${WEN_OUT} ############"
python eps_trajectory.py \
    --model_key "${MODEL}" \
    --text_dir "${TEXT_DIR}" --memo_dir "${WEN_MEMO}" \
    --output_dir "${WEN_OUT}" \
    --num_tp ${num_tp} --num_mtp ${num_mtp} --num_plot ${num_plot} \
    --num_inference_steps ${NFE} --cfg_guidance ${cfg} \
    --seed ${SEED} --num_samples ${num_samples} --device "${DEVICE}"

# ---- 3. Overlay 비교 ----
echo ""
echo "############ [3/3] Overlay compare -> ${OVERLAY} ############"
python plot_memo_proxy_overlay.py \
    --chen_dir "${CHEN_OUT}" --wen_dir "${WEN_OUT}" \
    --num_tp ${num_tp} \
    --output "${OVERLAY}"

echo ""
echo "[Done] overlay: ${OVERLAY}"
echo "       chen  : ${CHEN_OUT}"
echo "       wen   : ${WEN_OUT}"
