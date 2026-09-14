#!/bin/bash
# ===================================================================
#  [trd-only] baseline 4종 — GT별 trade-off curve 생성 (inference·eval 재사용)
#
#  각 method의 trd/ 를 GT별로 분리:
#    {method trd root}/cvpr_2025/{csv,plot}/   ← eval/cvpr_2025/total_metrics.csv
#    {method trd root}/sdv1_500/{csv,plot}/   ← eval/sdv1_500/total_metrics.csv
#  (score 예: .../lr=0.01/trd/{cvpr_2025,sdv1_500}/)
#
#  knob 목록은 존재하는 디렉토리에서 자동 발견 (glob).
#  실행: ori_memo/ 에서  bash shells/eval_cvpr2025/sd14/baselines/run_trd_cvpr2025_all.sh
# ===================================================================
export PYTHONUNBUFFERED=1

seed=42
batch=1
GT_VARIANTS=(cvpr_2025 sdv1_500)

# ---------------- [1] jeon: lr 스윕 ----------------
JEON_ROOT="workdir/memorization/sd14_base/baselines/jeon/stable-diffusion-v1-4/CFG=7.5_NFE=50"
JEON_LRS=()
for d in "${JEON_ROOT}"/lr=*/thres=8.2/ms=10/seed=${seed}; do
    [[ -d "$d" ]] && JEON_LRS+=("${d##*/lr=}"); done
JEON_LRS=("${JEON_LRS[@]/\/thres=*/}")
echo "[jeon] lr: ${JEON_LRS[*]}"
for GT in "${GT_VARIANTS[@]}"; do
    python collect_trd.py --method jeon --path "${JEON_ROOT}" \
        --lr_list ${JEON_LRS[@]} --seed ${seed} --batch ${batch} --tl 8.2 --oi 10 \
        --eval_sub ${GT} --trd_dir "${JEON_ROOT}/trd/${GT}"
    python plot_trd.py --csv "${JEON_ROOT}/trd/${GT}/csv/total_metrics.csv" \
        --out_dir "${JEON_ROOT}/trd/${GT}/plot"
done

# ---------------- [2] ren: c1 스윕 ----------------
REN_ROOT="workdir/memorization/sd14_base/baselines/ren/stable-diffusion-v1-4/CFG=7.5_NFE=50"
REN_C1S=()
for d in "${REN_ROOT}"/c1=*/seed=${seed}; do
    [[ -d "$d" ]] && REN_C1S+=("${d##*/c1=}"); done
REN_C1S=("${REN_C1S[@]/\/seed=*/}")
echo "[ren] c1: ${REN_C1S[*]}"
for GT in "${GT_VARIANTS[@]}"; do
    python collect_trd.py --method ren --path "${REN_ROOT}" \
        --lr_list ${REN_C1S[@]} --seed ${seed} --batch ${batch} \
        --eval_sub ${GT} --trd_dir "${REN_ROOT}/trd/${GT}"
    python plot_trd.py --csv "${REN_ROOT}/trd/${GT}/csv/total_metrics.csv" \
        --out_dir "${REN_ROOT}/trd/${GT}/plot"
done

# ---------------- [3] score: tl 스윕 (lr 고정 0.01) ----------------
SCORE_BASE="workdir/memorization/sd14_base/baselines/init_score_noise/NFE=50/per_sample/CFG=7.0"
SCORE_PATH="${SCORE_BASE}"                       # pattern: lr=/tl=/oi=/seed=
SCORE_TL_ROOT="${SCORE_BASE}/lr=0.01"
SCORE_TLS=()
for d in "${SCORE_PATH}"/lr=0.01/tl=*/oi=1000/seed=${seed}; do
    [[ -d "$d" ]] && SCORE_TLS+=("$(basename $(dirname $(dirname "$d")))"); done
SCORE_TLS=("${SCORE_TLS[@]#tl=}")          # "tl=0.7" → "0.7"
echo "[score] tl: ${SCORE_TLS[*]}"
for GT in "${GT_VARIANTS[@]}"; do
    python collect_trd.py --method init_score_noise --path "${SCORE_PATH}" \
        --lr 0.01 --lr_list ${SCORE_TLS[@]} --seed ${seed} --batch ${batch} --oi 1000 \
        --eval_sub ${GT} --trd_dir "${SCORE_TL_ROOT}/trd/${GT}"
    python plot_trd.py --csv "${SCORE_TL_ROOT}/trd/${GT}/csv/total_metrics.csv" \
        --out_dir "${SCORE_TL_ROOT}/trd/${GT}/plot"
done

# ---------------- [4] wen: tl 스윕 ----------------
WEN_ROOT="workdir/memorization/sd14_base/baselines/wen/stable-diffusion-v1-4/CFG=7.5_NFE=50"
WEN_TLS=()
for d in "${WEN_ROOT}"/tl=*/it=10/seed=${seed}; do
    [[ -d "$d" ]] && WEN_TLS+=("${d##*/tl=}"); done
WEN_TLS=("${WEN_TLS[@]/\/it=*/}")
echo "[wen] tl: ${WEN_TLS[*]}"
for GT in "${GT_VARIANTS[@]}"; do
    python collect_trd.py --method wen --path "${WEN_ROOT}" \
        --lr_list ${WEN_TLS[@]} --seed ${seed} --batch ${batch} --oi 10 \
        --eval_sub ${GT} --trd_dir "${WEN_ROOT}/trd/${GT}"
    python plot_trd.py --csv "${WEN_ROOT}/trd/${GT}/csv/total_metrics.csv" \
        --out_dir "${WEN_ROOT}/trd/${GT}/plot"
done

echo "[Done] GT별 trd — jeon/ren/score/wen × {cvpr_2025, sdv1_500}"
