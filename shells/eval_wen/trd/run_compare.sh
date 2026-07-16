#!/bin/bash
# ===================================================================
#  최종 모델 비교 trade-off curve: Han et al. vs ours (overlay)
#
#   각 path 아래 trd/csv/total_metrics.csv 를 가져와 한 plot 에 겹쳐 그림.
#   → {OUT_BASE}/trd/compare_{clipscore,pickscore,imagereward}.png
#
#  실행 (ori_memo/ 에서, conda div_DM 활성화 후):
#    HAN_PATH=workdir/.../init_score_noise/NFE=50/per_sample/CFG=7.5 \
#    OURS_PATH=workdir/.../init_opti/.../gap=3 \
#        bash shells/eval_wen/trd/run_compare.sh
# ===================================================================

# =========================== 1. [Parser] ===========================
HAN_PATH="${HAN_PATH:?HAN_PATH 필수 (Han et al. trd 부모 — 그 아래 trd/csv/ 가 있어야 함)}"
OURS_PATH="${OURS_PATH:?OURS_PATH 필수 (ours trd 부모)}"
OUT_BASE="${OUT_BASE:-workdir/memorization/sd14_memor_LAION2B_40k}"   # 최종 비교 산출물 부모

cd "${ROOT_DIR:-.}"                    # ori_memo/ 에서 실행 가정

HAN_CSV="${HAN_PATH}/trd/csv/total_metrics.csv"
OURS_CSV="${OURS_PATH}/trd/csv/total_metrics.csv"

echo "========================================="
echo "  model comparison (Han vs ours)"
echo "  Han csv : ${HAN_CSV}"
echo "  Ours csv: ${OURS_CSV}"
echo "  OUT     : ${OUT_BASE}/trd/"
echo "========================================="

# =========================== 2. [plot] ===========================
python plot_compare.py \
    --han_csv  "${HAN_CSV}" \
    --ours_csv "${OURS_CSV}" \
    --out_dir  "${OUT_BASE}/trd"

echo ""
echo "[Done] → ${OUT_BASE}/trd/compare_{clipscore,pickscore,imagereward}.png"
