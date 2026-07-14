#!/bin/bash
# ===================================================================
#  Webster Memorized GT 다운로드 (HuggingFace → datasets/cvpr2025_webster_gt)
#
#  사용:  bash download_webster_gt.sh
# ===================================================================
set -e
cd "$(dirname "$0")"

echo "================== [INFO]: Download Webster GT =================="
huggingface-cli download \
    ParkSol/cvpr2025_webster_gt \
    --repo-type dataset \
    --local-dir datasets/cvpr2025_webster_gt

echo ""
echo "[Done] → datasets/cvpr2025_webster_gt/"
ls datasets/cvpr2025_webster_gt/*.jpg 2>/dev/null | wc -l
echo "장 다운로드 완료"
