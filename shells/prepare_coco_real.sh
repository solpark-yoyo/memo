#!/bin/bash
# ===================================================================
#  Resize COCO val2014 → 512×512 (center-crop) for PRDC real set
# ===================================================================

SHELL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SHELL_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

src="datasets/ms_coco/ori/val2014"
size=256
dst="datasets/ms_coco/resize/val2014_${size}"
workers=32

echo "================== [INFO]: Resize val2014 → ${size}×${size} =================="
python prepare_coco_real.py \
    --src "${src}" --dst "${dst}" \
    --size ${size} --workers ${workers}

echo "[Done] ${dst}"
