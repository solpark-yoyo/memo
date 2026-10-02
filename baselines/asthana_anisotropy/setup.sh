#!/bin/bash
# asthana_anisotropy baseline 초기 설정
# pipe.py 등 대용량 파일을 원본 repo에서 받아옴
# 실행: bash baselines/asthana_anisotropy/setup.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="https://raw.githubusercontent.com/rohanasthana/memorization-anisotropy/main"

echo "[setup] fetching local_model/pipe.py ..."
curl -fsSL "${REPO}/local_model/pipe.py" -o "${SCRIPT_DIR}/local_model/pipe.py"

echo "[setup] fetching prompt files ..."
for f in sd1_mem.txt sd1_nmem.txt sd2_mem.txt sd2_nmem.txt RV_mem.txt RV_nmem.txt sample_mitigation.txt; do
    curl -fsSL "${REPO}/prompts/${f}" -o "${SCRIPT_DIR}/prompts/${f}" 2>/dev/null || true
done

echo "[setup] done. pipe.py + prompts ready."
