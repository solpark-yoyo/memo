#!/bin/bash
# ===================================================================
#  [eval-only] Ren MemAttn — 이중 GT SSCD eval (inference 재사용)
#  GT-1: cvpr2025_webster_gt (캡션 브리지) → eval/cvpr_2025/
#  GT-2: sdv1_500_mem_groundtruth (직접 매핑) → eval/sdv1_500/
#  생성: eval_web/sd14/baselines/run_memo_chen_ren.sh 결과
#
#  실행: ori_memo/ 에서  bash shells/eval_cvpr2025/sd14/baselines/run_eval_cvpr2025_ren.sh
# ===================================================================
export PYTHONUNBUFFERED=1

# =========================== 1. [Config] ===========================
device="${DEVICE:-cuda:0}"
gpu="${device##*:}"
seed=42
num_samples=10
num_images_per_prompt=1
model_tag="stable-diffusion-v1-4"

# 존재하는 c1 디렉토리 전체 glob 순회

webster_gt="datasets/memo/eval_sscd/sd14/cvpr2025_webster_gt"
sdv1_gt="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
output_path="workdir/memorization/sd14_base/baselines/ren/${model_tag}/CFG=7.5_NFE=50"

# =========================== 2. [캡션 브리지] ===========================
bridge_dir="workdir/memorization/sd14_base/baselines/_bridge"
bridge_csv="${bridge_dir}/cvpr2025_webster_bridge.csv"
if [[ ! -f "${bridge_csv}" ]]; then
    bash "$(dirname "$0")/run_eval_cvpr2025_ddim.sh" >/dev/null 2>&1 || true
fi
if [[ ! -f "${bridge_csv}" ]]; then
    echo "[error] 브리지 csv 생성 실패 — run_eval_cvpr2025_ddim.sh 의 브리지 섹션 확인"; exit 1
fi

# =========================== 3. [이중 SSCD eval] ===========================
for gen_dir in "${output_path}"/c1=*/seed="${seed}"; do
    if [[ ! -d "${gen_dir}/result" ]]; then
        echo "[skip] ${gen_dir}/result 없음"; continue
    fi
    echo ""
    echo "---- ren ${gen_dir%%/seed*} (${gen_dir%%/seed*##*/c1=}) ----"
    for OUT in cvpr_2025 sdv1_500; do mkdir -p "${gen_dir}/eval/${OUT}"; done

    python compute_sscd_gt.py \
        --gen_dir "${gen_dir}/result" --ref_dir "${webster_gt}" \
        --mapping_csv "${bridge_csv}" \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu "${gpu}" \
        --output_csv "${gen_dir}/eval/cvpr_2025/chen_sscd_gt_metrics.csv"

    python compute_sscd_gt.py \
        --gen_dir "${gen_dir}/result" --ref_dir "${sdv1_gt}" \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu "${gpu}" \
        --output_csv "${gen_dir}/eval/sdv1_500/chen_sscd_gt_metrics.csv"

    # ---- total_metrics.csv 조립: GT 무관한 T2I 복사 + merge ----
    # T2I(CLIP/Pick/ImgR)는 (이미지, 프롬pt)만 보므로 GT와 무관 — 부모 eval의 t2i 재사용
    if [[ -f "${gen_dir}/eval/chen_t2i_metrics.csv" ]]; then
        for OUT in cvpr_2025 sdv1_500; do
            cp "${gen_dir}/eval/chen_t2i_metrics.csv" "${gen_dir}/eval/${OUT}/chen_t2i_metrics.csv"
        done
        for OUT in cvpr_2025 sdv1_500; do
            python merge_benchmark.py --collect_dir "${gen_dir}/eval/${OUT}" \
                && echo "  [total] ${gen_dir}/eval/${OUT}/total_metrics.csv" || true
        done
    else
        echo "[warn] t2i 없음 — total_metrics 미생성: ${gen_dir}/eval/chen_t2i_metrics.csv"
    fi
done

echo "[Done] ren 이중 GT eval — ${output_path}/c1=*/seed=42/eval/{cvpr_2025,sdv1_500}/"
