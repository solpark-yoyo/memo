#!/bin/bash
# ===================================================================
#  [eval-only] init_score_noise (Han) — 이중 GT SSCD eval (inference 재사용)
#  GT-1: cvpr2025_webster_gt (캡션 브리지) → eval/cvpr_2025/
#  GT-2: sdv1_500_mem_groundtruth (직접 매핑) → eval/sdv1_500/
#  생성: eval_web/sd14/baselines/run_memo_chen_score.sh 결과
#  주의: Han 러너는 이미지가 gen_dir 바로 아래 (result/ 서브디렉토리 없음)
#
#  실행: ori_memo/ 에서  bash shells/eval_cvpr2025/sd14/baselines/run_eval_cvpr2025_score.sh
# ===================================================================
export PYTHONUNBUFFERED=1

# =========================== 1. [Config] ===========================
device="${DEVICE:-cuda:0}"
gpu="${device##*:}"
seed=42
num_samples=10
num_images_per_prompt=1

NFE=50
cfg_initnoise=7.0
lr=0.01
optim_iters=1000
# 존재하는 tl 디렉토리 전체 glob 순회 (0.7~1.5)

webster_gt="datasets/memo/eval_sscd/sd14/cvpr2025_webster_gt"
sdv1_gt="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
output_path="workdir/memorization/sd14_base/baselines/init_score_noise/NFE=${NFE}"

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
for gen_dir in "${output_path}"/per_sample/CFG="${cfg_initnoise}"/lr="${lr}"/tl=*/oi="${optim_iters}"/seed="${seed}"; do
    if ! ls "${gen_dir}"/img_*.png >/dev/null 2>&1; then
        echo "[skip] ${gen_dir} 이미지 없음"; continue
    fi
    echo ""
    echo "---- score ${gen_dir##*/tl=} (tl=$(basename $(dirname $(dirname ${gen_dir})))) ----"
    for OUT in cvpr_2025 sdv1_500; do mkdir -p "${gen_dir}/eval/${OUT}"; done

    python compute_sscd_gt.py \
        --gen_dir "${gen_dir}" --ref_dir "${webster_gt}" \
        --mapping_csv "${bridge_csv}" \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu "${gpu}" \
        --output_csv "${gen_dir}/eval/cvpr_2025/chen_sscd_gt_metrics.csv"

    python compute_sscd_gt.py \
        --gen_dir "${gen_dir}" --ref_dir "${sdv1_gt}" \
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

echo "[Done] init_score_noise 이중 GT eval — per_sample/.../eval/{cvpr_2025,sdv1_500}/"
