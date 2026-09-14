#!/bin/bash
# ===================================================================
#  [eval-only] DDIM 기준선 — 이중 GT SSCD eval (inference 재사용)
#  GT-1: cvpr2025_webster_gt (캡션 브리지) → eval/cvpr_2025/
#  GT-2: sdv1_500_mem_groundtruth (직접 매핑) → eval/sdv1_500/
#  생성: eval_web/sd14/baselines/run_memo_chen_ddim.sh 결과 (sdv1_500_mem 순서)
#
#  실행: ori_memo/ 에서  bash shells/eval_cvpr2025/sd14/baselines/run_eval_cvpr2025_ddim.sh
# ===================================================================
export PYTHONUNBUFFERED=1

# =========================== 1. [Config] ===========================
device="${DEVICE:-cuda:0}"
gpu="${device##*:}"
seed=42
num_samples=10
num_images_per_prompt=1
model="stable-diffusion-v1-4"

gen_dir="workdir/memorization/sd14_base/baselines/ddim/${model}/CFG=7.5_NFE=50/seed=${seed}/batch=1"
webster_gt="datasets/memo/eval_sscd/sd14/cvpr2025_webster_gt"
sdv1_gt="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"

# =========================== 2. [캡션 브리지] ===========================
# 생성 순서(sdv1_500_mem.txt 행 i) → 캡션 일치 → webster GT ref (없으면 MISSING)
bridge_dir="workdir/memorization/sd14_base/baselines/_bridge"
bridge_csv="${bridge_dir}/cvpr2025_webster_bridge.csv"
if [[ ! -f "${bridge_csv}" ]]; then
python - <<EOF
import csv, os
gen = [l.strip() for l in open("examples/assets/sdv1_500_mem.txt") if l.strip()]
web = [l.strip() for l in open("examples/assets/cvpr2025_memo_prompt.txt") if l.strip()]
cap2ref = {}
for row in csv.DictReader(open("${webster_gt}/prompt_to_ref.csv")):
    cap2ref[web[int(row["prompt_idx"])]] = row["ref_file"]
os.makedirs("${bridge_dir}", exist_ok=True)
with open("${bridge_csv}", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["prompt_idx", "ref_file"])
    for i, c in enumerate(gen):
        w.writerow([i, cap2ref.get(c, "MISSING")])
n_ok = sum(1 for c in gen if c in cap2ref)
print(f"[bridge] 캡션 매칭 {n_ok}/{len(gen)} → ${bridge_csv}")
EOF
fi

# =========================== 3. [이중 SSCD eval] ===========================
for OUT in cvpr_2025 sdv1_500; do mkdir -p "${gen_dir}/eval/${OUT}"; done

echo "==== [GT-1] cvpr2025_webster_gt (브리지) → eval/cvpr_2025/ ===="
python compute_sscd_gt.py \
    --gen_dir "${gen_dir}/result" --ref_dir "${webster_gt}" \
    --mapping_csv "${bridge_csv}" \
    --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
    --gpu "${gpu}" \
    --output_csv "${gen_dir}/eval/cvpr_2025/chen_sscd_gt_metrics.csv"

echo "==== [GT-2] sdv1_500_mem_groundtruth → eval/sdv1_500/ ===="
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

echo "[Done] ${gen_dir}/eval/{cvpr_2025,sdv1_500}/chen_sscd_gt_metrics.csv"
