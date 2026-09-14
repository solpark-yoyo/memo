#!/bin/bash
# ===================================================================
#  [eval-only] Wen prompt-aug — 이중 GT SSCD eval (inference 재사용)
#
#  ★ wen 세대 결과는 cvpr2025_memo_prompt 순서로 생성됨 (구 run_memo_chen_wen.sh):
#    - GT-1 cvpr2025_webster_gt: 순서 동일 → 직접 매핑 (브리지 불필요)
#    - GT-2 sdv1_500_mem_groundtruth: 역방향 캡션 브리지 (cvpr idx→캡션→sdv1 idx→ref)
#  → eval/cvpr_2025/ + eval/sdv1_500/ 각각 sscd + t2i(복사) + total_metrics.csv
#
#  실행: ori_memo/ 에서  bash shells/eval_cvpr2025/sd14/baselines/run_eval_cvpr2025_wen.sh
# ===================================================================
export PYTHONUNBUFFERED=1

# =========================== 1. [Config] ===========================
device="${DEVICE:-cuda:0}"
gpu="${device##*:}"
seed=42
num_samples=10
num_images_per_prompt=1
model_tag="stable-diffusion-v1-4"

optim_iters=10
target_loss_list=(0.7 1.3 1.9 2.4 3.0)

webster_gt="datasets/memo/eval_sscd/sd14/cvpr2025_webster_gt"
sdv1_gt="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
output_path="workdir/memorization/sd14_base/baselines/wen/${model_tag}/CFG=7.5_NFE=50"

# =========================== 2. [역방향 캡션 브리지] ===========================
# 생성 순서(cvpr2025_memo_prompt 행 i) → 캡션 → sdv1 GT idx → sdv1 ref
bridge_dir="workdir/memorization/sd14_base/baselines/_bridge"
rev_csv="${bridge_dir}/sdv1_500_rev_bridge.csv"
if [[ ! -f "${rev_csv}" ]]; then
python - <<EOF
import csv, os
cvpr = [l.strip() for l in open("examples/assets/cvpr2025_memo_prompt.txt") if l.strip()]
sdv1 = [l.strip() for l in open("examples/assets/sdv1_500_mem.txt") if l.strip()]
sdv1_idx = {c: i for i, c in enumerate(sdv1)}
idx2ref = {}
for row in csv.DictReader(open("${sdv1_gt}/prompt_to_ref.csv")):
    idx2ref[int(row["prompt_idx"])] = row["ref_file"]
os.makedirs("${bridge_dir}", exist_ok=True)
with open("${rev_csv}", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["prompt_idx", "ref_file"])
    n_ok = 0
    for i, c in enumerate(cvpr):
        j = sdv1_idx.get(c)
        ref = idx2ref.get(j, "MISSING") if j is not None else "MISSING"
        if ref != "MISSING":
            n_ok += 1
        w.writerow([i, ref])
print(f"[rev-bridge] 캡션 매칭 {n_ok}/{len(cvpr)} → ${rev_csv}")
EOF
fi

# =========================== 3. [이중 SSCD eval + total] ===========================
for tl in "${target_loss_list[@]}"; do
    gen_dir="${output_path}/tl=${tl}/it=${optim_iters}/seed=${seed}"
    if [[ ! -d "${gen_dir}/result" ]]; then
        echo "[skip] ${gen_dir}/result 없음"; continue
    fi
    echo ""
    echo "---- wen tl=${tl} ----"
    for OUT in cvpr_2025 sdv1_500; do mkdir -p "${gen_dir}/eval/${OUT}"; done

    echo "== [GT-1] cvpr2025_webster_gt (직접 매핑 — 생성 순서 동일) =="
    python compute_sscd_gt.py \
        --gen_dir "${gen_dir}/result" --ref_dir "${webster_gt}" \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu "${gpu}" \
        --output_csv "${gen_dir}/eval/cvpr_2025/chen_sscd_gt_metrics.csv"

    echo "== [GT-2] sdv1_500_mem_groundtruth (역방향 브리지) =="
    python compute_sscd_gt.py \
        --gen_dir "${gen_dir}/result" --ref_dir "${sdv1_gt}" \
        --mapping_csv "${rev_csv}" \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu "${gpu}" \
        --output_csv "${gen_dir}/eval/sdv1_500/chen_sscd_gt_metrics.csv"

    # ---- total_metrics.csv 조립: GT 무관한 T2I 복사 + merge ----
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

echo "[Done] wen 이중 GT eval — ${output_path}/tl=*/it=10/seed=42/eval/{cvpr_2025,sdv1_500}/"
