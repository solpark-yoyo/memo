#!/bin/bash
# ===================================================================
#  [cvpr2025] 1-1 memo_proxy / optimization: latent x_t (lr sweep)
#
#  eval_web/sd14/run_memo_twd_gap.sh 의 cvpr2025 버전 — 차이:
#    · 프롬pt: cvpr2025_memo_prompt.txt (Webster 순서로 생성)
#    · GT 이중 평가 (baseline cvpr2025 관행 동일):
#        - eval/cvpr_2025/ : cvpr2025_webster_gt 직접 매핑 (생성 순서 동일)
#        - eval/sdv1_500/  : sdv1_500_mem_groundtruth 역방향 캡션 브리지
#        - 각 폴더 sscd + t2i(복사) + total_metrics.csv
#    · trd: GT별 분리 → {trd_path}/trd/{cvpr_2025,sdv1_500}/{csv,plot}
#    · workdir: twd_gap_cvpr2025/ (eval_web twd_gap 결과와 분리)
#
#  loss: memo_proxy = ||ε_ref − ε_s(x_s, s)||²/D (ⓑ~ⓔ terminal head)
#  파이프라인: DDIM forward → x_t optimize (memo loss) → gap 전진 → resume
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/eval_cvpr2025/sd14/run_memo_twd_gap.sh
# ===================================================================

# =========================== 1. [Parser] ===========================
gpu=0
model="stable-diffusion-v1-4"
NFE=50
cfg=7.5
cfg_start_ratio="${CFGSR:-0.10}"       # staged CFG: early step conditional만
cfgsr_cond_flag=""; [[ "${cfg_start_ratio}" != "0.0" ]] && cfgsr_cond_flag="--cfgsr_cond"
SEED=42
num_samples=10
num_images_per_prompt=1
b_size=${num_images_per_prompt}
DEVICE="cuda:${gpu}"
GT_VARIANTS=(cvpr_2025 sdv1_500)

# twd_gap (1-1) optimization config
init_steps="${INIT_STEPS:-2}"
num_steps="${NUM_STEPS:-9}"
gap_steps="${GAP_STEPS:-4}"
base_s_ratio="${BASE_S_RATIO:-0.5}"
lambda_align="${LAMBDA_ALIGN:-0.1}"

# ★ lr sweep (LR_LIST 환경변수로 오버라이드 가능)
lr_list=(${LR_LIST:-0.01 0.02 0.03 0.04})

# prompts — cvpr2025 (Webster) 세트
text_name="cvpr2025_memo_prompt.txt"
prompt_dir="examples/assets/${text_name}"
t2i_prompt_dir="examples/assets/${text_name}"
webster_gt="datasets/memo/eval_sscd/sd14/cvpr2025_webster_gt"
sdv1_gt="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# workdir — eval_web twd_gap(sd v1 세트)과 분리
model_key="ckpt/${model}"
base_dir="workdir/memorization/sd14_base"
trd_path="${base_dir}/twd_gap_cvpr2025/CFG=${cfg}_NFE=${NFE}/cfgsr=${cfg_start_ratio}/init=${init_steps}/nsteps=${num_steps}/gap=${gap_steps}"

cd "${ROOT_DIR:-.}"

# =========================== 1-1. [역방향 캡션 브리지] ===========================
# 생성 순서(cvpr2025_memo_prompt 행 i) → 캡션 → sdv1 GT idx → sdv1 ref (wen 스크립트와 공유)
bridge_dir="${base_dir}/baselines/_bridge"
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
        if ref != "MISSING": n_ok += 1
        w.writerow([i, ref])
print(f"[rev-bridge] 캡션 매칭 {n_ok}/{len(cvpr)} → ${rev_csv}")
EOF
fi

echo "========================================="
echo "  [cvpr2025] memo_proxy x_t Optimization (lr sweep)"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  seed=${SEED}"
echo "  opti_mode=xt (grad ⓑ~ⓔ only, x_T 불변)"
echo "  init_steps=${init_steps}  num_steps=${num_steps}  gap=${gap_steps}"
echo "  base_s_ratio=${base_s_ratio}  lambda_align=${lambda_align}"
echo "  cfg_start_ratio=${cfg_start_ratio}  batch=${b_size}"
echo "  prompt=${text_name} (Webster)  lr_list=(${lr_list[*]})"
echo "  trd_path=${trd_path}"
echo "========================================="

# =========================== 2. [lr sweep] ===========================
for lr in "${lr_list[@]}"; do
    OUTPUT_DIR="${trd_path}/lr=${lr}/seed=${SEED}/batch=${b_size}"
    echo ""
    echo "---- lr=${lr} ----"

    # inference + optimize
    python run_ini_opti.py \
        --opti_mode xt \
        --NFE ${NFE} --cfg ${cfg} \
        --model_key ${model_key} \
        --cfg_start_ratio ${cfg_start_ratio} ${cfgsr_cond_flag} \
        --init_steps ${init_steps} \
        --num_steps ${num_steps} --gap_steps ${gap_steps} \
        --lr ${lr} \
        --base_s_ratio ${base_s_ratio} --lambda_align ${lambda_align} \
        --type_memo_loss minimization \
        --base_seed ${SEED} --num_seeds ${num_images_per_prompt} \
        --prompt_dir ${prompt_dir} --num_samples ${num_samples} \
        --device ${DEVICE} \
        --output_dir ${OUTPUT_DIR}

    # ---- 이중 GT eval: eval/{cvpr_2025,sdv1_500}/ (각 total_metrics.csv 포함) ----
    init_eval="${OUTPUT_DIR}/eval"
    mkdir -p ${init_eval}
    for OUT in "${GT_VARIANTS[@]}"; do mkdir -p "${init_eval}/${OUT}"; done

    # GT-1: webster 직접 매핑 (생성 순서 동일)
    python compute_sscd_gt.py \
        --gen_dir ${OUTPUT_DIR}/result --ref_dir ${webster_gt} \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu ${gpu} --output_csv "${init_eval}/cvpr_2025/chen_sscd_gt_metrics.csv"

    # GT-2: sdv1 역방향 브리지
    python compute_sscd_gt.py \
        --gen_dir ${OUTPUT_DIR}/result --ref_dir ${sdv1_gt} \
        --mapping_csv "${rev_csv}" \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu ${gpu} --output_csv "${init_eval}/sdv1_500/chen_sscd_gt_metrics.csv"

    # T2I (GT 무관 — 1회 측정, 양쪽 폴더에 복사)
    python -m compute_t2i_metrics \
        --eval_dir ${OUTPUT_DIR}/result --prompt_dir ${t2i_prompt_dir} \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --output_csv ${init_eval}/chen_t2i_metrics.csv \
        --device ${DEVICE} ${CS_FLAG}
    for OUT in "${GT_VARIANTS[@]}"; do
        cp "${init_eval}/chen_t2i_metrics.csv" "${init_eval}/${OUT}/chen_t2i_metrics.csv"
        python merge_benchmark.py --collect_dir "${init_eval}/${OUT}" \
            && echo "  [total] ${init_eval}/${OUT}/total_metrics.csv" || true
    done
    echo "  [CHECK] eval:"; /bin/ls ${init_eval}/*/*.csv 2>/dev/null
done

# =========================== 3. [Trade-off — GT별] ===========================
echo ""
echo "==== [trade-off] GT별 → ${trd_path}/trd/{cvpr_2025,sdv1_500}/ ===="
for GT in "${GT_VARIANTS[@]}"; do
    python collect_trd.py --method init_opti --path "${trd_path}" \
        --lr_list ${lr_list[@]} --seed ${SEED} --batch ${b_size} \
        --eval_sub ${GT} --trd_dir "${trd_path}/trd/${GT}"
    python plot_trd.py --csv "${trd_path}/trd/${GT}/csv/total_metrics.csv" \
        --out_dir "${trd_path}/trd/${GT}/plot"
done

echo ""
echo "[Done] → ${trd_path}/trd/{cvpr_2025,sdv1_500}/{csv,plot}/"
