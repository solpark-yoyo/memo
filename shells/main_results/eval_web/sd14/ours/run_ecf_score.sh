#!/bin/bash
# ===================================================================
#  ⑯ ecf_score optimization: latent x_t (lr sweep) — Chen eval
#  loss: L = Σ_k[(A(w_k)-exp(-w_k²/2))²+B(w_k)²] / Σ_k w_k
#        score = ε_c (_nc_p, ⓑ UNet 조건부 예측, 추가 호출 없음)
#        per-sample 정규화: z_i = (ε_c^i - μ_i)/σ_i  (C×H×W=16384 차원)
#        → ECF가 N(0,1)에서 벗어날수록 loss↑ → ε_c Gaussianity 강제
#        ∇ UNet 통과 → zt_leaf (⑮ ecf_loss 의 latent 직접 버전과 대비)
#
#  run_ini_opti.py --opti_mode xt --xt_loss ecf_score
#
#  파이프라인 (각 lr 마다):
#    1. DDIM forward 0 → init_steps (no grad)
#    2. x_t optimize: ecf_score (ε_c ECF 기반 Gaussianity, ∇ UNet 경유)
#       각 update: zt → (ⓑ) ε_c → z_normalize → ECF → loss → backward → step
#       → DDIM gap (no grad) 전진
#    3. DDIM resume → image
#    4. eval: SSCD-to-GT + T2I
#  종료 후: trade-off curve
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/main_results/eval_web/sd14/ours/run_ecf_score.sh
# ===================================================================

# =========================== 0. [Timer] ===========================
SECONDS=0

# =========================== 1. [Config] ===========================
gpu=0
model="stable-diffusion-v1-4"
NFE=50
cfg=7.5
cfg_start_ratio="${CFGSR:-0.15}"
cfgsr_cond_flag=""; [[ "${cfg_start_ratio}" != "0.0" ]] && cfgsr_cond_flag="--cfgsr_cond"
SEED=42
num_samples=10
num_images_per_prompt=4
b_size=${num_images_per_prompt}

num_eval_prompts="${NUM_EVAL_PROMPTS:-${num_samples}}"
(( num_eval_prompts > num_samples )) && num_eval_prompts=${num_samples}
num_eval=$((num_eval_prompts * num_images_per_prompt))
DEVICE="cuda:${gpu}"

# ecf_score optimization config
init_steps="${INIT_STEPS:-2}"
num_steps="${NUM_STEPS:-9}"
gap_steps="${GAP_STEPS:-4}"
base_s_ratio="${BASE_S_RATIO:-0.5}"
lambda_align="${LAMBDA_ALIGN:-0.0}"

# ECF hyperparams (env var로 override 가능)
ecf_K="${ECF_K:-10}"
ecf_w_max="${ECF_W_MAX:-3.0}"

# ★ lr sweep (LR_LIST 환경변수로 오버라이드 가능)
lr_list=(${LR_LIST:-0.05 0.10 0.15 0.20})

# prompts
text_name="sdv1_500_mem.txt"
prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
t2i_prompt_dir="examples/assets/${text_name}"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# workdir
model_key="ckpt/${model}"
base_dir="workdir/memorization/sd14_base"
trd_path="${base_dir}/ours/ecf_score/CFG=${cfg}_NFE=${NFE}/cfgsr=${cfg_start_ratio}/init=${init_steps}/nsteps=${num_steps}/gap=${gap_steps}/K=${ecf_K}/wmax=${ecf_w_max}"

cd "${ROOT_DIR:-.}"

echo "========================================="
echo "  ecf_score x_t Optimization (lr sweep) — Chen"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  seed=${SEED}"
echo "  opti_mode=xt --xt_loss ecf_score (UNet 1회: ⓑ→ε_c ECF, ∇UNet→zt_leaf)"
echo "  init_steps=${init_steps}  num_steps=${num_steps}  gap=${gap_steps}"
echo "  ecf_K=${ecf_K}  ecf_w_max=${ecf_w_max}"
echo "  cfg_start_ratio=${cfg_start_ratio}  batch=${b_size}"
echo "  lr_list=(${lr_list[*]})"
echo "  num_eval=${num_eval} imgs (eval prompts=${num_eval_prompts})"
echo "  trd_path=${trd_path}"
echo "========================================="

# =========================== 2. [lr sweep] ===========================
for lr in "${lr_list[@]}"; do
    OUTPUT_DIR="${trd_path}/batch=${b_size}/seed=${SEED}/lr=${lr}"
    echo ""
    echo "---- lr=${lr} ----"

    python run_ini_opti.py \
        --opti_mode xt \
        --xt_loss ecf_score \
        --NFE ${NFE} --cfg ${cfg} \
        --model_key ${model_key} \
        --cfg_start_ratio ${cfg_start_ratio} ${cfgsr_cond_flag} \
        --init_steps ${init_steps} \
        --num_steps ${num_steps} --gap_steps ${gap_steps} \
        --lr ${lr} \
        --base_s_ratio ${base_s_ratio} --lambda_align ${lambda_align} \
        --ecf_K ${ecf_K} --ecf_w_max ${ecf_w_max} \
        --type_memo_loss minimization \
        --base_seed ${SEED} --num_seeds ${num_images_per_prompt} \
        --prompt_dir ${prompt_dir} --num_samples ${num_samples} \
        --device ${DEVICE} \
        --output_dir ${OUTPUT_DIR}

    init_eval="${OUTPUT_DIR}/eval/${num_eval_prompts}"
    mkdir -p ${init_eval}
    python compute_sscd_gt.py \
        --gen_dir ${OUTPUT_DIR}/result --ref_dir ${gt_ref_dir} \
        --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu ${gpu} --output_csv ${init_eval}/chen_sscd_gt_metrics.csv
    python -m compute_t2i_metrics \
        --eval_dir ${OUTPUT_DIR}/result --prompt_dir ${t2i_prompt_dir} \
        --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_images_per_prompt} \
        --output_csv ${init_eval}/chen_t2i_metrics.csv \
        --device ${DEVICE} ${CS_FLAG}
    python merge_benchmark.py --collect_dir ${init_eval}
    echo "  [CHECK] eval:"; /bin/ls ${init_eval}/*.csv 2>/dev/null
done

# =========================== 3. [Trade-off] ===========================
trd_dir="${trd_path}/batch=${b_size}/seed=${SEED}/trd/${num_eval_prompts}"
echo ""
echo "==== [trade-off] collect_trd + plot_trd → ${trd_dir}/ ===="
python collect_trd.py --method init_opti --path "${trd_path}" \
    --lr_list ${lr_list[@]} --seed ${SEED} --batch ${b_size} --eval_sub ${num_eval_prompts} \
    --trd_dir "${trd_dir}"
python plot_trd.py --csv "${trd_dir}/csv/total_metrics.csv" --out_dir "${trd_dir}/plot"

echo ""
echo "[Done] → ${trd_dir}/{csv,plot}/"
echo "[Elapsed] $((SECONDS/3600))hrs $(( (SECONDS%3600)/60 ))min $((SECONDS%60))sec (total ${SECONDS}sec)"
