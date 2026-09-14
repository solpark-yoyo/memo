#!/bin/bash
# ===================================================================
#  1-1 memo_proxy / optimization: latent x_t (lr sweep) — Chen eval
#  loss: memo_proxy = ||ε_ref − ε_s(x_s, s)||²/D
#        (x_t → ε_cfg → Tweedie x̂₀ → x_s → ε_s — ⓑ~ⓔ terminal head)
#
#  run_memo_spectral_eps.sh 패턴의 twd_gap 버전 — 최적화 엔트리만
#  run_ini_opti.py --opti_mode xt (optimize_xt: 중간 latent x_t 를 leaf 로
#  순차 갱신, grad 는 ⓑ~ⓔ만 흐르고 x_T 불변, AdjointDPM 불필요).
#
#  파이프라인 (각 lr 마다):
#    1. DDIM forward 0 → init_steps (no grad)
#    2. x_t optimize: memo_proxy minimize
#       각 update: zt → ε_cfg → x̂₀ → x_s → ε_s → loss → backward → step
#       → DDIM gap (no grad) 전진
#    3. DDIM resume → image
#    4. eval: SSCD-to-GT + T2I
#  종료 후: trade-off curve
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/eval_chen/run_memo_twd_gap.sh
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
num_images_per_prompt=5
b_size=${num_images_per_prompt}
DEVICE="cuda:${gpu}"

# eps_ref_mse (⑤) optimization config
init_steps="${INIT_STEPS:-2}"
num_steps="${NUM_STEPS:-9}"
gap_steps="${GAP_STEPS:-4}"
base_s_ratio="${BASE_S_RATIO:-0.5}"
lambda_align="${LAMBDA_ALIGN:-0.1}"
xt_loss="${XT_LOSS:-eps_ref_mse}"   # ⑤ (기본) | memo_proxy(①) — --xt_loss 갈래

# ★ lr sweep (LR_LIST 환경변수로 오버라이드 가능)
lr_list=(${LR_LIST:-0.00 0.04 0.08 0.12})

# prompts
text_name="cvpr2025_memo_prompt.txt"
prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/cvpr2025_webster_gt"
t2i_prompt_dir="examples/assets/${text_name}"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# workdir
model_key="ckpt/${model}"
base_dir="workdir/memorization/sd14_base"
trd_path="${base_dir}/eps_ref_mse/CFG=${cfg}_NFE=${NFE}/cfgsr=${cfg_start_ratio}/init=${init_steps}/nsteps=${num_steps}/gap=${gap_steps}"

cd "${ROOT_DIR:-.}"

echo "========================================="
echo "  memo_proxy x_t Optimization (lr sweep) — Chen"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  seed=${SEED}"
echo "  opti_mode=xt (grad ⓑ~ⓔ only, x_T 불변)"
echo "  init_steps=${init_steps}  num_steps=${num_steps}  gap=${gap_steps}"
echo "  base_s_ratio=${base_s_ratio}  lambda_align=${lambda_align}"
echo "  cfg_start_ratio=${cfg_start_ratio}  batch=${b_size}"
echo "  lr_list=(${lr_list[*]})"
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
        --xt_loss ${xt_loss} \
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

    # eval
    init_eval="${OUTPUT_DIR}/eval"
    mkdir -p ${init_eval}
    python compute_sscd_gt.py \
        --gen_dir ${OUTPUT_DIR}/result --ref_dir ${gt_ref_dir} \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu ${gpu} --output_csv ${init_eval}/chen_sscd_gt_metrics.csv
    python -m compute_t2i_metrics \
        --eval_dir ${OUTPUT_DIR}/result --prompt_dir ${t2i_prompt_dir} \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --output_csv ${init_eval}/chen_t2i_metrics.csv \
        --device ${DEVICE} ${CS_FLAG}
    python merge_benchmark.py --collect_dir ${init_eval}
    echo "  [CHECK] eval:"; /bin/ls ${init_eval}/*.csv 2>/dev/null
done

# =========================== 3. [Trade-off] ===========================
echo ""
echo "==== [trade-off] collect_trd + plot_trd → ${trd_path}/trd/ ===="
python collect_trd.py --method init_opti --path "${trd_path}" \
    --lr_list ${lr_list[@]} --seed ${SEED} --batch ${b_size}
python plot_trd.py --csv "${trd_path}/trd/csv/total_metrics.csv" --out_dir "${trd_path}/trd/plot"

echo ""
echo "[Done] → ${trd_path}/trd/{csv,plot}/"
