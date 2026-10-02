#!/bin/bash
# ===================================================================
#  ⑬ twd_loss optimization: latent x_t (lr sweep) — Chen eval
#  loss: L = w_twd_mag·Σ proxy^i  +  w_twd_std·Σ_{i<j}|proxy^i − proxy^j|
#        proxy^i = ‖ε_ref − ε_s^i‖²/D  (per-seed Tweedie gap)
#        (UNet 2회(ⓑ+ⓔ) — memo_proxy 와 동일 chain, 집계 방식만 다름)
#
#  run_memo_twd_gap.sh 패턴의 twd_loss 버전
#  run_ini_opti.py --opti_mode xt --xt_loss twd_loss
#
#  파이프라인 (각 lr 마다):
#    1. DDIM forward 0 → init_steps (no grad)
#    2. x_t optimize: twd_loss (proxy 합 + proxy 편차)
#       각 update: zt → ε_cfg → x̂₀ → x_s → ε_s → proxy → loss → backward → step
#       → DDIM gap (no grad) 전진
#    3. DDIM resume → image
#    4. eval: SSCD-to-GT + T2I
#  종료 후: trade-off curve
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/main_results/eval_web/sd14/ours/run_twd_loss.sh
# ===================================================================

# =========================== 0. [Timer] ===========================
SECONDS=0

# =========================== 1. [Parser] ===========================
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

# twd_loss optimization config
init_steps="${INIT_STEPS:-2}"
num_steps="${NUM_STEPS:-9}"
gap_steps="${GAP_STEPS:-4}"
base_s_ratio="${BASE_S_RATIO:-0.5}"
lambda_align="${LAMBDA_ALIGN:-0.0}"

# twd_loss weights (env var로 override 가능)
w_twd_mag="${W_TWD_MAG:-0.00}"
w_twd_std="${W_TWD_STD:-0.05}"

# ★ lr sweep (LR_LIST 환경변수로 오버라이드 가능)
lr_list=(${LR_LIST:-0.05 0.10 0.15 0.20})

# prompts
text_name="sdv1_500_mem.txt"
prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
t2i_prompt_dir="examples/assets/${text_name}"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# workdir — w_twd_mag/w_twd_std 를 경로에 포함
model_key="ckpt/${model}"
base_dir="workdir/memorization/sd14_base"
trd_path="${base_dir}/ours/twd_loss/CFG=${cfg}_NFE=${NFE}/cfgsr=${cfg_start_ratio}/init=${init_steps}/nsteps=${num_steps}/gap=${gap_steps}/w_twd_mag=${w_twd_mag}/w_twd_std=${w_twd_std}"

cd "${ROOT_DIR:-.}"

echo "========================================="
echo "  twd_loss x_t Optimization (lr sweep) — Chen"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  seed=${SEED}"
echo "  opti_mode=xt --xt_loss twd_loss (UNet 2회: ⓑ+ⓔ chain)"
echo "  init_steps=${init_steps}  num_steps=${num_steps}  gap=${gap_steps}"
echo "  w_twd_mag=${w_twd_mag}  w_twd_std=${w_twd_std}"
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
        --xt_loss twd_loss \
        --NFE ${NFE} --cfg ${cfg} \
        --model_key ${model_key} \
        --cfg_start_ratio ${cfg_start_ratio} ${cfgsr_cond_flag} \
        --init_steps ${init_steps} \
        --num_steps ${num_steps} --gap_steps ${gap_steps} \
        --lr ${lr} \
        --base_s_ratio ${base_s_ratio} --lambda_align ${lambda_align} \
        --w_twd_mag ${w_twd_mag} --w_twd_std ${w_twd_std} \
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
