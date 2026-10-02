#!/bin/bash
# ===================================================================
#  ⑭ twd_target optimization: latent x_t (lr sweep) — Chen eval
#  loss: L = w_twd_mean·mean(proxy) + w_twd_std·std(proxy)
#        proxy^i = ‖x_T_init − ε_s^i‖²/D  (self-referential Tweedie gap)
#        (UNet 2회(ⓑ+ⓔ) — twd_loss 와 동일 chain; eps_ref = x_T_init)
#
#  early stopping: twd_mean < twd_mean_threshold AND twd_std < twd_std_threshold
#  → opti_num 도달 전에 종료 가능
#
#  run_ini_opti.py --opti_mode xt --xt_loss twd_target
#
#  파이프라인 (각 lr 마다):
#    1. DDIM forward 0 → init_steps (no grad)
#    2. x_t optimize: twd_target (self-referential proxy mean+std)
#       early stop: mean<twd_mean_threshold AND std<twd_std_threshold
#       → DDIM gap (no grad) 전진
#    3. DDIM resume → image
#    4. eval: SSCD-to-GT + T2I
#  종료 후: trade-off curve
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/main_results/eval_web/sd14/ours/run_twd_target.sh
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

# twd_target optimization config
init_steps="${INIT_STEPS:-2}"
num_steps="${NUM_STEPS:-9}"
gap_steps="${GAP_STEPS:-4}"
base_s_ratio="${BASE_S_RATIO:-0.5}"
lambda_align="${LAMBDA_ALIGN:-0.0}"

# twd_target params
opti_num="${OPTI_NUM:-3}"                    # max iterations
w_twd_mean="${W_TWD_MEAN:-0.10}"               # weight for mean(proxy) term
w_twd_std="${W_TWD_STD:-0.10}"                 # weight for std(proxy) term
twd_mean_threshold="${TWD_MEAN_THR:-0.75}"    # early stop: twd_mean < this
twd_std_threshold="${TWD_STD_THR:-0.50}"      # early stop: twd_std < this

# ★ lr sweep (LR_LIST 환경변수로 오버라이드 가능)
lr_list=(${LR_LIST:-0.05 0.10 0.15 0.20})

# prompts
text_name="sdv1_500_mem.txt"
prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
t2i_prompt_dir="examples/assets/${text_name}"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# workdir — threshold/weight 를 경로에 포함
model_key="ckpt/${model}"
base_dir="workdir/memorization/sd14_base"
trd_path="${base_dir}/ours/twd_target/CFG=${cfg}_NFE=${NFE}/cfgsr=${cfg_start_ratio}/init=${init_steps}/nsteps=${num_steps}/gap=${gap_steps}/opti_num=${opti_num}/w_mean=${w_twd_mean}/w_std=${w_twd_std}/thr=${twd_mean_threshold}_${twd_std_threshold}"

cd "${ROOT_DIR:-.}"

echo "========================================="
echo "  twd_target x_t Optimization (lr sweep) — Chen"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  seed=${SEED}"
echo "  opti_mode=xt --xt_loss twd_target (self-referential, UNet 2회: ⓑ+ⓔ chain)"
echo "  init_steps=${init_steps}  num_steps=${num_steps}  gap=${gap_steps}  opti_num=${opti_num}"
echo "  w_twd_mean=${w_twd_mean}  w_twd_std=${w_twd_std}"
echo "  early stop: twd_mean<${twd_mean_threshold} AND twd_std<${twd_std_threshold}"
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
        --xt_loss twd_target \
        --NFE ${NFE} --cfg ${cfg} \
        --model_key ${model_key} \
        --cfg_start_ratio ${cfg_start_ratio} ${cfgsr_cond_flag} \
        --init_steps ${init_steps} \
        --num_steps ${num_steps} \
        --gap_steps ${gap_steps} \
        --opti_num ${opti_num} \
        --lr ${lr} \
        --base_s_ratio ${base_s_ratio} --lambda_align ${lambda_align} \
        --w_twd_mean ${w_twd_mean} --w_twd_std ${w_twd_std} \
        --twd_mean_threshold ${twd_mean_threshold} \
        --twd_std_threshold ${twd_std_threshold} \
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
