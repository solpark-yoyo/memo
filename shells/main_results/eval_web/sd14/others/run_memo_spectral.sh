#!/bin/bash
# ===================================================================
#  x_t spectral optimization (lr sweep) — Chen eval
#  inference timestep t 에서 x_t 의 compact spectral L2² 를 minimize.
#
#  파이프라인 (각 lr 마다):
#    1. DDIM forward 0 → init_steps (no grad)
#    2. x_t optimize: spectral L2² minimize (FFT only, no DDIM chain backward)
#       각 update: zt → FFT loss → backward → step → detach → DDIM gap (no grad)
#    3. DDIM continue → image
#    4. eval: SSCD-to-GT + T2I
#  종료 후: trade-off curve
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/eval_chen/run_memo_spectral.sh
# ===================================================================

# =========================== 1. [Parser] ===========================
gpu=0
model="stable-diffusion-v1-4"
NFE=50
cfg=7.5
cfg_start_ratio="${CFGSR:-0.05}"       # staged CFG: early step conditional만
cfgsr_cond_flag=""; [[ "${cfg_start_ratio}" != "0.0" ]] && cfgsr_cond_flag="--cfgsr_cond"
SEED=42
num_samples=10
num_images_per_prompt=5
b_size=${num_images_per_prompt}
DEVICE="cuda:${gpu}"

# spectral optimization config
init_steps="${INIT_STEPS:-2}"
num_steps="${NUM_STEPS:-5}"
gap_steps="${GAP_STEPS:-4}"
block_size="${BLOCK_SIZE:-16}"

# ★ lr sweep (cfgsr=0.15 nsteps 축: NUM_STEPS 환경변수로 3/5/10 — 5는 기존)
lr_list=(0.00 0.04 0.08 0.12)

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
trd_path="${base_dir}/spectral_opt/loss=xt/CFG=${cfg}_NFE=${NFE}/cfgsr=${cfg_start_ratio}/init=${init_steps}/nsteps=${num_steps}/gap=${gap_steps}"

cd "${ROOT_DIR:-.}"

echo "========================================="
echo "  x_t Spectral Optimization (lr sweep) — Chen"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  seed=${SEED}"
echo "  init_steps=${init_steps}  num_steps=${num_steps}  gap=${gap_steps}"
echo "  cfg_start_ratio=${cfg_start_ratio}  block_size=${block_size}  batch=${b_size}"
echo "  lr_list=(${lr_list[*]})"
echo "  trd_path=${trd_path}"
echo "========================================="

# =========================== 2. [lr sweep] ===========================
for lr in "${lr_list[@]}"; do
    OUTPUT_DIR="${trd_path}/lr=${lr}/seed=${SEED}/batch=${b_size}"
    echo ""
    echo "---- lr=${lr} ----"

    # inference + optimize
    python optimize_xt_spectral.py \
        --NFE ${NFE} --cfg ${cfg} \
        --model_key ${model_key} \
        --cfg_start_ratio ${cfg_start_ratio} ${cfgsr_cond_flag} \
        --init_steps ${init_steps} \
        --num_steps ${num_steps} --gap_steps ${gap_steps} \
        --lr ${lr} --block_size ${block_size} \
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
