#!/bin/bash
# ===================================================================
#  init_score_noise (Han et al. NeurIPS 2025) — Chen CVPR 2025 setup
#  일반 SD1.4 + Webster memorized prompts (global/local)
#
#  conda env: div_DM
#    bash shells/run_memo_score_chen.sh
# ===================================================================

# =========================== 1. [Config] ===========================
gpu=0
NFE=50
cfg_initnoise=7.5
seed=42
num_samples=10
batch=5
num_images_per_prompt=${batch}

# init_score_noise (Han) params 
target_loss=0.9
optim_iters=10
lr_list=(0.0 0.02 0.04 0.06)

# Chen setup (일반 SD1.4 + Webster)
model_id="ckpt/stable-diffusion-v1-4"
text_name="cvpr2025_memo_prompt.txt"
t2i_prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/cvpr2025_webster_gt"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# =========================== 2. [Workdir] ===========================
base_dir="workdir/memorization/sd14_base"
output_path="${base_dir}/init_score_noise/NFE=${NFE}"

for lr in "${lr_list[@]}"; do
    gen_dir="${output_path}/per_sample/CFG=${cfg_initnoise}/lr=${lr}/tl=${target_loss}/oi=${optim_iters}/seed=${seed}"
    eval_dir="${gen_dir}/eval"
    echo "==================== lr=${lr} =========================="
    echo "  gen_dir: ${gen_dir}"

    # # =========================== 3. [Inference] ===========================
    # echo "================== [INFO]: init_score_noise Inference (Webster prompt) =================="
    # python baselines/init_score_noise/generate_init_score_noise.py \
    #     --method adj_init_noise --per_sample \
    #     --target_loss ${target_loss} --lr ${lr} --optim_iters ${optim_iters} \
    #     --guidance_scale ${cfg_initnoise} --seed ${seed} --num_prompts ${num_samples} \
    #     --n_samples_per_prompt ${num_images_per_prompt} --batch_size 1 \
    #     --num_inference_steps ${NFE} --model_id ${model_id} --gpu ${gpu} \
    #     --output_path "${output_path}" \
    #     --prompt_csv ${t2i_prompt_dir}

    # =========================== 4. [Eval] (Chen: SSCD + T2I) ===========================
    echo "================== [INFO]: Eval [init_score_noise] → ${eval_dir}/ =================="
    mkdir -p ${eval_dir}

    python compute_sscd_gt.py \
        --gen_dir ${gen_dir} --ref_dir ${gt_ref_dir} \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu ${gpu} \
        --output_csv ${eval_dir}/chen_sscd_gt_metrics.csv

    python -m compute_t2i_metrics \
        --eval_dir ${gen_dir} --prompt_dir ${t2i_prompt_dir} \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --output_csv ${eval_dir}/chen_t2i_metrics.csv \
        --device cuda:${gpu} ${CS_FLAG}

    python merge_benchmark.py --collect_dir ${eval_dir}

    echo "  [CHECK] eval files:"
    /bin/ls ${eval_dir}/*.csv 2>/dev/null
done

# =========================== 5. [Trade-off CSV + Plot] ===========================
# collect/plot (init_score_noise 만). csv/trd 아래 lr별 서브폴더 생성.
plot_dir="${base_dir}/init_score_noise/plot"
mkdir -p ${plot_dir}/trd ${plot_dir}/csv

echo "================== [INFO]: Collect T2I-SSCD trade-off (lr별 분할 포함) =================="
python collect_tradeoff.py --base_dir ${base_dir}/init_score_noise \
    --out ${plot_dir}/csv/tradeoff.csv --split_lr

echo "================== [INFO]: Plot trade-off curves → ${plot_dir}/trd/ (lr 표시 없음) =================="
for xm in clipscore pickscore imagereward; do
    python plot_tradeoff.py --csv ${plot_dir}/csv/tradeoff.csv --x_metric ${xm} \
        --out ${plot_dir}/trd/chen_tradeoff_${xm}.png \
        --methods init_score_noise
done

# trade-off curve PNG(전체 lr, lr 표시 없음)를 각 lr 폴더에 복사
for lr_dir in ${plot_dir}/csv/lr=*; do
    [ -d "${lr_dir}" ] || continue
    lr_name="$(basename "${lr_dir}")"
    mkdir -p "${plot_dir}/trd/${lr_name}"
    for xm in clipscore pickscore imagereward; do
        /bin/cp -f "${plot_dir}/trd/chen_tradeoff_${xm}.png" "${plot_dir}/trd/${lr_name}/" 2>/dev/null
    done
done

echo "[Done] → ${plot_dir}/csv/tradeoff.csv (+ lr=*/ )  +  ${plot_dir}/trd/chen_tradeoff_*.png (+ lr=*/ )"
