#!/bin/bash
# ===================================================================
#  init_score_noise (Han et al. NeurIPS 2025) — memorization baseline
#  Wen et al. (ICLR 2024) eval 기준 / sd14_memor_LAION2B_40k
#
#  conda env: div_DM (inference + eval 동일 env 사용)
#    bash shells/run_memo_score.sh
# ===================================================================
# PY="/home/geonsoo/anaconda3/envs/div_DM/bin/python"

# =========================== 1. [Config] ===========================
gpu=0
NFE=50
cfg_initnoise=7.5
seed=42
num_samples=15
batch=5
num_images_per_prompt=${batch}
target_loss=0.9
optim_iters=1000
lr=0.01
model_id="ckpt/sd14_memor_LAION2B_40k"
text_name="new_memorized_text_prompt.txt"

# Eval config
t2i_prompt_dir="examples/assets/${text_name}"
f_type="sscd"                                 # memorization → sscd (collapse 측정)
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# =========================== 2. [Workdir] ===========================
base_dir="workdir/memorization/sd14_memor_LAION2B_40k"
output_path="${base_dir}/init_score_noise/NFE=${NFE}"
gen_dir="${output_path}/per_sample/CFG=${cfg_initnoise}/lr=${lr}/tl=${target_loss}/oi=${optim_iters}/seed=${seed}"

echo "${gen_dir}"

# =========================== 3. [Inference] (div_DM env) ===========================
echo "================== [INFO]: init_score_noise Inference =================="
python baselines/init_score_noise/generate_init_score_noise.py \
    --method adj_init_noise --per_sample \
    --target_loss ${target_loss} --lr ${lr} --optim_iters ${optim_iters} \
    --guidance_scale ${cfg_initnoise} --seed ${seed} --num_prompts ${num_samples} \
    --n_samples_per_prompt ${num_images_per_prompt} --batch_size 1 \
    --num_inference_steps ${NFE} --model_id ${model_id} --gpu ${gpu} \
    --output_path "${output_path}"

echo "[Done] images at ${gen_dir}/"

# =========================== 4. [Eval] (div_DM env) ===========================
echo "================== [INFO]: Eval [init_score_noise] =================="
python compute_vendi_score.py \
    --eval_dir ${gen_dir} \
    --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
    --f_type ${f_type} \
    --output_csv ${gen_dir}/vendi_metrics.csv

python -m compute_t2i_metrics \
    --eval_dir ${gen_dir} --prompt_dir ${t2i_prompt_dir} \
    --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
    --output_csv ${gen_dir}/t2i_metrics.csv \
    --device cuda:${gpu} ${CS_FLAG}

# collect init_score_noise's t2i+vendi into one total_metrics.csv
python merge_benchmark.py --collect_dir ${gen_dir}

echo "[Done] metrics at ${gen_dir}/"
