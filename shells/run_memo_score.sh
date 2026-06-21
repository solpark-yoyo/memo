#!/bin/bash
# ===================================================================
#  init_score_noise (Han et al. NeurIPS 2025) — memorization baseline inference
#  per_sample variant.
#
#  ※ 수동으로 init_score_noise env 활성화 후 실행:
#    conda activate init_score_noise
#    bash shells/run_memo_score.sh
#  (eval은 ori_memo env에서 별도 수행)
# ===================================================================

# =========================== 1. [Config] ===========================
gpu=0
NFE=50
cfg_initnoise=7.5
seed=42
num_samples=20
batch=5
num_images_per_prompt=${batch}
target_loss=0.9
optim_iters=1000
lr=0.01
model_id="ckpt/stable-diffusion-v1-5"
text_name="memorized_prompts_membench.txt"

# Eval config
t2i_prompt_dir="examples/assets/${text_name}"
f_type="sscd"                                 # memorization → sscd (collapse 측정)
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# =========================== 2. [Workdir] ===========================
base_dir="workdir/memorization/sd15"
output_path="${base_dir}/init_score_noise/NFE=${NFE}"
gen_dir="${output_path}/per_sample/CFG=${cfg_initnoise}/lr=${lr}/tl=${target_loss}/oi=${optim_iters}/seed=${seed}"

echo "${gen_dir}"

# # =========================== 3. [Inference] (init_score_noise env) ===========================
# echo "================== [INFO]: init_score_noise Inference =================="
# python baselines/init_score_noise/generate_init_score_noise.py \
#     --method adj_init_noise --per_sample \
#     --target_loss ${target_loss} --lr ${lr} --optim_iters ${optim_iters} \
#     --guidance_scale ${cfg_initnoise} --seed ${seed} --num_prompts ${num_samples} \
#     --n_samples_per_prompt ${num_images_per_prompt} --batch_size 1 \
#     --num_inference_steps ${NFE} --model_id ${model_id} --gpu ${gpu} \
#     --output_path "${output_path}"

# echo "[Done] images at ${gen_dir}/"

# =========================== 4. [Eval] (ori_memo env) ===========================
# ※ inference는 init_score_noise env, eval은 ori_memo env에서 실행
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
