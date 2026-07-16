#!/bin/bash
# ===================================================================
#  Memorization Benchmark (Chen et al. CVPR/ICLR 2025 eval 기준)
#  DDIM baseline inference  (run_memo_chen.sh 기반)
#
#  Metrics: SSCD-to-GT + T2I → {method_dir}/eval/
#  Compute: elapsed_time, per_sample_time, peak_vram → {method_dir}/comp/
#  Reference: datasets/cvpr2025_webster_gt (Webster memorized GT)
#
#  환경: conda div_DM (실행 전 activate)
#  실행: ori_memo/ 디렉토리에서  bash shells/eval_chen/run_memo_chen_ddim.sh
# ===================================================================
# set -euo pipefail

# =========================== 1. [Parser] ===========================
gpu=0
model="stable-diffusion-v1-4"
method="ddim"
NFE=50
cfg_ddim=7.5
seed=42
num_samples=10
num_images_per_prompt=5
b_size=${num_images_per_prompt}
text_name="cvpr2025_memo_prompt.txt"

t2i_prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/cvpr2025_webster_gt"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# =========================== 2. [FLAG] ===========================
STD_FLAG="--model ${model} --method ${method} --device cuda:${gpu}"
ETC_FLAG="--NFE ${NFE} --seed ${seed}"
INF_FLAG="--b_size ${b_size} --num_samples ${num_samples} --num_images_per_prompt ${num_images_per_prompt}"
DIR_FLAG="--prompt_dir ${t2i_prompt_dir}"

# =========================== 3. [Workdir] ===========================
base_dir="workdir/memorization/sd14_base"
cfg_nfe_ddim="CFG=${cfg_ddim}_NFE=${NFE}"

ddim_dir="${base_dir}/ddim/${cfg_nfe_ddim}/seed=${seed}/batch=${b_size}"
ddim_eval="${ddim_dir}/eval"
ddim_comp="${ddim_dir}/comp"

model_key="ckpt/${model}"
total_imgs=$((num_samples * num_images_per_prompt))

echo "================== [INFO]: Model: ${model_key} =================="
echo "  DDIM  → ${ddim_dir}"

# =========================== 4. [Inference] (comp 자동 측정) ===========================
echo "================== [INFO]: DDIM Inference =================="
echo "  [CKPT] ${model_key}"
python -m examples.text_to_mscoco \
    ${STD_FLAG} ${ETC_FLAG} --cfg_guidance ${cfg_ddim} ${INF_FLAG} ${DIR_FLAG} \
    --workdir ${ddim_dir}

# =========================== 5. [Eval: DDIM] ===========================
echo "================== [INFO]: Eval [DDIM] → ${ddim_eval}/ =================="
mkdir -p ${ddim_eval}
python compute_sscd_gt.py \
    --gen_dir ${ddim_dir}/result --ref_dir ${gt_ref_dir} \
    --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
    --gpu ${gpu} \
    --output_csv ${ddim_eval}/chen_sscd_gt_metrics.csv
python -m compute_t2i_metrics \
    --eval_dir ${ddim_dir}/result --prompt_dir ${t2i_prompt_dir} \
    --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
    --output_csv ${ddim_eval}/chen_t2i_metrics.csv \
    --device cuda:${gpu} ${CS_FLAG}
python merge_benchmark.py --collect_dir ${ddim_eval}
echo "  [CHECK] eval files:"
/bin/ls ${ddim_eval}/*.csv 2>/dev/null

# # =========================== 6. [Trade-off CSV] (Chen eval) ===========================
# echo "================== [INFO]: Collect T2I-SSCD trade-off =================="
# python collect_tradeoff.py --base_dir ${base_dir} --out ${base_dir}/tradeoff.csv

# echo "================== [INFO]: Plot trade-off curves =================="
# for xm in clipscore pickscore imagereward; do
#     python plot_tradeoff.py --csv ${base_dir}/tradeoff.csv --x_metric ${xm} \
#         --out ${base_dir}/chen_tradeoff_${xm}.png \
#         --methods ddim
# done

echo "[Done] tradeoff.csv + chen_tradeoff_*.png → ${base_dir}/"
