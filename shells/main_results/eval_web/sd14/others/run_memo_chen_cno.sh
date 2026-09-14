#!/bin/bash
# ===================================================================
#  Memorization Benchmark (Chen et al. CVPR/ICLR 2025 eval 기준)
#  CNO(infoNCE) baseline inference  (run_memo_chen.sh 기반)
#
#  Metrics: SSCD-to-GT + T2I → {method_dir}/eval/
#  Compute: elapsed_time, per_sample_time, peak_vram → {method_dir}/comp/
#  Reference: datasets/cvpr2025_webster_gt (Webster memorized GT)
#
#  환경: conda div_DM (실행 전 activate)
#  실행: ori_memo/ 디렉토리에서  bash shells/eval_chen/run_memo_chen_cno.sh
# ===================================================================
# set -euo pipefail

# =========================== 1. [Parser] ===========================
gpu=0
model="stable-diffusion-v1-4"
method="ddim"
NFE=50
cfg_cno=6.0
seed=42
num_samples=10
num_images_per_prompt=5
b_size=${num_images_per_prompt}
text_name="cvpr2025_memo_prompt.txt"

# CNO (infoNCE) config
iopt_iter=3
iopt_lr=0.01
infoNCE_temp=0.1
window_size=16
gamma=1.0

t2i_prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/cvpr2025_webster_gt"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# =========================== 2. [FLAG] ===========================
STD_FLAG="--model ${model} --method ${method} --device cuda:${gpu}"
ETC_FLAG="--NFE ${NFE} --seed ${seed}"
INF_FLAG="--b_size ${b_size} --num_samples ${num_samples} --num_images_per_prompt ${num_images_per_prompt}"
DIR_FLAG="--prompt_dir ${t2i_prompt_dir}"

CNO_FLAG="--iopt_diverse --iopt_loss_type infoNCE \
    --i_opt_iter ${iopt_iter} --i_opt_lr ${iopt_lr} --iopt_cfg_tweedie \
    --infoNCE_temp ${infoNCE_temp} --window_size ${window_size} --gamma ${gamma} --n_aug_samples 0"

# =========================== 3. [Workdir] ===========================
base_dir="workdir/memorization/sd14_base"
cfg_nfe_cno="CFG=${cfg_cno}_NFE=${NFE}"

cno_dir="${base_dir}/cno_infoNCE/${cfg_nfe_cno}/temp=${infoNCE_temp}_win=${window_size}_gamma=${gamma}_iter=${iopt_iter}/seed=${seed}/batch=${b_size}"
cno_eval="${cno_dir}/eval"
cno_comp="${cno_dir}/comp"

model_key="ckpt/${model}"
total_imgs=$((num_samples * num_images_per_prompt))

echo "================== [INFO]: Model: ${model_key} =================="
echo "  CNO   → ${cno_dir}"

# =========================== 4. [Inference] (comp 자동 측정) ===========================
echo "================== [INFO]: CNO(InfoNCE) Inference =================="
echo "  [CKPT] ${model_key}"
python -m examples.text_to_mscoco \
    ${STD_FLAG} ${ETC_FLAG} --cfg_guidance ${cfg_cno} ${INF_FLAG} ${DIR_FLAG} \
    ${CNO_FLAG} \
    --workdir ${cno_dir}

# =========================== 5. [Eval: CNO] ===========================
echo "================== [INFO]: Eval [CNO] → ${cno_eval}/ =================="
mkdir -p ${cno_eval}
python compute_sscd_gt.py \
    --gen_dir ${cno_dir}/result --ref_dir ${gt_ref_dir} \
    --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
    --gpu ${gpu} \
    --output_csv ${cno_eval}/chen_sscd_gt_metrics.csv
python -m compute_t2i_metrics \
    --eval_dir ${cno_dir}/result --prompt_dir ${t2i_prompt_dir} \
    --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
    --output_csv ${cno_eval}/chen_t2i_metrics.csv \
    --device cuda:${gpu} ${CS_FLAG}
python merge_benchmark.py --collect_dir ${cno_eval}
echo "  [CHECK] eval files:"
/bin/ls ${cno_eval}/*.csv 2>/dev/null

# # =========================== 6. [Trade-off CSV] (Chen eval) ===========================
# echo "================== [INFO]: Collect T2I-SSCD trade-off =================="
# python collect_tradeoff.py --base_dir ${base_dir} --out ${base_dir}/tradeoff.csv

# echo "================== [INFO]: Plot trade-off curves =================="
# for xm in clipscore pickscore imagereward; do
#     python plot_tradeoff.py --csv ${base_dir}/tradeoff.csv --x_metric ${xm} \
#         --out ${base_dir}/chen_tradeoff_${xm}.png \
#         --methods cno_infoNCE
# done

echo "[Done] tradeoff.csv + chen_tradeoff_*.png → ${base_dir}/"
