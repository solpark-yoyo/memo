#!/bin/bash
# ===================================================================
#  [SD2.0] DDIM baseline — Memorization Benchmark 기준선
#  (sd14/baselines/run_memo_chen_ddim.sh 기반 — SD2 전용)
#
#  Metrics: SSCD-to-GT + T2I → {method_dir}/eval/
#  ★ Benchmark 기준선(baseline) 파악용 — 완화 없는 DDIM
#  프롬pt: sd2_mem219 (Webster SD2.0 memorized 219)
#  GT: datasets/memo/eval_sscd/sd20/sd2_mem_gt (195/219)
#
#  환경: conda div_DM (실행 전 activate)
#  실행: ori_memo/ 디렉토리에서  bash shells/eval_chen/sd20/baselines/run_sd20_ddim.sh
# ===================================================================
# set -euo pipefail

# =========================== 1. [Parser] ===========================
device="${DEVICE:-cuda:0}"
gpu="${device##*:}"
model="sd20"
model_tag="stable-diffusion-2-base"   # ★ sd2_mem219 발굴 모델 = 2-base (Webster verify_sdv2_bb_attack.sh 공식 지정)
method="ddim"
NFE=50
cfg_ddim=7.5
seed=42
num_samples=10
num_images_per_prompt=4
b_size=${num_images_per_prompt}
text_name="sd2_mem219.txt"

t2i_prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/memo/eval_sscd/sd20/sd2_mem_gt"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# =========================== 2. [FLAG] ===========================
STD_FLAG="--model ${model} --method ${method} --device cuda:${gpu}"
ETC_FLAG="--NFE ${NFE} --seed ${seed}"
INF_FLAG="--b_size ${b_size} --num_samples ${num_samples} --num_images_per_prompt ${num_images_per_prompt}"
DIR_FLAG="--prompt_dir ${t2i_prompt_dir}"

# =========================== 3. [Workdir] ===========================
base_dir="workdir/memorization/sd20_base"
cfg_nfe_ddim="CFG=${cfg_ddim}_NFE=${NFE}"

ddim_dir="${base_dir}/baselines/ddim/${model_tag}/${cfg_nfe_ddim}/seed=${seed}/batch=${b_size}"
ddim_eval="${ddim_dir}/eval"
ddim_comp="${ddim_dir}/comp"

model_key="ckpt/${model_tag}"
total_imgs=$((num_samples * num_images_per_prompt))

echo "================== [INFO]: Model: ${model_key} =================="
echo "  DDIM  → ${ddim_dir}"

# =========================== 4. [Inference] ===========================
echo "================== [INFO]: DDIM Inference SD2 =================="
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

echo "[Done] → ${ddim_eval}/"
