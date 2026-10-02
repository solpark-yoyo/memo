#!/bin/bash
# ===================================================================
#  CADS baseline — Chen eval (arXiv 2310.17347)
#  Condition-Annealed Diffusion Sampler: inference 시 conditioning에
#  스케줄 노이즈 주입 → memorization 완화 + 다양성 향상
#
#  ĉ = √γ(t)·c + s·√(1−γ(t))·n  (n~N(0,I))
#  γ: piecewise linear annealing (tau1→tau2 구간에서 1→0)
#
#  Metrics: SSCD-to-GT + T2I → eval/
#  환경: conda div_DM (ori_memo/ 에서 실행)
#  실행: bash shells/main_results/eval_web/sd14/baselines/run_memo_chen_cads.sh
# ===================================================================

SECONDS=0

# =========================== 1. [Config] ===========================
device="${DEVICE:-cuda:0}"
gpu="${device##*:}"
model="stable-diffusion-v1-4"
NFE=50
cfg=7.5
seed=42
num_samples=10
num_images_per_prompt=4
b_size=${num_images_per_prompt}

num_eval_prompts="${NUM_EVAL_PROMPTS:-${num_samples}}"
(( num_eval_prompts > num_samples )) && num_eval_prompts=${num_samples}
num_eval=$((num_eval_prompts * num_images_per_prompt))

# CADS hyperparams (env var로 override 가능)
tau1="${TAU1:-0.6}"
tau2="${TAU2:-0.9}"
noise_scale="${NOISE_SCALE:-0.25}"
psi="${PSI:-1.0}"
base_s_ratio="${BASE_S_RATIO:-0.5}"

text_name="sdv1_500_mem.txt"
prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
t2i_prompt_dir="examples/assets/${text_name}"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

model_key="ckpt/${model}"
base_dir="workdir/memorization/sd14_base"
output_dir="${base_dir}/baselines/cads/${model}/CFG=${cfg}_NFE=${NFE}/tau1=${tau1}_tau2=${tau2}/noise_scale=${noise_scale}/batch=${b_size}/seed=${seed}"
eval_dir="${output_dir}/eval/${num_eval_prompts}"

cd "${ROOT_DIR:-.}"

echo "========================================="
echo "  CADS baseline — Chen eval"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  seed=${seed}"
echo "  tau1=${tau1}  tau2=${tau2}  noise_scale=${noise_scale}  psi=${psi}  base_s_ratio=${base_s_ratio}"
echo "  num_eval=${num_eval} imgs (eval prompts=${num_eval_prompts})"
echo "  output_dir=${output_dir}"
echo "========================================="

# =========================== 2. [Inference] ===========================
python baselines/cads/cads_inference.py \
    --model_key ${model_key} \
    --NFE ${NFE} --cfg ${cfg} \
    --base_seed ${seed} \
    --device ${device} \
    --prompt_dir ${prompt_dir} \
    --num_samples ${num_samples} \
    --num_images_per_prompt ${num_images_per_prompt} \
    --tau1 ${tau1} --tau2 ${tau2} \
    --noise_scale ${noise_scale} --psi ${psi} \
    --base_s_ratio ${base_s_ratio} \
    --output_dir ${output_dir}

# =========================== 3. [Eval] ===========================
mkdir -p ${eval_dir}
python compute_sscd_gt.py \
    --gen_dir ${output_dir}/result --ref_dir ${gt_ref_dir} \
    --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_images_per_prompt} \
    --gpu ${gpu} --output_csv ${eval_dir}/chen_sscd_gt_metrics.csv
python -m compute_t2i_metrics \
    --eval_dir ${output_dir}/result --prompt_dir ${t2i_prompt_dir} \
    --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_images_per_prompt} \
    --output_csv ${eval_dir}/chen_t2i_metrics.csv \
    --device ${device} ${CS_FLAG}
python merge_benchmark.py --collect_dir ${eval_dir}

echo "  [CHECK] eval:"; /bin/ls ${eval_dir}/*.csv 2>/dev/null
echo ""
echo "[Done] → ${eval_dir}/"
echo "[Elapsed] $((SECONDS/3600))hrs $(( (SECONDS%3600)/60 ))min $((SECONDS%60))sec (total ${SECONDS}sec)"
