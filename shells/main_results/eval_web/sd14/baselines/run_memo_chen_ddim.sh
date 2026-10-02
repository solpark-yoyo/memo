#!/bin/bash
# ===================================================================
#  Memorization Benchmark (Chen et al. CVPR/ICLR 2025 eval 기준)
#  DDIM baseline inference  (run_memo_chen.sh 기반)
#
#  Metrics: SSCD-to-GT + T2I → {method_dir}/eval/
#  Compute: elapsed_time, per_sample_time, peak_vram → {method_dir}/comp/
#  ★ Benchmark 기준선(baseline) 파악용 — 완화 없는 DDIM
#  프롬pt: sdv1_500_mem (Wen/Webster SD1.4 memorized 500)
#  GT: datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth (폴더당 1장 대표 ref)
#
#  환경: conda div_DM (실행 전 activate)
#  실행: ori_memo/ 디렉토리에서  bash shells/eval_chen/run_memo_chen_ddim.sh
# ===================================================================
# set -euo pipefail

# =========================== 0. [Timer] ===========================
SECONDS=0    # 스크립트 전체 총 소요 시간 측정 (종료 시 [Elapsed] 출력)

# =========================== 1. [Parser] ===========================
device="${DEVICE:-cuda:0}"
gpu="${device##*:}"
model="stable-diffusion-v1-4"
method="ddim"
NFE=50
cfg_ddim=7.5
seed=42
num_samples=10
num_images_per_prompt=4
b_size=${num_images_per_prompt}

# ★ eval subset — 평가할 inference 이미지 수 NUM_EVAL(요청) → clamp 후 실제 평가 장수가
#   폴더명: eval/<num_eval>, trd/<num_eval> (예: 요청 100 > 생성 10장 → eval/10, trd/10)
num_eval="${NUM_EVAL:-100}"
num_eval_prompts=$((num_eval / num_images_per_prompt))
(( num_eval_prompts > num_samples )) && num_eval_prompts=${num_samples}
num_eval=$((num_eval_prompts * num_images_per_prompt))   # 실제 평가 이미지 수 = 폴더명
text_name="sdv1_500_mem.txt"

t2i_prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
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

ddim_dir="${base_dir}/baselines/ddim/${model}/${cfg_nfe_ddim}/batch=${b_size}/seed=${seed}"
ddim_eval="${ddim_dir}/eval/${num_eval}"
ddim_comp="${ddim_dir}/comp"

model_key="ckpt/${model}"
total_imgs=$((num_samples * num_images_per_prompt))

echo "================== [INFO]: Model: ${model_key} =================="
echo "  DDIM  → ${ddim_dir}"
echo "  num_eval=${num_eval} imgs (eval prompts=${num_eval_prompts}) → eval/${num_eval}, trd/${num_eval}"

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
    --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_images_per_prompt} \
    --gpu ${gpu} \
    --output_csv ${ddim_eval}/chen_sscd_gt_metrics.csv
python -m compute_t2i_metrics \
    --eval_dir ${ddim_dir}/result --prompt_dir ${t2i_prompt_dir} \
    --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_images_per_prompt} \
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

# ---- [Elapsed] 전체 config 총 소요 시간 ----
echo ""
echo "[Elapsed] $((SECONDS/3600))hrs $(( (SECONDS%3600)/60 ))min $((SECONDS%60))sec (total ${SECONDS}sec)"
