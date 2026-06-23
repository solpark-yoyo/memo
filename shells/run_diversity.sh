#!/bin/bash
# ===================================================================
#  Diversity Benchmark: DDIM vs CNO(infoNCE) vs init_opti
#  Metrics: T2I (CLIP/PickScore/ImageReward) + PRDC + Diversity
#
#  실행 한 번으로 inference → 모든 metric → merge 까지 전부 수행
# ===================================================================
# set -euo pipefail

# =========================== 1. [Parser] ===========================
# a. inference config
gpu=0
model="sd15"
method="ddim"
NFE=50
cfg_ddim=7.5
cfg_cno=6.0
cfg_init_opti=7.5
seed=42
b_size=5
num_samples=20
num_images_per_prompt=5
text_name="coco_v2.txt"

# b. CNO (infoNCE) config
iopt_iter=3
iopt_lr=0.01
infoNCE_temp=0.1
window_size=16
gamma=1.0

# c. init_opti config
init_steps=10
num_opt_steps=4
gap_steps=3
lr=0.01
base_s_ratio=0.5
lambda_align=0.1
init_opti_prompt_dir="examples/assets/coco_v2.txt"

# d. Eval config
prdc_real_dir="datasets/ms_coco/resize/val2014_512"
t2i_prompt_dir="examples/assets/${text_name}"
nearest_k=5
f_type="inception"                            # pixel | inception | sscd
cs_only=false                                 # true: CLIP score only

# =========================== 2. [FLAG] ===========================
STD_FLAG="--model ${model} --method ${method} --device cuda:${gpu}"
ETC_FLAG="--NFE ${NFE} --seed ${seed}"
INF_FLAG="--b_size ${b_size} --num_samples ${num_samples} --num_images_per_prompt ${num_images_per_prompt}"
DIR_FLAG="--prompt_dir ${t2i_prompt_dir}"

CNO_FLAG="--iopt_diverse --iopt_loss_type infoNCE \
--i_opt_iter ${iopt_iter} --i_opt_lr ${iopt_lr} --iopt_cfg_tweedie \
--infoNCE_temp ${infoNCE_temp} --window_size ${window_size} --gamma ${gamma} --n_aug_samples 0"

CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# =========================== 3. [Workdir] ===========================
base_dir="workdir/diversity/mscoco/sd15"
cfg_nfe_ddim="CFG=${cfg_ddim}_NFE=${NFE}"
cfg_nfe_cno="CFG=${cfg_cno}_NFE=${NFE}"
cfg_nfe_init="CFG=${cfg_init_opti}_NFE=${NFE}"

ddim_dir="${base_dir}/ddim/${cfg_nfe_ddim}/seed=${seed}"
cno_dir="${base_dir}/cno_infoNCE/${cfg_nfe_cno}/temp=${infoNCE_temp}_win=${window_size}_gamma=${gamma}_iter=${iopt_iter}/seed=${seed}"
init_dir="${base_dir}/init_opti/${cfg_nfe_init}/base_s_ratio=${base_s_ratio}_lambda_align=${lambda_align}/init=${init_steps}_nsteps=${num_opt_steps}_gap=${gap_steps}_lr=${lr}/seed=${seed}/batch=${b_size}"

echo "${ddim_dir}"
echo "${cno_dir}"
echo "${init_dir}"

# # =========================== 4. [Inference] ===========================
# echo "================== [INFO]: DDIM Inference =================="
# python -m examples.text_to_mscoco \
#     ${STD_FLAG} ${ETC_FLAG} --cfg_guidance ${cfg_ddim} ${INF_FLAG} ${DIR_FLAG} \
#     --workdir ${ddim_dir}

# echo "================== [INFO]: CNO(InfoNCE) Inference =================="
# python -m examples.text_to_mscoco \
#     ${STD_FLAG} ${ETC_FLAG} --cfg_guidance ${cfg_cno} ${INF_FLAG} ${DIR_FLAG} \
#     ${CNO_FLAG} \
#     --workdir ${cno_dir}

# echo "================== [INFO]: init_opti Inference =================="
# python run_ini_opti.py \
#     --NFE ${NFE} --cfg ${cfg_init_opti} --lr ${lr} \
#     --init_steps ${init_steps} --num_steps ${num_opt_steps} --gap_steps ${gap_steps} \
#     --base_s_ratio ${base_s_ratio} --lambda_align ${lambda_align} \
#     --base_seed ${seed} --num_seeds ${num_images_per_prompt} \
#     --prompt_dir ${init_opti_prompt_dir} --num_samples ${num_samples} \
#     --device cuda:${gpu} --output_dir ${init_dir}

# # =========================== 5. [Eval: DDIM] ===========================
# echo "================== [INFO]: Eval [DDIM] =================="
# mkdir -p "${ddim_dir}/logs"
# python -m compute_t2i_metrics \
#     --eval_dir ${ddim_dir}/result --prompt_dir ${t2i_prompt_dir} \
#     --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
#     --output_csv ${ddim_dir}/t2i_metrics.csv \
#     --device cuda:${gpu} ${CS_FLAG} \
#     --log_csv ${ddim_dir}/logs/t2i_log.csv --warn_log ${ddim_dir}/logs/t2i_warnings.log

# python -m prdc.prdc_cli \
#     -r ${prdc_real_dir} -f ${ddim_dir}/result \
#     -t R -o 64 -b $((num_samples * num_images_per_prompt)) -d cuda:${gpu} -k ${nearest_k} \
#     --num_real 1000 --num_fake $((num_samples * num_images_per_prompt)) \
#     --log_csv ${ddim_dir}/logs/prdc_log.csv --warn_log ${ddim_dir}/logs/prdc_warnings.log

# python compute_vendi_score.py \
#     --eval_dir ${ddim_dir}/result \
#     --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
#     --f_type ${f_type} \
#     --output_csv ${ddim_dir}/vendi_metrics.csv

# # collect DDIM's t2i+prdc+vendi into one total_metrics.csv
# python merge_benchmark.py --collect_dir ${ddim_dir}

# # =========================== 6. [Eval: CNO(InfoNCE)] ===========================
# echo "================== [INFO]: Eval [CNO(InfoNCE)] =================="
# mkdir -p "${cno_dir}/logs"
# python -m compute_t2i_metrics \
#     --eval_dir ${cno_dir}/result --prompt_dir ${t2i_prompt_dir} \
#     --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
#     --output_csv ${cno_dir}/t2i_metrics.csv \
#     --device cuda:${gpu} ${CS_FLAG} \
#     --log_csv ${cno_dir}/logs/t2i_log.csv --warn_log ${cno_dir}/logs/t2i_warnings.log

# python -m prdc.prdc_cli \
#     -r ${prdc_real_dir} -f ${cno_dir}/result \
#     -t R -o 64 -b $((num_samples * num_images_per_prompt)) -d cuda:${gpu} -k ${nearest_k} \
#     --num_real 1000 --num_fake $((num_samples * num_images_per_prompt)) \
#     --log_csv ${cno_dir}/logs/prdc_log.csv --warn_log ${cno_dir}/logs/prdc_warnings.log

# python compute_vendi_score.py \
#     --eval_dir ${cno_dir}/result \
#     --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
#     --f_type ${f_type} \
#     --output_csv ${cno_dir}/vendi_metrics.csv

# # collect CNO's t2i+prdc+vendi into one total_metrics.csv
# python merge_benchmark.py --collect_dir ${cno_dir}

# # =========================== 7. [Eval: init_opti] ===========================
# echo "================== [INFO]: Eval [init_opti] =================="
# mkdir -p "${init_dir}/logs"
# python -m compute_t2i_metrics \
#     --eval_dir ${init_dir}/result --prompt_dir ${t2i_prompt_dir} \
#     --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
#     --output_csv ${init_dir}/t2i_metrics.csv \
#     --device cuda:${gpu} ${CS_FLAG} \
#     --log_csv ${init_dir}/logs/t2i_log.csv --warn_log ${init_dir}/logs/t2i_warnings.log

# python -m prdc.prdc_cli \
#     -r ${prdc_real_dir} -f ${init_dir}/result \
#     -t R -o 64 -b $((num_samples * num_images_per_prompt)) -d cuda:${gpu} -k ${nearest_k} \
#     --num_real 1000 --num_fake $((num_samples * num_images_per_prompt)) \
#     --log_csv ${init_dir}/logs/prdc_log.csv --warn_log ${init_dir}/logs/prdc_warnings.log

# python compute_vendi_score.py \
#     --eval_dir ${init_dir}/result \
#     --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
#     --f_type ${f_type} \
#     --output_csv ${init_dir}/vendi_metrics.csv

# # collect init_opti's t2i+prdc+vendi into one total_metrics.csv
# python merge_benchmark.py --collect_dir ${init_dir}

# # =========================== 8. [Merge] ===========================
# echo "================== [INFO]: Merging All Metrics =================="
# python merge_benchmark.py \
#     --workdir ${base_dir} \
#     --methods DDIM CNO_infoNCE init_opti \
#     --ddim_dir ${ddim_dir} --cno_dir ${cno_dir} --init_opti_dir ${init_dir} \
#     --output ${base_dir}/comparison_report.csv

echo "[Done] Report: ${base_dir}/comparison_report.csv"
