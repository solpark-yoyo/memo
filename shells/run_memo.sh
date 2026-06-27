#!/bin/bash
# ===================================================================
#  Memorization Benchmark (Wen et al. ICLR 2024 eval 기준)
#  DDIM vs CNO(infoNCE) vs init_opti
#
#  Metrics: SSCD-to-GT + T2I → {method_dir}/eval/
#  Compute: elapsed_time, per_sample_time, peak_vram → {method_dir}/comp/
#  Reference: datasets/wen2024_memorized/ (Wen et al. 공개 GT)
# ===================================================================
# set -euo pipefail

# =========================== 1. [Parser] ===========================
gpu=0
model="sd14_memor_LAION2B_40k"
method="ddim"
NFE=50
cfg_ddim=7.5
cfg_cno=6.0
cfg_init_opti=7.5
seed=42
num_samples=10
num_images_per_prompt=5
b_size=${num_images_per_prompt}
text_name="wen2024_memorized_prompts.txt"

# b. CNO (infoNCE) config
iopt_iter=3
iopt_lr=0.01
infoNCE_temp=0.1
window_size=16
gamma=1.0

lr_list=(0.10)

for lr in "${lr_list[@]}"; do
    echo "==================== lr=${lr} =========================="
    init_steps=10
    num_opt_steps=4
    gap_steps=3
    base_s_ratio=0.5
    lambda_align=0.00
    init_opti_prompt_dir="examples/assets/${text_name}"

    t2i_prompt_dir="examples/assets/${text_name}"
    gt_ref_dir="datasets/wen2024_memorized"
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
    base_dir="workdir/memorization/sd14_memor_LAION2B_40k"
    cfg_nfe_ddim="CFG=${cfg_ddim}_NFE=${NFE}"
    cfg_nfe_cno="CFG=${cfg_cno}_NFE=${NFE}"
    cfg_nfe_init="CFG=${cfg_init_opti}_NFE=${NFE}"

    ddim_dir="${base_dir}/ddim/${cfg_nfe_ddim}/seed=${seed}"
    cno_dir="${base_dir}/cno_infoNCE/${cfg_nfe_cno}/temp=${infoNCE_temp}_win=${window_size}_gamma=${gamma}_iter=${iopt_iter}/seed=${seed}"
    init_dir="${base_dir}/init_opti/${cfg_nfe_init}/base_s_ratio=${base_s_ratio}_lambda_align=${lambda_align}/init=${init_steps}_nsteps=${num_opt_steps}_gap=${gap_steps}_lr=${lr}/seed=${seed}/batch=${b_size}"

    ddim_eval="${ddim_dir}/eval"
    cno_eval="${cno_dir}/eval"
    init_eval="${init_dir}/eval"
    ddim_comp="${ddim_dir}/comp"
    cno_comp="${cno_dir}/comp"
    init_comp="${init_dir}/comp"

    model_key="ckpt/${model}"
    total_imgs=$((num_samples * num_images_per_prompt))

    echo "================== [INFO]: Model: ${model_key} =================="
    echo "  DDIM  → ${ddim_dir}"
    echo "  CNO   → ${cno_dir}"
    echo "  init  → ${init_dir}"

    # =========================== 4. [Inference] (comp 자동 측정) ===========================
    # --- DDIM ---
    # echo "================== [INFO]: DDIM Inference =================="
    # echo "  [CKPT] ${model_key}"
    # python -m examples.text_to_mscoco \
    #     ${STD_FLAG} ${ETC_FLAG} --cfg_guidance ${cfg_ddim} ${INF_FLAG} ${DIR_FLAG} \
    #     --workdir ${ddim_dir}

    # --- CNO (infoNCE) ---
    # echo "================== [INFO]: CNO(InfoNCE) Inference =================="
    # echo "  [CKPT] ${model_key}"
    # python -m examples.text_to_mscoco \
    #     ${STD_FLAG} ${ETC_FLAG} --cfg_guidance ${cfg_cno} ${INF_FLAG} ${DIR_FLAG} \
    #     ${CNO_FLAG} \
    #     --workdir ${cno_dir}

    # --- init_opti ---
    echo "================== [INFO]: init_opti Inference =================="
    echo "  [CKPT] ${model_key}"
    python run_ini_opti.py \
        --NFE ${NFE} --cfg ${cfg_init_opti} --lr ${lr} \
        --model_key ${model_key} \
        --init_steps ${init_steps} --num_steps ${num_opt_steps} --gap_steps ${gap_steps} \
        --base_s_ratio ${base_s_ratio} --lambda_align ${lambda_align} \
        --base_seed ${seed} --num_seeds ${num_images_per_prompt} \
        --prompt_dir ${init_opti_prompt_dir} --num_samples ${num_samples} \
        --device cuda:${gpu} --output_dir ${init_dir}

    # =========================== 5. [Eval: DDIM] ===========================
    # echo "================== [INFO]: Eval [DDIM] → ${ddim_eval}/ =================="
    # mkdir -p ${ddim_eval}
    # python compute_sscd_gt.py \
    #     --gen_dir ${ddim_dir}/result --ref_dir ${gt_ref_dir} \
    #     --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
    #     --gpu ${gpu} \
    #     --output_csv ${ddim_eval}/sscd_gt_metrics.csv
    # python -m compute_t2i_metrics \
    #     --eval_dir ${ddim_dir}/result --prompt_dir ${t2i_prompt_dir} \
    #     --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
    #     --output_csv ${ddim_eval}/t2i_metrics.csv \
    #     --device cuda:${gpu} ${CS_FLAG}
    # python merge_benchmark.py --collect_dir ${ddim_eval}
    # echo "  [CHECK] eval files:"
    # /bin/ls ${ddim_eval}/*.csv 2>/dev/null

    # =========================== 6. [Eval: CNO] ===========================
    # echo "================== [INFO]: Eval [CNO] → ${cno_eval}/ =================="
    # mkdir -p ${cno_eval}
    # python compute_sscd_gt.py \
    #     --gen_dir ${cno_dir}/result --ref_dir ${gt_ref_dir} \
    #     --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
    #     --gpu ${gpu} \
    #     --output_csv ${cno_eval}/sscd_gt_metrics.csv
    # python -m compute_t2i_metrics \
    #     --eval_dir ${cno_dir}/result --prompt_dir ${t2i_prompt_dir} \
    #     --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
    #     --output_csv ${cno_eval}/t2i_metrics.csv \
    #     --device cuda:${gpu} ${CS_FLAG}
    # python merge_benchmark.py --collect_dir ${cno_eval}
    # echo "  [CHECK] eval files:"
    # /bin/ls ${cno_eval}/*.csv 2>/dev/null

    # =========================== 7. [Eval: init_opti] ===========================
    echo "================== [INFO]: Eval [init_opti] → ${init_eval}/ =================="
    mkdir -p ${init_eval}
    python compute_sscd_gt.py \
        --gen_dir ${init_dir}/result --ref_dir ${gt_ref_dir} \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu ${gpu} \
        --output_csv ${init_eval}/sscd_gt_metrics.csv

    python -m compute_t2i_metrics \
        --eval_dir ${init_dir}/result --prompt_dir ${t2i_prompt_dir} \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --output_csv ${init_eval}/t2i_metrics.csv \
        --device cuda:${gpu} ${CS_FLAG}

    python merge_benchmark.py --collect_dir ${init_eval}

    # eval 파일 검증
    echo "  [CHECK] eval files:"
    /bin/ls ${init_eval}/*.csv 2>/dev/null
    echo "  [CHECK] comp files:"
    /bin/ls ${init_comp}/*.csv 2>/dev/null

done
echo "[Done]"
