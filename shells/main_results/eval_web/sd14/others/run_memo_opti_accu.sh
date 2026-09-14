#!/bin/bash
# ===================================================================
#  init_opti + accumulate loss (adjointDPM) — Chen eval
#  type_memo_loss="accumulate"
#
#  accumulate: 각 update ui 에서 update_indices[0..ui] 의 proxy 를 누적.
#    예: init_steps=2, num_opt_steps=3, gap_steps=1 → update_indices=[2,3,4]
#      update 0: loss = proxy(step=2)
#      update 1: loss = (proxy(2) + proxy(3)) / 2
#      update 2: loss = (proxy(2) + proxy(3) + proxy(4)) / 3
#    각 active step 에서 terminal head → g_k 수집 → adjoint recursion 에 주입.
#
#  run_memo_init.sh 기반 (inference + eval + trade-off 통합)
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/eval_chen/run_memo_opti_accumulate.sh
#    GPER=1 bash shells/eval_chen/run_memo_opti_accumulate.sh
# ===================================================================

# =========================== 1. [Parser] ===========================
gpu=0
model="stable-diffusion-v1-4"
NFE=50
cfg_init_opti=7.5
cfg_start_ratio=0.0
type_memo_loss="accumulate"    # ★ accumulate mode (adjointDPM)
memo_threshold=0.3
gper="${GPER:-0}"
grad_prcd_flag=""; [[ "${gper}" == "1" ]] && grad_prcd_flag="--grad_prcd"
memothr_dir=""
seed=42
num_samples=10
num_images_per_prompt=5
b_size=${num_images_per_prompt}
text_name="cvpr2025_memo_prompt.txt"
init_opti_prompt_dir="examples/assets/${text_name}"
t2i_prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/cvpr2025_webster_gt"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# init_opti config (★ accumulate: gap=1, 짧은 chain)
num_opt_steps=3
gap_steps=1            # ★ accumulate 권장: gap=1
base_s_ratio=0.5
lambda_align=0.00

# sweep lists
init_steps_list=(2)
lr_list=(0.02 0.04 0.06 0.08)

# Workdir prefix
base_dir="workdir/memorization/sd14_base"
cfg_nfe_init="CFG=${cfg_init_opti}_NFE=${NFE}"
model_key="ckpt/${model}"
init_prefix="${base_dir}/init_opti/${cfg_nfe_init}/gper=${gper}/cfgsr=${cfg_start_ratio}/base_s_ratio=${base_s_ratio}_lambda_align=${lambda_align}/memoloss=${type_memo_loss}/${memothr_dir}"

cd "${ROOT_DIR:-.}"

echo "========================================="
echo "  init_opti accumulate (adjointDPM) — Chen"
echo "  model=${model}  NFE=${NFE}  cfg=${cfg_init_opti}  seed=${seed}  batch=${b_size}"
echo "  gper=${gper}  type_memo_loss=${type_memo_loss}"
echo "  init_steps=(${init_steps_list[*]})  num_opt_steps=${num_opt_steps}  gap=${gap_steps}"
echo "  lr=(${lr_list[*]})"
echo "========================================="

# =========================== 2. [init 별 루프] ===========================
for init_steps in "${init_steps_list[@]}"; do
    echo ""
    echo "############## init_steps=${init_steps} ##############"
    trd_path="${init_prefix}init=${init_steps}/nsteps=${num_opt_steps}/gap=${gap_steps}"
    echo "  trd_path(lr 부모) = ${trd_path}"

    for lr in "${lr_list[@]}"; do
        init_dir="${trd_path}/lr=${lr}/seed=${seed}/batch=${b_size}"
        init_eval="${init_dir}/eval"
        echo "---- init=${init_steps} lr=${lr} ----"

        python run_ini_opti.py \
            --NFE ${NFE} --cfg ${cfg_init_opti} --lr ${lr} \
            --model_key ${model_key} \
            --init_steps ${init_steps} --num_steps ${num_opt_steps} --gap_steps ${gap_steps} \
            --base_s_ratio ${base_s_ratio} --lambda_align ${lambda_align} \
            --cfg_start_ratio ${cfg_start_ratio} --type_memo_loss ${type_memo_loss} --memo_threshold ${memo_threshold} ${grad_prcd_flag} \
            --base_seed ${seed} --num_seeds ${num_images_per_prompt} \
            --prompt_dir ${init_opti_prompt_dir} --num_samples ${num_samples} \
            --device cuda:${gpu} --output_dir ${init_dir}

        mkdir -p ${init_eval}
        python compute_sscd_gt.py \
            --gen_dir ${init_dir}/result --ref_dir ${gt_ref_dir} \
            --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
            --gpu ${gpu} --output_csv ${init_eval}/chen_sscd_gt_metrics.csv
        python -m compute_t2i_metrics \
            --eval_dir ${init_dir}/result --prompt_dir ${t2i_prompt_dir} \
            --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
            --output_csv ${init_eval}/chen_t2i_metrics.csv \
            --device cuda:${gpu} ${CS_FLAG}
        python merge_benchmark.py --collect_dir ${init_eval}
        echo "  [CHECK] eval:"; /bin/ls ${init_eval}/*.csv 2>/dev/null
    done

    echo "==== [trade-off] ${trd_path}/trd/ ===="
    python collect_trd.py --method init_opti --path "${trd_path}" \
        --lr_list ${lr_list[@]} --seed ${seed} --batch ${b_size}
    python plot_trd.py --csv "${trd_path}/trd/csv/total_metrics.csv" --out_dir "${trd_path}/trd/plot"
done

echo ""
echo "[All Done] accumulate (adjointDPM) init=(${init_steps_list[*]}) 완료"
