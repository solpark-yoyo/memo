#!/bin/bash
# ===================================================================
#  Memorization Benchmark (Chen et al. eval) — init_opti 통합 shell
#  inference + eval + trade-off curve 를 한 번에 (init=1,3,5 각각 lr sweep)
#
#  각 init_steps 마다:
#    1) inference  : run_ini_opti.py (resume: 이미 생성된 prompt 는 skip)
#    2) eval       : SSCD-to-GT + T2I → {init_dir}/eval/
#    3) trade-off  : collect_trd + plot_trd → {trd_path}/trd/{csv,plot}/
#       trd_path = .../memoloss=minimization/init=${init_steps}/nsteps=2/gap=3  (lr 부모)
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/eval_chen/run_memo_init.sh
#    GPER=1 bash shells/eval_chen/run_memo_init.sh   # GPER gradient preconditioning on
# ===================================================================
# set -euo pipefail

# =========================== 1. [Parser] ===========================
gpu=0
model="stable-diffusion-v1-4"
method="ddim"
NFE=50
cfg_ddim=7.5
cfg_cno=6.0
cfg_init_opti=7.5
cfg_start_ratio=0.0           # staged CFG. 0.0 = 항상 CFG
type_memo_loss="minimization" # minimization | threshold
memo_threshold=0.3            # threshold 일 때만
gper="${GPER:-0}"             # GPER (arXiv:2602.08646): 0=off, 1=on. 경로 gper=${gper}/ 와 연동.
grad_prcd_flag=""; [[ "${gper}" == "1" ]] && grad_prcd_flag="--grad_prcd"
if [[ "${type_memo_loss}" == "threshold" ]]; then memothr_dir="memothr=${memo_threshold}/"; else memothr_dir=""; fi
seed=42
num_samples=10
num_images_per_prompt=5
b_size=${num_images_per_prompt}
text_name="cvpr2025_memo_prompt.txt"

# b. CNO (infoNCE) config (참고용 — 이 shell 은 init_opti 만 실행)
iopt_iter=3
iopt_lr=0.01
infoNCE_temp=0.1
window_size=16
gamma=1.0

# init_opti config
num_opt_steps=2
gap_steps=3
base_s_ratio=0.5
lambda_align=0.00
init_opti_prompt_dir="examples/assets/${text_name}"
t2i_prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/cvpr2025_webster_gt"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# ★ sweep lists
init_steps_list=(1 3 5)
lr_list=(0.02 0.04 0.06 0.08)

# Workdir prefix
base_dir="workdir/memorization/sd14_base"
cfg_nfe_init="CFG=${cfg_init_opti}_NFE=${NFE}"
model_key="ckpt/${model}"
init_prefix="${base_dir}/init_opti/${cfg_nfe_init}/gper=${gper}/cfgsr=${cfg_start_ratio}/base_s_ratio=${base_s_ratio}_lambda_align=${lambda_align}/memoloss=${type_memo_loss}/${memothr_dir}"

cd "${ROOT_DIR:-.}"  # ori_memo/ 에서 실행 가정; 다른 위치면 ROOT_DIR override

echo "========================================="
echo "  init_opti 통합 (inference + eval + trade-off) — Chen"
echo "  model=${model}  NFE=${NFE}  cfg=${cfg_init_opti}  seed=${seed}  batch=${b_size}"
echo "  gper=${gper}  cfgsr=${cfg_start_ratio}  base_s_ratio=${base_s_ratio}  lambda_align=${lambda_align}"
echo "  init_steps=(${init_steps_list[*]})  lr=(${lr_list[*]})"
echo "========================================="

# =========================== 2. [init 별 루프] ===========================
for init_steps in "${init_steps_list[@]}"; do
    echo ""
    echo "############## init_steps=${init_steps} ##############"
    trd_path="${init_prefix}init=${init_steps}/nsteps=${num_opt_steps}/gap=${gap_steps}"
    echo "  trd_path(lr 부모) = ${trd_path}"

    # ---- 2-1) lr sweep: inference + eval ----
    for lr in "${lr_list[@]}"; do
        init_dir="${trd_path}/lr=${lr}/seed=${seed}/batch=${b_size}"
        init_eval="${init_dir}/eval"
        echo "---- init=${init_steps} lr=${lr} ----"
        echo "  init → ${init_dir}"

        # (a) inference (resume: img 가 이미 모두 있으면 skip)
        python run_ini_opti.py \
            --NFE ${NFE} --cfg ${cfg_init_opti} --lr ${lr} \
            --model_key ${model_key} \
            --init_steps ${init_steps} --num_steps ${num_opt_steps} --gap_steps ${gap_steps} \
            --base_s_ratio ${base_s_ratio} --lambda_align ${lambda_align} \
            --cfg_start_ratio ${cfg_start_ratio} --type_memo_loss ${type_memo_loss} --memo_threshold ${memo_threshold} ${grad_prcd_flag} \
            --base_seed ${seed} --num_seeds ${num_images_per_prompt} \
            --prompt_dir ${init_opti_prompt_dir} --num_samples ${num_samples} \
            --device cuda:${gpu} --output_dir ${init_dir}

        # # (b) eval: SSCD-to-GT + T2I → merge
        # mkdir -p ${init_eval}
        # python compute_sscd_gt.py \
        #     --gen_dir ${init_dir}/result --ref_dir ${gt_ref_dir} \
        #     --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        #     --gpu ${gpu} \
        #     --output_csv ${init_eval}/chen_sscd_gt_metrics.csv
        # python -m compute_t2i_metrics \
        #     --eval_dir ${init_dir}/result --prompt_dir ${t2i_prompt_dir} \
        #     --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        #     --output_csv ${init_eval}/chen_t2i_metrics.csv \
        #     --device cuda:${gpu} ${CS_FLAG}
        # python merge_benchmark.py --collect_dir ${init_eval}
        # echo "  [CHECK] eval files:"; /bin/ls ${init_eval}/*.csv 2>/dev/null
    done

    # ---- 2-2) init별 trade-off curve ----
    echo "==== [trade-off] collect_trd + plot_trd → ${trd_path}/trd/ ===="
    python collect_trd.py --method init_opti --path "${trd_path}" \
        --lr_list ${lr_list[@]} --seed ${seed} --batch ${b_size}
    python plot_trd.py --csv "${trd_path}/trd/csv/total_metrics.csv" --out_dir "${trd_path}/trd/plot"
    echo "  [Done] init=${init_steps}: ${trd_path}/trd/{csv,plot}/"
done

# =========================== 3. [init=1,3,5 overlay 비교 plot] ===========================
echo ""
echo "==== [compare] init=(${init_steps_list[*]}) trade-off curve overlay ===="
compare_out="${init_prefix}trd_compare"
python plot_init_compare.py \
    --base_path "${init_prefix}" \
    --inits ${init_steps_list[@]} \
    --nsteps ${num_opt_steps} --gap ${gap_steps} \
    --out_dir "${compare_out}"
echo "[Done] init compare plot → ${compare_out}/tradeoff_compare_*.png"

echo ""
echo "[All Done] init=(${init_steps_list[*]}) 각각 inference+eval+trade-off + overlay 비교 완료"
