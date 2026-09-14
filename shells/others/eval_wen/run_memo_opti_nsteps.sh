#!/bin/bash
# ===================================================================
#  Memorization Benchmark (Wen et al. ICLR 2024 eval 기준)
#  ours(init_opti) — nsteps × lr sweep 버전
#
#  ★ run_memo_opti_inits.sh 기반. for 2 개(외부 nsteps, 내부 lr).
#    init_steps=5 고정 하고 num_opt_steps(nsteps) × lr 을 모두 sweep.
#  Metrics: SSCD-to-GT + T2I → {method_dir}/eval/
#  Reference: datasets/wen2024_memorized/ (Wen et al. 공개 GT)
#
#  환경: conda div_DM (실행 전 activate)
#  실행: ori_memo/ 디렉토리에서  bash shells/eval_wen/run_memo_opti_nsteps.sh
#  (trd curve 는 eval 종료 후 별도 shell 로)
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
text_name="new_memorized_text_prompt.txt"

# b. CNO (infoNCE) config
iopt_iter=3
iopt_lr=0.01
infoNCE_temp=0.1
window_size=16
gamma=1.0

# ★ sweep lists: 외부 nsteps, 내부 lr
nsteps_list=(1 3 5 7 10)
lr_list=(0.02 0.04 0.06 0.08)

# =========================== 2. [FLAG] ===========================
STD_FLAG="--model ${model} --method ${method} --device cuda:${gpu}"
ETC_FLAG="--NFE ${NFE} --seed ${seed}"
INF_FLAG="--b_size ${b_size} --num_samples ${num_samples} --num_images_per_prompt ${num_images_per_prompt}"
DIR_FLAG="--prompt_dir examples/assets/${text_name}"

CNO_FLAG="--iopt_diverse --iopt_loss_type infoNCE \
--i_opt_iter ${iopt_iter} --i_opt_lr ${iopt_lr} --iopt_cfg_tweedie \
--infoNCE_temp ${infoNCE_temp} --window_size ${window_size} --gamma ${gamma} --n_aug_samples 0"

base_dir="workdir/memorization/sd14_memor_LAION2B_40k"
cfg_nfe_init="CFG=${cfg_init_opti}_NFE=${NFE}"
model_key="ckpt/${model}"
gt_ref_dir="datasets/wen2024_memorized"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

cd "${ROOT_DIR:-.}"                    # ori_memo/ 에서 실행 가정

echo "========================================="
echo "  ours nsteps×lr sweep  (model=${model})"
echo "  nsteps_list=${nsteps_list[*]}"
echo "  lr_list=${lr_list[*]}"
echo "========================================="

# =========================== 3. [sweep: nsteps × lr] ===========================
for num_opt_steps in "${nsteps_list[@]}"; do
    echo ""
    echo "############################################################"
    echo "  nsteps=${num_opt_steps}"
    echo "############################################################"

    init_steps=5                        # ★ init_step=5 고정
    gap_steps=3
    base_s_ratio=0.5
    lambda_align=0.00
    cfg_start_ratio=0.0                 # staged CFG (0.0=항상 CFG)
    type_memo_loss="minimization"       # minimization | threshold
    memo_threshold=0.3                  # type_memo_loss=threshold 일 때만
grad_prcd_flag=""; [[ "${GRAD_PRCD:-0}" == "1" ]] && grad_prcd_flag="--grad_prcd"  # GPER gradient preconditioning (arXiv:2602.08646)
    if [[ "${type_memo_loss}" == "threshold" ]]; then memothr_dir="memothr=${memo_threshold}/"; else memothr_dir=""; fi
    init_opti_prompt_dir="examples/assets/${text_name}"
    t2i_prompt_dir="examples/assets/${text_name}"

    for lr in "${lr_list[@]}"; do
        echo ""
        echo "==================== nsteps=${num_opt_steps}  lr=${lr} =========================="

        # ---- [Workdir] ----
        init_dir="${base_dir}/init_opti/${cfg_nfe_init}/cfgsr=${cfg_start_ratio}/base_s_ratio=${base_s_ratio}_lambda_align=${lambda_align}/memoloss=${type_memo_loss}/${memothr_dir}init=${init_steps}/nsteps=${num_opt_steps}/gap=${gap_steps}/lr=${lr}/seed=${seed}/batch=${b_size}"
        init_eval="${init_dir}/eval"
        init_comp="${init_dir}/comp"
        echo "  init  → ${init_dir}"

        # ---- 4. [Inference: init_opti] ----
        echo "================== [INFO]: init_opti Inference (nsteps=${num_opt_steps}, lr=${lr}) =================="
        echo "  [CKPT] ${model_key}"
        python run_ini_opti.py \
            --NFE ${NFE} --cfg ${cfg_init_opti} --lr ${lr} \
            --model_key ${model_key} \
            --init_steps ${init_steps} --num_steps ${num_opt_steps} --gap_steps ${gap_steps} \
            --base_s_ratio ${base_s_ratio} --lambda_align ${lambda_align} \
            --cfg_start_ratio ${cfg_start_ratio} --type_memo_loss ${type_memo_loss} --memo_threshold ${memo_threshold} ${grad_prcd_flag} \
            --base_seed ${seed} --num_seeds ${num_images_per_prompt} \
            --prompt_dir ${init_opti_prompt_dir} --num_samples ${num_samples} \
            --device cuda:${gpu} --output_dir ${init_dir}

        # ---- 5. [Eval: init_opti] (Wen: SSCD-to-GT + T2I) ----
        echo "================== [INFO]: Eval [init_opti] → ${init_eval}/ =================="
        mkdir -p ${init_eval}
        python compute_sscd_gt.py \
            --gen_dir ${init_dir}/result --ref_dir ${gt_ref_dir} \
            --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
            --gpu ${gpu} \
            --output_csv ${init_eval}/chen_sscd_gt_metrics.csv

        python -m compute_t2i_metrics \
            --eval_dir ${init_dir}/result --prompt_dir ${t2i_prompt_dir} \
            --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
            --output_csv ${init_eval}/chen_t2i_metrics.csv \
            --device cuda:${gpu} ${CS_FLAG}

        python merge_benchmark.py --collect_dir ${init_eval}

        echo "  [CHECK] eval files:"
        /bin/ls ${init_eval}/*.csv 2>/dev/null
    done
done

echo ""
echo "[Done] nsteps×lr sweep 완료."
echo "  trd 생성: run_trd 계열 shell (각 nsteps 경로의 lr sweep)"
