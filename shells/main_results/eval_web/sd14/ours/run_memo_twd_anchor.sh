#!/bin/bash
# ===================================================================
#  ⑧ [Anchor: Twd_gap] initial noise 앵커 — s_Δ + Tweedie gap (lr sweep)
#  loss = ‖ε_uc(x_T) − ε_c(x_T)‖₂  +  w_twd · ‖ε_noise − ε_s‖²/D
#         (s_Δ: x_T에서 직접, per-sample norm | gap: x_T→x̂_{0|T}→x_s→ε_s)
#
#  x_T 직접 갱신 (adjoint 불필요 — UNet 2회 얕은 헤드에서 grad 곧장).
#  init/gap_steps 무의미 (항상 x_T), num_steps = x_T 갱신 횟수.
#
#  파이프라인 (각 lr 마다):
#    1. x_T optimize: anchor loss (num_steps회 갱신)
#    2. DDIM inference (x_T부터 전체 NFE)
#    3. eval: SSCD-to-GT(sdv1) + T2I → total_metrics.csv
#  종료 후: trade-off curve
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/eval_web/sd14/ours/run_memo_twd_anchor.sh
# ===================================================================

# =========================== 1. [Parser] ===========================
gpu="${GPU:-0}"
model="stable-diffusion-v1-4"
NFE=50
cfg=7.5
cfg_start_ratio="${CFGSR:-0.10}"       # staged CFG: early step conditional만
cfgsr_cond_flag=""; [[ "${cfg_start_ratio}" != "0.0" ]] && cfgsr_cond_flag="--cfgsr_cond"
SEED=42
num_samples=10
num_images_per_prompt=1
b_size=${num_images_per_prompt}
DEVICE="cuda:${gpu}"

# anchor (⑧) optimization config
num_steps="${NUM_STEPS:-9}"             # x_T 갱신 횟수 (init/gap_steps 무의미)
base_s_ratio="${BASE_S_RATIO:-0.5}"
w_twd="${W_TWD:-1.0}"             # Tweedie gap 가중치 (loss = s_Δ + w_twd·gap — ⑥ 관례 공용)

# ★ lr sweep (LR_LIST 환경변수로 오버라이드 가능)
lr_list=(${LR_LIST:-0.05 0.10 0.15 0.20})

# prompts — sdv1_500 (Wen) 세트
text_name="sdv1_500_mem.txt"
prompt_dir="examples/assets/${text_name}"
t2i_prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# workdir — anchor 전용 폴더 (knob=lr 가장 안쪽, trd는 batch/seed 아래)
model_key="ckpt/${model}"
base_dir="workdir/memorization/sd14_base"
trd_path="${base_dir}/ours/twd_anchor/CFG=${cfg}_NFE=${NFE}/cfgsr=${cfg_start_ratio}/w_twd=${w_twd}"

cd "${ROOT_DIR:-.}"

echo "========================================="
echo "  ⑧ [Anchor: Twd_gap] initial noise x_T Optimization (lr sweep)"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  seed=${SEED}"
echo "  --anchor  w_twd=${w_twd}  num_steps=${num_steps} (direct x_T update, no adjoint)"
echo "  base_s_ratio=${base_s_ratio}  num_samples=${num_samples}  batch=${b_size}"
echo "  cfg_start_ratio=${cfg_start_ratio}"
echo "  lr_list=(${lr_list[*]})"
echo "  trd_path=${trd_path}"
echo "========================================="

# =========================== 2. [lr sweep] ===========================
for lr in "${lr_list[@]}"; do
    OUTPUT_DIR="${trd_path}/batch=${b_size}/seed=${SEED}/lr=${lr}"
    echo ""
    echo "---- lr=${lr} ----"

    # inference + optimize
    python run_ini_opti.py \
        --anchor --w_twd ${w_twd} \
        --NFE ${NFE} --cfg ${cfg} \
        --model_key ${model_key} \
        --cfg_start_ratio ${cfg_start_ratio} ${cfgsr_cond_flag} \
        --num_steps ${num_steps} \
        --lr ${lr} \
        --base_s_ratio ${base_s_ratio} \
        --type_memo_loss minimization \
        --base_seed ${SEED} --num_seeds ${num_images_per_prompt} \
        --prompt_dir ${prompt_dir} --num_samples ${num_samples} \
        --device ${DEVICE} \
        --output_dir ${OUTPUT_DIR}

    # eval
    init_eval="${OUTPUT_DIR}/eval"
    mkdir -p ${init_eval}
    python compute_sscd_gt.py \
        --gen_dir ${OUTPUT_DIR}/result --ref_dir ${gt_ref_dir} \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu ${gpu} --output_csv ${init_eval}/chen_sscd_gt_metrics.csv
    python -m compute_t2i_metrics \
        --eval_dir ${OUTPUT_DIR}/result --prompt_dir ${t2i_prompt_dir} \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --output_csv ${init_eval}/chen_t2i_metrics.csv \
        --device ${DEVICE} ${CS_FLAG}
    python merge_benchmark.py --collect_dir ${init_eval}
    echo "  [CHECK] eval:"; /bin/ls ${init_eval}/*.csv 2>/dev/null
done

# =========================== 3. [Trade-off] ===========================
# collect_trd init_opti 신규 패턴: {path}/batch=B/seed=S/lr=*/eval — knob=lr 가장 안쪽
trd_dir="${trd_path}/batch=${b_size}/seed=${SEED}/trd"
mkdir -p ${trd_dir}/plot ${trd_dir}/csv
echo ""
echo "==== [trade-off] collect_trd + plot_trd → ${trd_dir}/ ===="
python collect_trd.py --method init_opti --path "${trd_path}" \
    --lr_list ${lr_list[@]} --seed ${SEED} --batch ${b_size} \
    --trd_dir "${trd_dir}"
python plot_trd.py --csv "${trd_dir}/csv/total_metrics.csv" --out_dir "${trd_dir}/plot"

echo ""
echo "[Done] → ${trd_dir}/{csv,plot}/"
