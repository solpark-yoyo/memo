#!/bin/bash
# ===================================================================
#  x_t spectral optimization (lr sweep) — sdv1_500 (Wen) eval
#  inference timestep t 에서 x_t 의 compact spectral L2² 를 minimize.
#
#  eval_web/sd14/others/run_memo_spectral.sh 의 ours 버전 — 차이:
#    · 프롬트/GT 경로 갱신 (cvpr2025_memo_prompt + cvpr2025_webster_gt
#      → sdv1_500_mem + datasets/memo/eval_sscd/sd14/...)
#    · batch=4 (ours 관례 — 프롬pt당 seed 수)
#    · lr_list 를 LR_LIST 환경변수로 오버라이드 가능
#    · 스크립트 전체 총 소요 시간 측정 ([Elapsed])
#  (loss=xt 갈래 — ours/run_memo_spectral_eps.sh 는 eps 갈래)
#
#  파이프라인 (각 lr 마다):
#    1. DDIM forward 0 → init_steps (no grad)
#    2. x_t optimize: spectral L2² minimize (FFT only, no DDIM chain backward)
#       각 update: zt → FFT loss → backward → step → detach → DDIM gap (no grad)
#    3. DDIM continue → image
#    4. eval: SSCD-to-GT + T2I → total_metrics.csv
#  종료 후: trade-off curve
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/eval_web/sd14/ours/run_memo_spectral.sh
# ===================================================================

# =========================== 0. [Timer] ===========================
SECONDS=0    # 스크립트 전체 총 소요 시간 측정 (종료 시 [Elapsed] 출력)

# =========================== 1. [Parser] ===========================
gpu=0
model="stable-diffusion-v1-4"
NFE=50
cfg=7.5
cfg_start_ratio="${CFGSR:-0.10}"       # staged CFG: early step conditional만
cfgsr_cond_flag=""; [[ "${cfg_start_ratio}" != "0.0" ]] && cfgsr_cond_flag="--cfgsr_cond"
SEED=42
num_samples=10
num_images_per_prompt=1                # ★ ours 관례: 프롬pt당 seed 수 (=batch)
b_size=${num_images_per_prompt}

# ★ eval subset — 평가할 inference 이미지 수 NUM_EVAL(요청) → clamp 후 실제 평가 장수가
#   폴더명: eval/<num_eval>, trd/<num_eval> (예: 요청 100 > 생성 10장 → eval/10, trd/10)
num_eval="${NUM_EVAL:-100}"
num_eval_prompts=$((num_eval / num_images_per_prompt))
(( num_eval_prompts > num_samples )) && num_eval_prompts=${num_samples}
num_eval=$((num_eval_prompts * num_images_per_prompt))   # 실제 평가 이미지 수 = 폴더명
DEVICE="cuda:${gpu}"

# spectral optimization config
init_steps="${INIT_STEPS:-2}"
num_steps="${NUM_STEPS:-5}"
gap_steps="${GAP_STEPS:-4}"
block_size="${BLOCK_SIZE:-16}"

# ★ lr sweep (LR_LIST 환경변수로 오버라이드 가능 — cfgsr=0.15 nsteps 축: NUM_STEPS 환경변수로 3/5/10)
lr_list=(${LR_LIST:-0.11 0.15 0.20})

# prompts — sdv1_500 (Wen) 세트
text_name="sdv1_500_mem.txt"
prompt_dir="examples/assets/${text_name}"
t2i_prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# workdir
model_key="ckpt/${model}"
base_dir="workdir/memorization/sd14_base"
trd_path="${base_dir}/ours/spectral_opt/loss=xt/CFG=${cfg}_NFE=${NFE}/cfgsr=${cfg_start_ratio}/init=${init_steps}/nsteps=${num_steps}/gap=${gap_steps}"

cd "${ROOT_DIR:-.}"

echo "========================================="
echo "  x_t Spectral Optimization (lr sweep) — sdv1_500 (Wen)"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  seed=${SEED}"
echo "  init_steps=${init_steps}  num_steps=${num_steps}  gap=${gap_steps}"
echo "  cfg_start_ratio=${cfg_start_ratio}  block_size=${block_size}  batch=${b_size}"
echo "  prompt=${text_name}  num_samples=${num_samples}"
echo "  lr_list=(${lr_list[*]})"
echo "  num_eval=${num_eval} imgs (eval prompts=${num_eval_prompts}) → eval/${num_eval}, trd/${num_eval}"
echo "  trd_path=${trd_path}"
echo "========================================="

# =========================== 2. [lr sweep] ===========================
for lr in "${lr_list[@]}"; do
    OUTPUT_DIR="${trd_path}/batch=${b_size}/seed=${SEED}/lr=${lr}"
    echo ""
    echo "---- lr=${lr} ----"

    # inference + optimize
    python optimize_xt_spectral.py \
        --NFE ${NFE} --cfg ${cfg} \
        --model_key ${model_key} \
        --cfg_start_ratio ${cfg_start_ratio} ${cfgsr_cond_flag} \
        --init_steps ${init_steps} \
        --num_steps ${num_steps} --gap_steps ${gap_steps} \
        --lr ${lr} --block_size ${block_size} \
        --base_seed ${SEED} --num_seeds ${num_images_per_prompt} \
        --prompt_dir ${prompt_dir} --num_samples ${num_samples} \
        --device ${DEVICE} \
        --output_dir ${OUTPUT_DIR}

    # eval
    init_eval="${OUTPUT_DIR}/eval/${num_eval}"
    mkdir -p ${init_eval}
    python compute_sscd_gt.py \
        --gen_dir ${OUTPUT_DIR}/result --ref_dir ${gt_ref_dir} \
        --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu ${gpu} --output_csv ${init_eval}/chen_sscd_gt_metrics.csv
    python -m compute_t2i_metrics \
        --eval_dir ${OUTPUT_DIR}/result --prompt_dir ${t2i_prompt_dir} \
        --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_images_per_prompt} \
        --output_csv ${init_eval}/chen_t2i_metrics.csv \
        --device ${DEVICE} ${CS_FLAG}
    python merge_benchmark.py --collect_dir ${init_eval}
    echo "  [CHECK] eval:"; /bin/ls ${init_eval}/*.csv 2>/dev/null
done

# =========================== 3. [Trade-off] ===========================
trd_dir="${trd_path}/batch=${b_size}/seed=${SEED}/trd/${num_eval}"
echo ""
echo "==== [trade-off] collect_trd + plot_trd → ${trd_dir}/ ===="
python collect_trd.py --method init_opti --path "${trd_path}" \
    --lr_list ${lr_list[@]} --seed ${SEED} --batch ${b_size} --eval_sub ${num_eval} \
    --trd_dir "${trd_dir}"
python plot_trd.py --csv "${trd_dir}/csv/total_metrics.csv" --out_dir "${trd_dir}/plot"

echo ""
echo "[Done] → ${trd_dir}/{csv,plot}/"

# ---- [Elapsed] 전체 config 총 소요 시간 ----
echo ""
echo "[Elapsed] $((SECONDS/3600))hrs $(( (SECONDS%3600)/60 ))min $((SECONDS%60))sec (total ${SECONDS}sec)"
