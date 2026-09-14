#!/bin/bash
# ===================================================================
#  ⑦ spec_low optimization: latent x_t (lr sweep) — sdv1_500 (Wen) eval
#  loss: spec_low = (1/B)Σ_{p<L_lh} E_p
#        ε_cfg(z_{t-1}, t-1) → rfft(ortho) → block energy → 저주파 대역 합
#        (candidate_proxy ⑦ spectral score low freq energy — 측정 GOOD 개형:
#          memo 58→101 vs text 52→4, 곡선-mean AUC 1.000, 2026-09-05)
#
#  ours/run_memo_eps_ref_mse.sh (⑤) 패턴의 spec_low 버전 — 차이:
#    · --xt_loss spec_low (⑦) | XT_LOSS 환경변수로 memo_proxy(①)/eps_ref_mse(⑤) 전환
#    · --lh_ratio LH_RATIO (기본 0.1 → P=512 중 앞 51 blocks, WG 기준 ≈51)
#    · trd_path 에 lh=${lh_ratio}/bs=${block_size} knob 추가 (method-level, lr은 최内층)
#      — XT_LOSS 오버라이드(①/⑤) 시 method 세그먼트를 실제 loss 명으로 전환
#
#  파이프라인 (각 lr 마다):
#    1. DDIM forward 0 → init_steps (no grad)
#    2. x_t optimize: spec_low minimize
#       각 update: z_{t-1} leaf → ε_cfg(z_{t-1}) → FFT → low-band sum → backward
#       → Adam step → DDIM gap (no grad) 전진
#    3. DDIM resume → image
#    4. eval: SSCD-to-GT + T2I → total_metrics.csv
#  종료 후: trade-off curve
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/eval_web/sd14/ours/run_memo_spec_low.sh
#    LR_LIST="0.11 0.15 0.20" bash shells/eval_web/sd14/ours/run_memo_spec_low.sh
# ===================================================================

# =========================== 0. [Timer] ===========================
SECONDS=0    # 스크립트 전체 총 소요 시간 측정 (종료 시 [Elapsed] 출력)

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

# ★ eval subset — 3차 프로토콜: NUM_EVAL = 프롬pt **종류 수** N (= eval/<N>, trd/<N>
#   폴더명). 실제 평가 장수 = N × ipp. num_samples(생성된 프롬pt 수)를 상한으로 clamp.
num_eval_prompts="${NUM_EVAL:-100}"
(( num_eval_prompts > num_samples )) && num_eval_prompts=${num_samples}
num_eval="${num_eval_prompts}"          # 폴더명 N = 종류 수 (3차 — 장수 아님)
DEVICE="cuda:${gpu}"

# spec_low (⑦) optimization config
xt_loss="${XT_LOSS:-spec_low}"       # ⑦ (기본) | memo_proxy(①) | eps_ref_mse(⑤)
lh_ratio="${LH_RATIO:-0.1}"          # low-band 비율 (P=512 → 앞 51 blocks, WG ≈51)
block_size="${BLOCK_SIZE:-16}"
init_steps="${INIT_STEPS:-2}"
num_steps="${NUM_STEPS:-9}"
gap_steps="${GAP_STEPS:-4}"
base_s_ratio="${BASE_S_RATIO:-0.5}"
lambda_align="${LAMBDA_ALIGN:-0.1}"

# ★ lr sweep (LR_LIST 환경변수로 오버라이드 가능)
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
# XT_LOSS 오버라이드 시 트리 오분류 방지 — method 세그먼트를 실제 loss 명으로,
# ⑦ 전용 knob(lh=, bs=)은 spec_low 일 때만 경로에 인코딩
if [[ "${xt_loss}" == "spec_low" ]]; then
    method_dir="spec_low"
    extra_knobs="/lh=${lh_ratio}/bs=${block_size}"
else
    method_dir="${xt_loss}"
    extra_knobs=""
fi
trd_path="${base_dir}/ours/${method_dir}/CFG=${cfg}_NFE=${NFE}/cfgsr=${cfg_start_ratio}/init=${init_steps}/nsteps=${num_steps}/gap=${gap_steps}${extra_knobs}"

cd "${ROOT_DIR:-.}"

echo "========================================="
echo "  spec_low (⑦) x_t Optimization (lr sweep) — sdv1_500 (Wen)"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  seed=${SEED}"
echo "  opti_mode=xt  xt_loss=${xt_loss} (grad head only, x_T fixed)"
echo "  lh_ratio=${lh_ratio} (block_size=${block_size})  loss=(1/B)Σ_{p<L_lh} E_p"
echo "  init_steps=${init_steps}  num_steps=${num_steps}  gap=${gap_steps}"
echo "  base_s_ratio=${base_s_ratio}  lambda_align=${lambda_align}"
echo "  cfg_start_ratio=${cfg_start_ratio}  batch=${b_size}"
echo "  prompt=${text_name}  lr_list=(${lr_list[*]})"
echo "  num_eval=${num_eval} prompt 종류 (=N, 3차 프로토콜) → eval/${num_eval}, trd/${num_eval}"
echo "  trd_path=${trd_path}"
echo "========================================="

# =========================== 2. [lr sweep] ===========================
for lr in "${lr_list[@]}"; do
    OUTPUT_DIR="${trd_path}/batch=${b_size}/seed=${SEED}/lr=${lr}"
    echo ""
    echo "---- lr=${lr} ----"

    # inference + optimize
    python run_ini_opti.py \
        --opti_mode xt \
        --xt_loss ${xt_loss} \
        --lh_ratio ${lh_ratio} --block_size ${block_size} \
        --NFE ${NFE} --cfg ${cfg} \
        --model_key ${model_key} \
        --cfg_start_ratio ${cfg_start_ratio} ${cfgsr_cond_flag} \
        --init_steps ${init_steps} \
        --num_steps ${num_steps} --gap_steps ${gap_steps} \
        --lr ${lr} \
        --base_s_ratio ${base_s_ratio} --lambda_align ${lambda_align} \
        --type_memo_loss minimization \
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
