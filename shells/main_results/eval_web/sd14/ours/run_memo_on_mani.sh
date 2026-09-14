#!/bin/bash
# ===================================================================
#  ⑥ [On-main] Compression into on-manifold — seed 인력 + w_twd·Twd_gap
#  loss = Σ_{i<j}‖x^i_t−x^j_t‖²/D  +  w_twd · memo_proxy(① head)
#
#  관측 동기: general 프롬pt는 seed별 x_t가 collapse, memorized는 심하게 벌어짐
#  → seed 인력으로 뭉치게 해 memorized를 general 거동으로 압축(compression)
#  ※ 원식 −Σ 리터럴(분리 방향)은 ON_MAIN_SPREAD=1 로 전환
#
#  파이프라인 (각 lr 마다):
#    1. DDIM forward 0 → init_steps (no grad)
#    2. x_t optimize: on_main loss (seed 4개 동시 — batch = num_seeds)
#    3. DDIM resume → 이미지 4장/프롬pt
#    4. eval: SSCD-to-GT(sdv1) + T2I → total_metrics.csv
#  종료 후: trade-off curve
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/eval_web/sd14/ours/run_memo_on_main.sh
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
num_seeds=4                            # ★ on_main: seed 4개 = loss_1의 인력 대상
batch_txt="${BT:-1}"                   # 프롬pt 배치 (블록 안쪽 쌍만 인력 — 교차 오염 없음)
b_size=${num_seeds}

# ★ eval subset — 평가할 inference 이미지 수 NUM_EVAL(요청) → clamp 후 실제 평가 장수가
#   폴더명: eval/<num_eval>, trd/<num_eval> (예: 요청 100 > 생성 10장 → eval/10, trd/10)
num_eval="${NUM_EVAL:-100}"
num_eval_prompts=$((num_eval / num_seeds))
(( num_eval_prompts > num_samples )) && num_eval_prompts=${num_samples}
num_eval=$((num_eval_prompts * num_seeds))               # 실제 평가 이미지 수 = 폴더명
DEVICE="cuda:${gpu}"

# on_main (⑥) optimization config
init_steps="${INIT_STEPS:-2}"
num_steps="${NUM_STEPS:-9}"
gap_steps="${GAP_STEPS:-4}"
base_s_ratio="${BASE_S_RATIO:-0.5}"
w_twd="${W_TWD:-6.0}"                  # loss_2 (Tweedie gap) 가중치
on_main_spread_flag=""; [[ "${ON_MAIN_SPREAD:-0}" == "1" ]] && on_main_spread_flag="--on_main_spread"

# ★ lr sweep (LR_LIST 환경변수로 오버라이드 가능)
lr_list=(${LR_LIST:-0.02 0.05 0.08 0.10})

# prompts — sdv1_500 (Wen) 세트
text_name="sdv1_500_mem.txt"
prompt_dir="examples/assets/${text_name}"
t2i_prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# workdir — on_main 전용 폴더
model_key="ckpt/${model}"
base_dir="workdir/memorization/sd14_base"
trd_path="${base_dir}/ours/on_main/CFG=${cfg}_NFE=${NFE}/cfgsr=${cfg_start_ratio}/init=${init_steps}/nsteps=${num_steps}/gap=${gap_steps}/w_twd=${w_twd}"

cd "${ROOT_DIR:-.}"

echo "========================================="
echo "  ⑥ [On-main] compression x_t Optimization (lr sweep) — sdv1_500"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  seed=${SEED}"
echo "  opti_mode=xt --on_main  w_twd=${w_twd} ${on_main_spread_flag}"
echo "  init_steps=${init_steps}  num_steps=${num_steps}  gap=${gap_steps}"
echo "  num_seeds=${num_seeds} (seed attraction targets)  num_samples=${num_samples}"
echo "  cfg_start_ratio=${cfg_start_ratio}"
echo "  lr_list=(${lr_list[*]})"
echo "  num_eval=${num_eval} imgs (eval prompts=${num_eval_prompts}) → eval/${num_eval}, trd/${num_eval}"
echo "  trd_path=${trd_path}"
echo "========================================="

# =========================== 2. [lr sweep] ===========================
for lr in "${lr_list[@]}"; do
    OUTPUT_DIR="${trd_path}/batch=${b_size}/seed=${SEED}/lr=${lr}"
    echo ""
    echo "---- lr=${lr} ----"

    # inference + optimize (batch_txt=1: batch 행 = 한 프롬pt의 seed 4개)
    python run_ini_opti.py \
        --opti_mode xt \
        --on_main --w_twd ${w_twd} ${on_main_spread_flag} \
        --NFE ${NFE} --cfg ${cfg} \
        --model_key ${model_key} \
        --cfg_start_ratio ${cfg_start_ratio} ${cfgsr_cond_flag} \
        --init_steps ${init_steps} \
        --num_steps ${num_steps} --gap_steps ${gap_steps} \
        --lr ${lr} \
        --base_s_ratio ${base_s_ratio} \
        --type_memo_loss minimization \
        --base_seed ${SEED} --num_seeds ${num_seeds} --batch_txt ${batch_txt} \
        --prompt_dir ${prompt_dir} --num_samples ${num_samples} \
        --device ${DEVICE} \
        --output_dir ${OUTPUT_DIR}

    # eval
    init_eval="${OUTPUT_DIR}/eval/${num_eval}"
    mkdir -p ${init_eval}
    python compute_sscd_gt.py \
        --gen_dir ${OUTPUT_DIR}/result --ref_dir ${gt_ref_dir} \
        --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_seeds} \
        --gpu ${gpu} --output_csv ${init_eval}/chen_sscd_gt_metrics.csv
    python -m compute_t2i_metrics \
        --eval_dir ${OUTPUT_DIR}/result --prompt_dir ${t2i_prompt_dir} \
        --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_seeds} \
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
