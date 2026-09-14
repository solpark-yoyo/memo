#!/bin/bash
# ===================================================================
#  Wen et al. (ICLR 2024) prompt augmentation — Chen eval (trade-off)
#  repo: baselines/wen_prompt_aug/ (YuxinWenRick/diffusion_memorization)
#  논문: arXiv 2407.21720 — ||ε_cond − ε_uncond|| 를 tolerance λ 이하로
#        낮추도록 prompt embedding 을 GD 최적화 (prompt augmentation)
#
#  ★ knob = optim_target_loss (λ, 완화 강도 — 작을수록 강함)
#    SOTA 값 3.0 포함 (공식 노트북 inference_time_mitigation.ipynb)
#    나머지: optim_lr=0.05 · optim_iters=10 (논문 고정)
#
#  환경: conda env "${CONDA_ENV:-div_DM}"
#    - local_sd_pipeline.py randn_tensor import는 try/except 폴백 패치 완료
#      (diffusers 0.18/0.36 양쪽 호환, 2026-08-30)
#
#  실행: ori_memo/ 에서  bash shells/eval_chen/baselines/run_memo_chen_wen.sh
# ===================================================================

# =========================== 0. [Env] ===========================
export PYTHONUNBUFFERED=1
SECONDS=0    # 스크립트 전체 총 소요 시간 측정 (종료 시 [Elapsed] 출력)

# =========================== 1. [Config] ===========================
device="${DEVICE:-cuda:0}"            # argparse --device 로 직접 전달 (예: DEVICE=cuda:3)
NFE=50
cfg_guidance=7.5
seed=42
num_samples=500
batch=4
num_images_per_prompt=${batch}

# ★ eval subset — 평가할 inference 이미지 수 NUM_EVAL(요청) → clamp 후 실제 평가 장수가
#   폴더명: eval/<num_eval>, trd/<num_eval> (예: 요청 100 > 생성 10장 → eval/10, trd/10)
num_eval="${NUM_EVAL:-100}"
num_eval_prompts=$((num_eval / num_images_per_prompt))
(( num_eval_prompts > num_samples )) && num_eval_prompts=${num_samples}
num_eval=$((num_eval_prompts * num_images_per_prompt))   # 실제 평가 이미지 수 = 폴더명
model_id="ckpt/stable-diffusion-v1-4"          # 로컬 ckpt (repo 인자 지원)
text_name="sdv1_500_mem.txt"
t2i_prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# Wen mitigation params — 논문 고정
optim_lr=0.05
optim_iters=10
optim_target_steps=0

# ★ λ sweep (완화 강도 축, 0.7~3.0 등간격 5점) — ★3.0 = 논문 SOTA, 작을수록 완화 강함
target_loss_list=(0.7 1.3 1.9 2.4 3.0)

# ★ 러너: repo 내부 wen_inference.py (inference_mem.py 경로 재현 + PNG 저장만 추가)
#    원본 inference_mem.py 는 PNG 저장 없음 (내부 평가 후 지표 print/wandb 만)
WEN_ENTRY="baselines/wen_prompt_aug/wen_inference.py"

# =========================== 2. [Prompt jsonl] ===========================
# Wen 은 jsonl({"caption": ...}) 형식 — 최초 1회 변환
wen_jsonl="examples/assets/sdv1_500_mem_wen.jsonl"
if [[ ! -f "${wen_jsonl}" ]] || [[ $(wc -l < "${wen_jsonl}") -lt ${num_samples} ]]; then
    python - <<EOF
import json
lines = [l.strip() for l in open("${t2i_prompt_dir}") if l.strip()][:${num_samples}]
with open("${wen_jsonl}", "w") as f:
    for i, cap in enumerate(lines):
        f.write(json.dumps({"caption": cap, "index": i}) + "\n")
print(f"[jsonl] ${wen_jsonl}: {len(lines)} captions")
EOF
fi

# =========================== 3. [Workdir] ===========================
# 저장 경로는 shell 변수(ckpt·cfg·NFE·knob)로 직접 조립
base_dir="workdir/memorization/sd14_base"
model_tag="$(basename ${model_id})"       # ckpt/stable-diffusion-v1-4 → stable-diffusion-v1-4
output_path="${base_dir}/baselines/wen/${model_tag}/CFG=${cfg_guidance}_NFE=${NFE}"


echo "========================================="
echo "  Wen prompt augmentation — Chen trade-off (env: ${CONDA_ENV})"
echo "  model=${model_id}  NFE=${NFE}  CFG=${cfg_guidance}  seed=${seed}"
echo "  optim_lr=${optim_lr}(fixed)  iters=${optim_iters}(fixed)"
echo "  ★ target_loss λ sweep=(${target_loss_list[*]})"
echo "  prompt=${text_name}  num_samples=${num_samples}  batch=${batch}"
echo "  num_eval=${num_eval} imgs (eval prompts=${num_eval_prompts}) → eval/${num_eval}, trd/${num_eval}"
echo "========================================="

# =========================== 4. [λ sweep] ===========================
for tl in "${target_loss_list[@]}"; do
    gen_dir="${output_path}/it=${optim_iters}/batch=${num_images_per_prompt}/seed=${seed}/tl=${tl}"
    eval_dir="${gen_dir}/eval/${num_eval}"
    echo ""
    echo "---- target_loss(λ)=${tl} ----"

    # 4-1. Inference (prompt embedding 최적화 + 생성)
    echo "================== [INFO]: Wen Inference (tl=${tl}) =================="
    python ${WEN_ENTRY} \
        --device ${device} \
        --run_name "chen_tl${tl}" \
        --dataset "${wen_jsonl}" \
        --model_id "${model_id}" \
        --num_images_per_prompt ${num_images_per_prompt} \
        --num_prompts ${num_samples} \
        --guidance_scale ${cfg_guidance} \
        --num_inference_steps ${NFE} \
        --gen_seed ${seed} \
        --optim_lr ${optim_lr} \
        --optim_iters ${optim_iters} \
        --optim_target_steps ${optim_target_steps} \
        --optim_target_loss ${tl} \
        --output_dir "${gen_dir}/result"
    # 래퍼 사양: result/img_XXXX_YY.png 저장 (XXXX=prompt idx, YY=sample idx)

    # 4-2. Eval (Chen: SSCD + T2I)
    echo "================== [INFO]: Eval → ${eval_dir}/ =================="
    mkdir -p ${eval_dir}

    python compute_sscd_gt.py \
        --gen_dir ${gen_dir}/result --ref_dir ${gt_ref_dir} \
        --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu "${device##*:}" \
        --output_csv ${eval_dir}/chen_sscd_gt_metrics.csv

    python -m compute_t2i_metrics \
        --eval_dir ${gen_dir}/result --prompt_dir ${t2i_prompt_dir} \
        --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_images_per_prompt} \
        --output_csv ${eval_dir}/chen_t2i_metrics.csv \
        --device ${device} ${CS_FLAG}

    python merge_benchmark.py --collect_dir ${eval_dir}
    echo "  [CHECK] eval:"; ls ${eval_dir}/*.csv 2>/dev/null
done

# =========================== 5. [Trade-off] ===========================
trd_path="${output_path}"
trd_dir="${trd_path}/it=${optim_iters}/batch=${num_images_per_prompt}/seed=${seed}/trd/${num_eval}"
mkdir -p ${trd_dir}/plot ${trd_dir}/csv

echo ""
echo "==== [trade-off] collect_trd + plot_trd → ${trd_dir}/ ===="
python collect_trd.py --method wen --path "${trd_path}" \
    --lr_list ${target_loss_list[@]} --seed ${seed} --batch ${num_images_per_prompt} --oi ${optim_iters} --eval_sub ${num_eval} \
    --trd_dir "${trd_dir}"
python plot_trd.py --csv "${trd_dir}/csv/total_metrics.csv" --out_dir "${trd_dir}/plot"

echo ""
echo "[Done] → ${trd_dir}/{csv,plot}/"

# ---- [Elapsed] 전체 config 총 소요 시간 ----
echo ""
echo "[Elapsed] $((SECONDS/3600))hrs $(( (SECONDS%3600)/60 ))min $((SECONDS%60))sec (total ${SECONDS}sec)"
