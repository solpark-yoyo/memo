#!/bin/bash
# ===================================================================
#  Jeon et al. (ICML 2025) SAIL sharpness 완화 — Chen eval (trade-off)
#  repo: baselines/jeon_sail/ (Dongjae0324/sharpness_memorization_diffusion)
#  논문: arXiv 2412.04140 — sharpness 검출(thres) 후 latent 를 GD 로
#        최적화해 gaussianity 회복 (Algorithm 2)
#
#  ★ knob = miti_lr (latent optimization 학습률 — 완화 강도)
#    SOTA 값 0.05 포함 (논문/공식 run_mitigation.sh + mitigate_mem.py 기본값)
#    나머지 고정: miti_thres=8.2 (l_thres, Alg.2) · budget=8 · max_steps=10
#
#  환경: conda env "${CONDA_ENV:-div_DM}"
#    - local_model.pipe import div_DM 검증 완료 (bf16)
#    - 최소 패치 #2: mitigate_mem.py 모델 id 하드코딩
#      'CompVis/stable-diffusion-v1-4' → 로컬 경로로 1회 자동 치환 (아래 2번)
#
#  실행: ori_memo/ 에서  bash shells/eval_chen/baselines/run_memo_chen_jeon.sh
# ===================================================================

# =========================== 0. [Env] ===========================
export PYTHONUNBUFFERED=1        # 진행 로그 즉시 flush (thres 에스컬레이션 관찰용)
SECONDS=0    # 스크립트 전체 총 소요 시간 측정 (종료 시 [Elapsed] 출력)

# =========================== 1. [Config] ===========================
device="${DEVICE:-cuda:0}"            # argparse --device 로 직접 전달 (예: DEVICE=cuda:3)
eval_gpu="${device##*:}"              # "cuda:0" → "0" (compute_sscd_gt --gpu는 int)
seed=42
num_samples=100
batch=4
num_images_per_prompt=${batch}

# ★ eval subset — 평가할 inference 이미지 수 NUM_EVAL(요청) → clamp 후 실제 평가 장수가
#   폴더명: eval/<num_eval>, trd/<num_eval> (예: 요청 100 > 생성 10장 → eval/10, trd/10)
num_eval="${NUM_EVAL:-100}"
num_eval_prompts=$((num_eval / num_images_per_prompt))
(( num_eval_prompts > num_samples )) && num_eval_prompts=${num_samples}
num_eval=$((num_eval_prompts * num_images_per_prompt))   # 실제 평가 이미지 수 = 폴더명
model_local="ckpt/stable-diffusion-v1-4"

# Chen prompts — jeon 은 plain txt 라인 형식 (그대로 사용)
t2i_prompt_dir="examples/assets/sdv1_500_mem.txt"
gt_ref_dir="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# ★ SAIL params — 논문 고정 (Algorithm 2)
miti_thres=8.2
miti_budget=8
miti_max_steps=10

# ★ miti_lr sweep (완화 강도 축) — ★0.05 = 논문 SOTA
miti_lr_list=(0.02 0.05 0.1 0.2)

# =========================== 2. [최소 패치·프롬pt 배치] ===========================
# 모델 id 하드코딩 → 로컬 ckpt 경로 치환 (1회, idempotent)
jeon_entry="baselines/jeon_sail/mitigate_mem.py"
if grep -q "CompVis/stable-diffusion-v1-4" ${jeon_entry}; then
    sed -i "s|'CompVis/stable-diffusion-v1-4'|'${model_local}'|g" ${jeon_entry}
    echo "[patch] model_id → ${model_local} (${jeon_entry})"
fi

# 프롬pt: examples/assets 원본을 --num_prompts 상한으로 직접 조회 (repo 사본 미사용)
jeon_prompt="baselines/jeon_sail/prompts/cvpr2025_memo_prompt.txt"
# =========================== 3. [Workdir] ===========================
# 저장 경로는 shell 변수(ckpt·cfg·NFE·knob)로 직접 조립
# (jeon mitigate_mem.py 는 cfg/NFE 인자 없음 → pipe 기본값 변수로 선언)
cfg=7.5
NFE=50
base_dir="workdir/memorization/sd14_base"
model_tag="$(basename ${model_local})"     # ckpt/stable-diffusion-v1-4 → stable-diffusion-v1-4
output_path="${base_dir}/baselines/jeon/${model_tag}/CFG=${cfg}_NFE=${NFE}"


echo "========================================="
echo "  Jeon SAIL sharpness mitigation — Chen trade-off (env: ${CONDA_ENV})"
echo "  model=${model_local}  seed=${seed}  gen_num(batch)=${num_images_per_prompt}"
echo "  thres=${miti_thres}(fixed)  budget=${miti_budget}  max_steps=${miti_max_steps}"
echo "  ★ miti_lr sweep=(${miti_lr_list[*]})"
echo "  prompt=${t2i_prompt_dir}  num_samples=${num_samples}"
echo "  num_eval=${num_eval} imgs (eval prompts=${num_eval_prompts}) → eval/${num_eval}, trd/${num_eval}"
echo "========================================="

# =========================== 4. [miti_lr sweep] ===========================
for mlr in "${miti_lr_list[@]}"; do
    gen_dir="${output_path}/thres=${miti_thres}/ms=${miti_max_steps}/batch=${num_images_per_prompt}/seed=${seed}/lr=${mlr}"
    eval_dir="${gen_dir}/eval/${num_eval}"

    echo ""
    echo "---- miti_lr=${mlr} ----"

    # 4-1. Inference (sharpness 검출 + latent 최적화 + 생성)
    echo "================== [INFO]: Jeon Inference (miti_lr=${mlr}) =================="
    python ${jeon_entry} \
        --device ${device} \
        --sd_ver 1 \
        --ckpt_path "${model_local}" \
        --data_path "${t2i_prompt_dir}" \
        --num_prompts ${num_samples} \
        --gen_num ${num_images_per_prompt} \
        --output_dir "${gen_dir}/result" \
        --gen_seed ${seed} \
        --prompt_type mem \
        --miti_thres ${miti_thres} \
        --miti_lr ${mlr} \
        --miti_budget ${miti_budget} \
        --miti_max_steps ${miti_max_steps}

    mkdir -p "${gen_dir}/result"   # 러너가 --output_dir로 여기에 직접 저장

    # 4-2. Eval (Chen: SSCD + T2I)
    echo "================== [INFO]: Eval → ${eval_dir}/ =================="
    mkdir -p ${eval_dir}

    python compute_sscd_gt.py \
        --gen_dir ${gen_dir}/result --ref_dir ${gt_ref_dir} \
        --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu ${eval_gpu} \
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
trd_dir="${output_path}/thres=${miti_thres}/ms=${miti_max_steps}/batch=${num_images_per_prompt}/seed=${seed}/trd/${num_eval}"
mkdir -p ${trd_dir}/plot ${trd_dir}/csv

echo ""
echo "==== [trade-off] collect_trd + plot_trd → ${trd_dir}/ ===="
python collect_trd.py --method jeon --path "${trd_path}" \
    --lr_list ${miti_lr_list[@]} --seed ${seed} --batch ${num_images_per_prompt} --tl ${miti_thres} --oi ${miti_max_steps} --eval_sub ${num_eval} \
    --trd_dir "${trd_dir}"
python plot_trd.py --csv "${trd_dir}/csv/total_metrics.csv" --out_dir "${trd_dir}/plot"

echo ""
echo "[Done] → ${trd_dir}/{csv,plot}/"

# ---- [Elapsed] 전체 config 총 소요 시간 ----
echo ""
echo "[Elapsed] $((SECONDS/3600))hrs $(( (SECONDS%3600)/60 ))min $((SECONDS%60))sec (total ${SECONDS}sec)"
