#!/bin/bash
# ===================================================================
#  Ren et al. (ECCV 2024) MemAttn cross-attention 완화 — Chen eval (trade-off)
#  repo: baselines/ren_memattn/ (renjie3/MemAttn)
#  논문: arXiv 2403.11052 — memorized 샘플의 특정 token 과집중 attention 을
#        BOS attention logit 배수(c1)로 완화 (순수 c1 프로토콜 — 2026-09-01:
#        --cross_attn_mask 제거. mask는 c1과 무관한 강한 이산 개입이라 c1=1.0이
#        no-op가 됨. 검증: c1=1.0 ≈ DDIM(CLIP 35.3/SSCD 0.87), c1=1.25 = CLIP 33.5/SSCD 0.38)
#
#  ★ knob = c1 (attention mask 스케일 계수)
#    SOTA 값 1.25 포함 (공식 shell local_cmd_inference_time_mitigation.sh)
#    c1=1.0 ≈ 완화 없음 기준점, 클수록 강함
#
#  환경: conda env "${CONDA_ENV:-div_DM}"
#    - refactored_classes.MemAttn import div_DM(0.36) 검증 완료
#    - text2img.py 내부 cache_dir 하드코딩은 --model_name 에
#      로컬 ckpt 경로를 주면 우회됨 (from_pretrained 이 local path 우선)
#
#  실행: ori_memo/ 에서  bash shells/eval_chen/baselines/run_memo_chen_ren.sh
# ===================================================================

# =========================== 0. [Env] ===========================
SECONDS=0    # 스크립트 전체 총 소요 시간 측정 (종료 시 [Elapsed] 출력)

# =========================== 1. [Config] ===========================
device="${DEVICE:-cuda:0}"            # 평가·shell 공용 장치 (예: DEVICE=cuda:3)
seed=42
num_samples=100
# ren_inference.py — text2img.py에 num_images_per_prompt 지원 추가 (2026-08-30)
# refactored UNet 주입 방식 계승 (stock UNet이면 return_attention kwarg 크래시)
batch=4
num_images_per_prompt=${batch}

# ★ eval subset — 평가할 inference 이미지 수 NUM_EVAL(요청) → clamp 후 실제 평가 장수가
#   폴더명: eval/<num_eval>, trd/<num_eval> (예: 요청 100 > 생성 10장 → eval/10, trd/10)
num_eval="${NUM_EVAL:-${num_samples}}"
num_eval_prompts=$((num_eval / num_images_per_prompt))
(( num_eval_prompts > num_samples )) && num_eval_prompts=${num_samples}
num_eval=$((num_eval_prompts * num_images_per_prompt))   # 실제 평가 이미지 수 = 폴더명
model_name="ckpt/stable-diffusion-v1-4"        # 로컬 ckpt 경로 (cache_dir 우회)

# Chen prompts — ren 은 {args.prompt}.txt 를 repo prompt/ 에서 읽음
text_name="sdv1_500_mem"
t2i_prompt_dir="examples/assets/${text_name}.txt"
gt_ref_dir="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# ★ c1 sweep (완화 강도 축) — ★1.25 = 논문 SOTA, 1.0 = 기준점
c1_list=(1.25 1.5 2.0 2.5)

# =========================== 2. [Prompt 배치] ===========================
# 프롬pt: examples/assets 원본을 --num_prompts 상한으로 직접 조회 (repo 사본 미사용)
ren_repo="baselines/ren_memattn"
# =========================== 3. [Workdir] ===========================
# 저장 경로는 shell 변수(ckpt·cfg·NFE·knob)로 직접 조립
# (ren text2img.py 는 cfg/NFE 인자 없음 → pipe 기본값 변수로 선언)
cfg=7.5
NFE=50
base_dir="workdir/memorization/sd14_base"
model_tag="$(basename ${model_name})"      # ckpt/stable-diffusion-v1-4 → stable-diffusion-v1-4
output_path="${base_dir}/baselines/ren/${model_tag}/CFG=${cfg}_NFE=${NFE}"


echo "========================================="
echo "  Ren MemAttn cross-attn mitigation — Chen trade-off (env: ${CONDA_ENV})"
echo "  model=${model_name}  seed=${seed}  batch/prompt=${num_images_per_prompt}"
echo "  ★ c1 sweep=(${c1_list[*]})"
echo "  prompt=${text_name}.txt  num_samples=${num_samples}"
echo "  num_eval=${num_eval} imgs (eval prompts=${num_eval_prompts}) → eval/${num_eval}, trd/${num_eval}"
echo "========================================="

# =========================== 4. [c1 sweep] ===========================
for c1 in "${c1_list[@]}"; do
    # ren 출력: {repo}/results/{job_id}_{args.prompt}_{output_name}_seed{seed}/
    job_id="chen_c1_${c1}"
    output_name="miti"
    gen_dir="${output_path}/batch=${num_images_per_prompt}/seed=${seed}/c1=${c1}"
    eval_dir="${gen_dir}/eval/${num_eval}"

    echo ""
    echo "---- c1=${c1} ----"

    # 4-1. Inference (repo 러너 ren_inference.py — batch 지원)
    echo "================== [INFO]: Ren Inference (c1=${c1}) =================="
    python ${ren_repo}/ren_inference.py \
        --model_name "${model_name}" \
        --prompt "examples/assets/${text_name}" \
        --num_prompts ${num_samples} \
        --job_id "${job_id}" \
        --output_name "${output_name}" \
        --seed ${seed} \
        --num_images_per_prompt ${num_images_per_prompt} \
        --device ${device} \
        --output_dir "${gen_dir}/result" \
        --save_ours \
        --local '' \
        --miti_mem \
        --c1 ${c1}

    mkdir -p "${gen_dir}/result"   # 러너가 --save_ours로 여기에 직접 저장

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
trd_dir="${trd_path}/batch=${num_images_per_prompt}/seed=${seed}/trd/${num_eval}"
mkdir -p ${trd_dir}/plot ${trd_dir}/csv

echo ""
echo "==== [trade-off] collect_trd + plot_trd → ${trd_dir}/ ===="
python collect_trd.py --method ren --path "${trd_path}" \
    --lr_list ${c1_list[@]} --seed ${seed} --batch ${num_images_per_prompt} --eval_sub ${num_eval} \
    --trd_dir "${trd_dir}"
python plot_trd.py --csv "${trd_dir}/csv/total_metrics.csv" --out_dir "${trd_dir}/plot"

echo ""
echo "[Done] → ${trd_dir}/{csv,plot}/"

# ---- [Elapsed] 전체 config 총 소요 시간 ----
echo ""
echo "[Elapsed] $((SECONDS/3600))hrs $(( (SECONDS%3600)/60 ))min $((SECONDS%60))sec (total ${SECONDS}sec)"
