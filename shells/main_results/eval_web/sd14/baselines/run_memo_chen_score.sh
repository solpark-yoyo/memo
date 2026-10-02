#!/bin/bash
# ===================================================================
#  init_score_noise (Han et al. NeurIPS 2025) — Chen eval
#  논문 정합 config: lr=0.01 고정, target_loss sweep, optim_iters=1000
#
#  per_sample mitigation:
#    x_T 를 최적화하여 ||ε_text - ε_uncond||₂ ≤ target_loss 달성
#    target_loss↓ → conditional guidance 약화 → memo 완화 강함
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/eval_chen/run_memo_chen_score.sh
# ===================================================================

# =========================== 0. [Timer] ===========================
SECONDS=0    # 스크립트 전체 총 소요 시간 측정 (종료 시 [Elapsed] 출력)

# =========================== 1. [Config] ===========================
gpu=0
NFE=50
cfg_initnoise=7.0          # 논문 기준 (SD v1.4)
seed=42
num_samples=5
batch=4
num_images_per_prompt=${batch}

# ★ eval subset — 평가할 inference 이미지 수 NUM_EVAL(요청) → clamp 후 실제 평가 장수가
#   폴더명: eval/<num_eval>, trd/<num_eval> (예: 요청 100 > 생성 10장 → eval/10, trd/10)
num_eval="${NUM_EVAL:-${num_samples}}"
num_eval_prompts=$((num_eval / num_images_per_prompt))
(( num_eval_prompts > num_samples )) && num_eval_prompts=${num_samples}
num_eval=$((num_eval_prompts * num_images_per_prompt))   # 실제 평가 이미지 수 = 폴더명
# init_score_noise (Han) params — 논문 정합
lr=0.01                   # ★ 고정 (논문: lr=0.01 for SD v1.4)
optim_iters=1000          # ★ 수렴까지 (논문: target 도달 시 자동 break)

# ★ target_loss sweep (논문의 control knob)
target_loss_list=(1.1 1.7 2.1 2.5)     # 완화 약화 구간 확장 (기준선 방향)


# Chen setup (일반 SD1.4 + Webster)
model_id="ckpt/stable-diffusion-v1-4"
text_name="sdv1_500_mem.txt"
t2i_prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# =========================== 2. [Workdir] ===========================
base_dir="workdir/memorization/sd14_base"
output_path="${base_dir}/baselines/init_score_noise/NFE=${NFE}"


echo "========================================="
echo "  init_score_noise (Han) — Chen eval"
echo "  model=${model_id}  NFE=${NFE}  CFG=${cfg_initnoise}  seed=${seed}"
echo "  lr=${lr} (fixed)  optim_iters=${optim_iters}"
echo "  target_loss=(${target_loss_list[*]})"
echo "  prompt=${text_name}  num_samples=${num_samples}  batch=${num_images_per_prompt}"
echo "  num_eval=${num_eval} imgs (eval prompts=${num_eval_prompts}) → eval/${num_eval}, trd/${num_eval}"
echo "========================================="

# =========================== 3. [target_loss sweep] ===========================
for tl in "${target_loss_list[@]}"; do
    gen_dir="${output_path}/per_sample/CFG=${cfg_initnoise}/lr=${lr}/oi=${optim_iters}/batch=${num_images_per_prompt}/seed=${seed}/tl=${tl}"
    eval_dir="${gen_dir}/eval/${num_eval}"
    echo ""
    echo "---- target_loss=${tl} ----"
    echo "  gen_dir: ${gen_dir}"

    # =========================== 3-1. [Inference] ===========================
    echo "================== [INFO]: init_score_noise Inference (tl=${tl}) =================="
    python baselines/init_score_noise/generate_init_score_noise.py \
        --method adj_init_noise --per_sample \
        --target_loss ${tl} --lr ${lr} --optim_iters ${optim_iters} \
        --guidance_scale ${cfg_initnoise} --seed ${seed} --num_prompts ${num_samples} \
        --n_samples_per_prompt ${num_images_per_prompt} --batch_size 1 \
        --num_inference_steps ${NFE} --model_id ${model_id} --gpu ${gpu} \
        --output_path "${output_path}" \
        --prompt_csv ${t2i_prompt_dir}

    # =========================== 3-2. [Eval] (Chen: SSCD + T2I) ===========================
    echo "================== [INFO]: Eval → ${eval_dir}/ =================="
    mkdir -p ${eval_dir}

    python compute_sscd_gt.py \
        --gen_dir ${gen_dir} --ref_dir ${gt_ref_dir} \
        --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu ${gpu} \
        --output_csv ${eval_dir}/chen_sscd_gt_metrics.csv

    python -m compute_t2i_metrics \
        --eval_dir ${gen_dir} --prompt_dir ${t2i_prompt_dir} \
        --num_prompts ${num_eval_prompts} --num_images_per_prompt ${num_images_per_prompt} \
        --output_csv ${eval_dir}/chen_t2i_metrics.csv \
        --device cuda:${gpu} ${CS_FLAG}

    python merge_benchmark.py --collect_dir ${eval_dir}
    echo "  [CHECK] eval:"; ls ${eval_dir}/*.csv 2>/dev/null
done

# =========================== 4. [Trade-off] ===========================
# target_loss sweep 결과를 trd 로 수집
trd_path="${output_path}/per_sample/CFG=${cfg_initnoise}"          # pattern 탐색 기준 (lr=/oi=/batch/seed/tl= 자동 조립)
trd_dir="${output_path}/per_sample/CFG=${cfg_initnoise}/lr=${lr}/oi=${optim_iters}/batch=${num_images_per_prompt}/seed=${seed}/trd/${num_eval}"  # 출력은 tl= 폴더 옆 (knob=tl)
mkdir -p ${trd_dir}/plot ${trd_dir}/csv
echo ""
echo "==== [trade-off] collect_trd + plot_trd → ${trd_dir}/ ===="
python collect_trd.py --method init_score_noise --path "${trd_path}" \
    --lr ${lr} --oi ${optim_iters} --lr_list ${target_loss_list[@]} --seed ${seed} --batch ${num_images_per_prompt} --eval_sub ${num_eval} \
    --trd_dir "${trd_dir}"
python plot_trd.py --csv "${trd_dir}/csv/total_metrics.csv" --out_dir "${trd_dir}/plot"

echo ""
echo "[Done] → ${trd_dir}/{csv,plot}/"

# ---- [Elapsed] 전체 config 총 소요 시간 ----
echo ""
echo "[Elapsed] $((SECONDS/3600))hrs $(( (SECONDS%3600)/60 ))min $((SECONDS%60))sec (total ${SECONDS}sec)"
