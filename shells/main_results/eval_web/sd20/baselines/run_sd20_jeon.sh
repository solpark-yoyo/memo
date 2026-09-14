#!/bin/bash
# ===================================================================
#  [SD2.0] Jeon et al. (ICML 2025) SAIL sharpness 완화 — trade-off
#  model: ckpt/stable-diffusion-2-base (--sd_ver 2)
#  prompt: sd2_mem219 head-10 · GT: datasets/memo/eval_sscd/sd20/sd2_mem_gt (jsonl URL 추출)
#  knob = miti_lr (latent optimization 학습률), ★0.05 = 논문 SOTA
#  batch=2 (miti 최적화 activation — 24GB OOM 대응, sd14와 동일)
#
#  실행: ori_memo/ 에서  bash shells/eval_sd2/run_sd20_jeon.sh
# ===================================================================

# =========================== 0. [Env] ===========================
export PYTHONUNBUFFERED=1

# =========================== 1. [Config] ===========================
device="${DEVICE:-cuda:0}"
eval_gpu="${device##*:}"
seed=42
num_samples=10
batch=1
num_images_per_prompt=${batch}
model_local="ckpt/stable-diffusion-2-base"

t2i_prompt_dir="examples/assets/sd2_mem219.txt"
gt_ref_dir="datasets/memo/eval_sscd/sd20/sd2_mem_gt"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# SAIL params — 논문 고정 (Algorithm 2)
miti_thres=1.55   # SD2 보정 (2026-08-30 프로브: 초기 loss 분포 median 1.56 — SD1 논문값 8.2는 스케일 불일치로 최적화 미발동)
miti_budget=8
miti_max_steps=10

miti_lr_list=(0.01 0.02 0.05 0.10 0.20)

# =========================== 2. [최소 패치·프롬pt 배치] ===========================
jeon_entry="baselines/jeon_sail/mitigate_mem.py"
# sd_ver=2 브랜치의 'stabilityai/stable-diffusion-2' → 로컬 ckpt 치환 (1회, idempotent)
if grep -q "'stabilityai/stable-diffusion-2'" ${jeon_entry}; then
    sed -i "s|'stabilityai/stable-diffusion-2'|'${model_local}'|g" ${jeon_entry}
    echo "[patch] sd2 model_id → ${model_local} (${jeon_entry})"
fi

jeon_prompt="baselines/jeon_sail/prompts/sd2_mem219.txt"
# =========================== 3. [Workdir] ===========================
cfg=7.5
NFE=50
base_dir="workdir/memorization/sd20_base"
model_tag="$(basename ${model_local})"
output_path="${base_dir}/baselines/jeon/${model_tag}/CFG=${cfg}_NFE=${NFE}"


echo "========================================="
echo "  [SD2.0] Jeon SAIL — trade-off (env: ${CONDA_ENV})"
echo "  model=${model_local}  seed=${seed}  gen_num=${num_images_per_prompt}"
echo "  thres=${miti_thres}  budget=${miti_budget}  ms=${miti_max_steps}"
echo "  ★ miti_lr=(${miti_lr_list[*]})  GT=${gt_ref_dir}"
echo "========================================="

# =========================== 4. [miti_lr sweep] ===========================
for mlr in "${miti_lr_list[@]}"; do
    gen_dir="${output_path}/lr=${mlr}/thres=${miti_thres}/ms=${miti_max_steps}/seed=${seed}"
    eval_dir="${gen_dir}/eval"

    echo ""
    echo "---- miti_lr=${mlr} ----"

    # 4-1. Inference (sharpness 검출 + latent 최적화 + 생성)
    echo "================== [INFO]: Jeon Inference SD2 (lr=${mlr}) =================="
    python ${jeon_entry} \
        --device ${device} \
        --sd_ver 2 \
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

    # 4-2. Eval (SSCD-to-GT + T2I)
    echo "================== [INFO]: Eval → ${eval_dir}/ =================="
    mkdir -p ${eval_dir}

    python compute_sscd_gt.py \
        --gen_dir ${gen_dir}/result --ref_dir ${gt_ref_dir} \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu "${eval_gpu}" \
        --output_csv ${eval_dir}/chen_sscd_gt_metrics.csv

    python -m compute_t2i_metrics \
        --eval_dir ${gen_dir}/result --prompt_dir ${t2i_prompt_dir} \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --output_csv ${eval_dir}/chen_t2i_metrics.csv \
        --device ${device} ${CS_FLAG}

    python merge_benchmark.py --collect_dir ${eval_dir}
    echo "  [CHECK] eval:"; ls ${eval_dir}/*.csv 2>/dev/null

done

# =========================== 5. [Trade-off] ===========================
trd_dir="${output_path}/trd"
mkdir -p ${trd_dir}/plot ${trd_dir}/csv

echo ""
echo "==== [trade-off] collect_trd + plot_trd → ${trd_dir}/ ===="
python collect_trd.py --method jeon --path "${output_path}" \
    --lr_list ${miti_lr_list[@]} --seed ${seed} --batch ${num_images_per_prompt} \
    --tl ${miti_thres} --oi ${miti_max_steps}
python plot_trd.py --csv "${output_path}/trd/csv/total_metrics.csv" --out_dir "${trd_dir}/plot"

echo ""
echo "[Done] → ${trd_dir}/{csv,plot}/"
