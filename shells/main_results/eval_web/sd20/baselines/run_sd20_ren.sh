#!/bin/bash
# ===================================================================
#  [SD2.0] Ren et al. (ECCV 2024) MemAttn — memorization trade-off
#  model: ckpt/stable-diffusion-2-base · prompt: sd2_mem219 (Webster SD2)
#  GT: datasets/memo/eval_sscd/sd20/sd2_mem_gt (jsonl URL 추출 195/219, 2026-08-30)
#  knob = c1 (attention mask 계수), ★1.25 = 논문 SOTA
#
#  실행: ori_memo/ 에서  bash shells/eval_sd2/run_sd20_ren.sh
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
model_name="ckpt/stable-diffusion-2-base"

text_name="sd2_mem219"                       # SD2 전용 Webster memorized 219
t2i_prompt_dir="examples/assets/${text_name}.txt"
gt_ref_dir="datasets/memo/eval_sscd/sd20/sd2_mem_gt"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

c1_list=(1.0 1.25 1.5 2.0 2.5)

# =========================== 2. [Prompt 배치] ===========================
ren_repo="baselines/ren_memattn"
# =========================== 3. [Workdir] ===========================
cfg=7.5
NFE=50
base_dir="workdir/memorization/sd20_base"
model_tag="$(basename ${model_name})"
output_path="${base_dir}/baselines/ren/${model_tag}/CFG=${cfg}_NFE=${NFE}"


echo "========================================="
echo "  [SD2.0] Ren MemAttn — trade-off (env: ${CONDA_ENV})"
echo "  model=${model_name}  seed=${seed}  batch=${num_images_per_prompt}"
echo "  ★ c1 sweep=(${c1_list[*]})  prompt=${text_name}.txt[0:10]"
echo "  GT=${gt_ref_dir}"
echo "========================================="

# =========================== 4. [c1 sweep] ===========================
for c1 in "${c1_list[@]}"; do
    job_id="sd20_c1_${c1}"
    gen_dir="${output_path}/c1=${c1}/seed=${seed}"
    eval_dir="${gen_dir}/eval"

    echo ""
    echo "---- c1=${c1} ----"

    # 4-1. Inference (refactored UNet 주입 러너 — SD2 unet도 동일 클래스로 로드)
    echo "================== [INFO]: Ren Inference SD2 (c1=${c1}) =================="
    # 러너가 --save_ours로 result/에 img_XXXX_YY.png 직접 저장
    python ${ren_repo}/ren_inference.py \
        --model_name "${model_name}" \
        --prompt "examples/assets/${text_name}" \
        --num_prompts ${num_samples} \
        --job_id "${job_id}" \
        --output_name "miti" \
        --seed ${seed} \
        --num_images_per_prompt ${num_images_per_prompt} \
        --device ${device} \
        --output_dir "${gen_dir}/result" \
        --save_ours \
        --local '' \
        --miti_mem \
        --c1 ${c1}

    mkdir -p "${gen_dir}/result"   # 러너가 --save_ours로 여기에 직접 저장

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
python collect_trd.py --method ren --path "${output_path}" \
    --lr_list ${c1_list[@]} --seed ${seed} --batch ${num_images_per_prompt}
python plot_trd.py --csv "${output_path}/trd/csv/total_metrics.csv" --out_dir "${trd_dir}/plot"

echo ""
echo "[Done] → ${trd_dir}/{csv,plot}/"
