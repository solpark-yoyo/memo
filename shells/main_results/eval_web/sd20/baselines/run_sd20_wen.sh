#!/bin/bash
# ===================================================================
#  [SD2.0] Wen et al. (ICLR 2024) prompt augmentation — trade-off
#  model: ckpt/stable-diffusion-2-base · prompt: sd2_mem219 head-10
#  GT: datasets/memo/eval_sscd/sd20/sd2_mem_gt · knob = target_loss λ (★3.0 = 논문 SOTA)
#
#  실행: ori_memo/ 에서  bash shells/eval_sd2/run_sd20_wen.sh
# ===================================================================

# =========================== 0. [Env] ===========================
export PYTHONUNBUFFERED=1

# =========================== 1. [Config] ===========================
device="${DEVICE:-cuda:0}"
eval_gpu="${device##*:}"
NFE=50
cfg_guidance=7.5
seed=42
num_samples=10
batch=1
num_images_per_prompt=${batch}
model_id="ckpt/stable-diffusion-2-base"
text_name="sd2_mem219.txt"
t2i_prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/memo/eval_sscd/sd20/sd2_mem_gt"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# Wen mitigation params — 논문 고정
optim_lr=0.05
optim_iters=10
optim_target_steps=0

target_loss_list=(0.7 1.3 1.9 2.4 3.0)

WEN_ENTRY="baselines/wen_prompt_aug/wen_inference.py"

# =========================== 2. [Prompt jsonl] ===========================
wen_jsonl="examples/assets/sd2_mem219_wen.jsonl"
if [[ ! -f "${wen_jsonl}" ]] || [[ $(wc -l < "${wen_jsonl}") -lt ${num_samples} ]]; then
    python - <<EOF
import json
lines = [l.strip() for l in open("${t2i_prompt_dir}") if l.strip()][:${num_samples}]
with open("${wen_jsonl}", "w") as f:
    for c in lines:
        f.write(json.dumps({"caption": c}) + "\n")
print(f"[prompt] {len(lines)} captions → ${wen_jsonl}")
EOF
fi

# =========================== 3. [Workdir] ===========================
base_dir="workdir/memorization/sd20_base"
model_tag="$(basename ${model_id})"
output_path="${base_dir}/baselines/wen/${model_tag}/CFG=${cfg_guidance}_NFE=${NFE}"


echo "========================================="
echo "  [SD2.0] Wen prompt-aug — trade-off (env: ${CONDA_ENV})"
echo "  model=${model_id}  NFE=${NFE}  CFG=${cfg_guidance}  seed=${seed}"
echo "  ★ λ=(${target_loss_list[*]})  batch=${num_images_per_prompt}  GT=${gt_ref_dir}"
echo "========================================="

# =========================== 4. [λ sweep] ===========================
for tl in "${target_loss_list[@]}"; do
    gen_dir="${output_path}/tl=${tl}/it=${optim_iters}/seed=${seed}"
    eval_dir="${gen_dir}/eval"
    echo ""
    echo "---- target_loss(λ)=${tl} ----"

    # 4-1. Inference (prompt embedding 최적화 + 생성)
    echo "================== [INFO]: Wen Inference SD2 (tl=${tl}) =================="
    python ${WEN_ENTRY} \
        --device ${device} \
        --run_name "sd20_tl${tl}" \
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
python collect_trd.py --method wen --path "${output_path}" \
    --lr_list ${target_loss_list[@]} --seed ${seed} --batch ${num_images_per_prompt} \
    --oi ${optim_iters}
python plot_trd.py --csv "${output_path}/trd/csv/total_metrics.csv" --out_dir "${trd_dir}/plot"

echo ""
echo "[Done] → ${trd_dir}/{csv,plot}/"
