#!/bin/bash
# ===================================================================
#  init_score_noise (Han et al. NeurIPS 2025) — memorization baseline
#  Wen et al. (ICLR 2024) eval 기준
#
#  Metrics: SSCD-to-GT + T2I → {gen_dir}/eval/
#  Compute: elapsed_time, per_sample_time, peak_vram → {gen_dir}/comp/
#
#  ※ 수동으로 init_score_noise env 활성화 후 실행:
#    conda activate init_score_noise
#    bash shells/run_memo_score.sh
#  (eval은 ori_memo env에서 별도 수행)
# ===================================================================

# =========================== 1. [Config] ===========================
gpu=0
NFE=50
cfg_initnoise=7.5
seed=42
num_samples=10
batch=5
num_images_per_prompt=${batch}
target_loss=0.9
optim_iters=1000
lr=0.01
model_id="ckpt/sd14_memor_LAION2B_40k"
text_name="wen2024_memorized_prompts.txt"

# Eval config (Wen et al. 기준)
t2i_prompt_dir="examples/assets/${text_name}"
gt_ref_dir="datasets/wen2024_memorized"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

# =========================== 2. [Workdir] ===========================
base_dir="workdir/memorization/sd14_memor_LAION2B_40k"
output_path="${base_dir}/init_score_noise/NFE=${NFE}"
gen_dir="${output_path}/per_sample/CFG=${cfg_initnoise}/lr=${lr}/tl=${target_loss}/oi=${optim_iters}/seed=${seed}"
eval_dir="${gen_dir}/eval"
comp_dir="${gen_dir}/comp"
total_imgs=$((num_samples * num_images_per_prompt))

echo "================== [INFO]: Model: ${model_id} =================="
echo "  gen_dir: ${gen_dir}"
echo "  total_images: ${total_imgs} (${num_samples} prompts × ${num_images_per_prompt} images)"

# # =========================== 3. [Inference + Bench] (init_score_noise env) ===========================
# echo "================== [INFO]: init_score_noise Inference =================="
# echo "  [CKPT] ${model_id}"
# mkdir -p ${comp_dir}
# python bench_inference.py \
#     --method init_score_noise --num_samples ${total_imgs} \
#     --output_csv ${comp_dir}/comp_metrics.csv --gpu ${gpu} -- \
#     python baselines/init_score_noise/generate_init_score_noise.py \
#     --method adj_init_noise --per_sample \
#     --target_loss ${target_loss} --lr ${lr} --optim_iters ${optim_iters} \
#     --guidance_scale ${cfg_initnoise} --seed ${seed} --num_prompts ${num_samples} \
#     --n_samples_per_prompt ${num_images_per_prompt} --batch_size 1 \
#     --num_inference_steps ${NFE} --model_id ${model_id} --gpu ${gpu} \
#     --output_path "${output_path}"

# echo "[Done] images at ${gen_dir}/"

# =========================== 4. [Eval] (ori_memo env) ===========================
# ※ inference는 init_score_noise env, eval은 ori_memo env에서 실행
echo "================== [INFO]: Eval [init_score_noise] → ${eval_dir}/ =================="
echo "  [CKPT] SSCD: ckpt/sscd_disc_mixup.torchscript.pt"
echo "  [CKPT] CLIP: ckpt/clip-vit-base-patch16"
echo "  [CKPT] PickScore: ckpt/PickScore_v1"
echo "  [CKPT] ImageReward: ckpt/ImageReward/ImageReward.pt"
echo "  [GT_REF] ${gt_ref_dir}"
mkdir -p ${eval_dir}

echo "  --- SSCD-to-GT ---"
python compute_sscd_gt.py \
    --gen_dir ${gen_dir} --ref_dir ${gt_ref_dir} \
    --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
    --gpu ${gpu} \
    --output_csv ${eval_dir}/sscd_gt_metrics.csv

echo "  --- T2I (CLIP / PickScore / ImageReward) ---"
python -m compute_t2i_metrics \
    --eval_dir ${gen_dir} --prompt_dir ${t2i_prompt_dir} \
    --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
    --output_csv ${eval_dir}/t2i_metrics.csv \
    --device cuda:${gpu} ${CS_FLAG}

echo "  --- Merge ---"
python merge_benchmark.py --collect_dir ${eval_dir}

# 결과 출력
echo ""
echo "================== [RESULT] init_score_noise =================="
echo "--- total_metrics.csv ---"
/bin/cat ${eval_dir}/total_metrics.csv 2>/dev/null
echo ""
echo "--- comp_metrics.csv ---"
/bin/cat ${comp_dir}/comp_metrics.csv 2>/dev/null
echo ""

# 파일 검증
echo "[CHECK] eval files:"
/bin/ls ${eval_dir}/*.csv 2>/dev/null
echo "[CHECK] comp files:"
/bin/ls ${comp_dir}/*.csv 2>/dev/null

echo ""
echo "[Done] metrics at ${eval_dir}/"
