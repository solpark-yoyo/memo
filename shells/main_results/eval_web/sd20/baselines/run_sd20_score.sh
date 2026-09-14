#!/bin/bash
# ===================================================================
#  [SD2.0] init_score_noise (Han et al. NeurIPS 2025) — trade-off
#  model: ckpt/stable-diffusion-2-base · prompt: sd2_mem219 head-10
#  GT: datasets/memo/eval_sscd/sd20/sd2_mem_gt · lr=0.01 고정, target_loss sweep (knob)
#
#  실행: ori_memo/ 에서  bash shells/eval_sd2/run_sd20_score.sh
# ===================================================================

# =========================== 0. [Env] ===========================
export PYTHONUNBUFFERED=1

# =========================== 1. [Config] ===========================
device="${DEVICE:-cuda:0}"
eval_gpu="${device##*:}"
gpu="${eval_gpu}"
NFE=50
cfg_initnoise=7.0          # sd14 프로토콜 승계 (Han 논문 기준)
seed=42
num_samples=10
batch=1
num_images_per_prompt=${batch}
model_id="ckpt/stable-diffusion-2-base"
t2i_prompt_dir="examples/assets/sd2_mem219.txt"
gt_ref_dir="datasets/memo/eval_sscd/sd20/sd2_mem_gt"
cs_only=false
CS_FLAG=""; [[ "${cs_only}" == "true" ]] && CS_FLAG="--cs_only"

lr=0.01                    # ★ 고정 (논문: lr=0.01)
optim_iters=1000           # target 도달 시 자동 break

target_loss_list=(0.7 0.9 1.1 1.3 1.5)

# =========================== 2. [Workdir] ===========================
base_dir="workdir/memorization/sd20_base"
output_path="${base_dir}/baselines/init_score_noise/NFE=${NFE}"


echo "========================================="
echo "  [SD2.0] init_score_noise (Han) — trade-off"
echo "  model=${model_id}  NFE=${NFE}  CFG=${cfg_initnoise}  seed=${seed}"
echo "  lr=${lr}  optim_iters=${optim_iters}"
echo "  target_loss=(${target_loss_list[*]})  GT=${gt_ref_dir}"
echo "========================================="

# =========================== 3. [target_loss sweep] ===========================
for tl in "${target_loss_list[@]}"; do
    gen_dir="${output_path}/per_sample/CFG=${cfg_initnoise}/lr=${lr}/tl=${tl}/oi=${optim_iters}/seed=${seed}"
    eval_dir="${gen_dir}/eval"
    echo ""
    echo "---- target_loss=${tl} ----"
    echo "  gen_dir: ${gen_dir}"

    # 3-1. Inference — 러너가 gen_dir 아래 img_XXXX_YY.png 저장
    echo "================== [INFO]: init_score_noise Inference SD2 (tl=${tl}) =================="
    python baselines/init_score_noise/generate_init_score_noise.py \
        --method adj_init_noise --per_sample \
        --target_loss ${tl} --lr ${lr} --optim_iters ${optim_iters} \
        --guidance_scale ${cfg_initnoise} --seed ${seed} --num_prompts ${num_samples} \
        --n_samples_per_prompt ${num_images_per_prompt} --batch_size 1 \
        --num_inference_steps ${NFE} --model_id ${model_id} --gpu ${gpu} \
        --output_path "${output_path}" \
        --prompt_csv ${t2i_prompt_dir}

    # 3-2. Eval (SSCD-to-GT + T2I)
    echo "================== [INFO]: Eval → ${eval_dir}/ =================="
    mkdir -p ${eval_dir}

    python compute_sscd_gt.py \
        --gen_dir ${gen_dir} --ref_dir ${gt_ref_dir} \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --gpu "${eval_gpu}" \
        --output_csv ${eval_dir}/chen_sscd_gt_metrics.csv

    python -m compute_t2i_metrics \
        --eval_dir ${gen_dir} --prompt_dir ${t2i_prompt_dir} \
        --num_prompts ${num_samples} --num_images_per_prompt ${num_images_per_prompt} \
        --output_csv ${eval_dir}/chen_t2i_metrics.csv \
        --device ${device} ${CS_FLAG}

    python merge_benchmark.py --collect_dir ${eval_dir}
    echo "  [CHECK] eval:"; ls ${eval_dir}/*.csv 2>/dev/null
done

# =========================== 4. [Trade-off] ===========================
# knob이 target_loss(tl)이므로 각 eval/total_metrics.csv를 label=tl로 수집
trd_path="${output_path}/per_sample/CFG=${cfg_initnoise}/lr=${lr}"
trd_dir="${trd_path}/oi=${optim_iters}/trd"
mkdir -p ${trd_dir}/plot ${trd_dir}/csv

echo ""
echo "==== [trade-off] 수집 + plot_trd → ${trd_dir}/ ===="
python - <<EOF
import csv, os
rows = []
for tl in "${target_loss_list[@]}".split():
    p = os.path.join("${trd_path}", f"tl={tl}", "oi=${optim_iters}", "seed=${seed}", "eval", "total_metrics.csv")
    if not os.path.exists(p):
        print(f"[warn] 없음: {p}"); continue
    m = {r[0]: r[1].split(",")[0] for r in (line.strip().split(",", 1) for line in open(p) if "," in line)}
    rows.append([tl, m.get("CLIPScore", ""), m.get("PickScore", ""),
                 m.get("ImageReward", ""), m.get("SSCD-to-GT (mean)", "")])
    print(f"[ok] tl={tl}: {m.get('SSCD-to-GT (mean)')} / CLIP {m.get('CLIPScore (mean)')}")
out = os.path.join("${trd_dir}", "csv", "total_metrics.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["label", "CLIP_mean", "Pick_mean", "ImgR_mean", "SSCD_mean"])
    w.writerows(rows)
print(f"[Done] {len(rows)} rows → {out}")
EOF
python plot_trd.py --csv "${trd_dir}/csv/total_metrics.csv" --out_dir "${trd_dir}/plot"

echo ""
echo "[Done] → ${trd_dir}/{csv,plot}/"
