#!/bin/bash
# ===================================================================
#  Memorization proxy vs denoising step plot
#  - memorized (또는 원하는) prompt 들에 대해 고정 x_T 로 DDIM forward 하며
#    각 step 의 normalized Tweedie-gap proxy 를 계산해 plot.
#
#  proxy(s) = || eps_ref - eps_s ||^2 / || eps_ref ||^2
#    eps_ref = x_T (fixed),  eps_s = eps_theta(x_s, s)
#    x_s = sqrt(alpha_s) x0_hat(s) + sqrt(1-alpha_s) x_T
#
#  실행: ori_memo 디렉토리에서  bash shells/plot_memo_proxy.sh
# ===================================================================

# =========================== 1. [Parser] ===========================
gpu=0
num_prompts=5
NFE=50
cfg=7.5
seed=42

# memorized prompt (normal 비교 원하면 coco_v2.txt 등으로 변경)
text_name="memorized_prompts_membench.txt"
prompt_dir="examples/assets/${text_name}"

# output
output="workdir/memo_proxy_vs_step_${text_name%.*}.png"

# =========================== 2. [Run] ===========================
python plot_memo_proxy.py \
    --num_prompts ${num_prompts} \
    --NFE ${NFE} --cfg ${cfg} --seed ${seed} \
    --prompt_dir ${prompt_dir} \
    --device cuda:${gpu} \
    --output ${output}

echo "[Done] plot: ${output}"
