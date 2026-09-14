#!/bin/bash
# =============================================================================
#  Proxy Trend:  x_t → ε_θ(x_t,t) → spectrum energy      ← 기본 (cfg_epsilon)
#                text(general) vs memorized 그룹을 denoising step 에 따라 비교
#
#  여기서 epsilon = cfg_epsilon = ε_uc + w·(ε_c − ε_uc)  (w = CFG)
#                  → DDIM update 에 실제로 쓰이는 CFG 결합 noise prediction
#
#  갈래 선택 (SIGNAL, argparse --signal 로 전달):
#    eps = x_t → cfg_epsilon → spectrum energy     (기본)
#    xt  = x_t → spectrum energy                   (latent baseline)
#
#  y-axis proxy 선택 (PROXY, argparse --proxy):
#    all = 전체 block 평균 | low = 저주파 밴드 평균 | high = 고주파 밴드 평균
#    (밴드는 FREQ_RATIO 로 산출: n_band=int(P*ratio), low=[0,n_band-1], high=[P-n_band,P-1])
#
#  scale: optimize_xt_spectral.py 의 spectral_l2_loss 와 동일
#         (mean(block energy)/B, 1 = white Gaussian)
#
#  저장 (OUTPUT_DIR 아래):
#    plot/proxy_trend_{signal}_{proxy}.png   그룹 mean ± std 비교
#    csv/proxy_trend_{signal}_{proxy}.csv    step,timestep,text_*,memo_*
#    npz/{text,memo}{i}_{signal}.npz         per-prompt 원본 (T,P)
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/others/run_eps_spectrum_energy.sh            # 기본: cfg_epsilon 갈래
#    SIGNAL=xt bash shells/others/run_eps_spectrum_energy.sh  # x_t baseline 갈래
#
#  오버라이드 예:
#    PROXY=low bash shells/others/run_eps_spectrum_energy.sh       # 저주파 밴드 y축
#    NUM_TP=5 NUM_MTP=5 bash shells/others/run_eps_spectrum_energy.sh
# =============================================================================
# set -euo pipefail

# ---- Config (환경변수 오버라이드 가능) ----
gpu="${GPU:-0}"
NFE="${NFE:-50}"
cfg="${CFG:-7.5}"
SEED="${SEED:-42}"
batch="${BATCH:-5}"
DEVICE="cuda:${gpu}"

SIGNAL="${SIGNAL:-eps}"                 # eps(기본)=cfg_epsilon | xt=x_t (spectrum energy 를 잴 신호)
PROXY="${PROXY:-all}"                   # all | low | high  (y-axis)
FREQ_RATIO="${FREQ_RATIO:-0.1}"         # low/high 밴드 비율
BLOCK_SIZE="${BLOCK_SIZE:-16}"          # 주파수 block 하나의 bin 수 B

num_samples=${batch}                    # seed 수 (prompt 당 평균 → mean ± std)
num_tp="${NUM_TP:-3}"                   # text(general) prompt 수 (plot 1개당)
num_mtp="${NUM_MTP:-1}"                 # memorized prompt 수 (plot 1개당)
num_plot="${NUM_PLOT:-5}"               # plot00, plot01 ... 비교 plot 개수

MODEL="${MODEL:-ckpt/stable-diffusion-v1-4}"                       # SD1.4
text_dir="${TEXT_DIR:-examples/assets/coco_v2.txt}"                # general prompt
memo_dir="${MEMO_DIR:-examples/assets/cvpr2025_memo_prompt.txt}"   # memorized prompt

# 저장 경로: signal 갈래별로 분리
base_dir="${EXP_ROOT:-workdir/exp_main/eps_trajectory}/results_proxy_trend"
OUTPUT_DIR="${base_dir}/signal=${SIGNAL}/ddim/CFG=${cfg}_NFE=${NFE}/seed=${SEED}/batch=${batch}"

echo "========================================="
echo "  Proxy Trend  (text vs memorized, DDIM, SD1.4)"
echo "  signal=${SIGNAL}  proxy=${PROXY}  freq_ratio=${FREQ_RATIO}"
echo "  NFE=${NFE}  CFG=${cfg}  SEED=${SEED}  batch=${batch}  block_size=${BLOCK_SIZE}"
echo "  text=${num_tp}개(${text_dir##*/})  memo=${num_mtp}개(${memo_dir##*/})  num_plot=${num_plot}"
echo "  model=${MODEL}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "========================================="

cd "${ROOT_DIR:-.}"   # ori_memo/ 에서 실행 가정; 다른 위치면 ROOT_DIR override

python eps_trajectory.py \
    --proxy_trend \
    --model_key ${MODEL} \
    --signal ${SIGNAL} \
    --proxy ${PROXY} \
    --freq_ratio ${FREQ_RATIO} \
    --block_size ${BLOCK_SIZE} \
    --num_inference_steps ${NFE} \
    --cfg_guidance ${cfg} \
    --seed ${SEED} \
    --device ${DEVICE} \
    --num_samples ${num_samples} \
    --output_dir ${OUTPUT_DIR} \
    --text_dir ${text_dir} \
    --memo_dir ${memo_dir} \
    --num_tp ${num_tp} \
    --num_mtp ${num_mtp} \
    --num_plot ${num_plot}

echo "Done. -> ${OUTPUT_DIR}"
