#!/bin/bash
# =============================================================================
#  Manifold Fixation (seed) — 고정 prompt 1개 × seed 개수 sweep
#  각 seed 개수(count)마다 base_seed부터 count개의 seed로 rollout →
#  step별 ‖z_t‖(latent magnitude)의 std를 seed 축으로 계산 → seed 수별 std table/curve
#  (general vs memorized 프롬프트의 seed 간 trajectory 분산(collapse 여부) 비교)
#
#  근거: general 프롬프트는 seed별 x_t가 collapse, memorized는 심하게 벌어짐
#  (memo/ori_memo/manifold.py --fixation seed, 실험 2-1)
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/main_exp/manifold/run_fixation_seed.sh
# =============================================================================

# ---- Config (전부 env var로 override 가능 — 다른 docker 컨테이너/경로에서도 그대로 재사용) ----
gpu="${GPU:-0}"                  # 예: GPU=3 bash ... 로 cuda:3 지정
model="${MODEL:-stable-diffusion-v1-4}"
NFE="${NFE:-50}"
cfg="${CFG:-7.5}"
SEED="${SEED:-42}"
DEVICE="cuda:${gpu}"

general_prompt="${GENERAL_PROMPT:-examples/assets/geneval_prompts.txt}"
memo_prompt="${MEMO_PROMPT:-examples/assets/cvpr2025_memo_prompt.txt}"
fix_general_idx="${FIX_GENERAL_IDX:-0}"
fix_memo_idx="${FIX_MEMO_IDX:-0}"
seed_list="${SEED_LIST:-5 10 50 100}"       # 고정 prompt에 뽑을 seed 개수 목록

model_key="${MODEL_KEY:-ckpt/${model}}"   # ckpt/ 레이아웃이 다르면 MODEL_KEY로 전체 경로 지정
out_root="${OUT_ROOT:-workdir/exp_main/manifold}"

# ---- Paths ----
# ori_memo/ 에서 실행 가정; 다른 위치(예: docker)면 ROOT_DIR 로 override.
cd "${ROOT_DIR:-.}"

echo "========================================="
echo "  Manifold Fixation (seed) — ||z_t|| std sweep"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  base_seed=${SEED}"
echo "  general_prompt=${general_prompt} (idx=${fix_general_idx})"
echo "  memo_prompt=${memo_prompt} (idx=${fix_memo_idx})"
echo "  seed_list=(${seed_list})"
echo "========================================="

python manifold.py \
    --fixation seed \
    --model_key ${model_key} \
    --device ${DEVICE} \
    --cfg ${cfg} --NFE ${NFE} --seed ${SEED} \
    --general_prompt ${general_prompt} \
    --memo_prompt ${memo_prompt} \
    --fix_general_idx ${fix_general_idx} \
    --fix_memo_idx ${fix_memo_idx} \
    --seed_list "${seed_list}" \
    --out_root ${out_root}

echo ""
echo "Done. -> ${out_root}/${model}/fixation_seed/gi=${fix_general_idx}_mi=${fix_memo_idx}/{csv,plot}/"
