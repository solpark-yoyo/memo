#!/bin/bash
# =============================================================================
#  Seed Std (per-prompt seed sensitivity, averaged over prompts)
#  [한 text prompt에 따른 latent std 계산 --> prompt 평균]
#
#  소스별 프롬프트 각각에 대해:
#    1) 같은 prompt에 seed(batch)개를 적용해 DDIM rollout
#    2) 임의의 denoising step에서 seed 축 std(‖z_t‖) 계산 (그 prompt의 seed 민감도)
#  → 소스 내 n_prompts개에 대해 위 std를 prompt 축으로 평균 → 소스별 std 곡선
#  → general·memo 전체 소스를 **한 plot에 동시 오버레이**
#  (색 계열: general=blue계 실선 원형 마커, memo=red계 실선 사각 마커)
#
#  소스:
#    general — gen_eval / for_diverse / drawbench / coco_v2 (examples/assets/)
#    memo    — webster / membench (examples/assets/)
#  ⚠ for_diverse.txt는 5줄뿐 — n_prompts를 초과 요청하면 5개로 clamp됨(경고 출력)
#
#  (memo/ori_memo/manifold.py --seedstd, 실험 4 / skill: manifold-ppt)
#
#  환경: conda div_DM (ori_memo/ 에서 실행)
#    bash shells/exp_main/manifold/run_seedstd.sh
# =============================================================================

# ---- Config (전부 env var로 override 가능 — 다른 docker 컨테이너/경로에서도 그대로 재사용) ----
gpu="${GPU:-0}"                  # 예: GPU=3 bash ... 로 cuda:3 지정
model="${MODEL:-stable-diffusion-v1-4}"
NFE="${NFE:-50}"
cfg="${CFG:-7.5}"
SEED="${SEED:-42}"
batch="${BATCH:-5}"              # num seeds per prompt (seed 축 std 계산 대상)
n_prompts="${N_PROMPTS:-25}"    # 소스별 최대 요청 prompt 수 (파일이 작으면 clamp) — prompt 축 평균 대상
DEVICE="cuda:${gpu}"

model_key="${MODEL_KEY:-ckpt/${model}}"   # ckpt/ 레이아웃이 다르면 MODEL_KEY로 전체 경로 지정
out_root="${OUT_ROOT:-workdir/exp_main/manifold}"
tag="${TAG:-general_memo}"

# 캐시 무시하고 강제로 새 rollout: FORCE_ROLLOUT=1 bash ...
force_rollout="${FORCE_ROLLOUT:-1}"
force_flag=""
if [[ "$force_rollout" == "1" ]] || [[ "$force_rollout" == "true" ]]; then
    force_flag="--force_rollout"
fi

# SOURCES — PROFILE 또는 SOURCES 환경변수로 설정
#   PROFILE=all      : 모든 소스 (gen_eval, for_diverse, drawbench, coco_v2, webster, membench)
#   PROFILE=minimal  : 최소 (gen_eval / webster)
#   (기본값)         : standard (gen_eval, drawbench, coco_v2 / webster, membench) — for_diverse 제외
#   SOURCES="..."    : 수동 지정 (PROFILE 우선순위 높음)
profile="${PROFILE:-standard}"

if [[ -n "${SOURCES:-}" ]]; then
    read -ra sources <<< "${SOURCES}"
elif [[ "$profile" == "all" ]]; then
    sources=(
        "general:gen_eval=examples/assets/gen_eval.txt"
        "general:for_diverse=examples/assets/for_diverse.txt"
        "general:drawbench=examples/assets/drawbench.txt"
        "general:coco_v2=examples/assets/coco_v2.txt"
        "memo:webster=examples/assets/sdv1_500_mem.txt"
        # "memo:membench=examples/assets/memorized_prompts_membench.txt"
    )
elif [[ "$profile" == "minimal" ]]; then
    sources=(
        "general:gen_eval=examples/assets/gen_eval.txt"
        "memo:webster=examples/assets/sdv1_500_mem.txt"
    )
else
    # default: standard (for_diverse 제외)
    sources=(
        "general:gen_eval=examples/assets/gen_eval.txt"
        "general:drawbench=examples/assets/drawbench.txt"
        "general:coco_v2=examples/assets/coco_v2.txt"
        "memo:webster=examples/assets/sdv1_500_mem.txt"
        # "memo:membench=examples/assets/memorized_prompts_membench.txt"
    )
fi

# ---- Paths ----
# ori_memo/ 에서 실행 가정; 다른 위치(예: docker)면 ROOT_DIR 로 override.
cd "${ROOT_DIR:-.}"

echo "=========================================="
echo "  [4. seedstd] Seed Std (per-prompt seed sensitivity, prompt-averaged)"
echo "  model=${model}  NFE=${NFE}  CFG=${cfg}  seed=${SEED}  batch=${batch}  n_prompts=${n_prompts}"
echo "  profile=${profile}  tag=${tag}  force_rollout=${force_rollout}"
echo "  sources: ${sources[*]}"
echo "=========================================="

python manifold.py \
    --seedstd ${tag} \
    --sources "${sources[@]}" \
    --model_key ${model_key} \
    --device ${DEVICE} \
    --cfg ${cfg} --NFE ${NFE} --seed ${SEED} --batch ${batch} \
    --n_prompts ${n_prompts} \
    --out_root ${out_root} \
    ${force_flag}

echo ""
echo "Done."
echo "  Paths:"
echo "    Latents: workdir/exp_main/manifold/latent_std/${model}/ddim/CFG=${cfg}_NFE=${NFE}/batch=${batch}/seed=${SEED}/"
echo "             ├── save/{general,memo}/{gen_eval,for_diverse,...}/"
echo "             │   ├── results/img_XXXX_YY.png"
echo "             │   ├── record/latents/latent.npz"
echo "             │   └── prompts/prompts.csv"
echo "             └── curve/prompts=${n_prompts}/"
echo "    Curves:  workdir/exp_main/manifold/latent_std/${model}/ddim/CFG=${cfg}_NFE=${NFE}/batch=${batch}/seed=${SEED}/curve/prompts=${n_prompts}/{csv,plot}/"
