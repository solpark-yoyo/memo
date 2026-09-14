#!/bin/bash
# ===================================================================
#  trd (ours = init_opti): cfgsr(deferring ratio) sweep
#    각 cfgsr 마다 lr sweep trade-off curve(csv+plot) 생성 → cfgsr 비교
#
#   cfgsr_list 각 cfgsr 에 대해:
#     path = {base_path}/cfgsr={cfgsr}/base_s_ratio=0.5_lambda_align=0.00/
#            memoloss=minimization/init=5/nsteps=5/gap=3
#     → collect_trd.py (lr sweep) → plot_trd.py
#     → {path}/trd/{csv,plot}/
#
#  ※ cfgsr_list 값은 실제 폴더명(float 표기) 기준: 0.0, 0.05, 0.1, 0.15
#     (0.00→0.0, 0.10→0.1 로 폴더가 생성되므로)
#  ※ 이 코드 수행 시점 = eval 이 끝난 상황 가정. eval 없는 cfgsr/lr 은 MISSING 표시.
#
#  실행 (ori_memo/ 에서, conda div_DM 활성화 후):
#    bash shells/eval_wen/trd/run_ours_cfgsr.sh
# ===================================================================

# =========================== 1. [Parser] ===========================
base_path="workdir/memorization/sd14_memor_LAION2B_40k/init_opti/CFG=7.5_NFE=50"
suffix="base_s_ratio=0.5_lambda_align=0.00/memoloss=minimization/init=5/nsteps=5/gap=3"
cfgsr_list=(0.0 0.05 0.10 0.15)        # deferring ratio (폴더명 기준)
lr_list=(0.02 0.04 0.06 0.08)
seed=42
batch=5

cd "${ROOT_DIR:-.}"         # ori_memo/ 에서 실행 가정; 다른 위치면 ROOT_DIR 로 override

echo "========================================="
echo "  trd (ours): cfgsr sweep"
echo "  base_path=${base_path}"
echo "  cfgsr_list=${cfgsr_list[*]}"
echo "  lr_list=${lr_list[*]}"
echo "  seed=${seed} batch=${batch}"
echo "========================================="

# =========================== 2. [per-cfgsr trd] ===========================
for cfgsr in "${cfgsr_list[@]}"; do
    path="${base_path}/cfgsr=${cfgsr}/${suffix}"
    echo ""
    echo "############ cfgsr=${cfgsr} → ${path}/trd/ ############"

    echo "  [csv]  collect_trd.py"
    python collect_trd.py --method init_opti --path "${path}" \
        --lr_list ${lr_list[@]} --seed ${seed} --batch ${batch}

    # csv 가 생성된 경우에만 plot
    if [[ -f "${path}/trd/csv/total_metrics.csv" ]]; then
        echo "  [plot] plot_trd.py"
        python plot_trd.py --csv "${path}/trd/csv/total_metrics.csv" --out_dir "${path}/trd/plot"
    else
        echo "  [skip] total_metrics.csv 없음 (eval 누락) — plot 생략"
    fi
done

echo ""
echo "[Done] 각 cfgsr trd → ${base_path}/cfgsr=*/${suffix}/trd/{csv,plot}/"
