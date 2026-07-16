#!/bin/bash
# ===================================================================
#  trd (ours = init_opti): lr sweep trade-off curve (csv + plot) 한 번에 생성
#
#   1) collect_trd.py  → {path}/trd/csv/{total_log,total_metrics}.csv
#   2) plot_trd.py     → {path}/trd/plot/tradeoff_{clipscore,pickscore,imagereward}.png
#
#  path = ours(init_opti) 의 lr 부모 (예: .../init=5/nsteps=5/gap=3)
#
#  실행 (ori_memo/ 에서, conda div_DM 활성화 후):
#    bash shells/eval_chen/trd/run_ours_lr.sh
# ===================================================================

# =========================== 1. [Parser] ===========================
path="workdir/memorization/sd14_memor_LAION2B_40k/init_opti/CFG=7.5_NFE=50/cfgsr=0.0/base_s_ratio=0.5_lambda_align=0.00/memoloss=minimization/init=5/nsteps=5/gap=3"
lr_list=(0.02 0.04 0.06 0.08)
seed=42
batch=5

cd "${ROOT_DIR:-.}"         # ori_memo/ 에서 실행 가정; 다른 위치면 ROOT_DIR 로 override

echo "========================================="
echo "  trd (ours): method=init_opti"
echo "  path=${path}"
echo "  lr_list=${lr_list[*]}"
echo "  seed=${seed} batch=${batch}"
echo "========================================="

# =========================== 2. [csv] ===========================
echo ""
echo "================== [INFO]: Collect trd csv → ${path}/trd/csv/ =================="
python collect_trd.py --method init_opti --path "${path}" \
    --lr_list ${lr_list[@]} --seed ${seed} --batch ${batch}

# =========================== 3. [plot] ===========================
echo ""
echo "================== [INFO]: Plot trade-off curves → ${path}/trd/plot/ =================="
python plot_trd.py --csv "${path}/trd/csv/total_metrics.csv" --out_dir "${path}/trd/plot"

echo ""
echo "[Done] → ${path}/trd/csv/{total_log,total_metrics}.csv  +  ${path}/trd/plot/tradeoff_*.png"
