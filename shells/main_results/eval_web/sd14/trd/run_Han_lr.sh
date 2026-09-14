#!/bin/bash
# ===================================================================
#  trd (init_score_noise = Han et al.): lr sweep trade-off curve — Chen
#   1) collect_trd.py → {path}/trd/csv/{total_log,total_metrics}.csv
#   2) plot_trd.py    → {path}/trd/plot/tradeoff_{clipscore,pickscore,imagereward}.png
#  path = init_score_noise 의 CFG 부모 (.../NFE=50/per_sample/CFG=7.5)
#  eval 경로 = {path}/lr={lr}/tl={tl}/oi={oi}/seed={seed}/eval
#  실행 (ori_memo/, conda div_DM): bash shells/eval_chen/trd/run_Han_lr.sh
# ===================================================================

# =========================== 1. [Parser] ===========================
path="workdir/memorization/sd14_base/init_score_noise/NFE=50/per_sample/CFG=7.5"
lr_list=(0.0 0.01 0.03 0.05)
seed=42
batch=5
tl=0.9                      # target_loss
oi=10                       # optim_iters

cd "${ROOT_DIR:-.}"         # ori_memo/ 에서 실행 가정

echo "========================================="
echo "  trd (init_score_noise): method=init_score_noise — Chen"
echo "  path=${path}"
echo "  lr_list=${lr_list[*]}"
echo "  seed=${seed} batch=${batch}  tl=${tl} oi=${oi}"
echo "========================================="

# =========================== 2. [csv] ===========================
echo ""
echo "================== [INFO]: Collect trd csv → ${path}/trd/csv/ =================="
python collect_trd.py --method init_score_noise --path "${path}" \
    --lr_list ${lr_list[@]} --seed ${seed} --batch ${batch} --tl ${tl} --oi ${oi}

# =========================== 3. [plot] ===========================
echo ""
echo "================== [INFO]: Plot trade-off curves → ${path}/trd/plot/ =================="
python plot_trd.py --csv "${path}/trd/csv/total_metrics.csv" --out_dir "${path}/trd/plot"

echo ""
echo "[Done] → ${path}/trd/csv/{total_log,total_metrics}.csv  +  ${path}/trd/plot/tradeoff_*.png"
