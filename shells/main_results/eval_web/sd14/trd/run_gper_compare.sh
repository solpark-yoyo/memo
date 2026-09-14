#!/bin/bash
# ===================================================================
#  trd: gper=0 vs gper=1 (또는 임의 gper 목록) trade-off curve 비교 — Chen
#
#   같은 init/nsteps/gap 에서 gper=0(init_opti 원본) vs gper=1(GPER on) 의
#   total_metrics.csv 를 읽어 한 plot 에 overlay → {out_dir}/tradeoff_compare_*.png
#
#  실행 (ori_memo/, conda div_DM):
#    bash shells/eval_chen/trd/run_gper_compare.sh                   # init=1, gpers=0 1
#    INIT=3 GPERS="0 1" bash shells/eval_chen/trd/run_gper_compare.sh
# ===================================================================

# =========================== 1. [Parser] ===========================
base_path="workdir/memorization/sd14_base/init_opti/CFG=7.5_NFE=50"
cfgsr="${CFGSR:-0.0}"
base_s_ratio="${BASE_S_RATIO:-0.5}"
lambda_align="${LAMBDA_ALIGN:-0.00}"
memoloss="${MEMOLOSS:-minimization}"
init_steps="${INIT:-1}"
nsteps="${NSTEPS:-2}"
gap="${GAP:-3}"
gpers="${GPERS:-0 1}"          # 비교할 gper 값들 (공백 구분)
y_metric="${Y_METRIC:-SSCD_mean}"

out_dir="${base_path}/trd_gper_compare/init=${init_steps}"

cd "${ROOT_DIR:-.}"  # ori_memo/ 에서 실행 가정

# gper별 total_metrics.csv 경로/라벨 구성
sub="cfgsr=${cfgsr}/base_s_ratio=${base_s_ratio}_lambda_align=${lambda_align}/memoloss=${memoloss}/init=${init_steps}/nsteps=${nsteps}/gap=${gap}/trd/csv/total_metrics.csv"
csvs=()
labels=()
for g in ${gpers}; do
    csvs+=("${base_path}/gper=${g}/${sub}")
    labels+=("gper=${g}")
done

echo "========================================="
echo "  trd: gper compare — Chen"
echo "  gpers=(${gpers})  init=${init_steps}  nsteps=${nsteps}  gap=${gap}"
echo "  cfgsr=${cfgsr}  base_s_ratio=${base_s_ratio}  lambda_align=${lambda_align}  memoloss=${memoloss}"
echo "========================================="

# 존재 확인 (없는 건 plot_gper_compare.py 가 skip)
for i in "${!csvs[@]}"; do
    if [ -f "${csvs[$i]}" ]; then
        echo "  [OK] ${labels[$i]}"
    else
        echo "  [X]  ${labels[$i]}: ${csvs[$i]} (결과 없음 → skip)"
    fi
done

# =========================== 2. [plot] ===========================
echo ""
echo "================== [INFO]: overlay trade-off curves → ${out_dir}/ =================="
CSV_LIST="${csvs[*]}" LABELS_LIST="${labels[*]}" OUT_DIR="${out_dir}" Y_METRIC="${y_metric}" python - <<'PYEOF'
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csvs   = os.environ["CSV_LIST"].split()
labels = os.environ["LABELS_LIST"].split()
out_dir = os.environ["OUT_DIR"]
y = os.environ.get("Y_METRIC", "SSCD_mean")

X = [("CLIP_mean","CLIPScore"), ("Pick_mean","PickScore"), ("ImgR_mean","ImageReward")]
COLORS = ["tab:red","tab:blue","tab:green","tab:purple","tab:orange","tab:brown"]
os.makedirs(out_dir, exist_ok=True)

def read_curve(p, xk):
    pts = []
    for r in csv.DictReader(open(p)):
        try: pts.append((float(r[xk]), float(r[y])))
        except (ValueError, KeyError): continue
    pts.sort(key=lambda q: q[0])
    return [q[0] for q in pts], [q[1] for q in pts]

for xk, xn in X:
    plt.figure(figsize=(8, 5.5))
    plotted = 0
    for i, cp in enumerate(csvs):
        if not os.path.isfile(cp):
            print(f"  [skip] {cp} 없음"); continue
        xs, ys = read_curve(cp, xk)
        if not xs:
            print(f"  [skip] {labels[i] if i<len(labels) else i}: {xn} 점 없음"); continue
        plt.plot(xs, ys, color=COLORS[i % len(COLORS)], marker="o",
                 linewidth=2.2, markersize=8,
                 label=labels[i] if i < len(labels) else f"curve{i}")
        plotted += 1
    if plotted < 2:
        print(f"  [skip] {xn}: 비교하려면 2개 이상 curve 필요 (현재 {plotted})")
        plt.close(); continue
    plt.xlabel(xn, fontsize=12)
    plt.ylabel(f"{y} (memorization, ↓ better)", fontsize=12)
    plt.title(f"{y} vs {xn} — gper compare", fontsize=12)
    plt.grid(True, alpha=0.3); plt.legend(fontsize=10); plt.tight_layout()
    out = os.path.join(out_dir, f"tradeoff_compare_{xn.lower()}.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"[plot] {out}  ({plotted} curves)")
PYEOF

echo ""
echo "[Done] → ${out_dir}/tradeoff_compare_{clipscore,pickscore,imagereward}.png"
