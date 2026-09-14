"""init=1,3,5 (또는 임의 init 목록) 의 trade-off curve 를 한 plot 에 overlay 비교.

각 init 의 {base_path}/init={init}/nsteps={nsteps}/gap={gap}/trd/csv/total_metrics.csv
를 읽어 x=CLIP/Pick/ImgR, y=SSCD_mean(↓ better) 으로 init별 색/라벨 overlay.

출력: {out_dir}/tradeoff_compare_{clipscore,pickscore,imagereward}.png

Usage:
    python plot_init_compare.py \
        --base_path workdir/.../memoloss=minimization/ \
        --inits 1 3 5 --nsteps 2 --gap 3 \
        --out_dir workdir/.../memoloss=minimization/trd_compare
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


X_METRICS = [
    ("CLIP_mean", "CLIPScore"),
    ("Pick_mean", "PickScore"),
    ("ImgR_mean", "ImageReward"),
]
COLORS = ["tab:red", "tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:brown"]


def read_curve(csv_path, x_key, y_metric):
    """total_metrics.csv → (xs, ys) sorted by x."""
    rows = list(csv.DictReader(open(csv_path)))
    pts = []
    for r in rows:
        try:
            pts.append((float(r[x_key]), float(r[y_metric])))
        except (ValueError, KeyError):
            continue
    pts.sort(key=lambda p: p[0])
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    ap = argparse.ArgumentParser(description="init 목록별 trade-off curve overlay 비교")
    ap.add_argument("--base_path", required=True,
                   help="init 부모 경로 (예: .../memoloss=minimization/). 끝 슬래시 무관.")
    ap.add_argument("--inits", type=int, nargs="+", default=[1, 3, 5],
                   help="비교할 init_steps 목록")
    ap.add_argument("--nsteps", type=int, default=2)
    ap.add_argument("--gap", type=int, default=3)
    ap.add_argument("--out_dir", required=True, help="출력 디렉토리")
    ap.add_argument("--y_metric", default="SSCD_mean", help="y축 (기본 SSCD_mean, ↓ better)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = args.base_path.rstrip("/") + "/"

    for x_key, x_name in X_METRICS:
        plt.figure(figsize=(8, 5.5))
        plotted = 0
        for i, init in enumerate(args.inits):
            csv_path = f"{base}init={init}/nsteps={args.nsteps}/gap={args.gap}/trd/csv/total_metrics.csv"
            if not os.path.isfile(csv_path):
                print(f"  [skip] init={init}: {csv_path} 없음")
                continue
            xs, ys = read_curve(csv_path, x_key, args.y_metric)
            if not xs:
                print(f"  [skip] init={init}: {x_name} 점 없음")
                continue
            plt.plot(xs, ys, color=COLORS[i % len(COLORS)], marker="o",
                     linewidth=2.2, markersize=8, label=f"init={init}")
            plotted += 1
        if plotted == 0:
            print(f"  [skip] {x_name}: 플롯할 init 없음")
            plt.close()
            continue
        plt.xlabel(x_name, fontsize=12)
        plt.ylabel(f"{args.y_metric} (memorization, ↓ better)", fontsize=12)
        plt.title(f"{args.y_metric} vs {x_name} — init compare (init={args.inits})", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        out = os.path.join(args.out_dir, f"tradeoff_compare_{x_name.lower()}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[plot] {out}  ({plotted} inits)")


if __name__ == "__main__":
    main()
