"""두 모델(Han et al. vs ours) 의 trd/csv/total_metrics.csv 를 overlay 비교.

각 모델 lr sweep 점들을 x=T2I(CLIP/Pick/ImgR), y=SSCD 로 한 plot 에 겹쳐 그림.
lr 라벨 없이 곡선만. → {out_dir}/compare_{clipscore,pickscore,imagereward}.png

Usage:
    python plot_compare.py \
        --han_csv  {han_path}/trd/csv/total_metrics.csv \
        --ours_csv {ours_path}/trd/csv/total_metrics.csv \
        --out_dir  {out_base}/trd
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


def read_rows(path):
    return list(csv.DictReader(open(path)))


def main():
    ap = argparse.ArgumentParser(description="2 모델 trd/csv overlay 비교 plot")
    ap.add_argument("--han_csv", required=True, help="Han et al. trd/csv/total_metrics.csv")
    ap.add_argument("--ours_csv", required=True, help="ours(init_opti) trd/csv/total_metrics.csv")
    ap.add_argument("--out_dir", required=True, help="출력 폴더 (예: {base}/trd)")
    ap.add_argument("--y_metric", default="SSCD_mean", help="y축 (기본 SSCD_mean, ↓ better)")
    ap.add_argument("--han_name", default="Han et al. (init_score_noise)")
    ap.add_argument("--ours_name", default="Ours (init_opti)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    models = [
        (args.han_name, args.han_csv, "#d62728", "^"),
        (args.ours_name, args.ours_csv, "#1f77b4", "o"),
    ]

    for x_key, x_name in X_METRICS:
        plt.figure(figsize=(7, 5))
        for name, csv_path, color, marker in models:
            if not os.path.exists(csv_path):
                print(f"[skip] {name}: {csv_path} 없음")
                continue
            pts = []
            for r in read_rows(csv_path):
                try:
                    pts.append((float(r[x_key]), float(r[args.y_metric])))
                except (ValueError, KeyError):
                    continue
            if not pts:
                continue
            pts.sort(key=lambda p: p[0])
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            plt.plot(xs, ys, color=color, marker=marker, linewidth=2.2, markersize=9, label=name)
        plt.xlabel(x_name, fontsize=12)
        plt.ylabel(f"{args.y_metric} (memorization, ↓ better)", fontsize=12)
        plt.title(f"{args.y_metric} vs {x_name} — model comparison", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        out = os.path.join(args.out_dir, f"compare_{x_name.lower()}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[plot] {out}")

    print(f"\n[Done] → {args.out_dir}/compare_*.png")


if __name__ == "__main__":
    main()
