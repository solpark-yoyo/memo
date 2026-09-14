"""trd/csv/total_metrics.csv → trade-off curve plot.

단일 method 의 lr sweep 점들을 x=metric(CLiP/Pick/ImgR), y=SSCD 로 연결.
lr 라벨/annotation 없이 곡선만. → {out_dir}/tradeoff_{metric}.png

Usage:
    python plot_trd.py --csv {path}/trd/csv/total_metrics.csv --out_dir {path}/trd/plot
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


def main():
    ap = argparse.ArgumentParser(description="trd total_metrics.csv → trade-off curve plot")
    ap.add_argument("--csv", required=True, help="trd/csv/total_metrics.csv")
    ap.add_argument("--out_dir", required=True, help="trd/plot/")
    ap.add_argument("--y_metric", default="SSCD_mean", help="y축 (기본 SSCD_mean, ↓ better)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(args.csv)))

    for x_key, x_name in X_METRICS:
        pts = []
        for r in rows:
            try:
                pts.append((float(r[x_key]), float(r[args.y_metric]), r.get("label", "")))
            except (ValueError, KeyError):
                continue
        if not pts:
            print(f"[skip] {x_name}: no valid points")
            continue
        pts.sort(key=lambda p: p[0])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        labels = [p[2] for p in pts]

        plt.figure(figsize=(7, 5))
        plt.plot(xs, ys, color="#1f77b4", marker="o", linewidth=2.2, markersize=9,
                 label="knob sweep")
        # 각 점 위에 knob 값(label) 표기
        for x, y, lab in zip(xs, ys, labels):
            if lab:
                plt.annotate(str(lab), (x, y), textcoords="offset points",
                             xytext=(0, 10), ha="center", fontsize=10, color="#333")
        plt.xlabel(x_name, fontsize=12)
        plt.ylabel(f"{args.y_metric} (memorization, ↓ better)", fontsize=12)
        plt.title(f"{args.y_metric} vs {x_name} — trade-off", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = os.path.join(args.out_dir, f"tradeoff_{x_name.lower()}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[plot] {out}  ({len(pts)} points)")


if __name__ == "__main__":
    main()
