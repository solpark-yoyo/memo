"""tradeoff.csv → SSCD vs (CLIP/Pick/IR) trade-off curve plot.

x축 metric 을 argparser 로 선택: clipscore | pickscore | imagereward
y축 = sscd (memorization, ↓ 좋음) → bottom-right 로 갈수록 좋은 메서드.

Usage:
    python plot_tradeoff.py --csv tradeoff.csv --x_metric clipscore --out chen_tradeoff.png
    python plot_tradeoff.py --csv tradeoff.csv --x_metric pickscore
    python plot_tradeoff.py --csv tradeoff.csv --x_metric imagereward
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="SSCD vs utility trade-off curve (Chen eval)")
    ap.add_argument("--csv", required=True, help="collect_tradeoff.py 결과 tradeoff.csv")
    ap.add_argument("--x_metric", default="clipscore",
                    choices=["clipscore", "pickscore", "imagereward"],
                    help="x축 metric (기본 clipscore)")
    ap.add_argument("--y_metric", default="sscd", help="y축 metric (기본 sscd, ↓ 좋음)")
    ap.add_argument("--methods", default="ddim,cno_infoNCE,init_score_noise,init_opti",
                    help="comma list of methods (collect_tradeoff 의 method 컬럼값). "
                         "지정한 것만/순서대로 plot")
    ap.add_argument("--out", default="tradeoff_curve.png")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    groups = defaultdict(list)
    for r in rows:
        try:
            x = float(r[args.x_metric])
            y = float(r[args.y_metric])
        except (ValueError, KeyError):
            continue
        groups[r.get("method", "")].append((x, y, r.get("lr", "")))

    plt.figure(figsize=(7, 5))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    markers = {"ddim": "s", "cno_infoNCE": "^", "init_score_noise": "D", "init_opti": "o",
               "spectral_opt": "*", "jeon": "v", "ren": "P", "wen": "X"}
    label_map = {"ddim": "DDIM", "cno_infoNCE": "CNO",
                 "init_score_noise": "Han (init_score_opti)", "init_opti": "Ours (init_opti)",
                 "spectral_opt": "Ours (spectral eps)", "jeon": "Jeon (SAIL)",
                 "ren": "Ren (MemAttn)", "wen": "Wen (prompt aug)"}
    # --methods 로 지정한 것만 / 순서대로 plot
    sel = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods = [m for m in sel if m in groups] + [m for m in groups if m not in sel]
    for i, method in enumerate(methods):
        pts = groups[method]
        pts.sort(key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = label_map.get(method, method or "(unknown)")
        plt.plot(xs, ys, marker=markers.get(method, "o"), color=colors[i % len(colors)],
                 label=label, linewidth=2, markersize=9)
        # lr annotate 제거 (사용자 요청)

    plt.xlabel(args.x_metric.upper())
    plt.ylabel(f"{args.y_metric.upper()} (memorization, ↓ better)")
    plt.title(args.title or f"{args.y_metric} vs {args.x_metric} — Chen eval trade-off")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[Done] → {args.out}  (x={args.x_metric}, y={args.y_metric})")


if __name__ == "__main__":
    main()
