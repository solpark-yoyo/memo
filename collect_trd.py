"""trade-off curve용 trd 폴더 관리 (다 method 지원).

{path}/trd/csv/  아래에:
  - total_log.csv     : eval_path, num_sample, label, CLIP/Pick/ImgR/SSCD 의 mean·std·max (전부)
  - total_metrics.csv : label, CLIP_mean, Pick_mean, ImgR_mean, SSCD_mean (mean 만)
{path}/trd/plot/ 폴더도 생성 (plot 저장용, 비워둠).

--method 로 eval 경로 패턴을 선택:
  - init_opti        : {path}/lr={lr}/seed={S}/batch={B}/eval          (label=lr)
  - init_score_noise : {path}/lr={lr}/tl={tl}/oi={oi}/seed={S}/eval    (label=lr)
  - ddim / cno       : {path} 아래 eval 자동 탐색 (단일, label=method 명)

Usage:
    # init_opti (기존)
    python collect_trd.py --path .../gap=3 --lr_list 0.02 0.04
    # init_score_noise
    python collect_trd.py --method init_score_noise --path .../CFG=7.5 --lr_list 0.0 0.02 --tl 0.9 --oi 10
    # ddim / cno (eval 부모 디렉토리를 path 로)
    python collect_trd.py --method ddim --path .../ddim/CFG=7.5_NFE=50/seed=42
"""
import argparse
import csv
import os

import numpy as np


def read_sscd(eval_dir):
    """sscd_gt_metrics.csv (또는 chen_sscd_gt_metrics.csv) → mean/std/max/num."""
    for name in ("chen_sscd_gt_metrics.csv", "sscd_gt_metrics.csv"):
        p = os.path.join(eval_dir, name)
        if not os.path.exists(p):
            continue
        per_prompt = []
        max_avg = None
        num = None
        with open(p) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split(",")
                head = parts[0].strip()
                if head == "num_prompts_evaluated":
                    try:
                        num = int(float(parts[1]))
                    except (IndexError, ValueError):
                        pass
                elif head == "max_sscd_to_gt_avg":
                    try:
                        max_avg = float(parts[1])
                    except (IndexError, ValueError):
                        pass
                elif head.isdigit() and len(parts) >= 3:
                    try:
                        per_prompt.append(float(parts[2]))
                    except ValueError:
                        pass
        arr = np.array(per_prompt) if per_prompt else np.array([])
        return {
            "mean": float(arr.mean()) if len(arr) else None,
            "std": float(arr.std()) if len(arr) else None,
            "max": max_avg,
            "num": num,
        }
    return {}


def read_t2i(eval_dir):
    """t2i_metrics.csv (또는 chen_t2i_metrics.csv) → {metric: (mean, std)}."""
    for name in ("chen_t2i_metrics.csv", "t2i_metrics.csv"):
        p = os.path.join(eval_dir, name)
        if not os.path.exists(p):
            continue
        out = {}
        with open(p) as f:
            for row in csv.reader(f):
                if len(row) >= 3 and row[0].strip() in ("CLIPScore", "PickScore", "ImageReward"):
                    try:
                        out[row[0].strip()] = (float(row[1]), float(row[2]))
                    except ValueError:
                        pass
        return out
    return {}


def get_eval_pairs(args):
    """(label, eval_dir) 목록을 method 별로 생성."""
    pairs = []
    if args.method == "init_opti":
        for lr in args.lr_list:
            d = os.path.join(args.path, f"lr={lr}", f"seed={args.seed}", f"batch={args.batch}", "eval")
            pairs.append((lr, d))
    elif args.method == "init_score_noise":
        for lr in args.lr_list:
            d = os.path.join(args.path, f"lr={lr}", f"tl={args.tl}", f"oi={args.oi}",
                             f"seed={args.seed}", "eval")
            pairs.append((lr, d))
    elif args.method in ("ddim", "cno"):
        # path 아래 eval 위치 후보 (batch 유/무)
        for cand in (
            os.path.join(args.path, "eval"),
            os.path.join(args.path, f"seed={args.seed}", "eval"),
            os.path.join(args.path, f"seed={args.seed}", f"batch={args.batch}", "eval"),
        ):
            if os.path.isdir(cand):
                pairs.append((args.method, cand))
                break
        if not pairs:
            print(f"[warn] {args.method}: {args.path} 아래 eval 디렉토리를 못 찾음")
    else:
        raise ValueError(f"unknown method: {args.method}")
    return pairs


def _fmt(v):
    return f"{v:.6f}" if isinstance(v, (int, float)) else ""


def main():
    ap = argparse.ArgumentParser(description="trd/csv/{total_log,total_metrics}.csv 생성 (다 method)")
    ap.add_argument("--method", default="init_opti",
                    choices=["init_opti", "init_score_noise", "ddim", "cno"],
                    help="eval 경로 패턴 선택")
    ap.add_argument("--path", required=True,
                    help="init_opti/init_score_noise: lr 부모 / ddim,cno: eval 부모(또는 eval 이 있는 dir)")
    ap.add_argument("--lr_list", nargs="*", default=[], help="lr 값들 (init_opti/init_score_noise)")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--batch", default="5")
    ap.add_argument("--tl", default="0.9", help="init_score_noise target_loss")
    ap.add_argument("--oi", default="10", help="init_score_noise optim_iters")
    args = ap.parse_args()

    trd_csv = os.path.join(args.path, "trd", "csv")
    trd_plot = os.path.join(args.path, "trd", "plot")
    os.makedirs(trd_csv, exist_ok=True)
    os.makedirs(trd_plot, exist_ok=True)

    log_fields = ["eval_path", "num_sample", "label",
                  "CLIP_mean", "CLIP_std",
                  "Pick_mean", "Pick_std",
                  "ImgR_mean", "ImgR_std",
                  "SSCD_mean", "SSCD", "SSCD_std"]
    metric_fields = ["label", "CLIP_mean", "Pick_mean", "ImgR_mean", "SSCD_mean"]

    log_rows = []
    for label, eval_dir in get_eval_pairs(args):
        sscd = read_sscd(eval_dir)
        t2i = read_t2i(eval_dir)
        clip = t2i.get("CLIPScore", (None, None))
        pick = t2i.get("PickScore", (None, None))
        imgr = t2i.get("ImageReward", (None, None))
        row = {
            "eval_path": eval_dir,
            "num_sample": sscd.get("num") if sscd.get("num") is not None else "",
            "label": label,
            "CLIP_mean": _fmt(clip[0]),
            "CLIP_std": _fmt(clip[1]),
            "Pick_mean": _fmt(pick[0]),
            "Pick_std": _fmt(pick[1]),
            "ImgR_mean": _fmt(imgr[0]),
            "ImgR_std": _fmt(imgr[1]),
            "SSCD_mean": _fmt(sscd.get("mean")),
            "SSCD": _fmt(sscd.get("max")),        # max_sscd_to_gt_avg
            "SSCD_std": _fmt(sscd.get("std")),
        }
        log_rows.append(row)
        print(f"  label={str(label):6s} CLIP={row['CLIP_mean']:>9s} Pick={row['Pick_mean']:>8s} "
              f"ImgR={row['ImgR_mean']:>8s} SSCD={row['SSCD_mean']:>8s} "
              f"(eval: {'OK' if os.path.isdir(eval_dir) else 'MISSING'})")

    if not log_rows:
        print("[Done] 수집된 eval 없음 — csv 미생성")
        return

    with open(os.path.join(trd_csv, "total_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_fields)
        w.writeheader()
        w.writerows(log_rows)

    with open(os.path.join(trd_csv, "total_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metric_fields)
        w.writeheader()
        for r in log_rows:
            w.writerow({k: r[k] for k in metric_fields})

    print(f"\n[Done] {len(log_rows)} rows ({args.method})")
    print(f"  {os.path.join(trd_csv, 'total_log.csv')}")
    print(f"  {os.path.join(trd_csv, 'total_metrics.csv')}")
    print(f"  {trd_plot}/  (plot 용 빈 폴더)")


if __name__ == "__main__":
    main()
