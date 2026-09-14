"""trade-off curve용 trd 폴더 관리 (다 method 지원).

{path}/trd/csv/  아래에:
  - total_log.csv     : eval_path, num_sample, label, CLIP/Pick/ImgR/SSCD 의 mean·std·max (전부)
  - total_metrics.csv : label, CLIP_mean, Pick_mean, ImgR_mean, SSCD_mean (mean 만)
{path}/trd/plot/ 폴더도 생성 (plot 저장용, 비워둠).

--method 로 eval 경로 패턴을 선택 (저장 레이아웃 2026-09-04 4차 변경:
  sweep knob(lr/tl/c1)이 batch/seed 아래(가장 안쪽)로 이동 — ours·jeon·ren·wen·han 전부.
  구 레이아웃(knob 이 바깥)은 폴백으로 수집):
  - init_opti        : {path}/batch={B}/seed={S}/lr={lr}/eval          (label=lr)
  - init_score_noise : {path}/lr={lr}/oi={oi}/batch={B}/seed={S}/tl={tl}/eval  (label=tl)
  - jeon / ren / wen : {path}/(thres/ms·it·)batch={B}/seed={S}/(lr|c1|tl)=/eval (label=knob)
  - ddim / cno       : {path} 아래 eval 자동 탐색 (단일, label=method 명)

eval subset 프로토콜 (2026-09-04): inference 전체 중 앞 N 장만 평가한 eval/${N} 에서 수집 시
--eval_sub ${N}, trd 출력은 --trd_dir ".../trd/${N}". 지표는 merge_benchmark 의
total_metrics.csv 를 우선 읽고, 없으면 chen_sscd_gt/chen_t2i 쌍으로 fallback
(SSCD max·num 은 항상 chen_sscd_gt_metrics.csv 에서).

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


def read_total_metrics(eval_dir):
    """merge_benchmark 가 남긴 total_metrics.csv (metric,mean,std) → {"metric": (mean, std)}.

    CLIPScore/PickScore/ImageReward/SSCD-to-GT (mean) 등을 read_t2i 와 같은 형태로 반환.
    파일이 없으면 None (이 경우 chen_*.csv 쌍으로 fallback).
    """
    p = os.path.join(eval_dir, "total_metrics.csv")
    if not os.path.exists(p):
        return None

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    out = {}
    with open(p) as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip():
                out[row[0].strip()] = (_f(row[1]), _f(row[2]) if len(row) >= 3 else None)
    return out or None


def _sb_dir(args, *prefix):
    """{prefix...}/batch={B}/seed={S}/eval — 신규 레이아웃 우선, 기존(seed→batch) 폴백."""
    for sb in (os.path.join(f"batch={args.batch}", f"seed={args.seed}"),
               os.path.join(f"seed={args.seed}", f"batch={args.batch}")):
        d = os.path.join(args.path, *prefix, sb, "eval")
        if os.path.isdir(d):
            return d
    return os.path.join(args.path, *prefix, f"batch={args.batch}", f"seed={args.seed}", "eval")


def _knob_dir(args, *prefix, knob):
    """{prefix...} 아래 knob(lr)·batch·seed 위치별 후보 — 신규(batch/seed 아래 knob) 우선, 구 레이아웃 폴백."""
    b, s = f"batch={args.batch}", f"seed={args.seed}"
    for segs in ((*prefix, b, s, knob),      # 신규 (2026-09-04 2차): batch/seed 아래 lr
                 (knob, *prefix, b, s),      # 2026-09-04 1차: lr 아래 batch/seed
                 (knob, *prefix, s, b),      # 구식: lr 아래 seed/batch
                 (knob, *prefix, s)):        # 최구식(jeon 초기): lr/thres/ms/seed (batch 없음)
        d = os.path.join(args.path, *segs, "eval")
        if os.path.isdir(d):
            return d
    return os.path.join(args.path, *prefix, b, s, knob, "eval")


def get_eval_pairs(args):
    """(label, eval_dir) 목록을 method 별로 생성."""
    pairs = []
    if args.method == "init_opti":
        for lr in args.lr_list:
            pairs.append((lr, _knob_dir(args, knob=f"lr={lr}")))
    elif args.method == "init_score_noise":
        if args.lr is not None:
            # tl-스윕 모드 — lr 고정, lr_list 값을 target_loss(tl)로 해석 (label=tl)
            # 신규(4차): lr/oi/batch/seed 아래 tl (knob=tl 가장 안쪽)
            for tl_val in args.lr_list:
                pairs.append((tl_val, _knob_dir(args, f"lr={args.lr}", f"oi={args.oi}", knob=f"tl={tl_val}")))
        else:
            # 기존 lr-스윕 (label=lr, tl 고정)
            for lr in args.lr_list:
                pairs.append((lr, _sb_dir(args, f"lr={lr}", f"tl={args.tl}", f"oi={args.oi}")))
    elif args.method == "wen":
        # {path}/it={oi}/batch={B}/seed={S}/tl={tl}/eval  (label=tl — knob: target_loss λ)
        for tl in args.lr_list:
            pairs.append((tl, _knob_dir(args, f"it={args.oi}", knob=f"tl={tl}")))
    elif args.method == "ren":
        # {path}/batch={B}/seed={S}/c1={c1}/eval  (label=c1 — knob: attention mask 계수)
        for c1 in args.lr_list:
            pairs.append((c1, _knob_dir(args, knob=f"c1={c1}")))
    elif args.method == "jeon":
        # {path}/thres={tl}/ms={oi}/batch={B}/seed={S}/lr={lr}/eval  (label=lr — knob: miti_lr)
        for lr in args.lr_list:
            pairs.append((lr, _knob_dir(args, f"thres={args.tl}", f"ms={args.oi}", knob=f"lr={lr}")))
    elif args.method in ("ddim", "cno"):
        # path 아래 eval 위치 후보 (batch 유/무 · 신규/기존 순서)
        for cand in (
            os.path.join(args.path, "eval"),
            _sb_dir(args),
            os.path.join(args.path, f"seed={args.seed}", "eval"),
        ):
            if os.path.isdir(cand):
                pairs.append((args.method, cand))
                break
        if not pairs:
            print(f"[warn] {args.method}: no eval dir found under {args.path}")
    else:
        raise ValueError(f"unknown method: {args.method}")
    if args.eval_sub:
        pairs = [(lab, os.path.join(d, args.eval_sub)) for lab, d in pairs]
    return pairs


def _fmt(v):
    return f"{v:.6f}" if isinstance(v, (int, float)) else ""


def main():
    ap = argparse.ArgumentParser(description="trd/csv/{total_log,total_metrics}.csv 생성 (다 method)")
    ap.add_argument("--method", default="init_opti",
                    choices=["init_opti", "init_score_noise", "ddim", "cno",
                             "wen", "ren", "jeon"],
                    help="eval 경로 패턴 선택 (wen/ren/jeon: 신규 baseline — knob 리스트는 --lr_list)")
    ap.add_argument("--path", required=True,
                    help="init_opti/init_score_noise: lr 부모 / ddim,cno: eval 부모(또는 eval 이 있는 dir)")
    ap.add_argument("--lr_list", nargs="*", default=[], help="lr 값들 (init_opti/init_score_noise)")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--batch", default="5")
    ap.add_argument("--tl", default="0.9", help="init_score_noise target_loss")
    ap.add_argument("--lr", default=None,
                    help="init_score_noise tl-스윕 모드 — lr 고정값. 주어지면 lr_list를 tl 값으로 해석")
    ap.add_argument("--oi", default="10", help="init_score_noise optim_iters")
    ap.add_argument("--eval_sub", default="",
                    help="eval 하위 폴더 (예: 100 — 앞 100장만 eval 하는 프로토콜의 eval/100, "
                         "또는 GT 폴더 cvpr_2025 / sdv1_500). 지정 시 {knob}/.../eval/{sub} 수집")
    ap.add_argument("--trd_dir", default=None,
                    help="trd 출력 디렉토리 (기본: {path}/trd). 예: lr 고정 스윕에서 "
                         "pattern 탐색은 상위 path, 출력은 lr=0.01/trd 아래에 두고 싶을 때")
    args = ap.parse_args()

    _trd_root = args.trd_dir if args.trd_dir else os.path.join(args.path, "trd")
    trd_csv = os.path.join(_trd_root, "csv")
    trd_plot = os.path.join(_trd_root, "plot")
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
        sscd = read_sscd(eval_dir)                # SSCD max·num 용 (chen_sscd_gt_metrics.csv)
        merged = read_total_metrics(eval_dir)     # ★ 우선: merge_benchmark 의 total_metrics.csv
        t2i = merged if merged else read_t2i(eval_dir)
        if merged:
            m = merged.get("SSCD-to-GT (mean)")
            if m and m[0] is not None:
                sscd["mean"], sscd["std"] = m
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
        print("[Done] no eval collected — csv not created")
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
    print(f"  {trd_plot}/  (empty dir for plots)")


if __name__ == "__main__":
    main()
