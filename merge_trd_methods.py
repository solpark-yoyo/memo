"""baselines 각 method 의 trd/<subset>/csv 를 하나로 병합 → 전체 비교 trd.

각 method 의 이미 정제된 trd csv (label,CLIP_mean,Pick_mean,ImgR_mean,SSCD_mean) 를
plot_tradeoff.py 입력 형식 (method,lr,clipscore,pickscore,imagereward,sscd) 으로 변환.
knob 이름(lr=/c1=/tl=)과 무관하게 원본 label 값을 그대로 lr 컬럼에 기록.

- collect_tradeoff.py 와 달리 **재귀 탐색 없음** — 명시된 trd 경로만 읽음 (others/ 미참조)
- ddim 은 trd 없음 → eval/<subset>/total_metrics.csv (metric,mean 형식) 에서 단일 점 추출
- 병합 후 plot_tradeoff.py 로 3종 (clipscore/pickscore/imagereward) 곡선 생성은 별도 호출

Usage (ori_memo/ 에서):
    python merge_trd_methods.py --base_dir workdir/memorization/sd14_base/baselines \
        --subset sdv1_500
"""
import argparse
import csv
import os


# method → trd/ 부모 경로 (base_dir 기준 상대 — 4차 레이아웃. jeon만 batch=1 run)
METHOD_TRD = {
    "jeon": "jeon/stable-diffusion-v1-4/CFG=7.5_NFE=50/thres=8.2/ms=10/batch=1/seed=42",
    "ren": "ren/stable-diffusion-v1-4/CFG=7.5_NFE=50/batch=4/seed=42",
    "wen": "wen/stable-diffusion-v1-4/CFG=7.5_NFE=50/it=10/batch=4/seed=42",
    "init_score_noise": "init_score_noise/NFE=50/per_sample/CFG=7.0/lr=0.01/oi=1000/batch=4/seed=42",
}
# ddim (완화 없음 기준선) — 단일 점, trd 없음 → eval/<subset>/total_metrics.csv
DDIM_EVAL = "ddim/stable-diffusion-v1-4/CFG=7.5_NFE=50/batch=1/seed=42/eval"


def load_method_trd(csv_path):
    """trd csv (label,CLIP_mean,Pick_mean,ImgR_mean,SSCD_mean) → rows."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "label": r["label"],
                "clipscore": r["CLIP_mean"],
                "pickscore": r["Pick_mean"],
                "imagereward": r["ImgR_mean"],
                "sscd": r["SSCD_mean"],
            })
    return rows


def load_ddim_point(csv_path):
    """ddim eval total_metrics.csv (metric,mean) → 1행."""
    m = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            m[r["metric"]] = r["mean"]
    return [{
        "label": "0.0",
        "clipscore": m["CLIPScore"],
        "pickscore": m["PickScore"],
        "imagereward": m["ImageReward"],
        "sscd": m["SSCD-to-GT (mean)"],
    }]


def main():
    ap = argparse.ArgumentParser(description="baselines trd/<subset> 병합 → 전체 비교 trd")
    ap.add_argument("--base_dir", required=True, help="baselines/ 루트")
    ap.add_argument("--subset", default="sdv1_500", help="trd 하위 프롬pt 세트 (기본 sdv1_500)")
    args = ap.parse_args()

    out_dir = os.path.join(args.base_dir, "trd", args.subset)
    os.makedirs(os.path.join(out_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plot"), exist_ok=True)

    merged = []
    for method, rel in METHOD_TRD.items():
        csv_path = os.path.join(args.base_dir, rel, "trd", args.subset, "csv", "total_metrics.csv")
        if not os.path.isfile(csv_path):
            print(f"[skip] {method}: trd csv 없음 — {csv_path}")
            continue
        rows = load_method_trd(csv_path)
        for r in rows:
            merged.append({"method": method, **r})
        print(f"[ok] {method}: {len(rows)} rows  ({rel}/trd/{args.subset})")

    ddim_csv = os.path.join(args.base_dir, DDIM_EVAL, args.subset, "total_metrics.csv")
    if os.path.isfile(ddim_csv):
        rows = load_ddim_point(ddim_csv)
        for r in rows:
            merged.append({"method": "ddim", **r})
        print(f"[ok] ddim: {len(rows)} row (단일 점, {DDIM_EVAL}/{args.subset})")
    else:
        print(f"[skip] ddim: eval csv 없음 — {ddim_csv}")

    out_csv = os.path.join(out_dir, "csv", "total_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "lr", "clipscore", "pickscore",
                                          "imagereward", "sscd"])
        w.writeheader()
        for r in merged:
            w.writerow({"method": r["method"], "lr": r["label"], "clipscore": r["clipscore"],
                        "pickscore": r["pickscore"], "imagereward": r["imagereward"],
                        "sscd": r["sscd"]})
    print(f"\n[Done] {len(merged)} rows → {out_csv}")
    print(f"       plot: python plot_tradeoff.py --csv {out_csv} "
          f"--methods ddim,init_score_noise,jeon,ren,wen --x_metric <m> --out {out_dir}/plot/...")


if __name__ == "__main__":
    main()
