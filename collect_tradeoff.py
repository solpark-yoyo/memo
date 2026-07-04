"""lr sweep eval 결과 수집 → (method, lr, SSCD, CLIP, Pick, IR) tradeoff.csv.

base_dir 아래 모든 eval 디렉토리(sscd_gt_metrics.csv + t2i_metrics.csv)를 순회해서
각 lr 설정의 (SSCD, CLIPscore, PickScore, ImageReward) 쌍을 모은다.

두 가지 디렉토리 명명 규칙을 모두 지원:
  - 슬래시형:  .../lr=0.04/seed=42/...           (run_memo.sh init_opti)
  - 언더스코어형: .../init=10_nsteps=4_gap=3_lr=0.04/seed=42/...  (과거 실행)
또한 lr 컬럼/폴더명은 소수점 2자리로 정규화(0.0 → 0.00)한다.

--split_lr 을 주면 --out 과 같은 디렉토리 아래 lr=<value>/tradeoff.csv 로
lr별 분할 파일을 추가로 생성한다.

Usage:
    python collect_tradeoff.py \\
        --base_dir workdir/memorization/sd14_base \\
        --out tradeoff.csv --split_lr
"""
import argparse
import csv
import os
import glob
from collections import defaultdict


FIELDNAMES = ["method", "lr", "sscd", "clipscore", "pickscore", "imagereward", "eval_dir"]


def norm_lr(lr):
    """lr 문자열 정규화: '0.0' → '0.00' (소수점 2자리). 빈 값은 그대로."""
    if lr == "" or lr is None:
        return ""
    try:
        return f"{float(lr):.2f}"
    except ValueError:
        return lr


def read_sscd(path):
    """sscd_gt_metrics.csv 에서 mean_sscd_to_gt_avg 추출."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("mean_sscd_to_gt_avg"):
                try:
                    return float(line.split(",")[1])
                except (IndexError, ValueError):
                    return None
    return None


def read_t2i(path):
    """t2i_metrics.csv 에서 CLIPScore/PickScore/ImageReward mean 추출."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[0] in ("CLIPScore", "PickScore", "ImageReward"):
                try:
                    out[row[0]] = float(row[1])
                except ValueError:
                    pass
    return out


def parse_meta(eval_dir):
    """경로에서 method/lr 추정. 두 명명 규칙 모두 지원.

    슬래시형:    .../sd14_base/init_opti/.../lr=0.04/seed=42/batch=5/eval
    언더스코어형: .../init=10_nsteps=4_gap=3_lr=0.04/seed=42/batch=5/eval
    """
    parts = eval_dir.replace("\\", "/").split("/")
    lr = ""
    # 1) 슬래시형: 토큰 자체가 "lr=0.04"
    for p in parts:
        if p.startswith("lr="):
            lr = p[3:]
    # 2) 언더스코어형: "init=10_nsteps=4_gap=3_lr=0.04" 토큰 안의 "_lr="
    if not lr:
        for p in parts:
            if "_lr=" in p:
                lr = p.split("_lr=")[-1].split("_")[0]
                break
    method = ""
    for m in ("init_opti", "cno_infoNCE", "ddim", "init_score_noise"):
        if m in parts:
            method = m
            break
    return method, norm_lr(lr)


def main():
    ap = argparse.ArgumentParser(description="lr sweep trade-off 쌍 수집")
    ap.add_argument("--base_dir", required=True, help="예: workdir/memorization/sd14_base")
    ap.add_argument("--out", default="tradeoff.csv")
    ap.add_argument("--split_lr", action="store_true",
                    help="같은 디렉토리 아래 lr=<value>/tradeoff.csv 로 lr별 분할 파일 추가 생성")
    ap.add_argument("--keep_empty", action="store_true",
                    help="sscd/metric 값이 비어있는 row도 유지 (기본은 제외)")
    ap.add_argument("--include_substr", default="",
                    help="eval_dir 경로에 이 부분문자열이 포함된 것만 수집 (예: 'init=10_nsteps=4')")
    ap.add_argument("--eval_only", action="store_true",
                    help="eval_dir 이 /eval 로 끝나는 것만 수집 (과거 실행 잔재인 batch=N/ 직하 파일 제외)")
    ap.add_argument("--lr_whitelist", default="",
                    help="이 lr(공백/콤마 구분, 소수점 2자리 정규화)만 수집. 예: '0.00 0.02 0.04 0.06'")
    args = ap.parse_args()

    wl = set()
    if args.lr_whitelist:
        wl = {norm_lr(x) for x in args.lr_whitelist.replace(",", " ").split() if x}

    # chen_ 접두 + 일반 모두 수집
    sscd_paths = set(
        glob.glob(os.path.join(args.base_dir, "**", "chen_sscd_gt_metrics.csv"), recursive=True)
        + glob.glob(os.path.join(args.base_dir, "**", "sscd_gt_metrics.csv"), recursive=True)
    )

    rows = []
    skipped_empty = 0
    for sscd_path in sorted(sscd_paths):
        eval_dir = os.path.dirname(sscd_path)
        if args.include_substr and args.include_substr not in eval_dir:
            continue
        if args.eval_only and not eval_dir.endswith("eval"):
            continue
        sscd = read_sscd(sscd_path)
        # 대응 t2i
        t2i_path = None
        for cand in ("chen_t2i_metrics.csv", "t2i_metrics.csv"):
            p = os.path.join(eval_dir, cand)
            if os.path.exists(p):
                t2i_path = p
                break
        t2i = read_t2i(t2i_path) if t2i_path else {}
        method, lr = parse_meta(eval_dir)
        if wl and lr not in wl:
            continue

        # 값이 전부 비어있는 row(skip) — sscd 없거나 t2i metric 전부 없으면 의미 없음
        has_value = sscd is not None or bool(t2i)
        if not has_value and not args.keep_empty:
            skipped_empty += 1
            continue

        rows.append({
            "method": method,
            "lr": lr,
            "sscd": f"{sscd:.6f}" if sscd is not None else "",
            "clipscore": f"{t2i['CLIPScore']:.6f}" if "CLIPScore" in t2i else "",
            "pickscore": f"{t2i['PickScore']:.6f}" if "PickScore" in t2i else "",
            "imagereward": f"{t2i['ImageReward']:.6f}" if "ImageReward" in t2i else "",
            "eval_dir": eval_dir,
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    print(f"[Done] {len(rows)} rows → {args.out}"
          + (f"  (빈 row {skipped_empty}개 제외)" if skipped_empty else ""))

    # lr별 분할 파일 생성
    if args.split_lr:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        out_name = os.path.basename(args.out)
        by_lr = defaultdict(list)
        for r in rows:
            if r["lr"]:
                by_lr[r["lr"]].append(r)
        for lr_val in sorted(by_lr.keys(), key=lambda x: float(x)):
            lr_rows = by_lr[lr_val]
            lr_dir = os.path.join(out_dir, f"lr={lr_val}")
            os.makedirs(lr_dir, exist_ok=True)
            lr_out = os.path.join(lr_dir, out_name)
            with open(lr_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDNAMES)
                w.writeheader()
                w.writerows(lr_rows)
            print(f"  [split_lr] lr={lr_val} → {lr_out} ({len(lr_rows)} rows)")

    for r in rows:
        print(f"  {r['method']:14s} lr={r['lr']:5s}  sscd={r['sscd']:>8s}  "
              f"clip={r['clipscore']:>8s}  pick={r['pickscore']:>8s}  ir={r['imagereward']:>8s}")


if __name__ == "__main__":
    main()
