"""
compute_sscd_gt.py — SSCD-to-GT: 생성 이미지 vs MemBench reference 원본 간 SSCD similarity 측정

Usage:
  python compute_sscd_gt.py \
      --gen_dir workdir/memorization/sd15/ddim/CFG=7.5_NFE=50/seed=42/result \
      --ref_dir datasets/membench_ref \
      --num_prompts 20 --num_images_per_prompt 5 \
      --gpu 0 \
      --output_csv workdir/.../sscd_gt_metrics.csv
"""
import os
import sys
import csv
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description="SSCD-to-GT: generated images vs MemBench reference")
    parser.add_argument("--gen_dir", required=True, help="생성 이미지 폴더")
    parser.add_argument("--ref_dir", default="datasets/membench_ref", help="reference 이미지 폴더")
    parser.add_argument("--mapping_csv", default=None,
                        help="prompt→ref 매핑 csv 경로 (기본: {ref_dir}/prompt_to_ref.csv). "
                             "생성 순서 ≠ GT 순서일 때 캡션 브리지로 만든 매핑을 전달")
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--num_images_per_prompt", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    import image_utils

    # reference prompt → file 매핑 로드 (브리지 매핑 우선)
    mapping_path = args.mapping_csv or os.path.join(args.ref_dir, "prompt_to_ref.csv")
    prompt_to_ref = {}
    with open(mapping_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ref_file"] != "MISSING":
                prompt_to_ref[int(row["prompt_idx"])] = row["ref_file"]
    print(f"[INFO] loaded {len(prompt_to_ref)} reference image mappings")

    # SSCD 모델
    sscd = image_utils.get_sscd().to(device)
    sscd_transform = image_utils.sscd_transforms()

    def get_sscd_embedding(path):
        img = Image.open(path).convert("RGB")
        x = sscd_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = sscd(x)
        return emb.squeeze(0)

    # 생성 이미지 로드
    gen_dir = args.gen_dir
    exts = (".png", ".jpg", ".jpeg")
    gen_files = sorted([f for f in os.listdir(gen_dir) if f.lower().endswith(exts)])
    npp = args.num_images_per_prompt

    results = []
    for pidx in range(args.num_prompts):
        if pidx not in prompt_to_ref:
            continue
        ref_path = os.path.join(args.ref_dir, prompt_to_ref[pidx])
        if not os.path.exists(ref_path):
            continue

        # 다중 GT (디렉토리 ref): gt_images/{LAION index}/ 폴더에서
        # 임의(정렬 첫 번째) 한 장의 SSCD feature를 대표 ref로 사용 (2026-09-01 프로토콜)
        if os.path.isdir(ref_path):
            ref_files = sorted(f for f in os.listdir(ref_path)
                               if f.lower().endswith((".jpg", ".jpeg", ".png")))
            if not ref_files:
                continue
            ref_path = os.path.join(ref_path, ref_files[0])

        try:
            ref_emb = get_sscd_embedding(ref_path)
        except Exception as e:
            # 손상/비이미지 GT 파일 — 무시하고 계속 (cvpr2025_webster_gt 등)
            print(f"[warn] skip corrupt ref (idx {pidx}): {ref_path} — {type(e).__name__}")
            continue

        sims = []
        for j in range(npp):
            gen_idx = pidx * npp + j
            if gen_idx >= len(gen_files):
                break
            gen_path = os.path.join(gen_dir, gen_files[gen_idx])
            try:
                gen_emb = get_sscd_embedding(gen_path)
            except Exception as e:
                print(f"[warn] skip corrupt gen ({os.path.basename(gen_path)}): {type(e).__name__}")
                continue
            sim = torch.nn.functional.cosine_similarity(
                ref_emb.unsqueeze(0), gen_emb.unsqueeze(0)
            ).item()
            sims.append(sim)

        if sims:
            mean_sim = sum(sims) / len(sims)
            max_sim = max(sims)
            results.append({
                "prompt_idx": pidx,
                "num_gen": len(sims),
                "mean_sscd_to_gt": f"{mean_sim:.6f}",
                "max_sscd_to_gt": f"{max_sim:.6f}",
                "sims": ";".join(f"{s:.6f}" for s in sims),
            })

    # CSV 저장
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_idx", "num_gen", "mean_sscd_to_gt", "max_sscd_to_gt", "sims"])
        writer.writeheader()
        writer.writerows(results)

        if results:
            mean_vals = [float(r["mean_sscd_to_gt"]) for r in results]
            max_vals = [float(r["max_sscd_to_gt"]) for r in results]
            f.write("\n")
            f.write(f"# Summary\n")
            f.write(f"num_prompts_evaluated,{len(results)}\n")
            f.write(f"mean_sscd_to_gt_avg,{sum(mean_vals)/len(mean_vals):.6f}\n")
            f.write(f"max_sscd_to_gt_avg,{sum(max_vals)/len(max_vals):.6f}\n")
            f.write(f"memorized_rate(mean>0.5),{sum(1 for v in mean_vals if v > 0.5)}/{len(mean_vals)}\n")

    print(f"[Done] {len(results)} prompts evaluated → {args.output_csv}")
    if results:
        mean_vals = [float(r["mean_sscd_to_gt"]) for r in results]
        print(f"  mean SSCD-to-GT: {sum(mean_vals)/len(mean_vals):.4f}")
        print(f"  memorized (mean>0.5): {sum(1 for v in mean_vals if v > 0.5)}/{len(mean_vals)}")


if __name__ == "__main__":
    main()
