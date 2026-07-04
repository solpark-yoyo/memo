"""Local memorization metric — Chen et al. CVPR 2025 (Eq.6) 기반.

Global memorization(SSCD>0.5)인 경우에만, BE attention mask m 영역의
pixel-level Euclidean distance 를 측정 → local memorization 정도.

  local_metric = 𝟙_{SSCD_global > τ} · ‖(x̂ − x) ∘ m‖₂

  x̂ : 생성 이미지 (image space, [3,H,W], 0~1)
  x  : training image (GT, [3,H,W], 0~1)
  m  : BE attention mask ([H,W], binary) — memorized local region
  τ  : global SSCD threshold (기본 0.5)

Usage (inference 후 별도):
    from compute_local_mem import local_mem_metric
    val = local_mem_metric(gen_img, gt_img, mask, sscd_global=0.7)

또는 batch eval:
    python compute_local_mem.py --gen_dir ... --ref_dir datasets/cvpr2025_webster_gt \\
        --mask_dir ... --global_sscd_csv .../chen_sscd_gt_metrics.csv
"""
import os
import csv
import argparse
import numpy as np
import torch
from PIL import Image


def _load_img_tensor(path, size=512):
    """이미지 → [3, size, size] float tensor (0~1)."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return t


def local_mem_metric(gen_img, gt_img, mask, sscd_global, threshold=0.5):
    """Eq.6 local memorization metric.

    Args:
        gen_img: [3,H,W] tensor (0~1) 또는 path
        gt_img:  [3,H,W] tensor (0~1) 또는 path
        mask:    [H,W] tensor (binary) 또는 path
        sscd_global: float (global SSCD cosine similarity)
        threshold: global SSCD 임계값 (기본 0.5)
    Returns:
        local_metric: float (0 if global SSCD <= threshold)
    """
    # ① indicator: global memorization 일 때만 local 측정
    if sscd_global <= threshold:
        return 0.0  # 𝟙_{SSCD>τ} = 0

    # ② tensor 로드/정규화
    if isinstance(gen_img, str):
        gen_img = _load_img_tensor(gen_img)
    if isinstance(gt_img, str):
        gt_img = _load_img_tensor(gt_img)
    if isinstance(mask, str):
        mask = np.load(mask) if mask.endswith('.npy') else \
               np.array(Image.open(mask).convert("L").resize(gen_img.shape[-2:], Image.NEAREST))
        mask = torch.from_numpy((mask > 127).astype(np.float32))

    # shape 맞춤
    H, W = gen_img.shape[-2:]
    if mask.shape != (H, W):
        mask = torch.nn.functional.interpolate(
            mask[None, None], size=(H, W), mode='nearest'
        )[0, 0]

    # ③ pixel distance 에 mask 적용: ‖(x̂ − x) ∘ m‖₂
    diff = (gen_img - gt_img) * mask  # [3,H,W], mask broadcast
    pixel_dist = diff.norm(p=2).item()
    return pixel_dist


def batch_local_eval(gen_dir, ref_dir, mask_dir, global_sscd_csv,
                     output_csv, num_prompts=500, num_images_per_prompt=5,
                     threshold=0.5, size=512):
    """batch local memorization 평가.

    Args:
        gen_dir: 생성 이미지 폴더 (img_{pid:04d}_{sid:02d}.png)
        ref_dir: GT 폴더 (prompt_to_ref.csv + {ref}.jpg)
        mask_dir: BE mask 폴더 (mask_{pid:04d}.npy 또는 .png)
        global_sscd_csv: global SSCD 결과 (chen_sscd_gt_metrics.csv)
        output_csv: 결과 저장
    """
    # ① global SSCD per prompt 로드
    prompt_sscd = {}
    with open(global_sscd_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("prompt_idx")
            if pid is None or pid == "":
                continue
            try:
                prompt_sscd[int(pid)] = float(row.get("mean_sscd_to_gt", 0))
            except (ValueError, KeyError):
                pass

    # ② prompt_to_ref 매핑
    prompt_to_ref = {}
    mapping = os.path.join(ref_dir, "prompt_to_ref.csv")
    if os.path.exists(mapping):
        with open(mapping) as f:
            for row in csv.DictReader(f):
                if row["ref_file"] != "MISSING":
                    prompt_to_ref[int(row["prompt_idx"])] = row["ref_file"]

    # ③ batch 계산
    results = []
    for pid in range(num_prompts):
        if pid not in prompt_sscd:
            continue
        sscd_g = prompt_sscd[pid]
        ref_path = prompt_to_ref.get(pid)
        if ref_path is None:
            continue
        ref_full = os.path.join(ref_dir, ref_path)
        if not os.path.exists(ref_full):
            continue

        # mask 경로 (npy 또는 png)
        mask_npy = os.path.join(mask_dir, f"mask_{pid:04d}.npy")
        mask_png = os.path.join(mask_dir, f"mask_{pid:04d}.png")
        mask_path = mask_npy if os.path.exists(mask_npy) else \
                    mask_png if os.path.exists(mask_png) else None
        if mask_path is None:
            continue

        # 각 생성 이미지에 대해 local metric
        local_vals = []
        for j in range(num_images_per_prompt):
            gen_path = os.path.join(gen_dir, f"img_{pid:04d}_{j:02d}.png")
            if not os.path.exists(gen_path):
                continue
            val = local_mem_metric(gen_path, ref_full, mask_path,
                                   sscd_global=sscd_g, threshold=threshold, size=size)
            local_vals.append(val)
        if local_vals:
            results.append({
                "prompt_idx": pid,
                "global_sscd": f"{sscd_g:.6f}",
                "memorized_global": int(sscd_g > threshold),
                "local_pixel_dist_mean": f"{sum(local_vals)/len(local_vals):.4f}",
                "local_pixel_dist_max": f"{max(local_vals):.4f}",
                "num_gen": len(local_vals),
            })

    # ④ 저장
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "prompt_idx", "global_sscd", "memorized_global",
            "local_pixel_dist_mean", "local_pixel_dist_max", "num_gen"
        ])
        w.writeheader()
        w.writerows(results)

        if results:
            vals = [float(r["local_pixel_dist_mean"]) for r in results]
            mem_g = [int(r["memorized_global"]) for r in results]
            f.write("\n# Summary\n")
            f.write(f"num_prompts_evaluated,{len(results)}\n")
            f.write(f"num_global_memorized,{sum(mem_g)}\n")
            f.write(f"local_pixel_dist_avg,{sum(vals)/len(vals):.4f}\n")
    print(f"[Done] {len(results)} prompts → {output_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Local memorization metric (Chen Eq.6)")
    ap.add_argument("--gen_dir", required=True, help="생성 이미지 폴더")
    ap.add_argument("--ref_dir", required=True, help="GT 폴더 (prompt_to_ref.csv 포함)")
    ap.add_argument("--mask_dir", required=True, help="BE mask 폴더 (mask_{pid:04d}.npy)")
    ap.add_argument("--global_sscd_csv", required=True, help="global SSCD 결과 csv")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--num_prompts", type=int, default=500)
    ap.add_argument("--num_images_per_prompt", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    batch_local_eval(args.gen_dir, args.ref_dir, args.mask_dir, args.global_sscd_csv,
                     args.output_csv, args.num_prompts, args.num_images_per_prompt,
                     args.threshold, args.size)
