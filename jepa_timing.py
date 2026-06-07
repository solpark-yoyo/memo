#!/usr/bin/env python3
"""
JEPA Jacobian SVD Timing Experiment
Compares Full-SVD vs RSVD computation time per sample.

Per-sample breakdown:
  time_jacobian_sec : Jacobian 계산 시간 (Full/RSVD 공통)
  time_svd_sec      : SVD 계산 시간 (Full vs RSVD 비교 핵심)
  time_total_sec    : 전체 시간 (= jacobian + svd)
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"

_CKPT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")
_DINOV2_HUB   = os.path.join(_CKPT_DIR, "jepa_model", "dinov2", "facebookresearch_dinov2_main")
_DINOV2_CKPTS = os.path.join(_CKPT_DIR, "jepa_model", "dinov2", "checkpoints")
_DINOV2_PTH   = {
    "dinov2_vits14": "dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14": "dinov2_vitb14_reg4_pretrain.pth",
    "dinov2_vitl14": "dinov2_vitl14_reg4_pretrain.pth",
}

import glob
import time
import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.hub as hub
import csv
from torch.autograd.functional import jacobian


# ============================================================
# Backbone loader (DINOv2 only — timing experiment)
# ============================================================
def load_backbone(backbone_name: str, device: torch.device):
    bname = backbone_name.lower()
    if bname not in _DINOV2_PTH:
        raise ValueError(f"Timing experiment supports only DINOv2 backbones: {list(_DINOV2_PTH)}")
    _pth = os.path.join(_DINOV2_CKPTS, _DINOV2_PTH[bname])
    print(f"[CKPT] Loading {bname}_reg  ← {_pth}")
    model = hub.load(_DINOV2_HUB, f"{bname}_reg", source='local', weights=_pth).to(device).eval()
    return model, model.embed_dim


# ============================================================
# Image loader
# ============================================================
def load_image_normalized(path: str, mean: torch.Tensor, std: torch.Tensor,
                          img_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return (x - mean) / std


# ============================================================
# SVD methods
# ============================================================
def svd_full(J: torch.Tensor, eps: float = 1e-6):
    """Full SVD: torch.linalg.svdvals (exact, all singular values)."""
    svdvals = torch.linalg.svdvals(J)          # (B, min(D,N))
    return svdvals.clip_(eps).log_().sum(1)    # (B,)


def svd_rsvd(J: torch.Tensor, topk: int, pi_q: int, oversample: int, eps: float = 1e-6):
    """Randomized SVD: power iteration + QR, top-k singular values."""
    B, D, N = J.shape
    r = topk + oversample
    Omega = torch.randn(B, N, r, device=J.device, dtype=J.dtype)
    Y = J @ Omega                               # (B, D, r)
    for _ in range(pi_q):
        Q, _ = torch.linalg.qr(Y)
        Z = J.transpose(-2, -1) @ Q            # (B, N, r)
        Q2, _ = torch.linalg.qr(Z)
        Y = J @ Q2
    Q, _ = torch.linalg.qr(Y)                  # (B, D, r)
    B_small = J.transpose(-2, -1) @ Q          # (B, N, r)
    _, sigmas, _ = torch.linalg.svd(B_small, full_matrices=False)
    return sigmas[:, :topk].clip_(eps).log_().sum(1)  # (B,)


# ============================================================
# Per-image timed scoring
# Returns (jepa_score, t_jacobian, t_svd)
# ============================================================
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def score_fullsvd(backbone, x, eps):
    # --- Jacobian ---
    _sync(); t0 = time.perf_counter()
    J = jacobian(lambda inp: backbone(inp).sum(0), inputs=x)
    _sync(); t1 = time.perf_counter()
    t_jac = t1 - t0

    # --- Full SVD ---
    with torch.inference_mode():
        J_flat = J.flatten(2).permute(1, 0, 2)   # (B, D, N)
    _sync(); t2 = time.perf_counter()
    with torch.inference_mode():
        score = svd_full(J_flat, eps)
    _sync(); t3 = time.perf_counter()
    t_svd = t3 - t2

    return score.item(), t_jac, t_svd


def score_rsvd(backbone, x, topk, pi_q, oversample, eps):
    # --- Jacobian ---
    _sync(); t0 = time.perf_counter()
    J = jacobian(lambda inp: backbone(inp).sum(0), inputs=x)
    _sync(); t1 = time.perf_counter()
    t_jac = t1 - t0

    # --- RSVD ---
    with torch.inference_mode():
        J_flat = J.flatten(2).permute(1, 0, 2)   # (B, D, N)
    _sync(); t2 = time.perf_counter()
    with torch.inference_mode():
        score = svd_rsvd(J_flat, topk, pi_q, oversample, eps)
    _sync(); t3 = time.perf_counter()
    t_svd = t3 - t2

    return score.item(), t_jac, t_svd


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="JEPA SVD timing: Full-SVD vs RSVD per sample")
    parser.add_argument("--img_root",      required=True,
                        help="Directory containing generated images")
    parser.add_argument("--backbone",      type=str, default="dinov2_vits14",
                        choices=list(_DINOV2_PTH.keys()))
    parser.add_argument("--img_size",      type=int, default=224)
    parser.add_argument("--svd_mode",      type=str, required=True,
                        choices=["full", "rsvd"],
                        help="full: torch.linalg.svdvals / rsvd: randomized SVD")
    # RSVD params (used only when --svd_mode rsvd)
    parser.add_argument("--rsvd_topk",     type=int, default=9)
    parser.add_argument("--rsvd_pi_q",     type=int, default=2)
    parser.add_argument("--rsvd_oversample", type=int, default=2)
    # Sampling
    parser.add_argument("--max_images",    type=int, default=50)
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--eps",           type=float, default=1e-6)
    parser.add_argument("--device",        default="cuda")
    # Output
    parser.add_argument("--time_csv",      type=str, required=True,
                        help="Path to save per-sample timing CSV")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    backbone, embed_dim = load_backbone(args.backbone, device)
    print(f"[INFO] backbone={args.backbone}  embed_dim={embed_dim}  "
          f"img_size={args.img_size}  svd_mode={args.svd_mode}")
    if args.svd_mode == "rsvd":
        print(f"[INFO] rsvd_topk={args.rsvd_topk}  pi_q={args.rsvd_pi_q}  "
              f"oversample={args.rsvd_oversample}")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Image paths
    img_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        img_paths += glob.glob(os.path.join(args.img_root, "**", ext), recursive=True)
        img_paths += glob.glob(os.path.join(args.img_root, ext))
    img_paths = sorted(set(img_paths))

    if args.max_images > 0:
        if args.random_sample:
            random.seed(args.seed)
            img_paths = random.sample(img_paths, min(args.max_images, len(img_paths)))
        else:
            img_paths = img_paths[:args.max_images]

    if not img_paths:
        raise RuntimeError(f"No images found in {args.img_root}")

    os.makedirs(os.path.dirname(os.path.abspath(args.time_csv)), exist_ok=True)

    rows_jac, rows_svd, rows_total, scores = [], [], [], []

    with open(args.time_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "time_jacobian_sec",
            f"time_svd_{args.svd_mode}_sec",
            "time_total_sec",
            "jepa_score",
        ])

        desc = f"Timing ({args.svd_mode.upper()} SVD, {args.img_size}x{args.img_size})"
        for p in tqdm(img_paths, desc=desc):
            x = load_image_normalized(p, mean.cpu(), std.cpu(), args.img_size).to(device)

            if args.svd_mode == "full":
                score_val, t_jac, t_svd = score_fullsvd(backbone, x, args.eps)
            else:
                score_val, t_jac, t_svd = score_rsvd(
                    backbone, x,
                    args.rsvd_topk, args.rsvd_pi_q, args.rsvd_oversample,
                    args.eps,
                )

            t_total = t_jac + t_svd
            rows_jac.append(t_jac)
            rows_svd.append(t_svd)
            rows_total.append(t_total)
            scores.append(score_val)

            writer.writerow([
                os.path.basename(p),
                f"{t_jac:.6f}",
                f"{t_svd:.6f}",
                f"{t_total:.6f}",
                f"{score_val:.10f}",
            ])
            del x

        # Summary
        n = len(img_paths)
        writer.writerow([])
        writer.writerow(["# Summary"])
        writer.writerow(["num_samples",              n])
        writer.writerow(["svd_mode",                 args.svd_mode])
        if args.svd_mode == "rsvd":
            writer.writerow(["rsvd_topk",            args.rsvd_topk])
            writer.writerow(["rsvd_pi_q",            args.rsvd_pi_q])
            writer.writerow(["rsvd_oversample",      args.rsvd_oversample])
        writer.writerow([])
        writer.writerow(["metric",
                         "total_sec",
                         "mean_per_sample_sec",
                         "std_per_sample_sec"])
        writer.writerow(["time_jacobian",
                         f"{sum(rows_jac):.4f}",
                         f"{np.mean(rows_jac):.6f}",
                         f"{np.std(rows_jac):.6f}"])
        writer.writerow([f"time_svd_{args.svd_mode}",
                         f"{sum(rows_svd):.4f}",
                         f"{np.mean(rows_svd):.6f}",
                         f"{np.std(rows_svd):.6f}"])
        writer.writerow(["time_total",
                         f"{sum(rows_total):.4f}",
                         f"{np.mean(rows_total):.6f}",
                         f"{np.std(rows_total):.6f}"])
        writer.writerow(["jepa_score",
                         "",
                         f"{np.mean(scores):.10f}",
                         f"{np.std(scores):.10f}"])

    print(f"\n[Done] Saved timing for {n} samples → {args.time_csv}")
    print("=" * 60)
    print(f"  SVD mode          : {args.svd_mode.upper()}")
    print(f"  Jacobian  total   : {sum(rows_jac):.3f}s  "
          f"| mean/sample : {np.mean(rows_jac)*1e3:.2f} ms")
    print(f"  SVD       total   : {sum(rows_svd):.3f}s  "
          f"| mean/sample : {np.mean(rows_svd)*1e3:.2f} ms")
    print(f"  Total     total   : {sum(rows_total):.3f}s  "
          f"| mean/sample : {np.mean(rows_total)*1e3:.2f} ms")
    print(f"  JEPA Score        : {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
