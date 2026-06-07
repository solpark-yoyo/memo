#!/usr/bin/env python3
import os
os.environ["XFORMERS_DISABLED"] = "1"  # Disable xformers to avoid inplace op issues with jacobian

_CKPT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")
_DINOV2_HUB   = os.path.join(_CKPT_DIR, "jepa_model", "dinov2", "facebookresearch_dinov2_main")
_DINOV2_CKPTS = os.path.join(_CKPT_DIR, "jepa_model", "dinov2", "checkpoints")
_DINOV2_PTH   = {
    "dinov2_vits14": "dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14": "dinov2_vitb14_reg4_pretrain.pth",
    "dinov2_vitl14": "dinov2_vitl14_reg4_pretrain.pth",
}
_DINOV3_DIR   = os.path.join(_CKPT_DIR, "jepa_model", "dinov3")
_DINOV3_VARIANTS = {
    "dinov3_vits16": "dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "dinov3-vitl16-pretrain-lvd1689m",
}

import glob
import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub as hub
import csv
from torch.autograd.functional import jacobian


# ============================================================
# Backbone loader
# Returns a callable: f(x: Tensor) -> Tensor (B, D)
# ============================================================
def load_backbone(backbone_name: str, device: torch.device):
    """
    Load JEPA backbone and return (callable, embed_dim).
    callable: (x: (B,3,H,W) normalized) -> (B, D)
    """
    bname = backbone_name.lower()

    if bname in _DINOV2_PTH:
        # ── DINOv2 via torch.hub (local) ──────────────────────────────
        _pth = os.path.join(_DINOV2_CKPTS, _DINOV2_PTH[bname])
        print(f"[CKPT] Loading DINOv2 {bname}_reg  ← {_pth}")
        model = hub.load(
            _DINOV2_HUB, f"{bname}_reg",
            source='local', weights=_pth,
        ).to(device).eval()
        # torch.hub DINOv2: model(x) -> (B, D)
        return model, model.embed_dim

    elif bname in _DINOV3_VARIANTS:
        # ── DINOv3 via HuggingFace (local) ────────────────────────────
        from transformers import AutoModel
        _local = os.path.join(_DINOV3_DIR, _DINOV3_VARIANTS[bname])
        print(f"[CKPT] Loading DINOv3 {bname}  ← {_local}")
        hf_model = AutoModel.from_pretrained(_local, local_files_only=True).to(device).eval()
        embed_dim = hf_model.config.hidden_size

        # Wrap HF output so jepa_score gets a plain tensor (B, D)
        class _DINOv3Callable(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                return self.m(pixel_values=x).last_hidden_state[:, 0]  # CLS token

        return _DINOv3Callable(hf_model).to(device).eval(), embed_dim

    else:
        raise ValueError(
            f"Unknown backbone: {backbone_name}. "
            f"Choose from: {list(_DINOV2_PTH.keys()) + list(_DINOV3_VARIANTS.keys())}"
        )


# ============================================================
# Image loader — resize to img_size x img_size, then normalize
# ============================================================
def load_image_normalized(path: str, mean: torch.Tensor, std: torch.Tensor,
                          img_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    return x


# ============================================================
# JEPA Score (Jacobian singular value sum)
# backbone: callable (B,3,H,W) -> (B,D)
# images:   (B, 3, H, W) normalized
# ============================================================
def jepa_score(backbone, images, eps=1e-6):
    J = jacobian(lambda x: backbone(x).sum(0), inputs=images)
    with torch.inference_mode():
        J = J.flatten(2).permute(1, 0, 2)  # (B, D, N)
        svdvals = torch.linalg.svdvals(J)
        score = svdvals.clip_(eps).log_().sum(1)  # (B,)
    return score


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root",     default="/media/usb_media/datasets/celeba/all/examples")
    parser.add_argument("--eps",          type=float, default=1e-6)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--max_images",   type=int, default=0)
    parser.add_argument("--random_sample", action="store_true",
                        help="Randomly sample images instead of taking first N")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--output_csv",   type=str, default=None,
                        help="Output CSV path (default: metrics/<basedir>.csv)")
    parser.add_argument("--log_csv",      type=str, default=None,
                        help="Path to save detailed log CSV")
    # ── backbone ─────────────────────────────────────────────────────────
    # [기존] dinov2_vits14 / vitb14 / vitl14
    # [신규] dinov3_vits16 / vitb16 / vitl16  (HuggingFace AutoModel, patch=16)
    parser.add_argument("--backbone",     type=str, default="dinov2_vits14",
                        choices=[
                            "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
                            "dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16",
                        ],
                        help="JEPA backbone. DINOv2: torch.hub (patch=14). "
                             "DINOv3: HuggingFace local (patch=16).")
    # ── img_size (신규) ───────────────────────────────────────────────────
    # DINOv2: 14의 배수 (기본 224=16*14)
    # DINOv3: 16의 배수 (기본 224=14*16)
    # 두 모델 모두 224 사용 가능하므로 기본값 동일
    parser.add_argument("--img_size",     type=int, default=224,
                        help="Input resolution for backbone. "
                             "DINOv2: multiple of 14 (e.g. 56, 112, 224). "
                             "DINOv3: multiple of 16 (e.g. 64, 128, 224). "
                             "Default: 224.")
    parser.add_argument("--warn_log",     type=str, default=None,
                        help="Redirect Python warnings to this file")
    args = parser.parse_args()

    if args.warn_log:
        import logging
        os.makedirs(os.path.dirname(os.path.abspath(args.warn_log)), exist_ok=True)
        logging.captureWarnings(True)
        _wh = logging.FileHandler(args.warn_log, mode='a')
        logging.getLogger('py.warnings').addHandler(_wh)

    # patch size 검증
    bname = args.backbone.lower()
    patch = 16 if 'dinov3' in bname else 14
    if args.img_size % patch != 0:
        raise ValueError(
            f"--img_size {args.img_size} must be a multiple of {patch} "
            f"(patch size for {args.backbone})."
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # Load backbone
    backbone, embed_dim = load_backbone(args.backbone, device)
    print(f"[INFO] backbone={args.backbone}  embed_dim={embed_dim}  img_size={args.img_size}")

    # ImageNet normalization (공통)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Image list
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

    if len(img_paths) == 0:
        raise RuntimeError("No images found")

    # CSV 경로
    if args.output_csv:
        csv_path = args.output_csv
    else:
        project_root = os.path.dirname(os.path.abspath(__file__))
        metrics_dir  = os.path.join(project_root, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        basedir  = os.path.basename(os.path.normpath(args.img_root))
        csv_path = os.path.join(metrics_dir, f"{basedir}_jepa.csv")

    # Compute JEPA scores
    all_scores = []
    desc = f"JEPA scoring ({args.backbone}, {args.img_size}x{args.img_size})"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", f"jepa_score({args.backbone})"])

        for p in tqdm(img_paths, desc=desc):
            x = load_image_normalized(p, mean.cpu(), std.cpu(), args.img_size).to(device)
            score     = jepa_score(backbone, x, eps=args.eps)
            score_val = score.item()
            all_scores.append(score_val)
            writer.writerow([os.path.basename(p), f"{score_val:.10f}"])
            del x

        mean_score = np.mean(all_scores)
        std_score  = np.std(all_scores)
        writer.writerow([])
        writer.writerow(["mean", f"{mean_score:.10f}"])
        writer.writerow(["std",  f"{std_score:.10f}"])

    print(f"[Done] Saved {len(img_paths)} scores to {csv_path}")
    print("=" * 50)
    print(f"  JEPA Score : {mean_score:.4f} ± {std_score:.4f}")
    print("=" * 50)

    if args.log_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_csv)), exist_ok=True)
        with open(args.log_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["# Config"])
            writer.writerow(["img_root",      args.img_root])
            writer.writerow(["backbone",      args.backbone])
            writer.writerow(["img_size",      args.img_size])
            writer.writerow(["max_images",    args.max_images])
            writer.writerow(["num_images",    len(img_paths)])
            writer.writerow(["random_sample", args.random_sample])
            writer.writerow(["seed",          args.seed])
            writer.writerow(["eps",           args.eps])
            writer.writerow([])
            writer.writerow(["# Results"])
            writer.writerow(["metric", "mean", "std"])
            writer.writerow([f"JEPA_Score({args.backbone})", f"{mean_score:.10f}", f"{std_score:.10f}"])
        print(f"[Done] JEPA log saved to: {args.log_csv}")


if __name__ == "__main__":
    main()
