"""
download_vendi_ckpt.py — Download inception_v3 weights to ckpt/ for offline vendi_score.

온라인 환경에서 실행하면 torchvision의 공식 가중치를 받아 ckpt/에 저장.
이후 get_inception()이 ckpt/를 우선 참조하므로 오프라인에서도 vendi_score 측정 가능.

Usage:
    python download_vendi_ckpt.py
"""
import os
import torch.hub
from torchvision.models import Inception_V3_Weights

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")


def download_inception():
    """inception_v3 (ImageNet) 가중치를 ckpt/ 로 다운로드."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    url = Inception_V3_Weights.IMAGENET1K_V1.url
    fname = url.split("/")[-1]                      # inception_v3_google-0cc3c7bd.pth
    dst = os.path.join(CKPT_DIR, fname)

    if os.path.exists(dst):
        print(f"[Skip] 이미 존재: {dst}")
        return dst

    print(f"[Download] {url}")
    print(f"        → {dst}")
    torch.hub.download_url_to_file(url, dst)
    size_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"[Done] {dst}  ({size_mb:.1f} MB)")
    return dst


if __name__ == "__main__":
    download_inception()
