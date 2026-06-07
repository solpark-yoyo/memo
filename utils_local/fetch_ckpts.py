#!/usr/bin/env python3
"""
jepa_sd/ckpt/ 에 필요한 체크포인트를 snapshot_download로 내려받습니다.

대상:
  compute_t2i_metrics.py 에서 필요한 것들
    - bert-base-uncased          (ImageReward → BLIP tokenizer)
    - clip-vit-base-patch16      (CLIPScore)
    - CLIP-ViT-H-14-laion2B-s32B-b79K  (PickScore processor)
    - PickScore_v1               (PickScore model)

사용법:
    python3 utils_local/fetch_ckpts.py
    python3 utils_local/fetch_ckpts.py --only bert-base-uncased
    python3 utils_local/fetch_ckpts.py --list
"""

import os
import sys
import argparse
from pathlib import Path

CKPT_ROOT = Path(__file__).parent.parent / "ckpt"

# (repo_id, local_dirname, repo_type)
TARGETS = [
    ("google-bert/bert-base-uncased",              "bert-base-uncased",               "model"),
    ("openai/clip-vit-base-patch16",               "clip-vit-base-patch16",           "model"),
    ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K",     "CLIP-ViT-H-14-laion2B-s32B-b79K", "model"),
    ("yuvalkirstain/PickScore_v1",                 "PickScore_v1",                    "model"),
]

# bert-base-uncased 는 tokenizer 파일만 있으면 되므로 모델 가중치는 제외
IGNORE_PATTERNS = {
    "bert-base-uncased": [
        "pytorch_model.bin", "model.safetensors", "tf_model*",
        "flax_model*", "rust_model*", "*.msgpack", "*.h5",
    ],
}

DEFAULT_IGNORE = ["*.msgpack", "flax_model*", "tf_model*", "*.h5"]


def download_one(repo_id: str, local_dirname: str, repo_type: str, force: bool):
    from huggingface_hub import snapshot_download

    local_dir = CKPT_ROOT / local_dirname

    if not force and local_dir.exists() and any(local_dir.iterdir()):
        print(f"[SKIP] {local_dirname}  (already exists)")
        return

    ignore = IGNORE_PATTERNS.get(local_dirname, DEFAULT_IGNORE)
    print(f"\n[DL] {repo_id}")
    print(f"     → {local_dir}")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(local_dir),
            ignore_patterns=ignore,
        )
        print(f"[OK] {local_dirname}")
    except Exception as e:
        print(f"[FAIL] {local_dirname}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Download jepa_sd checkpoints")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="Download only these local_dirname(s)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if directory already exists")
    parser.add_argument("--list", action="store_true",
                        help="Print available targets and exit")
    args = parser.parse_args()

    if args.list:
        for _, local_dirname, _ in TARGETS:
            exists = "✓" if (CKPT_ROOT / local_dirname).exists() else " "
            print(f"  [{exists}] {local_dirname}")
        return

    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] ckpt root: {CKPT_ROOT.resolve()}\n")

    for repo_id, local_dirname, repo_type in TARGETS:
        if args.only and local_dirname not in args.only:
            continue
        download_one(repo_id, local_dirname, repo_type, args.force)

    print("\n[Done]")


if __name__ == "__main__":
    main()
