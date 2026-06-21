"""
prepare_coco_real.py — Resize COCO val2014 to 512×512 for PRDC real reference.

Standard T2I benchmark preprocessing:
  Resize shortest side -> 512, then CenterCrop 512×512
  (preserves aspect ratio content, matches SD1.5 512×512 generation)

Usage:
  python prepare_coco_real.py \
      --src datasets/ms_coco/ori/val2014 \
      --dst datasets/ms_coco/resize/val2014_512 \
      --size 512 --workers 16
"""
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image


def process_one(args_tuple):
    src_path, dst_path, size = args_tuple
    if os.path.exists(dst_path):
        return dst_path, True  # already done
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            # resize shortest side to `size`, then center crop to size×size
            w, h = img.size
            scale = size / min(w, h)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = img.resize((new_w, new_h), Image.BILINEAR)
            left = (new_w - size) // 2
            top = (new_h - size) // 2
            img = img.crop((left, top, left + size, top + size))
            img.save(dst_path, quality=95)
        return dst_path, True
    except Exception as e:
        return f"{src_path}: {e}", False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True,
                        help="source image dir (e.g. datasets/ms_coco/ori/val2014)")
    parser.add_argument("--dst", type=str, required=True,
                        help="destination dir (e.g. datasets/ms_coco/resize/val2014_512)")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".JPEG")
    files = sorted([f for f in src.iterdir() if f.suffix in exts])
    print(f"[INFO] {len(files)} images found in {src}")
    print(f"[INFO] Resizing to {args.size}×{args.size} (center-crop) → {dst}")

    tasks = [(str(f), str(dst / (f.stem + ".png")), args.size) for f in files]

    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures)):
            _, ok = fut.result()
            if ok:
                done += 1
            else:
                failed += 1
            if (i + 1) % 2000 == 0:
                print(f"  ...{i+1}/{len(tasks)} processed ({done} ok, {failed} failed)")

    print(f"[Done] {done} resized, {failed} failed → {dst}")


if __name__ == "__main__":
    main()
