"""ref latent 사전 계산 — 대규모 이미지 풀의 step별 forward-noising latent.

manifold.py의 rollout_forward_noise와 동일 프로토콜:
  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε_shared (B=batch개 noise → batch_mean)
  - shared noise: torch.manual_seed(seed) → randn(N, batch, 4, 64, 64)
    (첫 100×5 블록이 n_prompts=100 실험 실행 시의 noise와 동일)
  - 연산 dtype: fp16 (rollout_forward_noise와 동일)

차이점: 결과 이미지를 results/에 따로 저장하지 않음 (원본 리사이즈본이 이미 있음).

출력: {out_root}/{model}/ddim/CFG/NFE/seed/size/batch/save/ref/{tag}/record/latent/latent.npz
  tag = --ref_images 경로에서 `ref/`와 마지막 `results` 사이 (예: laion_Aes_v2/512x512)
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from latent_diffusion import StableDiffusion  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="ref 풀 latent 사전 계산 (forward noising)")
    p.add_argument("--ref_images", type=str, required=True,
                   help="ref 이미지 디렉토리 (save/ref/{tag}/results 형태 권장)")
    p.add_argument("--n_images", type=int, default=5000)
    p.add_argument("--model_key", type=str,
                   default=os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-4"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--NFE", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=5)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--encode_bs", type=int, default=25)
    p.add_argument("--out_root", type=str, default=os.path.join(SCRIPT_DIR, "manifold"))
    p.add_argument("--no_compress", action="store_true", default=True,
                   help="대용량 npz는 비압축 저장 (기본 on — 로드·쓰기 속도)")
    args = p.parse_args()
    device = torch.device(args.device)

    # ---- 출력 경로: ref/{tag}/record/latent/latent.npz ----
    _parts = os.path.abspath(args.ref_images).rstrip("/").split(os.sep)
    if "ref" in _parts:
        tag = os.path.join(*_parts[_parts.index("ref") + 1:-1]) if _parts[-1] == "results" \
            else os.path.join(*_parts[_parts.index("ref") + 1:])
    else:
        tag = _parts[-1]
    model_tag = os.path.basename(args.model_key.rstrip("/"))
    out_npz = os.path.join(args.out_root, model_tag, "ddim",
                           f"CFG={args.cfg}_NFE={args.NFE}", f"seed={args.seed}",
                           "size=512x512", f"batch={args.batch}",
                           "save", "ref", tag, "record", "latent", "latent.npz")
    if os.path.exists(out_npz):
        print(f"[skip] 이미 존재: {out_npz}")
        return
    print(f"[out] {out_npz}")

    # ---- 모델 (VAE encode + scheduler alpha) ----
    from munch import munchify
    sd = StableDiffusion(solver_config=munchify({"num_sampling": args.NFE}),
                         model_key=args.model_key, device=device, seed=args.seed)

    # ---- 이미지 목록 ----
    from PIL import Image
    from torchvision import transforms
    files = sorted(f for f in os.listdir(args.ref_images)
                   if f.lower().endswith((".jpg", ".png", ".jpeg")))
    if len(files) < args.n_images:
        print(f"[warn] 요청 {args.n_images} > 파일 {len(files)} — 있는 만큼 사용")
        args.n_images = len(files)
    files = files[:args.n_images]
    print(f"[load] {args.n_images} images from {args.ref_images}")

    tf = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # ---- VAE encode → x_0 latent (CPU fp16 적재) ----
    t0 = time.time()
    x0_lat = torch.empty(args.n_images, 4, 64, 64, dtype=torch.float16)
    with torch.no_grad():
        for i in range(0, args.n_images, args.encode_bs):
            chunk = files[i:i + args.encode_bs]
            imgs = torch.stack([tf(Image.open(os.path.join(args.ref_images, f)).convert("RGB"))
                                for f in chunk]).to(device, sd.dtype)
            x0_lat[i:i + len(chunk)] = sd.encode(imgs).to(torch.float16).cpu()
            if (i // args.encode_bs) % 20 == 0:
                print(f"  [encode {i + len(chunk)}/{args.n_images}] {time.time()-t0:.0f}s", flush=True)
    print(f"[encode 완료] x_0 {tuple(x0_lat.shape)} — {time.time()-t0:.0f}s")

    # ---- shared noise: rollout 프로토콜과 동일 (첫 100×5 블록 정합) ----
    torch.manual_seed(args.seed)
    noise = torch.randn(args.n_images, args.batch, 4, 64, 64,
                        device=device, dtype=torch.float32)
    print(f"[shared-noise] {tuple(noise.shape)}")

    # ---- forward noising — step별 batch_mean → (T, n, 4, 64, 64) fp16 ----
    timesteps = list(sd.scheduler.timesteps)
    T = len(timesteps)
    out = np.empty((T, args.n_images, 4, 64, 64), dtype=np.float16)
    x0_dev = x0_lat.to(device, sd.dtype)
    with torch.no_grad():
        for i_t, t in enumerate(timesteps):
            at = sd.alpha(t)
            xt = at.sqrt() * x0_dev[:, None] + (1 - at).sqrt() * noise.to(sd.dtype)
            xt = xt.mean(dim=1)                                  # batch_mean
            out[i_t] = xt.to(torch.float16).cpu().numpy()
    del x0_dev, noise
    print(f"[noising 완료] x_t {out.shape} — {time.time()-t0:.0f}s")

    # ---- 저장 ----
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz,
             x_t=out,
             timesteps=np.asarray([int(t) for t in timesteps], dtype=np.int64),
             prompts=[f"ref_{i}" for i in range(out.shape[1])])
    sz = os.path.getsize(out_npz) / 1e9
    print(f"[Done] {out_npz} — {sz:.1f}GB, 총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
