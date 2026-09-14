"""ren MemAttn batch 러너 — text2img.py의 검증된 로드 방식 계승 + num_images_per_prompt 지원.

text2img.py 대비 차이:
  - --num_images_per_prompt (기본 5): 프롬pt당 k장
  - 저장명: "{prompt_id}_{text}_{k}.png" (k=샘플 인덱스) — collect 정규식 정합

호환 핵심 (2026-08-30 크래시 교훈):
  refactored_classes.refactored_unet_2d_condition.UNet2DConditionModel을 pipeline에
  주입해야 return_attention/miti_mem kwargs가 통과됨 (stock UNet은 TypeError).
"""

import argparse

parser = argparse.ArgumentParser(description="MemAttn mitigation — batch inference")
parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--local", type=str, default='',
                    help="CUDA_VISIBLE_DEVICES 설정 (하위호환용) — --device 지정 시 --device 가 우선")
parser.add_argument("--device", type=str, default="cuda:0",
                    help="연산 장치 (예: cuda, cuda:3) — --local(CUDA_VISIBLE_DEVICES)보다 우선")
parser.add_argument("--prompt", type=str, default="prompt/cvpr2025_memo_prompt")
parser.add_argument("--num_prompts", type=int, default=0,
                    help="사용할 프롬pt 상한 (0=전체) — examples/assets 원본을 직접 절제해 읽음")
parser.add_argument("--job_id", type=str, default='local')
parser.add_argument("--output_name", type=str, default='local')
parser.add_argument("--output_dir", type=str, default="./results",
                    help="출력 루트 디렉토리 (그 아래 {job_id}_{prompt}_{output_name}_seed{seed}/ 생성)")
parser.add_argument("--miti_mem", action="store_true", default=False)
parser.add_argument("--mask_length_minis1", action="store_true", default=False)
parser.add_argument("--cross_attn_mask", action="store_true", default=False)
parser.add_argument("--save_numpy", action="store_true", default=False)   # MemAttn 내부 참조용
parser.add_argument("--plot", action="store_true", default=False)         # MemAttn 내부 참조용
parser.add_argument("--layers_to_plot", type=int, nargs='*', default=[])  # MemAttn 내부 참조용
parser.add_argument('--c1', type=float, default=1)
parser.add_argument('--save_ours', action='store_true', default=False,
                    help='output_dir에 ours 규격(img_XXXX_YY.png)으로 직접 저장 — raw 네스팅/수집 생략')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_images_per_prompt', type=int, default=5)
parser.add_argument('--batch_txt', type=int, default=1,
                    help='한 번의 파이프라인 호출에 묶을 프롬pt 수 (1=기존 per-prompt 경로). '
                         '예: --batch_txt 5 --num_images_per_prompt 1 → 10프롬pt 2회 호출')
args = parser.parse_args()

import os

if args.local != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.local

import torch
from diffusers import DDIMScheduler
from refactored_classes.MemAttn import MemStableDiffusionPipeline as StableDiffusionPipeline
from refactored_classes.refactored_unet_2d_condition import UNet2DConditionModel

import numpy as np
import random


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

model_id = args.model_name
device = args.device

# ★ 핵심: refactored UNet 주입 (stock UNet이면 return_attention kwarg에서 크래시)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, unet=unet, safety_checker=None, torch_dtype=torch.float16)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# --save_ours: output_dir 자체가 결과 폴더 (raw 네스팅 없이 result/img_XXXX_YY.png 직접 저장)
save_dir = args.output_dir if args.save_ours else f"{args.output_dir}/{args.job_id}_{args.prompt}_{args.output_name}_seed{args.seed}"
os.makedirs(save_dir, exist_ok=True)

from time import time

# 프롬pt 목록 로드 (pid = 원본 파일 line_id — 마지막 배치가 짧아도 정합)
prompts_all = []
with open(f"{args.prompt}.txt", 'r') as file:
    for line_id, line in enumerate(file):
        if args.num_prompts and line_id >= args.num_prompts:
            break                                      # --num_prompts 상한 (0=전체)
        if line.strip():
            prompts_all.append((line_id, line.strip()))

nipp = args.num_images_per_prompt
counter = 0
t_total = 0.0
for bstart in range(0, len(prompts_all), max(args.batch_txt, 1)):
    chunk = prompts_all[bstart:bstart + max(args.batch_txt, 1)]
    args.prompt_id = chunk[0][0]
    # seed: 배치 첫 line_id 기준 (batch_txt=1이면 기존 per-prompt 스트림과 동일)
    set_seed(chunk[0][0] + args.seed)

    # 단일 프롬pt는 문자열 그대로 전달 (기존 경로 100% 보존), 2개 이상만 list 배치
    pipe_input = [p for _, p in chunk] if len(chunk) > 1 else chunk[0][1]
    start = time()
    images = pipe(pipe_input, num_images_per_prompt=nipp,
                  save_prefix=f"{save_dir}/{chunk[0][0]}", args=args).images
    dt = time() - start
    t_total += dt

    # 이미지 순서 = prompt-major (p0_s0..p0_s{k-1}, p1_s0..) — embeds 반복 순서와 동일
    n_saved = 0
    for j, (pid, prompt) in enumerate(chunk):
        save_name = '_'.join(prompt.split(' ')).replace('/', '<#>')
        for k in range(nipp):
            idx = j * nipp + k
            try:
                if args.save_ours:
                    images[idx].save(f"{save_dir}/img_{pid:04d}_{k:02d}.png")
                else:
                    images[idx].save(f"{save_dir}/{pid}_{save_name}_{k}.png")
                n_saved += 1
            except Exception as e:
                print(f"[save-fail] {pid}_{k}: {e}", flush=True)
    print(f"[gen] pid {chunk[0][0]}~{chunk[-1][0]}: {n_saved}장 ({dt:.1f}s)", flush=True)
    counter += len(chunk)

print(f"[Done] {counter} prompts × {nipp} (batch_txt={args.batch_txt}), "
      f"평균 {t_total / max(counter, 1):.1f}s/prompt → {save_dir}")
