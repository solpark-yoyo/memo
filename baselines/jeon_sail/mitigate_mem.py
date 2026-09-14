import argparse
from tqdm import tqdm

import torch
import os
from local_model.pipe import LocalStableDiffusionPipeline
from diffusers import DDIMScheduler, UNet2DConditionModel


def main(args):
    torch.set_default_dtype(torch.bfloat16)
    used_dtype = torch.bfloat16
    device = args.device

    # model_id 결정 — --ckpt_path 지정 시 sd_ver 무관 해당 로컬 경로 우선 사용,
    # 미지정 시 sd_ver 기본 경로 (sd_ver=1 → v1-4, sd_ver=2 → 2-base)
    if args.ckpt_path:
        model_id = args.ckpt_path
    elif args.sd_ver == 1:
        model_id = 'ckpt/stable-diffusion-v1-4'
    else:
        model_id = 'ckpt/stable-diffusion-2-base'

    # model_tag — ckpt basename → workdir 패밀리명 매핑:
    #   stable-diffusion-v1-4   → sd14_base
    #   stable-diffusion-v1-5   → sd15
    #   stable-diffusion-2-base → sd20_base
    #   stable-diffusion-2-1-base → sd21_base
    #   기타 → basename 그대로. ckpt_path 미지정 시 기존 sd1/sd2 폴백 유지
    ckpt_family_map = {
        'stable-diffusion-v1-4': 'sd14_base',
        'stable-diffusion-v1-5': 'sd15',
        'stable-diffusion-2-base': 'sd20_base',
        'stable-diffusion-2-1-base': 'sd21_base',
    }
    if args.ckpt_path:
        ckpt_name = os.path.basename(args.ckpt_path.rstrip('/'))
        model_tag = ckpt_family_map.get(ckpt_name, ckpt_name)
    else:
        model_tag = f'sd{args.sd_ver}'

    print(f'[model] model_id={model_id}  model_tag={model_tag}')

    if args.sd_ver == 1:
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder='unet', torch_dtype=used_dtype
        )
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            model_id,
            unet=unet,
            torch_dtype=used_dtype,
            safety_checker=None,
        )
    else:
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=used_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # 24GB GPU 대응 — miti 최적화(batch 16 UNet forward 2회 × autograd)의
    # activation 메모리 절약. gradient checkpointing은 재계산 기반으로 수치 동일.
    pipe.unet.enable_gradient_checkpointing()
        
    args.exp_type = 'miti' #mitigation task
    # --output_dir 지정 시 workdir result에 ours 규격(img_XXXX_YY.png)으로 직접 저장
    # → miti_outputs 스테이징 디렉토리 자체가 생성되지 않음
    if args.output_dir:
        gen_img_path = args.output_dir
    else:
        gen_img_path = f'./miti_outputs/{model_tag}/{args.prompt_type}'
    os.makedirs(gen_img_path, exist_ok=True)

    # 프롬pt 배치 청킹 — batch_txt=1이면 크기 1 청크 = 기존 per-prompt 경로 그대로
    prompts_all = []
    with open(args.data_path, 'r') as file:
        for line_id, line in enumerate(file):
            if args.num_prompts and line_id >= args.num_prompts:
                break                                    # --num_prompts 상한 (0=전체)
            if line.strip():
                prompts_all.append((line_id, line.strip()))

    for bstart in range(0, len(prompts_all), max(args.batch_txt, 1)):
        chunk = prompts_all[bstart:bstart + max(args.batch_txt, 1)]

        for pid, prompt in chunk:
            print(prompt)

        # 배치 첫 line_id 기준 seed — batch_txt=1에서 기존 스트림과 동일
        torch.manual_seed(args.gen_seed + chunk[0][0])

        pipe_input = [p for _, p in chunk] if len(chunk) > 1 else chunk[0][1]
        images = pipe(
            pipe_input,
            num_images_per_prompt=args.gen_num,
            args=args
        )
        gen_lst = images.images

        # 이미지 순서 = prompt-major (p0_s0..p0_s{k-1}, p1_s0..)
        for j, (pid, prompt) in enumerate(chunk):
            image_name = prompt.replace('/', '').replace('\\', '')  # Remove any slashes
            while image_name.startswith('"') or image_name.startswith("'"):
                image_name = image_name.strip('"').strip('"').strip("'").strip("'")
            for k in range(args.gen_num):
                if args.output_dir:
                    gen_lst[j * args.gen_num + k].save(
                        f"{gen_img_path}/img_{pid:04d}_{k:02d}.png")
                else:
                    gen_lst[j * args.gen_num + k].save(
                        f"{gen_img_path}/{pid}_{image_name}_{k}.png")
        print(f"[gen] pid {chunk[0][0]}~{chunk[-1][0]} ({len(chunk)} prompts)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion memorization")
    parser.add_argument("--sd_ver", default=1, type=int)
    parser.add_argument("--ckpt_path", default=None, type=str, help="로컬 ckpt 경로 (예: ckpt/stable-diffusion-2-base). 미지정 시 sd_ver 기본 경로 사용")
    parser.add_argument("--gen_num", default=4, type=int)
    parser.add_argument("--gen_seed", default=42, type=int)
    parser.add_argument("--prompt_type", default='mem', type=str)
    parser.add_argument("--data_path", default='prompts/sample_mitigation.txt', type=str)
    parser.add_argument("--num_prompts", default=0, type=int,
                        help='사용할 프롬pt 상한 (0=전체) — 원본 파일을 직접 절제해 읽음')
    
    ## Hyperparameters (check Appendix D)
    parser.add_argument("--miti_thres", default=8.2, type=float, help='l_thres (refer to Algorithm 2)')
    parser.add_argument("--miti_lr", default=0.05, type=float, help='learning rate for latent optimization')
    parser.add_argument("--miti_budget", default=8, type=int, help='batch size for simultaneous latent optimization')
    parser.add_argument("--miti_max_steps", default=10, type=int, help='max steps for latent optimization (may hurt gaussianity)')
    parser.add_argument("--device", default="cuda", type=str, help="연산 장치 (예: cuda, cuda:3)")
    parser.add_argument("--output_dir", default=None, type=str,
                        help='직접 저장 디렉토리 (ours 규격 img_XXXX_YY.png). 미지정 시 기존 ./miti_outputs 경로')
    parser.add_argument("--batch_txt", default=1, type=int,
                        help='한 번의 최적화+생성에 묶을 프롬pt 수 (1=기존 per-prompt 경로). '
                             '예: --batch_txt 5 --gen_num 1')
    parser.add_argument("--forward_chunk", default=32, type=int,
                        help='miti 배치 최적화의 UNet 1회 행 상한 (OOM 방지 chunking)')


    args = parser.parse_args()
    main(args)
