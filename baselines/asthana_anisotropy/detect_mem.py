import argparse
import os
from tqdm import tqdm
import random
import torch
import numpy as np
from local_model.pipe import LocalStableDiffusionPipeline
from diffusers import DDIMScheduler, UNet2DConditionModel


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    torch.set_default_dtype(torch.bfloat16)
    used_dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.sd_ver == 1:
        model_id = 'CompVis/stable-diffusion-v1-4'
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder='unet', torch_dtype=used_dtype
        )
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            model_id,
            unet=unet,
            torch_dtype=used_dtype,
            safety_checker=None,
            normalization=args.normalization,
        )
    elif args.sd_ver == 2:
        model_id = 'stabilityai/stable-diffusion-2'
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=used_dtype,
            safety_checker=None,
            requires_safety_checker=False,
            normalization=args.normalization,
        )
    else:
        model_id= 'SG161222/Realistic_Vision_V5.1_noVAE'
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=used_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    with open(args.data_path, 'r') as file:
        total_data_num = sum(1 for _ in file)

    args.exp_type = 'det'  # detection task
    stat_cont = torch.zeros((total_data_num, args.hvp_sampling_num))
    cos_cont = torch.zeros((total_data_num, args.hvp_sampling_num))

    with open(args.data_path, 'r') as file:
        for line_id, line in enumerate(file):
            set_seed(args.gen_seed)  # invariant to prompt ordering
            prompt = line.strip()
            print(prompt)

            hvp_res, cosine_res = pipe(
                prompt,
                num_images_per_prompt=args.gen_num,
                args=args,
                guidance_scale=args.guidance_scale,
                mode=args.mode,
            )
            stat_cont[line_id] = hvp_res
            cos_cont[line_id] = cosine_res

    os.makedirs('./det_outputs', exist_ok=True)
    save_path = (args.data_path).split('.')[0].split('/')[-1]

    torch.save(stat_cont, f'./det_outputs/{save_path}_gen{args.gen_num}_mode{args.mode}_seed{args.gen_seed}.pt')
    torch.save(cos_cont, f'./det_outputs/{save_path}_cosine_gen{args.gen_num}_mode{args.mode}_seed{args.gen_seed}.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion memorization")
    parser.add_argument("--sd_ver", default=1, type=int)
    parser.add_argument("--gen_num", default=1, type=int)
    parser.add_argument("--hvp_sampling_num", default=1, type=int)
    parser.add_argument("--gen_seed", default=42, type=int)
    parser.add_argument("--data_path", default='prompts/sd1_mem.txt', type=str)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--mode", default="x,c|x", type=str)
    parser.add_argument("--normalization", default="None", type=str)

    args = parser.parse_args()
    main(args)
