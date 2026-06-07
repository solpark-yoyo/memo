import torch
from diffusers import DDIMInverseScheduler
from cas.utils_ll import *

import blobfile as bf

from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import itertools
from tqdm import tqdm

import argparse

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def main():
    parser = argparse.ArgumentParser(description="Log-likelihood computation")
    parser.add_argument("--eval_dir", type=str, default="")
    parser.add_argument("--prompt_dir", type=str, default="examples/assets/coco_v2.txt")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--total_step", type=int, default=10)
    parser.add_argument("--approx_num", type=int, default=3)
    parser.add_argument("--rec_num", type=int, default=2)
    parser.add_argument("--resume_from", type=int, default=0)
    parser.add_argument("--method", type=str, default='ddim')
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sd20", "sdxl", "sdxl_lightning"])
    
    args = parser.parse_args()

    prompt_dir = args.prompt_dir
    num_samples = args.num_samples
    total_step = args.total_step
    approx_num = args.approx_num
    rec_num = args.rec_num
    resume_from = args.resume_from
    method = args.method
    model = args.model

    latent_dim = 16384 if (model == 'sd15' or model == 'sd20') else 65536

    output_file = os.path.join(args.eval_dir, "ll_metric.txt")

    eval_list = [
        args.eval_dir,
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler = DDIMInverseScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, compute_likelihood=False, set_alpha_to_zero=False)
    scheduler.set_timesteps(total_step)

    if model == 'sd15' or model == 'sd20':
        cas_preprocessor = CAS_preprocessor(model = model, scheduler = scheduler, device = device, num_inference_steps = total_step, approx_num = approx_num, rec_num = rec_num)
    else:
        cas_preprocessor = CAS_preprocessor_SDXL(model = model, scheduler = scheduler, device = device, num_inference_steps = total_step, approx_num = approx_num, rec_num = rec_num)
    cas_integrator = CAS_integrator(total_step, scheduler = scheduler)

    text_list = []
    with open(prompt_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # Only add non-empty lines
                text_list.append(stripped_line)
    prompts = text_list[resume_from : (resume_from + num_samples)] # Test for 10k MS-COCO validation

    # cas_dict = {}

    for eval_img_path in eval_list:
        
        cas_list = []
        img_files = _list_image_files_recursively(eval_img_path)[:num_samples]
        assert len(img_files) == len(prompts)
        eval_name = os.path.basename(eval_img_path)

        folder_name = f"metrics"
        os.makedirs(folder_name, exist_ok=True)
        save_name = f"{folder_name}/cas-ll-appnum={approx_num}_{model}_{method}_{eval_name}_{num_samples}.pt"
        
        if os.path.exists(save_name):
            cas_list = torch.load(save_name)
        else:
            for prompt, img_path in tqdm(zip(prompts, img_files), total=len(prompts)):
                image = Image.open(img_path)
                res = cas_preprocessor.preprocess(image, prompt)
                cas = cas_integrator.score(res['llhood']['total'], res['jacobian']['total'], res['dim'])
                ll = (cas / latent_dim) / np.log(2)
                print("ll: ", ll)
                cas_list.append(torch.tensor(ll))
                
            cas_list = torch.stack(cas_list)
            torch.save(cas_list, save_name)

        with open(output_file, 'w') as output:
            cas_mean = cas_list.mean().item()
            cas_sd = cas_list.std().item()
            output.write(f"log-likelihood (bpd): {cas_mean} +/- {cas_sd}\n")
            
        # cas_dict[f'{eval_name}'] = cas_list
        
        # methods = [keys for keys in cas_dict.keys()]

        # lss = ['-', '--', ':', '-.']
        # linestyle_cycle = itertools.cycle(lss)

        # # Iterate through the five airlines
        # for method in methods:
        #     ls = next(linestyle_cycle)
        #     subset = cas_dict[method]
            
        #     # Draw the density plot
        #     sns.distplot(subset, hist = False, kde = True,
        #                 kde_kws = {'shade': True, 'linewidth': 2, 'linestyle': ls}, 
        #                 label = method)


        # plt.legend(prop={'size': 13}, loc='upper center', bbox_to_anchor=(0.5, 1.41), ncol=2)
        # plt.xlabel('CAS', size=19)
        # plt.ylabel('Density', size=19)
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.grid(color = 'gray', linestyle = ':', alpha=0.5, linewidth = 1.0)
        # plt.show()
        # save_path = f"{folder_name}/density_plot_{eval_name}.png"
        # plt.savefig(save_path)
        # plt.close()


if __name__ == "__main__":
    main()