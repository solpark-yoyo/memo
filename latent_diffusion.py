"""
This module includes LDM-based inverse problem solvers.
Forward operators follow DPS and DDRM/DDNM.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import os

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

_CKPT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")
_DINOV2_HUB  = os.path.join(_CKPT_DIR, "jepa_model", "dinov2", "facebookresearch_dinov2_main")
_DINOV2_CKPTS = os.path.join(_CKPT_DIR, "jepa_model", "dinov2", "checkpoints")
_DINOV2_PTH  = {
    "dinov2_vits14": "dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14": "dinov2_vitb14_reg4_pretrain.pth",
    "dinov2_vitl14": "dinov2_vitl14_reg4_pretrain.pth",
}
_DINOV3_DIR  = os.path.join(_CKPT_DIR, "jepa_model", "dinov3")
_DINOV3_VARIANTS = {
    "dinov3_vits16": "dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "dinov3-vitl16-pretrain-lvd1689m",
}
_SSL_DIR = os.path.join(_CKPT_DIR, "jepa_model", "ssl")
_SSL_VARIANTS = {
    # MAE: reconstruction-based, no uniformity objective (facebook/vit-mae-base)
    "mae_vitb16":      "vit-mae-base",
    # Data2Vec-Vision: predicts contextualized representations, no uniformity (facebook/data2vec-vision-base)
    "data2vec_vitb16": "data2vec-vision-base",
}
from tqdm import tqdm

from torch.optim.adam import Adam
import copy
import math

from torchvision.utils import save_image
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from torch.optim.lr_scheduler import ExponentialLR
from utils_local.ptp_utils import AttendExciteAttnProcessor, AttentionStore

from utils_local.attn_utils import fn_show_attention, fn_smoothing_func, fn_get_topk, fn_clean_mask, fn_get_otsu_mask

####### Factory #######
__SOLVER__ = {}

def register_solver(name: str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls
    return wrapper

def get_solver(name: str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n+1, device=device)[:-1]
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

########################

class StableDiffusion():
    def __init__(self,
                 solver_config: Dict,
                #  model_key:str="runwayml/stable-diffusion-v1-5",
                 model_key:str=os.path.join(_CKPT_DIR, "stable-diffusion-v1-5"),
                 device: Optional[torch.device]=None,
                 seed: Optional[int]=42,
                 **kwargs):
        self.device = device

        self.dtype = kwargs.get("pipe_dtype", torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype, local_files_only=True).to(device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.model_key = model_key

        self.tokenizer_base = copy.deepcopy(pipe.tokenizer)
        self.text_encoder_base = copy.deepcopy(pipe.text_encoder)
        # self.unet_base = copy.deepcopy(pipe.unet)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", local_files_only=True)
        self.total_alphas = self.scheduler.alphas_cumprod.clone()

        self.sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        self.log_sigmas = self.sigmas.log()

        total_timesteps = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = total_timesteps // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod_default = self.scheduler.alphas_cumprod
        self.scheduler.alphas_cumprod_default = self.scheduler.alphas_cumprod_default.to(device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod]).to(device)

        # a dedicated generator for various purposes
        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(seed)
        
        
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.sample(*args, **kwargs)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Solver must implement sample() method.")
    
    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at
    
    def decode_latents(self, pred_x0):
        # decode latent into images
        pred_x0_img = self.vae.decode(
            1.0 / self.vae.config.scaling_factor * pred_x0.to(self.vae.dtype),
            return_dict=False,
        )[0]
        return (pred_x0_img / 2.0 + 0.5).clamp(0, 1)
    
    def _hook_fn(self, module, input, output, layer_name):
        self.features[layer_name] = output
    
    def _register_hooks(self):
        target_layers = {}

        # for i, down_block in enumerate(self.unet.down_blocks):
        #     layer_name = f"down_block_{i}"
        #     target_layers[layer_name] = down_block
            
        for i, up_block in enumerate(self.unet.up_blocks):
            layer_name = f"up_block_{i}"
            target_layers[layer_name] = up_block

        for name, layer in target_layers.items():
            layer.register_forward_hook(lambda m, i, o, name=name: self._hook_fn(m, i, o, name))
            
    def save_feature(self, module, input, output):
        global extracted_feature
        extracted_feature = output[0].clone()
    
    # def register_attention_control(self, attn_where="all", attn_type="both"):
    #     attn_procs = {}
    #     # cross_att_count = 0
    #     att_count = 0
    #     for name in self.unet.attn_processors.keys():
    #         if name.startswith("mid_block"):
    #             place_in_unet = "mid"
    #         elif name.startswith("up_blocks"):
    #             place_in_unet = "up"
    #         elif name.startswith("down_blocks"):
    #             place_in_unet = "down"
    #         else:
    #             continue
            

    #         if attn_where == "all" or place_in_unet == attn_where:
    #             att_count += 1
    #         attn_procs[name] = AttendExciteAttnProcessor(attnstore=self.attention_store, place_in_unet=place_in_unet, attn_where=attn_where, attn_type=attn_type)
            

    #     self.unet.set_attn_processor(attn_procs)
        
    #     if not attn_type == "both":
    #         att_count = int(att_count / 2)
    #     self.attention_store.num_att_layers = att_count
    
    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(attnstore=self.attention_store, place_in_unet=place_in_unet)

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    @torch.no_grad()
    def get_text_embed(self, null_prompt, prompt):
        """
        Get text embedding.
        args:
            null_prompt (str): null text
            prompt (str): guidance text
        """
        # null text embedding (negation)
        null_text_input = self.tokenizer(null_prompt,
                                         padding='max_length',
                                         max_length=self.tokenizer.model_max_length,
                                         return_tensors="pt",)
        null_text_embed = self.text_encoder(null_text_input.input_ids.to(self.device))[0]

        # text embedding (guidance)
        text_input = self.tokenizer(prompt,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    return_tensors="pt",
                                    truncation=True)
        text_embed = self.text_encoder(text_input.input_ids.to(self.device))[0]
        # import ipdb; ipdb.set_trace()

        return null_text_embed, text_embed
    
    def differentiable_get_text_embed(self, null_prompt, prompt):
        """
        Get text embedding.
        args:
            null_prompt (str): null text
            prompt (str): guidance text
        """
        # null text embedding (negation)
        null_text_input = self.tokenizer(null_prompt,
                                         padding='max_length',
                                         max_length=self.tokenizer.model_max_length,
                                         return_tensors="pt",)
        null_text_embed = self.text_encoder(null_text_input.input_ids.to(self.device))[0]

        # text embedding (guidance)
        text_input = self.tokenizer(prompt,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    return_tensors="pt",
                                    truncation=True)
        text_embed = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return null_text_embed, text_embed

    def encode(self, x):
        """
        xt -> zt
        """
        return self.vae.encode(x).latent_dist.sample() * 0.18215

    @torch.no_grad()
    def decode(self, zt):
        """
        zt -> xt
        """
        zt = 1/0.18215 * zt
        img = self.vae.decode(zt).sample.float()
        return img

    def predict_noise(self,
                      zt: torch.Tensor,
                      t: torch.Tensor,
                      uc: torch.Tensor,
                      c: torch.Tensor):
        """
        compuate epsilon_theta for null and condition
        args:
            zt (torch.Tensor): latent features
            t (torch.Tensor): timestep
            uc (torch.Tensor): null-text embedding
            c (torch.Tensor): text embedding
        """
        t_in = t.unsqueeze(0) if len(t.shape) == 0 else t
        # print("t_in.shape: ", t_in.shape)
        if uc is None:
            noise_c = self.unet(zt, t_in, encoder_hidden_states=c)['sample']
            noise_uc = noise_c
        elif c is None:
            noise_uc = self.unet(zt, t_in, encoder_hidden_states=uc)['sample']
            noise_c = noise_uc
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2) 
            t_in = torch.cat([t_in] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)

        return noise_uc, noise_c

    @torch.no_grad()
    def inversion(self,
                  z0: torch.Tensor,
                  uc: torch.Tensor,
                  c: torch.Tensor,
                  cfg_guidance: float=1.0):

        # initialize z_0
        zt = z0.clone().to(self.device)
         
        # loop
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM Inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            noise_uc, noise_c = self.predict_noise(zt, t, uc, c) 
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at_prev).sqrt() * noise_pred) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

        return zt
    
    def initialize_latent(self,
                          method: str='random',
                          src_img: Optional[torch.Tensor]=None,
                          b_size: int=1,
                          **kwargs):
        if method == 'ddim':
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('uc'),
                               kwargs.get('c'),
                               cfg_guidance=kwargs.get('cfg_guidance', 0.0))
        elif method == 'npi':
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('c'),
                               kwargs.get('c'),
                               cfg_guidance=1.0)
        elif method == 'random':
            size = kwargs.get('latent_dim', (b_size, 4, 64, 64))
            z = torch.randn(size).to(self.device)
        elif method == 'random_kdiffusion':
            size = kwargs.get('latent_dim', (b_size, 4, 64, 64))
            sigmas = kwargs.get('sigmas', [14.6146])
            z = torch.randn(size).to(self.device)
            z = z * (sigmas[0] ** 2 + 1) ** 0.5
        else: 
            raise NotImplementedError

        return z.requires_grad_()
    
    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def to_d(self, x, sigma, denoised):
        '''converts a denoiser output to a Karras ODE derivative'''
        return (x - denoised) / sigma.item()
    
    def calculate_input(self, x, sigma):
        return x / (sigma ** 2 + 1) ** 0.5
    
    def calculate_denoised(self, x, model_pred, sigma):
        return x - model_pred * sigma
    
    def kdiffusion_x_to_denoised(self, x, sigma, uc, c, cfg_guidance, t):
        xc = self.calculate_input(x, sigma)
        noise_uc, noise_c = self.predict_noise(xc, t, uc, c)
        noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
        denoised = self.calculate_denoised(x, noise_pred, sigma)
        uncond_denoised = self.calculate_denoised(x, noise_uc, sigma)
        return denoised, uncond_denoised

    @torch.enable_grad()
    def prompt_opt(self, zt, t, step, placeholder_token_ids_enc, uc, c_base, cfg_guidance, popt_kwargs):
        assert cfg_guidance > 0.0
        placeholder_string = popt_kwargs['placeholder_string']
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
        placeholder_symbol = placeholder_string.split("_")[0]

        decay_rate = popt_kwargs['lr_decay_rate']
        num_opt_tokens = popt_kwargs['num_opt_tokens']

        if (1. - step * decay_rate) <= 0 or popt_kwargs['debug_flag'] == 'no_opt':
            if (1. - step * decay_rate) <= 0:
                print("Learning rate is zero. Skipping prompt optimization and using the latest optimized embedding.")
            else:
                print("Debug flag is set to 'no_opt'. Skipping prompt optimization and using the latest optimized embedding.")
            prompt = self.prompt.copy()

            # add placeholder tokens only for prompt
            # prompt[1] = prompt[1] + " " + placeholder_string
            prompt_list = [prompt[1]]
            if popt_kwargs['placeholder_position'] == 'end':
                prompt_list = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list)]
            elif popt_kwargs['placeholder_position'] == 'start':
                prompt_list = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list)]
            prompt = prompt_list[0]

            print("Prompt: ", prompt)

            uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
            return c
        
        # print("self.text_enc_1.get_input_embeddings().weight.requires_grad: ", self.text_enc_1.get_input_embeddings().weight.requires_grad)
        # print("self.text_enc_2.get_input_embeddings().weight.requires_grad: ", self.text_enc_2.get_input_embeddings().weight.requires_grad)

        para = self.text_encoder.get_input_embeddings().parameters()
        optimizer = Adam(para, lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))
        # print("(before opt) self.text_enc_1.get_input_embeddings().weight.data[0]: ", self.text_enc_1.get_input_embeddings().weight.data[0])
        # print("(before opt) self.text_enc_1.get_input_embeddings().weight.data[49408]: ", self.text_enc_1.get_input_embeddings().weight.data[49408])
        # print("(before opt) self.text_enc_2.get_input_embeddings().weight.data[0]: ", self.text_enc_2.get_input_embeddings().weight.data[0])
        # print("(before opt) self.text_enc_2.get_input_embeddings().weight.data[49408]: ", self.text_enc_2.get_input_embeddings().weight.data[49408])

        # keep original embeddings as reference
        orig_embeds_params_enc = self.text_encoder.get_input_embeddings().weight.data.clone()

        prompt = self.prompt.copy()

        # add placeholder tokens only for prompt
        # prompt[1] = prompt[1] + " " + placeholder_string
        prompt_list = [prompt[1]]
        if popt_kwargs['placeholder_position'] == 'end':
            prompt_list = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompt_list = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list)]
        prompt[1] = prompt_list[0]

        print("Prompt: ", prompt)

        uc, c = self.differentiable_get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        at = self.alpha(t)

        t_mg = int(
            len(self.scheduler.alphas_cumprod_default) * popt_kwargs['p_ratio']
        )
        at_mg = self.scheduler.alphas_cumprod_default[t_mg]
        if popt_kwargs['dynamic_pr']:
            t_mg = int(
                len(self.scheduler.alphas_cumprod_default) - t
            )
            at_mg = self.scheduler.alphas_cumprod_default[t_mg]
        if popt_kwargs['dynamic_pr_rev']:
            next_t = t - self.skip + 1
            t_mg = int(
                len(self.scheduler.alphas_cumprod_default) - next_t
            )
            at_mg = self.scheduler.alphas_cumprod[t_mg]

        t_mg = torch.tensor(t_mg).to(t.device)
        
        for i in range(popt_kwargs['p_opt_iter']):
            if (popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0) or popt_kwargs['cfg_traj_opt']:
                noise_uc, noise_c = self.predict_noise(zt, t, uc.detach(), c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            else:
                _, noise_pred = self.predict_noise(zt, t, None, c)
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            rand_noise = torch.randn(noise_pred.shape, device=noise_pred.device, dtype=noise_pred.dtype, generator=self.generator)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise

            if popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0:
                if popt_kwargs['Cv_ablation']:
                    noise_uc_s, noise_c_s = self.predict_noise(zs, t_mg, uc.detach(), c)
                else:
                    noise_uc_s, noise_c_s = self.predict_noise(zs, t_mg, uc.detach(), c_base.detach())
                noise_pred_s = noise_uc_s + cfg_guidance * (noise_c_s - noise_uc_s)
            else:
                if popt_kwargs['Cv_ablation']:
                    _, noise_pred_s = self.predict_noise(zs, t_mg, None, c)
                else:
                    _, noise_pred_s = self.predict_noise(zs, t_mg, None, c_base.detach())
            
            # tweedie (x0doublehat)
            z0s = (zs - (1-at_mg).sqrt() * noise_pred_s) / at_mg.sqrt()
            if popt_kwargs['p_opt_sg']:
                z0s = z0s.detach()

            assert z0t.shape == z0s.shape and len(z0t.shape) == 4
            if popt_kwargs['elatentlpips'] is not None:
                # print("using ellpips")
                ms = popt_kwargs['elatentlpips'](z0t, z0s, normalize=False)

            else:
                if popt_kwargs['sg_combine'] and not popt_kwargs['p_opt_sg']:
                    term_1 = (z0t.detach() - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
                    term_2 = (z0t - z0s.detach()).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
                    ms = term_1 + popt_kwargs['sg_lambda'] * term_2
                else:
                    ms = (z0t - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
            loss = -1 * ms.mean()
            # print("ms: ", ms)

            optimizer.zero_grad()
            loss.backward()
            # print the gradient
            # print("self.text_enc_1.get_input_embeddings().weight.grad: ", self.text_enc_1.get_input_embeddings().weight.grad)
            # print("self.text_enc_2.get_input_embeddings().weight.grad: ", self.text_enc_2.get_input_embeddings().weight.grad)

            # import ipdb; ipdb.set_trace()

            optimizer.step()
            # print("(after opt, before restore) self.text_enc_1.get_input_embeddings().weight.data[0]: ", self.text_enc_1.get_input_embeddings().weight.data[0])
            # print("(after opt, before restore) self.text_enc_1.get_input_embeddings().weight.data[49408]: ", self.text_enc_1.get_input_embeddings().weight.data[49408])
            # print("(after opt, before restore) self.text_enc_2.get_input_embeddings().weight.data[0]: ", self.text_enc_2.get_input_embeddings().weight.data[0])
            # print("(after opt, before restore) self.text_enc_2.get_input_embeddings().weight.data[49408]: ", self.text_enc_2.get_input_embeddings().weight.data[49408])
            # print("orig_embeds_params_enc1 == self.text_enc_1.get_input_embeddings().weight: ", torch.equal(orig_embeds_params_enc1, self.text_enc_1.get_input_embeddings().weight))
            # print("orig_embeds_params_enc2 == self.text_enc_2.get_input_embeddings().weight: ", torch.equal(orig_embeds_params_enc2, self.text_enc_2.get_input_embeddings().weight))

            # Let's make sure we don't update any embedding weights besides the newly added token
            self.restore_embedding(placeholder_token_ids_enc, orig_embeds_params_enc, self.tokenizer, self.text_encoder)
            
            # print("(after opt, after restore) self.text_enc_1.get_input_embeddings().weight.data[0]: ", self.text_enc_1.get_input_embeddings().weight.data[0])
            # print("(after opt, after restore) self.text_enc_1.get_input_embeddings().weight.data[49408]: ", self.text_enc_1.get_input_embeddings().weight.data[49408])
            # print("(after opt, after restore) self.text_enc_2.get_input_embeddings().weight.data[0]: ", self.text_enc_2.get_input_embeddings().weight.data[0])
            # print("(after opt, after restore) self.text_enc_2.get_input_embeddings().weight.data[49408]: ", self.text_enc_2.get_input_embeddings().weight.data[49408])
            # import ipdb; ipdb.set_trace()
            if not i == popt_kwargs['p_opt_iter'] - 1:
                uc, c = self.differentiable_get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
            else:
                uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        return c
    
    
    
    @torch.enable_grad()
    def batch_prompt_opt(self, zt, ts, step, placeholder_token_ids_enc, uc, c_base, cfg_guidance, popt_kwargs):
        assert cfg_guidance > 0.0
        placeholder_string = popt_kwargs['placeholder_string']
        decay_rate = popt_kwargs['lr_decay_rate']
        num_opt_tokens = popt_kwargs['num_opt_tokens']

        if (1. - step * decay_rate) <= 0 or popt_kwargs['debug_flag'] == 'no_opt':
            if (1. - step * decay_rate) <= 0:
                print("Learning rate is zero. Skipping prompt optimization and using the latest optimized embedding.")
            else:
                print("Debug flag is set to 'no_opt'. Skipping prompt optimization and using the latest optimized embedding.")
            prompts = self.prompts.copy()
            null_prompts = self.null_prompts.copy()

            # add placeholder tokens only for prompt
            # assert popt_kwargs['num_opt_tokens'] == 1
            assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
            placeholder_symbol = placeholder_string.split("_")[0]
            # prompts = [p + " " + f"{placeholder_symbol}_{idx}" for idx, p in enumerate(prompts)]
            if popt_kwargs['placeholder_position'] == 'end':
                prompts = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompts)]
            elif popt_kwargs['placeholder_position'] == 'start':
                prompts = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompts)]
            # print("Prompts: ", prompts)

            _, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)
            return c
        
        # print("self.text_enc_1.get_input_embeddings().weight.requires_grad: ", self.text_enc_1.get_input_embeddings().weight.requires_grad)
        # print("self.text_enc_2.get_input_embeddings().weight.requires_grad: ", self.text_enc_2.get_input_embeddings().weight.requires_grad)

        para = self.text_encoder.get_input_embeddings().parameters()
        optimizer = Adam(para, lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))
        # print("(before opt) self.text_enc_1.get_input_embeddings().weight.data[0]: ", self.text_enc_1.get_input_embeddings().weight.data[0])
        # print("(before opt) self.text_enc_1.get_input_embeddings().weight.data[49408]: ", self.text_enc_1.get_input_embeddings().weight.data[49408])
        # print("(before opt) self.text_enc_2.get_input_embeddings().weight.data[0]: ", self.text_enc_2.get_input_embeddings().weight.data[0])
        # print("(before opt) self.text_enc_2.get_input_embeddings().weight.data[49408]: ", self.text_enc_2.get_input_embeddings().weight.data[49408])

        # keep original embeddings as reference
        orig_embeds_params_enc = self.text_encoder.get_input_embeddings().weight.data.clone()

        prompts = self.prompts.copy()
        null_prompts = self.null_prompts.copy()
        b_size = len(prompts)

        # add placeholder tokens only for prompt
        # assert num_opt_tokens == 1
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
        placeholder_symbol = placeholder_string.split("_")[0]
        # prompts = [p + " " + f"{placeholder_symbol}_{idx}" for idx, p in enumerate(prompts)]
        if popt_kwargs['placeholder_position'] == 'end':
            prompts = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompts)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompts = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompts)]
        # print("Prompts: ", prompts)

        _, c = self.differentiable_get_text_embed(null_prompt=null_prompts, prompt=prompts)

        at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)

        t_mg = int(
            len(self.scheduler.alphas_cumprod_default) * popt_kwargs['p_ratio']
        )
        ts_mg = torch.full((b_size,), t_mg, device=ts.device, dtype=torch.long)
        at_mg = self.scheduler.alphas_cumprod_default[ts_mg].view(b_size, 1, 1, 1)
        if popt_kwargs['dynamic_pr']:
            t_mg = int(
                len(self.scheduler.alphas_cumprod_default) - ts[0].item()
            )
            ts_mg = torch.full((b_size,), t_mg, device=ts.device, dtype=torch.long)
            at_mg = self.scheduler.alphas_cumprod_default[ts_mg].view(b_size, 1, 1, 1)
        if popt_kwargs['dynamic_pr_rev']:
            next_t = ts[0].item() - self.skip + 1
            t_mg = int(
                len(self.scheduler.alphas_cumprod_default) - next_t
            )
            ts_mg = torch.full((b_size,), t_mg, device=ts.device, dtype=torch.long)
            at_mg = self.scheduler.alphas_cumprod[ts_mg].view(b_size, 1, 1, 1)
        
        for i in range(popt_kwargs['p_opt_iter']):
            if (popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0) or popt_kwargs['cfg_traj_opt']:
                # noise_uc, noise_c = self.predict_noise(zt, ts, uc.detach(), c)
                noise_c = self.unet(zt, ts, encoder_hidden_states=c)['sample']
                with torch.no_grad():
                    noise_uc = self.unet(zt, ts, encoder_hidden_states=uc.detach())['sample']
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            else:
                _, noise_pred = self.predict_noise(zt, ts, None, c)
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            rand_noise = torch.randn(noise_pred.shape, device=noise_pred.device, dtype=noise_pred.dtype, generator=self.generator)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise

            if popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0:
                if popt_kwargs['Cv_ablation']:
                    # noise_uc_s, noise_c_s = self.predict_noise(zs, ts_mg, uc.detach(), c)
                    noise_c_s = self.unet(zs, ts_mg, encoder_hidden_states=c)['sample']
                    with torch.no_grad():
                        noise_uc_s = self.unet(zs, ts_mg, encoder_hidden_states=uc.detach())['sample']
                else:
                    # noise_uc_s, noise_c_s = self.predict_noise(zs, ts_mg, uc.detach(), c_base.detach())
                    noise_c_s = self.unet(zs, ts_mg, encoder_hidden_states=c_base.detach())['sample']
                    with torch.no_grad():
                        noise_uc_s = self.unet(zs, ts_mg, encoder_hidden_states=uc.detach())['sample']
                noise_pred_s = noise_uc_s + cfg_guidance * (noise_c_s - noise_uc_s)
            else:
                if popt_kwargs['Cv_ablation']:
                    _, noise_pred_s = self.predict_noise(zs, ts_mg, None, c)
                else:
                    _, noise_pred_s = self.predict_noise(zs, ts_mg, None, c_base.detach())
            
            # tweedie (x0doublehat)
            z0s = (zs - (1-at_mg).sqrt() * noise_pred_s) / at_mg.sqrt()
            if popt_kwargs['p_opt_sg']:
                z0s = z0s.detach()

            assert z0t.shape == z0s.shape and len(z0t.shape) == 4
            if popt_kwargs['elatentlpips'] is not None:
                # print("using ellpips")
                ms = popt_kwargs['elatentlpips'](z0t, z0s, normalize=False)

            else:
                if popt_kwargs['sg_combine'] and not popt_kwargs['p_opt_sg']:
                    term_1 = (z0t.detach() - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
                    term_2 = (z0t - z0s.detach()).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
                    ms = term_1 + popt_kwargs['sg_lambda'] * term_2
                else:
                    ms = (z0t - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
            loss = -1 * ms.sum()
            # loss = -1 * ms.sum()
            # print("ms: ", ms)

            optimizer.zero_grad()
            loss.backward()

            # for test
            # gradients = [p.grad for p in self.text_encoder.get_input_embeddings().parameters() if p.grad is not None][0]
            # updated_gradients = gradients[49408:]
            # import ipdb; ipdb.set_trace()
            optimizer.step()

            # Let's make sure we don't update any embedding weights besides the newly added token
            self.restore_embedding(placeholder_token_ids_enc, orig_embeds_params_enc, self.tokenizer, self.text_encoder)
            
            if not i == popt_kwargs['p_opt_iter'] - 1:
                _, c = self.differentiable_get_text_embed(null_prompt=null_prompts, prompt=prompts)
            else:
                _, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)

        return c

    @torch.enable_grad()
    def prompt_opt_dpmpp_2m(self, x, sigmas, step, placeholder_token_ids_enc, uc, c_base, cfg_guidance, popt_kwargs):
        assert cfg_guidance > 0.0
        placeholder_string = popt_kwargs['placeholder_string']
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
        placeholder_symbol = placeholder_string.split("_")[0]

        decay_rate = popt_kwargs['lr_decay_rate']
        num_opt_tokens = popt_kwargs['num_opt_tokens']

        if (1. - step * decay_rate) <= 0 or popt_kwargs['debug_flag'] == 'no_opt':
            if (1. - step * decay_rate) <= 0:
                print("Learning rate is zero. Skipping prompt optimization and using the latest optimized embedding.")
            else:
                print("Debug flag is set to 'no_opt'. Skipping prompt optimization and using the latest optimized embedding.")
            prompt = self.prompt.copy()

            # add placeholder tokens only for prompt
            prompt_list = [prompt[1]]
            if popt_kwargs['placeholder_position'] == 'end':
                prompt_list = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list)]
            elif popt_kwargs['placeholder_position'] == 'start':
                prompt_list = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list)]
            prompt = prompt_list[0]

            print("Prompt: ", prompt)

            uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
            return c

        para = self.text_encoder.get_input_embeddings().parameters()
        optimizer = Adam(para, lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))

        # keep original embeddings as reference
        orig_embeds_params_enc = self.text_encoder.get_input_embeddings().weight.data.clone()

        prompt = self.prompt.copy()

        # add placeholder tokens only for prompt
        prompt_list = [prompt[1]]
        if popt_kwargs['placeholder_position'] == 'end':
            prompt_list = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompt_list = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list)]
        prompt[1] = prompt_list[0]

        print("Prompt: ", prompt)

        uc, c = self.differentiable_get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        sigma = sigmas[step]
        new_t = self.timestep(sigma).to(self.device)

        step_mg = int(
            len(sigmas) * popt_kwargs['p_ratio']
        )
        if popt_kwargs['dynamic_pr_rev']:
            step_mg = int(len(sigmas) - step - 1)

        sigma_mg = sigmas[step_mg]
        new_t_mg = self.timestep(sigma_mg).to(self.device)
        new_t_mg = torch.tensor(new_t_mg).to(new_t.device)

        for i in range(popt_kwargs['p_opt_iter']):
            if popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0:
                x0t, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)
            else:
                xc = self.calculate_input(x, sigma)
                _, noise_pred = self.predict_noise(xc, new_t, None, c)
                x0t = self.calculate_denoised(x, noise_pred, sigma)

            # add noise
            rand_noise = torch.randn(noise_pred.shape, device=noise_pred.device, dtype=noise_pred.dtype, generator=self.generator)
            xs = x0t + rand_noise * sigma_mg

            if popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0:
                x0s, _ = self.kdiffusion_x_to_denoised(xs, sigma_mg, uc.detach(), c_base.detach(), cfg_guidance, new_t_mg)
            else:
                xsc = self.calculate_input(xs, sigma_mg)
                _, noise_pred_s = self.predict_noise(xsc, new_t_mg, None, c_base.detach())
                x0s = self.calculate_denoised(xs, noise_pred_s, sigma_mg)

            if popt_kwargs['p_opt_sg']:
                x0s = x0s.detach()

            assert x0t.shape == x0s.shape and len(x0t.shape) == 4
            if popt_kwargs['sg_combine'] and not popt_kwargs['p_opt_sg']:
                term_1 = (x0t.detach() - x0s).reshape(x0t.shape[0], -1).norm(p=2.0, dim=-1)
                term_2 = (x0t - x0s.detach()).reshape(x0t.shape[0], -1).norm(p=2.0, dim=-1)
                ms = term_1 + popt_kwargs['sg_lambda'] * term_2
            else:
                ms = (x0t - x0s).reshape(x0t.shape[0], -1).norm(p=2.0, dim=-1)
            loss = -1 * ms.mean()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # Let's make sure we don't update any embedding weights besides the newly added token
            self.restore_embedding(placeholder_token_ids_enc, orig_embeds_params_enc, self.tokenizer, self.text_encoder)

            if not i == popt_kwargs['p_opt_iter'] - 1:
                uc, c = self.differentiable_get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
            else:
                uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        return c

    @torch.enable_grad()
    def batch_prompt_opt_dpmpp_2m(self, x, sigmas, step, placeholder_token_ids_enc, uc, c_base, cfg_guidance, popt_kwargs):
        assert cfg_guidance > 0.0
        placeholder_string = popt_kwargs['placeholder_string']
        decay_rate = popt_kwargs['lr_decay_rate']
        num_opt_tokens = popt_kwargs['num_opt_tokens']

        if (1. - step * decay_rate) <= 0 or popt_kwargs['debug_flag'] == 'no_opt':
            if (1. - step * decay_rate) <= 0:
                print("Learning rate is zero. Skipping prompt optimization and using the latest optimized embedding.")
            else:
                print("Debug flag is set to 'no_opt'. Skipping prompt optimization and using the latest optimized embedding.")
            prompts = self.prompts.copy()
            null_prompts = self.null_prompts.copy()

            # add placeholder tokens only for prompt
            assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
            placeholder_symbol = placeholder_string.split("_")[0]
            if popt_kwargs['placeholder_position'] == 'end':
                prompts = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompts)]
            elif popt_kwargs['placeholder_position'] == 'start':
                prompts = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompts)]

            _, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)
            return c

        para = self.text_encoder.get_input_embeddings().parameters()
        optimizer = Adam(para, lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))

        # keep original embeddings as reference
        orig_embeds_params_enc = self.text_encoder.get_input_embeddings().weight.data.clone()

        prompts = self.prompts.copy()
        null_prompts = self.null_prompts.copy()
        b_size = len(prompts)

        # add placeholder tokens only for prompt
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
        placeholder_symbol = placeholder_string.split("_")[0]
        if popt_kwargs['placeholder_position'] == 'end':
            prompts = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompts)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompts = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompts)]

        _, c = self.differentiable_get_text_embed(null_prompt=null_prompts, prompt=prompts)

        sigma = sigmas[step]
        new_t = self.timestep(sigma).to(self.device)

        # expand for batch
        sigma = sigma.expand(b_size, 1, 1, 1).to(self.device)
        new_t = new_t.expand(b_size)

        step_mg = int(
            len(sigmas) * popt_kwargs['p_ratio']
        )
        if popt_kwargs['dynamic_pr_rev']:
            step_mg = int(len(sigmas) - step - 1)

        sigma_mg = sigmas[step_mg]
        new_t_mg = self.timestep(sigma_mg).to(self.device)
        new_t_mg = torch.tensor(new_t_mg).to(new_t.device)

        # expand for batch
        sigma_mg = sigma_mg.expand(b_size, 1, 1, 1).to(self.device)
        new_t_mg = new_t_mg.expand(b_size)

        for i in range(popt_kwargs['p_opt_iter']):
            if popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0:
                x0t, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)
            else:
                xc = self.calculate_input(x, sigma)
                _, noise_pred = self.predict_noise(xc, new_t, None, c)
                x0t = self.calculate_denoised(x, noise_pred, sigma)

            # add noise
            rand_noise = torch.randn(noise_pred.shape, device=noise_pred.device, dtype=noise_pred.dtype, generator=self.generator)
            xs = x0t + rand_noise * sigma_mg

            if popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0:
                x0s, _ = self.kdiffusion_x_to_denoised(xs, sigma_mg, uc.detach(), c_base.detach(), cfg_guidance, new_t_mg)
            else:
                xsc = self.calculate_input(xs, sigma_mg)
                _, noise_pred_s = self.predict_noise(xsc, new_t_mg, None, c_base.detach())
                x0s = self.calculate_denoised(xs, noise_pred_s, sigma_mg)

            if popt_kwargs['p_opt_sg']:
                x0s = x0s.detach()

            assert x0t.shape == x0s.shape and len(x0t.shape) == 4
            if popt_kwargs['sg_combine'] and not popt_kwargs['p_opt_sg']:
                term_1 = (x0t.detach() - x0s).reshape(x0t.shape[0], -1).norm(p=2.0, dim=-1)
                term_2 = (x0t - x0s.detach()).reshape(x0t.shape[0], -1).norm(p=2.0, dim=-1)
                ms = term_1 + popt_kwargs['sg_lambda'] * term_2
            else:
                ms = (x0t - x0s).reshape(x0t.shape[0], -1).norm(p=2.0, dim=-1)
            loss = -1 * ms.sum()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # Let's make sure we don't update any embedding weights besides the newly added token
            self.restore_embedding(placeholder_token_ids_enc, orig_embeds_params_enc, self.tokenizer, self.text_encoder)

            if not i == popt_kwargs['p_opt_iter'] - 1:
                _, c = self.differentiable_get_text_embed(null_prompt=null_prompts, prompt=prompts)
            else:
                _, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)

        return c



    def restore_embedding(self, placeholder_token_ids, orig_embeds_params, tokenizer, text_enc):
        index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
        index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

        with torch.no_grad():
            text_enc.get_input_embeddings().weight[
                index_no_updates
            ] = orig_embeds_params[index_no_updates]
    
    
    def initialize_embedding(self, tokenizer, text_enc, popt_kwargs, b_size=1):
        num_opt_tokens = popt_kwargs['num_opt_tokens'] * b_size # assignging popt_kwargs['num_opt_tokens'] tokens per each sample
        init_type = popt_kwargs['init_type']
        init_word = popt_kwargs['init_word']
        init_gau_scale = popt_kwargs['init_gau_scale']
        init_rand_vocab = popt_kwargs['init_rand_vocab']
        num_vocab = len(tokenizer)

        assert init_type in ['default', 'word', 'gaussian', 'gaussian_white']
        
        if 'gaussian' in init_type:
            token_embeds_base = text_enc.get_input_embeddings().weight.data.detach().clone()

        placeholder_string = popt_kwargs['placeholder_string']
        # assert popt_kwargs['num_opt_tokens'] == 1 # for now, we only support one token
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2 # the tokens should take the form of "*_0"
        
        placeholder_tokens = [placeholder_string]
        additional_tokens = []
        
        placeholder_symbol = placeholder_string.split("_")[0]
        for i in range(1, num_opt_tokens):
            print("Additional placeholder token: ", f"{placeholder_symbol}_{i}")
            additional_tokens.append(f"{placeholder_symbol}_{i}")
        placeholder_tokens += additional_tokens
        print("Placeholder tokens: ", placeholder_tokens)

        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        print("Number of tokens added to tokenizer: ", num_added_tokens)
        if num_added_tokens != num_opt_tokens:
            # print(f"The tokenizer already contains the token {placeholder_string}.")
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_string}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        print("Placeholder token ids: ", placeholder_token_ids)

        # Save random states before resize_token_embeddings (which consumes global random state internally)
        cpu_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state(self.device)

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_enc.resize_token_embeddings(len(tokenizer))

        # Restore random states to prevent resize_token_embeddings from affecting subsequent randomness
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state, self.device)

        # Re-initialize newly added token embeddings with self.generator for reproducibility
        token_embeds = text_enc.get_input_embeddings().weight.data
        with torch.no_grad():
            for token_id in placeholder_token_ids:
                token_embeds[token_id] = torch.randn(
                    token_embeds[token_id].shape,
                    device=token_embeds.device,
                    dtype=token_embeds.dtype,
                    generator=self.generator
                )

        if init_type == 'word':
            if not init_rand_vocab:
                assert init_word != ""
                # Convert the initializer_token, placeholder_token to ids
                token_ids = tokenizer.encode(init_word, add_special_tokens=False)
                # Check if initializer_token is a single token or a sequence of tokens
                if len(token_ids) > 1:
                    raise ValueError("The initializer token must be a single token.")
                
                initializer_token_id = token_ids[0]
                
                token_embeds = text_enc.get_input_embeddings().weight.data

                with torch.no_grad():
                    for token_id in placeholder_token_ids:
                        print(f"Initialize token id {token_id} as the token embeddin of {init_word}.")
                        # print(f"token_embeds[{token_id}] (before replacement): ", token_embeds[token_id])
                        token_embeds[token_id] = token_embeds[initializer_token_id].clone()
                        # print(f"token_embeds[{token_id}] (after replacement): ", token_embeds[token_id])
            else:
                token_embeds = text_enc.get_input_embeddings().weight.data
                with torch.no_grad():
                    for token_id in placeholder_token_ids:
                        rand_idx = torch.randint(0, num_vocab, (1,), generator=self.generator)
                        print(f"Initialize token id {token_id} as a random vocabulary of index {rand_idx}.")
                        token_embeds[token_id] = token_embeds[rand_idx].clone()


        elif 'gaussian' in init_type:
            embeds_mean = token_embeds_base.mean(dim=0)
            if init_type == 'gaussian_white':
                var_vector = (token_embeds_base ** 2 - embeds_mean.unsqueeze(0) ** 2).mean(dim=0)
                embeds_cov = torch.diag(var_vector) * (init_gau_scale ** 2)
            elif init_type == 'gaussian':
                embeds_cov = torch.einsum('ij,ik->jk', token_embeds_base, token_embeds_base) / token_embeds_base.shape[0]
                embeds_cov = embeds_cov.float() * (init_gau_scale ** 2)

            # Compute Cholesky decomposition for sampling: x = mean + L @ z, where z ~ N(0, I)
            L = torch.linalg.cholesky(embeds_cov)

            token_embeds = text_enc.get_input_embeddings().weight.data
            with torch.no_grad():
                for token_id in placeholder_token_ids:
                    print(f"Initialize token id {token_id} as a multivariate normal distribution ({init_type}).")
                    z = torch.randn(embeds_mean.shape, device=embeds_mean.device, dtype=embeds_mean.dtype, generator=self.generator)
                    token_embeds[token_id] = embeds_mean + L @ z
        elif init_type == 'default':
            print("Default initialization of newly-added embeddings.")


        # Freeze all parameters except for the token embeddings in text encoder
        text_enc.text_model.encoder.requires_grad_(False)
        text_enc.text_model.final_layer_norm.requires_grad_(False)
        text_enc.text_model.embeddings.position_embedding.requires_grad_(False)

        return placeholder_token_ids
    

    @torch.enable_grad()
    def latent_opt(self, zt, t, step, uc, c, etc_kwargs):
        decay_rate = etc_kwargs['l_lr_decay_rate']

        if (1. - step * decay_rate) <= 0:
            print("Learning rate is zero. Skipping latent optimization.")
            return zt
        
        zt = zt.detach()
        zt.requires_grad = True

        optimizer = Adam([zt], lr=etc_kwargs['l_opt_lr'] * (1. - step * decay_rate))

        at = self.alpha(t)

        t_mg = int(
            len(self.scheduler.alphas_cumprod_default) * etc_kwargs['l_p_ratio']
        )
        if etc_kwargs['l_dynamic_pr']:
            t_mg = int(
                len(self.scheduler.alphas_cumprod_default) - t
            )

        at_mg = self.scheduler.alphas_cumprod_default[t_mg]
        t_mg = torch.tensor(t_mg).to(t.device)
        
        for i in range(etc_kwargs['l_opt_iter']):
            _, noise_pred = self.predict_noise(zt, t, None, c.detach())
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            rand_noise = torch.randn_like(noise_pred, device=noise_pred.device)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
            
            _, noise_pred_s = self.predict_noise(zs, t_mg, None, c.detach())
            
            # tweedie (x0doublehat)
            z0s = (zs - (1-at_mg).sqrt() * noise_pred_s) / at_mg.sqrt()
            if etc_kwargs['l_opt_sg']:
                z0s = z0s.detach()

            assert z0t.shape == z0s.shape and len(z0t.shape) == 4
            ms = (z0t - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
            loss = -1 * ms.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return zt.detach()
    

    @torch.enable_grad()
    def batch_latent_opt(self, zt, ts, step, uc, c, etc_kwargs):
        decay_rate = etc_kwargs['l_lr_decay_rate']

        if (1. - step * decay_rate) <= 0:
            print("Learning rate is zero. Skipping latent optimization.")
            return zt
        
        zt = zt.detach()
        zt.requires_grad = True
        b_size = len(zt)

        optimizer = Adam([zt], lr=etc_kwargs['l_opt_lr'] * (1. - step * decay_rate))

        at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)

        t_mg = int(
            len(self.scheduler.alphas_cumprod_default) * etc_kwargs['l_p_ratio']
        )
        if etc_kwargs['l_dynamic_pr']:
            t_mg = int(
                len(self.scheduler.alphas_cumprod_default) - ts[0].item()
            )
            # print("using dynamic_pr. t_mg is : ", t_mg)

        ts_mg = torch.full((b_size,), t_mg, device=ts.device, dtype=torch.long)
        at_mg = self.scheduler.alphas_cumprod_default[ts_mg].view(b_size, 1, 1, 1)
        
        for i in range(etc_kwargs['l_opt_iter']):
            _, noise_pred = self.predict_noise(zt, ts, None, c.detach())
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            rand_noise = torch.randn_like(noise_pred, device=noise_pred.device)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
            
            _, noise_pred_s = self.predict_noise(zs, ts_mg, None, c.detach())
            
            # tweedie (x0doublehat)
            z0s = (zs - (1-at_mg).sqrt() * noise_pred_s) / at_mg.sqrt()
            if etc_kwargs['l_opt_sg']:
                z0s = z0s.detach()

            assert z0t.shape == z0s.shape and len(z0t.shape) == 4
            ms = (z0t - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
            loss = -1 * ms.sum()
            # print("ms: ", ms)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return zt.detach()
    
    @torch.enable_grad()
    def text_emb_opt(self, zt, t, step, uc, c, c_base, popt_kwargs):
        decay_rate = popt_kwargs['lr_decay_rate']

        if (1. - step * decay_rate) <= 0:
            print("Learning rate is zero. Skipping text-embedding optimization and using the latest optimized embedding.")
            return c
        
        c = torch.nn.Parameter(c.detach())
        optimizer = Adam([c], lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))
        
        at = self.alpha(t)

        t_mg = int(
            len(self.scheduler.alphas_cumprod_default) * popt_kwargs['p_ratio']
        )
        if popt_kwargs['dynamic_pr']:
            t_mg = int(
                len(self.scheduler.alphas_cumprod_default) - t
            )

        at_mg = self.scheduler.alphas_cumprod_default[t_mg]
        t_mg = torch.tensor(t_mg).to(t.device)
        
        for i in range(popt_kwargs['p_opt_iter']):
            _, noise_pred = self.predict_noise(zt, t, None, c)
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            rand_noise = torch.randn_like(noise_pred, device=noise_pred.device)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
            
            _, noise_pred_s = self.predict_noise(zs, t_mg, None, c_base.detach())
            
            # tweedie (x0doublehat)
            z0s = (zs - (1-at_mg).sqrt() * noise_pred_s) / at_mg.sqrt()
            if popt_kwargs['p_opt_sg']:
                z0s = z0s.detach()

            assert z0t.shape == z0s.shape and len(z0t.shape) == 4
            if popt_kwargs['sg_combine'] and not popt_kwargs['p_opt_sg']:
                term_1 = (z0t.detach() - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
                term_2 = (z0t - z0s.detach()).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
                ms = term_1 + popt_kwargs['sg_lambda'] * term_2
            else:
                ms = (z0t - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
            loss = -1 * ms.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return c.detach()


    @torch.enable_grad()
    def batch_text_emb_opt(self, zt, ts, step, uc, c, c_base, popt_kwargs):
        decay_rate = popt_kwargs['lr_decay_rate']
        b_size = len(c)

        if (1. - step * decay_rate) <= 0:
            print("Learning rate is zero. Skipping text-embedding optimization and using the latest optimized embedding.")
            return c
        
        c = torch.nn.Parameter(c.detach())
        optimizer = Adam([c], lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))

        at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)

        t_mg = int(
            len(self.scheduler.alphas_cumprod_default) * popt_kwargs['p_ratio']
        )
        if popt_kwargs['dynamic_pr']:
            t_mg = int(
                len(self.scheduler.alphas_cumprod_default) - ts[0].item()
            )
        ts_mg = torch.full((b_size,), t_mg, device=ts.device, dtype=torch.long)
        at_mg = self.scheduler.alphas_cumprod_default[ts_mg].view(b_size, 1, 1, 1)
        
        for i in range(popt_kwargs['p_opt_iter']):
            _, noise_pred = self.predict_noise(zt, ts, None, c)
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            rand_noise = torch.randn_like(noise_pred, device=noise_pred.device)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
            
            _, noise_pred_s = self.predict_noise(zs, ts_mg, None, c_base.detach())
            
            # tweedie (x0doublehat)
            z0s = (zs - (1-at_mg).sqrt() * noise_pred_s) / at_mg.sqrt()
            if popt_kwargs['p_opt_sg']:
                z0s = z0s.detach()

            assert z0t.shape == z0s.shape and len(z0t.shape) == 4
            if popt_kwargs['sg_combine'] and not popt_kwargs['p_opt_sg']:
                term_1 = (z0t.detach() - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
                term_2 = (z0t - z0s.detach()).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
                ms = term_1 + popt_kwargs['sg_lambda'] * term_2
            else:
                ms = (z0t - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
            loss = -1 * ms.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return c.detach()
    
    @torch.enable_grad()
    def null_text_emb_opt(self, zt, t, step, uc, c, uc_base, cfg_guidance, popt_kwargs):
        decay_rate = popt_kwargs['lr_decay_rate']

        if (1. - step * decay_rate) <= 0:
            print("Learning rate is zero. Skipping text-embedding optimization and using the latest optimized embedding.")
            return uc
        
        uc = torch.nn.Parameter(uc.detach())
        optimizer = Adam([uc], lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))
        
        at = self.alpha(t)

        t_mg = int(
            len(self.scheduler.alphas_cumprod_default) * popt_kwargs['p_ratio']
        )
        if popt_kwargs['dynamic_pr']:
            t_mg = int(
                len(self.scheduler.alphas_cumprod_default) - t
            )

        at_mg = self.scheduler.alphas_cumprod_default[t_mg]
        t_mg = torch.tensor(t_mg).to(t.device)
        
        for i in range(popt_kwargs['p_opt_iter']):
            noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            rand_noise = torch.randn_like(noise_pred, device=noise_pred.device)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
            
            # noise_uc_s, noise_c_s = self.predict_noise(zs, t_mg, uc_base.detach(), c)
            # noise_pred_s = noise_uc_s + cfg_guidance * (noise_c_s - noise_uc_s)
            _, noise_pred_s = self.predict_noise(zs, t_mg, None, c)
            
            # tweedie (x0doublehat)
            z0s = (zs - (1-at_mg).sqrt() * noise_pred_s) / at_mg.sqrt()
            if popt_kwargs['p_opt_sg']:
                z0s = z0s.detach()

            assert z0t.shape == z0s.shape and len(z0t.shape) == 4
            if popt_kwargs['sg_combine'] and not popt_kwargs['p_opt_sg']:
                term_1 = (z0t.detach() - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
                term_2 = (z0t - z0s.detach()).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
                ms = term_1 + popt_kwargs['sg_lambda'] * term_2
            else:
                ms = (z0t - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
            loss = -1 * ms.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return uc.detach()


    @torch.enable_grad()
    def batch_null_text_emb_opt(self, zt, ts, step, uc, c, uc_base, cfg_guidance, popt_kwargs):
        decay_rate = popt_kwargs['lr_decay_rate']
        b_size = len(uc)

        if (1. - step * decay_rate) <= 0:
            print("Learning rate is zero. Skipping text-embedding optimization and using the latest optimized embedding.")
            return uc
        
        uc = torch.nn.Parameter(uc.detach())
        optimizer = Adam([uc], lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))

        at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)

        t_mg = int(
            len(self.scheduler.alphas_cumprod_default) * popt_kwargs['p_ratio']
        )
        if popt_kwargs['dynamic_pr']:
            t_mg = int(
                len(self.scheduler.alphas_cumprod_default) - ts[0].item()
            )
        ts_mg = torch.full((b_size,), t_mg, device=ts.device, dtype=torch.long)
        at_mg = self.scheduler.alphas_cumprod_default[ts_mg].view(b_size, 1, 1, 1)
        
        for i in range(popt_kwargs['p_opt_iter']):
            noise_uc, noise_c = self.predict_noise(zt, ts, uc, c)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            rand_noise = torch.randn_like(noise_pred, device=noise_pred.device)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
            
            # noise_uc_s, noise_c_s = self.predict_noise(zs, ts_mg, uc_base.detach(), c)
            # noise_pred_s = noise_uc_s + cfg_guidance * (noise_c_s - noise_uc_s)
            _, noise_pred_s = self.predict_noise(zs, ts_mg, None, c)
            
            # tweedie (x0doublehat)
            z0s = (zs - (1-at_mg).sqrt() * noise_pred_s) / at_mg.sqrt()
            if popt_kwargs['p_opt_sg']:
                z0s = z0s.detach()

            assert z0t.shape == z0s.shape and len(z0t.shape) == 4
            if popt_kwargs['sg_combine'] and not popt_kwargs['p_opt_sg']:
                term_1 = (z0t.detach() - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
                term_2 = (z0t - z0s.detach()).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
                ms = term_1 + popt_kwargs['sg_lambda'] * term_2
            else:
                ms = (z0t - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
            loss = -1 * ms.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return uc.detach()
    
    @torch.enable_grad()
    def popt_diverse(self, zt, ts, step, placeholder_token_ids_enc, uc, c_base, cfg_guidance, popt_kwargs):
        assert cfg_guidance > 0.0
        placeholder_string = popt_kwargs['placeholder_string']
        decay_rate = popt_kwargs['lr_decay_rate']
        num_opt_tokens = popt_kwargs['num_opt_tokens']

        if (1. - step * decay_rate) <= 0 or popt_kwargs['debug_flag'] == 'no_opt':
            if (1. - step * decay_rate) <= 0:
                print("Learning rate is zero. Skipping prompt optimization and using the latest optimized embedding.")
            else:
                print("Debug flag is set to 'no_opt'. Skipping prompt optimization and using the latest optimized embedding.")
            prompts = self.prompts.copy()
            null_prompts = self.null_prompts.copy()

            # add placeholder tokens only for prompt
            # assert popt_kwargs['num_opt_tokens'] == 1
            assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
            placeholder_symbol = placeholder_string.split("_")[0]
            # prompts = [p + " " + f"{placeholder_symbol}_{idx}" for idx, p in enumerate(prompts)]
            if popt_kwargs['placeholder_position'] == 'end':
                prompts = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompts)]
            elif popt_kwargs['placeholder_position'] == 'start':
                prompts = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompts)]
            # print("Prompts: ", prompts)

            _, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)
            return c
        
        # print("self.text_enc_1.get_input_embeddings().weight.requires_grad: ", self.text_enc_1.get_input_embeddings().weight.requires_grad)
        # print("self.text_enc_2.get_input_embeddings().weight.requires_grad: ", self.text_enc_2.get_input_embeddings().weight.requires_grad)

        para = self.text_encoder.get_input_embeddings().parameters()
        optimizer = Adam(para, lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))
        # print("(before opt) self.text_enc_1.get_input_embeddings().weight.data[0]: ", self.text_enc_1.get_input_embeddings().weight.data[0])
        # print("(before opt) self.text_enc_1.get_input_embeddings().weight.data[49408]: ", self.text_enc_1.get_input_embeddings().weight.data[49408])
        # print("(before opt) self.text_enc_2.get_input_embeddings().weight.data[0]: ", self.text_enc_2.get_input_embeddings().weight.data[0])
        # print("(before opt) self.text_enc_2.get_input_embeddings().weight.data[49408]: ", self.text_enc_2.get_input_embeddings().weight.data[49408])

        # keep original embeddings as reference
        orig_embeds_params_enc = self.text_encoder.get_input_embeddings().weight.data.clone()

        prompts = self.prompts.copy()
        null_prompts = self.null_prompts.copy()
        b_size = len(prompts)
        assert b_size > 1 # batch size should be larger than 1 for diverse generation

        # add placeholder tokens only for prompt
        # assert num_opt_tokens == 1
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
        placeholder_symbol = placeholder_string.split("_")[0]
        # prompts = [p + " " + f"{placeholder_symbol}_{idx}" for idx, p in enumerate(prompts)]
        if popt_kwargs['placeholder_position'] == 'end':
            prompts = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompts)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompts = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompts)]
        # print("Prompts: ", prompts)

        _, c = self.differentiable_get_text_embed(null_prompt=null_prompts, prompt=prompts)

        at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
        
        for i in range(popt_kwargs['p_opt_iter']):
            if (popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0) or popt_kwargs['cfg_traj_opt']:
                noise_uc, noise_c = self.predict_noise(zt, ts, uc.detach(), c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            else:
                _, noise_pred = self.predict_noise(zt, ts, None, c)
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
            loss_per_sample = torch.zeros(b_size, device=zt.device)
            for i in range(b_size):
                for j in range(b_size):
                    if i == j:
                        continue
                    repel_from = z0t[j]
                    if popt_kwargs['p_opt_sg']:
                        repel_from = repel_from.detach()
                    if popt_kwargs['elatentlpips'] is not None:
                        loss_per_sample[i] += popt_kwargs['elatentlpips'](z0t[i].unsqueeze(0), repel_from.unsqueeze(0), normalize=False).squeeze()
                    else:
                        loss_per_sample[i] += (z0t[i] - repel_from).reshape(-1).norm(p=2.0)
            loss = -1 * loss_per_sample.sum() # encouraging z0t to be diverse
            if popt_kwargs['diverse_reg']:
                reg = (c - c_base).norm(p=2.0, dim=-1).mean(dim=-1) # (b, )
                loss += popt_kwargs['reg_lambda'] * reg.sum()
                print("reg: ", reg)
                
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # Let's make sure we don't update any embedding weights besides the newly added token
            self.restore_embedding(placeholder_token_ids_enc, orig_embeds_params_enc, self.tokenizer, self.text_encoder)
            
            if not i == popt_kwargs['p_opt_iter'] - 1:
                _, c = self.differentiable_get_text_embed(null_prompt=null_prompts, prompt=prompts)
            else:
                _, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)

        return c
    
    # Byungjun's implementation of InfoNCE
    @torch.enable_grad()
    def popt_diverse_infoNCE(self, zt, ts, step, placeholder_token_ids_enc, uc, c_base, cfg_guidance, popt_kwargs):
        assert cfg_guidance > 0.0
        placeholder_string = popt_kwargs['placeholder_string']
        decay_rate = popt_kwargs['lr_decay_rate']
        num_opt_tokens = popt_kwargs['num_opt_tokens']

        para = self.text_encoder.get_input_embeddings().parameters()
        optimizer = Adam(para, lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))

        # keep original embeddings as reference
        orig_embeds_params_enc = self.text_encoder.get_input_embeddings().weight.data.clone()

        prompts = self.prompts.copy()
        null_prompts = self.null_prompts.copy()
        b_size = len(prompts)
        assert b_size > 1 # batch size should be larger than 1 for diverse generation

        # add placeholder tokens only for prompt
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
        placeholder_symbol = placeholder_string.split("_")[0]
        if popt_kwargs['placeholder_position'] == 'end':
            prompts = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompts)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompts = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompts)]
        else:
            frag_pt = [1,1,1,1] 
            assert len(frag_pt) == b_size
            for idx, p in enumerate(prompts):
                left_phrase = p.split(" ")[:frag_pt[idx]]
                right_phrase = p.split(" ")[frag_pt[idx]:]
                prompts[idx] = " ".join(left_phrase) + " " + " ".join(f"*_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + " ".join(right_phrase)
        print("Prompts : ")
        
        for p in prompts:
            print(p)
            
        _, c = self.differentiable_get_text_embed(null_prompt=null_prompts, prompt=prompts)

        at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
        
        temperature = popt_kwargs['infoNCE_temp']  # typical value (can tune)
        gamma = popt_kwargs['gamma']
        print("temperature: ", temperature)
        print("gamma: ", gamma)

        with torch.no_grad():
            _, noise_pred_base = self.predict_noise(zt, ts, None, c_base)
        z0t_base = (zt - (1 - at).sqrt() * noise_pred_base) / at.sqrt()  # shape: (B, C, H, W)
        z0t_base = z0t_base.clone().detach()

        w = popt_kwargs['window_size'] # pooling output shape (output shape becomes [B, C, w, w] for w > 1, [B, C] for w = 1)
        for i in range(popt_kwargs['p_opt_iter']):
            _, noise_pred = self.predict_noise(zt, ts, None, c)

            # tweedie (x0hat)
            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()  # shape: (B, C, H, W)

            # import ipdb; ipdb.set_trace()
            if w == 1:
                z0t_base_flat = z0t_base.mean(dim=[2, 3])
                z0t_flat = z0t.mean(dim=[2, 3])

                # Normalize for cosine similarity
                z0t_base_flat = torch.nn.functional.normalize(z0t_base_flat, dim=1)
                z0t_flat = torch.nn.functional.normalize(z0t_flat, dim=1)

                # Similarity matrix (cosine similarity)
                diag_sim_matrix = torch.matmul(z0t_base_flat, z0t_flat.T) / temperature # (B, B) 
                non_diag_sim_matrix = torch.matmul(z0t_flat, z0t_flat.T) / temperature # (B, B)

                sim_matrix = non_diag_sim_matrix.clone()
                diag_indices = torch.arange(b_size)

                # 대각선 성분을 diag_sim_matrix 값으로 덮어쓰기
                sim_matrix[diag_indices, diag_indices] = diag_sim_matrix[diag_indices, diag_indices] / gamma  # alpha > 1이면 positive pair 영향 약해짐
                
                # use stopgrad for testing the impact of postive pair
                # sim_matrix[diag_indices, diag_indices] = diag_sim_matrix[diag_indices, diag_indices].detach() / gamma  # alpha > 1이면 positive pair 영향 약해짐

                labels = torch.arange(b_size, device=z0t.device)
                loss = torch.nn.functional.cross_entropy(sim_matrix, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:
                z0t_base_pooled = torch.nn.functional.adaptive_avg_pool2d(z0t_base, (w, w))  # (B, 4, w, w)
                z0t_pooled = torch.nn.functional.adaptive_avg_pool2d(z0t, (w, w))  # (B, 4, w, w)
                total_loss = 0.0
                for i in range(w):
                    for j in range(w):
                        # Step 2: Collect patch-wise vectors → shape (B, 4)
                        base_patch_vecs = z0t_base_pooled[:, :, i, j]  # (B, C)
                        patch_vecs = z0t_pooled[:, :, i, j]  # (B, C)

                        # Step 3: Normalize for cosine similarity
                        base_patch_vecs = F.normalize(base_patch_vecs, dim=1)  # (B, C)
                        patch_vecs = F.normalize(patch_vecs, dim=1)  # (B, C)

                        # Step 4: Similarity matrix (cosine similarity)
                        diag_sim_matrix = torch.matmul(base_patch_vecs, patch_vecs.T) / temperature # (B, B) 
                        non_diag_sim_matrix = torch.matmul(patch_vecs, patch_vecs.T) / temperature # (B, B)

                        sim_matrix = non_diag_sim_matrix.clone()
                        diag_indices = torch.arange(b_size)

                        # 대각선 성분을 diag_sim_matrix 값으로 덮어쓰기
                        sim_matrix[diag_indices, diag_indices] = diag_sim_matrix[diag_indices, diag_indices] / gamma  # alpha > 1이면 positive pair 영향 약해짐

                        # Step 5: InfoNCE loss
                        labels = torch.arange(b_size, device=sim_matrix.device)
                        loss_ij = F.cross_entropy(sim_matrix, labels)

                        # Accumulate
                        total_loss += loss_ij

                # Step 6: Mean over all patches
                total_loss = total_loss / (w * w)

                # Step 7: Backprop
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            # Let's make sure we don't update any embedding weights besides the newly added token
            self.restore_embedding(placeholder_token_ids_enc, orig_embeds_params_enc, self.tokenizer, self.text_encoder)
            
            if not i == popt_kwargs['p_opt_iter'] - 1:
                _, c = self.differentiable_get_text_embed(null_prompt=null_prompts, prompt=prompts)
            else:
                _, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)

        return c


    @torch.enable_grad()
    def iopt_diverse(self, zt, ts, step, uc, c, cfg_guidance, etc_kwargs):
        b_size = len(zt)
        zt_initial = zt.clone().detach()
                
        if etc_kwargs['iopt_dist']:
            assert len(zt.shape) == 4
            # data_shape = (1, zt.shape[1], zt.shape[2], zt.shape[3])
            data_shape = zt.shape
            log_var, mu = torch.zeros(data_shape, device=self.device), torch.zeros(data_shape, device=self.device) # diagonal covariance
            log_var, mu = log_var.clone().detach().requires_grad_(True), mu.clone().detach().requires_grad_(True)
        
            optimizer = Adam([log_var, mu], lr=etc_kwargs['i_opt_lr'], eps=1e-3)
            zt = zt_initial * (torch.exp(0.5 * log_var)) + mu
            
        elif etc_kwargs['res_opt']:
            assert len(zt.shape) == 4
            data_shape = zt.shape
            res = torch.zeros(data_shape, device=self.device)
            res = res.clone().detach().requires_grad_(True)
            
            optimizer = Adam([res], lr=etc_kwargs['i_opt_lr'], eps=1e-3)
            zt = zt_initial + res
        
        else:
            zt = zt.detach()
            # zt = torch.randn_like(zt, device=self.device)# initialize with random noise
            zt.requires_grad = True
            optimizer = Adam([zt], lr=etc_kwargs['i_opt_lr'])
        
        if etc_kwargs['iopt_lr_decay_rate'] != 1.0:
            scheduler = ExponentialLR(optimizer, gamma=etc_kwargs['iopt_lr_decay_rate'])

        at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
        
        if etc_kwargs['iopt_loss_type'] == 'infoNCE':
            with torch.no_grad():
                _, noise_pred_base = self.predict_noise(zt, ts, None, c)
                z0t_base = (zt - (1 - at).sqrt() * noise_pred_base) / at.sqrt()  # shape: (B, C, H, W)
                z0t_base = z0t_base.clone().detach()
        
        n_aug_samples = etc_kwargs['n_aug_samples']
        if n_aug_samples > 0:
            with torch.no_grad():
                zt_aug = torch.randn(n_aug_samples, zt.shape[1], zt.shape[2], zt.shape[3], device=zt.device, generator=self.generator)
                ts_aug = ts.detach().clone().repeat(n_aug_samples)
                c_aug = c.detach().clone().repeat(n_aug_samples, 1, 1)
                at_aug = self.scheduler.alphas_cumprod[ts_aug].view(n_aug_samples, 1, 1, 1)
                
                _, noise_pred_aug = self.predict_noise(zt_aug, ts_aug, None, c_aug)
                z0t_aug = (zt_aug - (1-at_aug).sqrt() * noise_pred_aug) / at_aug.sqrt()
                
                if etc_kwargs['iopt_loss_type'] == 'infoNCE':
                    z0t_base = torch.cat([z0t_base, z0t_aug], dim=0)

        assert len(zt.shape) == 4

        for i in range(etc_kwargs['i_opt_iter']):
            assert etc_kwargs['iopt_steps_for_z0t'] > 0
            if etc_kwargs['iopt_steps_for_z0t'] > 1:
                num_steps = etc_kwargs['iopt_steps_for_z0t']
                z0t = self.get_z0_ms(zt, c, uc, cfg_guidance, num_steps, etc_kwargs)

            else: # etc_kwargs['iopt_steps_for_z0t'] == 1
                if etc_kwargs['iopt_cfg_tweedie']:
                    if etc_kwargs['iopt_msd']:
                        with torch.no_grad():
                            noise_uc, noise_c = self.predict_noise(zt, ts, uc, c)
                            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
                    else:
                        noise_uc, noise_c = self.predict_noise(zt, ts, uc, c)
                        noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
                
                else:
                    if etc_kwargs['iopt_msd']:
                        with torch.no_grad():
                            print("iopt_msd is True.")
                            _, noise_pred = self.predict_noise(zt, ts, None, c)
                    else:
                        _, noise_pred = self.predict_noise(zt, ts, None, c)
                        
                # tweedie (x0hat)
                z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
                
                if etc_kwargs['Unet_features']:
                    if etc_kwargs['blended_diff']:
                        print("Blending z0t with zt.")
                        b_fac = (1-at).sqrt()
                        z0t_blend = b_fac * z0t + (1-b_fac) * zt
                        self.unet(z0t_blend, ts, encoder_hidden_states=c)
                        # self.predict_noise(z0t_blend, ts, None, c)
                    
                    # z0t_f_list = []
                    # for l_num in range(4):
                    #     z0t_f_list.append(self.features[f'down_block_{l_num}'][-1][-1])
                    # l_num = 0
                    # z0t_f = self.features[f'down_block_{l_num}'][-1][-1] # (b, 1280, 8, 8) for sd15                    
                    z0t_f = extracted_feature
            
            
            # if etc_kwargs['iopt_loss_type'] == 'infoNCE': # Byungjun's implementation of InfoNCE
            #     print("InfoNCE loss.")
            #     temperature = 0.1  # typical value (can tune)
            #     # Global Average Pooling → (B, C)
            #     z0t_flat = z0t.mean(dim=[2, 3])

            #     # Normalize for cosine similarity
            #     z0t_flat = torch.nn.functional.normalize(z0t_flat, dim=1)

            #     # Similarity matrix (cosine similarity)
            #     sim_matrix = torch.matmul(z0t_flat, z0t_flat.T)  # (B, B)

            #     # Scale by temperature
            #     sim_matrix = sim_matrix / temperature  # sharper distribution

            #     # Mask out diagonal (self-similarity) to avoid using them as positives
            #     # mask = torch.eye(b_size, device=z0t.device).bool()
            #     # sim_matrix.masked_fill_(mask, -1e5)

            #     # Use each row as query, and treat the rest as negatives
            #     # Label doesn't matter — goal is to push away other samples
            #     # Maximize entropy: No need for explicit positives
            #     labels = torch.arange(b_size, device=z0t.device)

            #     # print(sim_matrix.shape)
            #     # InfoNCE: Each sample is trying to not align with any of the others
            #     loss = torch.nn.functional.cross_entropy(sim_matrix, labels)
            
            if etc_kwargs['iopt_loss_type'] == 'infoNCE': # Byungjun's implementation of InfoNCE
                print("InfoNCE loss.")
                temperature = etc_kwargs['infoNCE_temp']  # typical value (can tune)
                gamma = etc_kwargs['gamma']
                w = etc_kwargs['window_size'] # pooling output shape (output shape becomes [B, C, w, w] for w > 1, [B, C] for w = 1)
                # _, noise_pred = self.predict_noise(zt, ts, None, c)
        
                # tweedie (x0hat)
                # z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()  # shape: (B, C, H, W)
                
                if n_aug_samples > 0:
                    # augment on existing z0t
                    print("Using augmentation with n_aug_samples: ", n_aug_samples)
                    z0t = torch.cat([z0t, z0t_aug], dim=0)  # (B + n_aug_samples, C, H, W)
                
                # import ipdb; ipdb.set_trace()
                if w == 1:
                    z0t_base_flat = z0t_base.mean(dim=[2, 3])
                    z0t_flat = z0t.mean(dim=[2, 3])
        
                    # Normalize for cosine similarity
                    z0t_base_flat = torch.nn.functional.normalize(z0t_base_flat, dim=1)
                    z0t_flat = torch.nn.functional.normalize(z0t_flat, dim=1)
        
                    # Similarity matrix (cosine similarity)
                    diag_sim_matrix = torch.matmul(z0t_base_flat, z0t_flat.T) / temperature # (B, B) 
                    non_diag_sim_matrix = torch.matmul(z0t_flat, z0t_flat.T) / temperature # (B, B)
        
                    sim_matrix = non_diag_sim_matrix.clone()
                    diag_indices = torch.arange(b_size)
        
                    # 대각선 성분을 diag_sim_matrix 값으로 덮어쓰기
                    sim_matrix[diag_indices, diag_indices] = diag_sim_matrix[diag_indices, diag_indices] / gamma  # alpha > 1이면 positive pair 영향 약해짐
        
                    labels = torch.arange(b_size, device=z0t.device)
                    loss = torch.nn.functional.cross_entropy(sim_matrix.float(), labels)
                    # loss = torch.nn.functional.cross_entropy(sim_matrix, labels)
        
                else:
                    z0t_base_pooled = torch.nn.functional.adaptive_avg_pool2d(z0t_base, (w, w))  # (B, 4, w, w)
                    z0t_pooled = torch.nn.functional.adaptive_avg_pool2d(z0t, (w, w))  # (B, 4, w, w)
                    loss = 0.0

                    B, C, w, _ = z0t_base_pooled.shape
                    P = w * w  # number of patches

                    # Step 1: Flatten spatial dimensions
                    z0t_base_vecs = z0t_base_pooled.view(B, C, P)     # (B, C, P)
                    z0t_vecs = z0t_pooled.view(B, C, P)               # (B, C, P)

                    # Step 2: Normalize across channels
                    z0t_base_vecs = F.normalize(z0t_base_vecs, dim=1)  # (B, C, P)
                    z0t_vecs = F.normalize(z0t_vecs, dim=1)            # (B, C, P)

                    # Step 3: Compute cosine similarity across patches (for InfoNCE)
                    # diag_sim_matrix[b, p] = cosine(z0t_base_vecs[b,:,p], z0t_vecs[b,:,p])
                    diag_sim_matrix = torch.einsum('bcp,bcp->bp', z0t_base_vecs, z0t_vecs) / temperature  # (B, P)

                    # Full similarity: each patch position p has a (B x B) similarity matrix
                    # sim_matrix[p] = cosine(z0t_vecs[:, :, p], z0t_vecs[:, :, p])
                    non_diag_sim_matrix = torch.einsum('bcp, dcp -> pbd', z0t_vecs, z0t_vecs) / temperature  # (P, B, B)
                    sim_matrix = non_diag_sim_matrix.clone()

                    # Step 4: Adjust diagonals
                    diag_indices = torch.arange(B, device=sim_matrix.device)
                    patch_indices = torch.arange(P, device=sim_matrix.device)
                    
                    sim_matrix[patch_indices[:, None], diag_indices, diag_indices] = (diag_sim_matrix / gamma).T  # (P, B)

                    # Step 5: Compute InfoNCE loss
                    labels = torch.arange(B, device=sim_matrix.device)
                    loss_ij = F.cross_entropy(sim_matrix.float(), labels.unsqueeze(0).repeat(P, 1))  # (P,)
                    # loss_ij = F.cross_entropy(sim_matrix, labels.unsqueeze(0).repeat(P, 1))  # (P,)
                    loss = loss_ij.mean()

                    print("Loss:", loss)

                    
            elif etc_kwargs['iopt_loss_type'] == 'attn_entropy':
                if etc_kwargs['attn_where'] == 'all':
                    from_where = ("up", "down", "mid")
                elif etc_kwargs['attn_where'] == 'up':
                    from_where = ("up",)
                elif etc_kwargs['attn_where'] == 'down':
                    from_where = ("down",)
                elif etc_kwargs['attn_where'] == 'mid':
                    from_where = ("mid",)
                
                aggregate_cross_attention_maps = self.attention_store.aggregate_attention(
                    from_where=from_where, is_cross=True)
                
                # cross attention map preprocessing
                assert len(aggregate_cross_attention_maps.shape) == 4 # (b_size, attn_res, attn_res, n_tokens)
                cross_attention_maps = aggregate_cross_attention_maps[:, :, :, 1:-1]
                cross_attention_maps = cross_attention_maps * 100
                
                # interested_token_indices = [1, 4] # astronaut, moon
                # cross_attention_maps = cross_attention_maps[:, :, :, interested_token_indices]
                
                # smoothed_maps = []
                
                # for iii in range(cross_attention_maps.shape[0]):
                #     per_sample = []
                #     for jjj in range(cross_attention_maps.shape[-1]):
                #         smoothed = fn_smoothing_func(cross_attention_maps[iii, :, :, jjj])
                #         per_sample.append(smoothed.unsqueeze(-1))  # shape: (H, W, 1)
                #     per_sample = torch.cat(per_sample, dim=-1)  # shape: (H, W, num_tokens)
                #     smoothed_maps.append(per_sample.unsqueeze(0))  # shape: (1, H, W, num_tokens)

                # cross_attention_maps = torch.cat(smoothed_maps, dim=0)  # shape: (B, H, W, num_tokens)
                
                if etc_kwargs['attn_map_num'] != 'all':
                    attn_map_num = int(etc_kwargs['attn_map_num'])
                    assert attn_map_num <= cross_attention_maps.shape[-1] and attn_map_num > 0
                    
                    # selecting the most significant 10 tokens
                    resp_strength = cross_attention_maps.mean(dim=[0, 1, 2])
                    _, str_idx = resp_strength.topk(attn_map_num)
                    cross_attention_maps = cross_attention_maps[:, :, :, str_idx]
                
                cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)
                if i == 0:
                    cross_attention_maps_initial = cross_attention_maps.clone().detach()
                
                losses = torch.zeros(b_size, device=zt.device)
                for i in range(b_size):
                    for j in range(b_size):
                        if i == j:
                            if etc_kwargs['attn_reg']:
                                ce = -1 * cross_attention_maps_initial[i] * torch.log(cross_attention_maps[i])
                                ce = ce.sum(dim=-1) # (attn_res, attn_res)
                                ce = ce.mean()
                                losses[i] += ce # entropy minimization
                            else:
                                continue
                        if etc_kwargs['i_opt_sg']:
                            ce = -1 * zt_ca_maps[j].detach() * torch.log(zt_ca_maps[i])
                        else:
                            ce = -1 * cross_attention_maps[j] * torch.log(cross_attention_maps[i])
                        ce = ce.sum(dim=-1) # (attn_res, attn_res)
                        ce = ce.mean()
                        losses[i] += -1 * ce # entropy maximization
                
                if etc_kwargs['lambda_sa'] != 0.0:
                    print("Self-attention entropy maximization.")
                    zt_sa_maps = self.attention_store.aggregate_attention(
                            from_where=from_where, is_cross=False)
                    for i in range(b_size):
                        for j in range(b_size):
                            if i == j:
                                continue
                            if etc_kwargs['i_opt_sg']:
                                ce = -1 * zt_sa_maps[j].detach() * torch.log(zt_sa_maps[i])
                            else:
                                ce = -1 * zt_sa_maps[j] * torch.log(zt_sa_maps[i])
                            ce = ce.sum(dim=-1) # (attn_res, attn_res)
                            ce = ce.mean()
                            losses[i] += -1 * etc_kwargs['lambda_sa'] * ce # entropy maximization
                        
                loss = losses.sum()
                
            if etc_kwargs['iopt_loss_type'] == 'self_attn_entropy':
                if etc_kwargs['attn_where'] == 'all':
                    from_where = ("up", "down", "mid")
                elif etc_kwargs['attn_where'] == 'up':
                    from_where = ("up",)
                elif etc_kwargs['attn_where'] == 'down':
                    from_where = ("down",)
                elif etc_kwargs['attn_where'] == 'mid':
                    from_where = ("mid",)
                    
                zt_sa_maps = self.attention_store.aggregate_attention(
                        from_where=from_where, is_cross=False)
                
                losses = torch.zeros(b_size, device=zt.device)
                for i in range(b_size):
                    for j in range(b_size):
                        if i == j:
                            continue
                        if etc_kwargs['i_opt_sg']:
                            ce = -1 * zt_sa_maps[j].detach() * torch.log(zt_sa_maps[i])
                        else:
                            ce = -1 * zt_sa_maps[j] * torch.log(zt_sa_maps[i])
                        ce = ce.sum(dim=-1) # (attn_res, attn_res)
                        ce = ce.mean()
                        losses[i] += -1 * ce # entropy maximization
                        
                loss = losses.sum()
                
            if etc_kwargs['iopt_loss_type'] == 'dist_eps':
                loss_per_sample = torch.zeros(b_size, device=zt.device)
                for i in range(b_size):
                    for j in range(b_size):
                        if i == j:
                            continue
                        repel_from = noise_pred[j].clone()
                        if etc_kwargs['i_opt_sg'] or b_size == 1:
                            # print("Detaching repel_from.")
                            repel_from = repel_from.detach()
                        
                        loss_per_sample[i] += (noise_pred[i] - repel_from).reshape(-1).norm(p=2.0)
                        
                loss = -1 * loss_per_sample.sum() # encouraging z0t to be diverse
                
            if etc_kwargs['iopt_loss_type'] == 'cossim_eps':
                loss_per_sample = torch.zeros(b_size, device=zt.device)
                noise_pred = noise_pred / noise_pred.norm(p=2.0, dim=[1,2,3], keepdim=True)
                for i in range(b_size):
                    for j in range(b_size):
                        if i == j:
                            continue
                        repel_from = noise_pred[j].clone()
                        if etc_kwargs['i_opt_sg'] or b_size == 1:
                            # print("Detaching repel_from.")
                            repel_from = repel_from.detach()                        
                        
                        loss_per_sample[i] += (noise_pred[i] * repel_from).sum()
                        
                loss = loss_per_sample.sum() # encouraging z0t to be diverse

            if etc_kwargs['iopt_loss_type'] == 'dist':
                if etc_kwargs['Unet_features']:
                    z0t = z0t_f
                loss_per_sample = torch.zeros(b_size, device=zt.device)
                for i in range(b_size):
                    for j in range(b_size):
                        if i == j:
                            continue
                        repel_from = z0t[j].clone()
                        if etc_kwargs['i_opt_sg'] or b_size == 1:
                            # print("Detaching repel_from.")
                            repel_from = repel_from.detach()
                        if etc_kwargs['elatentlpips'] is not None:
                            loss_per_sample[i] += etc_kwargs['elatentlpips'](z0t[i].unsqueeze(0), repel_from.unsqueeze(0), normalize=False).squeeze()
                        else:
                            loss_per_sample[i] += (z0t[i] - repel_from).reshape(-1).norm(p=2.0)
                        
                loss = -1 * loss_per_sample.sum() # encouraging z0t to be diverse
                # print("losses: ", loss_per_sample)

                if etc_kwargs['iopt_ms_lambda'] != 0.0:
                    t_mg = int(
                    len(self.scheduler.alphas_cumprod_default) * etc_kwargs['ims_p_ratio']
                    )
                    ts_mg = torch.full((b_size,), t_mg, device=ts.device, dtype=torch.long)
                    at_mg = self.scheduler.alphas_cumprod_default[ts_mg].view(b_size, 1, 1, 1)

                    # add noise
                    noise_shape = noise_pred.shape
                    rand_noise = torch.randn(noise_shape, device=noise_pred.device, generator=self.generator)
                    zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
                    
                    if etc_kwargs['iopt_cfg_tweedie']:
                        ## for non-Ampere gpus
                        # noise_s_uc = self.unet(zs, ts_mg, encoder_hidden_states=uc)['sample']
                        # noise_s_c = self.unet(zs, ts_mg, encoder_hidden_states=c)['sample']
                        noise_s_uc, noise_s_c = self.predict_noise(zs, ts_mg, uc, c)
                        noise_pred_s = noise_s_uc + cfg_guidance * (noise_s_c - noise_s_uc)
                    else:
                        _, noise_pred_s = self.predict_noise(zs, ts_mg, None, c.detach())
                    
                    # tweedie (x0doublehat)
                    z0s = (zs - (1-at_mg).sqrt() * noise_pred_s) / at_mg.sqrt()
                    # if etc_kwargs['i_opt_sg']:
                        # z0s = z0s.detach()

                    assert z0t.shape == z0s.shape and len(z0t.shape) == 4
                    if etc_kwargs['elatentlpips'] is not None:
                        ms = etc_kwargs['elatentlpips'](z0t, z0s, normalize=False)

                    else:
                        ms = (z0t - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
                    ms_reg = ms.sum() #* -1
                    loss += etc_kwargs['iopt_ms_lambda'] * ms_reg

            elif etc_kwargs['iopt_loss_type'] == 'contrastive':
                # get z0s
                t_mg = int(
                    len(self.scheduler.alphas_cumprod_default) * etc_kwargs['ims_p_ratio']
                )
                ts_mg = torch.full((b_size,), t_mg, device=ts.device, dtype=torch.long)
                at_mg = self.scheduler.alphas_cumprod_default[ts_mg].view(b_size, 1, 1, 1)

                # add noise
                noise_shape = zt.shape
                rand_noise = torch.randn(noise_shape, device=self.device, generator=self.generator)
                zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
                
                if etc_kwargs['iopt_cfg_tweedie']:
                    ## for non-Ampere gpus
                    # noise_s_uc = self.unet(zs, ts_mg, encoder_hidden_states=uc)['sample']
                    # noise_s_c = self.unet(zs, ts_mg, encoder_hidden_states=c)['sample']
                    noise_s_uc, noise_s_c = self.predict_noise(zs, ts_mg, uc, c)
                    noise_pred_s = noise_s_uc + cfg_guidance * (noise_s_c - noise_s_uc)
                else:
                    _, noise_pred_s = self.predict_noise(zs, ts_mg, None, c.detach())
                
                # tweedie (x0doublehat)
                z0s = (zs - (1-at_mg).sqrt() * noise_pred_s) / at_mg.sqrt()
                if etc_kwargs['Unet_features']:
                    if etc_kwargs['blended_diff']:
                        print("Blending z0s with zs.")
                        b_fac = (1-at_mg).sqrt()
                        z0s_blend = b_fac * z0s + (1-b_fac) * zs
                        self.unet(z0s_blend, ts_mg, encoder_hidden_states=c)
                        
                    # z0s_f_list = []
                    # for l_num in range(4):
                    #     z0s_f_list.append(self.features[f'down_block_{l_num}'][-1][-1])
                    # z0s_f = self.features[f'down_block_{l_num}'][-1][-1] # (b, 1280, 8, 8) for sd15
                    z0s_f = extracted_feature
                
                assert (etc_kwargs['elatentlpips'] == None) or (etc_kwargs['dreamsim'] == None)
                
                if etc_kwargs['elatentlpips'] is not None:
                    # # ellpips-allfeat
                    # # map to feature space of ellpips
                    
                    # z0t_list = etc_kwargs['elatentlpips'].net.forward(z0t)
                    # z0s_list = etc_kwargs['elatentlpips'].net.forward(z0s)

                    # loss = 0.0
                    # for z0t_now, z0s_now in zip(z0t_list, z0s_list):
                    #     if etc_kwargs['ellpips_pooling'] != 'none':
                    #         if etc_kwargs['ellpips_pooling'] == 'mean':
                    #             z0t_now = z0t_now.mean(dim=1) # average the channels
                    #             z0s_now = z0s_now.mean(dim=1) # average the channels
                    #         elif etc_kwargs['ellpips_pooling'] == 'max':
                    #             z0t_now = z0t_now.max(dim=1)[0]
                    #             z0s_now = z0s_now.max(dim=1)[0]
                    #         elif etc_kwargs['ellpips_pooling'] == 'median':
                    #             z0t_now = z0t_now.median(dim=1)[0]
                    #             z0s_now = z0s_now.median(dim=1)[0]
                    #         elif etc_kwargs['ellpips_pooling'] == 'sum':
                    #             z0t_now = z0t_now.sum(dim=1)
                    #             z0s_now = z0s_now.sum(dim=1)
                    #         elif etc_kwargs['ellpips_pooling'] == 'min':
                    #             z0t_now = z0t_now.min(dim=1)[0]
                    #             z0s_now = z0s_now.min(dim=1)[0]
                    #         z0t_now = z0t_now.unsqueeze(1)
                    #         z0s_now = z0s_now.unsqueeze(1)
                    #     # normalize z0t and z0s (without it, the optimization could break)
                    #     z0t_normed = z0t_now / z0t_now.norm(p=2.0, dim=[1,2,3], keepdim=True)
                    #     z0s_normed = z0s_now / z0s_now.norm(p=2.0, dim=[1,2,3], keepdim=True)

                    #     # similarities with positive pairs
                    #     pos_sim = ((z0t_normed * z0s_normed).sum(dim=[1,2,3]) / etc_kwargs['iopt_temp']).exp()

                    #     # similarities with negative pairs
                    #     neg_sim = torch.zeros(b_size, device=zt.device)
                    #     for i in range(b_size):
                    #         for j in range(b_size):
                    #             # similarity between z0t[i] and z0s[j]
                    #             if i == j:
                    #                 continue
                    #             else:
                    #                 neg_sim[i] += ((z0t_normed[i] * z0t_normed[j]).sum() / etc_kwargs['iopt_temp']).exp()
                    #                 if etc_kwargs['neg_aug']:
                    #                     neg_sim[i] += ((z0t_normed[i] * z0s_normed[j]).sum() / etc_kwargs['iopt_temp']).exp()

                    #     # compute loss
                    #     loss += -1 * (pos_sim / (pos_sim + neg_sim)).log().sum()

                    # map to feature space of ellpips
                    z0t = etc_kwargs['elatentlpips'].net.forward(z0t)[0]
                    z0s = etc_kwargs['elatentlpips'].net.forward(z0s)[0]
                    
                    if etc_kwargs['ellpips_pooling'] != 'none':
                        if etc_kwargs['ellpips_pooling'] == 'mean':
                            z0t = z0t.mean(dim=1) # average the channels
                            z0s = z0s.mean(dim=1) # average the channels
                        elif etc_kwargs['ellpips_pooling'] == 'max':
                            z0t = z0t.max(dim=1)[0]
                            z0s = z0s.max(dim=1)[0]
                        elif etc_kwargs['ellpips_pooling'] == 'median':
                            z0t = z0t.median(dim=1)[0]
                            z0s = z0s.median(dim=1)[0]
                        elif etc_kwargs['ellpips_pooling'] == 'sum':
                            z0t = z0t.sum(dim=1)
                            z0s = z0s.sum(dim=1)
                        elif etc_kwargs['ellpips_pooling'] == 'min':
                            z0t = z0t.min(dim=1)[0]
                            z0s = z0s.min(dim=1)[0]
                        z0t = z0t.unsqueeze(1)
                        z0s = z0s.unsqueeze(1)

                    # normalize z0t and z0s (without it, the optimization could break)
                    z0t_normed = z0t / z0t.norm(p=2.0, dim=[1,2,3], keepdim=True)
                    z0s_normed = z0s / z0s.norm(p=2.0, dim=[1,2,3], keepdim=True)

                    # similarities with positive pairs
                    pos_sim = ((z0t_normed * z0s_normed).sum(dim=[1,2,3]) / etc_kwargs['iopt_temp']).exp()

                    # similarities with negative pairs
                    neg_sim = torch.zeros(b_size, device=zt.device)
                    for i in range(b_size):
                        for j in range(b_size):
                            # similarity between z0t[i] and z0s[j]
                            if i == j:
                                continue
                            else:
                                neg_sim[i] += ((z0t_normed[i] * z0t_normed[j]).sum() / etc_kwargs['iopt_temp']).exp()
                                if etc_kwargs['neg_aug']:
                                    neg_sim[i] += ((z0t_normed[i] * z0s_normed[j]).sum() / etc_kwargs['iopt_temp']).exp()

                    # compute loss
                    loss = -1 * (pos_sim / (pos_sim + neg_sim)).log().sum()
                    
                elif etc_kwargs['dreamsim'] is not None:
                    pred_x0t_img, pred_x0s_img = self.decode_latents(z0t), self.decode_latents(z0s)
                    dreamsim_x0t_embs = etc_kwargs['dreamsim'][0](
                        etc_kwargs['dreamsim'][2](pred_x0t_img.float())
                    )
                    dreamsim_x0s_embs = etc_kwargs['dreamsim'][0](
                        etc_kwargs['dreamsim'][2](pred_x0s_img.float())
                    )
                    # import ipdb; ipdb.set_trace()
                    dreamsim_x0t_embs_normed = dreamsim_x0t_embs / dreamsim_x0t_embs.norm(p=2.0, dim=-1, keepdim=True)
                    dreamsim_x0s_embs_normed = dreamsim_x0s_embs / dreamsim_x0s_embs.norm(p=2.0, dim=-1, keepdim=True)
                    
                    # similarities with positive pairs
                    pos_sim = ((dreamsim_x0t_embs_normed * dreamsim_x0s_embs_normed).sum(dim=-1) / etc_kwargs['iopt_temp']).exp()

                    # similarities with negative pairs
                    neg_sim = torch.zeros(b_size, device=zt.device)
                    for i in range(b_size):
                        for j in range(b_size):
                            # similarity between z0t[i] and z0s[j]
                            if i == j:
                                continue
                            else:
                                neg_sim[i] += ((dreamsim_x0t_embs_normed[i] * dreamsim_x0t_embs_normed[j]).sum() / etc_kwargs['iopt_temp']).exp()
                                if etc_kwargs['neg_aug']:
                                    neg_sim[i] += ((dreamsim_x0t_embs_normed[i] * dreamsim_x0s_embs_normed[j]).sum() / etc_kwargs['iopt_temp']).exp()

                    # compute loss
                    loss = -1 * (pos_sim / (pos_sim + neg_sim)).log().sum()

                else:
                    if etc_kwargs['Unet_features']:
                        loss = 0.0
                        # for z0t_f, z0s_f in zip(z0t_f_list, z0s_f_list):
                            # normalize z0t and z0s (without it, the optimization could break)
                        z0t_normed = z0t_f / z0t_f.norm(p=2.0, dim=[1,2,3], keepdim=True)
                        z0s_normed = z0s_f / z0s_f.norm(p=2.0, dim=[1,2,3], keepdim=True)

                        # similarities with positive pairs
                        pos_sim = ((z0t_normed * z0s_normed).sum(dim=[1,2,3]) / etc_kwargs['iopt_temp']).exp()

                        # similarities with negative pairs
                        neg_sim = torch.zeros(b_size, device=zt.device)
                        for i in range(b_size):
                            for j in range(b_size):
                                # similarity between z0t[i] and z0s[j]
                                if i == j:
                                    continue
                                else:
                                    neg_sim[i] += ((z0t_normed[i] * z0t_normed[j]).sum() / etc_kwargs['iopt_temp']).exp()
                                    if etc_kwargs['neg_aug']:
                                        neg_sim[i] += ((z0t_normed[i] * z0s_normed[j]).sum() / etc_kwargs['iopt_temp']).exp()

                        # compute loss
                        loss += -1 * (pos_sim / (pos_sim + neg_sim)).log().sum()
                        
                    else:
                        # normalize z0t and z0s (without it, the optimization could break)
                        z0t_normed = z0t / z0t.norm(p=2.0, dim=[1,2,3], keepdim=True)
                        z0s_normed = z0s / z0s.norm(p=2.0, dim=[1,2,3], keepdim=True)

                        # similarities with positive pairs
                        pos_sim = ((z0t_normed * z0s_normed).sum(dim=[1,2,3]) / etc_kwargs['iopt_temp']).exp()

                        # similarities with negative pairs
                        neg_sim = torch.zeros(b_size, device=zt.device)
                        for i in range(b_size):
                            for j in range(b_size):
                                # similarity between z0t[i] and z0s[j]
                                if i == j:
                                    continue
                                else:
                                    neg_sim[i] += ((z0t_normed[i] * z0t_normed[j]).sum() / etc_kwargs['iopt_temp']).exp()
                                    if etc_kwargs['neg_aug']:
                                        neg_sim[i] += ((z0t_normed[i] * z0s_normed[j]).sum() / etc_kwargs['iopt_temp']).exp()

                        # compute loss
                        loss = -1 * (pos_sim / (pos_sim + neg_sim)).log().sum()
                    
            elif etc_kwargs['iopt_loss_type'] == 'contrastive_v2':
                # construct positive and negative pairs with augmentations by ellpips
                assert etc_kwargs['elatentlpips'] is not None
                z0t_aug, _ = etc_kwargs['elatentlpips'].augment(z0t, z0t.detach())
                # import ipdb; ipdb.set_trace()
                z0t = etc_kwargs['elatentlpips'].net.forward(z0t)[0]
                z0t_aug = etc_kwargs['elatentlpips'].net.forward(z0t_aug)[0]
                
                z0t_normed = z0t / z0t.norm(p=2.0, dim=[1,2,3], keepdim=True)
                z0t_aug_normed = z0t_aug / z0t_aug.norm(p=2.0, dim=[1,2,3], keepdim=True)
                
                pos_sim = ((z0t_normed * z0t_aug_normed).sum(dim=[1,2,3]) / etc_kwargs['iopt_temp']).exp()
                
                # similarities with negative pairs
                neg_sim = torch.zeros(b_size, device=zt.device)
                for i in range(b_size):
                    for j in range(b_size):
                        # similarity between z0t[i] and z0s[j]
                        if i == j:
                            continue
                        else:
                            neg_sim[i] += ((z0t_normed[i] * z0t_aug_normed[j]).sum() / etc_kwargs['iopt_temp']).exp()
                            if etc_kwargs['neg_aug']:
                                neg_sim[i] += ((z0t_normed[i] * z0t_normed[j]).sum() / etc_kwargs['iopt_temp']).exp()

                # compute loss
                loss = -1 * (pos_sim / (pos_sim + neg_sim)).log().sum()
                
            elif etc_kwargs['iopt_loss_type'] == 'contrastive_v3':
                z0t = etc_kwargs['elatentlpips'].net.forward(z0t)[0]
                z0t_normed = z0t / z0t.norm(p=2.0, dim=[1,2,3], keepdim=True)
                
                z0t_initial = (zt_initial - (1-at).sqrt() * noise_pred) / at.sqrt()
                z0t_initial = etc_kwargs['elatentlpips'].net.forward(z0t_initial)[0]
                z0t_initial_normed = z0t_initial / z0t_initial.norm(p=2.0, dim=[1,2,3], keepdim=True)
                
                # similarities with positive pairs
                pos_sim = ((z0t_normed * z0t_initial_normed).sum(dim=[1,2,3]) / etc_kwargs['iopt_temp']).exp()
                
                # similarities with negative pairs
                neg_sim = torch.zeros(b_size, device=zt.device)
                for i in range(b_size):
                    for j in range(b_size):
                        # similarity between z0t[i] and z0s[j]
                        if i == j:
                            continue
                        else:
                            neg_sim[i] += ((z0t_normed[i] * z0t_normed[j]).sum() / etc_kwargs['iopt_temp']).exp()

                # compute loss
                loss = -1 * (pos_sim / (pos_sim + neg_sim)).log().sum()
                
            elif etc_kwargs['iopt_loss_type'] == 'contrastive_patch':
                # get z0s
                t_mg = int(
                    len(self.scheduler.alphas_cumprod_default) * etc_kwargs['ims_p_ratio']
                )
                ts_mg = torch.full((b_size,), t_mg, device=ts.device, dtype=torch.long)
                at_mg = self.scheduler.alphas_cumprod_default[ts_mg].view(b_size, 1, 1, 1)

                # add noise
                noise_shape = zt.shape
                rand_noise = torch.randn(noise_shape, device=self.device, generator=self.generator)
                zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
                
                if etc_kwargs['iopt_cfg_tweedie']:
                    ## for non-Ampere gpus
                    # noise_s_uc = self.unet(zs, ts_mg, encoder_hidden_states=uc)['sample']
                    # noise_s_c = self.unet(zs, ts_mg, encoder_hidden_states=c)['sample']
                    noise_s_uc, noise_s_c = self.predict_noise(zs, ts_mg, uc, c)
                    noise_pred_s = noise_s_uc + cfg_guidance * (noise_s_c - noise_s_uc)
                else:
                    _, noise_pred_s = self.predict_noise(zs, ts_mg, None, c.detach())
                
                # tweedie (x0doublehat)
                z0s = (zs - (1-at_mg).sqrt() * noise_pred_s) / at_mg.sqrt()
                
                if etc_kwargs['elatentlpips'] is not None:
                    z0t = etc_kwargs['elatentlpips'].net.forward(z0t)[0]
                    z0s = etc_kwargs['elatentlpips'].net.forward(z0s)[0]
                    
                b_size, num_ch, _, _ = z0t.shape
                # import ipdb; ipdb.set_trace()
                patch_size = etc_kwargs.get('patch_size', 16)
                stride = etc_kwargs.get('stride', patch_size)
                # print("patch_size: ", patch_size)
                
                # Unfold into patches
                z0t_patches = z0t.unfold(2, patch_size, stride).unfold(3, patch_size, stride)  # (b_size, c, num_patches_h, num_patches_w, patch_size, patch_size)
                z0s_patches = z0s.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    
                # Reshape to (b_size, num_patches, c, patch_size, patch_size)
                z0t_patches = z0t_patches.permute(0, 2, 3, 1, 4, 5).reshape(b_size, -1, num_ch, patch_size, patch_size)
                z0s_patches = z0s_patches.permute(0, 2, 3, 1, 4, 5).reshape(b_size, -1, num_ch, patch_size, patch_size)
                
                # import ipdb; ipdb.set_trace()
                # Normalize patches
                z0t_normed = z0t_patches.flatten(2) / z0t_patches.flatten(2).norm(p=2.0, dim=2, keepdim=True)
                z0s_normed = z0s_patches.flatten(2) / z0s_patches.flatten(2).norm(p=2.0, dim=2, keepdim=True)
                
                num_patches = z0t_patches.shape[1]
                
                pos_sim = torch.zeros(b_size, device=z0t.device)
                for i in range(b_size):
                    for j in range(num_patches):
                        for k in range(num_patches):
                            if j <= k:
                                pos_sim[i] += ((z0t_normed[i, j] * z0s_normed[i, k]).sum() / etc_kwargs['iopt_temp']).exp()
                                
                neg_sim = torch.zeros(b_size, device=z0t.device)
                for ii in range(b_size):
                    for jj in range(b_size):
                        if ii != jj:
                            for j in range(num_patches):
                                for k in range(num_patches):
                                    if j <= k:
                                        neg_sim[ii] += ((z0t_normed[ii, j] * z0s_normed[jj, k]).sum() / etc_kwargs['iopt_temp']).exp()
                                        
                # compute loss
                loss = -1 * (pos_sim / (pos_sim + neg_sim)).log().sum()
                
            elif 'dist_NN' in etc_kwargs['iopt_loss_type']:
                assert etc_kwargs['elatentlpips'] is not None
                with torch.no_grad():
                    z0t_lpips = etc_kwargs['elatentlpips'].net.forward(z0t)[0] # (b_size, 4, 64, 64)
                
                # z0t: (b_size, channels, height, width)
                # Flatten each sample in the batch to (b_size, -1) for pairwise distance calculation
                b_size = z0t_lpips.shape[0]
                z0t_lpips_flat = z0t_lpips.view(b_size, -1)  # Shape: (b_size, features)

                # Normalize for cosine similarity, if needed
                # z0t_flat = F.normalize(z0t_flat, p=2, dim=1)

                # Compute pairwise distances using squared L2 norm (optional: use cosine similarity instead)
                dist_matrix = torch.cdist(z0t_lpips_flat, z0t_lpips_flat, p=2)  # Shape: (b_size, b_size)

                # Mask diagonal to ignore self-distances
                dist_matrix.fill_diagonal_(float('inf'))

                # Find the nearest neighbor indices and distances
                NN_distances, NN_indices = dist_matrix.min(dim=1)
                # print("NN_distances: ", NN_distances)
                # print("NN_indices: ", NN_indices)
                # import ipdb; ipdb.set_trace()
                
                repel_from = z0t[NN_indices]
                if 'sg' in etc_kwargs['iopt_loss_type']:
                    # print("Detaching repel_from.")
                    repel_from = repel_from.detach()
                    
                losses = etc_kwargs['elatentlpips'](z0t, repel_from, normalize=False)
                loss = -1 * losses.sum()
                
                if etc_kwargs['iopt_ms_lambda'] != 0.0:
                    t_mg = int(
                    len(self.scheduler.alphas_cumprod_default) * etc_kwargs['ims_p_ratio']
                    )
                    ts_mg = torch.full((b_size,), t_mg, device=ts.device, dtype=torch.long)
                    at_mg = self.scheduler.alphas_cumprod_default[ts_mg].view(b_size, 1, 1, 1)

                    # add noise
                    noise_shape = noise_pred.shape
                    rand_noise = torch.randn(noise_shape, device=noise_pred.device, generator=self.generator)
                    zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
                    
                    if etc_kwargs['iopt_cfg_tweedie']:
                        ## for non-Ampere gpus
                        # noise_s_uc = self.unet(zs, ts_mg, encoder_hidden_states=uc)['sample']
                        # noise_s_c = self.unet(zs, ts_mg, encoder_hidden_states=c)['sample']
                        noise_s_uc, noise_s_c = self.predict_noise(zs, ts_mg, uc, c)
                        noise_pred_s = noise_s_uc + cfg_guidance * (noise_s_c - noise_s_uc)
                    else:
                        _, noise_pred_s = self.predict_noise(zs, ts_mg, None, c.detach())
                    
                    # tweedie (x0doublehat)
                    z0s = (zs - (1-at_mg).sqrt() * noise_pred_s) / at_mg.sqrt()
                    # if etc_kwargs['i_opt_sg']:
                        # z0s = z0s.detach()

                    assert z0t.shape == z0s.shape and len(z0t.shape) == 4
                    if etc_kwargs['elatentlpips'] is not None:
                        ms = etc_kwargs['elatentlpips'](z0t, z0s, normalize=False)

                    else:
                        ms = (z0t - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
                    ms_reg = ms.sum() #* -1
                    loss += etc_kwargs['iopt_ms_lambda'] * ms_reg
                
            elif 'dist_r' in etc_kwargs['iopt_loss_type']:
                assert etc_kwargs['iopt_r_th'] != float('inf')
                assert etc_kwargs['elatentlpips'] is not None
                with torch.no_grad():
                    z0t_lpips = etc_kwargs['elatentlpips'].net.forward(z0t)[0] # (b_size, 4, 64, 64)
                
                b_size = z0t_lpips.shape[0]
                z0t_lpips_flat = z0t_lpips.view(b_size, -1)
                
                z0t_lpips_flat_normalize = F.normalize(z0t_lpips_flat, p=2, dim=1)
                
                # Compute pairwise distances
                dist_matrix = torch.cdist(z0t_lpips_flat_normalize, z0t_lpips_flat_normalize, p=2)
                
                # Mask diagonal to ignore self-distances
                dist_matrix.fill_diagonal_(float('inf'))
                print("dist_matrix: ", dist_matrix)
                
                # escape the optimization if there is no entry in the matrix satisfying the condition
                if dist_matrix[dist_matrix < etc_kwargs['iopt_r_th']].numel() == 0:
                    print("No entry in the matrix satisfies the condition. Skipping the optimization.")
                    break
                
                loss = 0.0
                for i in range(b_size):
                    for j in range(b_size):
                        if i == j:
                            continue
                        # incorporate close instances into the loss functinon
                        if dist_matrix[i][j] < etc_kwargs['iopt_r_th']:
                            repel_from = z0t[j]
                            if 'sg' in etc_kwargs['iopt_loss_type']:
                                # print("Detaching repel_from.")
                                repel_from = repel_from.detach()
                            loss += etc_kwargs['elatentlpips'](z0t[i].unsqueeze(0), repel_from.unsqueeze(0), normalize=False).squeeze()
                
                loss = -1 * loss
                
            optimizer.zero_grad()
            loss.backward()
            # print("zt.norm():", zt.norm().item())
            # print("zt.grad.norm():", zt.grad.norm().item())
            optimizer.step()
            
            if etc_kwargs['entropy_reg_lambda'] != 0:
                self.attention_store = AttentionStore(attn_res=etc_kwargs['attn_res'], num_heads=self.unet.config.attention_head_dim)
                self.register_attention_control(etc_kwargs['attn_where'], etc_kwargs['attn_type'])
                
                self.predict_noise(zt, ts, None, c)

                if etc_kwargs['attn_where'] == 'all':
                    from_where = ("up", "down", "mid")
                elif etc_kwargs['attn_where'] == 'up':
                    from_where = ("up",)
                elif etc_kwargs['attn_where'] == 'down':
                    from_where = ("down",)
                elif etc_kwargs['attn_where'] == 'mid':
                    from_where = ("mid",)
                
                aggregate_cross_attention_maps = self.attention_store.aggregate_attention(
                    from_where=from_where, is_cross=True)
                
                # cross attention map preprocessing
                assert len(aggregate_cross_attention_maps.shape) == 4
                cross_attention_maps = aggregate_cross_attention_maps[:, :, :, 1:-1]
                cross_attention_maps = cross_attention_maps * 100
                zt_ca_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)
                
                losses = torch.zeros(b_size, device=zt.device)
                for i in range(b_size):
                    for j in range(b_size):
                        if i == j:
                            continue
                        ce = -1 * zt_ca_maps[j].detach() * torch.log(zt_ca_maps[i])
                        ce = ce.sum(dim=-1) # (attn_res, attn_res)
                        ce = ce.mean()
                        losses[i] += -1 * ce # entropy maximization
                        
                loss = losses.sum() * etc_kwargs['entropy_reg_lambda']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # del self.unet
                # self.unet = self.unet_base
                print("Entropy regularization is done.")
            
            if etc_kwargs['kld_reg_lambda'] != 0.0:
                print("Using KLD regularization.")
                assert etc_kwargs['iopt_dist']
                kld_loss = self.fn_calc_kld_loss_func(log_var, mu) # returns (b_size, )
                print("kld_loss: ", kld_loss)
                loss = etc_kwargs['kld_reg_lambda'] * kld_loss.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if etc_kwargs['w_reg_lambda'] != 0.0:
                print("Using Wasserstein regularization.")
                assert etc_kwargs['iopt_dist']
                w_loss = self.fn_calc_W_loss_func(log_var, mu)
                print("w_loss: ", w_loss)
                loss = etc_kwargs['w_reg_lambda'] * w_loss.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if etc_kwargs['res_reg_lambda'] != 0.0:
                assert etc_kwargs['res_opt']
                res_losses = (res ** 2).sum(dim=[1,2,3])
                print("res_losses: ", res_losses)
                loss = etc_kwargs['res_reg_lambda'] * res_losses.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if etc_kwargs['norm_reg_lambda'] != 0.0:
                z_dim = zt.shape[1] * zt.shape[2] * zt.shape[3]
                # losses = ((zt ** 2).sum(dim=[1,2,3]) - z_dim) ** 2
                losses = (zt.norm(p=2.0, dim=[1,2,3]) - math.sqrt(z_dim)) ** 2
                print("norm_losses: ", losses)
                loss = etc_kwargs['norm_reg_lambda'] * losses.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if etc_kwargs['norm_reg']:
                assert not etc_kwargs['iopt_dist']
                z_dim = zt.shape[1] * zt.shape[2] * zt.shape[3]
                
                norm_th = math.sqrt(z_dim)
                norm_th_up = norm_th + etc_kwargs['norm_offset']
                norm_th_down = norm_th - etc_kwargs['norm_offset']
                
                while True:
                    z_norm = zt.view(b_size, z_dim).norm(p=2.0, dim=1)
                    print("z_norm: ", z_norm)
                    mask = (z_norm > norm_th_up) | (z_norm < norm_th_down)
                    if not mask.any():
                        break
                    optimizer.zero_grad()
                    # norm_loss = ((z_norm[mask] - norm_th) ** 2).sum()
                    norm_loss = (z_norm[mask] - norm_th).abs().sum()
                    norm_loss.backward()
                    optimizer.step()
                    
            if etc_kwargs['prox_reg']:
                assert not etc_kwargs['iopt_dist']
                
                while True:
                    # losses = (zt - zt_initial).norm(p=2.0, dim=[1,2,3])
                    losses = (zt - zt_initial).abs().mean(dim=[1,2,3])
                    print("prox_losses: ", losses)
                    mask = losses > etc_kwargs['prox_th']
                    if not mask.any():
                        break
                    optimizer.zero_grad()
                    prox_loss = losses[mask].sum()
                    prox_loss.backward()
                    optimizer.step()
                
            
            if etc_kwargs['iopt_dist']:
                zt = zt_initial * (torch.exp(0.5 * log_var)) + mu
                if etc_kwargs['use_kld_loss']:
                    kld_loss = self.fn_calc_kld_loss_func(log_var, mu) # returns (b_size, )
                    # print("kld_loss: ", kld_loss)
                    # print("kld_loss.shape: ", kld_loss.shape)
                    # import ipdb; ipdb.set_trace()
                    # while kld_loss > etc_kwargs['kld_th']: # 0.001
                    #     optimizer.zero_grad()
                    #     kld_loss = kld_loss.mean()
                    #     kld_loss.backward()
                    #     optimizer.step()
                    #     kld_loss = self.fn_calc_kld_loss_func(log_var, mu)
                    while True:
                        mask = kld_loss > etc_kwargs['kld_th']
                        # break if all elements are less than kld_th
                        if not mask.any():
                            break
                        optimizer.zero_grad()
                        kld_loss = kld_loss[mask].sum()
                        kld_loss.backward()
                        optimizer.step()
                        kld_loss = self.fn_calc_kld_loss_func(log_var, mu)                      
                    zt = zt_initial * (torch.exp(0.5 * log_var)) + mu
                if etc_kwargs['use_w_loss']:
                    w_loss = self.fn_calc_W_loss_func(log_var, mu)
                    print("w_loss: ", w_loss)
                    while True:
                        mask = w_loss > etc_kwargs['w_th']
                        # break if all elements are less than w_th
                        if not mask.any():
                            break
                        optimizer.zero_grad()
                        w_loss = w_loss[mask].sum()
                        w_loss.backward()
                        optimizer.step()
                        w_loss = self.fn_calc_W_loss_func(log_var, mu)
                    zt = zt_initial * (torch.exp(0.5 * log_var)) + mu
                    
            elif etc_kwargs['res_opt']:
                zt = zt_initial + res
            
            if etc_kwargs['iopt_lr_decay_rate'] != 1.0:
                scheduler.step()
                print("Current learning rate: ", scheduler.get_last_lr())
                
        # if 'attn_entropy' in etc_kwargs['iopt_loss_type']:
        #     self.unet = self.unet_base

        return zt.detach()
    
    @torch.enable_grad()
    def iopt_diverse_dds(self, zt, ts, step, uc, c, cfg_guidance, etc_kwargs):
        b_size = len(zt)
        zt = zt.detach()
        
        at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
        
        assert etc_kwargs['iopt_loss_type'] == 'infoNCE' # For now, only serve for InfoNCE
        
        if etc_kwargs['iopt_loss_type'] == 'infoNCE':
            with torch.no_grad():
                _, noise_pred_base = self.predict_noise(zt, ts, None, c)
                z0t_base = (zt - (1 - at).sqrt() * noise_pred_base) / at.sqrt()  # shape: (B, C, H, W)
                z0t_base = z0t_base.clone().detach()
                
        for i in range(etc_kwargs['i_opt_iter']):
            with torch.no_grad():
                _, noise_pred = self.predict_noise(zt, ts, None, c)
            
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
            z0t = z0t.detach().requires_grad_()
            
            if etc_kwargs['iopt_loss_type'] == 'infoNCE': # Byungjun's implementation of InfoNCE
                print("InfoNCE loss.")
                temperature = etc_kwargs['infoNCE_temp']  # typical value (can tune)
                gamma = etc_kwargs['gamma']
                w = etc_kwargs['window_size'] # pooling output shape (output shape becomes [B, C, w, w] for w > 1, [B, C] for w = 1)
                
                if w == 1:
                    z0t_base_flat = z0t_base.mean(dim=[2, 3])
                    z0t_flat = z0t.mean(dim=[2, 3])
        
                    # Normalize for cosine similarity
                    z0t_base_flat = torch.nn.functional.normalize(z0t_base_flat, dim=1)
                    z0t_flat = torch.nn.functional.normalize(z0t_flat, dim=1)
        
                    # Similarity matrix (cosine similarity)
                    diag_sim_matrix = torch.matmul(z0t_base_flat, z0t_flat.T) / temperature # (B, B) 
                    non_diag_sim_matrix = torch.matmul(z0t_flat, z0t_flat.T) / temperature # (B, B)
        
                    sim_matrix = non_diag_sim_matrix.clone()
                    diag_indices = torch.arange(b_size)
        
                    # 대각선 성분을 diag_sim_matrix 값으로 덮어쓰기
                    sim_matrix[diag_indices, diag_indices] = diag_sim_matrix[diag_indices, diag_indices] / gamma  # alpha > 1이면 positive pair 영향 약해짐
        
                    labels = torch.arange(b_size, device=z0t.device)
                    loss = torch.nn.functional.cross_entropy(sim_matrix, labels)
        
                else:
                    z0t_base_pooled = torch.nn.functional.adaptive_avg_pool2d(z0t_base, (w, w))  # (B, 4, w, w)
                    z0t_pooled = torch.nn.functional.adaptive_avg_pool2d(z0t, (w, w))  # (B, 4, w, w)
                    loss = 0.0

                    B, C, w, _ = z0t_base_pooled.shape
                    P = w * w  # number of patches

                    # Step 1: Flatten spatial dimensions
                    z0t_base_vecs = z0t_base_pooled.view(B, C, P)     # (B, C, P)
                    z0t_vecs = z0t_pooled.view(B, C, P)               # (B, C, P)

                    # Step 2: Normalize across channels
                    z0t_base_vecs = F.normalize(z0t_base_vecs, dim=1)  # (B, C, P)
                    z0t_vecs = F.normalize(z0t_vecs, dim=1)            # (B, C, P)

                    # Step 3: Compute cosine similarity across patches (for InfoNCE)
                    # diag_sim_matrix[b, p] = cosine(z0t_base_vecs[b,:,p], z0t_vecs[b,:,p])
                    diag_sim_matrix = torch.einsum('bcp,bcp->bp', z0t_base_vecs, z0t_vecs) / temperature  # (B, P)

                    # Full similarity: each patch position p has a (B x B) similarity matrix
                    # sim_matrix[p] = cosine(z0t_vecs[:, :, p], z0t_vecs[:, :, p])
                    non_diag_sim_matrix = torch.einsum('bcp, dcp -> pbd', z0t_vecs, z0t_vecs) / temperature  # (P, B, B)
                    sim_matrix = non_diag_sim_matrix.clone()

                    # Step 4: Adjust diagonals
                    diag_indices = torch.arange(B, device=sim_matrix.device)
                    patch_indices = torch.arange(P, device=sim_matrix.device)
                    
                    sim_matrix[patch_indices[:, None], diag_indices, diag_indices] = (diag_sim_matrix / gamma).T  # (P, B)

                    # Step 5: Compute InfoNCE loss
                    labels = torch.arange(B, device=sim_matrix.device)
                    loss_ij = F.cross_entropy(sim_matrix, labels.unsqueeze(0).repeat(P, 1))  # (P,)
                    loss = loss_ij.mean()

                    print("Loss:", loss)
                    
                # Update z0t with the minimizing loss direction
                loss.backward()
                with torch.no_grad():
                    z0t_new = z0t - etc_kwargs['i_opt_lr'] * z0t.grad
                
                    # Obtain new zt based on z0t_new
                    zt = at.sqrt() * z0t_new + (1 - at).sqrt() * noise_pred
                # z0t.grad = None
        
        return zt.detach()
        
        
        
    
    def fn_calc_kld_loss_func(self, log_var, mu):
        # return torch.mean(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp()), dim=0) # (1, C, H, W) version
        return -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=[1,2,3]) # (B, C, H, W) version
    
    def fn_calc_W_loss_func(self, log_var, mu):
        # Compute the covariance matrix diagonal
        variance = torch.exp(log_var)  # Variance is exp(log_var)
        
        # Wasserstein distance components
        # Mean term: ||mu||^2
        mean_term = torch.sum(mu**2, dim=(1, 2, 3))
        
        # Covariance term: Tr(Sigma + I - 2*sqrt(Sigma))
        identity = torch.ones_like(variance)  # Identity matrix in diagonal form
        sqrt_variance = torch.sqrt(variance)  # Element-wise square root
        covariance_term = torch.sum(variance + identity - 2 * sqrt_variance, dim=(1, 2, 3))
        
        return mean_term + covariance_term

    @torch.enable_grad()
    def get_z0_ms(self, zt, c, uc, cfg_guidance, num_steps, etc_kwargs):

        # cfg_guidance = 0.6 # cfgpp
        b_size = len(zt)
        skipped_scheduler = DDIMScheduler.from_pretrained(self.model_key, subfolder="scheduler", local_files_only=True)
        total_timesteps = len(skipped_scheduler.timesteps)

        skipped_scheduler.set_timesteps(num_steps, device=self.device)
        skip = total_timesteps // num_steps

        skipped_scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), skipped_scheduler.alphas_cumprod]).to(self.device)

        for step, t in enumerate(skipped_scheduler.timesteps):
            ts = torch.full((b_size,), t, device=self.device, dtype=torch.long)

            at = skipped_scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
            # if step == 0:
            #     print("t: ", t)
            #     print("at[0]: ", at[0])
            at_prev = skipped_scheduler.alphas_cumprod[ts - skip].view(b_size, 1, 1, 1)

            if etc_kwargs['iopt_msd']:
                with torch.no_grad():
                    noise_uc, noise_c = self.predict_noise(zt, ts, uc, c)
                    noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            else:
                noise_uc, noise_c = self.predict_noise(zt, ts, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred
            # zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc # cfgpp
            
            # if step == 0:
            #     img = self.decode(z0t)
            #     img = (img / 2 + 0.5).clamp(0, 1)
            #     img = img.detach().cpu()
            #     save_image(img, f"grid_z0T_total={num_steps}.png", normalize=True, nrow=int(b_size/2))
                
            #     z0t_c = (zt - (1-at).sqrt() * noise_c) / at.sqrt()
            #     img = self.decode(z0t_c)
            #     img = (img / 2 + 0.5).clamp(0, 1)
            #     img = img.detach().cpu()
            #     save_image(img, f"grid_z0T_c_total={num_steps}.png", normalize=True, nrow=int(b_size/2))\
                
            #     noise_pred_cfgpp = noise_uc + 0.6 * (noise_c - noise_uc)
            #     z0t_cfgpp = (zt - (1-at).sqrt() * noise_pred_cfgpp) / at.sqrt()
            #     img = self.decode(z0t_cfgpp)
            #     img = (img / 2 + 0.5).clamp(0, 1)
            #     img = img.detach().cpu()
            #     save_image(img, f"grid_z0T_cfgpp_total={num_steps}.png", normalize=True, nrow=int(b_size/2))

        # # test: visualize grided images by saving them
        # img = self.decode(z0t)
        # img = (img / 2 + 0.5).clamp(0, 1)
        # img = img.detach().cpu()
        # # save as grided images using torchvision.utils.make_grid
        # save_image(img, f"grid_z0_{num_steps}-step.png", normalize=True, nrow=int(b_size/2))

        return z0t

    @torch.enable_grad()
    def iopt_diverse_single(self, zt, ts, c, etc_kwargs):
        b_size = len(zt)
        assert b_size == 1            

        zt = zt.detach()
        zt.requires_grad = True
        at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)

        optimizer = Adam([zt], lr=etc_kwargs['i_opt_lr'])

        n_aug_samples = etc_kwargs['n_aug_samples']

        zt_aug = torch.randn(n_aug_samples, zt.shape[1], zt.shape[2], zt.shape[3], device=zt.device, generator=self.generator)
        ts_aug = ts.detach().clone().repeat(n_aug_samples)
        c_aug = c.detach().clone().repeat(n_aug_samples, 1, 1)
        at_aug = self.scheduler.alphas_cumprod[ts_aug].view(n_aug_samples, 1, 1, 1)

        assert len(zt.shape) == 4

        for i in range(etc_kwargs['i_opt_iter']):
            _, noise_pred = self.predict_noise(zt, ts, None, c)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            with torch.no_grad():
                _, noise_pred_aug = self.predict_noise(zt_aug, ts_aug, None, c_aug)
                z0t_aug = (zt_aug - (1-at_aug).sqrt() * noise_pred_aug) / at_aug.sqrt()


            if etc_kwargs['elatentlpips'] is not None:
                losses = etc_kwargs['elatentlpips'](z0t.repeat(len(z0t_aug), 1, 1, 1), z0t_aug, normalize=False).squeeze()
            else:
                losses = (z0t - z0t_aug.detach()).reshape(n_aug_samples, -1).norm(p=2.0, dim=-1)

            loss = -1 * losses.sum()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        return zt.detach()
    
    
    def cads(self, c, t, tau1=0.6, tau2=0.9, noise_scale=0.25, psi=1.0, rescale=True):        
        def linear_schedule(t, tau1, tau2):
            t_ratio = t / len(self.scheduler.alphas_cumprod_default)
            # print('t_ratio: ', t_ratio)
            if t_ratio <= tau1:
                # print('t_ratio <= tau1')
                return 1.0
            if t_ratio >= tau2:
                # print('t_ratio >= tau2')
                return 0.0
            # print('tau1 < t_ratio < tau2')
            gamma = (tau2 - t_ratio) / (tau2 - tau1)
            return gamma
        
        def add_noise(y_in, gamma, noise_scale, psi, rescale):
            use_avg_emb = False
            if use_avg_emb:
                print("Improved CADS: Using average embedding as noise.")
                init_gau_scale = 1.0

                text_enc = self.text_encoder.to(torch.float32)

                token_embeds_base = text_enc.get_input_embeddings().weight.data.detach().clone()

                embeds_mean = token_embeds_base.mean(dim=0)
                var_vector = (token_embeds_base ** 2 - embeds_mean.unsqueeze(0) ** 2).mean(dim=0)
                embeds_cov = torch.diag(var_vector) * (init_gau_scale ** 2)
                mvn = torch.distributions.MultivariateNormal(embeds_mean, covariance_matrix=embeds_cov)
                avg_emb = mvn.sample()


            # dim of y_in: (b, 77, 768)
            y_in_mean, y_in_std = y_in.mean(dim=[1,2], keepdim=True), y_in.std(dim=[1,2], keepdim=True)
            sqrt_gamma = gamma ** 0.5

            if not use_avg_emb:
                # use a dedicated local generator for augmentation to preserve global randomness
                y_in_shape = y_in.shape
                y = sqrt_gamma * y_in + noise_scale * (1 - sqrt_gamma) * torch.randn(y_in_shape, device=y_in.device, generator=self.generator)
            else:
                y = sqrt_gamma * y_in + noise_scale * (1 - sqrt_gamma) * avg_emb
            if rescale:
                y_scaled = (y - y.mean(dim=[1,2], keepdim=True)) / (y.std(dim=[1,2], keepdim=True)) * y_in_std + y_in_mean
                y = psi * y_scaled + (1 - psi) * y
            return y
        gamma = linear_schedule(t, tau1, tau2)
        return add_noise(c, gamma=gamma, noise_scale=noise_scale, psi=psi, rescale=rescale)



###########################################
# Base version
###########################################

@register_solver("ddim")
class BaseDDIM(StableDiffusion):
    """
    Basic DDIM solver for SD.
    Useful for text-to-image generation
    """

    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               popt_kwargs=None,
               etc_kwargs=None,
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """

        self.prompt = prompt
        
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        
        c_base = c.detach().clone()
        uc_base = uc.detach().clone()

        # Initialize zT
        zt = self.initialize_latent()
        if etc_kwargs['trunc_tau'] != 1.0:
            # zt = zt * math.sqrt(etc_kwargs['trunc_tau'])
            print(f"scaling zT by trunc_tau: {etc_kwargs['trunc_tau']}")
            zt = zt * etc_kwargs['trunc_tau']
        zt = zt.requires_grad_()

        if popt_kwargs['prompt_opt']:
            self.text_encoder = self.text_encoder.to(torch.float32)
            placeholder_token_ids_enc = self.initialize_embedding(self.tokenizer, self.text_encoder, popt_kwargs)
            self.vae.requires_grad_(False)

        if popt_kwargs['text_emb_opt']:
            c_opt_in = c.detach().clone()
        if popt_kwargs['null_text_emb_opt']:
            uc_opt_in = uc.detach().clone()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            # for prompt-opt
            if popt_kwargs['prompt_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                c = self.prompt_opt(
                    zt.detach(),
                    t,
                    step,
                    placeholder_token_ids_enc,
                    uc,
                    c_base,
                    cfg_guidance,
                    popt_kwargs
                )
            else:
                if popt_kwargs['prompt_opt'] and popt_kwargs['base_prompt_after_popt']:
                    c = c_base.detach().clone()

            if popt_kwargs['text_emb_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                c = self.text_emb_opt(
                    zt.detach(),
                    t,
                    step,
                    uc,
                    c_opt_in,
                    c_base, 
                    popt_kwargs
                )
                c_opt_in = c.detach().clone()
            else:
                if popt_kwargs['text_emb_opt'] and popt_kwargs['base_prompt_after_popt']:
                    c = c_base.detach().clone()

            if popt_kwargs['null_text_emb_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                uc = self.null_text_emb_opt(
                    zt.detach(),
                    t,
                    step,
                    uc_opt_in,
                    c,
                    uc_base,
                    cfg_guidance,
                    popt_kwargs
                )
                uc_opt_in = uc.detach().clone()
            else:
                if popt_kwargs['null_text_emb_opt'] and popt_kwargs['base_prompt_after_popt']:
                    uc = uc_base.detach().clone()

            if etc_kwargs['latent_opt'] and t > etc_kwargs['l_t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % etc_kwargs['l_inter_rate'] == 0:
                zt = self.latent_opt(
                    zt.detach(),
                    t,
                    step,
                    uc,
                    c,
                    etc_kwargs
                )

            if etc_kwargs['use_cads']:
                c = self.cads(c_base, t.item())
                uc = self.cads(uc_base, t.item())

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # for random noise
            if etc_kwargs['ddim_eta'] > 0.0:
                sigma_t = etc_kwargs['ddim_eta'] * torch.sqrt((1 - at_prev) / (1 - at) * (1 - at / at_prev))
                noise_rand = torch.randn_like(zt) * sigma_t
                zt = at_prev.sqrt() * z0t + (1-at_prev-sigma_t**2).sqrt() * noise_pred + noise_rand

            # for deterministic case: eta = 0.0
            else:
                zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def batch_sample(self,
               cfg_guidance=7.5,
               prompts=[""],
               null_prompts=[""],
               popt_kwargs=None,
               etc_kwargs=None,
               callback_fn=None,
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """
        assert len(prompts) == len(null_prompts)
        assert isinstance(prompts, list) and isinstance(null_prompts, list)
        self.prompts = prompts
        self.null_prompts = null_prompts

        # reset tokenizer and text_encoder
        self.tokenizer = copy.deepcopy(self.tokenizer_base)
        self.text_encoder = copy.deepcopy(self.text_encoder_base)

        b_size = len(prompts)

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)
        c_base = c.detach().clone()
        uc_base = uc.detach().clone()
        
        if etc_kwargs['Unet_features']:
            # # dictionary for saving features
            # self.features = {}
            # # Hook registration
            # self._register_hooks()
            
            cross_attn_block = self.unet.up_blocks[1]  # CrossAttnUpBlock2D
            attention_module = cross_attn_block.attentions[-1]  # 마지막 Attention 모듈
            hook = attention_module.register_forward_hook(self.save_feature) # Hook 등록
            
        if 'attn_entropy' in etc_kwargs['iopt_loss_type']:
            # # Attention store
            self.attention_store = AttentionStore(etc_kwargs['attn_res'], num_heads=self.unet.config.attention_head_dim)
            self.register_attention_control()
        
        # Initialize zT
        if etc_kwargs['sync_initial_noise']:
            from utils_local.log_util import set_seed
            set_seed(etc_kwargs['seed'])
        zt = self.initialize_latent(b_size=b_size)
        if etc_kwargs['trunc_tau'] != 1.0:
            # zt = zt * math.sqrt(etc_kwargs['trunc_tau'])
            print(f"scaling zT by trunc_tau: {etc_kwargs['trunc_tau']}")
            zt = zt * etc_kwargs['trunc_tau']
        zt = zt.requires_grad_()
        
        if popt_kwargs['prompt_opt'] or popt_kwargs['popt_diverse']:
            self.text_encoder = self.text_encoder.to(torch.float32)
            placeholder_token_ids_enc = self.initialize_embedding(self.tokenizer, self.text_encoder, popt_kwargs, b_size=b_size)
            self.vae.requires_grad_(False)

        if popt_kwargs['text_emb_opt']:
            c_opt_in = c.detach().clone()
        if popt_kwargs['null_text_emb_opt']:
            uc_opt_in = uc.detach().clone()
            
        B = zt.shape[0]
        all_errs = []   # (T, B)
        opt_w_trace = []   # 매 스텝마다 길이 B의 optimal w를 저장 (CPU로)


        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            ts = torch.full((b_size,), t, device=self.device, dtype=torch.long)

            at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
            at_prev = self.scheduler.alphas_cumprod[ts - self.skip].view(b_size, 1, 1, 1)

            if etc_kwargs['iopt_diverse'] and step == 0:
                # if len(zt) == 1:
                #     print("iopt_diverse_single")
                #     zt = self.iopt_diverse_single(
                #         zt.detach(),
                #         ts,
                #         c_base,
                #         etc_kwargs
                #     )

                # else:
                #     zt = self.iopt_diverse(
                #         zt.detach(),
                #         ts,
                #         step,
                #         uc_base,
                #         c_base,
                #         cfg_guidance, 
                #         etc_kwargs
                #     )
                zt = self.iopt_diverse(
                    zt.detach(),
                    ts,
                    step,
                    uc_base,
                    c_base,
                    cfg_guidance, 
                    etc_kwargs
                )
                
            if etc_kwargs['iopt_diverse_dds'] and step == 0:
                zt = self.iopt_diverse_dds(
                    zt.detach(),
                    ts,
                    step,
                    uc_base,
                    c_base,
                    cfg_guidance, 
                    etc_kwargs
                )

            if popt_kwargs['prompt_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                c = self.batch_prompt_opt(
                    zt.detach(),
                    ts,
                    step,
                    placeholder_token_ids_enc,
                    uc,
                    c_base,
                    cfg_guidance, 
                    popt_kwargs
                )
            else:
                if popt_kwargs['prompt_opt'] and popt_kwargs['base_prompt_after_popt']:
                    c = c_base.detach().clone()

            if popt_kwargs['popt_diverse'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                if popt_kwargs['diverse_type'] == "repel":
                    c = self.popt_diverse(
                        zt.detach(),
                        ts,
                        step,
                        placeholder_token_ids_enc,
                        uc,
                        c_base,
                        cfg_guidance, 
                        popt_kwargs
                    )
                elif popt_kwargs['diverse_type'] == "infoNCE":
                    c = self.popt_diverse_infoNCE(
                        zt.detach(),
                        ts,
                        step,
                        placeholder_token_ids_enc,
                        uc,
                        c_base,
                        cfg_guidance, 
                        popt_kwargs
                    )
                else:
                    raise NotImplementedError(f"Unknown diverse type: {popt_kwargs['diverse_type']}")
            else:
                if popt_kwargs['popt_diverse'] and popt_kwargs['base_prompt_after_popt']:
                    c = c_base.detach().clone()

            if popt_kwargs['text_emb_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                c = self.batch_text_emb_opt(
                    zt.detach(),
                    ts,
                    step,
                    uc,
                    c_opt_in,
                    c_base,
                    popt_kwargs
                )
                c_opt_in = c.detach().clone()
            else:
                if popt_kwargs['text_emb_opt'] and popt_kwargs['base_prompt_after_popt']:
                    c = c_base.detach().clone()

            if popt_kwargs['null_text_emb_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                uc = self.batch_null_text_emb_opt(
                    zt.detach(),
                    ts,
                    step,
                    uc_opt_in,
                    c,
                    uc_base,
                    cfg_guidance,
                    popt_kwargs
                )
                uc_opt_in = uc.detach().clone()
            else:
                if popt_kwargs['null_text_emb_opt'] and popt_kwargs['base_prompt_after_popt']:
                    uc = uc_base.detach().clone()

            if etc_kwargs['latent_opt'] and t > etc_kwargs['l_t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % etc_kwargs['l_inter_rate'] == 0:
                zt = self.batch_latent_opt(
                    zt.detach(),
                    ts,
                    step,
                    uc,
                    c,
                    etc_kwargs
                )

            if etc_kwargs['use_cads']:
                c = self.cads(c_base, t.item())
                uc = self.cads(uc_base, t.item())
                
            
            noise_matching_w = False
            
                
            if etc_kwargs['use_ig']:
                print("use_ig")
                t_ratio = float(t) / len(self.scheduler.alphas_cumprod_default)
                if etc_kwargs['ig_end'] <= t_ratio <= etc_kwargs['ig_start']:
                    # use cfg
                    print("using cfg")
                    with torch.no_grad():
                        noise_uc, noise_c = self.predict_noise(zt, ts, uc, c)
                        noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
                else:
                    print("not using cfg")
                    with torch.no_grad():
                        _, noise_pred = self.predict_noise(zt, ts, None, c)
                        
            elif noise_matching_w:
                with torch.no_grad():
                    noise_uc, noise_c = self.predict_noise(zt, ts, uc, c)
                    noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            else:
                with torch.no_grad():
                    noise_uc, noise_c = self.predict_noise(zt, ts, uc, c)
                    noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
                    
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
                    
            with torch.no_grad():
                if noise_matching_w:
                    eps_true = torch.randn(zt.shape, device=zt.device, generator=self.generator)
                    xt_probe = at.sqrt() * z0t + (1 - at).sqrt() * eps_true
                    eps_uc_p, eps_c_p = self.predict_noise(xt_probe, ts, uc, c)
                    eps_pred_probe = eps_uc_p + cfg_guidance * (eps_c_p - eps_uc_p)
                    
                    per_sample_err = (eps_true - eps_pred_probe).float().flatten(1).norm(p=2, dim=1)
                    ratio = (((1 - at) / at).clamp(min=1e-12)).sqrt().sqrt().view(B)
                    per_sample_err = per_sample_err * ratio
                    
                    
                    # ---- config ----
                    w_lo = 1.0
                    w_hi = float(cfg_guidance) if float(cfg_guidance) >= 1.0 else 1.0
                    w_mid = (w_lo + w_hi) * 0.5
                    cfg_inside_unet = True   # <- UNet이 guidance scale을 직접 쓰는 구현이면 True

                    with torch.inference_mode():
                        B = zt.shape[0]

                        # 공통 준비: probe 에러를 재는 함수
                        def eps_error_for_w(w_scalar):
                            """
                            동일 xt_probe, ts, uc, c 에서 guidance=w 로 예측된 eps와 eps_true의
                            per-sample L2 error (ratio 포함)를 반환: shape [B]
                            """
                            if cfg_inside_unet:
                                # UNet이 w를 직접 받는 구현: 모델을 w마다 호출
                                # predict_noise_w는 w를 모델로 전달하도록 네 구현에 맞춰 래핑해줘.
                                eps_pred = self.predict_noise(
                                    xt_probe, ts, uc, c, guidance_scale=float(w_scalar)
                                )
                                # ↑ 네 구현에서 (uc,c) 따로 주는 대신 guidance_scale로 바로 CFG eps를 내면 이 라인만 쓰면 됨.
                                #   만약 여전히 (eps_uc, eps_c)를 반환한다면, 모델 안에서 w를 쓰더라도 여기서는 CFG를 조합하지 말고
                                #   네 함수 시그니처에 맞게 eps_pred를 얻어줘.
                            else:
                            # # 표준 CFG: UNet 출력은 w와 무관 -> 미리 구한 uc/cond로 선형결합
                                eps_pred = eps_uc_p + float(w_scalar) * (eps_c_p - eps_uc_p)

                            err = (eps_true - eps_pred).reshape(B, -1).norm(p=2, dim=1)
                            # ratio는 argmin에는 영향 없지만, 네 로깅 일관성을 위해 곱해줌
                            return err * ratio

                        # 1) 세 점에서 에러 평가
                        f1 = eps_error_for_w(w_lo)   # [B]
                        f2 = eps_error_for_w(w_mid)  # [B]
                        f3 = eps_error_for_w(w_hi)   # [B]

                        # 2) 3점 포물선 근사로 각 샘플별 w* 추정
                        #    공통 w라서 3x3 역행렬은 상수. (수치 안정 위해 직접 공식 사용)
                        #    세 점 (x1,f1),(x2,f2),(x3,f3) 의 꼭짓점:
                        #    w* = x2 + 0.5 * [ (x1 - x2)^2 (f2 - f3) - (x2 - x3)^2 (f1 - f2) ] /
                        #                       [ (x1 - x2) (f2 - f3) - (x2 - x3) (f1 - f2) ]
                        x1, x2, x3 = w_lo, w_mid, w_hi
                        d12 = (x1 - x2); d23 = (x2 - x3)
                        num = (d12**2) * (f2 - f3) - (d23**2) * (f1 - f2)      # [B]
                        den = (d12) * (f2 - f3) - (d23) * (f1 - f2)            # [B]
                        den = torch.where(den.abs() < 1e-12, torch.sign(den) * 1e-12, den)

                        w_star = x2 + 0.5 * (num / den)                        # [B]
                        # 3) 경계로 클램프 + 비정상(오목, NaN) 케이스는 세 점 중 최솟값으로 폴백
                        w_star = w_star.clamp(min=w_lo, max=w_hi)

                        # 상향 오목(a<=0) 또는 NaN 인 샘플은 폴백
                        # a의 부호를 직접 안 구해도, "세 점 중 최소 f"로 안전 폴백
                        f_stack = torch.stack([f1, f2, f3], dim=0)             # [3, B]
                        w_cands = torch.tensor([x1, x2, x3], device=f1.device, dtype=f1.dtype)  # [3]
                        idx_min = f_stack.argmin(dim=0)                        # [B]
                        w_fb = w_cands[idx_min]                                # [B]

                        bad = torch.isnan(w_star) | torch.isinf(w_star)
                        # 원하면 여기에 '오목성 판단' 로직(예: f2 > (f1+f3)/2 등)도 추가 가능
                        opt_w = torch.where(bad, w_fb, w_star)                 # [B]
                        
                        # ... opt_w 계산까지 끝난 직후에:
                        opt_w_trace.append(opt_w.detach().to('cpu', torch.float32))  # [B]


                        # (선택) 에러 값도 기록하고 싶으면:
                        # f_opt = eps_error_for_w(opt_w)  # w 벡터를 받게 구현했으면 한 번에 처리

                        # 4) 최적 w로 noise_pred/z0t 갱신
                        if cfg_inside_unet:
                            # 모델이 w에 따라 바로 CFG 예측을 내는 경우
                            
                            noise_pred = noise_uc + opt_w * (noise_c - noise_uc)
                            
                            # noise_pred = self.predict_noise(zt, ts, uc, c, guidance_scale=opt_w)
                        # else:
                        #     noise_pred = noise_uc + opt_w.view(B, 1, 1, 1) * (noise_c - noise_uc)

                        z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()

                    
                    
                    
                    
                    
                    
                    # # ----- find optimal w (closed-form, per-sample) -----
                    # w_min = 0.0
                    # w_max = 15.0
                    # # w_max = float(cfg_guidance) if float(cfg_guidance) >= 1.0 else 1.0

                    # # Δ = eps_c - eps_uc, r = eps_true - eps_uc
                    # delta = (eps_c_p - eps_uc_p).float().flatten(1)   # [B, D]
                    # resid = (eps_true - eps_uc_p).float().flatten(1)  # [B, D]

                    # num = (resid * delta).sum(dim=1)                  # [B]
                    # den = (delta * delta).sum(dim=1).clamp_min(1e-12) # [B]
                    # w_star = num / den                                # unconstrained optimum per sample
                    # opt_w = w_star.clamp(min=w_min, max=w_max)        # project to [1, cfg_guidance]

                    # (선택) 최적 w에서의 probe 오차를 보고 싶으면:
                    # eps_pred_opt_probe = eps_uc_p + opt_w.view(-1,1,1,1) * (eps_c_p - eps_uc_p)
                    # per_sample_err_opt = (eps_true - eps_pred_opt_probe).float().flatten(1).norm(p=2, dim=1)
                    # per_sample_err_opt = per_sample_err_opt * ratio  # ratio는 값만 스케일

                    # (선택) 나중에 곡선 플롯/저장을 위해 기록
                    # opt_w_trace.append(opt_w.detach().cpu())
                    # err_opt_trace.append(per_sample_err_opt.detach().cpu())
                    # -----------------------------------------------
                    
                    # noise_pred = noise_uc + opt_w.view(-1,1,1,1) * (noise_c - noise_uc)
                    # z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()

                
            # if noise_matching_w:
            #     eps_true = torch.randn(zt.shape, device=zt.device, generator=self.generator)
            #     xt_probe = at.sqrt() * z0t + (1 - at).sqrt() * eps_true
            #     eps_uc_p, eps_c_p = self.predict_noise(xt_probe, ts, uc, c)
            #     eps_pred_probe = eps_uc_p + cfg_guidance * (eps_c_p - eps_uc_p)
                
            #     per_sample_err = (eps_true - eps_pred_probe).float().flatten(1).norm(p=2, dim=1)
            #     ratio = (((1 - at) / at).clamp(min=1e-12)).sqrt().sqrt().view(B)
            #     per_sample_err = per_sample_err * ratio
                
            #     # find optimal w
                
            #     # re-calculate z0t based on the opt cfg weight
            #     noise_pred = noise_uc + opt_cfg_guidance * (noise_c - noise_uc)
            #     z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
            
            
            # 루프 내, 네 코드 그대로 두고 "find optimal w" 부분을 아래로 교체
            # 사전 준비(루프 시작 전에 한 번만 정의해도 됨): grid 해상도
            # num_w = 21  # or etc_kwargs.get('nmw_num_w', 21)

            # if noise_matching_w:
            #     B = zt.shape[0]

            #     # 1) 현재 step용 probe 구성 (네 코드 그대로)
            #     z0t_nmw = (zt - (1 - at).sqrt() * noise_c) / at.sqrt()
            #     eps_true = torch.randn(zt.shape, device=zt.device, generator=self.generator)
            #     xt_probe = at.sqrt() * z0t_nmw + (1 - at).sqrt() * eps_true

            #     # 2) probe에서 eps 예측 (uncond/cond 각각 한 번만 추론)
            #     eps_uc_p, eps_c_p = self.predict_noise(xt_probe, ts, uc, c)

            #     # 3) ratio ( (1-ā)/ā )^{1/4}  [B,]
            #     ratio = (((1 - at) / at).clamp(min=1e-12)).sqrt().sqrt().view(B).float()

            #     # 4) w-grid 준비: [1.0, cfg_guidance]
            #     w_hi = float(cfg_guidance) if float(cfg_guidance) >= 1.0 else 1.0
            #     w_grid = torch.linspace(1.0, w_hi, steps=num_w, device=zt.device, dtype=eps_uc_p.dtype)  # [G]

            #     # 5) 선형결합으로 CFG 예측들을 한 번에 계산: eps_pred(w) = eps_uc + w*(eps_c - eps_uc)
            #     delta = (eps_c_p - eps_uc_p)                                       # [B,C,H,W]
            #     eps_pred_grid = eps_uc_p.unsqueeze(0) + w_grid.view(-1, 1, 1, 1, 1) * delta.unsqueeze(0)  # [G,B,C,H,W]

            #     # 6) per-sample L2 에러 (각 w마다 B개): err_grid[g,b]
            #     diff = (eps_pred_grid - eps_true.unsqueeze(0)).float()             # [G,B,C,H,W]
            #     err_grid = diff.flatten(2).norm(p=2, dim=2)                        # [G,B]
            #     err_grid = err_grid * ratio.view(1, B)                              # ratio 적용

            #     # 7) 각 샘플별 최적 w 선택
            #     idx_min = err_grid.argmin(dim=0)                                    # [B]  (각 b의 최적 g 인덱스)
            #     opt_w   = w_grid[idx_min]                                           # [B]

            #     # (선택) 기록해두고 싶으면:
            #     opt_w_trace.append(opt_w.detach().cpu())        # 리스트에 쌓아서 나중에 CSV/플롯 가능
            #     # err_min_trace.append(err_grid[idx_min, torch.arange(B)])  # 최솟값도 저장하고 싶으면

            #     # 8) 최적 w로 noise_pred 및 z0t 갱신 (샘플별 스칼라 w를 브로드캐스트)
            #     noise_pred_opt = noise_uc + opt_w.view(B, 1, 1, 1) * (noise_c - noise_uc)
            #     z0t = (zt - (1 - at).sqrt() * noise_pred_opt) / at.sqrt()
    
            
            noise_matching_err = False
            if noise_matching_err:
                # noise matching loss
                eps_true = torch.randn(zt.shape, device=zt.device, generator=self.generator)
                xt_probe = at.sqrt() * z0t + (1 - at).sqrt() * eps_true
                eps_uc_p, eps_c_p = self.predict_noise(xt_probe, ts, uc, c)
                eps_pred_probe = eps_uc_p + cfg_guidance * (eps_c_p - eps_uc_p)

                per_sample_err = (eps_true - eps_pred_probe).float().flatten(1).norm(p=2, dim=1)
                ratio = (((1 - at) / at).clamp(min=1e-12)).sqrt().sqrt().view(B)
                per_sample_err = per_sample_err * ratio

                all_errs.append(per_sample_err.detach().cpu())

            # for random noise
            if etc_kwargs['ddim_eta'] > 0.0:
                sigma_t = etc_kwargs['ddim_eta'] * torch.sqrt((1 - at_prev) / (1 - at) * (1 - at / at_prev))
                noise_rand = torch.randn_like(zt) * sigma_t
                zt = at_prev.sqrt() * z0t + (1-at_prev-sigma_t**2).sqrt() * noise_pred + noise_rand

            # for deterministic case: eta = 0.0
            else:
                zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

            # ---- callback (draw_noisy / draw_tweedie) ----
            if callback_fn is not None:
                callback_kwargs = {
                    'z0t': z0t.detach(),
                    'zt': zt.detach(),
                    'decode': self.decode,
                }
                callback_kwargs = callback_fn(step, t, callback_kwargs)

        if noise_matching_w:
            import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
            save_dir = "./err_logs"
            os.makedirs(save_dir, exist_ok=True)

            # (T, B)로 스택
            opt_w_tb = torch.stack(opt_w_trace, dim=0).numpy()  # shape: (T, B)
            timesteps = self.scheduler.timesteps.detach().cpu().numpy()

            # ----- CSV 저장 -----
            # df_w = pd.DataFrame(opt_w_tb, columns=[f"sample_{i}" for i in range(opt_w_tb.shape[1])])
            # df_w.insert(0, "timestep", timesteps)
            # csv_path = os.path.join(save_dir, "opt_cfg_w_per_timestep.csv")
            # df_w.to_csv(csv_path, index=False)
            # print(f"Saved CSV to {csv_path}")

            # ----- per-sample 곡선 플롯 (라인 너무 많으면 일부만) -----
            B = opt_w_tb.shape[1]
            max_lines = 12
            idx_to_plot = np.arange(B) if B <= max_lines else np.linspace(0, B-1, max_lines, dtype=int)

            plt.figure(figsize=(7,4))
            for b in idx_to_plot:
                plt.plot(timesteps, opt_w_tb[:, b], label=f"sample {b}")

            ax = plt.gca()
            ax.invert_xaxis()  # 보통 큰 t → 작은 t로 진행
            ax.set_xlabel("timestep (t)")
            ax.set_ylabel("optimal CFG weight (w)")
            ax.set_title("Per-sample optimal w across timesteps")

            # ---- 여기서 y축 오프셋/지수 표기 제거 + 범위 고정 ----
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            w_lo = 1.0
            w_hi = float(cfg_guidance) if float(cfg_guidance) >= 1.0 else 1.0
            if w_hi > w_lo:
                ax.set_ylim(w_lo, w_hi)
            else:
                # cfg_guidance == 1.0 같은 퇴화 케이스에 살짝 마진
                ax.set_ylim(w_lo - 1e-3, w_lo + 1e-3)

            # 라인 수가 많을 땐 범례 생략(또는 표시된 subset만 보이게 하려면 len(idx_to_plot) 기준)
            if len(idx_to_plot) <= max_lines:
                ax.legend(ncol=2, fontsize=8)

            plt.tight_layout()
            png_path = os.path.join(save_dir, "opt_cfg_w_per_timestep.png")
            plt.savefig(png_path, dpi=200)
            plt.close()
            print(f"Saved plot to {png_path}")

            
            
            
            
            
            
            # B = opt_w_tb.shape[1]
            # max_lines = 12
            # idx_to_plot = np.arange(B) if B <= max_lines else np.linspace(0, B-1, max_lines, dtype=int)

            # plt.figure(figsize=(7,4))
            # for b in idx_to_plot:
            #     plt.plot(timesteps, opt_w_tb[:, b], label=f"sample {b}")
            # plt.gca().invert_xaxis()  # 보통 큰 t → 작은 t로 진행
            # plt.xlabel("timestep (t)")
            # plt.ylabel("optimal CFG weight (w)")
            # plt.title("Per-sample optimal w across timesteps")
            # if B <= max_lines:
            #     plt.legend(ncol=2, fontsize=8)
            # plt.tight_layout()
            # png_path = os.path.join(save_dir, "opt_cfg_w_per_timestep.png")
            # plt.savefig(png_path, dpi=200); plt.close()
            # print(f"Saved plot to {png_path}")
            
            
            
        
        # if noise_matching_w:
        #     #######################################
        #     # -------- plot and save as PNG --------
        #     import numpy as np
        #     import matplotlib.pyplot as plt
        #     import os
        #     save_dir = "./err_logs"
        #     os.makedirs(save_dir, exist_ok=True)
            
        #     # (T, B)로 스택
        #     opt_w_tb = torch.stack(opt_w_trace, dim=0).numpy()        # shape: (T, B)
        #     timesteps = self.scheduler.timesteps.detach().cpu().numpy()
        #     # ----- per-sample 곡선 플롯 (라인 수가 많으면 일부만 표시) -----
        #     B = opt_w_tb.shape[1]
        #     max_lines = 12
        #     idx_to_plot = np.arange(B) if B <= max_lines else np.linspace(0, B-1, max_lines, dtype=int)

        #     plt.figure(figsize=(7,4))
        #     for b in idx_to_plot:
        #         plt.plot(timesteps, opt_w_tb[:, b], label=f"sample {b}")
        #     plt.gca().invert_xaxis()  # 보통 timestep이 큰→작으로 진행
        #     plt.xlabel("timestep (t)")
        #     plt.ylabel("optimal CFG weight (w)")
        #     plt.title("Per-sample optimal w across timesteps")
        #     if B <= max_lines:
        #         plt.legend(ncol=2, fontsize=8)
        #     plt.tight_layout()

        #     png_path = os.path.join(save_dir, "opt_cfg_w_per_timestep.png")
        #     plt.savefig(png_path, dpi=200); plt.close()
        #     print(f"Saved plot to {png_path}")
        
        if noise_matching_err:
        
            #######################################
            # -------- plot and save as PNG --------
            import matplotlib.pyplot as plt
            import os
            
            # stack shape: (T, B)
            all_errs = torch.stack(all_errs, dim=0).numpy()
            timesteps = self.scheduler.timesteps.detach().cpu().numpy()
            
            save_dir = "./err_logs"
            os.makedirs(save_dir, exist_ok=True)
            
            plt.figure(figsize=(7,4))
            for b in range(B):
                plt.plot(timesteps, all_errs[:, b], label=f"sample {b}")
            plt.gca().invert_xaxis()
            plt.xlabel("timestep (t)")
            plt.ylabel("‖ε_true - ε_pred‖₂ × ratio")
            plt.title("Per-sample noise prediction error curves")
            plt.legend()
            plt.tight_layout()

            png_path = os.path.join(save_dir, "eps_err_per_sample.png")
            plt.savefig(png_path, dpi=200)
            plt.close()
            print(f"Saved plot to {png_path}")
            #######################################
        
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


@register_solver("dpm++_2s_a")
class DPMpp2sAncestralCFGSolver(StableDiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, **kwargs):
        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        # convert to karras sigma scheduler
        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=sigmas).to(torch.float16)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            new_t = self.timestep(sigma).to(self.device)
            
            with torch.no_grad():
                denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)

            sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i + 1])
            if sigma_down == 0:
                # Euler method
                d = self.to_d(x, sigmas[i], denoised)
                x = denoised + d * sigma_down
            else:
                # DPM-Solver++(2S)
                t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
                
                with torch.no_grad():
                    sigma_s = sigma_fn(s)
                    t_2 = self.timestep(sigma_s).to(self.device)
                    denoised_2, _ = self.kdiffusion_x_to_denoised(x_2, sigma_s, uc, c, cfg_guidance, t_2)
                
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
            # Noise addition
            if sigmas[i + 1] > 0:
                x = x + torch.randn_like(x) * sigma_up

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, new_t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    
@register_solver("dpm++_2m")
class DPMpp2mCFGSolver(StableDiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance, prompt=["", ""], callback_fn=None, popt_kwargs=None, **kwargs):
        self.prompt = prompt

        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        c_base = c.detach().clone()

        # convert to karras sigma scheduler
        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(1, 4, 64, 64),
                                   sigmas=sigmas).to(torch.float16)
        old_denoised = None # buffer

        if popt_kwargs['prompt_opt']:
            self.text_encoder = self.text_encoder.to(torch.float32)
            placeholder_token_ids_enc = self.initialize_embedding(self.tokenizer, self.text_encoder, popt_kwargs)
            self.vae.requires_grad_(False)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            new_t = self.timestep(sigma).to(self.device)

            # for prompt-opt
            if popt_kwargs['prompt_opt'] and i % popt_kwargs['inter_rate'] == 0:
                c = self.prompt_opt_dpmpp_2m(
                    x.detach(),
                    sigmas,
                    i,
                    placeholder_token_ids_enc,
                    uc,
                    c_base,
                    cfg_guidance,
                    popt_kwargs
                )
            else:
                if popt_kwargs['prompt_opt'] and popt_kwargs['base_prompt_after_popt']:
                    c = c_base.detach().clone()
            
            with torch.no_grad():
                denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)

            # solve ODE one step
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i+1])
            h = t_next - t
            if old_denoised is None or sigmas[i+1] == 0:
                x = denoised + self.to_d(x, sigmas[i], denoised) * sigmas[i+1]
            else:
                h_last = t - t_fn(sigmas[i-1])
                r = h_last / h
                extra1 = -torch.exp(-h) * denoised - (-h).expm1() * (denoised - old_denoised) / (2*r)
                extra2 = torch.exp(-h) * x
                x = denoised + extra1 + extra2
            old_denoised = denoised

            if callback_fn is not None:
                callback_kwargs = { 'z0t': denoised.detach(),
                                    'zt': x.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(i, new_t, callback_kwargs)
                denoised = callback_kwargs["z0t"]
                x = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def batch_sample(self, cfg_guidance, prompts=[""], null_prompts=[""], popt_kwargs=None, **kwargs):
        assert len(prompts) == len(null_prompts)
        assert isinstance(prompts, list) and isinstance(null_prompts, list)
        self.prompts = prompts
        self.null_prompts = null_prompts

        # reset tokenizer and text_encoder
        self.tokenizer = copy.deepcopy(self.tokenizer_base)
        self.text_encoder = copy.deepcopy(self.text_encoder_base)

        b_size = len(prompts)

        t_fn = lambda sigma: sigma.log().neg()
        sigma_fn = lambda t: t.neg().exp()
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)
        c_base = c.detach().clone()
        uc_base = uc.detach().clone()

        # convert to karras sigma scheduler
        total_sigmas = (1-self.total_alphas).sqrt() / self.total_alphas.sqrt()
        sigmas = get_sigmas_karras(len(self.scheduler.timesteps), total_sigmas.min(), total_sigmas.max(), rho=7.)
        # initialize
        x = self.initialize_latent(method="random_kdiffusion",
                                   latent_dim=(b_size, 4, 64, 64),
                                   b_size=b_size,
                                   sigmas=sigmas).to(torch.float16)
        old_denoised = None # buffer

        if popt_kwargs['prompt_opt']:
            self.text_encoder = self.text_encoder.to(torch.float32)
            placeholder_token_ids_enc = self.initialize_embedding(self.tokenizer, self.text_encoder, popt_kwargs, b_size=b_size)
            self.vae.requires_grad_(False)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for i, _ in enumerate(pbar):
            sigma = sigmas[i]
            new_t = self.timestep(sigma).to(self.device)

            # expand for batch
            sigma = sigma.expand(b_size, 1, 1, 1).to(self.device)
            new_t = new_t.expand(b_size)

            # for prompt-opt
            if popt_kwargs['prompt_opt'] and i % popt_kwargs['inter_rate'] == 0:
                c = self.batch_prompt_opt_dpmpp_2m(
                    x.detach(),
                    sigmas,
                    i,
                    placeholder_token_ids_enc,
                    uc,
                    c_base,
                    cfg_guidance,
                    popt_kwargs
                )
            else:
                if popt_kwargs['prompt_opt'] and popt_kwargs['base_prompt_after_popt']:
                    c = c_base.detach().clone()
            
            with torch.no_grad():
                denoised, _ = self.kdiffusion_x_to_denoised(x, sigma, uc, c, cfg_guidance, new_t)

            # solve ODE one step
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i+1])
            h = t_next - t
            if old_denoised is None or sigmas[i+1] == 0:
                x = denoised + self.to_d(x, sigmas[i], denoised) * sigmas[i+1]
            else:
                h_last = t - t_fn(sigmas[i-1])
                r = h_last / h
                extra1 = -torch.exp(-h) * denoised - (-h).expm1() * (denoised - old_denoised) / (2*r)
                extra2 = torch.exp(-h) * x
                x = denoised + extra1 + extra2
            old_denoised = denoised
        
        # for the last step, do not add noise
        img = self.decode(x)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


@register_solver("ddim_inversion")
class InversionDDIM(BaseDDIM):
    """
    Editing via WardSwap after inversion.
    Useful for text-guided image editing.
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               src_img,
               cfg_guidance=7.5,
               prompt=["","",""],
               callback_fn=None,
               **kwargs):
        
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=src_img,
                                    uc=uc,
                                    c=c,
                                    cfg_guidance=cfg_guidance)
        zt = zt.requires_grad_()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

@register_solver("ddim_edit")
class EditWardSwapDDIM(InversionDDIM):
    """
    Editing via WardSwap after inversion.
    Useful for text-guided image editing.
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               src_img,
               cfg_guidance=7.5,
               prompt=["","",""],
               callback_fn=None,
               **kwargs):
        
        # Text embedding
        uc, src_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        _, tgt_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[2])

        # Initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=src_img,
                                    uc=uc,
                                    c=src_c,
                                    cfg_guidance=cfg_guidance)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DDIM-edit")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, tgt_c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred
        
            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

###########################################
# CFG++ version
###########################################

@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    """
    DDIM solver for SD with CFG++.
    Useful for text-to-image generation
    """
    def __init__(self,
                 solver_config: Dict,
                #  model_key:str="runwayml/stable-diffusion-v1-5",
                 model_key:str="botp/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        super().__init__(solver_config, model_key, device, **kwargs)

    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               popt_kwargs=None,
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """
        
        self.prompt = prompt

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        c_base = c.detach().clone()

        # Initialize zT
        zt = self.initialize_latent()
        zt = zt.requires_grad_() # why zt is required grad?
        
        if popt_kwargs['prompt_opt']:
            self.text_encoder = self.text_encoder.to(torch.float32)
            placeholder_token_ids_enc = self.initialize_embedding(self.tokenizer, self.text_encoder, popt_kwargs)
            self.vae.requires_grad_(False)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            # for prompt-opt
            if popt_kwargs['prompt_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                c = self.prompt_opt(
                    zt.detach(),
                    t,
                    step,
                    placeholder_token_ids_enc,
                    uc,
                    c_base,
                    cfg_guidance, 
                    popt_kwargs
                )
            else:
                if popt_kwargs['base_prompt_after_popt']:
                    c = c_base.detach().clone()

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def batch_sample(self,
               cfg_guidance=7.5,
               prompts=[""],
               null_prompts=[""],
               popt_kwargs=None,
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """
        assert len(prompts) == len(null_prompts)
        assert isinstance(prompts, list) and isinstance(null_prompts, list)
        self.prompts = prompts
        self.null_prompts = null_prompts

        # reset tokenizer and text_encoder
        self.tokenizer = copy.deepcopy(self.tokenizer_base)
        self.text_encoder = copy.deepcopy(self.text_encoder_base)

        b_size = len(prompts)

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)
        c_base = c.detach().clone()

        # Initialize zT
        zt = self.initialize_latent(b_size=b_size)
        zt = zt.requires_grad_()
        
        if popt_kwargs['prompt_opt']:
            self.text_encoder = self.text_encoder.to(torch.float32)
            placeholder_token_ids_enc = self.initialize_embedding(self.tokenizer, self.text_encoder, popt_kwargs, b_size=b_size)
            self.vae.requires_grad_(False)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            ts = torch.full((b_size,), t, device=self.device, dtype=torch.long)

            at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
            at_prev = self.scheduler.alphas_cumprod[ts - self.skip].view(b_size, 1, 1, 1)

            # for prompt-opt
            if popt_kwargs['prompt_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                c = self.batch_prompt_opt(
                    zt.detach(),
                    ts,
                    step,
                    placeholder_token_ids_enc,
                    uc,
                    c_base,
                    cfg_guidance, 
                    popt_kwargs
                )
            else:
                if popt_kwargs['prompt_opt'] and popt_kwargs['base_prompt_after_popt']:
                    c = c_base.detach().clone()

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, ts, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

@register_solver("ddim_inversion_cfg++")
class InversionDDIMCFGpp(BaseDDIMCFGpp):
    """
    Editing via WardSwap after inversion.
    Useful for text-guided image editing.
    """
    @torch.no_grad()
    def inversion(self,
                  z0: torch.Tensor,
                  uc: torch.Tensor,
                  c: torch.Tensor,
                  cfg_guidance: float=1.0):

        # initialize z_0
        zt = z0.clone().to(self.device)
         
        # loop
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM Inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t-self.skip)

            noise_uc, noise_c = self.predict_noise(zt, t, uc, c) 
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at_prev).sqrt() * noise_uc) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

        return zt

    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               src_img,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               **kwargs):
        
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=src_img,
                                    uc=uc,
                                    c=c,
                                    cfg_guidance=cfg_guidance)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

@register_solver("ddim_edit_cfg++")
class EditWardSwapDDIMCFGpp(InversionDDIMCFGpp):
    """
    Editing via WardSwap after inversion.
    Useful for text-guided image editing.
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               src_img,
               cfg_guidance=7.5,
               prompt=["","",""],
               callback_fn=None,
               **kwargs):
        
        # Text embedding
        uc, src_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        _, tgt_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[2])

        # Initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=src_img,
                                    uc=uc,
                                    c=src_c,
                                    cfg_guidance=cfg_guidance)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DDIM-edit")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, tgt_c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc
        
            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


#############################
@register_solver("ddim_jepa")
class DDIMWithJEPA(StableDiffusion):
    """
    DDIM solver with JEPA guidance for SD1.5
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_jepa = None
        self.jepa_rng = None
        self.jepa_config = {}
    
    def setup_jepa(self, jepa_config):
        """Initialize JEPA model and config"""
        # Save global RNG state (torch.hub.load changes it)
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        try:
            self.jepa_config = jepa_config
            self.jg_img_size = self.jepa_config.get('jg_img_size', 224)
            self.seed = self.jepa_config.get('seed', 42)
            jepa_backbone = self.jepa_config.get('jepa_backbone', 'dinov2')

            if jepa_config.get('use_jepa', False):
                if 'dinov2' in jepa_backbone.lower():
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
                    _pth = os.path.join(_DINOV2_CKPTS, _DINOV2_PTH[jepa_backbone.lower()])
                    print(f"[JEPA] Loading {jepa_backbone}_reg  ← {_pth}")
                    backbone = torch.hub.load(
                        _DINOV2_HUB, f'{jepa_backbone.lower()}_reg',
                        source='local', weights=_pth,
                    ).to(self.device).eval()
                    self.f_jepa = DINOv2JEPAWrapper(backbone, self.jg_img_size).to(self.device).eval()
                    print(f"[JEPA] Initialized {jepa_backbone} for JEPA guidance")
                elif 'dinov3' in jepa_backbone.lower():
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
                    from transformers import AutoModel
                    _variant = _DINOV3_VARIANTS[jepa_backbone.lower()]
                    _local = os.path.join(_DINOV3_DIR, _variant)
                    print(f"[JEPA] Loading DINOv3 {jepa_backbone}  ← {_local}")
                    backbone = AutoModel.from_pretrained(_local, local_files_only=True).to(self.device).eval()
                    self.f_jepa = DINOv3JEPAWrapper(backbone, self.jg_img_size).to(self.device).eval()
                    print(f"[JEPA] Initialized {jepa_backbone} for JEPA guidance")
                elif jepa_backbone.lower() == 'mae_vitb16':
                    # [Ablation] MAE: reconstruction-based, non-uniform feature space
                    # Expected: JS guidance metrics degrade vs. uniform SSL (DINOv2)
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
                    from transformers import ViTMAEModel
                    _local = os.path.join(_SSL_DIR, _SSL_VARIANTS['mae_vitb16'])
                    print(f"[JEPA][Ablation] Loading MAE ViT-B/16 (non-uniform SSL)  ← {_local}")
                    backbone = ViTMAEModel.from_pretrained(_local, local_files_only=True).to(self.device).eval()
                    self.f_jepa = MAEJEPAWrapper(backbone, self.jg_img_size).to(self.device).eval()
                    print("[JEPA][Ablation] Initialized MAE ViT-B/16")
                elif jepa_backbone.lower() == 'data2vec_vitb16':
                    # [Ablation] Data2Vec-Vision: prediction-based, non-uniform feature space
                    # Expected: JS guidance metrics degrade vs. uniform SSL (DINOv2)
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
                    from transformers import Data2VecVisionModel
                    _local = os.path.join(_SSL_DIR, _SSL_VARIANTS['data2vec_vitb16'])
                    print(f"[JEPA][Ablation] Loading Data2Vec-Vision ViT-B/16 (non-uniform SSL)  ← {_local}")
                    backbone = Data2VecVisionModel.from_pretrained(_local, local_files_only=True).to(self.device).eval()
                    self.f_jepa = Data2VecJEPAWrapper(backbone, self.jg_img_size).to(self.device).eval()
                    print("[JEPA][Ablation] Initialized Data2Vec-Vision ViT-B/16")
                elif 'metaclip' in jepa_backbone.lower():
                    # Disable memory-efficient attention backends for MetaCLIP gradient computation
                    print("[JEPA] Disabling flash/mem_efficient attention backends for MetaCLIP gradient computation...")
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)

                    import open_clip
                    backbone, _, preprocess = open_clip.create_model_and_transforms(
                        'ViT-B-16-quickgelu', pretrained='metaclip_400m'
                    )
                    backbone = backbone.visual.to(self.device).eval()
                    self.f_jepa = MetaCLIPJEPAWrapper(backbone, self.jg_img_size).to(self.device).eval()
                    print("[JEPA] Initialized MetaCLIP for JEPA guidance")
                else:
                    raise ValueError(f"Unknown jepa_backbone: {jepa_backbone}. Choose 'dinov2_*', 'dinov3_*', 'metaclip', or 'ijepa'")

                self.jepa_rng = torch.Generator(device=self.device)
                self.jepa_rng.manual_seed(self.seed)
        finally:
            # Restore global RNG state
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

    def compute_jepa_gradient(self, zt, step, total_steps, t=None, at=None, at_prev=None, uc=None, c=None, cfg_guidance=7.5):
        """Compute JEPA score gradient w.r.t. noisy latent zt"""
        cfg = self.jepa_config
        eta = cfg.get('jepa_eta', 1.0)
        g_interval = cfg.get('g_interval', 3)
        g_start_t = cfg.get('g_start_t', 0.8)
        k = cfg.get('rsvd_topk', 3)
        q_steps = cfg.get('rsvd_pi_q', 2)
        p = cfg.get('rsvd_oversample', 2)
        use_normed_grad = cfg.get('use_normed_grad', True)
        jg_img_size = cfg.get('jg_img_size', 224)
        jg_schedule = cfg.get('jg_schedule', 'variance')
        jepa_backbone = cfg.get('jepa_backbone', 'dinov2')
        use_full_svd = cfg.get('use_full_svd', False)
        eps = 1e-8

        # Normalization stats based on backbone type
        _imagenet_backbone = ('dinov2', 'ijepa', 'dinov3', 'mae_vitb16', 'data2vec_vitb16')
        if any(k in jepa_backbone.lower() for k in _imagenet_backbone):
            # ImageNet normalization for DINOv2, DINOv3, I-JEPA, MAE, Data2Vec-Vision
            norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(zt.device)
            norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(zt.device)
        elif 'metaclip' in jepa_backbone.lower():
            # CLIP normalization stats for MetaCLIP
            norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(zt.device)
            norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(zt.device)
        else:
            raise ValueError(f"Unknown jepa_backbone: {jepa_backbone}. Choose 'dinov2_*', 'dinov3_*', 'mae_vitb16', 'data2vec_vitb16', 'metaclip', or 'ijepa'")

        # Check timing (t_ratio: 1.0 at start -> 0.0 at end)
        t_ratio = 1.0 - (step / total_steps)
        if step % g_interval != 0 or t_ratio > g_start_t:
            return None

        r = k + p

        with torch.enable_grad():
            zt_in = zt.detach().clone().requires_grad_(True)

            # Re-compute noise_pred with zt_in to build computational graph
            noise_uc, noise_c = self.predict_noise(zt_in, t, uc, c)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            # _, noise_pred = self.predict_noise(zt_in, t, None, c) # nocfg version

            # Predict clean latent from noisy latent
            z0t = (zt_in - (1-at).sqrt() * noise_pred) / at.sqrt()

            # Decode latent -> image [0,1]
            # Get VAE dtype and convert accordingly
            vae_dtype = next(self.vae.parameters()).dtype
            z_scaled = (z0t / 0.18215).to(vae_dtype)
            img = self.vae.decode(z_scaled).sample
            x0p = (img / 2 + 0.5).clamp(0, 1).float()

            # Resize and normalize for JEPA backbone
            x0p = F.interpolate(x0p, size=(jg_img_size, jg_img_size), mode="bilinear", align_corners=False)
            x0p = (x0p - norm_mean) / norm_std

            B, C, Hp, Wp = x0p.shape

            # # Manual Jv/JTu to avoid xFormers/SDPA jvp incompatibility
            # def Jv(v, create_graph=False):
            #     # Forward-mode AD via finite difference: (f(x + eps*v) - f(x)) / eps
            #     eps_fd = 1e-4
            #     with torch.no_grad():
            #         f_plus = self.f_jepa(x0p + eps_fd * v)
            #         f_base = self.f_jepa(x0p)
            #     return (f_plus - f_base) / eps_fd

            # def JTu(u, create_graph=False):
            #     # Reverse-mode AD via standard backward
            #     # Use x0p directly to maintain graph connection to zt_in
            #     out = self.f_jepa(x0p)
            #     grad = torch.autograd.grad(out, x0p, grad_outputs=u,
            #                               create_graph=create_graph, retain_graph=True)[0]
            #     return grad
            
            # for efficiency, compute f_base only once
            f_base = self.f_jepa(x0p)
            
            # Manual Jv/JTu to avoid xFormers/SDPA jvp incompatibility
            def Jv(v, create_graph=False):
                # Forward-mode AD via finite difference: (f(x + eps*v) - f(x)) / eps
                eps_fd = 1e-4
                with torch.no_grad():
                    f_plus = self.f_jepa(x0p + eps_fd * v)
                return (f_plus - f_base.detach()) / eps_fd

            def JTu(u, create_graph=False):
                # Reverse-mode AD via standard backward
                # Use x0p directly to maintain graph connection to zt_in
                grad = torch.autograd.grad(f_base, x0p, grad_outputs=u,
                                          create_graph=create_graph, retain_graph=True)[0]
                return grad



            # # EVD-based
            # # 1) Random Omega
            # Omega = torch.randn(B, r, C, Hp, Wp, device=x0p.device, dtype=x0p.dtype,
            #                 generator=self.jepa_rng)
            # Omega = Omega / (Omega.view(B, r, -1).norm(dim=2, keepdim=True).view(B, r, 1, 1, 1) + eps)

            # # 2) Y = J @ Omega
            # Y_cols = [Jv(Omega[:, j], create_graph=False) for j in range(r)]
            # Y = torch.stack(Y_cols, dim=2)

            # # 2.5) Subspace iteration
            # for _ in range(q_steps):
            #     Y_cols = []
            #     for j in range(r):
            #         wj = JTu(Y[:, :, j].detach(), create_graph=False)
            #         Y_cols.append(Jv(wj, create_graph=False))
            #     Y = torch.stack(Y_cols, dim=2)
            #     Y, _ = torch.linalg.qr(Y, mode="reduced")

            # Q, _ = torch.linalg.qr(Y, mode="reduced")
            # Q = Q.detach()

            # # This avoids Jv (which has no grad graph with finite diff)
            # JTQ_cols = []
            # for j in range(r):
            #     qj = Q[:, :, j]
            #     wj = JTu(qj, create_graph=True)  # J^T @ q_j, shape (B, C, H, W)
            #     JTQ_cols.append(wj)
            # # JTQ: list of r tensors, each (B, C, H, W)
            # # Stack into (B, r, C, H, W) then flatten spatial dims
            # JTQ = torch.stack(JTQ_cols, dim=1)  # (B, r, C, H, W)
            # JTQ_flat = JTQ.view(B, r, -1)  # (B, r, C*H*W)
            
            # # EVD-based
            # # M_ij = <JTQ_i, JTQ_j> via batch matmul
            # M = torch.bmm(JTQ_flat, JTQ_flat.transpose(1, 2))  # (B, r, r)

            # # 4) Eigenvalues -> JEPA loss (requires float32)
            # M_float = M.float()

            # evals = torch.linalg.eigvalsh(M_float)
            # evals_top = torch.clamp(evals[:, -k:], min=eps)
            # sigmas_top = torch.sqrt(evals_top)
            
            
            

            # SVD-based
            # Retry loop: resample Omega if SVD fails
            max_svd_retries = 3
            for svd_attempt in range(max_svd_retries):
                # 1) Random Omega
                Omega = torch.randn(B, r, C, Hp, Wp, device=x0p.device, dtype=x0p.dtype,
                                generator=self.jepa_rng)
                Omega = Omega / (Omega.view(B, r, -1).norm(dim=2, keepdim=True).view(B, r, 1, 1, 1) + eps)

                # 2) Y = J @ Omega
                Y_cols = [Jv(Omega[:, j], create_graph=False) for j in range(r)]
                Y = torch.stack(Y_cols, dim=2)

                # 2.5) Subspace iteration
                for _ in range(q_steps):
                    Y_cols = []
                    for j in range(r):
                        wj = JTu(Y[:, :, j].detach(), create_graph=False)
                        Y_cols.append(Jv(wj, create_graph=False))
                    Y = torch.stack(Y_cols, dim=2)
                    # QR decomposition requires float32
                    Y_float = Y.float()
                    Y_float, _ = torch.linalg.qr(Y_float, mode="reduced")
                    Y = Y_float.to(Y.dtype)

                # QR decomposition requires float32
                Y_float = Y.float()
                Q_float, _ = torch.linalg.qr(Y_float, mode="reduced")
                Q = Q_float.to(Y.dtype).detach()

                # This avoids Jv (which has no grad graph with finite diff)
                JTQ_cols = []
                for j in range(r):
                    qj = Q[:, :, j]
                    wj = JTu(qj, create_graph=True)  # J^T @ q_j, shape (B, C, H, W)
                    JTQ_cols.append(wj)
                # JTQ: list of r tensors, each (B, C, H, W)
                # Stack into (B, r, C, H, W) then flatten spatial dims
                JTQ = torch.stack(JTQ_cols, dim=1)  # (B, r, C, H, W)
                JTQ_flat = JTQ.view(B, r, -1)  # (B, r, C*H*W)
                
                # SVD-based
                # SVD is more numerically stable than eigendecomposition of M = JTQ @ JTQ^T
                JTQ_float = JTQ_flat.float()
                try:
                    sigmas = torch.linalg.svdvals(JTQ_float)  # (B, r), descending order
                    break  # success
                except torch._C._LinAlgError:
                    try:
                        # CUDA SVD failed, fallback to CPU (more robust algorithm)
                        sigmas = torch.linalg.svdvals(JTQ_float.cpu()).to(JTQ_float.device)
                        break  # success
                    except torch._C._LinAlgError:
                        if svd_attempt < max_svd_retries - 1:
                            print(f"[JEPA] SVD failed, resampling Omega (attempt {svd_attempt + 1}/{max_svd_retries})")
                            continue
                        else:
                            print(f"[JEPA] SVD failed after {max_svd_retries} attempts, skipping this step")
                            return None
            
            if use_full_svd:
                # Full Jacobian SVD: sigma_max per batch element
                sigmas_full = self.jepa_score_full_svd(x0p, f_base)  # (B,)
                jepa_loss = torch.log(torch.clamp(sigmas_full, min=eps)).sum()
            else:
                sigmas_top = torch.clamp(sigmas[:, :k], min=eps)
                # Sum over k (singular values) but keep batch dimension
                jepa_loss_per_sample = torch.log(sigmas_top).sum(dim=1)  # (B,)
                jepa_loss = jepa_loss_per_sample.sum()  # scalar for backward

            grad = torch.autograd.grad(jepa_loss, zt_in)[0]

        if use_normed_grad:
            max_grad = grad.abs().amax(dim=(1, 2, 3), keepdim=True)
            grad = grad / (max_grad + eps)

        # Compute variance scheduling (similar to guided-diffusion)
        if jg_schedule == 'variance':
            # Variance of the reverse process: variance = (1 - at_prev) / (1 - at) * (1 - at/at_prev)
            variance = ((1 - at_prev) / (1 - at) * (1 - at / at_prev)).clamp(min=eps)
            print(f"JEPA scaling at step {step}:", variance[0].item())
            final_grad = eta * variance * grad
            return final_grad
        elif jg_schedule == 'constant':
            print("JEPA constant scaling at step", step)
            return eta * grad

    def jepa_score_full_svd(self, x0p: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Full Jacobian SVD-based JEPA score.

        Computes the exact Jacobian J = d(feats)/d(x0p) per batch element,
        then returns sigma_max = sqrt(lambda_max(J^T J))  shape: (B,).

        Args:
            x0p:   preprocessed image tensor (B, C, H', W') — must be in autograd graph
            feats: JEPA feature tensor (B, D) computed from x0p via self.f_jepa
        """
        B, D = feats.shape
        eps = 1e-12

        sigmas = []
        for b in range(B):
            Jb = []
            for i in range(D):
                grad_out = torch.zeros_like(feats)
                grad_out[b, i] = 1.0
                grad_i = torch.autograd.grad(
                    outputs=feats,
                    inputs=x0p,
                    grad_outputs=grad_out,
                    retain_graph=True,
                    create_graph=True,
                )[0][b]  # (C, H', W')
                Jb.append(grad_i.reshape(-1))  # (C*H'*W',)

            Jb = torch.stack(Jb, dim=0)   # (D, C*H'*W')
            JTJ = Jb @ Jb.t()             # (D, D)
            eigvals = torch.linalg.eigvalsh(JTJ.float())  # ascending, (D,)
            lambda_max = eigvals[-1]
            sigma = torch.sqrt(lambda_max.clamp(min=eps))
            sigmas.append(sigma)

        return torch.stack(sigmas)  # (B,)

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self, cfg_guidance=7.5, prompt=["",""], callback_fn=None,
               popt_kwargs=None, etc_kwargs=None, **kwargs):
        self.prompt = prompt
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        zt = self.initialize_latent().requires_grad_()

        total_steps = len(self.scheduler.timesteps)
        use_jepa = self.jepa_config.get('use_jepa', False) and self.f_jepa is not None
        pbar = tqdm(self.scheduler.timesteps, desc="SD+JEPA" if use_jepa else "SD")

        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            # JEPA guidance on zt
            if use_jepa:
                jepa_grad = self.compute_jepa_gradient(zt, step, total_steps, t=t, at=at, at_prev=at_prev, uc=uc, c=c, cfg_guidance=cfg_guidance)
                if jepa_grad is not None:
                    zt = zt - jepa_grad
                    pbar.set_postfix({'jepa': 'applied'})

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # DDIM step
            if etc_kwargs['ddim_eta'] > 0.0:
                sigma_t = etc_kwargs['ddim_eta'] * torch.sqrt((1 - at_prev) / (1 - at) * (1 - at / at_prev))
                zt = at_prev.sqrt() * z0t + (1-at_prev-sigma_t**2).sqrt() * noise_pred + torch.randn_like(zt) * sigma_t
            else:
                zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(), 'zt': zt.detach(), 'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t, zt = callback_kwargs["z0t"], callback_kwargs["zt"]

        img = self.decode(z0t)
        return (img / 2 + 0.5).clamp(0, 1).detach().cpu()

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def batch_sample(self, cfg_guidance=7.5, prompts=[""], null_prompts=[""],
                     popt_kwargs=None, etc_kwargs=None, **kwargs):
        assert len(prompts) == len(null_prompts)
        self.prompts, self.null_prompts = prompts, null_prompts
        b_size = len(prompts)

        uc, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)
        zt = self.initialize_latent(b_size=b_size)

        total_steps = len(self.scheduler.timesteps)
        use_jepa = self.jepa_config.get('use_jepa', False) and self.f_jepa is not None
        pbar = tqdm(self.scheduler.timesteps, desc="SD+JEPA" if use_jepa else "SD")

        for step, t in enumerate(pbar):
            ts = torch.full((b_size,), t, device=self.device, dtype=torch.long)
            at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
            at_prev = self.scheduler.alphas_cumprod[ts - self.skip].view(b_size, 1, 1, 1)

            # JEPA guidance on zt
            if use_jepa:
                jepa_grad = self.compute_jepa_gradient(zt, step, total_steps, t=ts, at=at, at_prev=at_prev, uc=uc, c=c, cfg_guidance=cfg_guidance)
                if jepa_grad is not None:
                    zt = zt - jepa_grad
                    pbar.set_postfix({'jepa': 'applied'})
                    
            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, ts, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # DDIM step
            if etc_kwargs['ddim_eta'] > 0.0:
                sigma_t = etc_kwargs['ddim_eta'] * torch.sqrt((1 - at_prev) / (1 - at) * (1 - at / at_prev))
                zt = at_prev.sqrt() * z0t + (1-at_prev-sigma_t**2).sqrt() * noise_pred + torch.randn_like(zt) * sigma_t
            else:
                zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

        img = self.decode(z0t)
        return (img / 2 + 0.5).clamp(0, 1).detach().cpu()



# ============================================================
# DINOv2 JEPA Wrapper
# ============================================================

class DINOv2JEPAWrapper(nn.Module):
    # 14의 배수인 유효한 resolution들
    VALID_SIZES = [224, 168, 112, 84, 70, 56, 42, 28]

    def __init__(self, backbone, size=224):
        super().__init__()
        self.backbone = backbone

        # 14의 배수 검증
        if size % 14 != 0:
            valid_str = ", ".join(map(str, self.VALID_SIZES))
            raise ValueError(f"size must be multiple of 14 (patch size). Got {size}. Valid options: {valid_str}")

        self.size = size
        num_patches = size // 14
        print(f"[DINOv2] Resolution: {size}x{size} -> {num_patches}x{num_patches} = {num_patches**2} tokens")

        # ImageNet stats
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # x: (B,3,H,W) in [0,1]
        # resize
        # x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        # ImageNet normalize
        # x = (x - self.mean) / self.std
        # DINOv2 forward
        return self.backbone(x)


class DINOv3JEPAWrapper(nn.Module):
    # 16의 배수인 유효한 resolution들 (DINOv3 uses patch size 16)
    VALID_SIZES = [224, 192, 160, 128, 96, 64, 32]

    def __init__(self, backbone, size=224):
        super().__init__()
        self.backbone = backbone

        # 16의 배수 검증
        if size % 16 != 0:
            valid_str = ", ".join(map(str, self.VALID_SIZES))
            raise ValueError(f"size must be multiple of 16 (patch size). Got {size}. Valid options: {valid_str}")

        self.size = size
        num_patches = size // 16
        print(f"[DINOv3] Resolution: {size}x{size} -> {num_patches}x{num_patches} = {num_patches**2} tokens")

        # ImageNet stats (same as DINOv2)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # x: (B,3,H,W) normalized
        # DINOv3 HuggingFace API: returns BaseModelOutputWithPooling
        output = self.backbone(pixel_values=x)
        # CLS token from last hidden state: (B, embed_dim)
        return output.last_hidden_state[:, 0]


class MetaCLIPJEPAWrapper(nn.Module):
    # 16의 배수인 유효한 resolution들 (CLIP uses patch size 16)
    VALID_SIZES = [224, 112, 96, 80, 64, 48, 32]

    def __init__(self, backbone, size=224):
        super().__init__()
        self.backbone = backbone

        # 16의 배수 검증
        if size % 16 != 0:
            valid_str = ", ".join(map(str, self.VALID_SIZES))
            raise ValueError(f"size must be multiple of 16 (patch size). Got {size}. Valid options: {valid_str}")

        self.size = size
        num_patches = size // 16
        print(f"[MetaCLIP] Resolution: {size}x{size} -> {num_patches}x{num_patches} = {num_patches**2} tokens")

        # ImageNet stats (same as CLIP preprocessing)
        self.register_buffer(
            "mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # x: (B,3,H,W) in [0,1]
        # resize
        # x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        # CLIP normalize
        # x = (x - self.mean) / self.std
        # MetaCLIP forward
        return self.backbone(x)


class IJEPAJEPAWrapper(nn.Module):
    # 14의 배수인 유효한 resolution들 (I-JEPA ViT-H/14 uses patch size 14)
    VALID_SIZES = [224, 168, 112, 84, 70, 56, 42, 28]

    def __init__(self, backbone, size=224):
        super().__init__()
        self.backbone = backbone

        # 14의 배수 검증
        if size % 14 != 0:
            valid_str = ", ".join(map(str, self.VALID_SIZES))
            raise ValueError(f"size must be multiple of 14 (patch size). Got {size}. Valid options: {valid_str}")

        self.size = size
        num_patches = size // 14
        print(f"[I-JEPA] Resolution: {size}x{size} -> {num_patches}x{num_patches} = {num_patches**2} tokens")

        # ImageNet stats
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # x: (B,3,H,W) in [0,1]
        # resize
        # x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        # ImageNet normalize
        # x = (x - self.mean) / self.std
        # I-JEPA forward
        return self.backbone(x)


# ============================================================
# MAE JEPA Wrapper  [Ablation: non-uniform SSL]
# MAE (Masked Autoencoder) learns via reconstruction — no uniformity loss.
# Feature space is NOT spread uniformly → JS guidance expected to degrade.
# HuggingFace: facebook/vit-mae-base  (ViT-B/16, patch=16, embed=768)
# ============================================================

class MAEJEPAWrapper(nn.Module):
    # 16의 배수인 유효한 resolution들 (ViT-B/16 patch size 16)
    VALID_SIZES = [224, 192, 160, 128, 96, 64, 32]

    def __init__(self, backbone, size=224):
        super().__init__()
        self.backbone = backbone

        if size % 16 != 0:
            valid_str = ", ".join(map(str, self.VALID_SIZES))
            raise ValueError(f"size must be multiple of 16. Got {size}. Valid: {valid_str}")

        # Disable random masking for deterministic feature extraction
        self.backbone.config.mask_ratio = 0.0

        self.size = size
        num_patches = size // 16
        print(f"[MAE] Resolution: {size}x{size} -> {num_patches}x{num_patches} = {num_patches**2} patches (mask_ratio=0)")

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # x: (B,3,H,W) normalized
        # ViTMAEModel with mask_ratio=0: last_hidden_state (B, 1+num_patches, 768)
        output = self.backbone(pixel_values=x)
        # CLS token at index 0
        return output.last_hidden_state[:, 0]  # (B, 768)


# ============================================================
# Data2Vec-Vision JEPA Wrapper  [Ablation: non-uniform SSL]
# Data2Vec predicts contextualized top-k layer representations.
# Training objective is NOT contrastive → feature space is non-uniform.
# HuggingFace: facebook/data2vec-vision-base  (ViT-B/16, patch=16, embed=768)
# ============================================================

class Data2VecJEPAWrapper(nn.Module):
    # 16의 배수인 유효한 resolution들 (ViT-B/16 patch size 16)
    VALID_SIZES = [224, 192, 160, 128, 96, 64, 32]

    def __init__(self, backbone, size=224):
        super().__init__()
        self.backbone = backbone

        if size % 16 != 0:
            valid_str = ", ".join(map(str, self.VALID_SIZES))
            raise ValueError(f"size must be multiple of 16. Got {size}. Valid: {valid_str}")

        self.size = size
        num_patches = size // 16
        print(f"[Data2Vec] Resolution: {size}x{size} -> {num_patches}x{num_patches} = {num_patches**2} patches")

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        # x: (B,3,H,W) normalized
        # Data2VecVisionModel uses BEiT-style architecture with CLS token
        output = self.backbone(pixel_values=x)
        # pooler_output is the CLS token representation (B, 768)
        return output.pooler_output


#############################

if __name__ == "__main__":
    # print all list of solvers
    print(f"Possble solvers: {[x for x in __SOLVER__.keys()]}")
    
