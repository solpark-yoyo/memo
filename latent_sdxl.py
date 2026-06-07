from typing import Any, Optional, Tuple
import os
from safetensors.torch import load_file

_CKPT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")
_DINOV2_HUB  = os.path.join(_CKPT_DIR, "jepa_model", "dinov2", "facebookresearch_dinov2_main")
_DINOV2_CKPTS = os.path.join(_CKPT_DIR, "jepa_model", "dinov2", "checkpoints")
_DINOV2_PTH  = {
    "dinov2_vits14": "dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14": "dinov2_vitb14_reg4_pretrain.pth",
    "dinov2_vitl14": "dinov2_vitl14_reg4_pretrain.pth",
}

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers.models.attention_processor import (AttnProcessor2_0,
                                                  LoRAAttnProcessor2_0,
                                                  LoRAXFormersAttnProcessor,
                                                  XFormersAttnProcessor)
from tqdm import tqdm

from torch.optim.adam import Adam
import copy
import math

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

########################

class SDXL():
    def __init__(self, 
                 solver_config: dict,
                 model_key:str=os.path.join(_CKPT_DIR, "stable-diffusion-xl-base-1.0"),
                 dtype=torch.float16,
                 device='cuda',
                 seed: int=42):

        self.device = device
        pipe = StableDiffusionXLPipeline.from_pretrained(model_key, torch_dtype=dtype, local_files_only=True).to(device)
        self.dtype = dtype

        # avoid overflow in float16
        self.vae = AutoencoderKL.from_pretrained(os.path.join(_CKPT_DIR, "sdxl-vae-fp16-fix"), torch_dtype=dtype, local_files_only=True).to(device)

        self.tokenizer_1_base = copy.deepcopy(pipe.tokenizer)
        self.tokenizer_2_base = copy.deepcopy(pipe.tokenizer_2)
        self.text_enc_1_base = copy.deepcopy(pipe.text_encoder)
        self.text_enc_2_base = copy.deepcopy(pipe.text_encoder_2)
        self.unet = pipe.unet

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size

        # sampling parameters
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", local_files_only=True)
        N_ts = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = N_ts // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod_default = self.scheduler.alphas_cumprod
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])
        
        # a dedicated generator for various purposes
        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(seed)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.sample(*args, **kwargs)

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def _text_embed(self, prompt, tokenizer, text_enc, clip_skip):
        text_inputs = tokenizer(
            prompt,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        # print("prompt: ", prompt)
        # print("text_input_ids: ", text_input_ids)
        # also print string for the associated text tokens
        # print("text tokens: ", tokenizer.convert_ids_to_tokens(text_input_ids[0]))
        prompt_embeds = text_enc(text_input_ids.to(self.device), output_hidden_states=True)

        pool_prompt_embeds = prompt_embeds[0]
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            # +2 because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
        return prompt_embeds, pool_prompt_embeds

    @torch.no_grad()
    def get_text_embed(self, null_prompt_1, prompt_1, null_prompt_2=None, prompt_2=None, clip_skip=None):
        '''
        At this time, assume that batch_size = 1.
        We should extend the code to batch_size > 1.
        '''        
        # Encode the prompts
        # if prompt_2 is None, set same as prompt_1
        prompt_1 = [prompt_1] if isinstance(prompt_1, str) else prompt_1
        null_prompt_1 = [null_prompt_1] if isinstance(null_prompt_1, str) else null_prompt_1


        prompt_embed_1, pool_prompt_embed = self._text_embed(prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        if prompt_2 is None:
            prompt_embed = [prompt_embed_1]
        else:
            # Comment on diffusers' source code:
            # "We are only ALWAYS interested in the pooled output of the final text encoder"
            # i.e. we overwrite the pool_prompt_embed with the new one
            prompt_embed_2, pool_prompt_embed = self._text_embed(prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            prompt_embed = [prompt_embed_1, prompt_embed_2]
        
        null_embed_1, pool_null_embed = self._text_embed(null_prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        if null_prompt_2 is None:
            null_embed = [null_embed_1]
        else:
            null_embed_2, pool_null_embed = self._text_embed(null_prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            null_embed = [null_embed_1, null_embed_2]

        # concat embeds from two encoders
        null_prompt_embeds = torch.concat(null_embed, dim=-1)
        prompt_embeds = torch.concat(prompt_embed, dim=-1)

        return null_prompt_embeds, prompt_embeds, pool_null_embed, pool_prompt_embed            
    
    def _differentiable_text_embed(self, prompt, tokenizer, text_enc, clip_skip):
        text_inputs = tokenizer(
            prompt,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_enc(text_input_ids.to(self.device), output_hidden_states=True)

        pool_prompt_embeds = prompt_embeds[0]
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            # +2 because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
        return prompt_embeds, pool_prompt_embeds

    def differentiable_get_text_embed(self, null_prompt_1, prompt_1, null_prompt_2=None, prompt_2=None, clip_skip=None, te1_nograd=False, te2_nograd=False):
        '''
        At this time, assume that batch_size = 1.
        We should extend the code to batch_size > 1.
        '''        
        # Encode the prompts
        # if prompt_2 is None, set same as prompt_1
        prompt_1 = [prompt_1] if isinstance(prompt_1, str) else prompt_1
        null_prompt_1 = [null_prompt_1] if isinstance(null_prompt_1, str) else null_prompt_1

        if te1_nograd:
            prompt_embed_1, pool_prompt_embed = self._text_embed(prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        else:
            prompt_embed_1, pool_prompt_embed = self._differentiable_text_embed(prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)

        if prompt_2 is None:
            prompt_embed = [prompt_embed_1]
        else:
            # Comment on diffusers' source code:
            # "We are only ALWAYS interested in the pooled output of the final text encoder"
            # i.e. we overwrite the pool_prompt_embed with the new one
            if te2_nograd:
                prompt_embed_2, pool_prompt_embed = self._text_embed(prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            else:
                prompt_embed_2, pool_prompt_embed = self._differentiable_text_embed(prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            prompt_embed = [prompt_embed_1, prompt_embed_2]
        
        if te1_nograd:
            null_embed_1, pool_null_embed = self._text_embed(null_prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        else:
            null_embed_1, pool_null_embed = self._differentiable_text_embed(null_prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)

        if null_prompt_2 is None:
            null_embed = [null_embed_1]
        else:
            if te2_nograd:
                null_embed_2, pool_null_embed = self._text_embed(null_prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            else:
                null_embed_2, pool_null_embed = self._differentiable_text_embed(null_prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            null_embed = [null_embed_1, null_embed_2]

        # concat embeds from two encoders
        null_prompt_embeds = torch.concat(null_embed, dim=-1)
        prompt_embeds = torch.concat(prompt_embed, dim=-1)

        return null_prompt_embeds, prompt_embeds, pool_null_embed, pool_prompt_embed            

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor 

    # @torch.no_grad() 
    def decode(self, zt):
        # make sure the VAE is in float32 mode, as it overflows in float16
        # needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        # if needs_upcasting:
        #     self.upcast_vae()
        #     zt = zt.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        image = self.vae.decode(zt / self.vae.config.scaling_factor).sample.float()
        return image


    def predict_noise(self, zt, t, uc, c, added_cond_kwargs):
        t_in = t.unsqueeze(0)
        if uc is None:
            noise_c = self.unet(zt, t_in, encoder_hidden_states=c,
                                   added_cond_kwargs=added_cond_kwargs)['sample']
            noise_uc = noise_c
        elif c is None:
            noise_uc = self.unet(zt, t_in, encoder_hidden_states=uc,
                                   added_cond_kwargs=added_cond_kwargs)['sample']
            noise_c = noise_uc
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2)
            t_in = torch.cat([t_in] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed,
                                   added_cond_kwargs=added_cond_kwargs)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)

        return noise_uc, noise_c

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim):
        add_time_ids = list(original_size+crops_coords_top_left+target_size)
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        assert expected_add_embed_dim == passed_add_embed_dim, (
             f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               prompt1 = ["", ""],
               prompt2 = ["", ""],
               cfg_guidance:float=5.0,
               original_size: Optional[Tuple[int, int]]=None,
               crops_coords_top_left: Tuple[int, int]=(0, 0),
               target_size: Optional[Tuple[int, int]]=None,
               negative_original_size: Optional[Tuple[int, int]]=None,
               negative_crops_coords_top_left: Tuple[int, int]=(0, 0),
               negative_target_size: Optional[Tuple[int, int]]=None,
               clip_skip: Optional[int]=None,
               popt_kwargs: Optional[dict]=None,
               etc_kwargs: Optional[dict]=None,
               **kwargs):
        
        self.prompt1 = prompt1
        self.prompt2 = prompt2
        self.cfg_guidance = cfg_guidance
        self.original_size = original_size
        self.crops_coords_top_left = crops_coords_top_left
        self.target_size = target_size
        self.negative_original_size = negative_original_size
        self.negative_crops_coords_top_left = negative_crops_coords_top_left
        self.negative_target_size = negative_target_size
        self.clip_skip = clip_skip

        # 0. Default height and width to unet
        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # reset tokenizer and text_encoder
        self.tokenizer_1 = copy.deepcopy(self.tokenizer_1_base)
        self.tokenizer_2 = copy.deepcopy(self.tokenizer_2_base)
        self.text_enc_1 = copy.deepcopy(self.text_enc_1_base)
        self.text_enc_2 = copy.deepcopy(self.text_enc_2_base)

        # embedding
        (null_prompt_embeds,
         prompt_embeds,
         pool_null_embed,
         pool_prompt_embed) = self.get_text_embed(prompt1[0], prompt1[1], prompt2[0], prompt2[1], clip_skip)

        # prepare kwargs for SDXL
        add_text_embeds = pool_prompt_embed
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
            )
        else:
            negative_add_time_ids = add_time_ids
        negative_text_embeds = pool_null_embed

        # batch > 1: add_time_ids는 항상 (1, 6)으로 생성되므로 배치 크기에 맞게 확장
        b_size = prompt_embeds.shape[0]
        if b_size > 1:
            add_time_ids = add_time_ids.expand(b_size, -1).contiguous()
            negative_add_time_ids = negative_add_time_ids.expand(b_size, -1).contiguous()

        if cfg_guidance != 0.0 and cfg_guidance != 1.0:
            # do cfg
            add_text_embeds = torch.cat([negative_text_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_cond_kwargs = {
            'text_embeds': add_text_embeds.to(self.device),
            'time_ids': add_time_ids.to(self.device)
        }

        # reverse sampling
        zt = self.reverse_process(null_prompt_embeds, prompt_embeds, cfg_guidance, add_cond_kwargs, target_size, popt_kwargs=popt_kwargs, etc_kwargs=etc_kwargs, **kwargs)

        # decode
        with torch.no_grad():
            img = self.decode(zt)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

    def initialize_latent(self,
                          method: str='random',
                          src_img: Optional[torch.Tensor]=None,
                          add_cond_kwargs: Optional[dict]=None,
                          **kwargs):
        if method == 'ddim':
            assert src_img is not None, "src_img must be provided for inversion"
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('uc'),
                               kwargs.get('c'),
                               kwargs.get('cfg_guidance', 0.0),
                               add_cond_kwargs)
        elif method == 'npi':
            assert src_img is not None, "src_img must be provided for inversion"
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('c'),
                               kwargs.get('c'),
                               1.0,
                               add_cond_kwargs)
        elif method == 'random':
            size = kwargs.get('size', (1, 4, 128, 128))
            z = torch.randn(size).to(self.device)
        else: 
            raise NotImplementedError

        return z.requires_grad_()

    def inversion(self, z0, uc, c, cfg_guidance, add_cond_kwargs):
        # if we use cfg_guidance=0.0 or 1.0 for inversion, add_cond_kwargs must be splitted. 
        if cfg_guidance == 0.0 or cfg_guidance == 1.0:
            add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
            add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)

        zt = z0.clone().to(self.device)
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c  = self.predict_noise(zt, t, uc, c, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at_prev).sqrt() * noise_pred) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

        return zt
    
    def reverse_process(self, *args, **kwargs):
        raise NotImplementedError
    
    @torch.enable_grad()
    def prompt_opt(self, zt, t, step, placeholder_token_ids_enc1, placeholder_token_ids_enc2, null_prompt_embeds, prompt_embeds_base, add_cond_kwargs_base, cfg_guidance, popt_kwargs):
        assert cfg_guidance > 0.0
        placeholder_string = popt_kwargs['placeholder_string']
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
        placeholder_symbol = placeholder_string.split("_")[0]

        decay_rate = popt_kwargs['lr_decay_rate']
        num_opt_tokens = popt_kwargs['num_opt_tokens']

        if (1. - step * decay_rate) <= 0:
            print("Learning rate is zero. Skipping prompt optimization and using the latest optimized embedding.")
            prompt1 = self.prompt1.copy()
            prompt2 = self.prompt2.copy()

            # add placeholder tokens only for prompt
            # prompt1[1] = prompt1[1] + " " + placeholder_string
            prompt_list_1 = [prompt1[1]]
            if popt_kwargs['placeholder_position'] == 'end':
                prompt_list_1 = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list_1)]
            elif popt_kwargs['placeholder_position'] == 'start':
                prompt_list_1 = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list_1)]
            prompt1[1] = prompt_list_1[0]

            # prompt2[1] = prompt2[1] + " " + placeholder_string
            prompt_list_2 = [prompt2[1]]
            if popt_kwargs['placeholder_position'] == 'end':
                prompt_list_2 = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list_2)]
            elif popt_kwargs['placeholder_position'] == 'start':
                prompt_list_2 = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list_2)]
            prompt2[1] = prompt_list_2[0]
            # print("Prompt1: ", prompt1)
            # print("Prompt2: ", prompt2)

            with torch.no_grad():
                null_prompt_embeds, prompt_embeds, add_cond_kwargs = self.get_embed_from_prompt12(prompt1, prompt2)
            return prompt_embeds, add_cond_kwargs
        
        # print("self.text_enc_1.get_input_embeddings().weight.requires_grad: ", self.text_enc_1.get_input_embeddings().weight.requires_grad)
        # print("self.text_enc_2.get_input_embeddings().weight.requires_grad: ", self.text_enc_2.get_input_embeddings().weight.requires_grad)

        # para = list(self.text_enc_1.get_input_embeddings().parameters()) + list(self.text_enc_2.get_input_embeddings().parameters())
        # optimizer = Adam(para, lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))

        if popt_kwargs['te1']:
            para = self.text_enc_1.get_input_embeddings().parameters()
        elif popt_kwargs['te2']:
            para = self.text_enc_2.get_input_embeddings().parameters()
        else:
            para = list(self.text_enc_1.get_input_embeddings().parameters()) + list(self.text_enc_2.get_input_embeddings().parameters())
        optimizer = Adam(para, lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))
            


        # print("(before opt) self.text_enc_1.get_input_embeddings().weight.data[0]: ", self.text_enc_1.get_input_embeddings().weight.data[0])
        # print("(before opt) self.text_enc_1.get_input_embeddings().weight.data[49408]: ", self.text_enc_1.get_input_embeddings().weight.data[49408])
        # print("(before opt) self.text_enc_2.get_input_embeddings().weight.data[0]: ", self.text_enc_2.get_input_embeddings().weight.data[0])
        # print("(before opt) self.text_enc_2.get_input_embeddings().weight.data[49408]: ", self.text_enc_2.get_input_embeddings().weight.data[49408])

        # keep original embeddings as reference
        orig_embeds_params_enc1 = self.text_enc_1.get_input_embeddings().weight.data.clone()
        orig_embeds_params_enc2 = self.text_enc_2.get_input_embeddings().weight.data.clone()

        prompt1 = self.prompt1.copy()
        prompt2 = self.prompt2.copy()

        # add placeholder tokens only for prompt
        # prompt1[1] = prompt1[1] + " " + placeholder_string
        prompt_list_1 = [prompt1[1]]
        if popt_kwargs['placeholder_position'] == 'end':
            prompt_list_1 = [p + " " + " ".join(f"*_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list_1)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompt_list_1 = [" ".join(f"*_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list_1)]
        prompt1[1] = prompt_list_1[0]

        # prompt2[1] = prompt2[1] + " " + placeholder_string
        prompt_list_2 = [prompt2[1]]
        if popt_kwargs['placeholder_position'] == 'end':
            prompt_list_2 = [p + " " + " ".join(f"*_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list_2)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompt_list_2 = [" ".join(f"*_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list_2)]
        prompt2[1] = prompt_list_2[0]
        # print("Prompt1: ", prompt1)
        # print("Prompt2: ", prompt2)

        null_prompt_embeds, prompt_embeds, add_cond_kwargs = self.get_embed_from_prompt12(prompt1, prompt2)
        
        if popt_kwargs['debug_flag'] == 'no_opt':
            print("No optimization is performed (generated using init prompts).")
            return prompt_embeds, add_cond_kwargs

        at = self.scheduler.alphas_cumprod[t]

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

        if popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0:
            add_cond_kwargs_base_original = add_cond_kwargs_base.copy()
            add_cond_kwargs_base_original['text_embeds'] = add_cond_kwargs_base_original['text_embeds'].detach()
            add_cond_kwargs_base_original['time_ids'] = add_cond_kwargs_base_original['time_ids'].detach()

        add_cond_kwargs_base['text_embeds'] = add_cond_kwargs_base['text_embeds'][-1].unsqueeze(0).detach()
        add_cond_kwargs_base['time_ids'] = add_cond_kwargs_base['time_ids'][-1].unsqueeze(0).detach()
        
        for i in range(popt_kwargs['p_opt_iter']):
            if self.skip == 250 or self.skip == 125: # Lightning cases
                assert popt_kwargs['cfg_traj_opt'] == False # do not use this option for Lightning models

            if (popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0) or popt_kwargs['cfg_traj_opt']:
                if popt_kwargs['cfg_traj_opt']:
                    add_cond_kwargs_c = {}
                    add_cond_kwargs_c['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
                    add_cond_kwargs_c['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)

                    add_cond_kwargs_uc = {}
                    add_cond_kwargs_uc['text_embeds'] = add_cond_kwargs['text_embeds'][0].unsqueeze(0)
                    add_cond_kwargs_uc['time_ids'] = add_cond_kwargs['time_ids'][0].unsqueeze(0)
                    
                    noise_c = self.unet(zt, t, encoder_hidden_states=prompt_embeds,
                                    added_cond_kwargs=add_cond_kwargs_c)['sample']
                    with torch.no_grad():
                        noise_uc = self.unet(zt, t, encoder_hidden_states=null_prompt_embeds,
                                    added_cond_kwargs=add_cond_kwargs_uc)['sample']
                    noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
                else:
                    noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds.detach(), prompt_embeds, add_cond_kwargs)
                    noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
                

            else:
                add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
                add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)

                _, noise_pred = self.predict_noise(zt, t, None, prompt_embeds, add_cond_kwargs)



            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            # rand_noise = torch.randn_like(noise_pred, device=noise_pred.device)
            rand_noise = torch.randn(noise_pred.shape, device=noise_pred.device, dtype=noise_pred.dtype, generator=self.generator)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
            
            if popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0:
                noise_uc, noise_c = self.predict_noise(zt, t_mg, null_prompt_embeds.detach(), prompt_embeds_base.detach(), add_cond_kwargs_base_original)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            else:
                _, noise_pred_s = self.predict_noise(zs, t_mg, None, prompt_embeds_base.detach(), add_cond_kwargs_base)
            
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

            
            # # want to save zt, z0t, z0s as png files
            # with torch.no_grad():
            #     from torchvision.utils import save_image
            #     zt_img = self.decode(zt)
            #     zt_img = (zt_img / 2 + 0.5).clamp(0, 1)
            #     zt_img = zt_img.detach().cpu()
            #     save_image(zt_img, f"zt.png")

            #     z0t_img = self.decode(z0t)
            #     z0t_img = (z0t_img / 2 + 0.5).clamp(0, 1)
            #     z0t_img = z0t_img.detach().cpu()
            #     save_image(z0t_img, f"z0t.png")

            #     zs_img = self.decode(zs)
            #     zs_img = (zs_img / 2 + 0.5).clamp(0, 1)
            #     zs_img = zs_img.detach().cpu()
            #     save_image(zs_img, f"zs.png")

            #     z0s_img = self.decode(z0s)
            #     z0s_img = (z0s_img / 2 + 0.5).clamp(0, 1)
            #     z0s_img = z0s_img.detach().cpu()
            #     save_image(z0s_img, f"z0s.png")
            #     import sys; sys.exit()

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
            if popt_kwargs['te1']:
                self.restore_embedding(placeholder_token_ids_enc1, orig_embeds_params_enc1, self.tokenizer_1, self.text_enc_1)
            elif popt_kwargs['te2']:
                self.restore_embedding(placeholder_token_ids_enc2, orig_embeds_params_enc2, self.tokenizer_2, self.text_enc_2)
            else:
                self.restore_embedding(placeholder_token_ids_enc1, orig_embeds_params_enc1, self.tokenizer_1, self.text_enc_1)
                self.restore_embedding(placeholder_token_ids_enc2, orig_embeds_params_enc2, self.tokenizer_2, self.text_enc_2)
            
            # print("(after opt, after restore) self.text_enc_1.get_input_embeddings().weight.data[0]: ", self.text_enc_1.get_input_embeddings().weight.data[0])
            # print("(after opt, after restore) self.text_enc_1.get_input_embeddings().weight.data[49408]: ", self.text_enc_1.get_input_embeddings().weight.data[49408])
            # print("(after opt, after restore) self.text_enc_2.get_input_embeddings().weight.data[0]: ", self.text_enc_2.get_input_embeddings().weight.data[0])
            # print("(after opt, after restore) self.text_enc_2.get_input_embeddings().weight.data[49408]: ", self.text_enc_2.get_input_embeddings().weight.data[49408])
            # import ipdb; ipdb.set_trace()
            if not i == popt_kwargs['p_opt_iter'] - 1:
                null_prompt_embeds, prompt_embeds, add_cond_kwargs = self.get_embed_from_prompt12(prompt1, prompt2)
            else:
                with torch.no_grad():
                    null_prompt_embeds, prompt_embeds, add_cond_kwargs = self.get_embed_from_prompt12(prompt1, prompt2)

        return prompt_embeds, add_cond_kwargs
    
    @torch.enable_grad()
    def prompt_opt_alter_tes(self, zt, t, step, placeholder_token_ids_enc1, placeholder_token_ids_enc2, null_prompt_embeds, prompt_embeds_base, add_cond_kwargs_base, cfg_guidance, popt_kwargs):
        assert cfg_guidance > 0.0
        placeholder_string = popt_kwargs['placeholder_string']
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
        placeholder_symbol = placeholder_string.split("_")[0]

        decay_rate = popt_kwargs['lr_decay_rate']
        num_opt_tokens = popt_kwargs['num_opt_tokens']

        if (1. - step * decay_rate) <= 0:
            print("Learning rate is zero. Skipping prompt optimization and using the latest optimized embedding.")
            prompt1 = self.prompt1.copy()
            prompt2 = self.prompt2.copy()

            # add placeholder tokens only for prompt
            # prompt1[1] = prompt1[1] + " " + placeholder_string
            prompt_list_1 = [prompt1[1]]
            if popt_kwargs['placeholder_position'] == 'end':
                prompt_list_1 = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list_1)]
            elif popt_kwargs['placeholder_position'] == 'start':
                prompt_list_1 = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list_1)]
            prompt1[1] = prompt_list_1[0]

            # prompt2[1] = prompt2[1] + " " + placeholder_string
            prompt_list_2 = [prompt2[1]]
            if popt_kwargs['placeholder_position'] == 'end':
                prompt_list_2 = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list_2)]
            elif popt_kwargs['placeholder_position'] == 'start':
                prompt_list_2 = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list_2)]
            prompt2[1] = prompt_list_2[0]
            # print("Prompt1: ", prompt1)
            # print("Prompt2: ", prompt2)

            with torch.no_grad():
                null_prompt_embeds, prompt_embeds, add_cond_kwargs = self.get_embed_from_prompt12(prompt1, prompt2)
            return prompt_embeds, add_cond_kwargs
        
        # activate text_enc_1 and text_enc_2
        for param in self.text_enc_1.get_input_embeddings().parameters():
            param.requires_grad = True
        for param in self.text_enc_2.get_input_embeddings().parameters():
            param.requires_grad = True

        para_te1 = self.text_enc_1.get_input_embeddings().parameters()
        para_te2 = self.text_enc_2.get_input_embeddings().parameters()
        optimizer_te1 = Adam(para_te1, lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))
        optimizer_te2 = Adam(para_te2, lr=popt_kwargs['p_opt_lr_te2'] * (1. - step * decay_rate))

        # un-freeze text_enc_1, since we start from optimizing text_enc_1
        for param in self.text_enc_1.get_input_embeddings().parameters():
            param.requires_grad = True

        # freeze text_enc_2
        for param in self.text_enc_2.get_input_embeddings().parameters():
            param.requires_grad = False

        # keep original embeddings as reference
        orig_embeds_params_enc1 = self.text_enc_1.get_input_embeddings().weight.data.clone()
        orig_embeds_params_enc2 = self.text_enc_2.get_input_embeddings().weight.data.clone()

        prompt1 = self.prompt1.copy()
        prompt2 = self.prompt2.copy()

        # add placeholder tokens only for prompt
        prompt_list_1 = [prompt1[1]]
        if popt_kwargs['placeholder_position'] == 'end':
            prompt_list_1 = [p + " " + " ".join(f"*_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list_1)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompt_list_1 = [" ".join(f"*_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list_1)]
        prompt1[1] = prompt_list_1[0]

        prompt_list_2 = [prompt2[1]]
        if popt_kwargs['placeholder_position'] == 'end':
            prompt_list_2 = [p + " " + " ".join(f"*_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list_2)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompt_list_2 = [" ".join(f"*_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list_2)]
        prompt2[1] = prompt_list_2[0]

        null_prompt_embeds, prompt_embeds, add_cond_kwargs = self.get_embed_from_prompt12(prompt1, prompt2, te2_nograd=True)
        
        if popt_kwargs['debug_flag'] == 'no_opt':
            print("No optimization is performed (generated using init prompts).")
            return prompt_embeds, add_cond_kwargs

        at = self.scheduler.alphas_cumprod[t]

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
            # import ipdb; ipdb.set_trace()
            next_t = t - self.skip + 1
            t_mg = int(
                len(self.scheduler.alphas_cumprod_default) - next_t
            )
            at_mg = self.scheduler.alphas_cumprod[t_mg]
            print("using dynamic_pr. t_mg is : ", t_mg)
            print("at_mg: ", at_mg)
        
        t_mg = torch.tensor(t_mg).to(t.device)

        if popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0:
            add_cond_kwargs_base_original = add_cond_kwargs_base.copy()
            add_cond_kwargs_base_original['text_embeds'] = add_cond_kwargs_base_original['text_embeds'].detach()
            add_cond_kwargs_base_original['time_ids'] = add_cond_kwargs_base_original['time_ids'].detach()

        add_cond_kwargs_base['text_embeds'] = add_cond_kwargs_base['text_embeds'][-1].unsqueeze(0).detach()
        add_cond_kwargs_base['time_ids'] = add_cond_kwargs_base['time_ids'][-1].unsqueeze(0).detach()
        
        for i in range(popt_kwargs['p_opt_iter']):
            #################################################################
            ###################### optimize text_enc_1 ######################
            #################################################################

            if popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0:
                noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds.detach(), prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            else:
                add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
                add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)

                _, noise_pred = self.predict_noise(zt, t, None, prompt_embeds, add_cond_kwargs)
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            rand_noise = torch.randn_like(noise_pred, device=noise_pred.device)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
            
            if popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0:
                noise_uc, noise_c = self.predict_noise(zt, t_mg, null_prompt_embeds.detach(), prompt_embeds_base.detach(), add_cond_kwargs_base_original)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            else:
                _, noise_pred_s = self.predict_noise(zs, t_mg, None, prompt_embeds_base.detach(), add_cond_kwargs_base)
            
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
            loss = -1 * ms.mean()

            optimizer_te1.zero_grad()
            loss.backward()
            optimizer_te1.step()

            # Let's make sure we don't update any embedding weights besides the newly added token
            self.restore_embedding(placeholder_token_ids_enc1, orig_embeds_params_enc1, self.tokenizer_1, self.text_enc_1)


            #################################################################
            ###################### optimize text_enc_2 ######################
            #################################################################

            # un-freeze text_enc_2
            for param in self.text_enc_2.get_input_embeddings().parameters():
                param.requires_grad = True
            
            # freeze text_enc_1
            for param in self.text_enc_1.get_input_embeddings().parameters():
                param.requires_grad = False

            null_prompt_embeds, prompt_embeds, add_cond_kwargs = self.get_embed_from_prompt12(prompt1, prompt2, te1_nograd=True)

            if popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0:
                noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds.detach(), prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            else:
                add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
                add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)

                _, noise_pred = self.predict_noise(zt, t, None, prompt_embeds, add_cond_kwargs)
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            rand_noise = torch.randn_like(noise_pred, device=noise_pred.device)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
            
            if popt_kwargs['cfg_tweedie'] and cfg_guidance != 1.0:
                noise_uc, noise_c = self.predict_noise(zt, t_mg, null_prompt_embeds.detach(), prompt_embeds_base.detach(), add_cond_kwargs_base_original)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            else:
                _, noise_pred_s = self.predict_noise(zs, t_mg, None, prompt_embeds_base.detach(), add_cond_kwargs_base)
            
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
            loss = -1 * ms.mean()

            optimizer_te2.zero_grad()
            loss.backward()
            optimizer_te2.step()

            # Let's make sure we don't update any embedding weights besides the newly added token
            self.restore_embedding(placeholder_token_ids_enc2, orig_embeds_params_enc2, self.tokenizer_2, self.text_enc_2)

            if not i == popt_kwargs['p_opt_iter'] - 1:
                # un-freeze text_enc_1, since we will optimize text_enc_1 in the next iter
                for param in self.text_enc_1.get_input_embeddings().parameters():
                    param.requires_grad = True

                # freeze text_enc_2
                for param in self.text_enc_2.get_input_embeddings().parameters():
                    param.requires_grad = False
                
                null_prompt_embeds, prompt_embeds, add_cond_kwargs = self.get_embed_from_prompt12(prompt1, prompt2, te2_nograd=True)
            else:
                # freeze both encoders, since the optimization is done in the current timestep
                for param in self.text_enc_1.get_input_embeddings().parameters():
                    param.requires_grad = False
                for param in self.text_enc_2.get_input_embeddings().parameters():
                    param.requires_grad = False

                with torch.no_grad():
                    null_prompt_embeds, prompt_embeds, add_cond_kwargs = self.get_embed_from_prompt12(prompt1, prompt2)

        return prompt_embeds, add_cond_kwargs



    def get_embed_from_prompt12(self, prompt1, prompt2, te1_nograd=False, te2_nograd=False):
        # 0. Default height and width to unet
        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor

        original_size = self.original_size or (height, width)
        target_size = self.target_size or (height, width)

        # embedding
        (null_prompt_embeds,
         prompt_embeds,
         pool_null_embed,
         pool_prompt_embed) = self.differentiable_get_text_embed(prompt1[0], prompt1[1], prompt2[0], prompt2[1], self.clip_skip, te1_nograd, te2_nograd)

        # prepare kwargs for SDXL
        add_text_embeds = pool_prompt_embed
        add_time_ids = self._get_add_time_ids(
            original_size,
            self.crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
        )

        if self.negative_original_size is not None and self.negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                self.negative_original_size,
                self.negative_crops_coords_top_left,
                self.negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
            )
        else:
            negative_add_time_ids = add_time_ids
        negative_text_embeds = pool_null_embed

        if self.cfg_guidance != 0.0 and self.cfg_guidance != 1.0:
            # do cfg
            add_text_embeds = torch.cat([negative_text_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_cond_kwargs = {
            'text_embeds': add_text_embeds.to(self.device),
            'time_ids': add_time_ids.to(self.device)
        }
        return null_prompt_embeds, prompt_embeds, add_cond_kwargs
    
    # def get_embed_from_prompt12(self, prompt1, prompt2):
    #     # 0. Default height and width to unet
    #     height = self.default_sample_size * self.vae_scale_factor
    #     width = self.default_sample_size * self.vae_scale_factor

    #     original_size = self.original_size or (height, width)
    #     target_size = self.target_size or (height, width)

    #     # embedding
    #     (null_prompt_embeds,
    #      prompt_embeds,
    #      pool_null_embed,
    #      pool_prompt_embed) = self.differentiable_get_text_embed(prompt1[0], prompt1[1], prompt2[0], prompt2[1], self.clip_skip)

    #     # prepare kwargs for SDXL
    #     add_text_embeds = pool_prompt_embed
    #     add_time_ids = self._get_add_time_ids(
    #         original_size,
    #         self.crops_coords_top_left,
    #         target_size,
    #         dtype=prompt_embeds.dtype,
    #         text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
    #     )

    #     if self.negative_original_size is not None and self.negative_target_size is not None:
    #         negative_add_time_ids = self._get_add_time_ids(
    #             self.negative_original_size,
    #             self.negative_crops_coords_top_left,
    #             self.negative_target_size,
    #             dtype=prompt_embeds.dtype,
    #             text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
    #         )
    #     else:
    #         negative_add_time_ids = add_time_ids
    #     negative_text_embeds = pool_null_embed

    #     if self.cfg_guidance != 0.0 and self.cfg_guidance != 1.0:
    #         # do cfg
    #         add_text_embeds = torch.cat([negative_text_embeds, add_text_embeds], dim=0)
    #         add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    #     add_cond_kwargs = {
    #         'text_embeds': add_text_embeds.to(self.device),
    #         'time_ids': add_time_ids.to(self.device)
    #     }
    #     return null_prompt_embeds, prompt_embeds, add_cond_kwargs


    def restore_embedding(self, placeholder_token_ids, orig_embeds_params, tokenizer, text_enc):
        index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
        index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

        with torch.no_grad():
            text_enc.get_input_embeddings().weight[
                index_no_updates
            ] = orig_embeds_params[index_no_updates]


    # def initialize_embedding(self, tokenizer, text_enc, popt_kwargs):
    #     placeholder_string = popt_kwargs['placeholder_string']
    #     num_opt_tokens = popt_kwargs['num_opt_tokens']
    #     init_word = popt_kwargs['init_word']

    #     placeholder_tokens = [placeholder_string]
    #     additional_tokens = []
    #     for i in range(1, num_opt_tokens):
    #         print("Additional placeholder token: ", f"{placeholder_string}_{i}")
    #         additional_tokens.append(f"{placeholder_string}_{i}")
    #     placeholder_tokens += additional_tokens
    #     print("Placeholder tokens: ", placeholder_tokens)

    #     num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    #     print("Number of tokens added to tokenizer: ", num_added_tokens)
    #     if num_added_tokens != num_opt_tokens:
    #         # print(f"The tokenizer already contains the token {placeholder_string}.")
    #         raise ValueError(
    #             f"The tokenizer already contains the token {placeholder_string}. Please pass a different"
    #             " `placeholder_token` that is not already in the tokenizer."
    #         )
        
    #     if not init_word == "":
    #         # Convert the initializer_token, placeholder_token to ids
    #         token_ids = tokenizer.encode(init_word, add_special_tokens=False)
    #         # Check if initializer_token is a single token or a sequence of tokens
    #         if len(token_ids) > 1:
    #             raise ValueError("The initializer token must be a single token.")
            
    #         initializer_token_id = token_ids[0]

    #     placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    #     print("Placeholder token ids: ", placeholder_token_ids)

    #     # Resize the token embeddings as we are adding new special tokens to the tokenizer
    #     text_enc.resize_token_embeddings(len(tokenizer))

    #     # Initialise the newly added placeholder token with the embeddings of the initializer token
    #     token_embeds = text_enc.get_input_embeddings().weight.data
    #     if not init_word == "":
    #         with torch.no_grad():
    #             for token_id in placeholder_token_ids:
    #                 print("Token id: ", token_id)
    #                 print(f"token_embeds[{token_id}] (before replacement): ", token_embeds[token_id])
    #                 token_embeds[token_id] = token_embeds[initializer_token_id].clone()
    #                 print(f"token_embeds[{token_id}] (after replacement): ", token_embeds[token_id])

    #     # Freeze all parameters except for the token embeddings in text encoder
    #     text_enc.text_model.encoder.requires_grad_(False)
    #     text_enc.text_model.final_layer_norm.requires_grad_(False)
    #     text_enc.text_model.embeddings.position_embedding.requires_grad_(False)

    #     return placeholder_token_ids

    def initialize_embedding(self, tokenizer, text_enc, popt_kwargs, b_size=1):
        num_opt_tokens = popt_kwargs['num_opt_tokens'] * b_size # assignging popt_kwargs['num_opt_tokens'] tokens per each sample
        init_type = popt_kwargs['init_type']
        init_word = popt_kwargs['init_word']
        init_gau_scale = popt_kwargs['init_gau_scale']
        init_rand_vocab = popt_kwargs['init_rand_vocab']
        init_max_cs = popt_kwargs['init_max_cs']
        num_vocab = len(tokenizer)

        assert init_type in ['default', 'word', 'gaussian', 'gaussian_white']
        
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
            # among init_rand_vocab and init_min_cs, either one should be True
            assert (init_rand_vocab != init_max_cs) or (init_rand_vocab == False and init_max_cs == False)
            if init_rand_vocab:
                token_embeds = text_enc.get_input_embeddings().weight.data
                with torch.no_grad():
                    for token_id in placeholder_token_ids:
                        rand_idx = torch.randint(0, num_vocab, (1,), generator=self.generator)
                        print(f"Initialize token id {token_id} as a random vocabulary of index {rand_idx}.")
                        token_embeds[token_id] = token_embeds[rand_idx].clone()
            elif init_max_cs:
                assert self.prompt1 == self.prompt2 # for now, we only support the same prompt for both encoders
                # get rid of indices of special tokens in token_embeds_base
                special_token_ids = torch.tensor(tokenizer.all_special_ids)
                token_embeds_min_cs = token_embeds_base[~torch.isin(torch.arange(token_embeds_base.shape[0]), special_token_ids)]

                prompt1 = self.prompt1.copy()
                prompt1_ids = tokenizer.encode(prompt1[1], add_special_tokens=False, return_tensors='pt').squeeze()
                prompt1_embeds = token_embeds_min_cs[prompt1_ids]
                cos_sims = torch.einsum('ij,kj->ik', token_embeds_min_cs, prompt1_embeds).sum(dim=-1)
                min_idx = cos_sims.argmax()
                # import ipdb; ipdb.set_trace()
                
                token_embeds = text_enc.get_input_embeddings().weight.data
                with torch.no_grad():    
                    for token_id in placeholder_token_ids:
                        print(f"Initialize token id {token_id} as a min-cs vocabulary of index {min_idx}.")
                        token_embeds[token_id] = token_embeds_min_cs[min_idx].clone()
            else:
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
    def latent_opt(self, zt, t, step, null_prompt_embeds, prompt_embeds, add_cond_kwargs, etc_kwargs):
        decay_rate = etc_kwargs['l_lr_decay_rate']

        if (1. - step * decay_rate) <= 0:
            print("Learning rate is zero. Skipping prompt optimization.")
            return zt
        
        zt = zt.detach()
        zt.requires_grad = True

        optimizer = Adam([zt], lr=etc_kwargs['l_opt_lr'] * (1. - step * decay_rate))

        at = self.scheduler.alphas_cumprod[t]

        t_mg = int(
            len(self.scheduler.alphas_cumprod_default) * etc_kwargs['l_p_ratio']
        )
        if etc_kwargs['l_dynamic_pr']:
            t_mg = int(
                len(self.scheduler.alphas_cumprod_default) - t
            )
            print("using dynamic_pr. t_mg is : ", t_mg)
        at_mg = self.scheduler.alphas_cumprod_default[t_mg]
        t_mg = torch.tensor(t_mg).to(t.device)

        add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
        add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)
        
        for i in range(etc_kwargs['l_opt_iter']):
            _, noise_pred = self.predict_noise(zt, t, None, prompt_embeds, add_cond_kwargs)
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            # rand_noise = torch.randn_like(noise_pred, device=noise_pred.device)
            rand_noise = torch.randn(noise_pred.shape, device=noise_pred.device, dtype=noise_pred.dtype, generator=self.generator)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
            
            _, noise_pred_s = self.predict_noise(zs, t_mg, None, prompt_embeds.detach(), add_cond_kwargs)
            
            # tweedie (x0doublehat)
            z0s = (zs - (1-at_mg).sqrt() * noise_pred_s) / at_mg.sqrt()
            if etc_kwargs['l_opt_sg']:
                z0s = z0s.detach()

            assert z0t.shape == z0s.shape and len(z0t.shape) == 4
            ms = (z0t - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
            loss = -1 * ms.mean()
            # print("ms: ", ms)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return zt.detach()
    
    # def cads(self, c, t, tau1=0.6, tau2=0.9, noise_scale=0.25, psi=1.0, rescale=True):
    def cads(self, c, t, tau1=0.8, tau2=1.0, noise_scale=0.1, psi=1.0, rescale=True):
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
            # dim of y_in: (b, 77, 768)
            assert len(y_in.shape) == 3
            y_in_mean, y_in_std = y_in.mean(dim=[1,2], keepdim=True), y_in.std(dim=[1,2], keepdim=True)
            sqrt_gamma = gamma ** 0.5
            y = sqrt_gamma * y_in + noise_scale * (1 - sqrt_gamma) * torch.randn(y_in.shape, device=y_in.device, generator=self.generator)
            if rescale:
                y_scaled = (y - y.mean(dim=[1,2], keepdim=True)) / (y.std(dim=[1,2], keepdim=True)) * y_in_std + y_in_mean
                y = psi * y_scaled + (1 - psi) * y
            return y
        gamma = linear_schedule(t, tau1, tau2)
        return add_noise(c, gamma=gamma, noise_scale=noise_scale, psi=psi, rescale=rescale)
                

class SDXLLightning(SDXL):
    def __init__(self, 
                 solver_config: dict,
                 base_model_key:str=os.path.join(_CKPT_DIR, "stable-diffusion-xl-base-1.0"),
                 light_model_ckpt:str=os.path.join(_CKPT_DIR, "sdxl_lightning_4step_unet.safetensors"),
                 dtype=torch.float16,
                 device='cuda',
                 seed: int = 42):

        self.device = device

        # load the student model
        unet = UNet2DConditionModel.from_config(os.path.join(base_model_key, "unet"), local_files_only=True).to(device, torch.float16)
        ext = os.path.splitext(light_model_ckpt)[1]
        if ext == ".safetensors":
            state_dict = load_file(light_model_ckpt)
        else:
            state_dict = torch.load(light_model_ckpt, map_location="cpu")
        print(unet.load_state_dict(state_dict, strict=True))
        unet.requires_grad_(False)
        self.unet = unet

        #pipe2 = StableDiffusionXLPipeline.from_single_file(light_model_ckpt, torch_dtype=dtype).to(device)
        pipe = StableDiffusionXLPipeline.from_pretrained(base_model_key, unet=self.unet, torch_dtype=dtype, local_files_only=True).to(device)
        self.dtype = dtype

        # avoid overflow in float16
        self.vae = AutoencoderKL.from_pretrained(os.path.join(_CKPT_DIR, "sdxl-vae-fp16-fix"), torch_dtype=dtype, local_files_only=True).to(device)

        self.tokenizer_1_base = copy.deepcopy(pipe.tokenizer)
        self.tokenizer_2_base = copy.deepcopy(pipe.tokenizer_2)
        self.text_enc_1_base = copy.deepcopy(pipe.text_encoder)
        self.text_enc_2_base = copy.deepcopy(pipe.text_encoder_2)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size

        # sampling parameters
        self.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        self.total_alphas = self.scheduler.alphas_cumprod.clone()
        N_ts = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = N_ts // solver_config.num_sampling

        #self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod_default = self.scheduler.alphas_cumprod
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod]).to(device)
        
        # a dedicated generator for various purposes
        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(seed)

###########################################
# Base version
###########################################

@register_solver('ddim')
class BaseDDIM(SDXL):
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        popt_kwargs=None,
                        etc_kwargs=None,
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        if etc_kwargs['sync_initial_noise']:
            from utils_local.log_util import set_seed
            set_seed(etc_kwargs['seed'])
            
        zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        if etc_kwargs['trunc_tau'] != 1.0:
            # zt = zt * math.sqrt(etc_kwargs['trunc_tau'])
            print(f"scaling zT by trunc_tau: {etc_kwargs['trunc_tau']}")
            zt = zt * etc_kwargs['trunc_tau']

        # initialize embedding for prompt-opt
        if popt_kwargs['prompt_opt']:
            if popt_kwargs['te1']:
                self.text_enc_1 = self.text_enc_1.to(torch.float32)
                placeholder_token_ids_enc1 = self.initialize_embedding(self.tokenizer_1, self.text_enc_1, popt_kwargs)
                placeholder_token_ids_enc2 = None
            elif popt_kwargs['te2']:
                self.text_enc_2 = self.text_enc_2.to(torch.float32)
                placeholder_token_ids_enc1 = None
                placeholder_token_ids_enc2 = self.initialize_embedding(self.tokenizer_2, self.text_enc_2, popt_kwargs)
            else:
                self.text_enc_1 = self.text_enc_1.to(torch.float32)
                self.text_enc_2 = self.text_enc_2.to(torch.float32)
                placeholder_token_ids_enc1 = self.initialize_embedding(self.tokenizer_1, self.text_enc_1, popt_kwargs)
                placeholder_token_ids_enc2 = self.initialize_embedding(self.tokenizer_2, self.text_enc_2, popt_kwargs)
            self.vae.requires_grad_(False)

        prompt_embeds_base = prompt_embeds.detach().clone()
        null_prompt_embeds_base = null_prompt_embeds.detach().clone()
        add_cond_kwargs_base = add_cond_kwargs.copy()
        
        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            # for prompt-opt
            if popt_kwargs['prompt_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                if popt_kwargs['alter_tes']:
                    prompt_embeds, add_cond_kwargs = self.prompt_opt_alter_tes(
                        zt,
                        t,
                        step,
                        placeholder_token_ids_enc1,
                        placeholder_token_ids_enc2,
                        null_prompt_embeds,
                        prompt_embeds_base,
                        add_cond_kwargs_base,
                        cfg_guidance,
                        popt_kwargs
                    )
                else:
                    prompt_embeds, add_cond_kwargs = self.prompt_opt(
                        zt,
                        t,
                        step,
                        placeholder_token_ids_enc1,
                        placeholder_token_ids_enc2,
                        null_prompt_embeds,
                        prompt_embeds_base,
                        add_cond_kwargs_base,
                        cfg_guidance,
                        popt_kwargs
                    )
            else:
                if popt_kwargs['base_prompt_after_popt']:
                    prompt_embeds = prompt_embeds_base.detach().clone()
                    add_cond_kwargs = add_cond_kwargs_base.copy()

            if etc_kwargs['use_cads']:
                prompt_embeds = self.cads(prompt_embeds_base, t.item())
                null_prompt_embeds = self.cads(null_prompt_embeds_base, t.item())
                add_cond_kwargs = add_cond_kwargs_base.copy()
                add_cond_kwargs['text_embeds'] = self.cads(add_cond_kwargs_base['text_embeds'].unsqueeze(0), t.item()).squeeze(0)
                    
            if etc_kwargs['latent_opt'] and t > etc_kwargs['l_t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % etc_kwargs['l_inter_rate'] == 0:
                zt = self.latent_opt(
                    zt.detach(),
                    t,
                    step,
                    null_prompt_embeds,
                    prompt_embeds,
                    add_cond_kwargs,
                    etc_kwargs
                ) # return optimized prompts

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # for random noise
            if etc_kwargs['ddim_eta'] > 0.0:
                sigma_t = etc_kwargs['ddim_eta'] * torch.sqrt((1 - at_next) / (1 - at) * (1 - at / at_next))
                noise_rand = torch.randn_like(zt) * sigma_t
                zt = at_next.sqrt() * z0t + (1-at_next-sigma_t**2).sqrt() * noise_pred + noise_rand

            # for deterministic case: eta = 0.0
            else:
                zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = { 'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return z0t


@register_solver('ddim_lightning')
class BaseDDIMLight(BaseDDIM, SDXLLightning):
    def __init__(self, **kwargs):
        SDXLLightning.__init__(self, **kwargs)
    
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        popt_kwargs=None,
                        etc_kwargs=None,
                        **kwargs):
        assert cfg_guidance == 1.0, "CFG should be turned off in the lightning version"
        return super().reverse_process(null_prompt_embeds, 
                                        prompt_embeds, 
                                        cfg_guidance, 
                                        add_cond_kwargs, 
                                        shape, 
                                        callback_fn,
                                        popt_kwargs=popt_kwargs,
                                        etc_kwargs=etc_kwargs,
                                        **kwargs)


@register_solver("ddim_edit")
class EditWardSwapDDIM(BaseDDIM):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               prompt1 = ["", "", ""],
               prompt2 = ["", "", ""],
               cfg_guidance:float=5.0,
               original_size: Optional[Tuple[int, int]]=None,
               crops_coords_top_left: Tuple[int, int]=(0, 0),
               target_size: Optional[Tuple[int, int]]=None,
               negative_original_size: Optional[Tuple[int, int]]=None,
               negative_crops_coords_top_left: Tuple[int, int]=(0, 0),
               negative_target_size: Optional[Tuple[int, int]]=None,
               clip_skip: Optional[int]=None,
               **kwargs):

        # 0. Default height and width to unet
        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # embedding
        (null_prompt_embeds,
         src_prompt_embeds,
         pool_null_embed,
         pool_src_prompt_embed) = self.get_text_embed(prompt1[0], prompt1[1], prompt2[0], prompt2[1], clip_skip)

        (_,
         tgt_prompt_embeds,
         _,
         pool_tgt_prompt_embed) = self.get_text_embed(prompt1[0], prompt1[2], prompt2[0], prompt2[2], clip_skip)

        # prepare kwargs for SDXL
        add_src_text_embeds = pool_src_prompt_embed
        add_tgt_text_embeds = pool_tgt_prompt_embed

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=src_prompt_embeds.dtype,
            text_encoder_projection_dim=int(pool_src_prompt_embed.shape[-1]),
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=src_prompt_embeds.dtype,
                text_encoder_projection_dim=int(pool_src_prompt_embed.shape[-1]),
            )
        else:
            negative_add_time_ids = add_time_ids
        negative_text_embeds = pool_null_embed 

        if cfg_guidance != 0.0 and cfg_guidance != 1.0:
            # do cfg
            add_src_text_embeds = torch.cat([negative_text_embeds, add_src_text_embeds], dim=0)
            add_tgt_text_embeds = torch.cat([negative_text_embeds, add_tgt_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_src_cond_kwargs = {
            'text_embeds': add_src_text_embeds.to(self.device),
            'time_ids': add_time_ids.to(self.device)
        }

        add_tgt_cond_kwargs = {
            'text_embeds': add_tgt_text_embeds.to(self.device),
            'time_ids': add_time_ids.to(self.device)
        }

        # reverse sampling
        zt = self.reverse_process(null_prompt_embeds,
                                  src_prompt_embeds, 
                                  tgt_prompt_embeds,
                                  cfg_guidance,
                                  add_src_cond_kwargs,
                                  add_tgt_cond_kwargs,
                                  **kwargs)

        # decode
        with torch.no_grad():
            img = self.decode(zt)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

    def reverse_process(self,
                        null_prompt_embeds,
                        src_prompt_embeds,
                        tgt_prompt_embed,
                        cfg_guidance,
                        add_src_cond_kwargs,
                        add_tgt_cond_kwargs,
                        callback_fn=None,
                        popt_kwargs=None,
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=kwargs.get('src_img', None),
                                    uc=null_prompt_embeds,
                                    c=src_prompt_embeds,
                                    cfg_guidance=cfg_guidance,
                                    add_cond_kwargs=add_src_cond_kwargs)

        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_next = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, 
                                                       null_prompt_embeds,
                                                       tgt_prompt_embed,
                                                       add_tgt_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return z0t


###########################################
# CFG++ version
###########################################

@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(SDXL):
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        popt_kwargs=None,
                        etc_kwargs=None,
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        
        # initialize embedding for prompt-opt
        if popt_kwargs['prompt_opt']:
            self.text_enc_1 = self.text_enc_1.to(torch.float32)
            self.text_enc_2 = self.text_enc_2.to(torch.float32)
            placeholder_token_ids_enc1 = self.initialize_embedding(self.tokenizer_1, self.text_enc_1, popt_kwargs)
            placeholder_token_ids_enc2 = self.initialize_embedding(self.tokenizer_2, self.text_enc_2, popt_kwargs)
            self.vae.requires_grad_(False)
            # self.unet.requires_grad_(False)

        prompt_embeds_base = prompt_embeds.detach().clone()
        add_cond_kwargs_base = add_cond_kwargs.copy()

        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]
            
            # for prompt-opt
            if popt_kwargs['prompt_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                # prompt_embeds_base = prompt_embeds.detach().clone()
                # add_cond_kwargs_base = add_cond_kwargs.copy()
                prompt_embeds, add_cond_kwargs = self.prompt_opt(
                    zt,
                    t,
                    step,
                    placeholder_token_ids_enc1,
                    placeholder_token_ids_enc2,
                    null_prompt_embeds,
                    prompt_embeds_base,
                    add_cond_kwargs_base,
                    cfg_guidance,
                    popt_kwargs
                ) # return optimized prompts
            else:
                if popt_kwargs['base_prompt_after_popt']:
                    prompt_embeds = prompt_embeds_base.detach().clone()
                    add_cond_kwargs = add_cond_kwargs_base.copy()


            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = { 'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return z0t

@register_solver('ddim_cfg++_lightning')
class BaseDDIMCFGppLight(BaseDDIMCFGpp, SDXLLightning):
    def __init__(self, **kwargs):
        SDXLLightning.__init__(self, **kwargs)
    
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        popt_kwargs=None,
                        etc_kwargs=None,
                        **kwargs):
        assert cfg_guidance == 1.0, "CFG should be turned off in the lightning version"
        return super().reverse_process(null_prompt_embeds, 
                                        prompt_embeds, 
                                        cfg_guidance, 
                                        add_cond_kwargs, 
                                        shape,
                                        callback_fn,
                                        popt_kwargs=popt_kwargs,
                                        etc_kwargs=etc_kwargs,
                                        **kwargs)


@register_solver("ddim_edit_cfg++")
class EditWardSwapDDIMCFGpp(EditWardSwapDDIM):
    @torch.no_grad()
    def inversion(self, z0, uc, c, cfg_guidance, add_cond_kwargs):
        # if we use cfg_guidance=0.0 or 1.0 for inversion, add_cond_kwargs must be splitted. 
        if cfg_guidance == 0.0 or cfg_guidance == 1.0:
            add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
            add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)

        zt = z0.clone().to(self.device)
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            noise_uc, noise_c  = self.predict_noise(zt, t, uc, c, add_cond_kwargs)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at_prev).sqrt() * noise_uc) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

        return zt

    def reverse_process(self,
                        null_prompt_embeds,
                        src_prompt_embeds,
                        tgt_prompt_embed,
                        cfg_guidance,
                        add_src_cond_kwargs,
                        add_tgt_cond_kwargs,
                        callback_fn=None,
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=kwargs.get('src_img', None),
                                    uc=null_prompt_embeds,
                                    c=src_prompt_embeds,
                                    cfg_guidance=cfg_guidance,
                                    add_cond_kwargs=add_src_cond_kwargs)

        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_next = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, 
                                                       null_prompt_embeds,
                                                       tgt_prompt_embed,
                                                       add_tgt_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return z0t
#############################

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


#############################
@register_solver("ddim_jepa")
class DDIMWithJEPA(SDXL):
    """
    DDIM solver with JEPA guidance for SDXL
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
            jepa_backbone = self.jepa_config.get('jepa_backbone', 'dinov2_vits14')

            if jepa_config.get('use_jepa', False):
                if 'dinov2' in jepa_backbone.lower():
                    # Disable efficient/flash SDPA for DINOv2
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
                    raise ValueError(f"Unknown jepa_backbone: {jepa_backbone}. Choose 'dinov2_*' or 'metaclip'")

                self.jepa_rng = torch.Generator(device=self.device)
                self.jepa_rng.manual_seed(self.seed)
        finally:
            # Restore global RNG state
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

    def compute_jepa_gradient(self, zt, step, total_steps, t=None, at=None, at_prev=None,
                              uc=None, c=None, add_cond_kwargs=None, cfg_guidance=7.5):
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
        jepa_backbone = cfg.get('jepa_backbone', 'dinov2_vits14')
        eps = 1e-8

        # Normalization stats based on backbone type
        if 'dinov2' in jepa_backbone.lower():
            # ImageNet normalization stats for DINOv2
            norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(zt.device)
            norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(zt.device)
        elif 'metaclip' in jepa_backbone.lower():
            # CLIP normalization stats for MetaCLIP
            norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(zt.device)
            norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(zt.device)
        else:
            raise ValueError(f"Unknown jepa_backbone: {jepa_backbone}. Choose 'dinov2_*' or 'metaclip'")

        # Check timing (t_ratio: 1.0 at start -> 0.0 at end)
        t_ratio = 1.0 - (step / total_steps)
        if step % g_interval != 0 or t_ratio > g_start_t:
            return None

        r = k + p

        with torch.enable_grad():
            zt_in = zt.detach().clone().requires_grad_(True)

            # Re-compute noise_pred with zt_in to build computational graph
            noise_uc, noise_c = self.predict_noise(zt_in, t, uc, c, add_cond_kwargs)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # Predict clean latent from noisy latent
            z0t = (zt_in - (1-at).sqrt() * noise_pred) / at.sqrt()

            # Decode latent -> image [0,1]
            # Get VAE dtype and convert accordingly
            vae_dtype = next(self.vae.parameters()).dtype
            z_scaled = (z0t / self.vae.config.scaling_factor).to(vae_dtype)
            img = self.vae.decode(z_scaled).sample
            x0p = (img / 2 + 0.5).clamp(0, 1).float()

            # Resize and normalize for JEPA backbone
            x0p = F.interpolate(x0p, size=(jg_img_size, jg_img_size), mode="bilinear", align_corners=False)
            x0p = (x0p - norm_mean) / norm_std

            B, C, Hp, Wp = x0p.shape
            
            # Jx224
            f_base = self.f_jepa(x0p)
            
            # # Manual Jv/JTu to avoid xFormers/SDPA jvp incompatibility
            # def Jv(v, create_graph=False):
            #     # Forward-mode AD via finite difference: (f(x + eps*v) - f(x)) / eps
            #     eps_fd = 1e-4
            #     with torch.no_grad():
            #         f_base = self.f_jepa(x0p)
            #         f_plus = self.f_jepa(x0p + eps_fd * v)
            #     return (f_plus - f_base) / eps_fd

            # def JTu(u, create_graph=False):
            #     # Reverse-mode AD via standard backward
            #     # Use x0p directly to maintain graph connection to zt_in
            #     out = self.f_jepa(x0p)
            #     grad = torch.autograd.grad(out, x0p, grad_outputs=u,
            #                               create_graph=create_graph, retain_graph=True)[0]
            #     return grad
            
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

                # 4) Singular values -> JEPA loss (requires float32)
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
            print(f"JEPA scaling at step {step}:", variance.item())
            final_grad = eta * variance * grad
            return final_grad
        elif jg_schedule == 'constant':
            print("JEPA constant scaling at step", step)
            return eta * grad

    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        popt_kwargs=None,
                        etc_kwargs=None,
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        if etc_kwargs['sync_initial_noise']:
            from utils_local.log_util import set_seed
            set_seed(etc_kwargs['seed'])
            
        # ### DEBUG: reproducing loop for 135th image ###
        # B, C, Hp, Wp = (1, 3, 224, 224)
        # r = 5
        # print("DEBUG: looping to reach 135th image")    
        # for _ in tqdm(range(134)):
        #     zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        #     Omega = torch.randn(B, r, C, Hp, Wp, device=zt.device, dtype=zt.dtype,
        #                        generator=self.jepa_rng)
        #     Omega = torch.randn(B, r, C, Hp, Wp, device=zt.device, dtype=zt.dtype,
        #                        generator=self.jepa_rng)
        #     Omega = torch.randn(B, r, C, Hp, Wp, device=zt.device, dtype=zt.dtype,
        #                        generator=self.jepa_rng)
        # del Omega
        
        

        b_size = prompt_embeds.shape[0]
        zt = self.initialize_latent(size=(b_size, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        # zt = zt.requires_grad_()

        if etc_kwargs['trunc_tau'] != 1.0:
            print(f"scaling zT by trunc_tau: {etc_kwargs['trunc_tau']}")
            zt = zt * etc_kwargs['trunc_tau']

        total_steps = len(self.scheduler.timesteps)
        use_jepa = self.jepa_config.get('use_jepa', False) and self.f_jepa is not None
        
        if cfg_guidance == 1.0:
            null_prompt_embeds = None

        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL+JEPA' if use_jepa else 'SDXL')
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            # JEPA guidance on zt
            if use_jepa:
                jepa_grad = self.compute_jepa_gradient(
                    zt, step, total_steps, t=t, at=at, at_prev=at_next,
                    uc=null_prompt_embeds, c=prompt_embeds,
                    add_cond_kwargs=add_cond_kwargs, cfg_guidance=cfg_guidance
                )
                if jepa_grad is not None:
                    zt = zt - jepa_grad
                    pbar.set_postfix({'jepa': 'applied'})

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)                    
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # for random noise
            if etc_kwargs['ddim_eta'] > 0.0:
                sigma_t = etc_kwargs['ddim_eta'] * torch.sqrt((1 - at_next) / (1 - at) * (1 - at / at_next))
                noise_rand = torch.randn_like(zt) * sigma_t
                zt = at_next.sqrt() * z0t + (1-at_next-sigma_t**2).sqrt() * noise_pred + noise_rand

            # for deterministic case: eta = 0.0
            else:
                zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = { 'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        return z0t


@register_solver("ddim_jepa_lightning")
class DDIMWithJEPALight(DDIMWithJEPA, SDXLLightning):
    """
    DDIM solver with JEPA guidance for SDXL Lightning
    """
    def __init__(self, **kwargs):
        SDXLLightning.__init__(self, **kwargs)
        self.f_jepa = None
        self.jepa_rng = None
        self.jepa_config = {}

    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        popt_kwargs=None,
                        etc_kwargs=None,
                        **kwargs):
        assert cfg_guidance == 1.0, "CFG should be turned off in the lightning version"
        return DDIMWithJEPA.reverse_process(
            self,
            null_prompt_embeds,
            prompt_embeds,
            cfg_guidance,
            add_cond_kwargs,
            shape,
            callback_fn,
            popt_kwargs=popt_kwargs,
            etc_kwargs=etc_kwargs,
            **kwargs
        )

#############################

if __name__ == "__main__":
    # print all list of solvers
    print(f"Possble solvers: {[x for x in __SOLVER__.keys()]}")
        
