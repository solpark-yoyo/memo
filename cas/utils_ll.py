import os
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
from scipy.integrate import simps

_CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ckpt")
from diffusers import DDIMInverseScheduler, StableDiffusionPipeline, PNDMScheduler, AutoencoderKL
from scipy.signal import savgol_filter
import numpy as np
from typing import *
from torchvision import transforms
from copy import deepcopy


def simpsons_1_3_from_list(mylist):
    return simps(mylist)

def simpsons_3_8_from_list(mylist):
    n = len(mylist) - 1
    if n < 3 or n % 3 != 0:
        raise ValueError("For Simpson's 3/8 rule, length of mylist should be 1 modulo 3 (like 4, 7, 10, ...)")
    integral = mylist[0] + mylist[n]
    for i in range(1, n, 3):
        integral += 3 * (mylist[i] + mylist[i+1])
    for i in range(3, n-2, 3):
        integral += 2 * mylist[i]
    integral *= 3/8
    return integral

def multivariate_gaussian_log_likelihood(x):

    # Calculate the log likelihood for each observation in the batch
    num_samples = x.shape[0]
    x = x.view(num_samples, -1)
    exps = -0.5 * x.pow(2).sum(1)
    const = -0.5 * x.shape[1] * np.log(2 * np.pi)
    log_likelihoods = exps + const

    return log_likelihoods


class CAS_preprocessor(object):
    def __init__(self, model = 'sd15', dtype = torch.float32, device = 'cuda', scheduler = None, approx_num = 1, num_inference_steps = 50, epsilon = 1e-3, rec_num = 1):
        self.pipe, self.vae, self.unet, self.text_encoder = {}, {}, {}, {}
        self.dtype = dtype
        self.device = device
        self.scheduler = scheduler
        # if model == 'sd15': self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype = dtype, safety_checker = None).to(device)
        # if model == 'sd15': self.pipe = StableDiffusionPipeline.from_pretrained("pt-sk/stable-diffusion-1.5", torch_dtype = dtype, safety_checker = None).to(device)
        # if model == 'sd15': self.pipe = StableDiffusionPipeline.from_pretrained("benjamin-paine/stable-diffusion-v1-5", torch_dtype = dtype, safety_checker = None).to(device)
        if model == 'sd15': self.pipe = StableDiffusionPipeline.from_pretrained(os.path.join(_CKPT_DIR, "stable-diffusion-v1-5"), torch_dtype = dtype, safety_checker = None, local_files_only=True).to(device)
        elif model == 'sd20': self.pipe = StableDiffusionPipeline.from_pretrained(os.path.join(_CKPT_DIR, "stable-diffusion-2-base"), torch_dtype = dtype, safety_checker = None, local_files_only=True).to(device)
        # elif model == 'SD2.1': self.pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype = dtype, safety_checker = None).to(device)
        # elif model == 'SD2.1v': self.pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype = dtype, safety_checker = None).to(device)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        # if model == 'sd15' or model == 'sd20':
        #     self.vae = self.pipe.vae
        # elif model == 'sdxl' or model == 'sdxl_lightning':
        #     self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)
        self.vae = self.pipe.vae
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.unet = self.pipe.unet
        self.unet.eval()
        self.unet.requires_grad_(False)
        self.image_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), ])
        self.approx_num = approx_num
        self.num_inference_steps = num_inference_steps
        self.epsilon = epsilon
        self.rec_num = rec_num

    def preprocess(self, image, prompt):
        height = self.unet.config.sample_size * self.pipe.vae_scale_factor
        width = self.unet.config.sample_size * self.pipe.vae_scale_factor
        timesteps = self.scheduler.timesteps[1:]
        with torch.no_grad():
            text_embeddings = {}
            text_embeddings['cond'] = self.pipe._encode_prompt(prompt, self.device, 1, False, "")
            # text_embeddings['uncond'] = self.pipe._encode_prompt("", self.device, 1, False, "")
            image = self.image_transforms(image.resize((height,width)))
            image = image.unsqueeze(0).to(device = self.vae.device, dtype = self.vae.dtype)

            init_latents = self.vae.encode(image).latent_dist.mean
            latents = init_latents * 0.18215
            # latents = {'cond': latents, 'uncond': latents.clone().detach()}
            latents = {'cond': latents}
            # jacobian_trace_list = {key: [torch.zeros(self.approx_num).cuda()] for key in ['total', 'cond', 'uncond']}
            jacobian_trace_list = {key: [torch.zeros(self.approx_num).cuda()] for key in ['total', 'cond']}
            
            for t in tqdm(timesteps):
                rand_eps = torch.randint(low=0, high=2, size=(self.approx_num, *init_latents.shape[1:]), dtype = self.dtype).cuda() * 2 - 1
                grad_fn_eps = {}
                latents_default = deepcopy(latents)
                for rec_iter in range(self.rec_num):
                    for preprocess_type in latents.keys():
                        latent_model_input = self.scheduler.scale_model_input(latents[preprocess_type], t)
                        noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings[preprocess_type].detach()).sample
                        if rec_iter == 0:
                            latents_eps = latents[preprocess_type].clone() + self.epsilon * rand_eps
                            latent_eps_model_input = self.scheduler.scale_model_input(latents_eps, t)
                            noise_pred_eps = self.pipe.unet(latent_eps_model_input, t, encoder_hidden_states = text_embeddings[preprocess_type].expand(self.approx_num, -1, -1)).sample
                        
                            grad_fn_eps[preprocess_type] = (noise_pred_eps - noise_pred) / self.epsilon
                        latents[preprocess_type] = self.scheduler.step(noise_pred, t, latents_default[preprocess_type]).prev_sample

                # for preprocess_type in ['total', 'cond', 'uncond']:
                for preprocess_type in ['total', 'cond']:
                    if preprocess_type == 'total':
                        # tj_sample = torch.sum((grad_fn_eps['cond'] - grad_fn_eps['uncond']) * rand_eps, dim=tuple(range(1, len(rand_eps.shape))))
                        tj_sample = torch.sum((grad_fn_eps['cond']) * rand_eps, dim=tuple(range(1, len(rand_eps.shape))))
                    else:
                        tj_sample = torch.sum((grad_fn_eps[preprocess_type]) * rand_eps, dim=tuple(range(1, len(rand_eps.shape))))
                    jacobian_trace_list[preprocess_type].append(tj_sample)

            # for preprocess_type in ['total', 'cond', 'uncond']:
            for preprocess_type in ['total', 'cond']:
                jacobian_trace_list[preprocess_type] = torch.cat(jacobian_trace_list[preprocess_type]).reshape(-1,self.approx_num)

        res_dict = {'jacobian': {}, 'llhood': {}}
        
        # for preprocess_type in ['cond', 'uncond']:
        for preprocess_type in ['cond']:
            res_dict['jacobian'][preprocess_type] = jacobian_trace_list[preprocess_type].cpu().tolist()
            res_dict['llhood'][preprocess_type] = multivariate_gaussian_log_likelihood(latents[preprocess_type].cpu()).item()
        
        res_dict['jacobian']['total'] = jacobian_trace_list['total'].cpu().tolist()
        # res_dict['llhood']['total'] = res_dict['llhood']['cond'] - res_dict['llhood']['uncond']
        res_dict['llhood']['total'] = res_dict['llhood']['cond']
        res_dict['dim'] = latents['cond'].view(len(latents['cond']), -1).shape[1]
        return res_dict


class CAS_preprocessor_SDXL(object):
    def __init__(self, model = 'sd15', dtype = torch.float32, device = 'cuda', scheduler = None, approx_num = 1, num_inference_steps = 50, epsilon = 1e-3, rec_num = 1):
        self.pipe, self.vae, self.unet, self.text_encoder = {}, {}, {}, {}
        self.dtype = dtype
        self.device = device
        self.scheduler = scheduler
        from diffusers import StableDiffusionXLPipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(os.path.join(_CKPT_DIR, "stable-diffusion-xl-base-1.0"), torch_dtype = dtype, safety_checker = None, local_files_only=True).to(device)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        # if model == 'sd15' or model == 'sd20':
        #     self.vae = self.pipe.vae
        # elif model == 'sdxl' or model == 'sdxl_lightning':
        #     self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)
        self.vae = self.pipe.vae
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.unet = self.pipe.unet
        self.unet.eval()
        self.unet.requires_grad_(False)
        self.image_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]), ])
        self.approx_num = approx_num
        self.num_inference_steps = num_inference_steps
        self.epsilon = epsilon
        self.rec_num = rec_num

        self.tokenizer_1 = self.pipe.tokenizer
        self.tokenizer_2 = self.pipe.tokenizer_2
        self.text_enc_1 = self.pipe.text_encoder
        self.text_enc_2 = self.pipe.text_encoder_2

        # self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # self.default_sample_size = self.unet.config.sample_size

    @torch.no_grad()
    def _text_embed(self, prompt, tokenizer, text_enc, clip_skip):
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

    def preprocess(self, image, prompt):
        height = self.unet.config.sample_size * self.pipe.vae_scale_factor
        width = self.unet.config.sample_size * self.pipe.vae_scale_factor
        timesteps = self.scheduler.timesteps[1:]
        
        if isinstance(prompt, list):
            assert len(prompt) == 1
            prompt = prompt[0]
        assert isinstance(prompt, str)

        prompt1 = ["", prompt] # [null_prompt, prompt]
        prompt2 = ["", prompt] # [null_prompt, prompt]
        clip_skip = None
        # cfg_guidance = 1.0
        # height = self.default_sample_size * self.vae_scale_factor
        # width = self.default_sample_size * self.vae_scale_factor
        target_size = (1024, 1024)
        original_size = (height, width)
        crops_coords_top_left = (0, 0)

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

        # if negative_original_size is not None and negative_target_size is not None:
        #     negative_add_time_ids = self._get_add_time_ids(
        #         negative_original_size,
        #         negative_crops_coords_top_left,
        #         negative_target_size,
        #         dtype=prompt_embeds.dtype,
        #         text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
        #     )
        # else:
        #     negative_add_time_ids = add_time_ids
        # negative_text_embeds = pool_null_embed 

        # if cfg_guidance != 0.0 and cfg_guidance != 1.0:
        #     # do cfg
        #     add_text_embeds = torch.cat([negative_text_embeds, add_text_embeds], dim=0)
        #     add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_cond_kwargs = {
            'text_embeds': add_text_embeds.to(self.device),
            'time_ids': add_time_ids.to(self.device)
        }

        with torch.no_grad():
            text_embeddings = {}
            # text_embeddings['cond'] = self.pipe._encode_prompt(prompt, self.device, 1, False, "")
            text_embeddings['cond'] = prompt_embeds.detach().clone()
            # text_embeddings['uncond'] = self.pipe._encode_prompt("", self.device, 1, False, "")
            image = self.image_transforms(image.resize((height,width)))
            image = image.unsqueeze(0).to(device = self.vae.device, dtype = self.vae.dtype)

            init_latents = self.vae.encode(image).latent_dist.mean
            latents = init_latents * 0.18215
            # latents = {'cond': latents, 'uncond': latents.clone().detach()}
            latents = {'cond': latents}
            # jacobian_trace_list = {key: [torch.zeros(self.approx_num).cuda()] for key in ['total', 'cond', 'uncond']}
            jacobian_trace_list = {key: [torch.zeros(self.approx_num).cuda()] for key in ['total', 'cond']}
            
            for t in tqdm(timesteps):
                rand_eps = torch.randint(low=0, high=2, size=(self.approx_num, *init_latents.shape[1:]), dtype = self.dtype).cuda() * 2 - 1
                grad_fn_eps = {}
                latents_default = deepcopy(latents)
                for rec_iter in range(self.rec_num):
                    for preprocess_type in latents.keys():
                        latent_model_input = self.scheduler.scale_model_input(latents[preprocess_type], t)
                        noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings[preprocess_type].detach(), added_cond_kwargs=add_cond_kwargs).sample
                        if rec_iter == 0:
                            latents_eps = latents[preprocess_type].clone() + self.epsilon * rand_eps
                            latent_eps_model_input = self.scheduler.scale_model_input(latents_eps, t)
                            # add_cond_kwargs_exp = {}
                            # for key, value in add_cond_kwargs.items():
                            #     add_cond_kwargs_exp[key] = value.expand(self.approx_num, -1, -1)
                            #     print("add_cond_kwargs_exp[key].shape", add_cond_kwargs_exp[key].shape)
                            add_cond_kwargs_exp = add_cond_kwargs.copy()
                            # add_cond_kwargs_exp['text_embeds'] = add_cond_kwargs_exp['text_embeds'].expand(self.approx_num, -1, -1)
                            noise_pred_eps = self.pipe.unet(latent_eps_model_input, t, encoder_hidden_states = text_embeddings[preprocess_type].expand(self.approx_num, -1, -1), added_cond_kwargs=add_cond_kwargs_exp).sample
                        
                            grad_fn_eps[preprocess_type] = (noise_pred_eps - noise_pred) / self.epsilon
                        latents[preprocess_type] = self.scheduler.step(noise_pred, t, latents_default[preprocess_type]).prev_sample

                # for preprocess_type in ['total', 'cond', 'uncond']:
                for preprocess_type in ['total', 'cond']:
                    if preprocess_type == 'total':
                        # tj_sample = torch.sum((grad_fn_eps['cond'] - grad_fn_eps['uncond']) * rand_eps, dim=tuple(range(1, len(rand_eps.shape))))
                        tj_sample = torch.sum((grad_fn_eps['cond']) * rand_eps, dim=tuple(range(1, len(rand_eps.shape))))
                    else:
                        tj_sample = torch.sum((grad_fn_eps[preprocess_type]) * rand_eps, dim=tuple(range(1, len(rand_eps.shape))))
                    jacobian_trace_list[preprocess_type].append(tj_sample)

            # for preprocess_type in ['total', 'cond', 'uncond']:
            for preprocess_type in ['total', 'cond']:
                jacobian_trace_list[preprocess_type] = torch.cat(jacobian_trace_list[preprocess_type]).reshape(-1,self.approx_num)

        res_dict = {'jacobian': {}, 'llhood': {}}
        
        # for preprocess_type in ['cond', 'uncond']:
        for preprocess_type in ['cond']:
            res_dict['jacobian'][preprocess_type] = jacobian_trace_list[preprocess_type].cpu().tolist()
            res_dict['llhood'][preprocess_type] = multivariate_gaussian_log_likelihood(latents[preprocess_type].cpu()).item()
        
        res_dict['jacobian']['total'] = jacobian_trace_list['total'].cpu().tolist()
        # res_dict['llhood']['total'] = res_dict['llhood']['cond'] - res_dict['llhood']['uncond']
        res_dict['llhood']['total'] = res_dict['llhood']['cond']
        res_dict['dim'] = latents['cond'].view(len(latents['cond']), -1).shape[1]
        return res_dict


class CAS_integrator(object):
    def __init__(self, num_timesteps, scheduler = None):
        self.set_config(num_timesteps, scheduler = scheduler)
    def get_coef_list(self):
        alphas = np.array([self.scheduler.alphas_cumprod[(1000//self.num_timesteps)*i].item() for i in range(self.num_timesteps)])
        coef_list = (1/2/alphas/(1-alphas)**(0.5))[:-1]*(alphas[1:]-alphas[:-1])
        self.alphas = alphas
        return coef_list
    def set_config(self, num_timesteps, scheduler = None):
        self.num_timesteps = num_timesteps
        if scheduler != None:
            self.scheduler = scheduler
        else:
            scheduler = DDIMInverseScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, compute_likelihood=False, set_alpha_to_zero=False)
            scheduler.set_timesteps(num_timesteps)
            self.scheduler = scheduler
        self.timesteps = scheduler.timesteps
        self.coef_list = self.get_coef_list()
    def score(self, log_likelihood, jacobian_list, data_dim, jacobian_range = [1,-1], method = 'simpsons_3_8'):
        if isinstance(jacobian_list, list):
            jacobian_list = np.array(jacobian_list)
        if len(jacobian_list.shape) == 2:
            jacobian_list = jacobian_list.mean(axis = 1)
        vals = -jacobian_list[:-1]*self.coef_list
        vals = vals[jacobian_range[0]: jacobian_range[1]]
        # print("self.alphas[0]: ", self.alphas[0])
        # print("self.alphas[-1]: ", self.alphas[-1])
        # print("self.alphas.shape: ", self.alphas.shape)
        # import ipdb; ipdb.set_trace()
        consts = 0.5 * np.log(self.alphas[0] / self.alphas[-1]) + np.log(data_dim)
        if method == 'simpsons_1_3':
            return log_likelihood + simps(vals) + consts
        elif method == 'simpsons_3_8':
            return log_likelihood + simpsons_3_8_from_list(vals) + consts