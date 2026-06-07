from typing import Union, Optional
import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift
from torchvision.utils import save_image
import numpy as np

import warnings
from dreamsim import dreamsim
from torchvision import transforms

def draw_img(img: Union[torch.Tensor, np.ndarray],
            save_path:Optional[str]='test.png',
            nrow:Optional[int]=8,
            normalize:Optional[bool]=True):
    if isinstance(img, np.ndarray):
        img = torch.Tensor(img)

    save_image(img, fp=save_path, nrow=nrow, normalize=normalize)

def normalize(img: Union[torch.Tensor, np.ndarray]) \
                        -> Union[torch.Tensor, np.ndarray]:
    
    return (img - img.min())/(img.max()-img.min())
     
def to_np(img: torch.Tensor,
          mode: Optional[str]='NCHW') -> np.ndarray:

    assert mode in ['NCHW', 'NHWC']
    
    if mode == 'NCHW':
        img = img.permute(0,2,3,1) 

    return img.detach().cpu().numpy()

def fft2d(img: torch.Tensor,
          mode: Optional[str]='NCHW') -> torch.Tensor:

    assert mode in ['NCHW', 'NHWC']
    
    if mode == 'NCHW':
        return fftshift(fft2(img))
    elif mode == 'NHWC':
        img = img.permute(0,3,1,2)
        return fftshift(fft2(img))
    else:
        raise NameError    
    

def ifft2d(img: torch.Tensor,
           mode: Optional[str]='NCHW') -> torch.Tensor:

    assert mode in ['NCHW', 'NHWC']
    
    if mode == 'NCHW':
        return ifft2(ifftshift(img))
    elif mode == 'NHWC':
        img = ifft2(ifftshift(img))
        return img.permute(0,2,3,1)
    else:
        raise NameError    

def load_dreamsim(device):
    """load DreamSim model, preprocessing function, and transform function for decoded VAE latent"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dreamsim_base_model, dreamsim_preprocess = dreamsim(pretrained=True, cache_dir="dreamsim_ckpt")
        dreamsim_base_model = dreamsim_base_model.to(device)
        dreamsim_model = lambda x: dreamsim_base_model.embed(x)
        # [0, 1] latent to dreamsim input
        dreamsim_latent_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.functional.InterpolationMode.BICUBIC, antialias=True),
        ])
    return dreamsim_model, dreamsim_preprocess, dreamsim_latent_transform