"""WGNC projection — White Gaussian Noise Constraint.

Ported from GPER (Hwang & Sung, ICML 2026,
"Gradient Preconditioning for Efficient and Reliable Reward-Guided Generation",
arXiv:2602.08646) — https://github.com/KAIST-Visual-AI-Group/GPER/blob/main/wgnc.py

Gradient preconditioning: project the (reward / memo) gradient onto the white
Gaussian noise feasible set G before the Adam step:

    x <- Adam( x , Proj_G(grad) )

G is defined in the compact spectral domain with blockwise ℓ1 and ℓ2 norm
constraints that match CN(0,1) statistics (||y^(p)||_1 = √(π/2)·B,
||y^(p)||_2^2 = B, block size B). The projection is closed-form, O(N log N) via
FFT. Keeps each gradient update noise-aligned → prevents the latent from drifting
out of the white-Gaussian prior (reward hacking / memo overfitting).
"""
from __future__ import annotations

import math

import torch

Tensor = torch.Tensor
SQRT2 = 2.0 ** 0.5


def f_r_to_c(x: Tensor) -> Tensor:
    f = torch.fft.rfft(x.to(torch.float64), norm="ortho")
    f[0] = (f[0] + 1j * f[-1]) / SQRT2
    return f[:-1]


def f_c_to_r(f: Tensor) -> Tensor:
    f = torch.cat([f[:1].real * SQRT2, f[1:], f[:1].imag * SQRT2], dim=0)
    return torch.fft.irfft(f, norm="ortho")


@torch.no_grad()
def project_wgnc(y: Tensor, block_size: int = 16) -> Tensor:
    """Project a single flat real vector onto the white Gaussian noise feasible set."""
    shape = y.shape
    dtype = y.dtype
    n = y.numel()
    divisor = 2 * block_size
    if n % divisor:
        raise ValueError(
            f"WGNC projection requires the flattened latent dimension to be divisible by {divisor}, "
            f"got {n}. Use a latent size divisible by {divisor} or change block_size."
        )
    y = f_r_to_c(y.reshape(-1))
    eps = 1e-6
    gamma = math.pi / 4.0
    gamma_b = gamma * block_size
    start_j = math.floor(gamma_b)

    y = y.reshape(-1, block_size)
    ay = torch.abs(y)
    max_count = torch.sum(ay == torch.max(ay, dim=1, keepdim=True)[0], dim=1, keepdim=True)
    y = torch.where(start_j < max_count, y + torch.randn_like(y) * eps, y)

    ay = torch.abs(y)
    w = torch.sort(ay, dim=1, descending=True)[0]
    s1 = torch.cumsum(w, dim=1)[:, start_j:block_size]
    s2 = torch.cumsum(w.square(), dim=1)[:, start_j:block_size]
    j = torch.arange(start_j + 1, block_size + 1, device=y.device, dtype=y.real.dtype).view(1, -1)
    lam = (s1 - (gamma_b / (j - gamma_b) * (j * s2 - s1.square())).clamp_min(0).sqrt()) / j

    max_lam = w[:, start_j:block_size]
    min_lam = torch.cat([w[:, start_j + 1:], torch.full_like(w[:, :1], -1e9)], dim=1)
    lam = torch.where((max_lam > lam) & (lam >= min_lam), lam, torch.full_like(lam, -1e9))
    lam = torch.sort(lam, dim=1, descending=True)[0][:, :1]

    clipped = torch.clamp(ay - lam, min=0.0)
    denom = torch.clamp(torch.sum(clipped, dim=1, keepdim=True), min=eps)
    y = (gamma ** 0.5 * block_size) * (clipped / denom) * (y / torch.clamp(ay, min=eps))
    return f_c_to_r(y.reshape(-1)).to(dtype).reshape(shape)


@torch.no_grad()
def project_wgnc_batched(x: Tensor, block_size: int = 16) -> Tensor:
    """Apply WGNC projection per sample, preserving leading batch dim.

    x: (B, C, H, W) latent/grad. Each sample (C*H*W) is projected independently.
    C*H*W must be divisible by 2*block_size (=32 for B=16).
    """
    shape = x.shape
    x_flat = x.reshape(shape[0], -1)
    out = torch.stack([project_wgnc(x_flat[i], block_size) for i in range(shape[0])], dim=0)
    return out.reshape(shape)
