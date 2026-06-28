#!/usr/bin/env python
# diag_tidx_sweep.py — sweep t_idx, compare adjoint vs TRUE autograd.
# Crucially uses gradient checkpointing to bound memory at larger t_idx.
# This reproduces diag_cache_sweep.py but with the cache/forward made IDENTICAL
# between true and adjoint, and reports per-step diagnostics.
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import torch
from munch import munchify
from latent_diffusion import StableDiffusion

DEVICE = torch.device("cuda:0")
MODEL_KEY = os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5")


def cfg_combine(nuc, nc, cfg=7.5):
    return nuc + cfg * (nc - nuc)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()


def main():
    sd = StableDiffusion(solver_config=munchify({"num_sampling": 50}),
                         model_key=MODEL_KEY, device=DEVICE, seed=42)
    sd.unet.float(); sd.dtype = torch.float32
    sd.unet.enable_gradient_checkpointing()
    timesteps = list(sd.scheduler.timesteps)
    sigma = float(sd.scheduler.init_noise_sigma)
    skip = sd.skip
    cfg = 7.5
    s_idx = int(len(timesteps) * 0.5)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)
    prompt = "An astronaut on the moon"
    uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)
    uc = uc.float(); c = c.float()

    torch.manual_seed(0)
    x_T = torch.randn(1, 4, 64, 64, device=DEVICE, dtype=torch.float32)
    epsilon_ref = x_T.detach().clone()

    print(f"{'t_idx':>5} {'|g_true|':>12} {'|g_adj|':>12} {'cos':>9} {'ratio':>8}")
    for t_idx in [1, 2, 3, 4, 5]:
        # ---- TRUE full autograd (gradient checkpointing keeps memory bounded) ----
        x_Tl = x_T.clone().requires_grad_(True)
        zt = x_Tl * sigma
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t); at_prev = sd.alpha(t - skip)
            nuc, nc = sd.predict_noise(zt, t, uc, c); eps = cfg_combine(nuc, nc, cfg)
            x0h = (zt - (1 - at).sqrt() * eps) / at.sqrt()
            zt = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
            if step_idx == t_idx:
                x_end_true = zt; t_break = t; at_break = at; break
        eps_t = cfg_combine(*sd.predict_noise(x_end_true, t_break, uc, c))
        x0_hat = (x_end_true - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
        x_s = alpha_s.sqrt().to(torch.float32) * x0_hat + (1 - alpha_s).sqrt().to(torch.float32) * x_Tl.detach()
        nuc_s, nc_s = sd.predict_noise(x_s, s_target, uc, c)
        eps_s = cfg_combine(nuc_s, nc_s, cfg)
        loss = ((epsilon_ref - eps_s).reshape(1, -1) ** 2).mean()
        g_true = torch.autograd.grad(loss, x_Tl, retain_graph=True)[0]
        g_xend_true = torch.autograd.grad(loss, x_end_true, retain_graph=True)[0]

        # ---- ADJOINT: build cache, terminal head, then reverse recursion ----
        with torch.no_grad():
            xk = (x_T.detach() * sigma).clone(); cache = [xk.clone()]
            for step_idx, t in enumerate(timesteps):
                at = sd.alpha(t); at_prev = sd.alpha(t - skip)
                nuc, nc = sd.predict_noise(xk, t, uc, c); eps = cfg_combine(nuc, nc, cfg)
                x0h = (xk - (1 - at).sqrt() * eps) / at.sqrt()
                xk = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
                if step_idx == t_idx: break
                if (step_idx + 1) <= t_idx: cache.append(xk.clone())
        # verify cache matches true trajectory
        cache_err = (cache[t_idx] - x_end_true.detach()).abs().max().item()
        with torch.enable_grad():
            x_end = cache[t_idx].detach().clone().requires_grad_(True)
            eps_t2 = cfg_combine(*sd.predict_noise(x_end, t_break, uc, c))
            x0h2 = (x_end - (1 - at_break).sqrt() * eps_t2) / at_break.sqrt()
            x_s2 = alpha_s.sqrt().to(torch.float32) * x0h2 + (1 - alpha_s).sqrt().to(torch.float32) * x_T.detach()
            nuc_s2, nc_s2 = sd.predict_noise(x_s2, s_target, uc, c)
            eps_s2 = cfg_combine(nuc_s2, nc_s2, cfg)
            loss2 = ((epsilon_ref - eps_s2).reshape(1, -1) ** 2).mean()
            g = torch.autograd.grad(loss2, x_end)[0]
        g_term_cos = cos(g, g_xend_true)
        for k in range(t_idx, 0, -1):
            j = k - 1; t_j = timesteps[j]
            a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j - skip)
            A_j = (a_jp1 / a_j).sqrt()
            B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()
            xl = cache[j].detach().clone().requires_grad_(True)
            eps_j = cfg_combine(*sd.predict_noise(xl, t_j, uc, c))
            Jt_g = torch.autograd.grad(eps_j, xl, grad_outputs=g)[0]
            g = A_j * g + B_j * Jt_g
        g_adj = g * sigma
        c = cos(g_adj, g_true)
        ratio = (g_adj.norm() / (g_true.norm() + 1e-12)).item()
        print(f"{t_idx:>5} {g_true.norm().item():>12.4e} {g_adj.norm().item():>12.4e} {c:>9.4f} {ratio:>8.4f}  "
              f"(cache_err={cache_err:.1e}, term_cos={g_term_cos:.4f})")
        del x_Tl, g_true, g_adj, g; torch.cuda.empty_cache()

    sd.unet.half()


if __name__ == "__main__":
    main()
