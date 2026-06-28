"""
Refined C3: compare adjoint g_xT vs TRUE full-autograd dL/dx_T (the ground truth,
no finite-difference noise). This isolates whether the adjoint is WRONG or whether
the memo-loss signal itself is just tiny/noisy.

Also measure: |dL/dx_T| relative to |dL/dx_{t_idx}|, and the prompt-discriminability
of eps_s perturbations (does moving x_T actually move eps_s?).
"""
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import torch
from munch import munchify
from latent_diffusion import StableDiffusion

DEVICE = torch.device("cuda:0")
MODEL_KEY = os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5")

def cfg_combine(nuc, nc, cfg=7.5): return nuc + cfg * (nc - nuc)

def main():
    sd = StableDiffusion(solver_config=munchify({"num_sampling": 50}),
                         model_key=MODEL_KEY, device=DEVICE, seed=42)
    sd.unet.float(); sd.dtype = torch.float32
    sd.unet.enable_gradient_checkpointing()
    timesteps = list(sd.scheduler.timesteps)
    sigma = float(sd.scheduler.init_noise_sigma)
    cfg = 7.5
    t_idx = 4   # small chain so full-autograd ground truth fits in VRAM
    s_idx = int(len(timesteps) * 0.5); s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)

    prompt = "An astronaut on the moon"
    uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)
    uc = uc.float(); c = c.float()
    torch.manual_seed(0)
    x_T = torch.randn(1, 4, 64, 64, device=DEVICE, dtype=torch.float32)
    epsilon_ref = x_T.detach().clone()

    # ---- TRUE gradient: full autograd through the ENTIRE chain (x_T -> x_tidx -> loss) ----
    # This is the ground truth dL/dx_T. No FD, no adjoint recursion.
    x_T_leaf = x_T.clone().requires_grad_(True)
    zt = x_T_leaf * sigma
    for step_idx, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        nuc, nc = sd.predict_noise(zt, t, uc, c)
        eps = cfg_combine(nuc, nc, cfg)
        x0h = (zt - (1 - at).sqrt() * eps) / at.sqrt()
        zt = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
        if step_idx == t_idx:
            x_end_true = zt
            t_break = t; at_break = at
            break
    x_end_true = x_end_true  # already grad-connected to x_T_leaf
    eps_t = cfg_combine(*sd.predict_noise(x_end_true, t_break, uc, c))
    x0_hat = (x_end_true - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
    x_s = alpha_s.sqrt().to(torch.float32) * x0_hat \
          + (1 - alpha_s).sqrt().to(torch.float32) * x_T_leaf.detach()
    nuc_s, nc_s = sd.predict_noise(x_s, s_target, uc, c)
    eps_s = cfg_combine(nuc_s, nc_s, cfg)
    loss = ((epsilon_ref - eps_s).reshape(1, -1) ** 2).mean()
    g_xend_true = torch.autograd.grad(loss, x_end_true, retain_graph=True)[0]
    g_true = torch.autograd.grad(loss, x_T_leaf)[0]
    print(f"[TRUE] loss={loss.item():.6f}")
    print(f"[TRUE] |dL/dx_T| (full autograd, ground truth) = {g_true.norm().item():.6e}")
    print(f"[TRUE] |dL/dx_tidx| (terminal)                = {g_xend_true.norm().item():.6e}")
    print(f"[TRUE] ratio |dL/dx_T|/|dL/dx_tidx|           = {(g_true.norm()/g_xend_true.norm()).item():.6e}")
    print(f"  -> this ratio tells how much the chain attenuates/rotates the gradient.")

    # ---- ADJOINT g_xT (replicate recursion) ----
    with torch.no_grad():
        xk = (x_T.detach() * sigma).clone(); cache = [xk.clone()]
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
            nuc, nc = sd.predict_noise(xk, t, uc, c); eps = cfg_combine(nuc, nc, cfg)
            x0h = (xk - (1 - at).sqrt() * eps) / at.sqrt()
            xk = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
            if step_idx == t_idx: break
            if (step_idx + 1) <= t_idx: cache.append(xk.clone())
    with torch.enable_grad():
        x_end = cache[t_idx].detach().clone().requires_grad_(True)
        eps_t2 = cfg_combine(*sd.predict_noise(x_end, t_break, uc, c))
        x0h2 = (x_end - (1 - at_break).sqrt() * eps_t2) / at_break.sqrt()
        x_s2 = alpha_s.sqrt().to(torch.float32) * x0h2 + (1 - alpha_s).sqrt().to(torch.float32) * x_T.detach()
        eps_s2 = cfg_combine(*sd.predict_noise(x_s2, s_target, uc, c))
        loss2 = ((epsilon_ref - eps_s2).reshape(1, -1) ** 2).mean()
        g_term = torch.autograd.grad(loss2, x_end)[0]
    g = g_term
    Aprod = 1.0
    for k in range(t_idx, 0, -1):
        j = k - 1; t_j = timesteps[j]
        a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j - sd.skip)
        A_j = (a_jp1 / a_j).sqrt(); B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()
        Aprod *= float(A_j)
        xl = cache[j].detach().clone().requires_grad_(True)
        eps_j = cfg_combine(*sd.predict_noise(xl, t_j, uc, c))
        Jt_g = torch.autograd.grad(eps_j, xl, grad_outputs=g)[0]
        g = A_j * g + B_j * Jt_g
    g_adj = g * sigma

    cos = torch.nn.functional.cosine_similarity(g_adj.flatten().unsqueeze(0), g_true.flatten().unsqueeze(0)).item()
    print(f"\n[ADJOINT] |g_xT_adj|={g_adj.norm().item():.6e}  prod(A_j)={Aprod:.4f}")
    print(f"[CMP] cosine(adj, TRUE) = {cos:.6f}")
    print(f"[CMP] |adj|/|true|      = {(g_adj.norm()/ (g_true.norm()+1e-12)).item():.6f}")
    print(f"  -> cosine~1 & ratio~1 => adjoint EXACT (cache/recursion correct; inertia is loss-signal issue)")
    print(f"  -> cosine<<1          => adjoint WRONG relative to true chain gradient")
    print(f"  -> ratio>>1 with cos~1=> adjoint OVER-amplifies (Adam still moves, but maybe wrong scale)")

    # ---- prompt-discriminability: does x_T perturbation actually move eps_s? ----
    print("\n[BLIND] sensitivity of eps_s to x_T (true chain):")
    print(f"  |dL/dx_T|={g_true.norm().item():.4e}  vs |dL/dx_tidx|={g_xend_true.norm().item():.4e}")
    print(f"  If |dL/dx_T| << |dL/dx_tidx|: signal IS reaching x_T but ATTENUATED by chain.")
    print(f"  If |dL/dx_tidx| itself tiny: terminal memo-loss is BLIND (eps_s insensitive to x_tidx).")

    sd.unet.half()

if __name__ == "__main__":
    main()
