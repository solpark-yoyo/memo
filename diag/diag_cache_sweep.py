"""
Sweep t_idx in {1,2,3,4}: compare adjoint g_xT vs TRUE full-autograd dL/dx_T
at each chain length. If cosine degrades as t_idx grows, the recursion
accumulates a direction error (cache/linearization issue). If cosine is bad
even at t_idx=1, the defect is in the single-step recursion or terminal head.
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
    sigma = float(sd.scheduler.init_noise_sigma); cfg = 7.5
    s_idx = int(len(timesteps) * 0.5); s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)
    prompt = "An astronaut on the moon"
    uc, c = sd.get_text_embed(null_prompt="", prompt=prompt); uc = uc.float(); c = c.float()

    print(f"{'t_idx':>5} {'|g_true|':>12} {'|g_adj|':>12} {'cos(adj,true)':>15} {'|adj|/|true|':>13}")
    for t_idx in [1, 2, 3, 4]:
        torch.manual_seed(0)
        x_T = torch.randn(1, 4, 64, 64, device=DEVICE, dtype=torch.float32)
        epsilon_ref = x_T.detach().clone()

        # ---- TRUE ----
        x_Tl = x_T.clone().requires_grad_(True)
        zt = x_Tl * sigma
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
            nuc, nc = sd.predict_noise(zt, t, uc, c); eps = cfg_combine(nuc, nc, cfg)
            x0h = (zt - (1 - at).sqrt() * eps) / at.sqrt()
            zt = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
            if step_idx == t_idx:
                x_end_true = zt; t_break = t; at_break = at; break
        eps_t = cfg_combine(*sd.predict_noise(x_end_true, t_break, uc, c))
        x0_hat = (x_end_true - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
        x_s = alpha_s.sqrt().to(torch.float32) * x0_hat + (1 - alpha_s).sqrt().to(torch.float32) * x_Tl.detach()
        eps_s = cfg_combine(*sd.predict_noise(x_s, s_target, uc, c))
        loss = ((epsilon_ref - eps_s).reshape(1, -1) ** 2).mean()
        g_true = torch.autograd.grad(loss, x_Tl)[0]

        # ---- ADJOINT ----
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
            g = torch.autograd.grad(loss2, x_end)[0]
        for k in range(t_idx, 0, -1):
            j = k - 1; t_j = timesteps[j]
            a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j - sd.skip)
            A_j = (a_jp1 / a_j).sqrt(); B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()
            xl = cache[j].detach().clone().requires_grad_(True)
            eps_j = cfg_combine(*sd.predict_noise(xl, t_j, uc, c))
            Jt_g = torch.autograd.grad(eps_j, xl, grad_outputs=g)[0]
            g = A_j * g + B_j * Jt_g
        g_adj = g * sigma

        cos = torch.nn.functional.cosine_similarity(g_adj.flatten().unsqueeze(0), g_true.flatten().unsqueeze(0)).item()
        ratio = (g_adj.norm() / (g_true.norm() + 1e-12)).item()
        print(f"{t_idx:>5} {g_true.norm().item():>12.4e} {g_adj.norm().item():>12.4e} {cos:>15.4f} {ratio:>13.4f}")
        del x_Tl, g_true, g_adj
        torch.cuda.empty_cache()

    print("\nInterpretation:")
    print("  cos~1 at all t_idx => adjoint correct (cache-consistent); inertia = loss-signal issue")
    print("  cos degrades with t_idx => recursion/cache accumulates DIRECTION error => root cause")
    print("  cos<0 even at t_idx=1   => single-step recursion or terminal-head sign/coefficient bug")
    sd.unet.half()

if __name__ == "__main__":
    main()
