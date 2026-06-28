#!/usr/bin/env python
"""
TEST 1 ONLY: adjoint g_xT vs gold full-backprop g_xT correctness.
Short chain (t_idx=3) so full-backprop activations fit in 24GB VRAM.
Runs adjoint and gold in SEPARATE functions so each pass's graph is freed.
"""
import os, sys, torch, gc
SCRIPT_DIR = "/home/geonsoo/Desktop/Datasets/Parksol/memo/ori_memo"
sys.path.insert(0, SCRIPT_DIR)
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed

device = torch.device("cuda:0")
NUM_SAMPLING = 50
CKPT = os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5")

set_seed(42)
sd = StableDiffusion(solver_config=munchify({"num_sampling": NUM_SAMPLING}),
                     model_key=CKPT, device=device, seed=42)
sd.unet.float()
sd.dtype = torch.float32
sd.unet.enable_gradient_checkpointing()

cfg = 7.5
timesteps = list(sd.scheduler.timesteps)
s_idx = int(len(timesteps) * 0.5)
s_target = timesteps[s_idx]
alpha_s = sd.alpha(s_target)
SHORT = 2  # chain length

x_T_init = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
epsilon_ref = x_T_init.detach()

# COND-ONLY noise (no CFG => no 2x batch). This halves activation memory.
# The recursion correctness is independent of CFG: J_j = d eps_theta/d x_j,
# and we compare adjoint-VJP to gold-backprop VJP for the SAME forward map.
def predict_noise_condonly(sd, zt, t, c):
    """Single UNet call, cond only (no uncond). Returns eps directly."""
    t_in = t.unsqueeze(0) if len(t.shape) == 0 else t
    t_in = t_in.expand(zt.shape[0])
    return sd.unet(zt, t_in, encoder_hidden_states=c)['sample']

prompt = "An astronaut on the moon"
uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)
uc = uc.float(); c = c.float()

def cfg_combine_unused(nuc, noc):
    return nuc + cfg * (noc - nuc)

# ======================================================================
# PASS A: gold full-backprop g_xT (cond-only, grad through chain + head)
# ======================================================================
print(f"[PASS A] gold full-backprop (cond-only), chain={SHORT}")
torch.cuda.empty_cache()
x_T_g = x_T_init.clone().requires_grad_(True)
zt = x_T_g.to(sd.dtype) * sd.scheduler.init_noise_sigma
with torch.enable_grad():
    for step_idx, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        eps = predict_noise_condonly(sd, zt, t, c)
        x0h = (zt - (1 - at).sqrt() * eps) / at.sqrt()
        zt = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
        if step_idx == SHORT:
            break
    # reference x0 for align
    with torch.no_grad():
        x_end_val = zt.detach().clone()
        at_b = sd.alpha(timesteps[SHORT])
        eps_r = predict_noise_condonly(sd, x_end_val, timesteps[SHORT], c)
        x0_ref = (x_end_val - (1 - at_b).sqrt() * eps_r) / at_b.sqrt()
    # head on the grad-flowing state zt
    t_break = timesteps[SHORT]; at_break = sd.alpha(t_break)
    eps_t = predict_noise_condonly(sd, zt, t_break, c)
    x0_hat = (zt - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
    x_s = alpha_s.sqrt().to(sd.dtype) * x0_hat + (1 - alpha_s).sqrt().to(sd.dtype) * x_T_g.detach().to(sd.dtype)
    eps_s = predict_noise_condonly(sd, x_s, s_target, c)
    memo = (epsilon_ref.to(sd.dtype) - eps_s).reshape(1, -1).pow(2).mean(-1).mean()
    loss_g = memo + 0.1 * ((x0_hat.float() - x0_ref).reshape(1, -1).pow(2).mean(-1).mean())
    g_xT_gold = torch.autograd.grad(loss_g, x_T_g, retain_graph=False)[0].detach()
print(f"  ||g_xT_gold|| = {g_xT_gold.norm().item():.6e}  loss={loss_g.item():.6f}")
del x_T_g, zt, x0_hat, x_s, eps_s, eps_t, x_end_val, x0_ref, eps_r
gc.collect(); torch.cuda.empty_cache()

# ======================================================================
# PASS B: adjoint path. Forward no_grad (cache latents), then terminal head
# + adjoint recursion. Compare to gold.
# ======================================================================
print(f"[PASS B] adjoint (cond-only), chain={SHORT}")
torch.cuda.empty_cache()
x_k = (x_T_init * sd.scheduler.init_noise_sigma).clone()
xs = [x_k.clone()]
with torch.no_grad():
    for step_idx, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        eps = predict_noise_condonly(sd, x_k, t, c)
        x0h = (x_k - (1 - at).sqrt() * eps) / at.sqrt()
        x_k = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
        if step_idx == SHORT:
            break
        if (step_idx + 1) <= SHORT:
            xs.append(x_k.clone())
x_end_state = xs[SHORT].clone()
with torch.no_grad():
    at_b = sd.alpha(timesteps[SHORT])
    eps_r = predict_noise_condonly(sd, x_end_state, timesteps[SHORT], c)
    x0_ref = (x_end_state - (1 - at_b).sqrt() * eps_r) / at_b.sqrt()
with torch.enable_grad():
    xe = x_end_state.detach().clone().requires_grad_(True)
    t_break = timesteps[SHORT]; at_break = sd.alpha(t_break)
    eps_t = predict_noise_condonly(sd, xe, t_break, c)
    x0_hat = (xe - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
    x_s = alpha_s.sqrt().to(sd.dtype) * x0_hat + (1 - alpha_s).sqrt().to(sd.dtype) * x_T_init.detach().to(sd.dtype)
    eps_s = predict_noise_condonly(sd, x_s, s_target, c)
    memo = (epsilon_ref.to(sd.dtype) - eps_s).reshape(1, -1).pow(2).mean(-1).mean()
    loss_a = memo + 0.1 * ((x0_hat.float() - x0_ref).reshape(1, -1).pow(2).mean(-1).mean())
    g_terminal = torch.autograd.grad(loss_a, xe, retain_graph=False)[0].detach()
# adjoint recursion
g = g_terminal.clone()
for k in range(SHORT, 0, -1):
    j = k - 1; t_j = timesteps[j]
    a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j - sd.skip)
    A_j = (a_jp1 / a_j).sqrt()
    B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()
    xj = xs[j].detach().clone().requires_grad_(True)
    epsj = predict_noise_condonly(sd, xj, t_j, c)
    Jtg = torch.autograd.grad(epsj, xj, grad_outputs=g, retain_graph=False)[0]
    g = A_j * g + B_j * Jtg
    del xj, epsj, Jtg
g_xT_adj = (g * sd.scheduler.init_noise_sigma).detach()
print(f"  ||g_terminal|| = {g_terminal.norm().item():.6e}")
print(f"  ||g_xT_adj||   = {g_xT_adj.norm().item():.6e}  loss={loss_a.item():.6f}")

cos = torch.nn.functional.cosine_similarity(g_xT_adj.flatten(), g_xT_gold.flatten(), dim=0).item()
rel = ((g_xT_adj - g_xT_gold).norm() / g_xT_gold.norm()).item()
print(f"\n[RESULT] cos(adj, gold) = {cos:.6f}")
print(f"[RESULT] rel_err = {rel:.6e}")
print(f"  cos ~ 1.0 => recursion is EXACT (no recursion bug).")
print(f"  cos << 1  => recursion has a bug -> recursion is a root cause.")
