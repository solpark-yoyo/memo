#!/usr/bin/env python
"""
CONTROLLED correctness test: gold full-backprop vs adjoint, where BOTH passes
share the IDENTICAL forward latent sequence (single grad-enabled forward chain).

This isolates whether the adjoint recursion's VJP fold exactly matches the
autograd-computed product of Jacobians, removing any forward-path divergence.

Method:
  1. ONE grad-enabled forward chain produces zt (grad-flowing) AND we snapshot
     detached copies xs[] at each step (the adjoint's cached latents).
  2. Terminal head on the grad-flowing zt produces loss.
  3. GOLD: torch.autograd.grad(loss, x_T_leaf) -- autograd walks the whole graph.
  4. ADJOINT: take detached xs[] snapshots, rebuild terminal head on xs[SHORT]
     (fresh leaf), get g_terminal, then recurse with VJP folds.
  5. Compare cos(g_adj, g_gold).

If cos ~ 1.0 here: adjoint recursion is EXACT; the chain=1/2 cos drop in the
prior script was from forward-path fp divergence (different random/ungrad path),
NOT a recursion bug.
If cos << 1.0: there is a genuine adjoint bug (coeff/index/Jacobian mismatch).

cond-only, chain=2 (fits VRAM).
"""
import os, sys, torch, gc
SCRIPT_DIR = "/home/geonsoo/Desktop/Datasets/Parksol/memo/ori_memo"
sys.path.insert(0, SCRIPT_DIR)
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed

device = torch.device("cuda:0")
CKPT = os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5")
set_seed(42)
sd = StableDiffusion(solver_config=munchify({"num_sampling": 50}),
                     model_key=CKPT, device=device, seed=42)
sd.unet.float(); sd.dtype = torch.float32
# NO gradient checkpointing: ensures grad-forward and no_grad-forward produce
# BIT-IDENTICAL latent sequences, so snapshot diff == 0 and the adjoint-vs-gold
# comparison is clean (isolates the recursion math from fp recompute noise).
# sd.unet.enable_gradient_checkpointing()  # disabled

cfg = 7.5
timesteps = list(sd.scheduler.timesteps)
s_target = timesteps[int(len(timesteps)*0.5)]
alpha_s = sd.alpha(s_target)
SHORT = 1  # no-checkpointing chain=1 fits VRAM cleanly

x_T_init = torch.randn(1,4,64,64, device=device, dtype=torch.float32)
epsilon_ref = x_T_init.detach()

def noise_c(zt, t, c):
    t_in = t.unsqueeze(0) if len(t.shape)==0 else t
    t_in = t_in.expand(zt.shape[0])
    return sd.unet(zt, t_in, encoder_hidden_states=c)['sample']

uc, c = sd.get_text_embed(null_prompt="", prompt="An astronaut on the moon")
c = c.float()

# ======================================================================
# ONE grad-enabled forward chain; snapshot detached latents for adjoint.
# ======================================================================
x_T_leaf = x_T_init.clone().requires_grad_(True)
zt = x_T_leaf.to(sd.dtype) * sd.scheduler.init_noise_sigma
xs_snap = [zt.detach().clone()]   # xs[0] = scaled x_T
with torch.enable_grad():
    for step_idx, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        eps = noise_c(zt, t, c)
        x0h = (zt - (1-at).sqrt()*eps)/at.sqrt()
        zt = at_prev.sqrt()*x0h + (1-at_prev).sqrt()*eps
        if step_idx == SHORT:
            break
        if (step_idx+1) <= SHORT:
            xs_snap.append(zt.detach().clone())

# verify the snapshot matches the grad-flowing state values
x_end_gradflow = zt  # state at step SHORT (grad-flowing)
x_end_snap = xs_snap[SHORT]
snap_match = (x_end_gradflow.detach() - x_end_snap).abs().max().item()
print(f"[check] snapshot vs gradflow state max-abs diff = {snap_match:.3e}  (should be ~0)")

# reference x0 for align (no grad)
with torch.no_grad():
    at_b = sd.alpha(timesteps[SHORT])
    eps_r = noise_c(x_end_snap, timesteps[SHORT], c)
    x0_ref = (x_end_snap - (1-at_b).sqrt()*eps_r)/at_b.sqrt()

# ======================================================================
# GOLD: head on grad-flowing state, then autograd to x_T_leaf
# ======================================================================
with torch.enable_grad():
    t_break = timesteps[SHORT]; at_break = sd.alpha(t_break)
    eps_t = noise_c(x_end_gradflow, t_break, c)
    x0_hat = (x_end_gradflow - (1-at_break).sqrt()*eps_t)/at_break.sqrt()
    x_s = alpha_s.sqrt().to(sd.dtype)*x0_hat + (1-alpha_s).sqrt().to(sd.dtype)*x_T_leaf.detach().to(sd.dtype)
    eps_s = noise_c(x_s, s_target, c)
    memo = (epsilon_ref.to(sd.dtype) - eps_s).reshape(1,-1).pow(2).mean(-1).mean()
    loss_g = memo + 0.1*((x0_hat.float()-x0_ref).reshape(1,-1).pow(2).mean(-1).mean())
    g_gold = torch.autograd.grad(loss_g, x_T_leaf, retain_graph=False)[0].detach()
print(f"[GOLD ] ||g||={g_gold.norm().item():.6e}  loss={loss_g.item():.6f}")
del eps_t, x0_hat, x_s, eps_s, memo, loss_g, zt, x_end_gradflow
gc.collect(); torch.cuda.empty_cache()

# ======================================================================
# ADJOINT: rebuild head on detached snapshot xs[SHORT], recurse with VJP
# ======================================================================
with torch.enable_grad():
    xe = xs_snap[SHORT].detach().clone().requires_grad_(True)
    t_break = timesteps[SHORT]; at_break = sd.alpha(t_break)
    eps_t = noise_c(xe, t_break, c)
    x0_hat = (xe - (1-at_break).sqrt()*eps_t)/at_break.sqrt()
    x_s = alpha_s.sqrt().to(sd.dtype)*x0_hat + (1-alpha_s).sqrt().to(sd.dtype)*x_T_init.detach().to(sd.dtype)
    eps_s = noise_c(x_s, s_target, c)
    memo = (epsilon_ref.to(sd.dtype)-eps_s).reshape(1,-1).pow(2).mean(-1).mean()
    loss_a = memo + 0.1*((x0_hat.float()-x0_ref).reshape(1,-1).pow(2).mean(-1).mean())
    g_term = torch.autograd.grad(loss_a, xe, retain_graph=False)[0].detach()
    loss_a_val = loss_a.item()
print(f"[ADJ  ] ||g_terminal||={g_term.norm().item():.6e}  loss={loss_a_val:.6f}")

g = g_term.clone()
for k in range(SHORT,0,-1):
    j = k-1; t_j = timesteps[j]
    a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j-sd.skip)
    A_j = (a_jp1/a_j).sqrt()
    B_j = (1-a_jp1).sqrt()-(a_jp1*(1-a_j)/a_j).sqrt()
    xj = xs_snap[j].detach().clone().requires_grad_(True)
    epsj = noise_c(xj, t_j, c)
    Jtg = torch.autograd.grad(epsj, xj, grad_outputs=g, retain_graph=False)[0]
    g = A_j*g + B_j*Jtg
    del xj, epsj, Jtg
g_adj = (g*sd.scheduler.init_noise_sigma).detach()

cos = torch.nn.functional.cosine_similarity(g_adj.flatten(), g_gold.flatten(), dim=0).item()
rel = ((g_adj-g_gold).norm()/g_gold.norm()).item()
norm_ratio = g_adj.norm().item()/g_gold.norm().item()
print(f"\n[RESULT] chain={SHORT}")
print(f"  cos(g_adj, g_gold)   = {cos:.6f}")
print(f"  rel_err              = {rel:.6e}")
print(f"  ||g_adj||/||g_gold|| = {norm_ratio:.4f}")
print(f"\n  cos ~ 1.0 => adjoint recursion EXACT (any prior cos<1 was forward-path fp noise).")
print(f"  cos << 1.0 => genuine adjoint bug (coeff/index/Jacobian).")
