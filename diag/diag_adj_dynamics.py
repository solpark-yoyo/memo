"""
Perspective-4 diagnostic: ACTUAL adjoint recursion signal dynamics.

Loads SD1.5, runs ONE prompt through optimize_xT_adj with instrumentation
that records, at EVERY adjoint fold:
  - ||A_j * g||           (norm-preserving/amplifying branch)
  - ||B_j * Jt_g||        (Jacobian-transpose branch)
  - ratio ||B*Jt_g|| / ||A*g||   (does J branch dominate?)
  - cos(g_after, g_terminal)     (direction preservation vs terminal signal)
  - cos(g_after, g_before)       (per-fold direction change)

This directly answers: does the adjoint preserve the memo-minimize direction,
or rotate it into noise?
"""
import os, sys, torch
SCRIPT_DIR = "/home/geonsoo/Desktop/Datasets/Parksol/memo/ori_memo"
sys.path.insert(0, SCRIPT_DIR)
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed
import run_ini_opti as R

device = torch.device("cuda:0")
set_seed(42)

solver_config = munchify({"num_sampling": 50})
sd = StableDiffusion(solver_config=solver_config,
                     model_key=os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5"),
                     device=device, seed=42)
sd.unet.enable_gradient_checkpointing()

# --- replicate the terminal head + adjoint recursion inline, with full instrumentation ---
cfg = 7.5
uc, c = sd.get_text_embed(null_prompt="", prompt="An astronaut on the moon")
uc = uc.float(); c = c.float()
sd.unet.float(); sd.dtype = torch.float32

timesteps = list(sd.scheduler.timesteps)
init_steps, gap_steps, num_steps = 10, 3, 4
update_indices = [init_steps + i*gap_steps for i in range(num_steps)]
update_indices = [i for i in update_indices if i < len(timesteps)]
s_idx = int(len(timesteps) * 0.5)
s_target = timesteps[s_idx]
alpha_s = sd.alpha(s_target)

x_T_init = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
x_T = x_T_init.clone().requires_grad_(True)
epsilon_ref = x_T_init.detach()

def cfg_combine(nuc, nc):
    return nuc + cfg*(nc - nuc)

# reference trajectory for align loss
x0_orig_refs = {}
with torch.no_grad():
    zt_ref = x_T_init.to(sd.dtype) * sd.scheduler.init_noise_sigma
    for step_idx, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        nuc, nc = sd.predict_noise(zt_ref, t, uc, c)
        eps = cfg_combine(nuc, nc)
        x0h = (zt_ref - (1-at).sqrt()*eps)/at.sqrt()
        zt_ref = at_prev.sqrt()*x0h + (1-at_prev).sqrt()*eps
        if step_idx in update_indices:
            x0_orig_refs[step_idx] = x0h.detach().clone().float()

t_idx = update_indices[0]  # 10
print(f"\n{'='*80}\nINSTRUMENTED adjoint for t_idx={t_idx}  (prompt: 'An astronaut on the moon')\n{'='*80}")

# (1) forward chain, cache latents
x_k = (x_T.detach() * sd.scheduler.init_noise_sigma).clone()
xs = [x_k.clone()]
with torch.no_grad():
    for step_idx, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        nuc, nc = sd.predict_noise(x_k, t, uc, c)
        eps = cfg_combine(nuc, nc)
        x0h = (x_k - (1-at).sqrt()*eps)/at.sqrt()
        x_k = at_prev.sqrt()*x0h + (1-at_prev).sqrt()*eps
        if step_idx == t_idx:
            print(f"  [tweedie t_idx={t_idx}] alpha_t={at.item():.4f} SNR={(at/(1-at)).item():.3f}")
            break
        if (step_idx+1) <= t_idx:
            xs.append(x_k.clone())
x_end_state = xs[t_idx].clone()

# (2) terminal head -> g_terminal
with torch.enable_grad():
    x_end = x_end_state.detach().clone().requires_grad_(True)
    t_break = timesteps[t_idx]
    at_break = sd.alpha(t_break)
    eps_t = cfg_combine(*sd.predict_noise(x_end, t_break, uc, c))
    x0_hat = (x_end - (1-at_break).sqrt()*eps_t)/at_break.sqrt()
    x_s = alpha_s.sqrt().to(sd.dtype)*x0_hat + (1-alpha_s).sqrt().to(sd.dtype)*x_T.detach().to(sd.dtype)
    nuc_s, nc_s = sd.predict_noise(x_s, s_target, uc, c)
    eps_s = cfg_combine(nuc_s, nc_s)
    memo_proxy = (epsilon_ref.to(sd.dtype) - eps_s).reshape(1,-1).pow(2).mean(-1)
    loss_memo = memo_proxy.mean()
    loss_align = ((x0_hat.float() - x0_orig_refs[t_idx]).reshape(1,-1).pow(2).mean(-1).mean())
    loss = loss_memo + 0.1*loss_align
    print(f"  loss_memo={loss_memo.item():.6f}  loss_align={loss_align.item():.6f}")
    g = torch.autograd.grad(loss, x_end, retain_graph=False)[0]
    g_terminal = g.clone()
    g_terminal_norm = g_terminal.flatten().norm().item()
    print(f"  ||g_terminal|| = {g_terminal_norm:.6e}")

# (3) adjoint recursion with instrumentation
print(f"\n{'j':>3} {'||A*g||':>12} {'||B*Jtg||':>12} {'ratio JB/A':>11} {'cos(g,g_term)':>14} {'cos(g,g_prev)':>14}")
print("-"*80)
cos_term_init = torch.nn.functional.cosine_similarity(
    g_terminal.flatten(), g_terminal.flatten(), dim=0).item()
g_prev_for_cos = g_terminal.clone()
for k in range(t_idx, 0, -1):
    j = k - 1
    t_j = timesteps[j]
    a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j - sd.skip)
    A_j = (a_jp1/a_j).sqrt()
    B_j = (1-a_jp1).sqrt() - (a_jp1*(1-a_j)/a_j).sqrt()
    x_j_local = xs[j].detach().clone().requires_grad_(True)
    eps_j = cfg_combine(*sd.predict_noise(x_j_local, t_j, uc, c))
    Jt_g = torch.autograd.grad(eps_j, x_j_local, grad_outputs=g, retain_graph=False)[0]
    A_term = A_j * g
    B_term = B_j * Jt_g
    norm_A = A_term.flatten().norm().item()
    norm_B = B_term.flatten().norm().item()
    g = A_term + B_term
    cos_term = torch.nn.functional.cosine_similarity(
        g.flatten(), g_terminal.flatten(), dim=0).item()
    cos_prev = torch.nn.functional.cosine_similarity(
        g.flatten(), g_prev_for_cos.flatten(), dim=0).item()
    print(f"{j:>3} {norm_A:>12.4e} {norm_B:>12.4e} {norm_B/(norm_A+1e-12):>11.4f} {cos_term:>14.4f} {cos_prev:>14.4f}")
    g_prev_for_cos = g.clone()

g_final = g * sd.scheduler.init_noise_sigma
print(f"\n  ||g_final(x_T)|| = {g_final.flatten().norm().item():.6e}")
cos_final = torch.nn.functional.cosine_similarity(
    g_final.flatten(), g_terminal.flatten(), dim=0).item()
print(f"  cos(g_xT, g_terminal) = {cos_final:.4f}")
print(f"  -> If cos_final ~ 1.0: direction preserved (adjoint faithful).")
print(f"     If cos_final ~ 0 or <0: direction DESTROYED (g_xT points away from memo-min).")
print(f"\n  ALSO: compare g_terminal direction to the ACTUAL autograd baseline.")
# Compute the TRUE g_xT via standard backprop through the same forward chain (optimize_xT path)
# to see if adjoint g_xT aligns with the gold-standard full-backprop g_xT.
x_T2 = x_T_init.clone().requires_grad_(True)
zt = x_T2.to(sd.dtype) * sd.scheduler.init_noise_sigma
with torch.enable_grad():
    for step_idx, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        nuc, nc = sd.predict_noise(zt, t, uc, c)
        eps = cfg_combine(nuc, nc)
        x0h = (zt - (1-at).sqrt()*eps)/at.sqrt()
        zt = at_prev.sqrt()*x0h + (1-at_prev).sqrt()*eps
        if step_idx == t_idx:
            break
    # same head
    t_break = timesteps[t_idx]; at_break = sd.alpha(t_break)
    eps_t2 = cfg_combine(*sd.predict_noise(zt, t_break, uc, c))
    x0_hat2 = (zt - (1-at_break).sqrt()*eps_t2)/at_break.sqrt()
    x_s2 = alpha_s.sqrt().to(sd.dtype)*x0_hat2 + (1-alpha_s).sqrt().to(sd.dtype)*x_T2.detach().to(sd.dtype)
    nuc_s2, nc_s2 = sd.predict_noise(x_s2, s_target, uc, c)
    eps_s2 = cfg_combine(nuc_s2, nc_s2)
    memo2 = (epsilon_ref.to(sd.dtype) - eps_s2).reshape(1,-1).pow(2).mean(-1).mean()
    loss2 = memo2 + 0.1*((x0_hat2.float()-x0_orig_refs[t_idx]).reshape(1,-1).pow(2).mean(-1).mean())
    g_xT_gold = torch.autograd.grad(loss2, x_T2, retain_graph=False)[0]
print(f"\n  ||g_xT_gold (full backprop)|| = {g_xT_gold.flatten().norm().item():.6e}")
cos_gold_adj = torch.nn.functional.cosine_similarity(
    g_final.flatten(), g_xT_gold.flatten(), dim=0).item()
print(f"  cos(adjoint g_xT, gold backprop g_xT) = {cos_gold_adj:.4f}")
print(f"  -> THIS is the decisive test: adjoint is EXACT, so this cos should be ~1.0.")
print(f"     If ~1.0: adjoint recursion is mathematically correct (no adjoint bug).")
print(f"        => root cause of lr-insensitivity is NOT in the recursion, it is UPSTREAM")
print(f"           (terminal head signal: g_terminal itself is prompt-blind).")
print(f"     If <<1: adjoint recursion has a bug (cache/index/coeff error).")
