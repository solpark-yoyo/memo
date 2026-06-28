#!/usr/bin/env python
# diag_skeptic_verify.py
# SKEPTIC'S DECISIVE TEST of the finite-diff-gradcheck diagnosis.
#
# The diagnosis claims:
#   ROOT CAUSE = adjoint's "no_grad cache + detach re-leaf" design makes the
#   VJP operating-point DIFFER from the true differentiable forward, because
#   fp32 CUDA UNet gives slightly different outputs under no_grad vs grad.
#   => adjoint gradient is biased/non-exact => x_T inert to lr.
#
# This script runs THREE decisive tests that can REFUTE that claim:
#
# TEST A (operating-point shift): Does a real fp32 SD1.5 UNet actually give
#   DIFFERENT latents under no_grad vs grad forward, on the SAME chain?
#   -> If diff == 0 (bit-identical), the diagnosis's mechanism is FALSE and
#      the real-UNet mismatch measured earlier must come from ELSEWHERE.
#
# TEST B (controlled forward-shared correctness): Run ONE grad-enabled forward
#   chain. Snapshot detached copies xs[]. Then:
#     GOLD  = autograd.grad(loss, x_T_leaf) over the WHOLE grad graph.
#     ADJOINT = rebuild terminal head on detached xs[t_idx] + VJP recursion on xs[j].
#   If cos(adj, gold) ~ 1.0 here, the adjoint MATH is exact AND the operating-
#   point is IDENTICAL (both use the same forward). => the ONLY remaining
#   source of mismatch in the real (cache) path is the no_grad-vs-grad forward
#   divergence (TEST A). If TEST A says diff==0 but the real cache path still
#   mismatches, the cause is NOT operating-point shift.
#
# TEST C (terminal-head-only, cache vs grad): Isolate the 2-UNet terminal head.
#   Compare g_terminal computed from (i) no_grad-cached x_end vs (ii) grad-
#   flowing x_end from the same forward. If cos~1.0, terminal head has no
#   operating-point problem; the earlier rel_err 0.91 is a measurement artifact.
import os, sys, gc, torch
SCRIPT_DIR = "/home/geonsoo/Desktop/Datasets/Parksol/memo/ori_memo"
sys.path.insert(0, SCRIPT_DIR)
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed

device = torch.device("cuda:0")
CKPT = os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5")
CFG = 7.5


def cfg_combine(nuc, nc, cfg=CFG):
    return nuc + cfg * (nc - nuc)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def re(a, b):
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


set_seed(42)
sd = StableDiffusion(solver_config=munchify({"num_sampling": 50}),
                     model_key=CKPT, device=device, seed=42)
sd.unet.float()
sd.dtype = torch.float32
# Match the production path: gradient checkpointing ON
sd.unet.enable_gradient_checkpointing()

timesteps = list(sd.scheduler.timesteps)
sigma = float(sd.scheduler.init_noise_sigma)
skip = sd.skip
s_idx = int(len(timesteps) * 0.5)
s_target = timesteps[s_idx]
alpha_s = sd.alpha(s_target)

uc, c = sd.get_text_embed(null_prompt="", prompt="An astronaut on the moon")
uc = uc.float(); c = c.float()

x_T_init = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
eps_ref = x_T_init.detach()

# =====================================================================
# TEST A: no_grad forward chain vs grad forward chain — SAME x_T.
# Measure latent divergence at each step. Detach the grad chain so the
# only difference is the grad context of predict_noise.
# =====================================================================
print("=" * 78)
print("TEST A: no_grad forward vs grad forward (operating-point divergence)")
print("=" * 78)
T_IDX = 10
# --- no_grad chain (mirrors production adjoint forward) ---
with torch.no_grad():
    xk_nograd = (x_T_init * sigma).clone()
    xs_nograd = [xk_nograd.clone()]
    for step_idx, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - skip)
        nuc, noc = sd.predict_noise(xk_nograd, t, uc, c)
        eps = cfg_combine(nuc, noc)
        x0h = (xk_nograd - (1 - at).sqrt() * eps) / at.sqrt()
        xk_nograd = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
        if step_idx == T_IDX:
            break
        if (step_idx + 1) <= T_IDX:
            xs_nograd.append(xk_nograd.clone())

# --- grad chain, same x_T, but detached accumulation (values only) ---
xk_grad = (x_T_init * sigma).clone()
xs_grad = []
for step_idx, t in enumerate(timesteps):
    with torch.enable_grad():
        xk_in = xk_grad.detach().clone().requires_grad_(True)
        at = sd.alpha(t); at_prev = sd.alpha(t - skip)
        nuc, noc = sd.predict_noise(xk_in, t, uc, c)
        eps = cfg_combine(nuc, noc)
        x0h = (xk_in - (1 - at).sqrt() * eps) / at.sqrt()
        xk_next = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
    xs_grad.append(xk_in.detach().clone())  # input latent
    xk_grad = xk_next.detach().clone()      # carry values only (like no_grad)
    if step_idx == T_IDX:
        break

print(f"  compared steps 0..{T_IDX}, latent 1x4x64x64 fp32, grad-ckpt ON")
max_diff = 0.0
for j in range(min(len(xs_nograd), len(xs_grad))):
    d = (xs_nograd[j] - xs_grad[j]).abs().max().item()
    max_diff = max(max_diff, d)
    if j < 4 or j >= T_IDX - 1:
        rel = re(xs_nograd[j], xs_grad[j])
        print(f"    step {j:2d}: max|diff|={d:.3e}  rel_err={rel:.3e}  "
              f"cos={cos(xs_nograd[j], xs_grad[j]):.6f}")
print(f"  >>> MAX abs diff over all steps = {max_diff:.3e}")
if max_diff == 0.0:
    print("  >>> VERDICT A: no_grad and grad forward are BIT-IDENTICAL.")
    print("      => the diagnosis's 'operating-point shift' mechanism is REFUTED.")
    print("      => any real-path mismatch must come from a DIFFERENT source.")
else:
    print("  >>> VERDICT A: no_grad and grad forward DIVERGE => operating-point shift REAL.")
del xs_nograd, xs_grad, xk_nograd, xk_grad
gc.collect(); torch.cuda.empty_cache()

# =====================================================================
# TEST B: controlled correctness — ONE grad forward, gold vs adjoint.
# Both use the SAME forward latents, so operating point is identical.
# If cos(adj,gold)~1.0 here, adjoint math is exact; the real (cache) path's
# only divergence source is TEST A.
# =====================================================================
print("\n" + "=" * 78)
print("TEST B: controlled correctness (forward-shared, short chain t_idx=3)")
print("=" * 78)
SHORT = 3
x_T_g = x_T_init.clone().requires_grad_(True)
zt = x_T_g * sigma
xs_shared = [zt.detach().clone()]
for step_idx, t in enumerate(timesteps):
    with torch.enable_grad():
        at = sd.alpha(t); at_prev = sd.alpha(t - skip)
        nuc, noc = sd.predict_noise(zt, t, uc, c)
        eps = cfg_combine(nuc, noc)
        x0h = (zt - (1 - at).sqrt() * eps) / at.sqrt()
        zt = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
    if step_idx == SHORT:
        break
    if (step_idx + 1) <= SHORT:
        xs_shared.append(zt.detach().clone())
# terminal head on grad-flowing zt (= x_{SHORT})
with torch.enable_grad():
    t_break = timesteps[SHORT]; at_break = sd.alpha(t_break)
    eps_t = cfg_combine(*sd.predict_noise(zt, t_break, uc, c))
    x0_hat = (zt - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
    x_s = alpha_s.sqrt() * x0_hat + (1 - alpha_s).sqrt() * x_T_init.detach()
    nuc_s, noc_s = sd.predict_noise(x_s, s_target, uc, c)
    eps_s = cfg_combine(nuc_s, noc_s)
    memo = (eps_ref - eps_s).reshape(1, -1).pow(2).mean(-1).mean()
    loss = memo  # drop align for cleanliness; memo is the discriminating term
    g_xT_gold = torch.autograd.grad(loss, x_T_g, retain_graph=False)[0].detach()
# adjoint on the SHARED cached latents (operating point identical to gold)
with torch.enable_grad():
    xe = xs_shared[SHORT].detach().clone().requires_grad_(True)
    eps_t2 = cfg_combine(*sd.predict_noise(xe, t_break, uc, c))
    x0h2 = (xe - (1 - at_break).sqrt() * eps_t2) / at_break.sqrt()
    x_s2 = alpha_s.sqrt() * x0h2 + (1 - alpha_s).sqrt() * x_T_init.detach()
    nuc_s2, noc_s2 = sd.predict_noise(x_s2, s_target, uc, c)
    eps_s2 = cfg_combine(nuc_s2, noc_s2)
    memo2 = (eps_ref - eps_s2).reshape(1, -1).pow(2).mean(-1).mean()
    g_terminal_cached = torch.autograd.grad(memo2, xe, retain_graph=False)[0].detach()
g = g_terminal_cached.clone()
for k in range(SHORT, 0, -1):
    j = k - 1; t_j = timesteps[j]
    a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j - skip)
    A_j = (a_jp1 / a_j).sqrt()
    B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()
    with torch.enable_grad():
        xj = xs_shared[j].detach().clone().requires_grad_(True)
        epsj = cfg_combine(*sd.predict_noise(xj, t_j, uc, c))
        Jtg = torch.autograd.grad(epsj, xj, grad_outputs=g, retain_graph=False)[0]
    g = A_j * g + B_j * Jtg
g_xT_adj = (g * sigma).detach()
print(f"  ||g_xT gold||  = {g_xT_gold.norm().item():.4e}")
print(f"  ||g_xT adj||   = {g_xT_adj.norm().item():.4e}")
print(f"  cos(adj,gold)  = {cos(g_xT_adj, g_xT_gold):.6f}")
print(f"  rel_err        = {re(g_xT_adj, g_xT_gold):.4e}")
if cos(g_xT_adj, g_xT_gold) > 0.999:
    print("  >>> VERDICT B: adjoint is EXACT when forward is shared.")
    print("      => recursion math correct; operating-point is the ONLY possible cause.")
else:
    print("  >>> VERDICT B: adjoint MISMATCHES even with shared forward.")
    print("      => there is a recursion/index/coeff BUG independent of cache.")
del xs_shared, zt, xe, x_T_g, g_xT_gold, g_xT_adj, g, g_terminal_cached
gc.collect(); torch.cuda.empty_cache()

# =====================================================================
# TEST C: terminal-head-only, cached vs grad-flowing (same forward).
# =====================================================================
print("\n" + "=" * 78)
print("TEST C: terminal-head-only, cached x_end vs grad-flowing x_end")
print("=" * 78)
# build a 1-step grad forward so x_1 is grad-flowing AND we cache x_0
x_T_g2 = x_T_init.clone().requires_grad_(True)
zt2 = x_T_g2 * sigma
xs_c = [zt2.detach().clone()]
t0 = timesteps[0]; at0 = sd.alpha(t0); at0p = sd.alpha(t0 - skip)
with torch.enable_grad():
    nuc, noc = sd.predict_noise(zt2, t0, uc, c)
    eps = cfg_combine(nuc, noc)
    x0h = (zt2 - (1 - at0).sqrt() * eps) / at0.sqrt()
    zt2 = at0p.sqrt() * x0h + (1 - at0).sqrt() * eps
xs_c.append(zt2.detach().clone())
# terminal head on grad-flowing zt2 (=x_1)
t_b = timesteps[1]; at_b = sd.alpha(t_b)
with torch.enable_grad():
    eps_t = cfg_combine(*sd.predict_noise(zt2, t_b, uc, c))
    x0h_g = (zt2 - (1 - at_b).sqrt() * eps_t) / at_b.sqrt()
    x_s_g = alpha_s.sqrt() * x0h_g + (1 - alpha_s).sqrt() * x_T_init.detach()
    nuc_s, noc_s = sd.predict_noise(x_s_g, s_target, uc, c)
    eps_s_g = cfg_combine(nuc_s, noc_s)
    memo_g = (eps_ref - eps_s_g).reshape(1, -1).pow(2).mean(-1).mean()
    g_term_gradflow = torch.autograd.grad(memo_g, zt2, retain_graph=False)[0].detach()
# terminal head on cached leaf (detached) — same operating point values
with torch.enable_grad():
    xe = xs_c[1].detach().clone().requires_grad_(True)
    eps_t2 = cfg_combine(*sd.predict_noise(xe, t_b, uc, c))
    x0h_c = (xe - (1 - at_b).sqrt() * eps_t2) / at_b.sqrt()
    x_s_c = alpha_s.sqrt() * x0h_c + (1 - alpha_s).sqrt() * x_T_init.detach()
    nuc_s2, noc_s2 = sd.predict_noise(x_s_c, s_target, uc, c)
    eps_s_c = cfg_combine(nuc_s2, noc_s2)
    memo_c = (eps_ref - eps_s_c).reshape(1, -1).pow(2).mean(-1).mean()
    g_term_cached = torch.autograd.grad(memo_c, xe, retain_graph=False)[0].detach()
print(f"  terminal latent (x_1) cached vs gradflow: "
      f"max|diff|={(xs_c[1]-zt2.detach()).abs().max().item():.3e}")
print(f"  ||g_term gradflow|| = {g_term_gradflow.norm().item():.4e}")
print(f"  ||g_term cached||   = {g_term_cached.norm().item():.4e}")
print(f"  cos(cached,gradflow) = {cos(g_term_cached, g_term_gradflow):.6f}")
print(f"  rel_err              = {re(g_term_cached, g_term_gradflow):.4e}")
if cos(g_term_cached, g_term_gradflow) > 0.999:
    print("  >>> VERDICT C: terminal head has NO operating-point problem.")
else:
    print("  >>> VERDICT C: terminal head DOES differ cached vs gradflow.")

print("\n" + "=" * 78)
print("OVERALL SKEPTIC VERDICT")
print("=" * 78)
