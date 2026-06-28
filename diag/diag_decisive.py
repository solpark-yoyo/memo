#!/usr/bin/env python
# diag_decisive.py — DECISIVE test of the adjoint recursion vs true autograd.
#
# The prior diagnostic (diag_cache_sweep.py) reported cosine(adj,true)=-0.85 at t_idx=1.
# This script isolates WHY by testing several hypotheses, including the possibility
# that the diagnostic ITSELF had a bug (e.g. evaluating the adjoint VJP at a
# DIFFERENT latent/timestep than the true Jacobian).
#
# KEY DESIGN: we run BOTH the true gradient and the adjoint on the EXACT SAME
# forward trajectory, and we additionally decompose the single-step adjoint to
# compare A_j term and B_j*Jt_g term separately against the true one-step
# Jacobian-vector product.
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
    print(f"[setup] sigma(init_noise_sigma)={sigma}  skip={skip}")
    print(f"[setup] timesteps[0]={int(timesteps[0])} timesteps[1]={int(timesteps[1])} ... "
          f"alpha(ts[0])={sd.alpha(timesteps[0]).item():.4f} alpha(ts[1])={sd.alpha(timesteps[1]).item():.4f}")

    s_idx = int(len(timesteps) * 0.5)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)
    prompt = "An astronaut on the moon"
    uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)
    uc = uc.float(); c = c.float()

    torch.manual_seed(0)
    x_T = torch.randn(1, 4, 64, 64, device=DEVICE, dtype=torch.float32)
    epsilon_ref = x_T.detach().clone()

    # ============================================================
    # TEST 1: t_idx=1. Single forward step x_0 -> x_1.
    # Compare true dL/dx_T (full autograd) vs adjoint recursion.
    # ============================================================
    t_idx = 1
    print("\n" + "=" * 60)
    print(f"TEST 1: t_idx={t_idx} (single forward step)")
    print("=" * 60)

    # ---- TRUE full autograd ----
    x_Tl = x_T.clone().requires_grad_(True)
    zt = x_Tl * sigma                     # x_0 = x_T * sigma
    x_0 = zt.detach().clone()             # for reference
    # forward step 0: x_0 -> x_1
    t0 = timesteps[0]
    at0 = sd.alpha(t0); at0_prev = sd.alpha(t0 - skip)
    nuc, nc = sd.predict_noise(zt, t0, uc, c); eps = cfg_combine(nuc, nc, cfg)
    x0h = (zt - (1 - at0).sqrt() * eps) / at0.sqrt()
    zt_next = at0_prev.sqrt() * x0h + (1 - at0_prev).sqrt() * eps   # this is x_1
    x_end_true = zt_next

    # terminal head (same as optimize_xT_adj)
    t_break = timesteps[t_idx]; at_break = sd.alpha(t_break)
    eps_t = cfg_combine(*sd.predict_noise(x_end_true, t_break, uc, c))
    x0_hat = (x_end_true - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
    x_s = alpha_s.sqrt().to(torch.float32) * x0_hat + (1 - alpha_s).sqrt().to(torch.float32) * x_Tl.detach()
    nuc_s, nc_s = sd.predict_noise(x_s, s_target, uc, c)
    eps_s = cfg_combine(nuc_s, nc_s, cfg)
    loss = ((epsilon_ref - eps_s).reshape(1, -1) ** 2).mean()
    g_true = torch.autograd.grad(loss, x_Tl, retain_graph=True)[0]
    # also get dL/dx_1 (= dL/dx_end_true) for decomposition
    g_x1_true = torch.autograd.grad(loss, x_end_true, retain_graph=True)[0]
    print(f"  TRUE: loss={loss.item():.6e}  |g_true(dL/dx_T)|={g_true.norm().item():.6e}  |g_x1(dL/dx_1)|={g_x1_true.norm().item():.6e}")

    # ---- ADJOINT (mirrors optimize_xT_adj exactly) ----
    with torch.no_grad():
        xk = (x_T.detach() * sigma).clone()
        cache = [xk.clone()]
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t); at_prev = sd.alpha(t - skip)
            nuc, nc = sd.predict_noise(xk, t, uc, c); eps = cfg_combine(nuc, nc, cfg)
            x0h = (xk - (1 - at).sqrt() * eps) / at.sqrt()
            xk = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
            if step_idx == t_idx:
                break
            if (step_idx + 1) <= t_idx:
                cache.append(xk.clone())
    print(f"  cache: len={len(cache)}  |cache[0]-x_0|={(cache[0]-x_0).abs().max().item():.3e}  (should be 0)")
    print(f"         |cache[1]-x_end_true|={(cache[1]-x_end_true.detach()).abs().max().item():.3e}  (should be 0)")

    with torch.enable_grad():
        x_end = cache[t_idx].detach().clone().requires_grad_(True)
        eps_t2 = cfg_combine(*sd.predict_noise(x_end, t_break, uc, c))
        x0h2 = (x_end - (1 - at_break).sqrt() * eps_t2) / at_break.sqrt()
        x_s2 = alpha_s.sqrt().to(torch.float32) * x0h2 + (1 - alpha_s).sqrt().to(torch.float32) * x_T.detach()
        nuc_s2, nc_s2 = sd.predict_noise(x_s2, s_target, uc, c)
        eps_s2 = cfg_combine(nuc_s2, nc_s2, cfg)
        loss2 = ((epsilon_ref - eps_s2).reshape(1, -1) ** 2).mean()
        g = torch.autograd.grad(loss2, x_end)[0]   # terminal adjoint = dL/dx_1
    print(f"  ADJ terminal: loss2={loss2.item():.6e}  |g_terminal(dL/dx_1)|={g.norm().item():.6e}")
    print(f"  cos(g_terminal_adj, g_x1_true) = {cos(g, g_x1_true):.4f}  (should be ~1.0)")

    # ---- single adjoint step (j=0): g_0 = A_0*g_1 + B_0*(J_0^T g_1) ----
    j = 0; t_j = timesteps[j]
    a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j - skip)
    A_j = (a_jp1 / a_j).sqrt()
    B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()
    print(f"\n  step j=0: t_j={int(t_j)}  a_j(alpha(ts[0]))={a_j.item():.6f}  "
          f"a_jp1(alpha(ts[0]-skip))={a_jp1.item():.6f}  (NOTE: a_jp1 > a_j since cleaner)")
    print(f"            A_j={A_j.item():.6f}  B_j={B_j.item():+.6f}")
    x_j_local = cache[j].detach().clone().requires_grad_(True)
    eps_j = cfg_combine(*sd.predict_noise(x_j_local, t_j, uc, c))
    Jt_g = torch.autograd.grad(eps_j, x_j_local, grad_outputs=g)[0]
    g0_adj = A_j * g + B_j * Jt_g
    g_adj = g0_adj * sigma
    print(f"\n  FINAL: cos(g_adj, g_true) = {cos(g_adj, g_true):.4f}")
    print(f"         |g_adj|/|g_true|    = {(g_adj.norm()/g_true.norm()).item():.4f}")

    # ============================================================
    # TEST 2: DECOMPOSE the true one-step Jacobian to find the mismatch.
    # The TRUE single-step map is x_0 -> x_1 where
    #   x_1 = at0_prev.sqrt()*x0_hat(x_0) + (1-at0_prev).sqrt()*eps(x_0)
    #   x0_hat(x_0) = (x_0 - (1-at0).sqrt()*eps(x_0))/at0.sqrt()
    # i.e. x_1 = c1*x_0 + c2*eps(x_0) with:
    #   c1 = at0_prev.sqrt()/at0.sqrt()
    #   c2 = (1-at0_prev).sqrt() - at0_prev.sqrt()*(1-at0).sqrt()/at0.sqrt()
    # So the TRUE Jacobian d x_1/d x_0 = c1*I + c2*(d eps/d x_0)
    # Compare c1,c2 to A_j,B_j. They use DIFFERENT alpha indices!
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: compare adjoint A_j,B_j to TRUE affine coeffs c1,c2")
    print("=" * 60)
    # TRUE coeffs: forward step 0 uses t=timesteps[0], t_prev=timesteps[0]-skip
    c1 = (at0_prev / at0).sqrt()
    c2 = (1 - at0_prev).sqrt() - (at0_prev * (1 - at0) / at0).sqrt()
    print(f"  TRUE c1 (at0_prev/at0).sqrt()       = {c1.item():.6f}")
    print(f"  ADJ  A_j (a_jp1/a_j).sqrt()         = {A_j.item():.6f}")
    print(f"  diff c1-A_j                          = {(c1-A_j).item():.6e}")
    print(f"  TRUE c2 = (1-at0_prev).sqrt() - (at0_prev*(1-at0)/at0).sqrt() = {c2.item():.6f}")
    print(f"  ADJ  B_j = (1-a_jp1).sqrt()  - (a_jp1*(1-a_j)/a_j).sqrt()    = {B_j.item():.6f}")
    print(f"  diff c2-B_j                          = {(c2-B_j).item():.6e}")
    print(f"  NOTE: at0 = alpha(timesteps[0]) = {at0.item():.6f}, a_j = alpha(timesteps[0]) = {a_j.item():.6f}")
    print(f"        at0_prev = alpha(timesteps[0]-skip) = {at0_prev.item():.6f}")
    print(f"        a_jp1    = alpha(timesteps[0]-skip) = {a_jp1.item():.6f}")
    print(f"  => c1==A_j and c2==B_j IDENTICALLY? "
          f"c1==A_j:{torch.allclose(c1,A_j)}  c2==B_j:{torch.allclose(c2,B_j)}")

    # ============================================================
    # TEST 3: if coeffs match, the VJP must match too.
    # Compute TRUE d x_1/d x_0 . g  (one-step JVP) via autograd and compare
    # to A_j*g + B_j*(J^T g).
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: TRUE one-step Jacobian-vector vs adjoint recursion")
    print("=" * 60)
    x0_leaf = cache[0].detach().clone().requires_grad_(True)
    nuc_t, nc_t = sd.predict_noise(x0_leaf, t0, uc, c); eps_t0 = cfg_combine(nuc_t, nc_t, cfg)
    x0h_t = (x0_leaf - (1 - at0).sqrt() * eps_t0) / at0.sqrt()
    x1_from_leaf = at0_prev.sqrt() * x0h_t + (1 - at0_prev).sqrt() * eps_t0
    # TRUE d x_1/d x_0 ^T applied to g  (= JVP in reverse)
    Jt_true = torch.autograd.grad(x1_from_leaf, x0_leaf, grad_outputs=g)[0]
    print(f"  TRUE (d x_1/d x_0)^T g  : |.|={Jt_true.norm().item():.6e}")
    print(f"  ADJ  A_j*g + B_j*Jt_g   : |.|={g0_adj.norm().item():.6e}")
    print(f"  cos(TRUE_step, ADJ_step)= {cos(Jt_true, g0_adj):.4f}")
    print(f"  After *sigma: cos = {cos(Jt_true*sigma, g_adj):.4f}")

    # ============================================================
    # TEST 4: the full TRUE gradient dL/dx_0 should be Jt_true*sigma.
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 4: does TRUE dL/dx_T = (d x_1/d x_0)^T g_x1 * sigma ?")
    print("=" * 60)
    g_true_pred = Jt_true * sigma
    print(f"  cos(g_true, Jt_true*sigma) = {cos(g_true, g_true_pred):.4f}")
    print(f"  |g_true|={g_true.norm().item():.6e}  |Jt_true*sigma|={g_true_pred.norm().item():.6e}")

    sd.unet.half()


if __name__ == "__main__":
    main()
