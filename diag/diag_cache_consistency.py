"""
Diagnostic: cache-consistency + adjoint-recursion sanity for optimize_xT_adj.

Run inside the project conda env (groundit):
  /home/geonsoo/Desktop/Datasets/Parksol/.conda/envs/groundit/bin/python \
      /home/geonsoo/Desktop/Datasets/Parksol/memo/ori_memo/diag_cache_consistency.py

Checks (perspective: cache-consistency / off-by-one):
  C1. Forward cache xs[j] equals the UNet INPUT at forward step j (recompute check).
  C2. Adjoint coeffs A_j, B_j match the empirical Jacobian of the cached DDIM step
      (finite-difference the actual x_{j+1} = F_j(x_j) along a random direction).
  C3. The FULL adjoint gradient (xs cache path) matches a finite-difference estimate
      of dL/dx_T computed by perturbing x_T and re-running the WHOLE forward.
      -> If |adj - fd| / |fd| is small, the cache + recursion are CORRECT and the
         lr-inertia is NOT a cache bug. If large, cache/recursion is the root cause.
  C4. Decompose: compare adjoint g_xT vs pure-terminal approx prod(A_j)*dL/dx_{t_idx}.
      -> If they agree to within ~|B_j|, the recursion collapses to terminal-only
         (J_j ~ 0), proving the forward dynamics are washed out.
"""

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


@torch.no_grad()
def recompute_to(sd, x_T, timesteps, k, uc, c, cfg, sigma):
    """Independent re-run of k DDIM steps from x_T (no cache)."""
    x = x_T.detach() * sigma
    for step_idx, t in enumerate(timesteps):
        if step_idx == k:
            return x.clone()
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        nuc, nc = sd.predict_noise(x, t, uc, c)
        eps = cfg_combine(nuc, nc, cfg)
        x0h = (x - (1 - at).sqrt() * eps) / at.sqrt()
        x = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
    return x.clone()


def main():
    sd = StableDiffusion(solver_config=munchify({"num_sampling": 50}),
                         model_key=MODEL_KEY, device=DEVICE, seed=42)
    sd.unet.float(); sd.dtype = torch.float32
    timesteps = list(sd.scheduler.timesteps)
    skip = sd.skip
    sigma = float(sd.scheduler.init_noise_sigma)
    print(f"[setup] NFE=50 skip={skip} init_noise_sigma={sigma}")
    print(f"[setup] timesteps[:3]={timesteps[:3]} ... [-3:]={timesteps[-3:]}")

    prompt = "An astronaut on the moon"
    uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)
    uc = uc.float(); c = c.float()
    cfg = 7.5

    t_idx = 10            # first update step (init_steps=10)
    s_idx = int(len(timesteps) * 0.5)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)

    torch.manual_seed(0)
    x_T = torch.randn(1, 4, 64, 64, device=DEVICE, dtype=torch.float32)
    epsilon_ref = x_T.detach().clone()

    # ---------- C1: cache vs independent recompute ----------
    x_k = (x_T.detach() * sigma).clone()
    xs = [x_k.clone()]
    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
            nuc, nc = sd.predict_noise(x_k, t, uc, c)
            eps = cfg_combine(nuc, nc, cfg)
            x0h = (x_k - (1 - at).sqrt() * eps) / at.sqrt()
            x_k = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
            if step_idx == t_idx:
                break
            if (step_idx + 1) <= t_idx:
                xs.append(x_k.clone())
    print(f"\n[C1] len(xs)={len(xs)} (expected t_idx+1={t_idx+1})")
    max_err = 0.0
    for j in range(len(xs)):
        x_re = recompute_to(sd, x_T, timesteps, j, uc, c, cfg, sigma)
        err = (xs[j] - x_re).abs().max().item()
        max_err = max(max_err, err)
    print(f"[C1] max|x_cached[j] - x_recomputed[j]| over j=0..{len(xs)-1}: {max_err:.3e}")
    print(f"[C1] {'PASS' if max_err < 1e-4 else 'FAIL'}: cache latents match independent recompute.")

    # ---------- C2: empirical A_j, B_j vs coded A_j, B_j ----------
    print("\n[C2] coded vs empirical affine coeffs (FD along random dir):")
    v = torch.randn_like(xs[0])
    v = v / v.norm()
    worst = 0.0
    for k in range(t_idx, 0, -1):
        j = k - 1
        t_j = timesteps[j]
        a_j = float(sd.alpha(t_j)); a_jp1 = float(sd.alpha(t_j - sd.skip))
        A_code = (a_jp1 / a_j) ** 0.5
        B_code = (1 - a_jp1) ** 0.5 - (a_jp1 * (1 - a_j) / a_j) ** 0.5

        # empirical F_j(x_j) and F_j(x_j + h v) -> extract A_j, B_j*(Jv)
        def F(x_in):
            at = sd.alpha(t_j); at_prev = sd.alpha(t_j - sd.skip)
            nuc, nc = sd.predict_noise(x_in, t_j, uc, c)
            eps = cfg_combine(nuc, nc, cfg)
            x0h = (x_in - (1 - at).sqrt() * eps) / at.sqrt()
            return at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps, eps

        with torch.no_grad():
            f0, eps0 = F(xs[j])
            h = 1e-3
            fp, ep = F(xs[j] + h * v)
            # f(x+hv)-f(x) = A*h*v + B*(J*h*v)  => (fp-f0)/h = A*v + B*(Jv)
            emp = (fp - f0) / h
            A_v_plus_BJv = emp
            # coded prediction: A_code*v + B_code*(Jv); need Jv from eps perturbation
            Jv = (ep - eps0) / h
            coded_pred = A_code * v + B_code * Jv
            rel = (A_v_plus_BJv - coded_pred).norm().item() / (A_v_plus_BJv.norm().item() + 1e-12)
            worst = max(worst, rel)
            if k in (t_idx, t_idx // 2, 1):
                print(f"  k={k} j={j} t_j={int(t_j)}: A_code={A_code:.4f} B_code={B_code:+.4f}  "
                      f"rel_err(FD vs coded)={rel:.3e}  |Jv|={Jv.norm().item():.4e}")
    print(f"[C2] worst rel_err over all steps: {worst:.3e}  -> {'PASS (<1e-2)' if worst < 1e-2 else 'FAIL'}")

    # ---------- terminal loss + adjoint (replicate optimize_xT_adj head) ----------
    def loss_of_xT(x_T_var, use_adjoint=True):
        """Return scalar loss and (optionally) g w.r.t. x_T via adjoint."""
        # forward cache
        xk = (x_T_var.detach() * sigma).clone()
        cache = [xk.clone()]
        with torch.no_grad():
            for step_idx, t in enumerate(timesteps):
                at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
                nuc, nc = sd.predict_noise(xk, t, uc, c)
                eps = cfg_combine(nuc, nc, cfg)
                x0h = (xk - (1 - at).sqrt() * eps) / at.sqrt()
                xk = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
                if step_idx == t_idx:
                    break
                if (step_idx + 1) <= t_idx:
                    cache.append(xk.clone())
        x_end_state = cache[t_idx].clone()

        with torch.enable_grad():
            x_end = x_end_state.detach().clone().requires_grad_(True)
            t_break = timesteps[t_idx]; at_break = sd.alpha(t_break)
            eps_t = cfg_combine(*sd.predict_noise(x_end, t_break, uc, c))
            x0_hat = (x_end - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
            x_s = alpha_s.sqrt().to(torch.float32) * x0_hat \
                  + (1 - alpha_s).sqrt().to(torch.float32) * x_T_var.detach().to(torch.float32)
            nuc_s, nc_s = sd.predict_noise(x_s, s_target, uc, c)
            eps_s = cfg_combine(nuc_s, nc_s, cfg)
            memo = ((epsilon_ref - eps_s).reshape(1, -1) ** 2).mean()
            loss = memo
            g = torch.autograd.grad(loss, x_end)[0]   # dL/dx_{t_idx}

        if use_adjoint:
            for k in range(t_idx, 0, -1):
                j = k - 1
                t_j = timesteps[j]
                a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j - sd.skip)
                A_j = (a_jp1 / a_j).sqrt()
                B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()
                x_j_local = cache[j].detach().clone().requires_grad_(True)
                eps_j = cfg_combine(*sd.predict_noise(x_j_local, t_j, uc, c))
                Jt_g = torch.autograd.grad(eps_j, x_j_local, grad_outputs=g)[0]
                g = A_j * g + B_j * Jt_g
            g_xT = g * sigma
        else:
            g_xT = None
        return loss, g_xT, g  # g (terminal) for C4

    x_T_leaf = x_T.clone().requires_grad_(True)
    loss_val, g_adj, g_term = loss_of_xT(x_T_leaf, use_adjoint=True)
    print(f"\n[adjoint] loss={loss_val.item():.6f}  |g_xT_adj|={g_adj.norm().item():.6e}  "
          f"|g_terminal(dL/dx_tidx)|={g_term.norm().item():.6e}")

    # ---------- C3: full FD of dL/dx_T (re-run whole forward per perturbation) ----------
    print("\n[C3] finite-difference dL/dx_T (re-run full forward chain, no cache reuse):")
    v = torch.randn_like(x_T); v = v / v.norm()
    h = 1e-2
    with torch.no_grad():
        lp, _, _ = loss_of_xT(x_T + h * v, use_adjoint=False)
        lm, _, _ = loss_of_xT(x_T - h * v, use_adjoint=False)
    fd_dir = (lp - lm).item() / (2 * h)   # directional derivative along v
    adj_dir = (g_adj * v).sum().item()
    print(f"  directional deriv along v:  adjoint={adj_dir:.6e}  FD={fd_dir:.6e}")
    rel = abs(adj_dir - fd_dir) / (abs(fd_dir) + 1e-12)
    print(f"  |adj - FD| / |FD| = {rel:.3e}  -> {'PASS (<0.3 => cache/recursion CORRECT)' if rel < 0.3 else 'FAIL (cache/recursion suspect)'}")

    # ---------- C4: pure-terminal approximation ----------
    import numpy as np
    Aprod = 1.0
    for k in range(t_idx, 0, -1):
        j = k - 1; a_j = float(sd.alpha(timesteps[j])); a_jp1 = float(sd.alpha(timesteps[j] - sd.skip))
        Aprod *= (a_jp1 / a_j) ** 0.5
    g_term_approx = g_term * Aprod * sigma
    cos = torch.nn.functional.cosine_similarity(
        g_adj.flatten().unsqueeze(0), g_term_approx.flatten().unsqueeze(0)).item()
    ratio = g_adj.norm().item() / (g_term_approx.norm().item() + 1e-12)
    print(f"\n[C4] pure-terminal approx prod(A_j)*sigma*dL/dx_tidx:")
    print(f"  prod(A_j)={Aprod:.4f}  cosine(adj, terminal_approx)={cos:.4f}  |adj|/|term_approx|={ratio:.4f}")
    print(f"  -> cos~1 & ratio~1 means recursion collapsed to terminal-only (J_j~0):")
    print(f"     the forward chain's NONLINEAR dynamics are NOT reflected in g_xT.")
    print(f"     g_xT points along dL/dx_tidx regardless of x_T's actual influence path.")

    sd.unet.half()


if __name__ == "__main__":
    main()
