"""
Diagnostic: terminal-head memo gradient signal strength (perspective: terminal-signal).

Goal: quantify how much of g_terminal = dL/dx_{t_idx} actually comes from the
MEMO term, and WHY it is weak. Reproduces the EXACT terminal head of
run_ini_opti.optimize_xT_adj (lines 305-335) but instruments it.

Key chain (all under enable_grad on a leaf x_end):
    x_end = x_{t_idx}  (cached latent)
    eps_t = cfg_combine(unet(x_end, t_break))                 # UNet #1
    x0_hat = (x_end - sqrt(1-alpha_t)*eps_t) / sqrt(alpha_t)  # Tweedie
    x_s = sqrt(alpha_s)*x0_hat + sqrt(1-alpha_s)*x_T.detach() # STOP-GRAD on x_T
    eps_s = cfg_combine(unet(x_s, s_target))                  # UNet #2
    memo_proxy = ||epsilon_ref - eps_s||^2                     # epsilon_ref fixed
    loss = memo_proxy.mean() + lambda_align * loss_align

We measure:
  (1) alpha_t, alpha_s, sqrt(alpha_s), sqrt(1-alpha_s)  -> x_s composition
  (2) |d(memo)/d(x_end)| = terminal memo-gradient norm
  (3) |d(align)/d(x_end)| = terminal align-gradient norm  (compare)
  (4) Sensitivity of eps_s to x0_hat:  ||d(eps_s)/d(x0_hat)||  (Jacobian norm via VJP)
      -> if near 0, memo signal cannot enter x_end (the core of early-step blindness)
  (5) g_terminal magnitude as function of base_s_ratio in {0.1..0.9} and init_steps in {5,10,15,19}
      -> shows the terminal signal is structurally weak regardless of lr sweep.

Run:
  python diag_terminal_signal.py
"""

import os, sys, math
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
from munch import munchify
from latent_diffusion import StableDiffusion


def cfg_combine(nuc, noc, cfg=7.5):
    return nuc + cfg * (noc - nuc)


@torch.no_grad()
def forward_cache(sd, x_T_init, timesteps, t_idx, uc, c, cfg):
    """Reproduce forward no_grad chain of optimize_xT_adj lines 258-281; return xs[t_idx]."""
    x_k = (x_T_init * sd.scheduler.init_noise_sigma).clone()
    xs = [x_k.clone()]
    for step_idx, t in enumerate(timesteps):
        at = sd.alpha(t)
        at_prev = sd.alpha(t - sd.skip)
        nuc, noc = sd.predict_noise(x_k, t, uc, c)
        eps_theta = cfg_combine(nuc, noc, cfg)
        x0_hat = (x_k - (1 - at).sqrt() * eps_theta) / at.sqrt()
        x_k = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta
        if step_idx == t_idx:
            break
        if (step_idx + 1) <= t_idx:
            xs.append(x_k.clone())
    return xs[t_idx].clone()


def measure_one(sd, timesteps, uc, c, cfg, x_T_init, init_steps, base_s_ratio, lambda_align):
    t_idx = init_steps
    s_idx = int(len(timesteps) * base_s_ratio)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)
    t_break = timesteps[t_idx]
    alpha_t = sd.alpha(t_break)

    # cached terminal latent (no grad)
    x_end_state = forward_cache(sd, x_T_init, timesteps, t_idx, uc, c, cfg)

    # reference x0_hat for align loss (un-optimized trajectory). Use same cached path.
    with torch.no_grad():
        at = alpha_t
        nuc, noc = sd.predict_noise(x_end_state, t_break, uc, c)
        eps_theta = cfg_combine(nuc, noc, cfg)
        x0_ref = (x_end_state - (1 - at).sqrt() * eps_theta) / at.sqrt()

    epsilon_ref = x_T_init.detach()

    # ---- terminal head under enable_grad on x_end (EXACT copy of run_ini_opti:305-329) ----
    with torch.enable_grad():
        x_end = x_end_state.detach().clone().requires_grad_(True)

        eps_t = cfg_combine(*sd.predict_noise(x_end, t_break, uc, c), cfg)
        x0_hat = (x_end - (1 - alpha_t).sqrt() * eps_t) / alpha_t.sqrt()

        x_s = (alpha_s.sqrt() * x0_hat
               + (1 - alpha_s).sqrt() * x_T_init.detach())

        nuc_s, noc_s = sd.predict_noise(x_s, s_target, uc, c)
        eps_s = cfg_combine(nuc_s, noc_s, cfg)

        B = eps_s.shape[0]
        memo_proxy = (epsilon_ref - eps_s).reshape(B, -1).pow(2).mean(-1)
        loss_memo = memo_proxy.mean()
        loss_align = ((x0_hat - x0_ref).reshape(x0_hat.shape[0], -1).pow(2).mean(-1).mean())
        loss = loss_memo + lambda_align * loss_align

        # cache x0_hat value for the isolated Jacobian pass below
        x0_hat_val = x0_hat.detach().clone()
        loss_memo_val = loss_memo.item()
        loss_align_val = loss_align.item()

        # (2)(3) terminal grads split (compute sequentially to save memory)
        g_memo = torch.autograd.grad(loss_memo, x_end, retain_graph=True)[0].detach()
        memo_norm_val = g_memo.flatten().norm().item()
        del g_memo
        g_align = torch.autograd.grad(loss_align, x_end, retain_graph=True)[0].detach()
        align_norm_val = g_align.flatten().norm().item()
        del g_align
        g_total = torch.autograd.grad(loss, x_end, retain_graph=False)[0].detach()
        total_norm_val = g_total.flatten().norm().item()
        del g_total

    # (4) sensitivity d(eps_s)/d(x0_hat): isolated pass, fresh graph, freed after.
    # Run AFTER the head graph is gone so memory is clean.
    eps_s_Jnorm = float('nan')
    memo_to_xT = float('nan')
    with torch.enable_grad():
        torch.cuda.empty_cache()
        x0_hat_leaf = x0_hat_val.detach().clone().requires_grad_(True)
        x_s2 = (alpha_s.sqrt() * x0_hat_leaf + (1 - alpha_s).sqrt() * x_T_init.detach())
        nuc_s2, noc_s2 = sd.predict_noise(x_s2, s_target, uc, c)
        eps_s2 = cfg_combine(nuc_s2, noc_s2, cfg)
        v = torch.randn_like(eps_s2)
        Jt_v = torch.autograd.grad(eps_s2, x0_hat_leaf, grad_outputs=v, retain_graph=False)[0]
        eps_s_Jnorm = Jt_v.flatten().norm().item() / (v.flatten().norm().item() + 1e-12)
        # effective coupling of memo signal into x_end via Tweedie chain:
        # d x0_hat/d x_end = 1/sqrt(alpha_t) * (I - sqrt(1-alpha_t) d eps_t/d x_end)
        # dominant scale of memo->x_end ~ sqrt(alpha_s) * ||d eps_s/d x0_hat|| / sqrt(alpha_t)
        memo_to_xT = alpha_s.sqrt().item() * eps_s_Jnorm / alpha_t.sqrt().item()
        del eps_s2, nuc_s2, noc_s2, x_s2, Jt_v, v

    return {
        "t_idx": t_idx,
        "alpha_t": alpha_t.item(),
        "s_target": int(s_target),
        "alpha_s": alpha_s.item(),
        "sqrt_alpha_s": alpha_s.sqrt().item(),
        "sqrt_1malpha_s": (1 - alpha_s).sqrt().item(),
        "frac_x0hat": alpha_s.sqrt().item(),       # weight on x0_hat in x_s
        "frac_xT": (1 - alpha_s).sqrt().item(),    # weight on x_T.detach()
        "loss_memo": loss_memo_val,
        "loss_align": loss_align_val,
        "|g_memo|": memo_norm_val,
        "|g_align|": align_norm_val,
        "|g_total|": total_norm_val,
        "memo_share": memo_norm_val / (total_norm_val + 1e-12),
        "||d eps_s/d x0_hat||": eps_s_Jnorm,
        "memo_to_xT_via_x0hat": memo_to_xT,
    }


def main():
    device = torch.device("cuda:0")
    model_key = os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5")
    solver_config = munchify({"num_sampling": 50})
    sd = StableDiffusion(solver_config=solver_config, model_key=model_key, device=device, seed=42)
    sd.unet.float()
    sd.dtype = torch.float32
    sd.unet.enable_gradient_checkpointing()

    timesteps = list(sd.scheduler.timesteps)
    print(f"len(timesteps)={len(timesteps)}  skip={sd.skip}  init_noise_sigma={sd.scheduler.init_noise_sigma:.4f}")
    snr = lambda i: (sd.alpha(timesteps[i]) / (1 - sd.alpha(timesteps[i]))).item()
    print("[SNR] " + "  ".join(f"s{i}(a={sd.alpha(timesteps[i]).item():.4f},snr={snr(i):.2f})"
                               for i in range(0, len(timesteps), 5)))

    prompt = "An astronaut on the moon"
    uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)
    uc = uc.float(); c = c.float()

    torch.manual_seed(42)
    x_T_init = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)

    cfg = 7.5
    lambda_align = 0.1

    # ===== (A) base_s_ratio sweep at init_steps=10 =====
    print("\n" + "="*100)
    print("(A) base_s_ratio sweep at init_steps=10  -- measures x_s composition + memo gradient")
    print("="*100)
    hdr = f"{'s_ratio':>7} {'s_idx':>5} {'alpha_s':>8} {'w(x0h)':>8} {'w(xT)':>8} " \
          f"{'|g_memo|':>12} {'|g_align|':>12} {'memo_share':>10} {'||dEps/dx0h||':>14}"
    print(hdr)
    print("-"*len(hdr))
    for sr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        torch.cuda.empty_cache()
        r = measure_one(sd, timesteps, uc, c, cfg, x_T_init, init_steps=10,
                        base_s_ratio=sr, lambda_align=lambda_align)
        print(f"{sr:>7.2f} {r['s_target']:>5d} {r['alpha_s']:>8.4f} "
              f"{r['frac_x0hat']:>8.4f} {r['frac_xT']:>8.4f} "
              f"{r['|g_memo|']:>12.4e} {r['|g_align|']:>12.4e} "
              f"{r['memo_share']:>10.4f} {r['||d eps_s/d x0_hat||']:>14.4e}")

    # ===== (B) init_steps sweep at base_s_ratio=0.5 =====
    print("\n" + "="*100)
    print("(B) init_steps sweep at base_s_ratio=0.5  -- terminal-step effect")
    print("="*100)
    hdr2 = f"{'init':>5} {'alpha_t':>8} {'alpha_s':>8} {'w(x0h)':>8} {'w(xT)':>8} " \
           f"{'|g_memo|':>12} {'|g_total|':>12} {'memo_share':>10}"
    print(hdr2)
    print("-"*len(hdr2))
    for is_ in [5, 10, 15, 19, 25, 35]:
        if is_ >= len(timesteps):
            continue
        torch.cuda.empty_cache()
        r = measure_one(sd, timesteps, uc, c, cfg, x_T_init, init_steps=is_,
                        base_s_ratio=0.5, lambda_align=lambda_align)
        print(f"{is_:>5d} {r['alpha_t']:>8.4f} {r['alpha_s']:>8.4f} "
              f"{r['frac_x0hat']:>8.4f} {r['frac_xT']:>8.4f} "
              f"{r['|g_memo|']:>12.4e} {r['|g_total|']:>12.4e} "
              f"{r['memo_share']:>10.4f}")

    # ===== (C) detailed @ default (init=10, s_ratio=0.5) =====
    print("\n" + "="*100)
    print("(C) detailed decomposition @ init_steps=10, base_s_ratio=0.5 (the actual run config)")
    print("="*100)
    torch.cuda.empty_cache()
    r = measure_one(sd, timesteps, uc, c, cfg, x_T_init, init_steps=10,
                    base_s_ratio=0.5, lambda_align=lambda_align)
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k:>26s}: {v:.6e}")
        else:
            print(f"  {k:>26s}: {v}")

    print("\n[INTERPRETATION GUIDE]")
    print("- If w(xT) >> w(x0h) (sqrt(1-a_s) >> sqrt(a_s)): x_s dominated by STOP-GRAD x_T.")
    print("  => memo loss gradient reaches x_end ONLY through the tiny w(x0h)=sqrt(a_s) path.")
    print("- ||d eps_s/d x0_hat|| near 0: UNet eps_s is near-flat wrt x0_hat when input is")
    print("  near-pure-noise (early-step blindness). Memo signal cannot couple into x_end.")
    print("- memo_share << 1: align loss dominates g_terminal; memo contributes negligibly.")
    print("  => adjoint is exact but propagates a near-zero memo signal => x_T barely moves.")
    print("  This explains lr 0..0.08 flatness: the issue is terminal-head design, not adjoint.")


if __name__ == "__main__":
    main()
