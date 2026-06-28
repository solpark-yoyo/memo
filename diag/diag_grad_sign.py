"""
Micro-diagnostic: verify the adjoint gradient DIRECTION for the memo loss.
We run the adjoint ONCE (single update, t_idx=10), get g_xT = x_T.grad, then
manually probe the memo loss at x_T +/- eps*normalized(g) to see whether the
gradient actually points downhill.

  L(x_T)              : forward memo loss from scratch at x_T
  L(x_T + eps*g_hat)  : loss moving ALONG +gradient  (should INCREASE for a descent dir)
  L(x_T - eps*g_hat)  : loss moving AGAINST gradient (should DECREASE for a descent dir)

If L(x_T - eps*g) < L(x_T) < L(x_T + eps*g): gradient is a valid DESCENT direction
  => optimizer should reduce memo loss; the observed memo-INCREASE must come from
     something else (e.g. the loss being re-evaluated at a different t_idx each step,
     or lr too large overshooting).
If the opposite: the adjoint gradient points the WRONG way (sign/adjoint bug).

We reuse the EXACT terminal-head + adjoint code by importing optimize_xT_adj's
internal pieces via a thin re-implementation is risky; instead we call optimize_xT_adj
with lr=0 to obtain x_T.grad (it sets x_T.grad even when lr=0 — the optimizer just
does nothing), then probe L at perturbed x_T using the SAME forward+terminal-head.

To make the probe loss identical to the optimizer's, we replicate the forward-chain
+ terminal-head (2 UNets) memo loss as a standalone function.
"""

import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse, torch
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed
import run_ini_opti as RIO


def forward_to_x_at_idx(sd, x_T, timesteps, t_idx, uc, c, cfg):
    """Replicate the no_grad forward chain to x_{t_idx} (matches adjoint (1))."""
    x_k = (x_T.detach() * sd.scheduler.init_noise_sigma).clone()
    for step_idx, t in enumerate(timesteps):
        if step_idx == t_idx:
            return x_k.clone()
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        with torch.no_grad():
            nuc, nc = sd.predict_noise(x_k, t, uc, c)
        eps = nuc + cfg * (nc - nuc)
        x0h = (x_k - (1 - at).sqrt() * eps) / at.sqrt()
        x_k = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
    return x_k.clone()


def memo_loss_at(sd, x_T, timesteps, t_idx, uc, c, cfg,
                 alpha_s, s_target, epsilon_ref, init_noise_sigma):
    """Replicate the adjoint terminal-head memo loss (lines 305-326).
    Returns scalar loss (memo_proxy.mean())."""
    x_end_state = forward_to_x_at_idx(sd, x_T, timesteps, t_idx, uc, c, cfg)
    with torch.enable_grad():
        x_end = x_end_state.detach().clone().requires_grad_(True)
        t_break = timesteps[t_idx]
        at_break = sd.alpha(t_break)
        eps_t = (lambda nuc, nc: nuc + cfg * (nc - nuc))(*sd.predict_noise(x_end, t_break, uc, c))
        x0_hat = (x_end - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
        x_s = (alpha_s.sqrt() * x0_hat
               + (1 - alpha_s).sqrt() * x_T.detach())
        nuc_s, nc_s = sd.predict_noise(x_s, s_target, uc, c)
        eps_s = nuc_s + cfg * (nc_s - nuc_s)
        B = eps_s.shape[0]
        memo = (epsilon_ref - eps_s).reshape(B, -1).pow(2).mean(-1).mean()
    return memo.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", type=str,
                   default=os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--NFE", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--init_steps", type=int, default=10)
    p.add_argument("--base_s_ratio", type=float, default=0.5)
    p.add_argument("--probe_t_idx", type=int, default=10)
    p.add_argument("--eps", type=float, default=1e-2)
    p.add_argument("--prompt", type=str, default="An astronaut on the moon")
    args = p.parse_args()
    device = torch.device(args.device)

    sd = StableDiffusion(solver_config=munchify({"num_sampling": args.NFE}),
                         model_key=args.model_key, device=device, seed=args.seed)
    sd.unet.enable_gradient_checkpointing()
    uc, c = sd.get_text_embed(null_prompt="", prompt=args.prompt)
    uc = uc.float(); c = c.float()

    # run optimize_xT_adj with lr=0 to get x_T.grad for t_idx=init_steps (first update)
    set_seed(args.seed)
    sd.unet.half(); sd.dtype = torch.float16
    # capture the print to read g_xT and also grab x_T_init from a fixed seed
    import io, contextlib, re
    set_seed(args.seed)
    x_T_init = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
    # the optimizer creates its own x_T_init with randn; to match we must set seed identically
    set_seed(args.seed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # single update at init_steps
        RIO.optimize_xT_adj(sd, uc, c, args.cfg, device,
                            args.init_steps, 1, 1, 0.0, args.base_s_ratio, 0.1,
                            batch_size=1)
    log = buf.getvalue()
    g_vals = [float(x) for x in re.findall(r"\|g_xT\|=([\d.eE+\-]+)", log)]
    print(f"adjoint |g_xT| at t_idx={args.init_steps}: {g_vals[-1] if g_vals else 'n/a'}")

    # Now reconstruct x_T_init the SAME way the optimizer did (randn after set_seed)
    set_seed(args.seed)
    x_T_init = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
    sd.unet.float(); sd.dtype = torch.float32
    timesteps = list(sd.scheduler.timesteps)
    s_idx = int(len(timesteps) * args.base_s_ratio)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)
    epsilon_ref = x_T_init.detach()

    t_idx = args.probe_t_idx

    # Probe loss at x_T_init (unperturbed)
    L0 = memo_loss_at(sd, x_T_init, timesteps, t_idx, uc, c, args.cfg,
                      alpha_s, s_target, epsilon_ref, sd.scheduler.init_noise_sigma)
    # Re-derive the adjoint gradient at x_T_init (call adj once more, lr=0, read x_T.grad)
    # We re-run the optimizer but it creates a NEW x_T; instead we recompute g via a fresh
    # leaf by calling the optimizer which returns x_T_opt (= unchanged since lr=0). We need
    # x_T.grad. Simpler: replicate by calling optimize_xT_adj and parsing, but that resets x_T.
    # Accept the magnitude from the first run; for direction we need sign, so do a manual
    # adjoint-gradient-free finite-difference sign check:
    eps = args.eps
    g_hat = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
    # We need the ACTUAL adjoint gradient direction. Re-run optimizer, but instead of using
    # its internal x_T, we instrument by monkeypatching torch.optim.Adam.step to capture grad.
    captured = {}
    orig_step = torch.optim.Adam.step
    def spy_step(self, *a, **k):
        for pg in self.param_groups:
            for p in pg["params"]:
                if p.grad is not None and p.shape == x_T_init.shape:
                    captured["g"] = p.grad.detach().clone()
        return orig_step(self, *a, **k)
    torch.optim.Adam.step = spy_step
    set_seed(args.seed)
    sd.unet.half(); sd.dtype = torch.float16
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        RIO.optimize_xT_adj(sd, uc, c, args.cfg, device,
                            args.init_steps, 1, 1, 0.0, args.base_s_ratio, 0.1,
                            batch_size=1)
    torch.optim.Adam.step = orig_step
    g = captured.get("g", None)
    if g is None:
        print("FAILED to capture grad"); return
    g = g.float()
    gnorm = g.flatten().norm()
    g_hat = g / (gnorm + 1e-12)

    sd.unet.float(); sd.dtype = torch.float32
    # probe along +/- g_hat
    L_plus = memo_loss_at(sd, x_T_init + eps * g_hat * gnorm, timesteps, t_idx, uc, c, args.cfg,
                          alpha_s, s_target, epsilon_ref, sd.scheduler.init_noise_sigma)
    L_minus = memo_loss_at(sd, x_T_init - eps * g_hat * gnorm, timesteps, t_idx, uc, c, args.cfg,
                           alpha_s, s_target, epsilon_ref, sd.scheduler.init_noise_sigma)

    print(f"\n=== GRADIENT DIRECTION PROBE at t_idx={t_idx}, eps={eps} ===")
    print(f"  L(x_T)           = {L0:.6f}")
    print(f"  L(x_T + eps*g)   = {L_plus:.6f}   (along +grad)")
    print(f"  L(x_T - eps*g)   = {L_minus:.6f}  (along -grad, descent direction)")
    print(f"  |g_xT| = {gnorm.item():.6f}")
    print()
    if L_minus < L0 < L_plus:
        print("VERDICT: gradient is a valid DESCENT direction (L(-eps*g) < L0 < L(+eps*g)).")
        print("  => adjoint direction is CORRECT. memo-INCREASE across steps comes from")
        print("     something else: re-evaluating L at a later t_idx each step / lr overshoot /")
        print("     the loss landscape being redefined per-step (different x0_orig_ref).")
    elif L_plus < L0 < L_minus:
        print("VERDICT: gradient points UPHILL (L(+eps*g) < L0 < L(-eps*g)). SIGN BUG in adjoint!")
    else:
        print(f"VERDICT: non-monotone (L0={L0:.5f}, L+={L_plus:.5f}, L-={L_minus:.5f}); "
              f"delta(L-)={L_minus-L0:.2e}, delta(L+)={L_plus-L0:.2e}")


if __name__ == "__main__":
    main()
