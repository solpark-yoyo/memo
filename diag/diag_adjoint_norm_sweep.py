"""
Diagnostic: real-UNet adjoint gradient MAGNITUDE vs chain length (t_idx).

Perspective 6 corollary. Having established that optimize_xT_adj is
mathematically EXACT (diag_adjoint_vs_direct.py + diag_adjoint_toy.py), the
production lr-unresponsiveness must come from the gradient's DYNAMICS, not a
bug. This script measures, on the real SD1.5 UNet, how the adjoint gradient
norm and the |g_final|/|g_terminal| ratio behave as t_idx grows (1..10).

Two candidate failure modes:
  (F1) EXPLOSION: |g_xT| grows huge with t_idx. Adam normalizes per-element
       by sqrt(v)+eps, so direction is preserved but the implicit step is
       bounded -- huge grads do NOT produce huge x_T moves. The terminal
       head's signal (prompt-discriminating) gets amplified in a direction
       that may be ~orthogonal to the useful memo-mitigation direction.
  (F2) VANISHING (the original hypothesis): |g_xT| -> 0. Ruled OUT for the
       adjoint by diag_adjoint_toy.py (norm explodes there). This checks the
       real schedule.

The decisive quantity is the ratio |g_xT| / |g_terminal|: if it >> 1 and
growing, explosion; if -> 0, vanishing.
"""
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import argparse, torch
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed


def adjoint_grad_and_terminal(sd, uc, c, cfg, device, x_T_init, t_idx,
                              base_s_ratio, lambda_align, x0_orig_ref):
    sd.unet.float(); sd.dtype = torch.float32
    timesteps = list(sd.scheduler.timesteps)
    s_idx = int(len(timesteps) * base_s_ratio)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)
    x_T = x_T_init.clone().detach().requires_grad_(True)
    eps_ref = x_T_init.detach()

    def comb(nu, nc): return nu + cfg * (nc - nu)

    x_k = (x_T.detach() * sd.scheduler.init_noise_sigma).clone()
    xs = [x_k.clone()]
    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
            nu, nc = sd.predict_noise(x_k, t, uc, c)
            eps = comb(nu, nc)
            x0h = (x_k - (1 - at).sqrt() * eps) / at.sqrt()
            x_k = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
            if step_idx == t_idx:
                break
            if (step_idx + 1) <= t_idx:
                xs.append(x_k.clone())
    x_end_state = xs[t_idx].clone() if len(xs) > t_idx else xs[0].clone()

    with torch.enable_grad():
        x_end = x_end_state.detach().clone().requires_grad_(True)
        t_break = timesteps[t_idx]; at_break = sd.alpha(t_break)
        eps_t = comb(*sd.predict_noise(x_end, t_break, uc, c))
        x0_hat = (x_end - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
        x_s = (alpha_s.sqrt().to(sd.dtype) * x0_hat
               + (1 - alpha_s).sqrt().to(sd.dtype) * x_T.detach().to(sd.dtype))
        eps_s = comb(*sd.predict_noise(x_s, s_target, uc, c))
        B = eps_s.shape[0]
        memo = (eps_ref.to(sd.dtype) - eps_s).reshape(B, -1).pow(2).mean(-1)
        loss_memo = memo.mean()
        loss_align = ((x0_hat.float() - x0_orig_ref)
                      .reshape(B, -1).pow(2).mean(-1).mean())
        loss = loss_memo + lambda_align * loss_align
        g = torch.autograd.grad(loss, x_end, retain_graph=False)[0]
        g_terminal_norm = g.flatten().norm().item()
        g_terminal = g.detach().clone()

        if t_idx == 0:
            g = g * sd.scheduler.init_noise_sigma
        else:
            for k in range(t_idx, 0, -1):
                j = k - 1
                t_j = timesteps[j]
                a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j - sd.skip)
                A_j = (a_jp1 / a_j).sqrt()
                B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()
                x_j_local = xs[j].detach().clone().requires_grad_(True)
                eps_j = comb(*sd.predict_noise(x_j_local, t_j, uc, c))
                Jt_g = torch.autograd.grad(eps_j, x_j_local,
                                           grad_outputs=g, retain_graph=False)[0]
                g = A_j * g + B_j * Jt_g
            g = g * sd.scheduler.init_noise_sigma
    return g.detach(), g_terminal, loss.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--NFE", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--t_idx_list", type=int, nargs="+", default=[1,2,3,5,7,10])
    p.add_argument("--base_s_ratio", type=float, default=0.5)
    p.add_argument("--lambda_align", type=float, default=0.1)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--model_key", type=str,
                   default=os.path.join(SCRIPT_DIR, "ckpt", "sd14_memor_LAION2B_40k"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--prompt", type=str, default="An astronaut on the moon")
    args = p.parse_args()
    device = torch.device(args.device)

    sd = StableDiffusion(solver_config=munchify({"num_sampling": args.NFE}),
                         model_key=args.model_key, device=device, seed=args.base_seed)
    sd.unet.enable_gradient_checkpointing()
    uc, c = sd.get_text_embed(null_prompt="", prompt=args.prompt)
    uc = uc.float().to(device); c = c.float().to(device)
    sd.vae.to("cpu"); sd.text_encoder.to("cpu"); torch.cuda.empty_cache()

    set_seed(args.base_seed)
    x_T_init = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
    timesteps = list(sd.scheduler.timesteps)

    x0_orig_refs = {}
    sd.unet.float(); sd.dtype = torch.float32
    with torch.no_grad():
        zt = x_T_init.to(torch.float32) * sd.scheduler.init_noise_sigma
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
            nu, nc = sd.predict_noise(zt, t, uc, c)
            eps = nu + args.cfg * (nc - nu)
            x0h = (zt - (1 - at).sqrt() * eps) / at.sqrt()
            zt = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
            if step_idx in args.t_idx_list:
                x0_orig_refs[step_idx] = x0h.detach().clone().float()

    print("\n=== real-UNet adjoint gradient dynamics vs chain length ===")
    print(f"{'t_idx':>5} {'alpha_t':>9} {'|g_xT|':>12} {'|g_term|':>12} "
          f"{'|g_xT|/|g_term|':>16}  mode")
    for t_idx in args.t_idx_list:
        if t_idx >= len(timesteps):
            continue
        sd.unet.float(); sd.dtype = torch.float32
        g, g_term, loss = adjoint_grad_and_terminal(
            sd, uc, c, args.cfg, device, x_T_init, t_idx,
            args.base_s_ratio, args.lambda_align, x0_orig_refs[t_idx])
        torch.cuda.empty_cache()
        at_t = sd.alpha(timesteps[t_idx]).item()
        ratio = g.flatten().norm().item() / (g_term.flatten().norm().item() + 1e-12)
        if ratio > 3.0:
            mode = "EXPLODE"
        elif ratio < 0.1:
            mode = "VANISH"
        else:
            mode = "ok"
        print(f"{t_idx:>5} {at_t:>9.4f} {g.flatten().norm().item():>12.4e} "
              f"{g_term.flatten().norm().item():>12.4e} {ratio:>16.4e}  {mode}")
    print("\n(if EXPLODE grows with t_idx, Adam (scale-invariant) silently caps")
    print(" the step; direction may be ~orthogonal to memo-mitigation -> lr-flat)")
    sd.unet.half(); sd.dtype = torch.float16
    print("Done.")


if __name__ == "__main__":
    main()
