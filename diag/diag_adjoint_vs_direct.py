"""
Diagnostic: adjointDPM vs direct-autograd ground-truth on SHORT chains.

Perspective 6 (adjoint-vs-direct-backprop).
For t_idx in {1,2,3} (short chains where direct backprop is exact and does NOT
vanish), compute dL/dx_T two independent ways and compare:

  (A) DIRECT (ground truth): run the full DDIM chain under enable_grad from a
      leaf x_T, compute the SAME memo+align loss head, then
      torch.autograd.grad(loss, x_T).  This materializes the exact Jacobian
      product -- on a 1-3 step chain it is NOT vanishing, so it is the truth.

  (B) ADJOINT: replicate optimize_xT_adj's machinery (latent cache +
      terminal 2-UNet head via torch.autograd.grad + reverse recursion
      A_j*g + B_j*(J_j^T g)) and read off g*init_noise_sigma.

Both paths MUST share: identical x_T_init, identical t_idx, identical loss
formula (memo_proxy ||eps_ref - eps_s||^2 + lambda_align*||x0_hat-x0_orig||^2),
identical stop-gradients (eps_ref = x_T_init.detach(); x_T.detach() in x_s).

Verdict rule:
  relative error < 1e-2 AND cos sim > 0.999  -> adjoint EXACT (short chain OK)
  else                                          -> adjoint has an indexing/
                                                 algebra bug that will also
                                                 corrupt long chains.

Run:
  python diag_adjoint_vs_direct.py --device cuda:0
"""

import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse
import copy
import torch
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed


# ----------------------------------------------------------------------------
# (A) DIRECT backprop ground truth: leaf x_T -> full grad chain -> loss -> grad
# ----------------------------------------------------------------------------
def _ckpt_cfg_noise(sd, zt, t, uc, c, cfg):
    """CFG-combined epsilon_theta with activation checkpointing so the chain
    graph does NOT retain per-UNet activations. Gradient stays EXACT
    (checkpointing recomputes forward in backward; numerically identical)."""
    t_in = t.unsqueeze(0) if len(t.shape) == 0 else t
    c_embed = torch.cat([uc, c], dim=0)
    z_in = torch.cat([zt] * 2)
    t_in = t_in.expand(zt.shape[0])
    t_in = torch.cat([t_in] * 2)

    def run(_z, _t, _c):
        return sd.unet(_z, _t, encoder_hidden_states=_c)['sample']

    noise_pred = torch.utils.checkpoint.checkpoint(run, z_in, t_in, c_embed,
                                                   use_reentrant=False)
    noise_uc, noise_c = noise_pred.chunk(2)
    return noise_uc + cfg * (noise_c - noise_uc)


def grad_direct(sd, uc, c, cfg, device, x_T_init, t_idx,
                base_s_ratio, lambda_align, x0_orig_ref):
    """Exact dL/dx_T via direct backprop through the (short) DDIM chain.

    Each UNet call is activation-checkpointed so the full-chain graph fits in
    memory; checkpointing recomputes forward during backward and yields the
    EXACT same gradient as plain backprop (no approximation).

    Captures the input tensor to the UNet at step t_idx (x_{t_idx}) and builds
    the loss head from it exactly as optimize_xT line 87-99 does.
    """
    sd.unet.float()
    sd.dtype = torch.float32
    uc = uc.float()
    c = c.float()

    timesteps = list(sd.scheduler.timesteps)
    s_idx = int(len(timesteps) * base_s_ratio)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)

    x_T = x_T_init.clone().detach().requires_grad_(True)
    eps_ref = x_T_init.detach()  # fixed reference (stop-grad)

    # forward chain (WITH grad) up to t_idx, capturing pre-step input at t_idx
    zt = x_T * sd.scheduler.init_noise_sigma
    x_at_tidx = None
    for step_idx, t in enumerate(timesteps):
        if step_idx == t_idx:
            x_at_tidx = zt  # input to UNet at step t_idx (= x_{t_idx})
        at = sd.alpha(t)
        at_prev = sd.alpha(t - sd.skip)
        eps_theta = _ckpt_cfg_noise(sd, zt, t, uc, c, cfg)
        x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()
        zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta
        if step_idx == t_idx:
            break

    # Build the SAME loss head as optimize_xT line 87-99 from x_at_tidx.
    t_break = timesteps[t_idx]
    at_break = sd.alpha(t_break)
    eps_b = _ckpt_cfg_noise(sd, x_at_tidx, t_break, uc, c, cfg)
    x0_hat_head = (x_at_tidx - (1 - at_break).sqrt() * eps_b) / at_break.sqrt()

    x_s = (alpha_s.sqrt().to(sd.dtype) * x0_hat_head
           + (1 - alpha_s).sqrt().to(sd.dtype) * x_T.detach().to(sd.dtype))
    eps_s = _ckpt_cfg_noise(sd, x_s, s_target, uc, c, cfg)

    B = eps_s.shape[0]
    memo_proxy = (eps_ref.to(sd.dtype) - eps_s).reshape(B, -1).pow(2).mean(-1)
    loss_memo = memo_proxy.mean()
    loss_align = ((x0_hat_head.float() - x0_orig_ref)
                  .reshape(x0_hat_head.shape[0], -1).pow(2).mean(-1).mean())
    loss = loss_memo + lambda_align * loss_align

    g = torch.autograd.grad(loss, x_T, retain_graph=False)[0]
    return g.detach(), loss.item()


# ----------------------------------------------------------------------------
# (B) ADJOINT: replicate optimize_xT_adj machinery (cache + head + recursion)
# ----------------------------------------------------------------------------
def grad_adjoint(sd, uc, c, cfg, device, x_T_init, t_idx,
                 base_s_ratio, lambda_align, x0_orig_ref):
    """Adjoint dL/dx_T: forward no_grad cache + 2-UNet terminal head +
    reverse adjoint recursion. Replicates run_ini_opti.py:247-427 logic."""
    sd.unet.float()
    sd.dtype = torch.float32
    uc = uc.float()
    c = c.float()

    timesteps = list(sd.scheduler.timesteps)
    s_idx = int(len(timesteps) * base_s_ratio)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)

    x_T = x_T_init.clone().detach().requires_grad_(True)  # leaf (we only set .grad)
    eps_ref = x_T_init.detach()

    def cfg_combine(n_uc, n_c):
        return n_uc + cfg * (n_c - n_uc)

    # ---- (1) FORWARD: cache latents under no_grad ----
    x_k = (x_T.detach() * sd.scheduler.init_noise_sigma).clone()
    xs = [x_k.clone()]
    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t)
            at_prev = sd.alpha(t - sd.skip)
            noise_uc, noise_c = sd.predict_noise(x_k, t, uc, c)
            eps_theta = cfg_combine(noise_uc, noise_c)
            x0_hat = (x_k - (1 - at).sqrt() * eps_theta) / at.sqrt()
            x_k = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta
            if step_idx == t_idx:
                break
            if (step_idx + 1) <= t_idx:
                xs.append(x_k.clone())
    if len(xs) > t_idx:
        x_end_state = xs[t_idx].clone()
    else:
        # t_idx==0 edge: xs[0] is x_0
        x_end_state = xs[0].clone()

    # ---- (2) TERMINAL HEAD (2 UNets, enable_grad) ----
    with torch.enable_grad():
        x_end = x_end_state.detach().clone().requires_grad_(True)
        t_break = timesteps[t_idx]
        at_break = sd.alpha(t_break)
        eps_t = cfg_combine(*sd.predict_noise(x_end, t_break, uc, c))
        x0_hat = (x_end - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
        x_s = (alpha_s.sqrt().to(sd.dtype) * x0_hat
               + (1 - alpha_s).sqrt().to(sd.dtype) * x_T.detach().to(sd.dtype))
        noise_uc_s, noise_c_s = sd.predict_noise(x_s, s_target, uc, c)
        eps_s = cfg_combine(noise_uc_s, noise_c_s)
        B = eps_s.shape[0]
        memo_proxy = (eps_ref.to(sd.dtype) - eps_s).reshape(B, -1).pow(2).mean(-1)
        loss_memo = memo_proxy.mean()
        loss_align = ((x0_hat.float() - x0_orig_ref)
                      .reshape(x0_hat.shape[0], -1).pow(2).mean(-1).mean())
        loss = loss_memo + lambda_align * loss_align

        g = torch.autograd.grad(loss, x_end, retain_graph=False)[0]
        g_terminal_norm = g.flatten().norm().item()

        # ---- (3) ADJOINT RECURSION t_idx -> 0 ----
        if t_idx == 0:
            g_final = g * sd.scheduler.init_noise_sigma
        else:
            for k in range(t_idx, 0, -1):
                j = k - 1
                t_j = timesteps[j]
                a_j = sd.alpha(t_j)
                a_jp1 = sd.alpha(t_j - sd.skip)
                A_j = (a_jp1 / a_j).sqrt()
                B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()

                x_j_local = xs[j].detach().clone().requires_grad_(True)
                eps_j = cfg_combine(*sd.predict_noise(x_j_local, t_j, uc, c))
                Jt_g = torch.autograd.grad(eps_j, x_j_local,
                                           grad_outputs=g, retain_graph=False)[0]
                g = A_j * g + B_j * Jt_g
            g_final = g * sd.scheduler.init_noise_sigma

    return g_final.detach(), loss.item()


def compare(g_direct, g_adjoint):
    gd = g_direct.flatten()
    ga = g_adjoint.flatten()
    rel_err = (gd - ga).norm().item() / (gd.norm().item() + 1e-12)
    cos = torch.dot(gd, ga).item() / (gd.norm().item() * ga.norm().item() + 1e-12)
    return rel_err, cos, gd.norm().item(), ga.norm().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--NFE", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--t_idx_list", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--base_s_ratio", type=float, default=0.5)
    p.add_argument("--lambda_align", type=float, default=0.1)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--model_key", type=str,
                   default=os.path.join(SCRIPT_DIR, "ckpt", "sd14_memor_LAION2B_40k"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--prompt", type=str,
                   default="An astronaut on the moon")
    args = p.parse_args()
    device = torch.device(args.device)

    solver_config = munchify({"num_sampling": args.NFE})
    sd = StableDiffusion(solver_config=solver_config, model_key=args.model_key,
                         device=device, seed=args.base_seed)
    sd.unet.enable_gradient_checkpointing()

    # text embed (batch=1 for diagnostic) -- compute BEFORE moving encoders off
    uc, c = sd.get_text_embed(null_prompt="", prompt=args.prompt)
    uc = uc.float().to(device)
    c = c.float().to(device)

    # move VAE + text encoder to CPU to free VRAM for the UNet autograd graph
    sd.vae.to("cpu")
    sd.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # shared x_T_init
    set_seed(args.base_seed)
    x_T_init = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
    init_noise_sigma = sd.scheduler.init_noise_sigma
    print(f"[setup] init_noise_sigma={init_noise_sigma:.6f}  "
          f"skip={sd.skip}  NFE={args.NFE}")
    print(f"[setup] x_T_init |.|={x_T_init.norm().item():.4f}")

    timesteps = list(sd.scheduler.timesteps)

    # switch UNet to fp32 BEFORE anything that calls predict_noise
    sd.unet.float()
    sd.dtype = torch.float32

    # precompute x0_orig_ref for each t_idx (no-grad reference trajectory)
    x0_orig_refs = {}
    with torch.no_grad():
        zt_ref = x_T_init.to(sd.dtype) * init_noise_sigma
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t)
            at_prev = sd.alpha(t - sd.skip)
            noise_uc, noise_c = sd.predict_noise(zt_ref, t, uc, c)
            eps_theta = noise_uc + args.cfg * (noise_c - noise_uc)
            x0_hat = (zt_ref - (1 - at).sqrt() * eps_theta) / at.sqrt()
            zt_ref = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta
            if step_idx in args.t_idx_list:
                x0_orig_refs[step_idx] = x0_hat.detach().clone().float()

    print("\n=== adjoint vs direct-backprop (short chain) ===")
    print(f"{'t_idx':>5} {'|g_direct|':>12} {'|g_adjoint|':>12} "
          f"{'rel_err':>12} {'cos_sim':>10}  verdict")
    results = []
    for t_idx in args.t_idx_list:
        sd.unet.float(); sd.dtype = torch.float32
        g_d, loss_d = grad_direct(sd, uc, c, args.cfg, device, x_T_init, t_idx,
                                  args.base_s_ratio, args.lambda_align,
                                  x0_orig_refs[t_idx])
        torch.cuda.empty_cache()
        sd.unet.float(); sd.dtype = torch.float32
        g_a, loss_a = grad_adjoint(sd, uc, c, args.cfg, device, x_T_init, t_idx,
                                   args.base_s_ratio, args.lambda_align,
                                   x0_orig_refs[t_idx])
        torch.cuda.empty_cache()
        rel_err, cos, nd, na = compare(g_d, g_a)
        verdict = "EXACT" if (rel_err < 1e-2 and cos > 0.999) else "MISMATCH"
        results.append((t_idx, nd, na, rel_err, cos, loss_d, loss_a, verdict))
        print(f"{t_idx:>5} {nd:>12.4e} {na:>12.4e} "
              f"{rel_err:>12.4e} {cos:>10.6f}  {verdict}")
        print(f"       loss_direct={loss_d:.6f}  loss_adjoint={loss_a:.6f}  "
              f"(loss diff={abs(loss_d-loss_a):.2e})")

    # also: terminal-gradient-only comparison (g_terminal vs g at t_idx from direct)
    print("\n=== terminal-head-only sanity (g_terminal direct vs adjoint) ===")
    print("(if short-chain recursion EXACT but long chain vanishes, the recursion")
    print(" dynamics, not the head, explain the production lr-unresponsiveness.)")

    sd.unet.half()
    sd.dtype = torch.float16
    print("\nDone.")


if __name__ == "__main__":
    main()
