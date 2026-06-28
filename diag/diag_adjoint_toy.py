"""
Toy surrogate: adjoint recursion EXACTNESS on an arbitrary nonlinear eps_theta.

This isolates the adjoint algebra/indexing of optimize_xT_adj from GPU-memory
noise. The adjoint identity g_{j-1} = A_j g_j + B_j (J_j^T g_j) is a property
of the affine DDIM recursion x_{j+1} = A_j x_j + B_j eps_theta(x_j); it holds
for ANY differentiable eps_theta. So we substitute a small random conv-net for
eps_theta and compare the adjoint gradient to plain direct backprop on chains
of length 1, 2, 3 (and up to 10) -- all trivially feasible in CPU RAM.

If adjoint == direct here for a generic nonlinear eps_theta, the recursion
machinery is EXACT; any production-scale discrepancy is then attributable to
(a) recursion dynamics on the real long chain (vanishing), not a bug, or
(b) a real-UNet-only numerical issue, which diag_adjoint_vs_direct.py probes.

The coefficient/indexing logic is copied verbatim from run_ini_opti.py:356-399
and the forward cache from :258-281, with sd.alpha(t) replaced by a scalar
schedule so we can exercise the SAME indexing code path.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)


class ToyEpsNet(nn.Module):
    """Stands in for CFG-combined eps_theta: 4ch -> 4ch nonlinear map."""
    def __init__(self, ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, ch, 3, padding=1),
        )

    def forward(self, x, t):
        # ignore t in the toy (the real code passes t to the UNet; here the
        # Jacobian structure is what we test, and t-dependence doesn't change
        # the adjoint identity). Add a tiny t-dependent bias for realism.
        return self.net(x) + 0.01 * t * x


def make_schedule(n_total=1000, nfe=50):
    """Mimic sd.scheduler: 50 descending timesteps, skip = 1000//50 = 20,
    and a prepended-1.0 alpha table like latent_diffusion.py:113.

    alpha(t) returns alphas_cumprod[t] (with index 0 == 1.0 prepended), so
    alpha(t) for the raw DDIM index t maps to cumulative-prod alpha at t.
    We build a realistic cosine-ish schedule on [0, n_total].
    """
    skip = n_total // nfe
    # raw alphas_cumprod over 0..n_total (we prepend a 1.0 just like the code)
    steps = torch.arange(n_total + 1)
    # cosine schedule (scaled_beta) -> alphas_cumprod decreasing from ~1 to ~0
    acp = torch.cos(((steps / n_total) * torch.pi * 0.95)) ** 2
    acp = acp.clamp(min=1e-4)
    # prepend 1.0 exactly like latent_diffusion.py:113
    acp_full = torch.cat([torch.tensor([1.0]), acp])
    # sampling timesteps: descending from high to low (DDIM goes big->small t)
    # In diffusers DDIM, timesteps are descending. index into acp accordingly.
    # We mirror: timesteps[k] is a raw index; alpha(timesteps[k]) = acp_full[timesteps[k]].
    # Use evenly spaced descending raw indices.
    ts = torch.arange(n_total - 1, -1, -skip)  # descending
    ts = list(ts)
    return acp_full, ts, skip


def alpha(acp_full, t):
    """Mirror of sd.alpha(t): acp_full[t] (prepended-1.0 table)."""
    if t < 0:
        return acp_full[0]  # final_alpha_cumprod analog
    return acp_full[t]


def cfg_combine(n_uc, n_c, cfg=7.5):
    return n_uc + cfg * (n_c - n_uc)


def toy_predict_noise(epsnet, x, t, cfg=7.5):
    """Mimic predict_noise: returns (noise_uc, noise_c) from a batched call.
    Toy: two passes (uc and c) to mirror the real CFG structure."""
    noise_c = epsnet(x, t)
    noise_uc = epsnet(x.detach(), t).detach()  # uc path (stop-grad on input like real)
    return noise_uc, noise_c


# ---------------------------------------------------------------------------
# ADJOINT path: verbatim copy of run_ini_opti.py cache+head+recursion indexing
# ---------------------------------------------------------------------------
def grad_adjoint_toy(epsnet, x_T_init, t_idx, acp_full, ts, skip,
                     alpha_s, s_target, x0_orig_ref, lambda_align, cfg=7.5):
    x_T = x_T_init.clone().detach().requires_grad_(True)
    eps_ref = x_T_init.detach()

    # (1) forward cache (no_grad)
    x_k = (x_T.detach() * 1.0).clone()  # init_noise_sigma = 1.0 in toy
    xs = [x_k.clone()]
    with torch.no_grad():
        for step_idx, t in enumerate(ts):
            at = alpha(acp_full, t)
            at_prev = alpha(acp_full, t - skip)
            noise_uc, noise_c = toy_predict_noise(epsnet, x_k, t, cfg)
            eps_theta = cfg_combine(noise_uc, noise_c, cfg)
            x0_hat = (x_k - (1 - at).sqrt() * eps_theta) / at.sqrt()
            x_k = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta
            if step_idx == t_idx:
                break
            if (step_idx + 1) <= t_idx:
                xs.append(x_k.clone())
    x_end_state = xs[t_idx].clone() if len(xs) > t_idx else xs[0].clone()

    # (2) terminal head (enable_grad) -- exact copy of :305-335 indexing
    with torch.enable_grad():
        x_end = x_end_state.detach().clone().requires_grad_(True)
        t_break = ts[t_idx]
        at_break = alpha(acp_full, t_break)
        eps_t = cfg_combine(*toy_predict_noise(epsnet, x_end, t_break, cfg), cfg)
        x0_hat = (x_end - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
        x_s = (alpha_s.sqrt() * x0_hat + (1 - alpha_s).sqrt() * x_T.detach())
        eps_s = cfg_combine(*toy_predict_noise(epsnet, x_s, s_target, cfg), cfg)
        B = eps_s.shape[0]
        memo_proxy = (eps_ref - eps_s).reshape(B, -1).pow(2).mean(-1)
        loss_memo = memo_proxy.mean()
        loss_align = ((x0_hat - x0_orig_ref).reshape(B, -1).pow(2).mean(-1).mean())
        loss = loss_memo + lambda_align * loss_align
        g = torch.autograd.grad(loss, x_end, retain_graph=False)[0]

        # (3) adjoint recursion -- verbatim copy of :357-399 indexing/coeffs
        if t_idx == 0:
            g_final = g * 1.0
        else:
            for k in range(t_idx, 0, -1):
                j = k - 1
                t_j = ts[j]
                a_j = alpha(acp_full, t_j)
                a_jp1 = alpha(acp_full, t_j - skip)
                A_j = (a_jp1 / a_j).sqrt()
                B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()
                x_j_local = xs[j].detach().clone().requires_grad_(True)
                eps_j = cfg_combine(*toy_predict_noise(epsnet, x_j_local, t_j, cfg), cfg)
                Jt_g = torch.autograd.grad(eps_j, x_j_local,
                                           grad_outputs=g, retain_graph=False)[0]
                g = A_j * g + B_j * Jt_g
            g_final = g * 1.0  # * init_noise_sigma (=1)
    return g_final.detach(), loss.item()


# ---------------------------------------------------------------------------
# DIRECT backprop ground truth (full grad chain, feasible for toy)
# ---------------------------------------------------------------------------
def grad_direct_toy(epsnet, x_T_init, t_idx, acp_full, ts, skip,
                    alpha_s, s_target, x0_orig_ref, lambda_align, cfg=7.5):
    x_T = x_T_init.clone().detach().requires_grad_(True)
    eps_ref = x_T_init.detach()
    zt = x_T * 1.0
    x_at_tidx = None
    for step_idx, t in enumerate(ts):
        if step_idx == t_idx:
            x_at_tidx = zt
        at = alpha(acp_full, t)
        at_prev = alpha(acp_full, t - skip)
        noise_uc, noise_c = toy_predict_noise(epsnet, zt, t, cfg)
        eps_theta = cfg_combine(noise_uc, noise_c, cfg)
        x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()
        zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta
        if step_idx == t_idx:
            break
    t_break = ts[t_idx]
    at_break = alpha(acp_full, t_break)
    eps_b = cfg_combine(*toy_predict_noise(epsnet, x_at_tidx, t_break, cfg), cfg)
    x0_hat_head = (x_at_tidx - (1 - at_break).sqrt() * eps_b) / at_break.sqrt()
    x_s = (alpha_s.sqrt() * x0_hat_head + (1 - alpha_s).sqrt() * x_T.detach())
    eps_s = cfg_combine(*toy_predict_noise(epsnet, x_s, s_target, cfg), cfg)
    B = eps_s.shape[0]
    memo_proxy = (eps_ref - eps_s).reshape(B, -1).pow(2).mean(-1)
    loss_memo = memo_proxy.mean()
    loss_align = ((x0_hat_head - x0_orig_ref).reshape(B, -1).pow(2).mean(-1).mean())
    loss = loss_memo + lambda_align * loss_align
    g = torch.autograd.grad(loss, x_T, retain_graph=False)[0]
    return g.detach(), loss.item()


def main():
    acp_full, ts, skip = make_schedule()
    alpha_s = alpha(acp_full, ts[len(ts) // 2])  # base_s_ratio=0.5
    s_target = ts[len(ts) // 2]
    lambda_align = 0.1
    cfg = 7.5

    epsnet = ToyEpsNet(ch=4).cpu()
    for p in epsnet.parameters():
        p.requires_grad_(False)

    x_T_init = torch.randn(1, 4, 8, 8)

    # reference x0_orig_ref per t_idx (no-grad trajectory)
    x0_orig_refs = {}
    with torch.no_grad():
        zt = x_T_init * 1.0
        for step_idx, t in enumerate(ts):
            at = alpha(acp_full, t)
            at_prev = alpha(acp_full, t - skip)
            n_uc, n_c = toy_predict_noise(epsnet, zt, t, cfg)
            eps_theta = cfg_combine(n_uc, n_c, cfg)
            x0h = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()
            zt = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps_theta
            x0_orig_refs[step_idx] = x0h.detach().clone()

    print("=== TOY surrogate: adjoint vs direct-backprop (CPU, exact) ===")
    print(f"{'t_idx':>5} {'|g_direct|':>12} {'|g_adjoint|':>12} "
          f"{'rel_err':>12} {'cos_sim':>10}  verdict")
    for t_idx in [1, 2, 3, 5, 8]:
        if t_idx >= len(ts):
            continue
        g_d, ld = grad_direct_toy(epsnet, x_T_init, t_idx, acp_full, ts, skip,
                                  alpha_s, s_target, x0_orig_refs[t_idx],
                                  lambda_align, cfg)
        g_a, la = grad_adjoint_toy(epsnet, x_T_init, t_idx, acp_full, ts, skip,
                                   alpha_s, s_target, x0_orig_refs[t_idx],
                                   lambda_align, cfg)
        gd, ga = g_d.flatten(), g_a.flatten()
        rel_err = (gd - ga).norm().item() / (gd.norm().item() + 1e-12)
        cos = torch.dot(gd, ga).item() / (gd.norm().item() * ga.norm().item() + 1e-12)
        verdict = "EXACT" if (rel_err < 1e-3 and cos > 0.9999) else "MISMATCH"
        print(f"{t_idx:>5} {gd.norm().item():>12.4e} {ga.norm().item():>12.4e} "
              f"{rel_err:>12.4e} {cos:>10.6f}  {verdict}")
        print(f"       loss_direct={ld:.6f}  loss_adjoint={la:.6f}")


if __name__ == "__main__":
    main()
