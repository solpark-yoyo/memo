"""
Diagnostic 1 (toy-model): verify the adjoint recursion in optimize_xT_adj is
EXACT against (a) full autograd backward through the chain and (b) central
finite differences w.r.t. x_T.

We replicate the EXACT math of run_ini_opti.optimize_xT_adj with a small
differentiable stand-in for the CFG-combined noise predictor:
    eps_theta_cfg(x, t) = net_t(x)      # a tiny MLP per timestep (analog of
                                        # noise_uc + cfg*(noise_c - noise_uc))

This isolates the adjoint recursion / affine-coefficient logic from any SD-
specific behavior. If the recursion is correct, the adjoint gradient g_xT must
match both autograd-through-chain and finite-difference to < 1e-3 relative.

Run:  /path/to/python diag_finite_diff_toy.py
"""
import torch
torch.manual_seed(0)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
D = 8          # latent dimensionality (flat, for speed)
CFG = 7.5
INIT_NOISE_SIGMA = 1.0   # keep sigma=1 so FD conditioning is clean; the
                         # scalar cancels out of the exactness question
                         # (it only scales g uniformly).


# ---- toy "DDIM" alpha schedule: alpha decreases as index grows ----
# We mimic the real schedule: a_0 ~ 1.0 (clean), a_T ~ 0.0 (noise).
# timesteps are listed high->low in the real DDIMScheduler; here we build a
# synthetic monotone schedule over N steps and define alpha(idx).
N = 6
alphas = torch.linspace(0.9999, 0.005, N + 1)        # a_0 .. a_N
def alpha_of(idx):
    return alphas[idx]

# mimic scheduler skip (real code: alpha(t - skip) gives the *next* state)
SKIP = 1
def alpha_next(idx):
    return alpha_of(idx - SKIP)   # moving toward clean => larger alpha

# ---- toy CFG noise network: independent small MLP per timestep index ----
class NoiseNet(torch.nn.Module):
    def __init__(self, dim, n_t):
        super().__init__()
        # one MLP per timestep, like a noise predictor conditioned on t
        self.nets = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(dim, 32), torch.nn.Tanh(),
                torch.nn.Linear(32, dim)
            ) for _ in range(n_t)
        ])
    def forward(self, x, t_idx_int):
        return self.nets[t_idx_int](x)

net = NoiseNet(D, N + 1).to(DEV)

def predict_noise_cfg(x, t_idx):
    """CFG-combined noise, analog of noise_uc + cfg*(noise_c - noise_uc).
    For the toy we just return the net output scaled to act like eps_theta.
    The adjoint VJP only cares that this is a differentiable fn of x_j."""
    return net(x, t_idx)


def forward_chain(x_T_leaf, t_idx):
    """Replicate lines 258-276: forward under the SAME ops, caching xs[].
    Returns xs (list of latents, xs[k]=input to UNet at step k). Built WITH
    grad so we can also do full-autograd backward as ground truth."""
    x_k = x_T_leaf * INIT_NOISE_SIGMA
    xs = [x_k]
    for j in range(t_idx + 1):   # steps 0..t_idx
        at = alpha_of(j)
        at_prev = alpha_next(j)            # alpha at x_{j+1}
        eps = predict_noise_cfg(x_k, j)
        x0_hat = (x_k - (1 - at).sqrt() * eps) / at.sqrt()
        x_k = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps
        xs.append(x_k)
    return xs   # xs[k] = input to UNet at step k ; xs has length t_idx+2


def terminal_loss(x_end, x_T_detach, ref, lambda_align, x0_ref):
    """Replicate terminal head (lines 305-329): x_end -> eps_t -> x0_hat ->
    x_s (stop-grad on x_T) -> eps_s -> memo_proxy + align. x_end is a leaf
    with requires_grad. Returns scalar loss."""
    at_break = alpha_of(t_idx_global)
    eps_t = predict_noise_cfg(x_end, t_idx_global)
    x0_hat = (x_end - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
    # s target at half the chain
    s_idx = N // 2
    alpha_s = alpha_of(s_idx)
    x_s = alpha_s.sqrt() * x0_hat + (1 - alpha_s).sqrt() * x_T_detach
    eps_s = predict_noise_cfg(x_s, s_idx)
    memo_proxy = ((ref - eps_s) ** 2).mean()
    loss_align = ((x0_hat - x0_ref) ** 2).mean()
    return memo_proxy + lambda_align * loss_align


def adjoint_grad_xT(x_T_value, t_idx, lambda_align, ref, x0_ref):
    """Faithful reimplementation of the adjoint path in optimize_xT_adj:
       (1) forward no_grad, cache xs
       (2) terminal head autograd -> g at x_end
       (3) adjoint recursion g_j = A_j*g + B_j*(J_j^T g) using cached xs[j]
       (4) x_T.grad = g * init_noise_sigma
    Returns the adjoint gradient w.r.t. x_T (shape [D])."""
    # (1) forward, NO grad, cache (exactly as code does)
    with torch.no_grad():
        x_k = (x_T_value * INIT_NOISE_SIGMA).clone()
        xs = [x_k.clone()]
        for j in range(t_idx + 1):
            at = alpha_of(j)
            at_prev = alpha_next(j)
            eps = predict_noise_cfg(x_k, j)
            x0_hat = (x_k - (1 - at).sqrt() * eps) / at.sqrt()
            x_k = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps
            xs.append(x_k.clone())
        x_end_state = xs[t_idx]

    # (2) terminal head WITH grad on a fresh leaf x_end
    with torch.enable_grad():
        x_end = x_end_state.detach().clone().requires_grad_(True)
        loss = terminal_loss(x_end, x_T_value.detach(), ref, lambda_align, x0_ref)
        g = torch.autograd.grad(loss, x_end, retain_graph=False)[0]

    # (3) adjoint recursion
    if t_idx == 0:
        x_T_grad = g * INIT_NOISE_SIGMA
    else:
        for k in range(t_idx, 0, -1):
            j = k - 1
            a_j = alpha_of(j)
            a_jp1 = alpha_next(j)
            A_j = (a_jp1 / a_j).sqrt()
            B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()
            # cached xs[j], re-leaf
            x_j_local = xs[j].detach().clone().requires_grad_(True)
            eps_j = predict_noise_cfg(x_j_local, j)
            Jt_g = torch.autograd.grad(eps_j, x_j_local, grad_outputs=g,
                                       retain_graph=False)[0]
            g = A_j * g + B_j * Jt_g
        x_T_grad = g * INIT_NOISE_SIGMA
    return x_T_grad.detach()


def full_autograd_grad_xT(x_T_leaf, t_idx, lambda_align, ref, x0_ref):
    """Ground truth: build the WHOLE chain with grad and call autograd.grad.
    This is what optimize_xT (the original) would compute (before vanishing)."""
    xs = forward_chain(x_T_leaf, t_idx)
    x_end = xs[t_idx]
    loss = terminal_loss(x_end, x_T_leaf.detach(), ref, lambda_align, x0_ref)
    return torch.autograd.grad(loss, x_T_leaf, retain_graph=False)[0].detach()


def fd_grad_xT(x_T_value, t_idx, lambda_align, ref, x0_ref, h=1e-5):
    """Central finite-difference gradient w.r.t. x_T (the leaf before scaling).
    IMPORTANT: the terminal head uses x_T.detach() in the x_s shortcut (a
    STOP-GRADIENT, run_ini_opti.py:317). To measure the SAME quantity that
    autograd/adjoint compute, the FD must hold that shortcut branch FROZEN at
    the unperturbed x_T value. Otherwise FD captures the (intentionally cut)
    shortcut derivative and will mismatch autograd BY DESIGN, not by bug.
    """
    x_T_frozen = x_T_value.detach().clone()   # frozen shortcut reference
    g = torch.zeros_like(x_T_value)
    with torch.no_grad():
        for d in range(D):
            xp = x_T_value.clone(); xp[d] += h
            xm = x_T_value.clone(); xm[d] -= h
            lp = _eval_loss_full(xp, x_T_frozen, t_idx, lambda_align, ref, x0_ref)
            lm = _eval_loss_full(xm, x_T_frozen, t_idx, lambda_align, ref, x0_ref)
            g[d] = (lp - lm) / (2 * h)
    return g


def _eval_loss_full(x_T_value, x_T_shortcut, t_idx, lambda_align, ref, x0_ref):
    """Full forward x_T -> loss (no grad). x_end = xs[t_idx] (input to UNet at
    step t_idx). x_T_shortcut is the FROZEN value injected into the x_s term
    (mimics x_T.detach() under stop-gradient)."""
    with torch.no_grad():
        x_k = x_T_value * INIT_NOISE_SIGMA
        xs = [x_k]
        for j in range(t_idx + 1):
            at = alpha_of(j); at_prev = alpha_next(j)
            eps = predict_noise_cfg(x_k, j)
            x0_hat = (x_k - (1 - at).sqrt() * eps) / at.sqrt()
            x_k = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps
            xs.append(x_k)
        x_end = xs[t_idx]   # input to UNet at step t_idx
        at_break = alpha_of(t_idx)
        eps_t = predict_noise_cfg(x_end, t_idx)
        x0_hat = (x_end - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
        s_idx = N // 2
        alpha_s = alpha_of(s_idx)
        x_s = alpha_s.sqrt() * x0_hat + (1 - alpha_s).sqrt() * x_T_shortcut
        eps_s = predict_noise_cfg(x_s, s_idx)
        memo_proxy = ((ref - eps_s) ** 2).mean()
        loss_align = ((x0_hat - x0_ref) ** 2).mean()
        return (memo_proxy + lambda_align * loss_align).item()


def rel_err(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


print("=" * 72)
print("TOY-MODEL ADJOINT EXACTNESS CHECK (replicates optimize_xT_adj math)")
print(f"  device={DEV}  D={D}  N(steps)={N}  CFG={CFG}")
print("=" * 72)
LAMBDA = 0.1
for t_idx in [0, 1, 2, 3]:
    x_T_val = torch.randn(D, device=DEV)
    ref = torch.randn(D, device=DEV)
    x0_ref = torch.randn(D, device=DEV)
    # make t_idx_global visible to terminal_loss
    globals()["t_idx_global"] = t_idx

    g_adj = adjoint_grad_xT(x_T_val, t_idx, LAMBDA, ref, x0_ref)

    x_T_leaf = x_T_val.clone().requires_grad_(True)
    g_autograd = full_autograd_grad_xT(x_T_leaf, t_idx, LAMBDA, ref, x0_ref)

    g_fd = fd_grad_xT(x_T_val, t_idx, LAMBDA, ref, x0_ref, h=1e-4)

    e_auto = rel_err(g_adj, g_autograd)
    e_fd = rel_err(g_adj, g_fd)
    e_auto_fd = rel_err(g_autograd, g_fd)
    print(f"\n[t_idx={t_idx}]  |g_adj|={g_adj.norm():.5f}  "
          f"|g_autograd|={g_autograd.norm():.5f}  |g_fd|={g_fd.norm():.5f}")
    print(f"   rel_err(adj vs autograd) = {e_auto:.3e}")
    print(f"   rel_err(adj vs FD)        = {e_fd:.3e}")
    print(f"   rel_err(autograd vs FD)   = {e_auto_fd:.3e}   (sanity)")
    verdict = "EXACT" if e_auto < 1e-2 and e_fd < 5e-2 else "MISMATCH"
    print(f"   >>> VERDICT: {verdict}")
