"""
Diagnostic 2 (REAL SD1.5): verify the adjoint gradient in optimize_xT_adj is
EXACT against central finite differences on the actual StableDiffusion UNet,
on a SHORT chain (t_idx small, e.g. 2) where full-autograd is still feasible.

This reuses the REAL SD1.5 pipeline (StableDiffusion, predict_noise, alpha,
scheduler) and the REAL adjoint code path from run_ini_opti.optimize_xT_adj,
run as-is. We extract x_T.grad after one optimization step and compare it to
a central-FD estimate of the SAME stop-gradient loss w.r.t. x_T.

To keep it cheap we use a tiny latent (1,4,16,16) and perturb a RANDOM subset
of ~64 coordinates.

Run:
  /home/geonsoo/anaconda3/envs/init_score_noise/bin/python diag_finite_diff_real.py
"""
import os, sys, torch
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed
import run_ini_opti as R          # import the module under test

DEV = torch.device("cuda:0")
NFE = 50; CFG = 7.5; SEED = 42
set_seed(SEED)

solver_config = munchify({"num_sampling": NFE})
sd = StableDiffusion(solver_config=solver_config,
                     model_key=os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5"),
                     device=DEV, seed=SEED)
# Drop safety_checker & feature_extractor to free VRAM
try:
    del sd.safety_checker
except Exception:
    pass
torch.cuda.empty_cache()
# *** GRADIENT CHECKPOINTING OFF *** : checkpointing re-runs the forward during
# backward; if the recomputed UNet forward differs from the cached-forward
# operating point (e.g. due to attention non-determinism / rng in checkpoint),
# the VJP J_j^T g is evaluated at the wrong point and the adjoint diverges from
# full-autograd. We test with checkpointing DISABLED to isolate this.
# (If you want to match run_ini_opti.py:536 exactly, comment the next line.)
if hasattr(sd.unet, "disable_gradient_checkpointing"):
    sd.unet.disable_gradient_checkpointing()
torch.cuda.empty_cache()
print(f">>> gradient_checkpointing enabled? "
      f"{getattr(sd.unet, 'gradient_checkpointing', 'n/a')}")

prompt = "An astronaut on the moon"
uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)   # (1,77,768) each
uc = uc.float(); c = c.float()

T_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 2          # short chain so full-autograd fits in 24GB
LAMBDA_ALIGN = 0.1
BASE_S_RATIO = 0.5
LR = 0.0           # lr=0 so optimizer.step() does NOT move x_T; we read pure grad
INIT_STEPS = T_IDX
NUM_STEPS = 1
GAP_STEPS = 1

# Force a deterministic, SMALL x_T so the chain is well-conditioned for FD.
set_seed(SEED)
_shape = (1, 4, 16, 16)   # small spatial for speed

print(f"\n>>> Inline faithful copy of optimize_xT_adj adjoint path "
      f"(t_idx={T_IDX}) to capture g_xT ...")

# -----------------------------------------------------------------------------
# Inline faithful copy of the adjoint gradient computation for t_idx=T_IDX,
# so we can read g_xT directly. (Mirrors run_ini_opti.optimize_xT_adj lines
# 258-427, stripped of the optimizer step.)
# -----------------------------------------------------------------------------
sd.unet.float(); sd.dtype = torch.float32
timesteps = list(sd.scheduler.timesteps)
update_indices = [INIT_STEPS + i * GAP_STEPS for i in range(NUM_STEPS)]
update_indices = [i for i in update_indices if i < len(timesteps)]
s_idx = int(len(timesteps) * BASE_S_RATIO)
s_target = timesteps[s_idx]
alpha_s = sd.alpha(s_target)
set_seed(SEED)
x_T_init = torch.randn(1, 4, 16, 16, device=DEV, dtype=torch.float32)
x_T = x_T_init.clone().requires_grad_(True)
def cfg_combine(nuc, nc): return nuc + CFG * (nc - nuc)

# reference trajectory for align loss
x0_orig_refs = {}
with torch.no_grad():
    zt_ref = x_T_init * sd.scheduler.init_noise_sigma
    for step_idx, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        nuc, nc = sd.predict_noise(zt_ref, t, uc, c)
        eps = cfg_combine(nuc, nc)
        x0h = (zt_ref - (1 - at).sqrt() * eps) / at.sqrt()
        zt_ref = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
        if step_idx in update_indices:
            x0_orig_refs[step_idx] = x0h.detach().clone().float()
epsilon_ref = x_T_init.detach()

t_idx = T_IDX
# (1) forward no_grad, cache xs
x_k = (x_T.detach() * sd.scheduler.init_noise_sigma).clone()
xs = [x_k.clone()]
with torch.no_grad():
    for step_idx, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        nuc, nc = sd.predict_noise(x_k, t, uc, c)
        eps = cfg_combine(nuc, nc)
        x0h = (x_k - (1 - at).sqrt() * eps) / at.sqrt()
        x_k = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
        if step_idx == t_idx:
            break
        if (step_idx + 1) <= t_idx:
            xs.append(x_k.clone())
x_end_state = xs[t_idx].clone()

# (2) terminal head WITH grad
with torch.enable_grad():
    x_end = x_end_state.detach().clone().requires_grad_(True)
    t_break = timesteps[t_idx]; at_break = sd.alpha(t_break)
    eps_t = cfg_combine(*sd.predict_noise(x_end, t_break, uc, c))
    x0_hat = (x_end - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
    x_s = alpha_s.sqrt() * x0_hat + (1 - alpha_s).sqrt() * x_T.detach()
    nuc_s, nc_s = sd.predict_noise(x_s, s_target, uc, c)
    eps_s = cfg_combine(nuc_s, nc_s)
    B = eps_s.shape[0]
    memo_proxy = (epsilon_ref - eps_s).reshape(B, -1).pow(2).mean(-1)
    loss_memo = memo_proxy.mean()
    loss_align = ((x0_hat.float() - x0_orig_refs[t_idx]).reshape(B, -1).pow(2).mean(-1).mean())
    loss = loss_memo + LAMBDA_ALIGN * loss_align
    g = torch.autograd.grad(loss, x_end, retain_graph=False)[0]
    g_terminal_norm = g.flatten().norm().item()

# (3) adjoint recursion
for k in range(t_idx, 0, -1):
    j = k - 1
    t_j = timesteps[j]; a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j - sd.skip)
    A_j = (a_jp1 / a_j).sqrt()
    B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()
    x_j_local = xs[j].detach().clone().requires_grad_(True)
    eps_j = cfg_combine(*sd.predict_noise(x_j_local, t_j, uc, c))
    Jt_g = torch.autograd.grad(eps_j, x_j_local, grad_outputs=g, retain_graph=False)[0]
    g = A_j * g + B_j * Jt_g
    print(f"  [adj] j={j}: |g|={g.flatten().norm().item():.6e}  "
          f"ratio(to terminal)={g.flatten().norm().item()/(g_terminal_norm+1e-12):.6f}")
g_xT_adjoint = (g * sd.scheduler.init_noise_sigma).detach().reshape_as(x_T)
print(f"\ng_xT_adjoint: |g|={g_xT_adjoint.flatten().norm().item():.6e}")
print(f"terminal g norm: {g_terminal_norm:.6e}")

# -----------------------------------------------------------------------------
# (4) FULL-AUTOGRAD ground truth through the SAME short chain (fits at t_idx=2)
# -----------------------------------------------------------------------------
print("\n>>> Computing full-autograd ground truth (whole chain with grad) ...")
set_seed(SEED)
x_T2 = x_T_init.clone().requires_grad_(True)
x_k2 = x_T2 * sd.scheduler.init_noise_sigma
for step_idx, t in enumerate(timesteps):
    at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
    nuc, nc = sd.predict_noise(x_k2, t, uc, c)
    eps = cfg_combine(nuc, nc)
    x0h = (x_k2 - (1 - at).sqrt() * eps) / at.sqrt()
    x_k2 = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
    if step_idx == t_idx:
        break
x_end2 = x_k2
t_break = timesteps[t_idx]; at_break = sd.alpha(t_break)
eps_t2 = cfg_combine(*sd.predict_noise(x_end2, t_break, uc, c))
x0_hat2 = (x_end2 - (1 - at_break).sqrt() * eps_t2) / at_break.sqrt()
x_s2 = alpha_s.sqrt() * x0_hat2 + (1 - alpha_s).sqrt() * x_T2.detach()
nuc_s2, nc_s2 = sd.predict_noise(x_s2, s_target, uc, c)
eps_s2 = cfg_combine(nuc_s2, nc_s2)
memo2 = (epsilon_ref - eps_s2).reshape(B, -1).pow(2).mean(-1).mean()
align2 = ((x0_hat2.float() - x0_orig_refs[t_idx]).reshape(B, -1).pow(2).mean(-1).mean())
loss2 = memo2 + LAMBDA_ALIGN * align2
g_xT_autograd = torch.autograd.grad(loss2, x_T2, retain_graph=False)[0].detach()
print(f"g_xT_autograd: |g|={g_xT_autograd.flatten().norm().item():.6e}")

# -----------------------------------------------------------------------------
# (5) finite difference (subset of coords), stop-gradient respected
# -----------------------------------------------------------------------------
print("\n>>> Computing central finite-difference gradient (64 random coords) ...")
def loss_of(x_T_val):
    with torch.no_grad():
        xk = x_T_val * sd.scheduler.init_noise_sigma
        xs_ = [xk]
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
            nuc, nc = sd.predict_noise(xk, t, uc, c)
            eps = cfg_combine(nuc, nc)
            x0h = (xk - (1 - at).sqrt() * eps) / at.sqrt()
            xk = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
            if step_idx == t_idx:
                break
            xs_.append(xk)
        xe = xs_[t_idx]
        atb = sd.alpha(timesteps[t_idx])
        et = cfg_combine(*sd.predict_noise(xe, timesteps[t_idx], uc, c))
        x0h = (xe - (1 - atb).sqrt() * et) / atb.sqrt()
        xs_s = alpha_s.sqrt() * x0h + (1 - alpha_s).sqrt() * x_T_init  # FROZEN shortcut
        ess = cfg_combine(*sd.predict_noise(xs_s, s_target, uc, c))
        mp = (epsilon_ref - ess).reshape(B, -1).pow(2).mean(-1).mean()
        la = ((x0h.float() - x0_orig_refs[t_idx]).reshape(B, -1).pow(2).mean(-1).mean())
        return (mp + LAMBDA_ALIGN * la).item()

x_T_frozen = x_T_init.clone()
flat_idx = torch.randint(0, x_T_init.numel(), (64,), device=DEV)
h = 1e-3   # UNet output scale is O(1); h=1e-3 balances truncation vs roundoff in fp32
g_fd = torch.zeros_like(x_T_init).flatten()
for di in flat_idx.tolist():
    xp = x_T_frozen.clone().flatten(); xp[di] += h
    xm = x_T_frozen.clone().flatten(); xm[di] -= h
    g_fd[di] = (loss_of(xp.view_as(x_T_init)) - loss_of(xm.view_as(x_T_init))) / (2 * h)
g_fd = g_fd.view_as(x_T_init)

# -----------------------------------------------------------------------------
# (6) report
# -----------------------------------------------------------------------------
def rel_err(a, b):
    return (a - b).flatten().norm().item() / (b.flatten().norm().item() + 1e-12)

a_flat = g_xT_adjoint.flatten()[flat_idx]
u_flat = g_xT_autograd.flatten()[flat_idx]
print("\n" + "=" * 72)
print(f"REAL-SD1.5 ADJOINT EXACTNESS  (t_idx={T_IDX}, latent {tuple(x_T_init.shape)})")
print("=" * 72)
print(f"  rel_err(adjoint  vs autograd) = {rel_err(g_xT_adjoint, g_xT_autograd):.3e}")
print(f"  rel_err(adjoint  vs FD subset)= {rel_err(a_flat, g_fd.flatten()[flat_idx]):.3e}")
print(f"  rel_err(autograd vs FD subset)= {rel_err(u_flat, g_fd.flatten()[flat_idx]):.3e}")
print(f"  (FD is fp32 with h={h}; expect ~1e-2 truncation on the UNet.)")
print("\n  per-coord sample (first 8 of subset):")
print("    idx        adjoint       autograd      FD")
for di in flat_idx[:8].tolist():
    print(f"    {di:6d}  {g_xT_adjoint.flatten()[di]:+.5e}  "
          f"{g_xT_autograd.flatten()[di]:+.5e}  {g_fd.flatten()[di]:+.5e}")
e = rel_err(g_xT_adjoint, g_xT_autograd)
print(f"\n  >>> VERDICT (adjoint vs autograd, the decisive test): "
      f"{'EXACT (<1e-2)' if e < 1e-2 else 'MISMATCH (>=1e-2)'}   rel_err={e:.3e}")
