#!/usr/bin/env python
"""
DECISIVE VERIFICATION of the prior recursion-dynamics diagnosis.

Prior claim to test:
  (1) adjoint recursion is mathematically faithful (A_j>1 => amplifying, not vanishing)
  (2) cos(g_xT, g_terminal) ~ 0.32 (68% direction rotation)
  (3) ROOT CAUSE is NOT recursion but UPSTREAM: g_terminal is prompt-blind
      (early-step alpha_t~0.044, SNR~0.046 => x0_hat is garbage)

This script runs the THREE decisive tests:
  TEST 1 (correctness): adjoint g_xT vs gold full-backprop g_xT.
       cos~1.0 => recursion is exact (no recursion bug). [confirms prior claim (1)]
  TEST 2 (prompt-blindness of TERMINAL): cos(g_terminal_A, g_terminal_B) across 2 prompts.
       cos>0.9 => g_terminal is prompt-blind => root cause is UPSTREAM. [prior claim (3)]
       cos<0.5 => g_terminal IS prompt-discriminative => recursion rotation is the culprit.
  TEST 3 (prompt-blindness of FINAL x_T gradient): cos(g_xT_A, g_xT_B) across 2 prompts.
       If TEST2 shows blindness but TEST3 does not, recursion somehow restores signal
       (contradicts prior claim). If both blind, root cause is firmly upstream.

Also tests 3+ prompt pairs for robustness, and a memorized vs non-memorized contrast.

Run: CUDA_VISIBLE_DEVICES=0 python diag_verdict.py
"""
import os, sys, torch, gc
SCRIPT_DIR = "/home/geonsoo/Desktop/Datasets/Parksol/memo/ori_memo"
sys.path.insert(0, SCRIPT_DIR)
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed

device = torch.device("cuda:0")
NUM_SAMPLING = 50
CKPT = os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5")

set_seed(42)
sd = StableDiffusion(solver_config=munchify({"num_sampling": NUM_SAMPLING}),
                     model_key=CKPT, device=device, seed=42)
sd.unet.float()
sd.dtype = torch.float32
sd.unet.enable_gradient_checkpointing()

cfg = 7.5
timesteps = list(sd.scheduler.timesteps)
init_steps, gap_steps = 10, 3
t_idx = init_steps
s_idx = int(len(timesteps) * 0.5)
s_target = timesteps[s_idx]
alpha_s = sd.alpha(s_target)
alpha_t_break = sd.alpha(timesteps[t_idx])
print(f"[setup] skip={sd.skip}, t_idx={t_idx}, alpha_t={alpha_t_break.item():.4f}, "
      f"SNR_t={(alpha_t_break/(1-alpha_t_break)).item():.4f}")
print(f"[setup] s_target={s_target}, alpha_s={alpha_s.item():.4f}, "
      f"sqrt(a_s)={alpha_s.sqrt().item():.4f}, sqrt(1-a_s)={(1-alpha_s).sqrt().item():.4f}")
print(f"[setup] NOTE: x_s = {alpha_s.sqrt().item():.4f}*x0_hat + {(1-alpha_s).sqrt().item():.4f}*x_T.detach()")

# Fixed initial noise (same for all prompts — isolates the prompt effect)
x_T_init = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
epsilon_ref = x_T_init.detach()


def cfg_combine(nuc, noc):
    return nuc + cfg * (noc - nuc)


def run(prompt, seed=42):
    """Run forward cache + terminal head + adjoint recursion for one prompt.

    Returns: g_terminal, g_xT (adjoint final), g_xT_gold (full backprop), loss.
    Uses the SAME fixed x_T_init so the ONLY difference across prompts is the text embedding.
    """
    set_seed(seed)
    torch.cuda.empty_cache()
    uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)
    uc = uc.float(); c = c.float()

    # ---- (1) forward chain (no_grad), cache latents xs[] ----
    x_k = (x_T_init * sd.scheduler.init_noise_sigma).clone()
    xs = [x_k.clone()]
    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
            nuc, noc = sd.predict_noise(x_k, t, uc, c)
            eps = cfg_combine(nuc, noc)
            x0h = (x_k - (1 - at).sqrt() * eps) / at.sqrt()
            x_k = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
            if step_idx == t_idx:
                break
            if (step_idx + 1) <= t_idx:
                xs.append(x_k.clone())
    x_end_state = xs[t_idx].clone()

    # ---- reference x0_hat for align loss ----
    with torch.no_grad():
        at = alpha_t_break
        nuc, noc = sd.predict_noise(x_end_state, timesteps[t_idx], uc, c)
        eps_ref_traj = cfg_combine(nuc, noc)
        x0_ref = (x_end_state - (1 - at).sqrt() * eps_ref_traj) / at.sqrt()

    # ---- (2) terminal head (enable_grad) -> g_terminal ----
    with torch.enable_grad():
        x_end = x_end_state.detach().clone().requires_grad_(True)
        t_break = timesteps[t_idx]
        at_break = sd.alpha(t_break)
        eps_t = cfg_combine(*sd.predict_noise(x_end, t_break, uc, c))
        x0_hat = (x_end - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
        x_s = alpha_s.sqrt().to(sd.dtype) * x0_hat + (1 - alpha_s).sqrt().to(sd.dtype) * x_T_init.detach().to(sd.dtype)
        nuc_s, noc_s = sd.predict_noise(x_s, s_target, uc, c)
        eps_s = cfg_combine(nuc_s, noc_s)
        memo_proxy = (epsilon_ref.to(sd.dtype) - eps_s).reshape(1, -1).pow(2).mean(-1)
        loss_memo = memo_proxy.mean()
        loss_align = ((x0_hat.float() - x0_ref).reshape(1, -1).pow(2).mean(-1).mean())
        loss = loss_memo + 0.1 * loss_align
        g_terminal = torch.autograd.grad(loss, x_end, retain_graph=False)[0].detach()

    # ---- (3) adjoint recursion: g_terminal -> g_xT (short chain, t_idx=10) ----
    g = g_terminal.clone()
    for k in range(t_idx, 0, -1):
        j = k - 1
        t_j = timesteps[j]
        a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j - sd.skip)
        A_j = (a_jp1 / a_j).sqrt()
        B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()
        x_j_local = xs[j].detach().clone().requires_grad_(True)
        eps_j = cfg_combine(*sd.predict_noise(x_j_local, t_j, uc, c))
        Jt_g = torch.autograd.grad(eps_j, x_j_local, grad_outputs=g, retain_graph=False)[0]
        g = A_j * g + B_j * Jt_g
        del x_j_local, eps_j, Jt_g
    g_xT_adj = (g * sd.scheduler.init_noise_sigma).detach()

    # cleanup
    del xs, x_end_state, x_end, g, x0_hat, x_s, eps_s, nuc_s, noc_s, eps_t
    torch.cuda.empty_cache()
    return {
        "g_terminal": g_terminal,
        "g_xT_adj": g_xT_adj,
        "loss_memo": loss_memo.item(),
        "loss_align": loss_align.item(),
    }


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


# ======================================================================
# TEST 1: CORRECTNESS — adjoint g_xT vs gold full-backprop g_xT
# Run on a SHORT chain (t_idx=3) so full-backprop activations fit in VRAM.
# The point is: is the recursion mathematically exact? If exact at chain=3,
# it is exact at chain=10 (same code path, just more folds).
# ======================================================================
print("\n" + "=" * 80)
print("TEST 1: ADJOINT vs GOLD BACKPROP correctness (SHORT chain t_idx=3)")
print("=" * 80)
# temporarily override t_idx for correctness test only
_t_idx_orig = t_idx
_short = 3
def run_short(prompt, seed=42):
    """Short-chain correctness test. Self-contained to avoid clobbering run()."""
    set_seed(seed)
    torch.cuda.empty_cache()
    uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)
    uc = uc.float(); c = c.float()
    x_T_g = x_T_init.clone().requires_grad_(True)
    eps_ref_loc = x_T_init.detach()
    # forward WITH grad, cache states for adjoint AND gold at once
    zt = x_T_g.to(sd.dtype) * sd.scheduler.init_noise_sigma
    xs_loc = [zt.detach().clone()]
    for step_idx, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        nuc, noc = sd.predict_noise(zt, t, uc, c)
        eps = cfg_combine(nuc, noc)
        x0h = (zt - (1 - at).sqrt() * eps) / at.sqrt()
        zt = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
        if step_idx == _short:
            break
        if (step_idx + 1) <= _short:
            xs_loc.append(zt.detach().clone())
    x_end_state = xs_loc[_short].clone()
    with torch.no_grad():
        at = sd.alpha(timesteps[_short])
        nuc, noc = sd.predict_noise(x_end_state, timesteps[_short], uc, c)
        x0_ref = (x_end_state - (1 - at).sqrt() * cfg_combine(nuc, noc)) / at.sqrt()
    # terminal head on the grad-flowing state
    with torch.enable_grad():
        t_break = timesteps[_short]; at_break = sd.alpha(t_break)
        eps_t = cfg_combine(*sd.predict_noise(zt, t_break, uc, c))
        x0_hat = (zt - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
        x_s = alpha_s.sqrt().to(sd.dtype) * x0_hat + (1 - alpha_s).sqrt().to(sd.dtype) * x_T_g.detach().to(sd.dtype)
        nuc_s, noc_s = sd.predict_noise(x_s, s_target, uc, c)
        eps_s = cfg_combine(nuc_s, noc_s)
        memo = (eps_ref_loc.to(sd.dtype) - eps_s).reshape(1,-1).pow(2).mean(-1).mean()
        loss = memo + 0.1 * ((x0_hat.float() - x0_ref).reshape(1,-1).pow(2).mean(-1).mean())
        g_xT_gold = torch.autograd.grad(loss, x_T_g, retain_graph=False)[0].detach()
        g_terminal = torch.autograd.grad(loss, zt, retain_graph=False)[0]
    # NOW recompute terminal head on cached x_end (no-grad forward) to get g_terminal
    # from the SAME values (the adjoint path uses cached latents, not the grad-flowing zt).
    # For correctness, we compare adjoint(g_terminal_from_cache) to gold(g_xT).
    # Rebuild terminal head on cached leaf for adjoint g_terminal:
    with torch.enable_grad():
        xe = x_end_state.detach().clone().requires_grad_(True)
        eps_t2 = cfg_combine(*sd.predict_noise(xe, t_break, uc, c))
        x0h2 = (xe - (1-at_break).sqrt()*eps_t2)/at_break.sqrt()
        x_s2 = alpha_s.sqrt().to(sd.dtype)*x0h2 + (1-alpha_s).sqrt().to(sd.dtype)*x_T_init.detach().to(sd.dtype)
        nuc_s2,noc_s2 = sd.predict_noise(x_s2, s_target, uc, c)
        eps_s2 = cfg_combine(nuc_s2,noc_s2)
        memo2 = (eps_ref_loc.to(sd.dtype)-eps_s2).reshape(1,-1).pow(2).mean(-1).mean()
        loss2 = memo2 + 0.1*((x0h2.float()-x0_ref).reshape(1,-1).pow(2).mean(-1).mean())
        g_terminal_cached = torch.autograd.grad(loss2, xe, retain_graph=False)[0].detach()
    # adjoint recursion on cached latents
    g = g_terminal_cached.clone()
    for k in range(_short,0,-1):
        j = k-1; t_j = timesteps[j]
        a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j-sd.skip)
        A_j = (a_jp1/a_j).sqrt(); B_j = (1-a_jp1).sqrt()-(a_jp1*(1-a_j)/a_j).sqrt()
        xj = xs_loc[j].detach().clone().requires_grad_(True)
        epsj = cfg_combine(*sd.predict_noise(xj, t_j, uc, c))
        Jtg = torch.autograd.grad(epsj, xj, grad_outputs=g, retain_graph=False)[0]
        g = A_j*g + B_j*Jtg
        del xj, epsj, Jtg
    g_xT_adj = (g*sd.scheduler.init_noise_sigma).detach()
    del xs_loc, x_end_state, zt, xe, x0_hat, x0h2, x_s, x_s2, eps_s, eps_s2, x_T_g
    torch.cuda.empty_cache()
    return g_xT_adj, g_xT_gold, g_terminal_cached

r_adj, r_gold, r_term = run_short("An astronaut on the moon")
print(f"  chain length = {_short}")
print(f"  ||g_terminal (cached)||  = {r_term.norm().item():.6e}")
print(f"  ||g_xT (adjoint)||       = {r_adj.norm().item():.6e}")
print(f"  ||g_xT (gold full-bp)||  = {r_gold.norm().item():.6e}")
print(f"  cos(adj, gold)           = {cos(r_adj, r_gold):.6f}")
print(f"  rel_err ||adj-gold||/||gold|| = {((r_adj-r_gold).norm()/r_gold.norm()).item():.6e}")
print(f"  DECISION:")
print(f"    cos(adj,gold) ~ 1.0  => recursion is EXACT (prior claim (1) confirmed: no recursion bug).")
print(f"    cos(adj,gold) << 1   => recursion has a CACHE/INDEX/COEFF bug -> recursion IS a root cause.")
del r_adj, r_gold, r_term
gc.collect(); torch.cuda.empty_cache()

# ======================================================================
# TEST 2 + 3: PROMPT-BLINDNESS — g_terminal AND g_xT across multiple prompts
# The SAME fixed x_T_init is used; only the text embedding differs.
# ======================================================================
print("\n" + "=" * 80)
print("TEST 2 & 3: PROMPT-BLINDNESS (fixed x_T_init, only text embedding varies)")
print("=" * 80)
prompts = [
    ("astronaut", "An astronaut on the moon"),
    ("tiger", "Portrait of Tiger in black and white by Lukas Holas"),
    ("marvel", "Captain Marvel Exclusive Ccxp Poster Released Online By Marvel"),
    ("forest", "A serene forest with sunlight filtering through the trees"),
    ("city", "A futuristic cyberpunk city skyline at night with neon lights"),
]
results = {}
for name, p in prompts:
    print(f"\n  Running prompt: '{p}' ...")
    results[name] = run(p)
    gc.collect()
    torch.cuda.empty_cache()

print("\n  --- Per-prompt norms ---")
print(f"  {'prompt':>12} {'||g_term||':>12} {'||g_xT_adj||':>14} "
      f"{'cos(xT,term)':>13} {'loss_memo':>11}")
for name, _ in prompts:
    r = results[name]
    print(f"  {name:>12} {r['g_terminal'].norm().item():>12.4e} "
          f"{r['g_xT_adj'].norm().item():>14.4e} "
          f"{cos(r['g_xT_adj'], r['g_terminal']):>13.4f} {r['loss_memo']:>11.6f}")

print("\n  --- TEST 2: g_terminal prompt-blindness (pairwise cosine) ---")
names = [n for n, _ in prompts]
print(f"  {'':>12}", end="")
for n2 in names:
    print(f" {n2:>11}", end="")
print()
max_cos_term = 0.0
min_cos_term = 1.0
for n1 in names:
    print(f"  {n1:>12}", end="")
    for n2 in names:
        c = cos(results[n1]['g_terminal'], results[n2]['g_terminal'])
        if n1 != n2:
            max_cos_term = max(max_cos_term, c)
            min_cos_term = min(min_cos_term, c)
        print(f" {c:>11.4f}", end="")
    print()

print("\n  --- TEST 3: g_xT (adjoint FINAL) prompt-blindness (pairwise cosine) ---")
print(f"  {'':>12}", end="")
for n2 in names:
    print(f" {n2:>11}", end="")
print()
max_cos_xT = 0.0
min_cos_xT = 1.0
for n1 in names:
    print(f"  {n1:>12}", end="")
    for n2 in names:
        c = cos(results[n1]['g_xT_adj'], results[n2]['g_xT_adj'])
        if n1 != n2:
            max_cos_xT = max(max_cos_xT, c)
            min_cos_xT = min(min_cos_xT, c)
        print(f" {c:>11.4f}", end="")
    print()

print("\n" + "=" * 80)
print("VERDDECISION CRITERIA")
print("=" * 80)
print(f"  g_terminal cross-prompt cos: min={min_cos_term:.4f}  max={max_cos_term:.4f}")
print(f"  g_xT(adj)  cross-prompt cos: min={min_cos_xT:.4f}  max={max_cos_xT:.4f}")
print()
print("  CASE A (prior diagnosis CONFIRMED):")
print("    cos(adj,gold)~1.0 AND g_terminal cross-prompt cos>0.9")
print("    => recursion exact, g_terminal prompt-blind.")
print("    => ROOT CAUSE is UPSTREAM terminal-head (early-step SNR), NOT recursion.")
print()
print("  CASE B (prior diagnosis REFUTED — recursion/dynamics matters):")
print("    cos(adj,gold)~1.0 AND g_terminal cross-prompt cos<0.5 BUT g_xT cross>0.9")
print("    => g_terminal IS discriminative, but recursion ROTATES it to a common direction.")
print("    => recursion dynamics is the culprit (lr-insensitive because all prompts push x_T")
print("       the same way regardless of the memo signal).")
print()
print("  CASE C (recursion has a BUG):")
print("    cos(adj,gold) << 1.0")
print("    => the adjoint implementation is wrong; recursion itself is the root cause.")
print()
print("  CASE D (memo proxy itself carries no memo signal):")
print("    g_terminal cross-prompt cos>0.9 AND g_xT cross>0.9 AND ||g|| non-trivial")
print("    => the proxy ||eps_ref - eps_s||^2 is structurally blind (eps_ref=x_T_init makes")
print("       the target the same noise for all prompts, so the loss landscape is identical).")
