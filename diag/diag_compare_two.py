#!/usr/bin/env python
# diag_compare_two.py — Find WHY diag_decisive gives cos=1.0 but
# diag_tidx_sweep gives cos=-0.85 at t_idx=1. The ONLY difference
# must be in how the forward/cache is built. We run BOTH constructions
# side by side on the SAME x_T and compare the cached latents.
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


def main():
    sd = StableDiffusion(solver_config=munchify({"num_sampling": 50}),
                         model_key=MODEL_KEY, device=DEVICE, seed=42)
    sd.unet.float(); sd.dtype = torch.float32
    sd.unet.enable_gradient_checkpointing()
    timesteps = list(sd.scheduler.timesteps)
    sigma = float(sd.scheduler.init_noise_sigma)
    skip = sd.skip
    cfg = 7.5
    s_idx = int(len(timesteps) * 0.5)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)
    prompt = "An astronaut on the moon"
    uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)
    uc = uc.float(); c = c.float()

    torch.manual_seed(0)
    x_T = torch.randn(1, 4, 64, 64, device=DEVICE, dtype=torch.float32)
    epsilon_ref = x_T.detach().clone()
    t_idx = 1

    # ---- Construction A (diag_decisive style): single explicit step ----
    x_Tl = x_T.clone().requires_grad_(True)
    zt = x_Tl * sigma
    x_0_A = zt.detach().clone()
    t0 = timesteps[0]
    at0 = sd.alpha(t0); at0_prev = sd.alpha(t0 - skip)
    nuc, nc = sd.predict_noise(zt, t0, uc, c); eps = cfg_combine(nuc, nc, cfg)
    x0h = (zt - (1 - at0).sqrt() * eps) / at0.sqrt()
    zt_next_A = at0_prev.sqrt() * x0h + (1 - at0_prev).sqrt() * eps
    x_end_A = zt_next_A

    # ---- Construction B (sweep style): loop with break ----
    x_Tl2 = x_T.clone().requires_grad_(True)
    zt2 = x_Tl2 * sigma
    for step_idx, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - skip)
        nuc2, nc2 = sd.predict_noise(zt2, t, uc, c); eps2 = cfg_combine(nuc2, nc2, cfg)
        x0h2 = (zt2 - (1 - at).sqrt() * eps2) / at.sqrt()
        zt2 = at_prev.sqrt() * x0h2 + (1 - at_prev).sqrt() * eps2
        if step_idx == t_idx:
            x_end_B = zt2; t_break_B = t; at_break_B = at; break

    print("Forward comparison (TRUE path, two constructions):")
    print(f"  |x_end_A - x_end_B| = {(x_end_A - x_end_B).abs().max().item():.6e}")
    print(f"  t_break_B = {int(t_break_B)}  (should be timesteps[1]={int(timesteps[1])})")
    print(f"  at_break_B = {at_break_B.item():.6f}  vs alpha(ts[1])={sd.alpha(timesteps[1]).item():.6f}")

    # ---- Construction C (adjoint cache style) ----
    with torch.no_grad():
        xk = (x_T.detach() * sigma).clone(); cache = [xk.clone()]
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t); at_prev = sd.alpha(t - skip)
            nuc3, nc3 = sd.predict_noise(xk, t, uc, c); eps3 = cfg_combine(nuc3, nc3, cfg)
            x0h3 = (xk - (1 - at).sqrt() * eps3) / at.sqrt()
            xk = at_prev.sqrt() * x0h3 + (1 - at_prev).sqrt() * eps3
            if step_idx == t_idx: break
            if (step_idx + 1) <= t_idx: cache.append(xk.clone())
    print("\nCache comparison:")
    print(f"  len(cache) = {len(cache)}")
    print(f"  |cache[0] - x_0_A| = {(cache[0] - x_0_A).abs().max().item():.6e}")
    print(f"  |cache[1] - x_end_A| = {(cache[1] - x_end_A.detach()).abs().max().item():.6e}")
    print(f"  |cache[1] - x_end_B.detach()| = {(cache[1] - x_end_B.detach()).abs().max().item():.6e}")

    # CRITICAL: in the sweep, cache[t_idx] is the OUTPUT of step 0 (appended at step_idx 0).
    # But x_end_B is the output of step 1 (step_idx==1 break). These are DIFFERENT steps!
    # x_end_B = state AFTER step 1 (i.e., x_2), cache[1] = state AFTER step 0 (i.e., x_1).
    # So the sweep's "true" gradient is dL/dx_2 propagated, while adjoint starts at x_1.
    print("\n*** INDEX ANALYSIS ***")
    print(f"  cache[t_idx=1] = cache[1] = state after forward step 0 = x_{{timesteps[1] input}}")
    print(f"  x_end_B (TRUE break at step_idx==1) = state AFTER step 1 = x_{{timesteps[2] input}}")
    print(f"  => TRUE terminal head evaluates at timesteps[t_idx]=timesteps[1]={int(timesteps[1])}")
    print(f"     but x_end_B is the latent that was produced by step_idx=1 using timesteps[1]={int(timesteps[1])}")
    print(f"     so x_end_B is the INPUT to step 2, NOT step 1!")
    print(f"  x_0_A is INPUT to step 0 (timesteps[0]={int(timesteps[0])}).")
    print(f"  zt_next_A (x_end_A) is OUTPUT of step 0 = INPUT to step 1 = x at timesteps[1].")
    print(f"  cache[1] = OUTPUT of step 0 = INPUT to step 1. Same as x_end_A. CONSISTENT.")

    sd.unet.half()


if __name__ == "__main__":
    main()
