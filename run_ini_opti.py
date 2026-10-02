"""
ini_opti: Optimize x_T via ||ε - ε_s||² at multiple DDIM steps (starting from start_step).
Then DDIM inference with optimized x_T.

Example: init_steps=10, num_steps=4, gap_steps=3
  -> gradient at step 10, 13, 16, 19
"""

import sys, os, csv
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed
from torchvision.utils import save_image
from wgnc import project_wgnc_batched


def cfg_eff_at(sd, step_idx, cfg):
    """Staged CFG: denoising 초반(step_idx < cfg_start_ratio * NFE)은 null(unconditional)만
    사용하고, 그 이후 step부터 정상 CFG 계수를 적용.

      cfg_start_ratio = 0.0  →  항상 정상 CFG (기존 동작, opt-in 아님)
      cfg_start_ratio = 0.3  →  step 0~int(0.3*NFE)-1 까지 eps = noise_uc (null 만),
                                 그 이후 step 부터 eps = noise_uc + cfg·(noise_c - noise_uc)

    step_idx 기준:
      - DDIM forward/reference 의 eps_theta → 현재 denoising step_idx
      - memo proxy 의 eps_s(x_s, s_target)  → s_target 의 인덱스 s_idx
    """
    ratio = getattr(sd, "cfg_start_ratio", 0.0) or 0.0
    if ratio <= 0:
        return cfg
    total = len(sd.scheduler.timesteps)
    if step_idx < int(total * ratio):
        # cfgsr 구간: 기본 null(cfg=0, unconditional만).
        # --cfgsr_cond 켜면 conditional(cfg=1 → text prompt eps) 사용.
        return 1.0 if getattr(sd, "cfgsr_cond", False) else 0.0
    return cfg

MEMO_PROMPTS = {
    "astronaut_on_the_moon":  "An astronaut on the moon",
    "captain_marvel":         "Captain Marvel Exclusive Ccxp Poster Released Online By Marvel",
    "tiger_portrait":         "Portrait of Tiger in black and white by Lukas Holas",
}


def compute_memo_loss(memo_proxy, type_memo_loss="minimization", memo_threshold=0.0):
    """memo_proxy: (B,) per-sample memorization proxy = ||ε_ref - ε_s||²/D.

    스칼라 loss 반환 (MINIMIZE → memorization 완화).

    type_memo_loss:
      - "minimization": loss = mean(proxy)
          proxy 전체를 낮춘다 (기존 동작).
      - "threshold":    loss = mean(relu(proxy - memo_threshold))
          proxy > memo_threshold 인 샘플/구간만 gradient 를 흘린다.
          proxy ≤ memo_threshold 이면 loss=0 → 해당 부분은 update 하지 않음
          (memo_proxy 가 이미 충분히 작으면 더 이상 밀지 않음).
    """
    if type_memo_loss == "threshold":
        return torch.clamp(memo_proxy - memo_threshold, min=0.0).mean()
    return memo_proxy.mean()


def optimize_xT(sd, uc, c, cfg, device, init_steps, num_steps, gap_steps, lr, base_s_ratio, lambda_align,
                type_memo_loss="minimization", memo_threshold=0.0, batch_size=1):
    """Optimize x_T by applying gradient at [init_steps, init_steps+gap_steps, ...]"""

    # ---- fp32 전환: gradient가 10~19 UNet chain을 생존하도록 ----
    _orig_dtype = sd.dtype
    sd.unet.float()
    sd.dtype = torch.float32
    uc = uc.float()
    c = c.float()

    timesteps = list(sd.scheduler.timesteps)
    # print(f"timesteps: {len(timesteps)}")
    # update target step indices
    update_indices = [init_steps + i * gap_steps for i in range(num_steps)]
    update_indices = [i for i in update_indices if i < len(timesteps)]
    # print(f"update_indices: {update_indices}")
    # s target for memo_proxy
    s_idx = int(len(timesteps) * base_s_ratio)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)

    x_T_init = torch.randn(batch_size, 4, 64, 64, device=device, dtype=torch.float32)
    x_T = x_T_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x_T], lr=lr)

    # Pre-compute x̂₀_orig (reference trajectory without optimization) at each update step
    x0_orig_refs = {}
    with torch.no_grad():
        # print(f"x_T_init: {x_T_init.shape}")
        # print(f"sd.scheduler.init_noise_sigma: {sd.scheduler.init_noise_sigma}")
        zt_ref = x_T_init.to(sd.dtype) * sd.scheduler.init_noise_sigma
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t)
            at_prev = sd.alpha(t - sd.skip)
            noise_uc, noise_c = sd.predict_noise(zt_ref, t, uc, c)
            eps_theta = noise_uc + cfg_eff_at(sd, step_idx, cfg) * (noise_c - noise_uc)

            x0_hat = (zt_ref - (1 - at).sqrt() * eps_theta) / at.sqrt() # Tweedie formula

            zt_ref = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta # DDIM Denoising Step
            if step_idx in update_indices: 
                x0_orig_refs[step_idx] = x0_hat.detach().clone().float()

    total_loss = 0.0
    # ε reference = x_T 와 독립인 fresh random noise (proxy reference).
    # x_s forward noise 는 여전히 x_T.detach() (아래 x_s 식) — forward consistency 유지.
    # reference 가 x_T 이면 proxy 가 self-referential 로 항상 0 으로 수렴 → memorization 신호 무의미.
    # 루프 밖에서 한 번만 정의 → 4번의 update 동안 변하지 않음
    epsilon_ref = torch.randn_like(x_T_init)

    for ui, t_idx in enumerate(update_indices):
        # print(f"t_idx: {t_idx}")
        optimizer.zero_grad()

        zt = x_T.to(sd.dtype) * sd.scheduler.init_noise_sigma

        # ================================================================
        # accumulate mode: update_indices[0..ui] 의 proxy 를 누적해서 loss
        # ================================================================
        if type_memo_loss == "accumulate":
            active = set(update_indices[:ui + 1])
            accumulated_proxy = 0.0
            n_active = 0
            for step_idx, t in enumerate(timesteps):
                at = sd.alpha(t)
                at_prev = sd.alpha(t - sd.skip)
                noise_uc, noise_c = sd.predict_noise(zt, t, uc, c)
                eps_theta = noise_uc + cfg_eff_at(sd, step_idx, cfg) * (noise_c - noise_uc)
                x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()
                zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta

                if step_idx in active:
                    x_s = alpha_s.sqrt().to(sd.dtype) * x0_hat + (1 - alpha_s).sqrt().to(sd.dtype) * epsilon_ref.to(sd.dtype)
                    noise_uc_s, noise_c_s = sd.predict_noise(x_s, s_target, uc, c)
                    eps_s = noise_uc_s + cfg_eff_at(sd, s_idx, cfg) * (noise_c_s - noise_uc_s)
                    B = eps_s.shape[0]
                    proxy = (epsilon_ref.to(sd.dtype) - eps_s).reshape(B, -1).pow(2).mean()
                    accumulated_proxy = accumulated_proxy + proxy
                    n_active += 1
                    print(f"    [accumulate] step={step_idx} proxy={proxy.item():.6f} (n_active={n_active})")
                    del x_s, eps_s, noise_uc_s, noise_c_s

                if step_idx == t_idx:
                    break

            loss_memo = accumulated_proxy / n_active
            loss_align = (x0_hat.float() - x0_orig_refs[t_idx]).reshape(x0_hat.shape[0], -1).pow(2).mean(-1).mean()
            loss = loss_memo + lambda_align * loss_align

            print(f"  [acc] ui={ui} t_idx={t_idx}  loss_memo={loss_memo.item():.6f}  (accumulated {n_active} steps)")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"    [opt-acc] step {t_idx}: memo={loss_memo.item():.6f} "
                  f"|dxT|={((x_T.detach() - x_T_init).norm()).item():.4f}")

            del zt, x0_hat, eps_theta, noise_uc, noise_c
            torch.cuda.empty_cache()
            continue   # ★ 기존 (minimization/threshold) path 건너뜀

        # DDIM forward (with grad) up to t_idx
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t)
            at_prev = sd.alpha(t - sd.skip)
            noise_uc, noise_c = sd.predict_noise(zt, t, uc, c)
            eps_theta = noise_uc + cfg_eff_at(sd, step_idx, cfg) * (noise_c - noise_uc)
            x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()
            zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta # DDIM denoising step
            if step_idx == t_idx:
                _snr_t = (at / (1 - at)).item()
                print(f"    [tweedie t_idx={t_idx}] alpha_t={at.item():.4f}  SNR={_snr_t:.3f}  "
                      f"(1/sqrt(alpha)={(1/at.sqrt()).item():.2f}x amplification)")
                break

        # ---- memo proxy (eps_trajectory.py:107,111-116 참조) ----
        # x_s = √ᾱ_s·x̂₀ + √(1-ᾱ_s)·ε   (ε = x_T.detach() — trajectory 경로로만 gradient 흐름)
        # eps_trajectory.py:107 과 동일하게 ε를 detach.
        # x_T.detach() 안 하면 x_s→x_T shortcut gradient 생겨서 trajectory 의미 없어짐.
        x_s = alpha_s.sqrt().to(sd.dtype) * x0_hat + (1 - alpha_s).sqrt().to(sd.dtype) * epsilon_ref.to(sd.dtype)
        noise_uc_s, noise_c_s = sd.predict_noise(x_s, s_target, uc, c)
        eps_s = noise_uc_s + cfg_eff_at(sd, s_idx, cfg) * (noise_c_s - noise_uc_s)

        # memo proxy = ||ε - eps_s||² / D, batch-safe (eps_trajectory.py:115-116 과 동일)
        # reshape(B,-1).pow(2).mean(-1) → 샘플별 (B,) proxy; .mean() 으로 스칼라 loss
        B = eps_s.shape[0]
        memo_proxy = (epsilon_ref.to(sd.dtype) - eps_s).reshape(B, -1).pow(2).mean(-1)  # (B,)

        # 완화 목적: memo_proxy를 MINIMIZE.
        # 실측(eps_trajectory plot)에서 memorized prompt일수록 proxy가 큼(ε을 무시하고
        # memorized 방향 eps_s를 뱉기 때문). ∴ proxy↓ = 정상 denoiser(eps_s→ε)로 회귀 = 완화.
        loss_memo = compute_memo_loss(memo_proxy, type_memo_loss, memo_threshold)

        # text alignment loss: keep x̂₀ close to original trajectory (MSE, batch-safe)
        loss_align = (x0_hat.float() - x0_orig_refs[t_idx]).reshape(x0_hat.shape[0], -1).pow(2).mean(-1).mean()
        # print(f"loss_memo: {loss_memo}, loss_align: {loss_align}")

        loss = loss_memo + lambda_align * loss_align
        
        print(f"memo_proxy(↓=mitigate): {memo_proxy.mean().item():.6f}  loss_memo: {loss_memo.item():.6f}  loss_align: {loss_align.item():.6f}")

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"    [opt] step {t_idx}: memo={loss_memo.item():.6f} "
              f"align={loss_align.item():.6f} "
              f"|dxT|={((x_T.detach() - x_T_init).norm()).item():.4f}")

        # cleanup
        del zt, x0_hat, x_s, eps_s, noise_uc, noise_c, noise_uc_s, noise_c_s, eps_theta
        torch.cuda.empty_cache()

    x_T_opt = x_T.detach().clone()
    del x_T, optimizer, x0_orig_refs
    torch.cuda.empty_cache()

    # ---- fp16 복구 ----
    sd.unet.half()
    sd.dtype = _orig_dtype

    return x_T_opt, total_loss


def optimize_xT_adj(sd, uc, c, cfg, device,
                     init_steps, num_steps, gap_steps, lr, base_s_ratio, lambda_align,
                     adjoint_normalize=False, adjoint_fd_fallback=False, fd_eps=1e-3,
                     cache_latents=True, grad_vanish_threshold=1e-3, batch_size=1,
                     record_dir=None, record_dirs=None,
                     type_memo_loss="minimization", memo_threshold=0.0,
                     grad_prcd=False):
    """
    AdjointDPM version of optimize_xT.

    Replaces loss.backward() through 10-19 chained UNets (gradient vanishing)
    with an exact discrete adjoint ODE solve. The forward DDIM chain (η=0) is
    run under no_grad with latent-state checkpointing; the backward pass folds
    one UNet VJP (vector-Jacobian product) per step via a reverse recursion.

    Loss definition (unchanged from optimize_xT):
        L = ||ε_ref - ε_s||²/D  +  λ_align · ||x0_hat - x0_orig_ref||²

    Why this fixes vanishing:
      The exact gradient ∂L/∂x_T = ∏_{k=1}^{t_idx} (∂x_k/∂x_{k-1})ᵀ · g_{t_idx}.
      Standard autograd materializes the product, which → 0 as t_idx grows
      (each factor has spectral radius ≤ 1). The adjoint identity rewrites the
      SAME product as the solution of the linear reverse recursion
          g_{k-1} = (A_k·I + B_k·J_k)ᵀ · g_k
      where each step is one additive VJP fold (torch.autograd.grad, O(1)
      memory), never forming the product. Terminal g_{t_idx} comes from a
      tiny 2-UNet autograd head that never vanishes.

    DDIM step affine form (substitute Tweedie x0_hat into the update):
        x_k   = state at timestep timesteps[k]   (input to UNet at step k)
        a_k   = ᾱ(timesteps[k])
        a_kp1 = ᾱ(timesteps[k-1])  (next step is CLEANER, so a_{k+1} > a_k)
        A_k = sqrt(a_{k+1} / a_k)                                    (scalar)
        B_k = sqrt(1 - a_{k+1}) - sqrt(a_{k+1}·(1 - a_k) / a_k)      (scalar)
        x_{k+1} = A_k·x_k + B_k·eps_theta(x_k, t_k)
        ∂x_{k+1}/∂x_k = A_k·I + B_k·J_k   where J_k = ∂eps_theta/∂x_k

    Drop-in replacement for optimize_xT: identical positional args.

    Args:
        adjoint_normalize: (§7.3) rescale g to unit norm each step. Off by
            default; enable if |g| explodes on long chains. Preserves
            gradient direction (Adam is scale-invariant for direction).
        adjoint_fd_fallback: (§7.2) finite-difference Hutchinson VJP instead
            of exact autograd VJP. Only use if autograd on the UNet is
            unavailable. Adds variance. Default False.
        fd_eps: perturbation for the FD fallback.
        cache_latents: (§5) store x_k latents (~65KB each, ≤1.3MB total) so
            the adjoint does not recompute the forward chain per step.
            False → O(1) memory but 2× UNet calls. Default True.

    Returns:
        (x_T_optimized [1,4,64,64] fp32, total_loss float)
    """

    # ================================================================
    # (A) SETUP — identical to optimize_xT (lines 30-50)
    # ================================================================
    # Force fp32 for the UNet during optimization so the adjoint VJPs and
    # the A_k/B_k scalar arithmetic are numerically stable. fp16 restores
    # at the end (§: restore fp16).
    _orig_dtype = sd.dtype
    sd.unet.float()
    sd.dtype = torch.float32
    uc = uc.float()
    c = c.float()

    timesteps = list(sd.scheduler.timesteps)
    update_indices = [init_steps + i * gap_steps for i in range(num_steps)]
    update_indices = [i for i in update_indices if i < len(timesteps)]

    # s target for memo_proxy (unchanged)
    s_idx = int(len(timesteps) * base_s_ratio)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)

    x_T_init = torch.randn(batch_size, 4, 64, 64, device=device, dtype=torch.float32)
    x_T = x_T_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x_T], lr=lr)

    # ---- reference trajectory (NO grad) — lines 52-68, unchanged ----
    # Pre-compute x̂0_orig at each update step using the UN-optimized x_T.
    # Used by the alignment loss as a text-anchoring regularizer.
    x0_orig_refs = {}
    with torch.no_grad():
        zt_ref = x_T_init.to(sd.dtype) * sd.scheduler.init_noise_sigma
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t)
            at_prev = sd.alpha(t - sd.skip)
            noise_uc, noise_c = sd.predict_noise(zt_ref, t, uc, c)
            eps_theta = noise_uc + cfg_eff_at(sd, step_idx, cfg) * (noise_c - noise_uc)
            x0_hat = (zt_ref - (1 - at).sqrt() * eps_theta) / at.sqrt()
            zt_ref = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta
            if step_idx in update_indices:
                x0_orig_refs[step_idx] = x0_hat.detach().clone().float()

    def cfg_combine(noise_uc, noise_c, step_idx):
        """CFG combination with staged CFG (cfg_eff_at). step_idx = denoising step index."""
        return noise_uc + cfg_eff_at(sd, step_idx, cfg) * (noise_c - noise_uc)

    # ================================================================
    # (B) OPTIMIZATION LOOP — adjoint replaces loss.backward()
    # ================================================================
    total_loss = 0.0
    # ε reference = x_T 와 독립인 fresh random noise (proxy reference; 4번 update 동안 고정).
    # x_s forward noise 는 x_T.detach() (아래 x_s 식) 로 forward consistency 유지.
    epsilon_ref = torch.randn_like(x_T_init)

    # ---- record setup: batch(=seed)별 폴더 + update-step 누적 리스트 ----
    # 구조: record/img_PPPP/img_PPPP_BB/{grid.png, loss.csv}
    # grid.png = row(update step 횟수) × col(x_t, x_0|t, x_s)
    batch_record_imgs = None
    batch_dirs = None
    if record_dirs is not None:
        # 배치 모드(batch_txt>1) — main이 row→dir 매핑을 주입 (프롬pt별 record 분리)
        assert len(record_dirs) == batch_size
        batch_dirs = list(record_dirs)
        batch_record_imgs = [[] for _ in range(batch_size)]
    elif record_dir is not None:
        _prompt_tag = os.path.basename(record_dir.rstrip('/'))  # e.g. "img_0000"
        batch_dirs = []
        for _b in range(batch_size):
            _bd = os.path.join(record_dir, f"{_prompt_tag}_{_b:02d}")
            os.makedirs(_bd, exist_ok=True)
            batch_dirs.append(_bd)
        batch_record_imgs = [[] for _ in range(batch_size)]  # per-batch step 누적용

    for ui, t_idx in enumerate(update_indices):
        optimizer.zero_grad()

        # ---------- (1) FORWARD: x_T -> x_{t_idx}, NO grad, cache latents --
        # Run the DDIM chain under no_grad. We store only the latent states
        # x_k (each ~65KB fp32) — NOT activations. Peak memory is O(t_idx)
        # latents ≈ 1.3MB for t_idx=19, negligible vs the hundreds-of-MB
        # UNet activations the original code retained.
        #
        # xs[k] = latent at timestep timesteps[k] (input to UNet at step k).
        # xs[0] = scaled initial noise. xs has length t_idx+1 (steps 0..t_idx).
        x_k = (x_T.detach() * sd.scheduler.init_noise_sigma).clone()
        xs = [x_k.clone()] if cache_latents else None
        with torch.no_grad():
            # Forward / xs: cache
            for step_idx, t in enumerate(timesteps):
                at = sd.alpha(t)
                at_prev = sd.alpha(t - sd.skip)
                noise_uc, noise_c = sd.predict_noise(x_k, t, uc, c)
                eps_theta = cfg_combine(noise_uc, noise_c, step_idx)
                x0_hat = (x_k - (1 - at).sqrt() * eps_theta) / at.sqrt()
                x_k = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta # DDIM 
                if step_idx == t_idx:
                    # Match the SNR debug print of the original (lines 91-93)
                    _snr_t = (at / (1 - at)).item()
                    print(f"    [tweedie t_idx={t_idx}] alpha_t={at.item():.4f}  "
                          f"SNR={_snr_t:.3f}  (1/sqrt(alpha)={(1/at.sqrt()).item():.2f}x amplification)")
                    break
                # cache the NEXT state for steps we will visit in the adjoint
                if cache_latents and (step_idx + 1) <= t_idx:
                    xs.append(x_k.clone()) # cache
        # x_k is now x_{t_idx+1} (after DDIM step at step_idx==t_idx).
        # But the terminal head needs x_{t_idx} (BEFORE the DDIM step).
        # xs[t_idx] was cached at step_idx==t_idx-1 (appended after DDIM step).
        if cache_latents and len(xs) > t_idx:
            x_end_state = xs[t_idx].clone()   # x_{t_idx} (correct)
        else:
            # Fallback: recompute x_{t_idx} from x_T (no cache path)
            x_end_state = _recompute_to(sd, x_T, timesteps, t_idx, uc, c,
                                        cfg, sd.scheduler.init_noise_sigma).clone()

        # ---------- (2-ACC) ACCUMULATE MODE (adjointDPM) -----------------
        if type_memo_loss == "accumulate": # update_indices: [3,6,9,12]
            active = update_indices[:ui + 1] # t_idx, ui:2, active: [3,6,9]
            g_injections = {}
            loss_memo_acc = 0.0
            with torch.enable_grad():
                # accumulate calculate
                for k in active: 
                    x_end_k = xs[k].detach().clone().requires_grad_(True)
                    t_k = timesteps[k]; at_k = sd.alpha(t_k)
                    _nu, _nc = sd.predict_noise(x_end_k, t_k, uc, c)
                    eps_k = cfg_combine(_nu, _nc, k)
                    x0_k = (x_end_k - (1 - at_k).sqrt() * eps_k) / at_k.sqrt()
                    x_s_k = (alpha_s.sqrt().to(sd.dtype) * x0_k
                             + (1 - alpha_s).sqrt().to(sd.dtype) * epsilon_ref.to(sd.dtype))
                    _nus, _ncs = sd.predict_noise(x_s_k, s_target, uc, c)
                    eps_s_k = cfg_combine(_nus, _ncs, s_idx)
                    B = eps_s_k.shape[0]
                    proxy_k = (epsilon_ref.to(sd.dtype) - eps_s_k).reshape(B, -1).pow(2).mean(-1).mean()
                    loss_memo_acc = loss_memo_acc + proxy_k
                    g_injections[k] = torch.autograd.grad(proxy_k, x_end_k)[0]
                    print(f"  [acc-adj] step={k} proxy={proxy_k.item():.6f}")
                    del x_end_k, eps_k, x0_k, x_s_k, eps_s_k, _nu, _nc, _nus, _ncs
                n_acc = len(active)
                loss_memo = loss_memo_acc / n_acc
                # align at t_idx
                x_end = x_end_state.detach().clone().requires_grad_(True)
                t_break = timesteps[t_idx]; at_break = sd.alpha(t_break)
                _nut, _nct = sd.predict_noise(x_end, t_break, uc, c)
                eps_t = cfg_combine(_nut, _nct, t_idx)
                x0_hat = (x_end - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()
                loss_align = ((x0_hat.float() - x0_orig_refs[t_idx])
                              .reshape(x0_hat.shape[0], -1).pow(2).mean(-1).mean())
            loss = loss_memo + lambda_align * loss_align
            print(f"  [acc] ui={ui} t_idx={t_idx} loss_memo={loss_memo.item():.6f} ({n_acc} steps)")
            # adjoint recursion with g injections
            # active=[2,4]라고 하면 g_interval = (grad about proxy(time_step=2) w.r.t x_2,
            # ,grad about proxy(time_step=4) w.r.t x_4)
            g = g_injections[t_idx]
            g_terminal_norm = g.flatten().norm().item()
            if t_idx == 0:
                x_T.grad = (g * sd.scheduler.init_noise_sigma).detach().reshape_as(x_T)
            else:
                for kk in range(t_idx, 0, -1):
                    j = kk - 1
                    t_j = timesteps[j]; a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j - sd.skip)
                    A_j = (a_jp1 / a_j).sqrt()
                    B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()
                    x_j_local = xs[j].detach().clone().requires_grad_(True)
                    _nuj, _ncj = sd.predict_noise(x_j_local, t_j, uc, c)
                    eps_j = cfg_combine(_nuj, _ncj, j)
                    Jt_g = torch.autograd.grad(eps_j, x_j_local, grad_outputs=g)[0]
                    g = A_j * g + B_j * Jt_g
                    if j in g_injections:
                        g = g + g_injections[j]
                    del x_j_local, eps_j, _nuj, _ncj, Jt_g
            x_T.grad = (g * sd.scheduler.init_noise_sigma).detach().reshape_as(x_T)
            if grad_prcd and x_T.grad is not None:
                with torch.no_grad():
                    x_T.grad = project_wgnc_batched(x_T.grad.detach()).reshape_as(x_T)
            optimizer.step()
            total_loss += loss.item()
            print(f"  [opt-acc-adj] t_idx={t_idx} memo={loss_memo.item():.6f} "
                  f"|dxT|={((x_T.detach() - x_T_init).norm()).item():.4f}")
            del x_end, x0_hat, eps_t, _nut, _nct, g
            if xs is not None: del xs
            torch.cuda.empty_cache()
            continue

        # ---------- (2) TERMINAL ADJOINT g_{t_idx} via 2-UNet head --------
        # The loss head is shallow (only 2 UNets) and never vanishes, so we
        # backprop it with standard autograd on a fresh leaf x_end. This
        # produces the exact terminal condition g_{t_idx} = ∂L/∂x_{t_idx}
        # that the adjoint recursion then propagates back to x_T.
        #
        #   x_{t_idx} ──(Tweedie ε_θ)──> x0_hat ──(x_s)──> ε_s ──(L)──> scalar
        #                                  │
        #                                  └──(x0_hat also feeds align loss)──>
        #
        # ε_ref in x_s is x_T.detach() — a stop-gradient. This is INTENTIONAL
        # (original line 100): it blocks a shortcut gradient directly into
        # x_T and forces the memo loss to enter g_{t_idx} through x0_hat,
        # i.e. through the DDIM chain. The adjoint respects this.
        #
        # BUG12 fix: wrap the head + adjoint in an explicit enable_grad guard
        # so that grad flows even if the caller is under no_grad (defensive;
        # main() is not, but sibling optimizers latent_opt/prompt_opt are).
        with torch.enable_grad():
            x_end = x_end_state.detach().clone().requires_grad_(True)
            t_break = timesteps[t_idx]
            at_break = sd.alpha(t_break)

            # UNet call #1 (grad): CFG noise at the break timestep for Tweedie.
            # predict_noise is NOT decorated @no_grad, so grad flows here.
            _nu_t, _nc_t = sd.predict_noise(x_end, t_break, uc, c)
            eps_t = cfg_combine(_nu_t, _nc_t, t_idx)
            x0_hat = (x_end - (1 - at_break).sqrt() * eps_t) / at_break.sqrt()

            # x_s with DETACHED ε_ref (identical to original line 100).
            x_s = (alpha_s.sqrt().to(sd.dtype) * x0_hat
                   + (1 - alpha_s).sqrt().to(sd.dtype) * epsilon_ref.to(sd.dtype))

            # UNet call #2 (grad): CFG noise at s_target for the memo proxy.
            noise_uc_s, noise_c_s = sd.predict_noise(x_s, s_target, uc, c)
            eps_s = cfg_combine(noise_uc_s, noise_c_s, s_idx)

            # ---- memo proxy & losses — IDENTICAL formulas to original ----
            B = eps_s.shape[0]
            memo_proxy = (epsilon_ref.to(sd.dtype) - eps_s).reshape(B, -1).pow(2).mean(-1)
            loss_memo = compute_memo_loss(memo_proxy, type_memo_loss, memo_threshold)
            loss_align = ((x0_hat.float() - x0_orig_refs[t_idx])
                          .reshape(x0_hat.shape[0], -1).pow(2).mean(-1).mean())
            loss = loss_memo + lambda_align * loss_align

            _loss_tag = (f"[memo_loss={type_memo_loss},τ={memo_threshold}]"
                         if type_memo_loss == "threshold" else "[memo_loss=minimization]")
            print(f"{_loss_tag} proxy(↓=mitigate): {memo_proxy.mean().item():.6f}  "
                  f"loss_memo: {loss_memo.item():.6f}  loss_align: {loss_align.item():.6f}")

            # ---- record: 매 update step마다 batch(=seed)별 latent 누적 + loss.csv append ----
            # 그리드는 loop 종료 후 batch별 1장으로 합침: row=update step(init_opti 횟수), col=(x_t, x_0|t, x_s).
            if batch_dirs is not None:
                import csv as _csv
                with torch.no_grad():
                    _vae_dtype = next(sd.vae.parameters()).dtype   # VAE는 fp16 (UNet만 fp32 강제 중)
                    def _dec(z):
                        return (sd.decode(z.detach().to(_vae_dtype)) / 2 + 0.5).clamp(0, 1).cpu()
                    img_xt  = _dec(x_end)    # x_t  = x_{t_idx} (forward chain 결과)
                    img_x0t = _dec(x0_hat)   # x_0|t (Tweedie 추정)
                    img_xs  = _dec(x_s)      # x_s  = √α_s·x0_hat + √(1-α_s)·x_T.detach()
                    B = img_xt.shape[0]
                    eps_s_n   = eps_s.detach().reshape(B, -1).norm(dim=-1)
                    eps_ref_n = epsilon_ref.detach().to(sd.dtype).reshape(B, -1).norm(dim=-1)
                    for b in range(B):
                        # 이 update step의 (x_t, x_0|t, x_s) 3개 column을 batch별로 누적
                        batch_record_imgs[b].extend([img_xt[b], img_x0t[b], img_xs[b]])
                        # batch별 loss.csv: epsilon_s, epsilon(loss의 ε_ref), loss
                        _csv_path = os.path.join(batch_dirs[b], "loss.csv")
                        _wh = not os.path.exists(_csv_path)
                        with open(_csv_path, "a", newline="") as _f:
                            _w = _csv.writer(_f)
                            if _wh:
                                _w.writerow(["update_step", "t_idx", "eps_s_norm", "eps_ref_norm",
                                             "memo_proxy", "loss_memo", "loss"])
                            _w.writerow([ui, t_idx, f"{eps_s_n[b].item():.6f}",
                                         f"{eps_ref_n[b].item():.6f}", f"{memo_proxy[b].item():.6f}",
                                         f"{loss_memo.item():.6f}", f"{loss.item():.6f}"])

            # Terminal adjoint: ∂L/∂x_{t_idx} via the 2-UNet head (exact autograd).
            g = torch.autograd.grad(loss, x_end, retain_graph=False)[0]
            g_terminal_norm = g.flatten().norm().item()
            grad_vanished = False

            # ---------- (3) ADJOINT ODE BACKWARD: t_idx -> 0 ---------------
            # Recursion:  g_{j} = (A_j·I + B_j·J_j)ᵀ · g_{j+1}
            #                    = A_j · g_{j+1}  +  B_j · (J_jᵀ · g_{j+1})
            #
            # INDEXING (BUG1 fix): forward step j maps x_j -> x_{j+1} using
            # timestep timesteps[j] and Jacobian J_j = ∂eps_theta(x_j)/∂x_j.
            # To descend g_{t_idx} -> g_0 we must apply (DF_j)ᵀ for
            # j = t_idx-1, t_idx-2, ..., 0. So at loop iteration k we use the
            # PREVIOUS step j = k-1: timestep timesteps[j], latent xs[j],
            # coeffs from a_j = alpha(timesteps[j]), a_{j+1} = alpha(t_j - skip).
            #
            # J_jᵀ · g is an exact VJP via torch.autograd.grad — one UNet
            # forward+backward, O(1) activation memory. No Jacobian materialized.
            if t_idx == 0:
                # Edge case (§7.4): no chain. g is already ∂L/∂x_0; just apply
                # the init_noise_sigma scaling (§7.6) since x_0 = x_T·σ_init.
                x_T.grad = (g * sd.scheduler.init_noise_sigma).detach().reshape_as(x_T)
            else:
                for k in range(t_idx, 0, -1):
                    j = k - 1                       # forward step F_j: x_j -> x_{j+1}=x_k
                    t_j = timesteps[j]              # timestep of forward step j
                    a_j   = sd.alpha(t_j)           # alpha at x_j
                    a_jp1 = sd.alpha(t_j - sd.skip) # alpha at x_{j+1}; a_jp1 > a_j

                    # Affine coefficients (scalars, O(1)).
                    A_j = (a_jp1 / a_j).sqrt()
                    B_j = (1 - a_jp1).sqrt() - (a_jp1 * (1 - a_j) / a_j).sqrt()

                    # Recover x_j (input to UNet at forward step j). Detach +
                    # re-leaf so autograd builds the graph for THIS single VJP.
                    if cache_latents:
                        x_j_local = xs[j].detach().clone().requires_grad_(True)
                    else:
                        x_j_local = _recompute_to(sd, x_T, timesteps, j, uc, c,
                                                  cfg, sd.scheduler.init_noise_sigma)
                        x_j_local = x_j_local.detach().clone().requires_grad_(True)

                    if adjoint_fd_fallback:
                        # §7.2: finite-difference Hutchinson VJP (fallback only).
                        # Uses the identity Jᵀg = E_v[ v · (vᵀ · Jᵀg) ]; estimate
                        # J·v by central differences and contract with (vᵀg).
                        # NOTE: this is a single-sample (high-variance) estimate;
                        # default off, prefer the exact autograd path below.
                        v   = torch.randn_like(x_j_local)
                        _nu_p, _nc_p = sd.predict_noise(x_j_local + fd_eps * v, t_j, uc, c)
                        e_p = cfg_combine(_nu_p, _nc_p, j)
                        _nu_m, _nc_m = sd.predict_noise(x_j_local - fd_eps * v, t_j, uc, c)
                        e_m = cfg_combine(_nu_m, _nc_m, j)
                        Jv  = (e_p - e_m) / (2 * fd_eps)          # estimate of J·v
                        Jt_g = Jv * (g * v).sum()                 # J·v · (vᵀg), shape of x
                    else:
                        # Exact VJP: J_jᵀ · g via reverse-mode autograd.
                        # CFG-combined noise so the VJP captures
                        # J_uc + cfg·(J_c - J_uc) automatically.
                        _nu_j, _nc_j = sd.predict_noise(x_j_local, t_j, uc, c)
                        eps_j = cfg_combine(_nu_j, _nc_j, j)
                        Jt_g = torch.autograd.grad(eps_j, x_j_local,
                                                   grad_outputs=g,
                                                   retain_graph=False)[0]

                    # Adjoint recursion fold (additive — does NOT vanish).
                    # g_j = A_j·g_{j+1} + B_j·(J_jᵀ·g_{j+1})
                    g_prev = g.clone()
                    g = A_j * g + B_j * Jt_g

                    # ---- gradient magnitude tracking ----
                    g_norm = g.flatten().norm().item()
                    g_prev_norm = g_prev.flatten().norm().item()
                    ratio = g_norm / (g_prev_norm + 1e-12)
                    ratio_to_terminal = g_norm / (g_terminal_norm + 1e-12)
                    print(f"    [adj] step {j:2d}->0: |g|={g_norm:.6e}  "
                          f"|g_prev|={g_prev_norm:.6e}  "
                          f"ratio(step)={ratio:.4f}  "
                          f"ratio(to terminal)={ratio_to_terminal:.6f}")
                    if ratio_to_terminal < grad_vanish_threshold and not grad_vanished:
                        grad_vanished = True
                        print(f"    ⚠️  gradient dropped below {grad_vanish_threshold} of terminal at step {j} "
                              f"(t_idx={t_idx}, {j} steps from x_T)")

                    if adjoint_normalize:
                        # §7.3: optional magnitude control. Preserves direction;
                        # Adam is direction-only, so this does not change the
                        # update direction, only the implicit step magnitude.
                        g = g / (g.flatten().norm() + 1e-8)

                    del x_j_local, Jt_g
                    if not adjoint_fd_fallback:
                        del eps_j

                # §7.6: x_0 = x_T · init_noise_sigma (chain rule for the scalar).
                # x_T is the leaf, so the final gradient w.r.t. x_T picks up σ.
                x_T.grad = (g * sd.scheduler.init_noise_sigma).detach().reshape_as(x_T) # x_T.shape == [B,C,H,W]

        # Free the head graph immediately (only 2 UNets, but no need to keep).
        del x_end, eps_t, x0_hat, x_s, eps_s, noise_uc_s, noise_c_s

        # ---- gradient preconditioning (GPER, Hwang & Sung ICML 2026; arXiv:2602.08646) ----
        # project grad onto white Gaussian noise feasible set before the Adam step:
        #   x <- Adam(x, Proj_G(grad)). keeps each update noise-aligned → prevents
        #   the latent from drifting out of the white-Gaussian prior (memo overfitting).

        if grad_prcd and x_T.grad is not None:
            with torch.no_grad():
                x_T.grad = project_wgnc_batched(x_T.grad.detach()).reshape_as(x_T)

        optimizer.step()
        total_loss += loss.item()

        # Debug prints — mirrors original lines 125-127.
        grad_norm = x_T.grad.norm().item() if x_T.grad is not None else float('nan')
        print(f"    [opt-adj] step {t_idx}: memo={loss_memo.item():.6f} "
              f"align={loss_align.item():.6f} "
              f"|g_xT|={grad_norm:.4f} "
              f"|dxT|={((x_T.detach() - x_T_init).norm()).item():.4f}")

        # cleanup
        del x_end_state
        if xs is not None:
            del xs
        torch.cuda.empty_cache()

    # ---- record: update loop 종료 후 batch(=seed)별 그리드 저장 ----
    # row = update step(init_opti 횟수), col = (x_t, x_0|t, x_s)
    if batch_record_imgs is not None:
        from torchvision.utils import make_grid as _make_grid, save_image as _save_image
        for b in range(len(batch_record_imgs)):
            if batch_record_imgs[b]:
                _grid = _make_grid(batch_record_imgs[b], nrow=3)
                _save_image(_grid, os.path.join(batch_dirs[b], "grid.png"))

    x_T_opt = x_T.detach().clone()
    del x_T, optimizer, x0_orig_refs
    torch.cuda.empty_cache()

    # ---- restore fp16 (identical to original line 138) ----
    sd.unet.half()
    sd.dtype = _orig_dtype

    return x_T_opt, total_loss


# ===================================================================
#  ⑧ [Anchor: Twd_gap] (candidate_proxy.md ⑧) — x_T 직접 최적화 (adjoint 불필요)
#  loss = s_Δ + w_twd · memo_proxy(①)
#    s_Δ        = ‖ε_uc(x_T) − ε_c(x_T)‖₂   (per-sample L2 norm → batch mean)
#    memo_proxy = ‖ε_noise − ε_s‖²/D, 체인 x_T → x̂_{0|T} → x_s → ε_s (ⓑ~ⓔ head)
#  grad 가 UNet 2개짜리 얕은 헤드에서 x_T 로 곧장 흐름 — DDIM 연쇄(ⓐ)를 거치지
#  않으므로 AdjointDPM(optimize_xT_adj) 불필요. init_steps/gap_steps 무의미(항상
#  x_T), num_steps = x_T 갱신 횟수.
# ===================================================================
def optimize_xT_anchor(sd, uc, c, cfg, device,
                       num_steps, lr, base_s_ratio, w_twd=1.0,
                       batch_size=1, record_dir=None, record_dirs=None):
    """loss = s_Δ + w_twd · memo_proxy  (minimization 고정 — compute_memo_loss 갈래 없음)"""

    # ---- fp32 강제 (optimize_xT_adj 정책 동일) ----
    _orig_dtype = sd.dtype
    sd.unet.float()
    sd.dtype = torch.float32
    uc = uc.float()
    c = c.float()

    timesteps = list(sd.scheduler.timesteps)
    T_max = timesteps[0]
    at = sd.alpha(T_max)
    s_idx = int(len(timesteps) * base_s_ratio)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)

    x_T_init = torch.randn(batch_size, 4, 64, 64, device=device, dtype=torch.float32)
    x_T = x_T_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x_T], lr=lr)

    # ε_noise: x_s noising + gap 기준 noise (x_T 와 독립인 fresh noise, update 동안 고정)
    epsilon_ref = torch.randn_like(x_T_init)

    def cfg_combine(noise_uc, noise_c, step_idx):
        return noise_uc + cfg_eff_at(sd, step_idx, cfg) * (noise_c - noise_uc)

    # ---- record setup: batch(=seed)별 폴더 (optimize_xT_adj 패턴) ----
    # 구조: record/img_PPPP/img_PPPP_BB/{grid.png, loss.csv}
    # grid.png = row(update 횟수) × col(x_T, x_0|T, x_s)
    batch_record_imgs = None
    batch_dirs = None
    if record_dirs is not None:
        assert len(record_dirs) == batch_size
        batch_dirs = list(record_dirs)
        batch_record_imgs = [[] for _ in range(batch_size)]
    elif record_dir is not None:
        _prompt_tag = os.path.basename(record_dir.rstrip('/'))
        batch_dirs = []
        for _b in range(batch_size):
            _bd = os.path.join(record_dir, f"{_prompt_tag}_{_b:02d}")
            os.makedirs(_bd, exist_ok=True)
            batch_dirs.append(_bd)
        batch_record_imgs = [[] for _ in range(batch_size)]

    print(f"[anchor] T_max={T_max} alpha_T={at.item():.4f} "
          f"(1/sqrt(alpha)={(1/at.sqrt()).item():.2f}x amplification)  "
          f"s_target={s_target}(idx {s_idx})  w_twd={w_twd}")

    total_loss = 0.0
    for ui in range(num_steps):
        optimizer.zero_grad()
        with torch.enable_grad():
            zt = x_T * sd.scheduler.init_noise_sigma

            # UNet #1: x_T 에서 uncond/cond ε — s_Δ 와 eps_ref(CFG 결합) 동시 획득
            _nu, _nc = sd.predict_noise(zt, T_max, uc, c)
            B = zt.shape[0]
            s_delta_per = (_nc - _nu).reshape(B, -1).norm(dim=-1)   # (B,) per-sample L2 norm
            s_delta = s_delta_per.mean()
            eps_ref = cfg_combine(_nu, _nc, 0)

            # Tweedie x̂_{0|T} → x_s forward noising
            x0_hat = (zt - (1 - at).sqrt() * eps_ref) / at.sqrt()
            x_s = (alpha_s.sqrt().to(sd.dtype) * x0_hat
                   + (1 - alpha_s).sqrt().to(sd.dtype) * epsilon_ref.to(sd.dtype))

            # UNet #2: x_s 에서 ε_s → memo_proxy (① head)
            _nus, _ncs = sd.predict_noise(x_s, s_target, uc, c)
            eps_s = cfg_combine(_nus, _ncs, s_idx)
            memo_proxy = (epsilon_ref.to(sd.dtype) - eps_s).reshape(B, -1).pow(2).mean(-1)
            gap = memo_proxy.mean()
            loss = s_delta + w_twd * gap

            # grad 분리 로깅 (⑥ on_main 패턴): |g1|=s_Δ 기여, |g2|=gap 기여
            _g1_n = torch.autograd.grad(s_delta, x_T, retain_graph=True)[0].norm().item()
            _g2_n = torch.autograd.grad(w_twd * gap, x_T, retain_graph=True)[0].norm().item()
            print(f"  [opt-anchor {ui+1}/{num_steps}] s_delta={s_delta.item():.4f} "
                  f"gap={gap.item():.6f} |g1|={_g1_n:.4f} |g2|={_g2_n:.4f} "
                  f"loss={loss.item():.4f}")

            # ---- record: batch별 (x_T, x_0|T, x_s) 누적 + loss.csv append ----
            if batch_dirs is not None:
                import csv as _csv
                with torch.no_grad():
                    _vae_dtype = next(sd.vae.parameters()).dtype   # VAE는 fp16 (UNet만 fp32 강제 중)
                    def _dec(z):
                        return (sd.decode(z.detach().to(_vae_dtype)) / 2 + 0.5).clamp(0, 1).cpu()
                    img_xT = _dec(zt)
                    img_x0 = _dec(x0_hat)
                    img_xs = _dec(x_s)
                    for b in range(B):
                        batch_record_imgs[b].extend([img_xT[b], img_x0[b], img_xs[b]])
                        _csv_path = os.path.join(batch_dirs[b], "loss.csv")
                        _wh = not os.path.exists(_csv_path)
                        with open(_csv_path, "a", newline="") as _f:
                            _w = _csv.writer(_f)
                            if _wh:
                                _w.writerow(["update_step", "s_delta", "memo_proxy", "loss"])
                            _w.writerow([ui, f"{s_delta_per[b].item():.6f}",
                                         f"{memo_proxy[b].item():.6f}", f"{loss.item():.6f}"])

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"    |dxT|={((x_T.detach() - x_T_init).norm()).item():.4f}")

        del _nu, _nc, _nus, _ncs, eps_ref, x0_hat, x_s, eps_s
        torch.cuda.empty_cache()

    # ---- record: update loop 종료 후 batch별 그리드 저장 ----
    if batch_record_imgs is not None:
        from torchvision.utils import make_grid as _make_grid, save_image as _save_image
        for b in range(len(batch_record_imgs)):
            if batch_record_imgs[b]:
                _grid = _make_grid(batch_record_imgs[b], nrow=3)
                _save_image(_grid, os.path.join(batch_dirs[b], "grid.png"))

    x_T_opt = x_T.detach().clone()
    del x_T, optimizer
    torch.cuda.empty_cache()

    # ---- restore fp16 ----
    sd.unet.half()
    sd.dtype = _orig_dtype

    return x_T_opt, total_loss


# ===================================================================
#  ⑨ [Btw_anchor: batch-traction] (candidate_proxy.md ⑨) — x_T 직접 최적화
#  loss = s_Δ + w_btw · Σ_{i<j} ‖x_i − x_j‖₂
#    s_Δ = ‖ε_uc(x_T) − ε_c(x_T)‖₂   (⑧ 과 동일, per-sample L2 norm → batch mean)
#    btw = 프롬pt 블록 안쪽 x_T pairwise L2 norm 합 (⑥ loss_1 의 norm 버전) —
#          최소화하면 batch 샘플 인력(뭉침). memorized 프롬pt에서 batch 간 std가
#          압도적으로 크다는 관측(2026-09-07)의 잉여 spread 를 눌러 완화.
#          prompt-major 블록 [p0의 S개, p1의 S개, ...] 안쪽 쌍만 인력 —
#          batch_txt>1 에서도 프롬pt 간 오염 없음.
#  grad: s_Δ는 UNet 1회 경유, btw는 x_T 직접(UNet 무경유) — 역방향 UNet 1회로
#  ⑧(2회)보다 얕음. init/gap_steps 무의미, num_steps = x_T 갱신 횟수.
# ===================================================================
def optimize_xT_btw(sd, uc, c, cfg, device,
                    num_steps, lr, w_btw=1.0,
                    batch_size=1, record_dir=None, record_dirs=None, seeds_per_prompt=None):
    """loss = s_Δ + w_btw · Σ_{i<j}‖x_i−x_j‖₂  (minimization 고정 — compute_memo_loss 갈래 없음)"""
    assert batch_size >= 2, "btw: need >=2 samples in batch for pairwise term (--num_seeds>=2)"

    # ---- fp32 강제 (⑧ anchor 정책 동일) ----
    _orig_dtype = sd.dtype
    sd.unet.float()
    sd.dtype = torch.float32
    uc = uc.float()
    c = c.float()

    timesteps = list(sd.scheduler.timesteps)
    T_max = timesteps[0]

    x_T_init = torch.randn(batch_size, 4, 64, 64, device=device, dtype=torch.float32)
    x_T = x_T_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x_T], lr=lr)

    # ---- record setup: batch(=seed)별 폴더 (⑧ 패턴) ----
    # 구조: record/img_PPPP/img_PPPP_BB/{grid.png, loss.csv} — grid = row(update) × col(x_T)
    batch_record_imgs = None
    batch_dirs = None
    if record_dirs is not None:
        assert len(record_dirs) == batch_size
        batch_dirs = list(record_dirs)
        batch_record_imgs = [[] for _ in range(batch_size)]
    elif record_dir is not None:
        _prompt_tag = os.path.basename(record_dir.rstrip('/'))
        batch_dirs = []
        for _b in range(batch_size):
            _bd = os.path.join(record_dir, f"{_prompt_tag}_{_b:02d}")
            os.makedirs(_bd, exist_ok=True)
            batch_dirs.append(_bd)
        batch_record_imgs = [[] for _ in range(batch_size)]

    _S_pp = seeds_per_prompt if seeds_per_prompt is not None else batch_size
    print(f"[btw] T_max={T_max}  w_btw={w_btw}  batch={batch_size} "
          f"(blocks={batch_size // _S_pp} x S={_S_pp}, "
          f"pairs={batch_size // _S_pp * (_S_pp * (_S_pp - 1) // 2)})")

    total_loss = 0.0
    for ui in range(num_steps):
        optimizer.zero_grad()
        with torch.enable_grad():
            zt = x_T * sd.scheduler.init_noise_sigma

            # UNet 1회: x_T 에서 uncond/cond ε → s_Δ (⑧ 첫 항 그대로)
            _nu, _nc = sd.predict_noise(zt, T_max, uc, c)
            B = zt.shape[0]
            s_delta_per = (_nc - _nu).reshape(B, -1).norm(dim=-1)   # (B,) per-sample L2 norm
            s_delta = s_delta_per.mean()

            # btw: 프롬pt 블록별 x_T pairwise L2 norm 합 (⑥ loss_1 패턴 — batch_txt>1에서도
            #      프롬pt끼리 인력 오염 방지. 행 순서 = prompt-major: [p0의 S개, p1의 S개, ...])
            z_flat = x_T.reshape(B, -1)
            S_pp = seeds_per_prompt if seeds_per_prompt is not None else B
            assert B % S_pp == 0, f"btw: B={B} not a multiple of seeds_per_prompt={S_pp}"
            _l1_blocks = []
            for _n in range(B // S_pp):
                _zb = z_flat[_n * S_pp:(_n + 1) * S_pp]
                _db = torch.cdist(_zb, _zb)                           # (S,S) L2 norm
                _iub = torch.triu_indices(S_pp, S_pp, offset=1)
                _l1_blocks.append(_db[_iub[0], _iub[1]].sum())        # Σ_{i<j} ‖x_i−x_j‖₂
            btw = sum(_l1_blocks) if len(_l1_blocks) > 1 else _l1_blocks[0]
            loss = s_delta + w_btw * btw

            # grad 분리 로깅 (⑧ 패턴): |g1|=s_Δ 기여, |g2|=btw 기여
            _g1_n = torch.autograd.grad(s_delta, x_T, retain_graph=True)[0].norm().item()
            _g2_n = torch.autograd.grad(w_btw * btw, x_T, retain_graph=True)[0].norm().item()
            print(f"  [opt-btw {ui+1}/{num_steps}] s_delta={s_delta.item():.4f} "
                  f"btw={btw.item():.2f} |g1|={_g1_n:.4f} |g2|={_g2_n:.4f} "
                  f"loss={loss.item():.4f}")

            # ---- record: batch별 x_T 누적 + loss.csv append ----
            if batch_dirs is not None:
                import csv as _csv
                with torch.no_grad():
                    _vae_dtype = next(sd.vae.parameters()).dtype   # VAE는 fp16 (UNet만 fp32 강제 중)
                    img_xT = (sd.decode(zt.detach().to(_vae_dtype)) / 2 + 0.5).clamp(0, 1).cpu()
                    for b in range(B):
                        batch_record_imgs[b].append(img_xT[b])
                        _csv_path = os.path.join(batch_dirs[b], "loss.csv")
                        _wh = not os.path.exists(_csv_path)
                        with open(_csv_path, "a", newline="") as _f:
                            _w = _csv.writer(_f)
                            if _wh:
                                _w.writerow(["update_step", "s_delta", "btw", "loss"])
                            _w.writerow([ui, f"{s_delta_per[b].item():.6f}",
                                         f"{btw.item():.6f}", f"{loss.item():.6f}"])

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"    |dxT|={((x_T.detach() - x_T_init).norm()).item():.4f}")

        del _nu, _nc
        torch.cuda.empty_cache()

    # ---- record: update loop 종료 후 batch별 그리드 저장 ----
    if batch_record_imgs is not None:
        from torchvision.utils import make_grid as _make_grid, save_image as _save_image
        for b in range(len(batch_record_imgs)):
            if batch_record_imgs[b]:
                _grid = _make_grid(batch_record_imgs[b], nrow=1)
                _save_image(_grid, os.path.join(batch_dirs[b], "grid.png"))

    x_T_opt = x_T.detach().clone()
    del x_T, optimizer
    torch.cuda.empty_cache()

    # ---- restore fp16 ----
    sd.unet.half()
    sd.dtype = _orig_dtype

    return x_T_opt, total_loss


# ===================================================================
#  ⑦/⑩ spectral score band energy (candidate_proxy.md ⑦ low · ⑩ high)
#  ε_cfg → rfft(ortho) → block energy → 대역별 sum (÷B) — differentiable
#  (측정 체계 eps_trajectory.py --proxy_trend --band_agg sum --lh_ratio 와 동일 스케일)
# ===================================================================
def spectral_low_band_energy(eps, block_size=16, lh_ratio=0.1, band="low"):
    """⑦/⑩ L = (1/B)Σ_{band} E_p — ε_cfg 스펙트럼 대역별 block energy 합 (per-sample).

    eps (B,C,H,W) → flatten (B,N) → rfft(ortho) → Hermitian 제거 y (B,N/2)
    → (B,P,B) reshape → E_p = ‖y^(p)‖² (B,P) → n_band = int(P·lh_ratio)개 block 선택
    → 합 ÷ block_size.
    band="low"  (⑦): 앞 n_band개 block(저주파) — energies[:, :n_band]
    band="high" (⑩): 뒤 n_band개 block(고주파) — energies[:, P-n_band:]
    스케일: white Gaussian 기준 ≈ n_band (예: lh=0.1, P=512 → ≈51) —
    ④ spectral_l2_loss의 mean_p(E_p)/B 와 동일 체계, 집계만 대역별 sum.

    differentiable — UNet head(ε_cfg)에서 z_{t-1}까지 backward 가능.
    Returns: (B,) per-sample (batch mean은 호출부에서)
    """
    SQRT2 = 2.0 ** 0.5
    B_bs = eps.shape[0]
    B = block_size
    xf = eps.reshape(B_bs, -1).to(torch.float64)        # (B_bs, N)
    f = torch.fft.rfft(xf, norm="ortho")                # (B_bs, N/2+1)
    f[:, 0] = (f[:, 0] + 1j * f[:, -1]) / SQRT2         # compact spectral (per sample)
    y = f[:, :-1]                                       # (B_bs, N/2)
    pad = (-y.shape[1]) % B
    if pad:
        y = torch.nn.functional.pad(y, (0, pad))
    y = y.reshape(B_bs, -1, B)                          # (B_bs, P, B)
    energies = y.abs().square().sum(dim=2).real          # (B_bs, P) E_p
    P = energies.shape[1]
    n_band = max(1, min(int(P * lh_ratio), P // 2))
    if band == "low":
        sel = energies[:, :n_band]
    elif band == "high":
        sel = energies[:, P - n_band:]
    else:
        raise ValueError(f"band must be 'low' or 'high', got {band!r}")
    return sel.sum(dim=1) / float(B)                     # (B_bs,) 대역별 sum ÷B


def optimize_xt(sd, uc, c, cfg, device,
                init_steps, num_steps, gap_steps, lr, base_s_ratio, lambda_align,
                batch_size=1, record_dir=None, record_dirs=None,
                type_memo_loss="minimization", memo_threshold=0.0,
                grad_prcd=False, xt_loss="memo_proxy",
                on_main=False, w_twd=1.0, on_main_spread=False, seeds_per_prompt=None,
                block_size=16, lh_ratio=0.1,
                w_mag=0.0, w_std=0.0,
                w_twd_mag=0.0, w_twd_std=0.0,
                opti_num=20, w_twd_mean=1.0,
                twd_mean_threshold=0.10, twd_std_threshold=0.05,
                ecf_K=10, ecf_w_max=3.0):
    """1-1 (candidate_proxy.md): memo_proxy loss 를 중간 latent x_t 에만 흘려 최적화.

    optimize_xT_adj 와의 차이: grad 가 ⓐ(x_T→x_t DDIM 연쇄)를 거슬러 전파되지 않음 —
    update 시점의 x_t 를 leaf 로 만들어 Adam 갱신 후 gap_steps 전진
    (optimize_xt_spectral.py::run_spectral_opt 패턴). AdjointDPM 불필요, update 1회당
    backward 는 terminal head(ⓑ~ⓔ, 2-UNet) 뿐.

    Chain (registry 1-1, v4 — DDIM denoising 먼저, 이후 Adam update):
      ① z_{t-1} ← DDIM(z_t)                        (η=0, no grad)
      ② L = memo_head(z_{t-1}) = ‖ε_ref−ε_s‖²/D   (loss 를 z_{t-1} 에서 계산, align 없음)
      ③ z_{t-1} ← Adam(z_{t-1}, ∇_{z_{t-1}} L)    ← update 적용점 = z_{t-1} (norm ≈ lr·√(B·D))
      → backward UNet 2회(head 만)

    Returns: (zt_final [B,4,64,64], last_forwarded int, total_loss float)
      zt_final = 마지막 update 이후 gap_steps 전진한 latent (이후 inference 재개용),
      last_forwarded = zt_final 의 step index
    """
    if type_memo_loss == "accumulate":
        raise NotImplementedError(
            "optimize_xt (mode=xt) supports minimization/threshold only — "
            "accumulate is for xT mode (optimize_xT_adj)")

    # ---- fp32 강제: UNet backward 정밀도 (optimize_xT_adj 정책과 동일) ----
    _orig_dtype = sd.dtype
    sd.unet.float()
    sd.dtype = torch.float32
    uc = uc.float()
    c = c.float()

    timesteps = list(sd.scheduler.timesteps)
    update_indices = [init_steps + i * gap_steps for i in range(num_steps)]
    update_indices = [i for i in update_indices if i < len(timesteps)]
    # ★ NFE 초과 방지: update(t_idx) 후 gap 전진 + resume 구간(step ≥ t_idx+gap+1)이
    #   최소 1 step 확보되도록 — 벗어나는 update는 종료(제외).
    #   (예: init=15/nsteps=9/gap=4, NFE=50 → 마지막 update 47은 47+4 전진으로 schedule
    #    소진 → 제외, 실제 마지막 update = 43, resume 48~49 확보)
    _n_updates_planned = len(update_indices)
    update_indices = [i for i in update_indices if i + gap_steps < len(timesteps)]
    if len(update_indices) < _n_updates_planned:
        print(f"  [nfe-guard] dropped {_n_updates_planned - len(update_indices)} update(s) "
              f"(NFE={len(timesteps)} limit) — actual updates={update_indices}")
    if not update_indices:
        raise ValueError(f"update_indices empty: init_steps={init_steps} >= NFE={len(timesteps)}")

    # s target for memo_proxy (unchanged)
    s_idx = int(len(timesteps) * base_s_ratio)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)

    x_T_init = torch.randn(batch_size, 4, 64, 64, device=device, dtype=torch.float32)

    def cfg_combine(noise_uc, noise_c, step_idx):
        return noise_uc + cfg_eff_at(sd, step_idx, cfg) * (noise_c - noise_uc)

    # ε reference (update 동안 고정)
    # twd_target: self-referential (x_T_init 자체) — proxy = ‖x_T_init − ε_s‖²/D
    # others:     x_T 와 독립인 fresh random noise
    if xt_loss == "twd_target":
        epsilon_ref = x_T_init.detach()
    else:
        epsilon_ref = torch.randn_like(x_T_init)

    # ---- record setup: batch(=seed)별 폴더 (optimize_xT_adj 패턴) ----
    batch_record_imgs = None
    batch_dirs = None
    if record_dirs is not None:
        # 배치 모드(batch_txt>1) — main이 row→dir 매핑을 주입 (프롬pt별 record 분리)
        assert len(record_dirs) == batch_size
        batch_dirs = list(record_dirs)
        batch_record_imgs = [[] for _ in range(batch_size)]
    elif record_dir is not None:
        _prompt_tag = os.path.basename(record_dir.rstrip('/'))  # e.g. "img_0000"
        batch_dirs = []
        for _b in range(batch_size):
            _bd = os.path.join(record_dir, f"{_prompt_tag}_{_b:02d}")
            os.makedirs(_bd, exist_ok=True)
            batch_dirs.append(_bd)
        batch_record_imgs = [[] for _ in range(batch_size)]

    # ---- 1. no_grad DDIM 0→init_steps (ⓐ — grad 여기서 끊김) ----
    zt = (x_T_init.to(sd.dtype) * sd.scheduler.init_noise_sigma).clone()
    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            if step_idx == init_steps:
                break
            at = sd.alpha(t)
            at_prev = sd.alpha(t - sd.skip)
            noise_uc, noise_c = sd.predict_noise(zt, t, uc, c)
            eps_theta = cfg_combine(noise_uc, noise_c, step_idx)
            x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()
            zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta
    print(f"  [forward] DDIM 0 → step {init_steps} (no grad, ⓐ excluded from ∇)")

    total_loss = 0.0
    _twd_m_final, _twd_s_final = None, None
    _last_processed_tidx = update_indices[0]
    # ---- 2. update 마다: DDIM denoising 먼저 → z_{t-1} 에서 Adam update → 전진 ----
    for ui, t_idx in enumerate(update_indices):
        t_upd = timesteps[t_idx]
        at_upd = sd.alpha(t_upd)
        at_prev = sd.alpha(t_upd - sd.skip)          # ᾱ at step 후 상태 (z_{t-1})
        t_prev = timesteps[min(t_idx + 1, len(timesteps) - 1)]  # z_{t-1} 의 timestep

        # ---- Step 1: DDIM denoising 먼저 (no grad): z_t → z_{t-1} ----
        with torch.no_grad():
            _nu_t, _nc_t = sd.predict_noise(zt, t_upd, uc, c)
            eps_t = cfg_combine(_nu_t, _nc_t, t_idx)
            x0_hat_t = (zt - (1 - at_upd).sqrt() * eps_t) / at_upd.sqrt()  # Tweedie
            z_prev0 = (at_prev.sqrt().to(sd.dtype) * x0_hat_t
                       + (1 - at_prev).sqrt().to(sd.dtype) * eps_t)        # DDIM step (η=0)

        # ---- Step 2: memo loss 를 z_{t-1}(leaf) 에서 계산 → Adam 으로 update ----
        zt_leaf = z_prev0.detach().to(torch.float32).clone().requires_grad_(True)  # z_{t-1}
        optimizer = torch.optim.Adam([zt_leaf], lr=lr)

        if xt_loss == "twd_target":
            # ⑭ inner while loop: 같은 t_idx에서 opti_num번까지 반복, threshold 충족 시 early stop
            # DDIM step은 위(Step 1)에서 1회 수행 — zt_leaf(z_{t-1})를 매 inner iter에서 갱신
            _inner_count = 0
            while True:
                optimizer.zero_grad()
                with torch.enable_grad():
                    _nu_p, _nc_p = sd.predict_noise(zt_leaf, t_prev, uc, c)
                    eps_p = cfg_combine(_nu_p, _nc_p, t_idx + 1)
                    x0_hat = (zt_leaf - (1 - at_prev).sqrt() * eps_p) / at_prev.sqrt()
                    x_s = (alpha_s.sqrt().to(sd.dtype) * x0_hat
                           + (1 - alpha_s).sqrt().to(sd.dtype) * epsilon_ref.to(sd.dtype))
                    noise_uc_s, noise_c_s = sd.predict_noise(x_s, s_target, uc, c)
                    eps_s = cfg_combine(noise_uc_s, noise_c_s, s_idx)
                    B = eps_s.shape[0]
                    proxy = (epsilon_ref.to(sd.dtype) - eps_s).reshape(B, -1).pow(2).mean(-1)
                    twd_mean_now = proxy.mean()
                    twd_std_now = (proxy.std() if B > 1
                                   else torch.zeros((), device=proxy.device, dtype=proxy.dtype))
                    loss = w_twd_mean * twd_mean_now + w_twd_std * twd_std_now
                _twd_m = twd_mean_now.item()
                _twd_s = twd_std_now.item()
                _twd_m_final, _twd_s_final = _twd_m, _twd_s
                print(f"  [twd_target {ui+1}/{len(update_indices)} inner {_inner_count+1}/{opti_num}] "
                      f"twd_mean={_twd_m:.4f}  twd_std={_twd_s:.4f}  "
                      f"(target: <{twd_mean_threshold} / <{twd_std_threshold})")
                if _twd_m < twd_mean_threshold and _twd_s < twd_std_threshold:
                    print(f"  [twd_target] EARLY STOP — threshold met at inner {_inner_count+1}")
                    break
                loss.backward()
                if grad_prcd and zt_leaf.grad is not None:
                    with torch.no_grad():
                        zt_leaf.grad = project_wgnc_batched(zt_leaf.grad.detach()).reshape_as(zt_leaf)
                _g_norm = zt_leaf.grad.norm().item() if zt_leaf.grad is not None else float('nan')
                print(f"    [opt-xt {ui+1}/{len(update_indices)} inner {_inner_count+1}/{opti_num}] "
                      f"t_idx={t_idx} loss={loss.item():.6f} "
                      f"|g_xt|={_g_norm:.4f} "
                      f"|dxt|={(zt_leaf.detach() - z_prev0).norm().item():.4f}")
                optimizer.step()
                total_loss += loss.item()
                _inner_count += 1
                if _inner_count >= opti_num:
                    print(f"  [twd_target] OPTI_NUM ({opti_num}) REACHED at t_idx={t_idx}")
                    break
            memo_proxy = proxy
            loss_memo = loss
        else:
            optimizer.zero_grad()
            with torch.enable_grad():
                _nu_p, _nc_p = sd.predict_noise(zt_leaf, t_prev, uc, c)          # ⓑ UNet @ z_{t-1}
                eps_p = cfg_combine(_nu_p, _nc_p, t_idx + 1)
                x0_hat = (zt_leaf - (1 - at_prev).sqrt() * eps_p) / at_prev.sqrt()  # ⓒ Tweedie

                if on_main:
                    # ⑥ [On-main] Compression into on-manifold — seed 간 인력 + w_twd·Tweedie gap
                    # loss_1 = Σ_{i<j} ‖z^i − z^j‖²/D  (per-pair ÷D — proxy 스케일 정합)
                    #          : batch 행 = 한 프롬pt의 seed들 (batch_txt=1 전제) — 최소화하면
                    #            seed latent 뭉침(general의 collapse 회복) = compression
                    #            --on_main_spread 시 부호 반전(원식 −Σ 리터럴, seed 분리 방향)
                    # loss_2 = memo_proxy(① Twd_gap head ⓑ~ⓔ) per-sample mean
                    # total = loss_1 + w_twd · loss_2
                    assert zt_leaf.shape[0] >= 2, "on_main: need >=2 seeds (--num_seeds>=2, --batch_txt=1)"
                    x_s = (alpha_s.sqrt().to(sd.dtype) * x0_hat
                           + (1 - alpha_s).sqrt().to(sd.dtype) * epsilon_ref.to(sd.dtype))  # ⓓ
                    noise_uc_s, noise_c_s = sd.predict_noise(x_s, s_target, uc, c)  # ⓔ UNet
                    eps_s = cfg_combine(noise_uc_s, noise_c_s, s_idx)

                    B = eps_s.shape[0]
                    memo_proxy = (epsilon_ref.to(sd.dtype) - eps_s).reshape(B, -1).pow(2).mean(-1)  # (B,)
                    loss_2 = memo_proxy.mean()

                    z_flat = zt_leaf.reshape(B, -1).float()                    # (B, D)
                    D_dim = z_flat.shape[-1]
                    # 프롬pt 블록별 pairwise — batch_txt>1에서도 프롬pt끼리 인력 오염 방지
                    # (행 순서 = prompt-major: [p0의 S개 seed, p1의 S개 seed, ...])
                    S_pp = seeds_per_prompt if seeds_per_prompt is not None else B
                    assert B % S_pp == 0, f"on_main: B={B} not a multiple of seeds_per_prompt={S_pp}"
                    _l1_blocks = []
                    for _n in range(B // S_pp):
                        _zb = z_flat[_n * S_pp:(_n + 1) * S_pp]
                        _d2b = torch.cdist(_zb, _zb).pow(2)
                        _iub = torch.triu_indices(S_pp, S_pp, offset=1)
                        _l1_blocks.append((_d2b[_iub[0], _iub[1]] / D_dim).sum())
                    loss_1 = sum(_l1_blocks) if len(_l1_blocks) > 1 else _l1_blocks[0]
                    spread_now = loss_1.item()                                  # 관측용 (프롬pt 블록 합)
                    if on_main_spread:
                        loss_1 = -loss_1

                    loss = loss_1 + w_twd * loss_2
                    loss_memo = loss                              # print/record alias
                elif xt_loss == "eps_ref_mse":
                    # ⑤: L = ‖ε_ref − ε_cfg(z_{t-1}, t-1)‖²/D — 재포워드 없이 score 직접 (UNet 1회)
                    #    양수 minimize (2026-09-05 복원 — 09-01부터 무문서화 음수(maximize) 유입,
                    #    명세·측정 방향(L 높음=memorized)과 모순되어 제거)
                    B = eps_p.shape[0]
                    memo_proxy = (epsilon_ref.to(sd.dtype) - eps_p).reshape(B, -1).pow(2).mean(-1)  # (B,)
                    loss = memo_proxy.mean()  # 단순 MSE — threshold·compute_memo_loss 없음
                    loss_memo = loss          # record용 alias
                    x_s = x0_hat  # ⑤는 x_s 포워드 없음 — record 열 placeholder
                    eps_s = eps_p  # record(eps_s_norm) 용 placeholder — ⑤에서는 ε_cfg 기준
                elif xt_loss in ("spec_low", "spec_high"):
                    # ⑦/⑩: L = (1/B)Σ_{band} E_p — ε_cfg(z_{t-1}, t-1) 스펙트럼 block energy
                    #    대역 합 (minimize). ⑦ low: 앞 L_lh개 block — memo 58→101 vs text 52→4
                    #    (측정 2026-09-05) | ⑩ high: 뒤 L_lh개 block — ⑦의 고주파 대칭 변형
                    #    (2026-09-07). UNet 1회(ⓑ) + FFT — 재포워드 없음 (⑤ 패턴)
                    _band = "low" if xt_loss == "spec_low" else "high"
                    memo_proxy = spectral_low_band_energy(eps_p, block_size, lh_ratio, band=_band)  # (B,)
                    loss = memo_proxy.mean()   # 단순 minimization — threshold·compute_memo_loss 없음
                    loss_memo = loss           # record용 alias
                    x_s = x0_hat  # ⑦/⑩은 x_s 포워드 없음 — record 열 placeholder (⑤ 패턴)
                    eps_s = eps_p  # record(eps_s_norm) 용 placeholder — ⑦/⑩에서는 ε_cfg 기준
                elif xt_loss == "twd_loss":
                    # ⑬: L = w_twd_mag·loss_twd_mag + w_twd_std·loss_twd_std
                    #   proxy^i = ‖ε_ref − ε_s^i‖²/D      (per-seed Tweedie gap, (B,))
                    #   loss_twd_mag = Σ_i proxy^i          (proxy 합 최소화 — memo_proxy 변형)
                    #   loss_twd_std = Σ_{i<j} |proxy^i − proxy^j|  (seed 간 proxy 편차 최소화)
                    # batch = prompt-major: [p0의 S개 seed, p1의 S개 seed, ...]
                    # — 프롬pt 블록별 집계 (on_main ⑥ 패턴 승계, 프롬pt 간 오염 없음)
                    # UNet 2회(ⓑ+ⓔ) — memo_proxy 와 동일 chain, loss 집계 방식만 다름
                    x_s = (alpha_s.sqrt().to(sd.dtype) * x0_hat
                           + (1 - alpha_s).sqrt().to(sd.dtype) * epsilon_ref.to(sd.dtype))  # ⓓ
                    noise_uc_s, noise_c_s = sd.predict_noise(x_s, s_target, uc, c)  # ⓔ UNet
                    eps_s = cfg_combine(noise_uc_s, noise_c_s, s_idx)

                    B = eps_s.shape[0]
                    proxy = (epsilon_ref.to(sd.dtype) - eps_s).reshape(B, -1).pow(2).mean(-1)  # (B,)

                    S_pp = seeds_per_prompt if seeds_per_prompt is not None else B
                    assert B % S_pp == 0, f"twd_loss: B={B} not divisible by seeds_per_prompt={S_pp}"

                    _loss_twd_mag = 0.0
                    _loss_twd_std = 0.0
                    for _n in range(B // S_pp):
                        _pr = proxy[_n * S_pp:(_n + 1) * S_pp]  # (S_pp,)
                        _loss_twd_mag = _loss_twd_mag + _pr.sum()
                        if S_pp >= 2:
                            _diff = (_pr.unsqueeze(1) - _pr.unsqueeze(0)).abs()  # (S_pp, S_pp)
                            _iub = torch.triu_indices(S_pp, S_pp, offset=1)
                            _loss_twd_std = _loss_twd_std + _diff[_iub[0], _iub[1]].sum()

                    loss = w_twd_mag * _loss_twd_mag + w_twd_std * _loss_twd_std
                    memo_proxy = proxy   # (B,) — loss와 공유 grad 그래프, print/record 열 재사용
                    loss_memo = loss     # record alias
                elif xt_loss == "latent_loss":
                    # ⑫: L = w_mag·loss_mag + w_std·loss_std
                    #   loss_mag = Σ_i ‖z_t^i‖₂           (per-sample L2 norm 합 — magnitude 최소화)
                    #   loss_std = Σ_{i<j} ‖z_t^i−z_t^j‖₂ (pairwise L2 distance 합 — seed 간 응집/분리)
                    # batch = prompt-major: [p0의 S개 seed, p1의 S개 seed, ...]
                    # — 프롬pt 블록별 pairwise (프롬pt 간 오염 없음, on_main ⑥ 패턴 승계)
                    # UNet 1회(ⓑ)는 이미 수행됨 — 이 갈래에서는 z_{t-1}(zt_leaf) 직접 사용
                    B = zt_leaf.shape[0]
                    z_flat = zt_leaf.reshape(B, -1).float()  # (B, D)

                    S_pp = seeds_per_prompt if seeds_per_prompt is not None else B
                    assert B % S_pp == 0, f"latent_loss: B={B} not divisible by seeds_per_prompt={S_pp}"

                    _loss_mag = 0.0
                    _loss_std = 0.0
                    for _n in range(B // S_pp):
                        _zb = z_flat[_n * S_pp:(_n + 1) * S_pp]  # (S_pp, D)
                        _loss_mag = _loss_mag + _zb.norm(dim=1).sum()
                        if S_pp >= 2:
                            _iub = torch.triu_indices(S_pp, S_pp, offset=1)
                            _loss_std = _loss_std + torch.cdist(_zb, _zb)[_iub[0], _iub[1]].sum()

                    loss = w_mag * _loss_mag + w_std * _loss_std
                    loss_memo = loss       # record alias
                    # per-sample proxy: magnitude (print/csv 열 재사용)
                    memo_proxy = zt_leaf.detach().reshape(B, -1).norm(dim=1)  # (B,)
                    x_s = x0_hat  # 재포워드 없음 — record 열 placeholder
                    eps_s = eps_p  # record(eps_s_norm) 용 placeholder

                elif xt_loss == "ecf_loss":
                    # ⑮ ECF: zt_leaf(= z_{t-1}, DDIM denoised) 에 직접 ECF loss
                    # z_t → z_{t-1}(zt_leaf) 는 Step 1 no_grad 에서 이미 수행됨
                    _B = zt_leaf.shape[0]
                    x_flat = zt_leaf.reshape(_B, -1).float()     # (B, D)
                    _mu  = x_flat.mean(dim=1, keepdim=True)
                    _sig = x_flat.std(dim=1, keepdim=True).clamp(min=1e-8)
                    z_flat  = (x_flat - _mu) / _sig              # per-sample 정규화
                    w_k = (torch.arange(1, ecf_K + 1, dtype=torch.float32, device=device)
                           * (ecf_w_max / ecf_K))                # (K,)
                    angles  = w_k.view(-1, 1, 1) * z_flat.unsqueeze(0)  # (K, B, D)
                    A_emp   = angles.cos().mean(dim=(1, 2))      # (K,)
                    B_emp   = angles.sin().mean(dim=(1, 2))      # (K,)
                    A_theo  = (-0.5 * w_k.pow(2)).exp()
                    ecf = (w_k * ((A_emp - A_theo).pow(2) + B_emp.pow(2))).sum() / w_k.sum()
                    loss = ecf
                    loss_memo = loss
                    memo_proxy = torch.full((_B,), ecf.detach().item(), device=device)
                    x_s  = x0_hat   # record placeholder
                    eps_s = eps_p   # record placeholder

                elif xt_loss == "ecf_score":
                    # ⑯ ECF-score: zt_leaf → UNet(ⓑ) → ε_c(_nc_p) → per-sample normalize → ECF
                    #   → loss → ∇ 를 UNet 통과해 zt_leaf 로 전파
                    # score의 Gaussianity 강제 (⑮ ecf_loss의 latent 직접 버전과 대비)
                    # UNet 1회(ⓑ) — _nc_p 는 위 line 에서 이미 with grad 로 계산됨 (추가 호출 없음)
                    _B = _nc_p.shape[0]
                    x_flat = _nc_p.reshape(_B, -1).float()       # (B, D)
                    _mu  = x_flat.mean(dim=1, keepdim=True)
                    _sig = x_flat.std(dim=1, keepdim=True).clamp(min=1e-8)
                    z_flat  = (x_flat - _mu) / _sig              # per-sample 정규화
                    w_k = (torch.arange(1, ecf_K + 1, dtype=torch.float32, device=device)
                           * (ecf_w_max / ecf_K))                # (K,)
                    angles  = w_k.view(-1, 1, 1) * z_flat.unsqueeze(0)  # (K, B, D)
                    A_emp   = angles.cos().mean(dim=(1, 2))      # (K,)
                    B_emp   = angles.sin().mean(dim=(1, 2))      # (K,)
                    A_theo  = (-0.5 * w_k.pow(2)).exp()
                    ecf = (w_k * ((A_emp - A_theo).pow(2) + B_emp.pow(2))).sum() / w_k.sum()
                    loss = ecf
                    loss_memo = loss
                    memo_proxy = torch.full((_B,), ecf.detach().item(), device=device)
                    x_s  = x0_hat   # record placeholder
                    eps_s = eps_p   # record placeholder

                elif xt_loss == "latent_mag_loss":
                    # ⑰ latent_mag_loss: L = mean_b(‖zt_leaf^b‖₂) — latent norm 최소화
                    # UNet 없음 — zt_leaf 에 직접 gradient
                    _B = zt_leaf.shape[0]
                    mag = zt_leaf.reshape(_B, -1).norm(dim=1)    # (B,)
                    loss = mag.mean()
                    loss_memo = loss
                    memo_proxy = mag.detach()                     # (B,)
                    x_s  = x0_hat   # record placeholder
                    eps_s = eps_p   # record placeholder

                else:
                    x_s = (alpha_s.sqrt().to(sd.dtype) * x0_hat
                           + (1 - alpha_s).sqrt().to(sd.dtype) * epsilon_ref.to(sd.dtype))  # ⓓ
                    noise_uc_s, noise_c_s = sd.predict_noise(x_s, s_target, uc, c)  # ⓔ UNet
                    eps_s = cfg_combine(noise_uc_s, noise_c_s, s_idx)

                    B = eps_s.shape[0]
                    memo_proxy = (epsilon_ref.to(sd.dtype) - eps_s).reshape(B, -1).pow(2).mean(-1)  # (B,)
                    loss_memo = compute_memo_loss(memo_proxy, type_memo_loss, memo_threshold)
                    loss = loss_memo  # align 없음 — memo loss 만

            _loss_tag = (f"[memo_loss={type_memo_loss},τ={memo_threshold}]"
                         if type_memo_loss == "threshold" else "[memo_loss=minimization]")
            print(f"{_loss_tag} proxy(↓=mitigate): {memo_proxy.mean().item():.6f}  "
                  f"loss: {loss_memo.item():.6f}")

            if on_main:
                # 리뷰 반영: 두 항의 grad 기여 분리 로깅 (lr/w_twd sweep 해석용)
                _g1_n = torch.autograd.grad(loss_1, zt_leaf, retain_graph=True)[0].norm().item()
                _g2_n = torch.autograd.grad(w_twd * loss_2, zt_leaf, retain_graph=True)[0].norm().item()
            loss.backward()
            if grad_prcd and zt_leaf.grad is not None:
                with torch.no_grad():
                    zt_leaf.grad = project_wgnc_batched(zt_leaf.grad.detach()).reshape_as(zt_leaf)
            optimizer.step()  # ★ z_{t-1} ← Adam(z_{t-1}, ∇_{z_{t-1}} L)
            total_loss += loss.item()

            _g_norm = zt_leaf.grad.norm().item() if zt_leaf.grad is not None else float('nan')
            print(f"    [opt-xt {ui + 1}/{len(update_indices)}] t_idx={t_idx} "
                  f"memo={loss_memo.item():.6f} "
                  f"|g_xt|={_g_norm:.4f} "
                  f"|dxt|={(zt_leaf.detach() - z_prev0).norm().item():.4f}"
                  + (f" spread={spread_now:.4f} twd={loss_2.item():.6f} "
                     f"|g1|={_g1_n:.4f} |g2|={_g2_n:.4f}" if on_main else ""))

            # ---- record: (x_t, x_0|t, x_s) 누적 + loss.csv append (기존 패턴) ----
            if batch_dirs is not None:
                import csv as _csv
                with torch.no_grad():
                    _vae_dtype = next(sd.vae.parameters()).dtype
                    def _dec(z):
                        return (sd.decode(z.detach().to(_vae_dtype)) / 2 + 0.5).clamp(0, 1).cpu()
                    img_xt = _dec(zt_leaf.detach())
                    img_x0t = _dec(x0_hat)
                    img_xs = _dec(x_s)
                    B_ = img_xt.shape[0]
                    eps_s_n = eps_s.detach().reshape(B_, -1).norm(dim=-1)
                    eps_ref_n = epsilon_ref.detach().to(sd.dtype).reshape(B_, -1).norm(dim=-1)
                    for b in range(B_):
                        batch_record_imgs[b].extend([img_xt[b], img_x0t[b], img_xs[b]])
                        _csv_path = os.path.join(batch_dirs[b], "loss.csv")
                        _wh = not os.path.exists(_csv_path)
                        with open(_csv_path, "a", newline="") as _f:
                            _w = csv.writer(_f)
                            if _wh:
                                _w.writerow(["update_step", "t_idx", "eps_s_norm", "eps_ref_norm",
                                             "memo_proxy", "loss_memo", "loss"])
                            _w.writerow([ui, t_idx, f"{eps_s_n[b].item():.6f}",
                                         f"{eps_ref_n[b].item():.6f}", f"{memo_proxy[b].item():.6f}",
                                         f"{loss_memo.item():.6f}", f"{loss.item():.6f}"])

        # ---- Step 3: 갱신된 z_{t-1} 에서 gap 전진 (no grad) → 다음 update 위치로 ----
        zt = zt_leaf.detach().clone()
        del zt_leaf, optimizer, z_prev0, x0_hat_t, eps_t, x0_hat, x_s, eps_s, eps_p
        del _nu_t, _nc_t, _nu_p, _nc_p
        if xt_loss == "twd_target":
            del noise_uc_s, noise_c_s
        torch.cuda.empty_cache()

        with torch.no_grad():
            for step_idx in range(t_idx + 2, min(t_idx + 1 + gap_steps, len(timesteps))):
                t = timesteps[step_idx]
                at = sd.alpha(t)
                at_prev = sd.alpha(t - sd.skip)
                noise_uc, noise_c = sd.predict_noise(zt, t, uc, c)
                eps_theta = cfg_combine(noise_uc, noise_c, step_idx)
                x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()
                zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta
        _last_processed_tidx = t_idx   # 매 정상 반복 종료 후 갱신

    if xt_loss == "twd_target" and _twd_m_final is not None:
        print(f"\n[twd_target DONE] final twd_mean={_twd_m_final:.4f}  twd_std={_twd_s_final:.4f}"
              f"  (threshold: mean<{twd_mean_threshold}, std<{twd_std_threshold})\n")

    last_forwarded = min(_last_processed_tidx + gap_steps, len(timesteps) - 1)

    # ---- record: grid.png (row=update step, col=x_t, x_0|t, x_s) ----
    if batch_record_imgs is not None:
        from torchvision.utils import make_grid as _make_grid, save_image as _save_image
        for b in range(len(batch_record_imgs)):
            if batch_record_imgs[b]:
                _save_image(_make_grid(batch_record_imgs[b], nrow=3),
                            os.path.join(batch_dirs[b], "grid.png"))

    zt_final = zt.detach().clone()
    sd.unet.half()
    sd.dtype = _orig_dtype

    return zt_final, last_forwarded, total_loss


@torch.no_grad()
def _recompute_to(sd, x_T, timesteps, k, uc, c, cfg, init_noise_sigma):
    """
    Recompute x_k from x_T by running k DDIM steps under no_grad.
    Only used when cache_latents=False (§5 edge case). Trades ~2x UNet
    calls for O(1) memory.
    """
    x = x_T.detach() * init_noise_sigma
    for step_idx, t in enumerate(timesteps):
        if step_idx == k:
            return x.clone()
        at = sd.alpha(t)
        at_prev = sd.alpha(t - sd.skip)
        noise_uc, noise_c = sd.predict_noise(x, t, uc, c)
        eps = noise_uc + cfg_eff_at(sd, step_idx, cfg) * (noise_c - noise_uc)
        x0h = (x - (1 - at).sqrt() * eps) / at.sqrt()
        x = at_prev.sqrt() * x0h + (1 - at_prev).sqrt() * eps
    return x.clone()


@torch.no_grad()
def ddim_inference(sd, x_T, uc, c, cfg):
    """Standard DDIM from given x_T."""
    zt = x_T.to(sd.dtype) * sd.scheduler.init_noise_sigma
    for step_idx, t in enumerate(sd.scheduler.timesteps):
        at = sd.alpha(t)
        at_prev = sd.alpha(t - sd.skip)
        noise_uc, noise_c = sd.predict_noise(zt, t, uc, c)
        eps_theta = noise_uc + cfg * (noise_c - noise_uc)
        x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()
        zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta
    img = sd.decode(x0_hat)
    return (img / 2 + 0.5).clamp(0, 1)


@torch.no_grad()
def ddim_inference_with_proxy(sd, x_T, uc, c, cfg, base_s_ratio,
                              record_dir, prompt_tag, filename="memo_proxy_ddim.png"):
    """x_T opti 종료 후 DDIM sampling 하며 모든 denoising step 에서 memo_proxy 수집.

    memo_proxy(t) = || ε_ref - ε_s(x_s(t), s) ||^2 / D
      - ε_ref = fresh random noise (x_T 와 독립) — proxy reference
      - x̂₀|t  = Tweedie(x_t, t)
      - x_s    = √ᾱ_s · x̂₀|t + √(1-ᾱ_s) · x_T_noise,  s = 고정 mid-noise (base_s_ratio)
                 (x_T_noise = 최적화된 x_T, DDIM 입력 noise — forward consistency 유지)
      - ε_s    = ε_θ(x_s, s)
    ε_ref 가 x_T 와 독립이므로 self-referential 0-수렴이 없고, ε_s 가 memorized
    direction 으로 쏠릴수록 proxy 가 커짐 → memorization 신호.

    batch(= num_seeds) 차원으로 mean ± std 를 계산해 plot + csv → record_dir/.
    record_dir=None 이면 plot/csv 저장 없이 생성만 (batch_txt>1 모드).
    """
    if record_dir is not None:
        os.makedirs(record_dir, exist_ok=True)
    timesteps = list(sd.scheduler.timesteps)
    s_idx = int(len(timesteps) * base_s_ratio)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)

    x_T_noise = x_T.detach().to(sd.dtype)             # 최적화된 x_T = DDIM 입력 noise (x_s forward 용)
    epsilon_ref = torch.randn_like(x_T).to(sd.dtype)  # proxy reference (fresh random, x_T 와 독립)
    zt = x_T.to(sd.dtype) * sd.scheduler.init_noise_sigma

    step_indices = []
    memo_proxy_per_step = []                        # (B,) numpy per step

    for step_idx, t in enumerate(timesteps):
        at = sd.alpha(t)
        at_prev = sd.alpha(t - sd.skip)
        noise_uc, noise_c = sd.predict_noise(zt, t, uc, c)
        eps_theta = noise_uc + cfg * (noise_c - noise_uc)
        x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()

        # ---- memo proxy at this step (eps_trajectory.py:107-116 과 동일) ----
        x_s = (alpha_s.sqrt().to(sd.dtype) * x0_hat
               + (1 - alpha_s).sqrt().to(sd.dtype) * epsilon_ref)
        noise_uc_s, noise_c_s = sd.predict_noise(x_s, s_target, uc, c)
        eps_s = noise_uc_s + cfg_eff_at(sd, s_idx, cfg) * (noise_c_s - noise_uc_s)
        B = eps_s.shape[0]
        memo_proxy = (epsilon_ref - eps_s).reshape(B, -1).pow(2).mean(-1)   # (B,)
        memo_proxy_per_step.append(memo_proxy.float().cpu().numpy())
        step_indices.append(step_idx)

        # DDIM step (η=0)
        zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta

    proxy_arr = np.stack(memo_proxy_per_step, axis=0)   # (num_steps, B)
    mean = proxy_arr.mean(axis=1)
    std = proxy_arr.std(axis=1)
    steps = np.asarray(step_indices)

    if record_dir is not None:
        # ---- plot: batch mean ± std (+개별 batch faint) ----
        fig, ax = plt.subplots(figsize=(10, 6))
        for b in range(B):
            ax.plot(steps, proxy_arr[:, b], color="tab:blue", alpha=0.12, linewidth=0.8)
        ax.plot(steps, mean, color="tab:blue", linewidth=2.2, marker="o", markersize=4,
                label="mean memo_proxy")
        ax.fill_between(steps, mean - std, mean + std, color="tab:blue", alpha=0.2,
                        label=f"±1 std (batch n={B})")
        ax.set_xlabel("Denoising Step", fontsize=12)
        ax.set_ylabel(r"memo_proxy  $\|\,\epsilon_{ref} - \epsilon_s\,\|^2 / D$", fontsize=12)
        ax.set_title(
            f"memo_proxy vs denoising step (post x_T opti) — {prompt_tag}\n"
            f"batch mean ± std (n={B}, s_idx={s_idx}, base_s_ratio={base_s_ratio})",
            fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plot_path = os.path.join(record_dir, filename)
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[plot] memo_proxy (post-opti DDIM) -> {plot_path}")

        # ---- csv: step, mean, std, per-sample ----
        csv_path = os.path.join(record_dir, "memo_proxy_ddim.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "mean", "std"] + [f"sample_{b:02d}" for b in range(B)])
            for si, m, sd_v, row in zip(steps, mean, std, proxy_arr):
                w.writerow([int(si), f"{m:.6f}", f"{sd_v:.6f}"]
                           + [f"{v:.6f}" for v in row])
        print(f"[csv]  memo_proxy (post-opti DDIM) -> {csv_path}")

        # ---- twd_mean / twd_std explicit plot ----
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(steps, mean, color="tab:blue", linewidth=2.2, marker="o", markersize=4)
        ax1.set_ylabel(r"twd_mean  $E[\|\epsilon_{ref}-\epsilon_s\|^2/D]$", fontsize=11)
        ax1.set_title(
            f"twd_mean / twd_std vs denoising step — {prompt_tag}\n"
            f"(n={B}, s_idx={s_idx}, base_s_ratio={base_s_ratio})",
            fontsize=11)
        ax1.set_ylim(0, 5)
        ax1.grid(True, alpha=0.3)
        ax2.plot(steps, std, color="tab:orange", linewidth=2.2, marker="s", markersize=4)
        ax2.set_xlabel("Denoising Step", fontsize=12)
        ax2.set_ylabel(r"twd_std  $\mathrm{Std}[\|\epsilon_{ref}-\epsilon_s\|^2/D]$", fontsize=11)
        ax2.set_ylim(0, 1.2)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        twd_plot_path = os.path.join(record_dir, "twd_gap_inference.png")
        plt.savefig(twd_plot_path, dpi=150)
        plt.close()
        print(f"[plot] twd_mean/twd_std (post-opti DDIM) -> {twd_plot_path}")

        # ---- twd csv: step, twd_mean, twd_std ----
        twd_csv_path = os.path.join(record_dir, "twd_gap_inference.csv")
        with open(twd_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "twd_mean", "twd_std"])
            for si, m, sd_v in zip(steps, mean, std):
                w.writerow([int(si), f"{m:.6f}", f"{sd_v:.6f}"])
        print(f"[csv]  twd_mean/twd_std (post-opti DDIM) -> {twd_csv_path}")
    else:
        plot_path = None

    img = sd.decode(x0_hat)
    return (img / 2 + 0.5).clamp(0, 1), plot_path


@torch.no_grad()
def ddim_inference_with_proxy_from(sd, zt, start_idx, uc, c, cfg, base_s_ratio,
                                   record_dir, prompt_tag, filename="memo_proxy_ddim.png"):
    """optimize_xt(1-1) 용 resume inference — 중간 latent zt 에서 DDIM 재개하며 남은
    denoising step 의 memo_proxy 를 수집 (plot/csv 구조는 ddim_inference_with_proxy 와 동일).

    zt = optimize_xt 의 zt_final (start_idx step 까지 전진 완료 상태) —
    step start_idx+1 부터 NFE-1 까지 샘플링.
    """
    if record_dir is not None:
        os.makedirs(record_dir, exist_ok=True)
    timesteps = list(sd.scheduler.timesteps)
    s_idx = int(len(timesteps) * base_s_ratio)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)

    epsilon_ref = torch.randn_like(zt).to(sd.dtype)  # proxy reference (fresh random)
    zt = zt.to(sd.dtype)

    step_indices = []
    memo_proxy_per_step = []

    for step_idx in range(start_idx + 1, len(timesteps)):
        t = timesteps[step_idx]
        at = sd.alpha(t)
        at_prev = sd.alpha(t - sd.skip)
        noise_uc, noise_c = sd.predict_noise(zt, t, uc, c)
        eps_theta = noise_uc + cfg * (noise_c - noise_uc)
        x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()

        # ---- memo proxy at this step (ddim_inference_with_proxy 와 동일) ----
        x_s = (alpha_s.sqrt().to(sd.dtype) * x0_hat
               + (1 - alpha_s).sqrt().to(sd.dtype) * epsilon_ref)
        noise_uc_s, noise_c_s = sd.predict_noise(x_s, s_target, uc, c)
        eps_s = noise_uc_s + cfg_eff_at(sd, s_idx, cfg) * (noise_c_s - noise_uc_s)
        B = eps_s.shape[0]
        memo_proxy = (epsilon_ref - eps_s).reshape(B, -1).pow(2).mean(-1)   # (B,)
        memo_proxy_per_step.append(memo_proxy.float().cpu().numpy())
        step_indices.append(step_idx)

        # DDIM step (η=0)
        zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta

    proxy_arr = np.stack(memo_proxy_per_step, axis=0)   # (num_steps, B)
    mean = proxy_arr.mean(axis=1)
    std = proxy_arr.std(axis=1)
    B = proxy_arr.shape[1]
    steps = np.asarray(step_indices)

    if record_dir is not None:
        # ---- plot: batch mean ± std (+개별 batch faint) ----
        fig, ax = plt.subplots(figsize=(10, 6))
        for b in range(B):
            ax.plot(steps, proxy_arr[:, b], color="tab:blue", alpha=0.12, linewidth=0.8)
        ax.plot(steps, mean, color="tab:blue", linewidth=2.2, marker="o", markersize=4,
                label="mean memo_proxy")
        ax.fill_between(steps, mean - std, mean + std, color="tab:blue", alpha=0.2,
                        label=f"±1 std (batch n={B})")
        ax.set_xlabel("Denoising Step", fontsize=12)
        ax.set_ylabel(r"memo_proxy  $\|\,\epsilon_{ref} - \epsilon_s\,\|^2 / D$", fontsize=12)
        ax.set_title(
            f"memo_proxy vs denoising step (post x_t opti, resume@{start_idx + 1}) — {prompt_tag}\n"
            f"batch mean ± std (n={B}, s_idx={s_idx}, base_s_ratio={base_s_ratio})",
            fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plot_path = os.path.join(record_dir, filename)
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[plot] memo_proxy (post-opti DDIM, resume) -> {plot_path}")

        csv_path = os.path.join(record_dir, "memo_proxy_ddim.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "mean", "std"] + [f"sample_{b:02d}" for b in range(B)])
            for si, m, sd_v, row in zip(steps, mean, std, proxy_arr):
                w.writerow([int(si), f"{m:.6f}", f"{sd_v:.6f}"]
                           + [f"{v:.6f}" for v in row])
        print(f"[csv]  memo_proxy (post-opti DDIM, resume) -> {csv_path}")

        # ---- twd_mean / twd_std explicit plot ----
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(steps, mean, color="tab:blue", linewidth=2.2, marker="o", markersize=4)
        ax1.set_ylabel(r"twd_mean  $E[\|\epsilon_{ref}-\epsilon_s\|^2/D]$", fontsize=11)
        ax1.set_title(
            f"twd_mean / twd_std vs denoising step (resume@{start_idx + 1}) — {prompt_tag}\n"
            f"(n={B}, s_idx={s_idx}, base_s_ratio={base_s_ratio})",
            fontsize=11)
        ax1.set_ylim(0, 5)
        ax1.grid(True, alpha=0.3)
        ax2.plot(steps, std, color="tab:orange", linewidth=2.2, marker="s", markersize=4)
        ax2.set_xlabel("Denoising Step", fontsize=12)
        ax2.set_ylabel(r"twd_std  $\mathrm{Std}[\|\epsilon_{ref}-\epsilon_s\|^2/D]$", fontsize=11)
        ax2.set_ylim(0, 1.2)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        twd_plot_path = os.path.join(record_dir, "twd_gap_inference.png")
        plt.savefig(twd_plot_path, dpi=150)
        plt.close()
        print(f"[plot] twd_mean/twd_std (post-opti DDIM, resume) -> {twd_plot_path}")

        twd_csv_path = os.path.join(record_dir, "twd_gap_inference.csv")
        with open(twd_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "twd_mean", "twd_std"])
            for si, m, sd_v in zip(steps, mean, std):
                w.writerow([int(si), f"{m:.6f}", f"{sd_v:.6f}"])
        print(f"[csv]  twd_mean/twd_std (post-opti DDIM, resume) -> {twd_csv_path}")
    else:
        plot_path = None

    img = sd.decode(x0_hat)
    return (img / 2 + 0.5).clamp(0, 1), plot_path


def load_prompts(prompt_dir, num_samples):
    """prompt 파일에서 앞 num_samples개 prompt를 순서대로 로드."""
    with open(prompt_dir, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    return prompts[:num_samples]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opti_mode", type=str, default="xT", choices=["xT", "xt"],
                   help="최적화 대상 변수: xT(기본) = optimize_xT_adj — x_T(initial noise) 갱신, "
                        "grad 가 DDIM 연쇄(ⓐ)를 타고 x_T 까지 (AdjointDPM) | "
                        "xt = optimize_xt — 중간 latent x_t 갱신 (registry 1-1), "
                        "grad 는 ⓑ~ⓔ terminal head 만, x_T 불변")
    p.add_argument("--NFE", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--cfg_start_ratio", type=float, default=0.0,
                   help="denoising step 중 CFG 적용 시작 ratio. "
                        "step_idx < ratio*NFE 동안은 null(unconditional)만 사용하고, "
                        "그 이후 step부터 정상 CFG. 0.0 = 항상 CFG (기존 동작). 예: 0.3 = 초반 30%% null")
    p.add_argument("--cfgsr_cond", action="store_true",
                   help="cfgsr 구간(step < cfg_start_ratio*NFE)의 eps 를 null(cfg=0) 대신 "
                        "conditional(text prompt, cfg=1) 로 사용")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--init_steps", type=int, default=10,
                   help="DDIM steps before first gradient update")
    p.add_argument("--gap_steps", type=int, default=3,
                   help="Interval between updates")
    p.add_argument("--num_steps", type=int, default=4,
                   help="Number of gradient updates")
    p.add_argument("--base_s_ratio", type=float, default=0.5)
    p.add_argument("--lambda_align", type=float, default=0.1,
                   help="Weight for text alignment regularization")
    p.add_argument("--type_memo_loss", type=str, default="minimization",
                   choices=["minimization", "threshold", "accumulate"],
                   help="minimization | threshold | accumulate "
                        "(여러 step 의 proxy 를 누적해서 loss)")
    p.add_argument("--memo_threshold", type=float, default=0.3,
                   help="τ for type_memo_loss=threshold (memo_proxy=||ε-ε_s||²/D 스케일; "
                        "memorized proxy 전형적으로 ~0.1-0.5, 모델/prompt 마다 튜닝)")
    p.add_argument("--grad_prcd", action="store_true",
                   help="GPER gradient preconditioning (arXiv:2602.08646): project grad onto "
                        "white Gaussian noise feasible set (WGNC, block_size=16) before Adam step")
    p.add_argument("--xt_loss", type=str, default="memo_proxy",
                   choices=["memo_proxy", "eps_ref_mse", "spec_low", "spec_high",
                            "latent_loss", "twd_loss", "twd_target", "ecf_loss", "ecf_score",
                            "latent_mag_loss"],
                   help="opti_mode=xt 의 loss 갈래: memo_proxy(① ‖ε_ref−ε_s‖²/D, 재포워드) | "
                        "eps_ref_mse(⑤ ‖ε_ref−ε_cfg(z_{t-1},t-1)‖²/D — x_s 재포워드 없이 "
                        "score 직접, UNet 1회) | "
                        "spec_low(⑦ ε_cfg 저주파 block energy 합 (1/B)Σ_{p<L_lh}E_p — "
                        "rfft(ortho) → block → low-band sum, UNet 1회) | "
                        "spec_high(⑩ ε_cfg 고주파 block energy 합 (1/B)Σ_{p≥P−L_lh}E_p — "
                        "⑦의 고주파 대칭 변형, 뒤 int(P·LH_ratio)개 block, UNet 1회) | "
                        "latent_loss(⑫ w_mag·Σ‖z_t^i‖ + w_std·Σ_{i<j}‖z_t^i−z_t^j‖ — "
                        "latent magnitude/seed 다양성 직접 제어, UNet 1회(ⓑ)만) | "
                        "twd_loss(⑬ w_twd_mag·Σproxy^i + w_twd_std·Σ_{i<j}|proxy^i−proxy^j| — "
                        "Tweedie gap proxy 합+편차 제어, UNet 2회(ⓑ+ⓔ))")
    p.add_argument("--lh_ratio", type=float, default=0.1,
                   help="⑦/⑩ spec_low·spec_high 의 대역 선택 비율 LH_ratio (block 총 수 P "
                        "기준 — low: 앞 int(P·LH_ratio)개 / high: 뒤 int(P·LH_ratio)개 block "
                        "선택, 측정 eps_trajectory.py --lh_ratio 와 동일). "
                        "예: 0.1 → P=512 중 51 blocks (WG 기준 ≈51). default 0.1")
    p.add_argument("--w_mag", type=float, default=0.0,
                   help="⑫ latent_loss: loss_mag(Σ_i‖z_t^i‖₂) 가중치. "
                        "0.0 이면 magnitude 항 비활성 (기본). --xt_loss latent_loss 와 함께 사용")
    p.add_argument("--w_std", type=float, default=0.0,
                   help="⑫ latent_loss: loss_std(Σ_{i<j}‖z_t^i−z_t^j‖₂) 가중치. "
                        "최소화 시 seed 뭉침, 음수(-w_std)는 seed 분리 방향. "
                        "--xt_loss latent_loss 와 함께 사용")
    p.add_argument("--w_twd_mag", type=float, default=0.0,
                   help="⑬ twd_loss: loss_twd_mag(Σ_i proxy^i) 가중치. "
                        "proxy^i = ‖ε_ref−ε_s^i‖²/D — seed 별 Tweedie gap 합 최소화. "
                        "--xt_loss twd_loss 와 함께 사용")
    p.add_argument("--w_twd_std", type=float, default=0.0,
                   help="⑬ twd_loss: loss_twd_std(Σ_{i<j}|proxy^i−proxy^j|) 가중치. "
                        "seed 간 proxy 편차 최소화 — 고 proxy seed 를 저 proxy 방향으로 당김. "
                        "--xt_loss twd_loss 와 함께 사용. "
                        "⑭ twd_target: proxy.std() 가중치 (w_twd_std·std(proxy))")
    p.add_argument("--opti_num", type=int, default=20,
                   help="⑭ twd_target: 최대 최적화 반복 횟수 (num_steps 대체). "
                        "twd_mean < twd_mean_threshold AND twd_std < twd_std_threshold "
                        "이면 early stop — 그 전에 opti_num 에 도달하면 종료. default 20")
    p.add_argument("--w_twd_mean", type=float, default=1.0,
                   help="⑭ twd_target: loss_twd_mean(proxy.mean()) 가중치. "
                        "자기참조 proxy 평균을 0으로 당기는 항. --xt_loss twd_target 와 함께 사용")
    p.add_argument("--twd_mean_threshold", type=float, default=0.10,
                   help="⑭ twd_target early stop 조건: twd_mean < 이 값. "
                        "proxy = ‖x_T_init − ε_s‖²/D 스케일 — white-Gaussian 기준 ≈0 (완전 완화). "
                        "memorized 전형값 ~0.1–0.5. default 0.10")
    p.add_argument("--twd_std_threshold", type=float, default=0.05,
                   help="⑭ twd_target early stop 조건: twd_std < 이 값. "
                        "batch 내 seed 간 proxy 산포 — memorized는 크고(seed 별 경로 산발), "
                        "general은 작음. default 0.05")
    p.add_argument("--ecf_K", type=int, default=10,
                   help="⑮ ecf_loss: 주파수 격자 크기 K (w_k = k·w_max/K, k=1..K). default 10")
    p.add_argument("--ecf_w_max", type=float, default=3.0,
                   help="⑮ ecf_loss: 최대 주파수 w_max (격자 = [w_max/K, ..., w_max]). "
                        "N(0,1) CF가 급감하기 전 민감 대역 포함 권장. default 3.0")
    p.add_argument("--block_size", type=int, default=16,
                   help="⑦/⑩ spec_low·spec_high 의 주파수 block 크기 B (측정 체계 기본값과 "
                        "동일). default 16")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--on_main", action="store_true",
                   help="⑥ [On-main] compression: loss = loss_1(seed 인력) + w_twd·Twd_gap "
                        "— loss_1 = Σ_{i<j}‖x^i_t−x^j_t‖²/D 최소화(seed 뭉침, general의 collapse 회복). "
                        "batch 행 = prompt-major seed 블록 (--num_seeds>=2, batch_txt 호환 — 블록 안쪽 쌍만 인력). "
                        "출력은 output_dir 아래 on_main/ 폴더에 저장")
    p.add_argument("--w_twd", type=float, default=1.0,
                   help="Tweedie gap 가중치 — ⑥ on_main: loss_1 + w_twd·loss_2 | "
                        "⑧ anchor: s_Δ + w_twd·memo_proxy")
    p.add_argument("--on_main_spread", action="store_true",
                   help="⑥ loss_1 부호 반전(원식 −Σ 리터럴) — 최소화 시 seed 분리 방향")
    p.add_argument("--anchor", action="store_true",
                   help="⑧ [Anchor: Twd_gap]: loss = s_Δ + w_twd·memo_proxy(①) — "
                        "s_Δ = ‖ε_uc(x_T)−ε_c(x_T)‖₂ (x_T에서 직접, per-sample norm), "
                        "gap 체인 x_T→x̂_{0|T}→x_s→ε_s. x_T 직접 갱신 (adjoint 불필요 — "
                        "UNet 2회 얕은 헤드). init/gap_steps 무시, num_steps=갱신 횟수. "
                        "출력은 output_dir 아래 twd_anchor/ 폴더에 저장")
    p.add_argument("--btw_anchor", action="store_true",
                   help="⑨ [Btw_anchor]: loss = s_Δ + w_btw·Σ_{i<j}‖x_i−x_j‖₂ — "
                        "s_Δ = ‖ε_uc(x_T)−ε_c(x_T)‖₂ (⑧과 동일), btw 항 = x_T pairwise "
                        "L2 norm 합 (최소화 → batch 인력·뭉침 — memorized의 압도적 batch "
                        "std 관측 대응). 프롬pt 블록 안쪽 쌍만 인력 (batch_txt 호환 — "
                        "프롬pt 간 오염 없음, ⑥ 패턴 승계). UNet 1회 얕은 헤드, "
                        "x_T 직접 갱신, --num_seeds≥2 필수. "
                        "출력은 output_dir 아래 btw_anchor/ 폴더에 저장")
    p.add_argument("--w_btw", type=float, default=1.0,
                   help="⑨ btw_anchor: batch pairwise 인력 가중치 (loss = s_Δ + w_btw·btw)")
    p.add_argument("--batch_txt", type=int, default=1,
                   help="한 번의 최적화에 묶을 프롬pt 수 (1=기존 per-prompt 경로). "
                        "batch=num_seeds×batch_txt — 24GB에서는 10 이하 권장")
    p.add_argument("--num_seeds", type=int, default=5,
                   help="images per prompt (different seed each)")
    p.add_argument("--prompt_dir", type=str,
                   default=os.path.join(SCRIPT_DIR, "examples", "assets", "coco_v2.txt"),
                   help="prompt file (default: coco_v2.txt)")
    p.add_argument("--num_samples", type=int, default=10,
                   help="number of prompts to use from prompt_dir")
    p.add_argument("--model_key", type=str, default=os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default=os.path.join(SCRIPT_DIR, "workdir", "ini_opti", "memorized"))
    args = p.parse_args()

    # ---- ⑤/⑦ xt_loss 갈래 가드: memo_proxy 외 loss 는 opti_mode=xt 전용 ----
    # (xT 모드의 optimize_xT_adj은 xt_loss 를 받지 않음 — 조용한 무시 방지)
    if args.xt_loss != "memo_proxy":
        assert args.opti_mode == "xt", \
            f"--xt_loss {args.xt_loss} is for opti_mode=xt only (xT mode is fixed to memo_proxy(①))"

    # ---- ⑥ on_main 가드 + 전용 출력 폴더 ----
    if args.on_main:
        assert args.opti_mode == "xt", "on_main requires opti_mode=xt (reuses Twd_gap head)"
        assert args.num_seeds >= 2, "on_main: num_seeds>=2 required for seed attraction"
        assert args.xt_loss == "memo_proxy", "on_main is based on xt_loss=memo_proxy (① head) — exclusive with eps_ref_mse"
        assert args.type_memo_loss == "minimization", "on_main bypasses compute_memo_loss — threshold/accumulate unsupported"
        if "on_main" not in args.output_dir.split(os.sep):
            args.output_dir = os.path.join(args.output_dir, "on_main")
        print(f"[on_main] w_twd={args.w_twd} spread_mode={args.on_main_spread} → {args.output_dir}")

    # ---- ⑧ anchor 가드 + 전용 출력 폴더 ----
    if args.anchor:
        assert args.opti_mode == "xT", "anchor requires opti_mode=xT (direct initial-noise update)"
        assert args.type_memo_loss == "minimization", "anchor supports minimization only"
        assert not args.on_main, "anchor and on_main are mutually exclusive (on_main is xt-only)"
        if "twd_anchor" not in args.output_dir.split(os.sep):
            args.output_dir = os.path.join(args.output_dir, "twd_anchor")
        print(f"[anchor] w_twd={args.w_twd} num_steps={args.num_steps} "
              f"(direct x_T update, no adjoint) → {args.output_dir}")

    # ---- ⑨ btw_anchor 가드 + 전용 출력 폴더 ----
    if args.btw_anchor:
        assert args.opti_mode == "xT", "btw_anchor requires opti_mode=xT (direct initial-noise update)"
        assert args.type_memo_loss == "minimization", "btw_anchor supports minimization only"
        assert not args.anchor, "btw_anchor and anchor(⑧) are mutually exclusive"
        assert not args.on_main, "btw_anchor and on_main are mutually exclusive (on_main is xt-only)"
        assert args.num_seeds >= 2, "btw_anchor: num_seeds>=2 required for pairwise attraction"
        if "btw_anchor" not in args.output_dir.split(os.sep):
            args.output_dir = os.path.join(args.output_dir, "btw_anchor")
        print(f"[btw] w_btw={args.w_btw} num_steps={args.num_steps} "
              f"(direct x_T update, batch traction) → {args.output_dir}")
    device = torch.device(args.device)

    # ---- 벤치마크 측정 시작 ----
    import time
    comp_rows = []   # (sample_idx, time_per_sample_sec, peak_vram_GB) per inference image
    _total_time_s = 0.0     # 모든 batch 처리 시간 합 (총 소요 시간)
    _total_peak_gb = 0.0    # 전체 실행 중 최대 VRAM (batch peak 들의 max)

    solver_config = munchify({"num_sampling": args.NFE})
    sd = StableDiffusion(solver_config=solver_config, model_key=args.model_key, device=device, seed=args.base_seed)
    sd.unet.enable_gradient_checkpointing()
    sd.cfg_start_ratio = args.cfg_start_ratio   # staged CFG: 초반 ratio*NFE step 동안 null 만, 이후 정상 CFG
    sd.cfgsr_cond = args.cfgsr_cond             # cfgsr 구간 eps: False=null(cfg=0), True=cond(text prompt, cfg=1)

    update_steps = [args.init_steps + i * args.gap_steps for i in range(args.num_steps)]
    _cfg_n = int(args.NFE * args.cfg_start_ratio) if args.cfg_start_ratio > 0 else 0
    print(f"NFE={args.NFE} CFG={args.cfg} cfg_start_ratio={args.cfg_start_ratio} "
          f"(step 0~{_cfg_n-1} null-only) lr={args.lr} update_steps={update_steps}")
    print(f"type_memo_loss={args.type_memo_loss} "
          + (f"memo_threshold={args.memo_threshold}" if args.type_memo_loss == "threshold" else "(full minimization)"))
    print(f"prompt_dir={args.prompt_dir} num_samples={args.num_samples} num_seeds(per prompt)={args.num_seeds}")

    # SNR schedule summary (where Tweedie x0_hat becomes signal-bearing)
    _ts = sd.scheduler.timesteps
    _snr_of = lambda i, t: (sd.alpha(t) / (1 - sd.alpha(t))).item()
    print("[SNR schedule] " + "  ".join(f"s{i}={_snr_of(i, t):.2f}" for i, t in enumerate(_ts) if i % 5 == 0))
    _snr1 = next((i for i, t in enumerate(_ts) if _snr_of(i, t) >= 1.0), None)
    _snr2 = next((i for i, t in enumerate(_ts) if _snr_of(i, t) >= 2.0), None)
    print(f"  -> SNR>=1 at step {_snr1} (base_s_ratio>={_snr1/len(_ts):.2f}),  SNR>=2 at step {_snr2}")

    # load prompts from file (aligned with DDIM/CNO)
    prompts = load_prompts(args.prompt_dir, args.num_samples)

    # flat result dir (sequential naming: idx = i*num_seeds + j, matching text_to_mscoco)
    result_dir = os.path.join(args.output_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    # save ordered prompts for T2I pairing
    with open(os.path.join(args.output_dir, "prompts.txt"), "w") as f:
        for prompt in prompts:
            f.write(prompt + "\n")

    # DDIM(text_to_mscoco)과 동일: set_seed 1회 후 prompt마다 batch randn 연속 (reset X)
    set_seed(args.base_seed)

    # ---- 프롬pt 배치 청킹 (batch_txt=1이면 크기 1 청크 = 기존 per-prompt 경로와
    #      동일 연산 순서·RNG 스트림 유지) ----
    S = args.num_seeds
    for i0 in range(0, len(prompts), max(args.batch_txt, 1)):
        chunk = prompts[i0:i0 + max(args.batch_txt, 1)]
        N = len(chunk)

        # ---- resume: 청크 내 전 프롬pt가 이미 생성됐으면 skip ----
        if all(all(os.path.exists(os.path.join(result_dir, f"img_{i0+n:04d}_{j:02d}.png"))
                   for j in range(S)) for n in range(N)):
            print(f"[{i0+1}~{i0+N}/{len(prompts)}] SKIP ({S} images already exist)")
            continue
        print(f"\n[{i0+1}~{i0+N}/{len(prompts)}] ({[p[:40] for p in chunk]}) "
              f"(seeds={S} × txt={N})")
        if N * S > 10:
            print(f"  [warn] batch={N*S} > 10 — near 24GB VRAM limit (reduce batch_txt if OOM)")

        # text embedding — 프롬pt별 개별 인코딩 후 cat (padding 간섭 없음;
        # tokenizer가 padding='max_length'(77 고정)라 배치/단일 인코딩 결과 동일)
        _embs = [sd.get_text_embed(null_prompt="", prompt=p) for p in chunk]
        uc = _embs[0][0]                                   # null embed — 프롬pt 무관 공용
        uc_batch = uc.repeat(N * S, 1, 1)
        c_batch = torch.cat([c_.repeat(S, 1, 1) for _, c_ in _embs], dim=0)  # prompt-major

        # 초기 noise — prompt별 순차 draw (batch_txt=1이면 기존 스트림과 bit-identical)
        x_T_init_batch = torch.cat(
            [torch.randn(S, 4, 64, 64, device=device, dtype=torch.float32) for _ in range(N)],
            dim=0)
        # ---- per-batch compute cost 측정 시작 (optimize + inference) ----
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        _t0 = time.perf_counter()

        # record: N=1 기존 단일 경로 / N>1 프롬pt별 디렉토리 + row→dir 매핑 주입
        if N == 1:
            record_dir = os.path.join(args.output_dir, "record", f"img_{i0:04d}")
            os.makedirs(record_dir, exist_ok=True)
            record_dirs = None
        else:
            record_dir = None
            record_dirs = []
            for n in range(N):
                _tag = f"img_{i0+n:04d}"
                for j in range(S):
                    _bd = os.path.join(args.output_dir, "record", _tag, f"{_tag}_{j:02d}")
                    os.makedirs(_bd, exist_ok=True)
                    record_dirs.append(_bd)

        # Phase 1+2: optimize (opti_mode 분기) → inference + step-wise memo_proxy 수집
        if args.opti_mode == "xt":
            # 1-1: 중간 latent x_t 최적화 (ⓑ~ⓔ만 grad, x_T 불변) → 이후 step 에서 DDIM 재개
            zt_opt_batch, last_forwarded, loss = optimize_xt(
                sd, uc_batch, c_batch, args.cfg, device,
                args.init_steps, args.num_steps, args.gap_steps, args.lr,
                args.base_s_ratio, args.lambda_align,
                batch_size=N * S,
                record_dir=record_dir, record_dirs=record_dirs,
                type_memo_loss=args.type_memo_loss,
                memo_threshold=args.memo_threshold,
                grad_prcd=args.grad_prcd,
                xt_loss=args.xt_loss,
                on_main=args.on_main,
                w_twd=args.w_twd,
                on_main_spread=args.on_main_spread,
                seeds_per_prompt=args.num_seeds,
                block_size=args.block_size,
                lh_ratio=args.lh_ratio,
                w_mag=args.w_mag,
                w_std=args.w_std,
                w_twd_mag=args.w_twd_mag,
                w_twd_std=args.w_twd_std,
                opti_num=args.opti_num,
                w_twd_mean=args.w_twd_mean,
                twd_mean_threshold=args.twd_mean_threshold,
                twd_std_threshold=args.twd_std_threshold,
                ecf_K=args.ecf_K,
                ecf_w_max=args.ecf_w_max,
            )
            print(f"  x_t_opt[{args.xt_loss}{'/on_main' if args.on_main else ''}]: "
                  f"{zt_opt_batch.shape}  loss: {loss:.4f}  "
                  f"(inference resume from step {last_forwarded + 1})")
            # N>1에선 프롬pt 혼합 mean±std plot이 무의미 → record 저장 없이 생성만
            img_batch, _ = ddim_inference_with_proxy_from(
                sd, zt_opt_batch, last_forwarded, uc_batch, c_batch, args.cfg,
                args.base_s_ratio, record_dir,
                prompt_tag=(f"img_{i0:04d}" if N == 1 else None))
        else:
            if args.anchor:
                # ⑧ anchor: x_T 직접 최적화 (s_Δ + Twd_gap, adjoint 없음)
                x_T_opt_batch, loss = optimize_xT_anchor(
                    sd, uc_batch, c_batch, args.cfg, device,
                    args.num_steps, args.lr, args.base_s_ratio,
                    w_twd=args.w_twd,
                    batch_size=N * S,
                    record_dir=record_dir, record_dirs=record_dirs,
                )
                print(f"  x_T_opt[anchor]: {x_T_opt_batch.shape}  loss: {loss:.4f}")
            elif args.btw_anchor:
                # ⑨ btw_anchor: x_T 직접 최적화 (s_Δ + batch pairwise 인력, adjoint 없음)
                x_T_opt_batch, loss = optimize_xT_btw(
                    sd, uc_batch, c_batch, args.cfg, device,
                    args.num_steps, args.lr,
                    w_btw=args.w_btw,
                    batch_size=N * S,
                    record_dir=record_dir, record_dirs=record_dirs,
                    seeds_per_prompt=args.num_seeds,
                )
                print(f"  x_T_opt[btw_anchor]: {x_T_opt_batch.shape}  loss: {loss:.4f}")
            else:
                # 기존: x_T(initial noise) 최적화 (AdjointDPM) → x_T 부터 전체 DDIM
                x_T_opt_batch, loss = optimize_xT_adj(
                    sd, uc_batch, c_batch, args.cfg, device,
                    args.init_steps, args.num_steps, args.gap_steps, args.lr,
                    args.base_s_ratio, args.lambda_align,
                    batch_size=N * S,
                    record_dir=record_dir, record_dirs=record_dirs,
                    type_memo_loss=args.type_memo_loss,
                    memo_threshold=args.memo_threshold,
                    grad_prcd=args.grad_prcd,
                )
                print(f"  x_T_opt: {x_T_opt_batch.shape}  loss: {loss:.4f}")

            # Phase 2: DDIM inference (batch) + step-wise memo_proxy 수집 → plot
            img_batch, _ = ddim_inference_with_proxy(
                sd, x_T_opt_batch, uc_batch, c_batch, args.cfg,
                args.base_s_ratio, record_dir,
                prompt_tag=(f"img_{i0:04d}" if N == 1 else None))

        torch.cuda.synchronize()
        _t1 = time.perf_counter()
        _batch_time_s = _t1 - _t0
        _batch_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        _per_sample_sec = (_batch_time_s / (N * S))        # sec (단위 통일)
        _total_time_s += _batch_time_s
        _total_peak_gb = max(_total_peak_gb, _batch_peak_gb)

        # save  (init_score_noise 라벨링 표준: img_{prompt:04d}_{sample:02d}.png)
        for n in range(N):
            for j in range(S):
                idx = n * S + j
                fname = f"img_{i0+n:04d}_{j:02d}.png"
                save_image(img_batch[idx], os.path.join(result_dir, fname))
                comp_rows.append(((i0 + n) * S + j, _per_sample_sec, _batch_peak_gb))
                print(f"  sample[{idx}] -> result/{fname}")

    # ---- comp_metrics.csv: per-sample time/VRAM + mean/std ----
    import statistics as _st
    comp_dir = os.path.join(args.output_dir, "comp")
    os.makedirs(comp_dir, exist_ok=True)
    comp_csv = os.path.join(comp_dir, "comp_metrics.csv")
    _times = [r[1] for r in comp_rows]
    _vrams = [r[2] for r in comp_rows]
    _n = len(comp_rows)
    _mean_t = sum(_times) / _n if _n else 0.0
    _mean_v = sum(_vrams) / _n if _n else 0.0
    _std_t = _st.pstdev(_times) if _n > 1 else 0.0
    _std_v = _st.pstdev(_vrams) if _n > 1 else 0.0
    with open(comp_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "time_per_sample_sec", "peak_vram_GB"])
        for r in comp_rows:
            writer.writerow([r[0], f"{r[1]:.2f}", f"{r[2]:.4f}"])
        writer.writerow([])
        writer.writerow(["statistic", "time_per_sample_sec", "peak_vram_GB"])
        writer.writerow(["mean", f"{_mean_t:.2f}", f"{_mean_v:.4f}"])
        writer.writerow(["std",  f"{_std_t:.2f}", f"{_std_v:.4f}"])
        writer.writerow([])
        writer.writerow(["total_time_sec", f"{_total_time_s:.2f}"])
        writer.writerow(["total_peak_vram_GB", f"{_total_peak_gb:.4f}"])
    print(f"\n[BENCH] init_opti: {_n} samples | "
          f"time_per_sample mean={_mean_t:.4f}s std={_std_t:.4f}s | "
          f"peak_vram mean={_mean_v:.3f}GB std={_std_v:.3f}GB")
    print(f"[BENCH] saved → {comp_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
