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
    return 0.0 if step_idx < int(total * ratio) else cfg

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
                type_memo_loss="minimization", memo_threshold=0.0):
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

    x_T_init = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
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
    # ε reference = 최적화 전 원본 noise로 고정 (eps_trajectory.py:74 구조와 동일)
    # 루프 밖에서 한 번만 정의 → 4번의 update 동안 변하지 않음
    epsilon_ref = x_T_init.detach()

    for ui, t_idx in enumerate(update_indices):
        # print(f"t_idx: {t_idx}")
        optimizer.zero_grad()

        zt = x_T.to(sd.dtype) * sd.scheduler.init_noise_sigma

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
        x_s = alpha_s.sqrt().to(sd.dtype) * x0_hat + (1 - alpha_s).sqrt().to(sd.dtype) * x_T.detach().to(sd.dtype)
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
                     record_dir=None,
                     type_memo_loss="minimization", memo_threshold=0.0):
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
    # ε reference = 최적화 전 원본 noise로 고정 (4번 update 동안 변하지 않음)
    epsilon_ref = x_T_init.detach()

    # ---- record setup: batch(=seed)별 폴더 + update-step 누적 리스트 ----
    # 구조: record/img_PPPP/img_PPPP_BB/{grid.png, loss.csv}
    # grid.png = row(update step 횟수) × col(x_t, x_0|t, x_s)
    batch_record_imgs = None
    batch_dirs = None
    if record_dir is not None:
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
            for step_idx, t in enumerate(timesteps):
                at = sd.alpha(t)
                at_prev = sd.alpha(t - sd.skip)
                noise_uc, noise_c = sd.predict_noise(x_k, t, uc, c)
                eps_theta = cfg_combine(noise_uc, noise_c, step_idx)
                x0_hat = (x_k - (1 - at).sqrt() * eps_theta) / at.sqrt()
                x_k = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta
                if step_idx == t_idx:
                    # Match the SNR debug print of the original (lines 91-93)
                    _snr_t = (at / (1 - at)).item()
                    print(f"    [tweedie t_idx={t_idx}] alpha_t={at.item():.4f}  "
                          f"SNR={_snr_t:.3f}  (1/sqrt(alpha)={(1/at.sqrt()).item():.2f}x amplification)")
                    break
                # cache the NEXT state for steps we will visit in the adjoint
                if cache_latents and (step_idx + 1) <= t_idx:
                    xs.append(x_k.clone())
        # x_k is now x_{t_idx+1} (after DDIM step at step_idx==t_idx).
        # But the terminal head needs x_{t_idx} (BEFORE the DDIM step).
        # xs[t_idx] was cached at step_idx==t_idx-1 (appended after DDIM step).
        if cache_latents and len(xs) > t_idx:
            x_end_state = xs[t_idx].clone()   # x_{t_idx} (correct)
        else:
            # Fallback: recompute x_{t_idx} from x_T (no cache path)
            x_end_state = _recompute_to(sd, x_T, timesteps, t_idx, uc, c,
                                        cfg, sd.scheduler.init_noise_sigma).clone()

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
                   + (1 - alpha_s).sqrt().to(sd.dtype) * x_T.detach().to(sd.dtype))

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
            if record_dir is not None and batch_dirs is not None:
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
                x_T.grad = (g * sd.scheduler.init_noise_sigma).detach().reshape_as(x_T)

        # Free the head graph immediately (only 2 UNets, but no need to keep).
        del x_end, eps_t, x0_hat, x_s, eps_s, noise_uc_s, noise_c_s

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
    if record_dir is not None and batch_record_imgs is not None:
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
      - ε_ref = 최적화된 x_T (DDIM 입력 noise)
      - x̂₀|t  = Tweedie(x_t, t)
      - x_s    = √ᾱ_s · x̂₀|t + √(1-ᾱ_s) · ε_ref,  s = 고정 mid-noise (base_s_ratio)
      - ε_s    = ε_θ(x_s, s)
    denoising 이 진행될수록 x̂₀|t → x₀ 이므로 x_s → true forward, ε_s → ε_ref.
    ∴ memo_proxy 는 단조 감소 경향(= memorization 완화)을 보여야 함.

    batch(= num_seeds) 차원으로 mean ± std 를 계산해 plot + csv → record_dir/.
    """
    os.makedirs(record_dir, exist_ok=True)
    timesteps = list(sd.scheduler.timesteps)
    s_idx = int(len(timesteps) * base_s_ratio)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)

    epsilon_ref = x_T.detach().to(sd.dtype)        # 최적화된 x_T = DDIM 입력 noise
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

    img = sd.decode(x0_hat)
    return (img / 2 + 0.5).clamp(0, 1), plot_path


def load_prompts(prompt_dir, num_samples):
    """prompt 파일에서 앞 num_samples개 prompt를 순서대로 로드."""
    with open(prompt_dir, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    return prompts[:num_samples]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--NFE", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--cfg_start_ratio", type=float, default=0.0,
                   help="denoising step 중 CFG 적용 시작 ratio. "
                        "step_idx < ratio*NFE 동안은 null(unconditional)만 사용하고, "
                        "그 이후 step부터 정상 CFG. 0.0 = 항상 CFG (기존 동작). 예: 0.3 = 초반 30%% null")
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
                   choices=["minimization", "threshold"],
                   help="minimization: loss=mean(proxy) 전체 완화 | "
                        "threshold: loss=mean(relu(proxy-τ)), proxy>τ 일 때만 grad 흘림")
    p.add_argument("--memo_threshold", type=float, default=0.3,
                   help="τ for type_memo_loss=threshold (memo_proxy=||ε-ε_s||²/D 스케일; "
                        "memorized proxy 전형적으로 ~0.1-0.5, 모델/prompt 마다 튜닝)")
    p.add_argument("--base_seed", type=int, default=42)
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

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] \"{prompt}\" (batch={args.num_seeds})")

        # text embedding (1회 계산, batch로 복제)
        uc, c = sd.get_text_embed(null_prompt="", prompt=prompt) # sen : [1,77,768]
        uc_batch = uc.repeat(args.num_seeds, 1, 1) # [num_images_per_prompt,77,768]
        c_batch = c.repeat(args.num_seeds, 1, 1) # num_images_per_prompt == num_seeds

        # 초기 noise — DDIM(text_to_mscoco)과 동일: main set_seed 1회 후
        # prompt마다 torch.randn([num_seeds,4,64,64]) batch 연속 생성 (reset X)
        x_T_init_batch = torch.randn(args.num_seeds, 4, 64, 64, device=device, dtype=torch.float32)
        print(f"  x_T batch: {x_T_init_batch.shape}")

        # ---- per-batch compute cost 측정 시작 (optimize + inference) ----
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        _t0 = time.perf_counter()

        # Phase 1: optimize x_T (batch)
        record_dir = os.path.join(args.output_dir, "record", f"img_{i:04d}")
        os.makedirs(record_dir, exist_ok=True)
        x_T_opt_batch, loss = optimize_xT_adj(
            sd, uc_batch, c_batch, args.cfg, device,
            args.init_steps, args.num_steps, args.gap_steps, args.lr,
            args.base_s_ratio, args.lambda_align,
            batch_size=args.num_seeds,
            record_dir=record_dir,
            type_memo_loss=args.type_memo_loss,
            memo_threshold=args.memo_threshold,
        )
        print(f"  x_T_opt: {x_T_opt_batch.shape}  loss: {loss:.4f}")

        # Phase 2: DDIM inference (batch) + step-wise memo_proxy 수집 → plot
        # x_T opti 종료 후 DDIM sampling 하며 각 denoising step 의 memo_proxy 를
        # batch(num_seeds) mean ± std 로 시각화 → record/img_{i:04d}/
        img_batch, _ = ddim_inference_with_proxy(
            sd, x_T_opt_batch, uc_batch, c_batch, args.cfg,
            args.base_s_ratio, record_dir, prompt_tag=f"img_{i:04d}")

        torch.cuda.synchronize()
        _t1 = time.perf_counter()
        _batch_time_s = _t1 - _t0
        _batch_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        _per_sample_sec = (_batch_time_s / args.num_seeds)   # sec (단위 통일)
        _total_time_s += _batch_time_s
        _total_peak_gb = max(_total_peak_gb, _batch_peak_gb)

        # save  (init_score_noise 라벨링 표준: img_{prompt:04d}_{sample:02d}.png)
        for j in range(args.num_seeds):
            fname = f"img_{i:04d}_{j:02d}.png"
            save_image(img_batch[j], os.path.join(result_dir, fname))
            comp_rows.append((i * args.num_seeds + j, _per_sample_sec, _batch_peak_gb))
            print(f"  sample[{j}] -> result/{fname}  (batch noise from single seed={args.base_seed})")

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
