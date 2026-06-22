"""
ini_opti: Optimize x_T via ||ε - ε_s||² at multiple DDIM steps (starting from start_step).
Then DDIM inference with optimized x_T.

Example: init_steps=10, num_steps=4, gap_steps=3
  -> gradient at step 10, 13, 16, 19
"""

import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse, torch
from tqdm import tqdm
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed
from torchvision.utils import save_image

MEMO_PROMPTS = {
    "astronaut_on_the_moon":  "An astronaut on the moon",
    "captain_marvel":         "Captain Marvel Exclusive Ccxp Poster Released Online By Marvel",
    "tiger_portrait":         "Portrait of Tiger in black and white by Lukas Holas",
}


def optimize_xT(sd, uc, c, cfg, device, init_steps, num_steps, gap_steps, lr, base_s_ratio, lambda_align):
    """Optimize x_T by applying gradient at [init_steps, init_steps+gap_steps, ...]"""

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
            eps_theta = noise_uc + cfg * (noise_c - noise_uc)

            x0_hat = (zt_ref - (1 - at).sqrt() * eps_theta) / at.sqrt() # Tweedie formula

            zt_ref = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta # DDIM Denoising Step
            if step_idx in update_indices: 
                x0_orig_refs[step_idx] = x0_hat.detach().clone().float()

    total_loss = 0.0
    for ui, t_idx in enumerate(update_indices):
        # print(f"t_idx: {t_idx}")
        optimizer.zero_grad()

        # ε reference = 현재 trajectory를 seed하는 초기 noise ε.
        # eps_trajectory.py:74 의 epsilon_original(=zt) 구조와 동일 —
        # "trajectory를 만든 noise 자체"를 reference로 쓴다 (fresh gaussian 아님).
        epsilon_ref = x_T.detach()

        zt = x_T.to(sd.dtype) * sd.scheduler.init_noise_sigma

        # DDIM forward (with grad) up to t_idx
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t)
            at_prev = sd.alpha(t - sd.skip)
            noise_uc, noise_c = sd.predict_noise(zt, t, uc, c)
            eps_theta = noise_uc + cfg * (noise_c - noise_uc)
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
        eps_s = noise_uc_s + cfg * (noise_c_s - noise_uc_s)

        # memo proxy = ||ε - eps_s||² / D, batch-safe (eps_trajectory.py:115-116 과 동일)
        # reshape(B,-1).pow(2).mean(-1) → 샘플별 (B,) proxy; .mean() 으로 스칼라 loss
        B = eps_s.shape[0]
        memo_proxy = (epsilon_ref.to(sd.dtype) - eps_s).reshape(B, -1).pow(2).mean(-1)  # (B,)

        # 완화 목적: memo_proxy를 MINIMIZE.
        # 실측(eps_trajectory plot)에서 memorized prompt일수록 proxy가 큼(ε을 무시하고
        # memorized 방향 eps_s를 뱉기 때문). ∴ proxy↓ = 정상 denoiser(eps_s→ε)로 회귀 = 완화.
        loss_memo = memo_proxy.mean()   # 스칼라: batch 평균 (최소화 → 완화)

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
    return x_T_opt, total_loss


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


def load_prompts(prompt_dir, num_samples):
    """prompt 파일에서 앞 num_samples개 prompt를 순서대로 로드."""
    with open(prompt_dir, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    return prompts[:num_samples]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--NFE", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
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

    solver_config = munchify({"num_sampling": args.NFE})
    sd = StableDiffusion(solver_config=solver_config, model_key=args.model_key, device=device, seed=args.base_seed)
    sd.unet.enable_gradient_checkpointing()

    update_steps = [args.init_steps + i * args.gap_steps for i in range(args.num_steps)]
    print(f"NFE={args.NFE} CFG={args.cfg} lr={args.lr} update_steps={update_steps}")
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

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] \"{prompt}\"")

        for j in range(args.num_seeds):
            seed = args.base_seed + j * 100
            set_seed(seed)
            uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)

            # Phase 1: optimize x_T
            # print(f"sd: {type(sd)}")
            # print(f"uc: {uc.shape}")
            # print(f"c: {c.shape}")
            x_T_opt, loss = optimize_xT(sd, uc, c, args.cfg, device,
                                         args.init_steps, args.num_steps,
                                         args.gap_steps, args.lr, args.base_s_ratio,
                                         args.lambda_align)
            
            print(f"x_T_opt: {x_T_opt.shape}") # VAE latent : [1,4,64,64]
            print(f"loss: {loss}")
            # Phase 2: DDIM inference
            img = ddim_inference(sd, x_T_opt, uc, c, args.cfg)

            # sequential flat naming so VendiScore groups (num_seeds consecutive = 1 prompt)
            idx = i * args.num_seeds + j
            fname = f"{idx:05d}.png"
            save_image(img, os.path.join(result_dir, fname))
            print(f"  prompt={i} seed={seed} loss={loss:.1f} -> result/{fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()
