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


def optimize_xT(sd, uc, c, cfg, device, init_steps, num_steps, gap_steps, lr, base_s_ratio=0.5):
    """Optimize x_T by applying gradient at [init_steps, init_steps+gap_steps, ...]"""

    timesteps = list(sd.scheduler.timesteps)

    # update target step indices
    update_indices = [init_steps + i * gap_steps for i in range(num_steps)]
    update_indices = [i for i in update_indices if i < len(timesteps)]

    # s target for memo_proxy
    s_idx = int(len(timesteps) * base_s_ratio)
    s_target = timesteps[s_idx]
    alpha_s = sd.alpha(s_target)

    x_T = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([x_T], lr=lr)

    total_loss = 0.0
    for ui, t_idx in enumerate(update_indices):
        optimizer.zero_grad()
        epsilon = x_T
        zt = x_T.to(sd.dtype) * sd.scheduler.init_noise_sigma

        # DDIM forward (with grad) up to t_idx
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t)
            at_prev = sd.alpha(t - sd.skip)
            noise_uc, noise_c = sd.predict_noise(zt, t, uc, c)
            eps_theta = noise_uc + cfg * (noise_c - noise_uc)
            x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()
            zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta
            if step_idx == t_idx:
                break

        # memo_proxy loss at this step
        x_s = alpha_s.sqrt().to(sd.dtype) * x0_hat + (1 - alpha_s).sqrt().to(sd.dtype) * epsilon.to(sd.dtype)
        noise_uc_s, noise_c_s = sd.predict_noise(x_s, s_target, uc, c)
        eps_s = noise_uc_s + cfg * (noise_c_s - noise_uc_s)
        loss = (epsilon.to(sd.dtype) - eps_s).reshape(1, -1).pow(2).sum()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # cleanup
        del zt, x0_hat, x_s, eps_s, noise_uc, noise_c, noise_uc_s, noise_c_s, eps_theta
        torch.cuda.empty_cache()

    x_T_opt = x_T.detach().clone()
    del x_T, optimizer
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
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--num_seeds", type=int, default=8)
    p.add_argument("--model_key", type=str, default=os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default=os.path.join(SCRIPT_DIR, "workdir", "ini_opti", "memorized"))
    p.add_argument("--prompts", nargs="+", type=str, default=None,
                   help="Run only specified prompt folder names (e.g. tiger_portrait)")
    args = p.parse_args()
    device = torch.device(args.device)

    solver_config = munchify({"num_sampling": args.NFE})
    sd = StableDiffusion(solver_config=solver_config, model_key=args.model_key, device=device, seed=args.base_seed)
    sd.unet.enable_gradient_checkpointing()

    update_steps = [args.init_steps + i * args.gap_steps for i in range(args.num_steps)]
    print(f"NFE={args.NFE} CFG={args.cfg} lr={args.lr} update_steps={update_steps}")

    prompts_to_run = MEMO_PROMPTS
    if args.prompts:
        prompts_to_run = {k: v for k, v in MEMO_PROMPTS.items() if k in args.prompts}

    for folder, prompt in prompts_to_run.items():
        cfg_dir = os.path.join(args.output_dir, folder,
                               f"NFE={args.NFE}", f"CFG={args.cfg}",
                               f"init_steps={args.init_steps}",
                               f"num_steps={args.num_steps}",
                               f"gap_steps={args.gap_steps}",
                               f"lr={args.lr}")
        os.makedirs(cfg_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, folder, "prompt.txt"), "w") as f:
            f.write(prompt + "\n")

        print(f"\n[{folder}] \"{prompt}\"")

        for si in range(args.num_seeds):
            seed = args.base_seed + si * 100
            set_seed(seed)
            uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)

            # Phase 1: optimize x_T
            x_T_opt, loss = optimize_xT(sd, uc, c, args.cfg, device,
                                         args.init_steps, args.num_steps,
                                         args.gap_steps, args.lr, args.base_s_ratio)

            # Phase 2: DDIM inference
            img = ddim_inference(sd, x_T_opt, uc, c, args.cfg)

            fname = f"NFE={args.NFE}_CFG={args.cfg}_lr={args.lr}_steps={args.num_steps}_gap={args.gap_steps}_init={args.init_steps}_seed={seed}_{folder}.png"
            save_image(img, os.path.join(cfg_dir, fname))
            print(f"  seed={seed} loss={loss:.1f} -> {fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()
