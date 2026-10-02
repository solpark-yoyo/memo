"""
Tweedie domain gap tracker for HuggingFace DDIM pipelines.

Self-referential form (eps_ref = x_T):
  x0_hat = (x_t - sqrt(1-alpha_t)*eps_cfg) / sqrt(alpha_t)
  x_s    = sqrt(alpha_s)*x0_hat + sqrt(1-alpha_s)*x_T
  eps_s  = UNet(x_s, s_target, text_embeds)
  proxy  = ||x_T - eps_s||^2 / D   shape (B,)

  twd_mean(step) = proxy.mean()
  twd_std(step)  = proxy.std()          (across seeds in batch)

Usage inside a pipeline __call__:
    tracker = TwdGapTracker(self.unet, self.scheduler,
                            prompt_embeds,   # (2*B, seq, dim)
                            guidance_scale, x_T)
    # in denoising loop, BEFORE scheduler.step():
    if tracker is not None:
        tracker.record(latents, noise_pred, t, i)
    # after loop:
    if tracker is not None:
        tracker.save(record_dir, tag="method_name")
"""
import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TwdGapTracker:
    def __init__(self, unet, scheduler, prompt_embeds, cfg, x_T, base_s_ratio=0.5):
        """
        Args:
          unet:          HF UNet2DConditionModel
          scheduler:     DDIMScheduler (set_timesteps already called)
          prompt_embeds: (2*B, seq, dim) — cat([uncond, cond])
          cfg:           guidance_scale
          x_T:           (B, C, H, W) initial latent noise (AFTER any pre-optimization)
          base_s_ratio:  mid-noise ratio for s_target
        """
        self.unet = unet
        self.scheduler = scheduler
        self.prompt_embeds = prompt_embeds.detach()
        self.cfg = float(cfg)
        self.device = x_T.device
        self.dtype = x_T.dtype

        self.eps_ref = x_T.detach()    # (B, C, H, W) — self-referential
        self.B = x_T.shape[0]

        timesteps = scheduler.timesteps
        s_idx = int(len(timesteps) * base_s_ratio)
        self.s_target = int(timesteps[s_idx].item())
        self.alphas_cumprod = scheduler.alphas_cumprod.to(self.device)
        self.alpha_s = self.alphas_cumprod[self.s_target].to(self.dtype)

        self.records = []  # list of (step_idx, proxy_np (B,))

    @torch.no_grad()
    def record(self, latents, noise_pred_cfg, t, step_idx):
        """Call BEFORE scheduler.step(). latents = x_t, noise_pred_cfg = eps after CFG."""
        t_val = int(t.item()) if hasattr(t, "item") else int(t)
        alpha_t = self.alphas_cumprod[t_val].to(self.dtype)

        x_t = latents.detach().to(self.dtype)
        eps_cfg = noise_pred_cfg.detach().to(self.dtype)

        # Tweedie
        x0_hat = (x_t - (1 - alpha_t).sqrt() * eps_cfg) / alpha_t.sqrt()

        # Forward to s_target (self-referential: x_T as noise)
        x_s = self.alpha_s.sqrt() * x0_hat + (1 - self.alpha_s).sqrt() * self.eps_ref

        # UNet at s_target
        x_s_in = torch.cat([x_s] * 2)
        t_s = torch.full((2 * self.B,), self.s_target, device=self.device, dtype=torch.long)
        out = self.unet(x_s_in, t_s,
                        encoder_hidden_states=self.prompt_embeds.to(self.device),
                        return_dict=False)[0]
        eps_uc, eps_c = out.chunk(2)
        eps_s = eps_uc + self.cfg * (eps_c - eps_uc)

        proxy = (self.eps_ref - eps_s).reshape(self.B, -1).pow(2).mean(-1).float().cpu()
        self.records.append((step_idx, proxy.numpy()))

    def save(self, record_dir, tag=""):
        if not self.records:
            return
        os.makedirs(record_dir, exist_ok=True)
        steps = np.array([r[0] for r in self.records])
        proxy_arr = np.stack([r[1] for r in self.records], axis=0)  # (T, B)
        twd_mean = proxy_arr.mean(axis=1)
        twd_std = proxy_arr.std(axis=1)
        sfx = f"_{tag}" if tag else ""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(steps, twd_mean, color="tab:blue", linewidth=2.2, marker="o", markersize=4)
        ax1.set_ylabel(r"twd_mean  $E[\|\mathbf{x}_T - \epsilon_s\|^2/D]$", fontsize=11)
        title = f"twd_mean / twd_std — {tag}" if tag else "twd_mean / twd_std"
        title += f"\n(B={self.B}, s_target={self.s_target})"
        ax1.set_title(title, fontsize=11)
        ax1.set_ylim(0, 6)
        ax1.grid(True, alpha=0.3)
        ax2.plot(steps, twd_std, color="tab:orange", linewidth=2.2, marker="s", markersize=4)
        ax2.set_xlabel("Denoising Step", fontsize=12)
        ax2.set_ylabel(r"twd_std  $\mathrm{Std}[\|\mathbf{x}_T - \epsilon_s\|^2/D]$", fontsize=11)
        ax2.set_ylim(0, 6)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(record_dir, f"twd_gap_inference{sfx}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[twd_gap] plot -> {plot_path}")

        csv_path = os.path.join(record_dir, f"twd_gap_inference{sfx}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "twd_mean", "twd_std"])
            for s, m, sd in zip(steps, twd_mean, twd_std):
                w.writerow([int(s), f"{m:.6f}", f"{sd:.6f}"])
        print(f"[twd_gap] csv  -> {csv_path}")
