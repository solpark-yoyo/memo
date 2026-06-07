"""
Plot cross-seed variance of ||ε - ε_s||² for memorization comparison.

Memorized prompt  → low variance across seeds (model always predicts same ε_s)
Normal prompt     → high variance across seeds (model's prediction depends on ε)

Reads .npz files from eps_trajectory.py --num_samples N output.
"""

import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt


def load_multi_results(result_dir):
    """Load all eps_traj_multi_*.npz files, sorted by index."""
    files = sorted(glob.glob(os.path.join(result_dir, "eps_traj_multi_*.npz")))
    all_data = []
    for f in files:
        data = dict(np.load(f, allow_pickle=True))
        all_data.append(data)
    return all_data


def plot_cross_seed_variance(all_data, memo_indices, output_dir,
                             filename="cross_seed_variance.png"):
    """
    Plot variance of ||ε - ε_s||² across seeds at each denoising step.

    Left  : mean ± std for each prompt
    Right : std alone (lower = more memorized)
    """
    n = len(all_data)
    fig, (ax_mean, ax_std) = plt.subplots(1, 2, figsize=(16, 6))

    colors = []
    for i in range(n):
        colors.append("red" if i in memo_indices else f"tab:{['blue','green','orange','purple'][i % 4]}")

    for idx, data in enumerate(all_data):
        prompt = str(data["prompt"])
        label = (prompt[:40] + "...") if len(prompt) > 40 else prompt
        if idx in memo_indices:
            label = f"[MEMO] {label}"

        all_curves = data["all_eps_diff_sq"]          # (num_seeds, num_steps)
        steps = data["step_indices"]
        mean = all_curves.mean(axis=0)
        std  = all_curves.std(axis=0)

        is_memo = idx in memo_indices
        lw = 2.5 if is_memo else 1.5
        ls = "--" if is_memo else "-"

        ax_mean.plot(steps, mean, linewidth=lw, linestyle=ls,
                     color=colors[idx], label=label, alpha=0.9)
        ax_mean.fill_between(steps, mean - std, mean + std,
                             color=colors[idx], alpha=0.15)

        ax_std.plot(steps, std, linewidth=lw, linestyle=ls,
                    color=colors[idx], label=label, alpha=0.9)

    # --- Left: mean ± std ---
    ax_mean.set_xlabel("Denoising Step", fontsize=12)
    ax_mean.set_ylabel(r"$\|\, \epsilon - \epsilon_s \,\|^2$  (mean ± std)", fontsize=12)
    ax_mean.set_title("Mean ± Std across seeds", fontsize=13)
    ax_mean.legend(fontsize=7, loc="best", framealpha=0.9)
    ax_mean.grid(True, alpha=0.3)

    # --- Right: std only ---
    ax_std.set_xlabel("Denoising Step", fontsize=12)
    ax_std.set_ylabel(r"Std of $\|\, \epsilon - \epsilon_s \,\|^2$", fontsize=12)
    ax_std.set_title("Cross-seed Std  (lower = more memorized)", fontsize=13)
    ax_std.legend(fontsize=7, loc="best", framealpha=0.9)
    ax_std.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {path}")
    plt.close()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results_eps_multiseed"))
    p.add_argument("--memo_indices", nargs="+", type=int, default=[3],
                   help="0-based indices of memorized prompts")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Defaults to result_dir")
    args = p.parse_args()

    output_dir = args.output_dir or args.result_dir
    all_data = load_multi_results(args.result_dir)
    print(f"Loaded {len(all_data)} prompt results from {args.result_dir}")

    plot_cross_seed_variance(all_data, set(args.memo_indices), output_dir)


if __name__ == "__main__":
    main()
