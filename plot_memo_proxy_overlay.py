"""
Overlay memorization-proxy curves across prompt sources.

Reads the per-prompt proxy CSVs produced by eps_trajectory.py (multi-sample mode):

    {group_dir}/plot{NN}/csv/proxy_{idx}.csv
        columns: step,proxy_mean,proxy_std   (proxy_mean = batch mean over seeds)

memo vs text split by --num_tp:
    text (coco general) = idx 0 .. num_tp-1
    memo                = idx num_tp ..                     (tail)

Groups:
    Chen (cvpr2025_memo_prompt)        -> --chen_dir
    Wen  (new_memorized_text_prompt)   -> --wen_dir
    General (coco_v2)                  -> pooled text curves from chen_dir + wen_dir

Outputs:
    {output}              overlay_compare.png  (group mean ± SEM, linear + log axes)
    {output no ext}.npz   raw per-prompt curves per group + group-mean curves

Usage (run from ori_memo/):
    python plot_memo_proxy_overlay.py \
        --chen_dir results_eps_trajectory/sd14/ddim/CFG=7.5_NFE=50/seed=42/batch=5/chen \
        --wen_dir  results_eps_trajectory/sd14/ddim/CFG=7.5_NFE=50/seed=42/batch=5/wen \
        --num_tp 3 \
        --output  results_eps_trajectory/sd14/ddim/CFG=7.5_NFE=50/seed=42/batch=5/overlay_compare.png
"""
import os
import argparse
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===================================================================
#  IO helpers
# ===================================================================
def read_proxy_csv(path):
    """Read eps_trajectory.py proxy CSV -> (steps[N], mean[N], std[N])."""
    steps, mean, std = [], [], []
    with open(path) as f:
        next(f)  # header: step,proxy_mean,proxy_std
        for line in f:
            line = line.strip()
            if not line:
                continue
            s, m, sd = line.split(",")
            steps.append(int(s))
            mean.append(float(m) if m else float("nan"))
            std.append(float(sd) if sd else 0.0)
    return np.array(steps), np.array(mean), np.array(std)


def collect_group(group_dir, num_tp):
    """Walk plot*/csv/proxy_*.csv under group_dir.

    Returns dict:
        steps : (T,)      denoising step indices
        memo  : (Pm, T)   per-prompt proxy_mean curves for memorized prompts (tail idx)
        text  : (Pt, T)   per-prompt proxy_mean curves for coco general prompts (head idx)
    """
    memo_curves, text_curves = [], []
    steps_ref = None
    plot_dirs = sorted(glob.glob(os.path.join(group_dir, "plot*")))
    if not plot_dirs:
        print(f"  [warn] no plot*/ dirs under {group_dir}")
    for pd in plot_dirs:
        csvs = sorted(glob.glob(os.path.join(pd, "csv", "proxy_*.csv")))
        for idx, c in enumerate(csvs):
            steps, mean, _ = read_proxy_csv(c)
            if steps_ref is None:
                steps_ref = steps
            if idx < num_tp:
                text_curves.append(mean)
            else:
                memo_curves.append(mean)
    return {
        "steps": steps_ref if steps_ref is not None else np.array([]),
        "memo": np.asarray(memo_curves) if memo_curves else np.empty((0, 0)),
        "text": np.asarray(text_curves) if text_curves else np.empty((0, 0)),
    }


def _sem(a):
    """Standard error of the mean across rows (prompts). a: (P, T) -> (T,)."""
    if a.size == 0 or a.shape[0] < 1:
        return np.zeros(a.shape[1] if a.ndim == 2 else 0)
    return a.std(axis=0) / max(np.sqrt(a.shape[0]), 1.0)


# ===================================================================
#  Plot
# ===================================================================
def plot_overlay(groups, output_png, output_npz):
    """groups: ordered dict name -> {'steps','curves'(P,T)}. Saves png + npz."""
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(18, 7))
    palette = {
        "Chen (cvpr2025)":      "tab:red",
        "Wen (new_memorized)":  "tab:purple",
        "General (coco)":       "tab:blue",
    }

    for name, g in groups.items():
        curves = g["curves"]
        if curves.size == 0:
            print(f"  [skip] {name}: empty curves")
            continue
        steps = g["steps"]
        mean = curves.mean(axis=0)
        sem = _sem(curves)
        color = palette.get(name, "gray")
        n = curves.shape[0]
        for ax, use_log in [(ax_lin, False), (ax_log, True)]:
            ax.plot(steps, mean, color=color, linewidth=2.3, marker="o", markersize=3,
                    label=f"{name} (n={n})")
            ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.2)
            if use_log:
                ax.set_yscale("log")

    for ax, t in [(ax_lin, "linear"), (ax_log, "log")]:
        ax.set_xlabel("Denoising Step", fontsize=12)
        ax.set_ylabel(r"$||\epsilon - \epsilon_s||^2\ /\ D$", fontsize=12)
        ax.set_title(f"{t} scale", fontsize=13)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        "Memorization proxy by prompt source  "
        "(SD1.4, DDIM, CFG=7.5, NFE=50; mean ± SEM across prompts)",
        fontsize=13,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {output_png}")

    # raw curves dump
    dump = {}
    first_steps = next(iter(groups.values()))["steps"]
    dump["steps"] = first_steps
    for name, g in groups.items():
        key = name.split(" ")[0].lower()
        dump[f"{key}_curves"] = g["curves"]
        dump[f"{key}_mean"] = g["curves"].mean(axis=0) if g["curves"].size else np.array([])
    np.savez(output_npz, **dump)
    print(f"[saved] {output_npz}")


# ===================================================================
#  Main
# ===================================================================
def main():
    p = argparse.ArgumentParser(
        description="Overlay memo_proxy curves: Chen (cvpr2025) vs Wen (new_memorized) vs coco general")
    p.add_argument("--chen_dir", required=True,
                   help="eps_trajectory output dir for cvpr2025 (Chen) memo prompts")
    p.add_argument("--wen_dir", required=True,
                   help="eps_trajectory output dir for new_memorized (Wen) memo prompts")
    p.add_argument("--num_tp", type=int, default=3,
                   help="number of coco general prompts per plot (= head index count; memo = tail)")
    p.add_argument("--output", default="overlay_compare.png",
                   help="output overlay png path")
    args = p.parse_args()

    chen = collect_group(args.chen_dir, args.num_tp)
    wen = collect_group(args.wen_dir, args.num_tp)

    # general (coco) = pool text curves from both runs (same coco source distribution)
    if chen["text"].size and wen["text"].size:
        general_curves = np.concatenate([chen["text"], wen["text"]], axis=0)
    elif chen["text"].size:
        general_curves = chen["text"]
    else:
        general_curves = wen["text"]

    steps = chen["steps"] if chen["steps"].size else wen["steps"]

    groups = {
        "Chen (cvpr2025)":     {"steps": steps, "curves": chen["memo"]},
        "Wen (new_memorized)": {"steps": steps, "curves": wen["memo"]},
        "General (coco)":      {"steps": steps, "curves": general_curves},
    }

    print("=" * 60)
    for name, g in groups.items():
        print(f"  {name:22s}: {g['curves'].shape[0]:>4d} prompts")
    print("=" * 60)

    npz_out = os.path.splitext(args.output)[0] + ".npz"
    plot_overlay(groups, args.output, npz_out)


if __name__ == "__main__":
    main()
