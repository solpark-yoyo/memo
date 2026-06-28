"""
Diagnostic (perspective 7: dxT-lr-diagnostic).
Calls optimize_xT (chained-autograd baseline) and optimize_xT_adj (adjointDPM)
DIRECTLY (not via main()) across lr in {0.0, 0.01, 0.03, 0.05} on ONE prompt.

Captures per-update-step:
  (a) |x_T_final - x_T_init|  (dxT)        -> must grow monotonically with lr
  (b) |x_T.grad|              (g_xT norm)  -> constant vs lr? magnitude?
  (c) ratio-to-terminal       (adjoint)    -> does the recursion preserve signal?

We monkeypatch the print() used inside the optimizers via stdout capture so we
can also read the [adj] line traces. But the primary readouts come from re-
deriving the numbers inside a thin re-implementation of the two optimizers that
returns the raw tensors. To stay faithful, we instead just import the real
optimizers and additionally instrument by wrapping torch.autograd.grad in the
adjoint path is NOT needed: the optimizers already print |g_xT| and |dxT|.

Strategy:
  - For each (optimizer_name, lr): call optimizer, parse the printed
    "|g_xT|=" and "|dxT|=" lines, compute final dxT ourselves.
  - Print a summary table at the end.

Usage:
  python diag_dxT_lr.py [--model_key ...] [--device cuda:0]
"""

import sys, os, re, io, contextlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse
import torch
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed

import run_ini_opti as RIO  # the module under test


def run_one(optimizer_fn, sd, uc, c, cfg, device, lr,
            init_steps=10, num_steps=4, gap_steps=3,
            base_s_ratio=0.5, lambda_align=0.1, batch_size=1,
            pass_batch_size=True):
    """Call a single optimizer run, capture stdout, return parsed metrics + final dxT."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        kwargs = dict(
            init_steps=init_steps, num_steps=num_steps, gap_steps=gap_steps, lr=lr,
            base_s_ratio=base_s_ratio, lambda_align=lambda_align,
        )
        if pass_batch_size:
            kwargs["batch_size"] = batch_size
        x_T_opt, total_loss = optimizer_fn(sd, uc, c, cfg, device, **kwargs)
    log = buf.getvalue()

    # parse per-update-step lines: "...|g_xT|=NUM ...|dxT|=NUM"
    g_xt_vals = [float(x) for x in re.findall(r"\|g_xT\|=([\d.eE+\-]+)", log)]
    dxT_vals = [float(x) for x in re.findall(r"\|dxT\|=([\d.eE+\-]+)", log)]
    # parse terminal + ratio-to-terminal from adjoint
    g_terminal = [float(x) for x in re.findall(r"\|g\|=([\d.eE+\-]+)", log)]
    ratio_terminal = [float(x) for x in re.findall(r"ratio\(to terminal\)=([\d.eE+\-]+)", log)]
    memo_vals = [float(x) for x in re.findall(r"loss_memo: ([\d.eE+\-]+)", log)]
    return {
        "log": log,
        "g_xT": g_xt_vals,          # list over update steps
        "dxT": dxT_vals,            # list over update steps
        "ratio_terminal": ratio_terminal,
        "g_step": g_terminal,
        "memo": memo_vals,
        "final_loss": total_loss,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--NFE", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--lrs", type=str, default="0.0,0.01,0.03,0.05")
    p.add_argument("--init_steps", type=int, default=10)
    p.add_argument("--gap_steps", type=int, default=3)
    p.add_argument("--num_steps", type=int, default=4)
    p.add_argument("--base_s_ratio", type=float, default=0.5)
    p.add_argument("--lambda_align", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", type=str,
                   default="An astronaut on the moon")
    p.add_argument("--model_key", type=str,
                   default=os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--optimizers", type=str, default="adj,baseline",
                   help="comma list: adj,baseline")
    p.add_argument("--batch_size", type=int, default=1)
    args = p.parse_args()
    device = torch.device(args.device)

    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    opt_names = [s.strip() for s in args.optimizers.split(",") if s.strip()]

    solver_config = munchify({"num_sampling": args.NFE})
    sd = StableDiffusion(solver_config=solver_config,
                         model_key=args.model_key, device=device, seed=args.seed)
    sd.unet.enable_gradient_checkpointing()

    # fixed text embeddings (re-used for every run; optimizer casts .float() itself)
    uc, c = sd.get_text_embed(null_prompt="", prompt=args.prompt)
    uc_b = uc.repeat(args.batch_size, 1, 1)
    c_b = c.repeat(args.batch_size, 1, 1)

    print("=" * 78)
    print(f"DIAG dxT-lr | prompt='{args.prompt}' | NFE={args.NFE} CFG={args.cfg}")
    print(f"init_steps={args.init_steps} gap={args.gap_steps} num={args.num_steps} "
          f"base_s_ratio={args.base_s_ratio} lambda_align={args.lambda_align} batch={args.batch_size}")
    print("=" * 78)

    results = {}
    for oname in opt_names:
        if oname == "adj":
            fn = RIO.optimize_xT_adj
            pass_bs = True
        elif oname == "baseline":
            fn = RIO.optimize_xT
            pass_bs = False
        else:
            continue
        results[oname] = {}
        for lr in lrs:
            # reset seed so x_T_init is identical across optimizers AND across lr
            # (so the only variable is lr).
            set_seed(args.seed)
            # ensure model is fp16 before each run (optimizers flip to fp32 then back)
            sd.unet.half(); sd.dtype = torch.float16
            try:
                res = run_one(fn, sd, uc_b, c_b, args.cfg, device, lr,
                              init_steps=args.init_steps, num_steps=args.num_steps,
                              gap_steps=args.gap_steps, base_s_ratio=args.base_s_ratio,
                              lambda_align=args.lambda_align, batch_size=args.batch_size,
                              pass_batch_size=pass_bs)
            except Exception as e:
                import traceback
                print(f"[{oname}] lr={lr} FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                results[oname][lr] = {"error": str(e)}
                torch.cuda.empty_cache()
                continue
            results[oname][lr] = res
            # aggressively free between runs to avoid OOM accumulation
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # ---- summary table ----
    print("\n" + "=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)
    for oname in opt_names:
        if oname not in results:
            continue
        print(f"\n### optimizer = {oname}")
        print(f"{'lr':>6} | {'|g_xT| (final upd)':>20} | {'|dxT| (final upd)':>20} | "
              f"{'memo (final upd)':>18} | {'ratio_term range':>22}")
        for lr in lrs:
            r = results[oname].get(lr, None)
            if r is None or "error" in r:
                print(f"{lr:>6.3f} | ERROR: {r.get('error') if r else 'n/a'}")
                continue
            g = r["g_xT"][-1] if r["g_xT"] else float("nan")
            dx = r["dxT"][-1] if r["dxT"] else float("nan")
            mm = r["memo"][-1] if r["memo"] else float("nan")
            rt = r["ratio_terminal"]
            rt_str = (f"{min(rt):.2e}..{max(rt):.2e} (n={len(rt)})") if rt else "n/a (baseline)"
            print(f"{lr:>6.3f} | {g:>20.6e} | {dx:>20.6e} | {mm:>18.6f} | {rt_str:>22}")

    # ---- dxT-vs-lr monotonicity verdict ----
    print("\n" + "-" * 78)
    print("dxT-vs-lr MONOTONICITY (final |dxT| per lr):")
    for oname in opt_names:
        if oname not in results:
            continue
        seq = []
        for lr in lrs:
            r = results[oname].get(lr, None)
            if r is None or "error" in r:
                seq.append((lr, None))
            else:
                seq.append((lr, r["dxT"][-1] if r["dxT"] else float("nan")))
        vals = [v for _, v in seq if v is not None]
        monotone = all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1)) if len(vals) > 1 else True
        print(f"  {oname:10s}: " + ", ".join(f"lr={lr}->{v:.4e}" if v is not None else f"lr={lr}->ERR" for lr, v in seq)
              + f"   | monotone_increasing={monotone}")

    # dump full raw logs to files for the record
    logdir = os.path.join(SCRIPT_DIR, "diag_dxT_lr_out")
    os.makedirs(logdir, exist_ok=True)
    for oname in opt_names:
        for lr in lrs:
            r = results.get(oname, {}).get(lr, None)
            if r is None or "error" in r:
                continue
            with open(os.path.join(logdir, f"{oname}_lr{lr}.log"), "w") as f:
                f.write(r["log"])
    print(f"\nFull per-run logs written to {logdir}/")


if __name__ == "__main__":
    main()
