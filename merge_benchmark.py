"""
merge_benchmark.py — Compare DDIM / CNO(infoNCE) / init_opti

Reads per-method metric CSVs and produces a unified comparison table.

Usage:
    python merge_benchmark.py \
        --workdir benchmark_results \
        --methods DDIM CNO_infoNCE init_opti \
        --ddim_dir benchmark_results/ddim/NFE=50_CFG=7.5 \
        --cno_dir benchmark_results/cno_infoNCE/NFE=50_CFG=7.5_temp=0.1_win=1_gamma=1.0_iter=3 \
        --init_opti_dir benchmark_results/init_opti/NFE=50_CFG=7.5_init=10_nsteps=4_gap=3_lr=0.01 \
        --output benchmark_results/comparison_report.csv
"""
import argparse
import csv
import os
from collections import OrderedDict


# ──────────────────── CSV readers ────────────────────

def read_t2i(path):
    """Read *_t2i.csv → {metric: (mean, std)}"""
    metrics = {}
    if not os.path.isfile(path):
        return metrics
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3 and row[0] in ("CLIPScore", "PickScore", "ImageReward"):
                metrics[row[0]] = (row[1], row[2])
    return metrics


def read_prdc(path):
    """Read prdc_metrics.csv → {metric: (value, '')}
    Skips the leading '# Config' metadata block and finds the
    Precision/Recall/Density/Coverage header row."""
    metrics = {}
    if not os.path.isfile(path):
        return metrics
    prdc_cols = {"Precision", "Recall", "Density", "Coverage"}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if row and row[0] in prdc_cols:
                header = row
                break
        if header is None:
            return metrics
        values = next(reader, None)
        if values is None:
            return metrics
        for h, v in zip(header, values):
            metrics[h] = (v, "")
    return metrics


def read_vendi(path):
    """Read vendi_metrics.csv → {metric: (mean, std)}.

    MeanPairwiseSimilarity is renamed to 'MSS (<f_type> checkpoint)',
    where <f_type> is parsed from the # Config block (e.g. sscd → 'SSCD checkpoint').
    """
    metrics = {}
    if not os.path.isfile(path):
        return metrics
    f_type = None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # parse f_type from the # Config metadata block
            if len(row) >= 2 and row[0].strip() == "f_type":
                f_type = row[1].strip()
            if len(row) >= 3 and row[0] in ("VendiScore", "MeanPairwiseSimilarity"):
                if row[0] == "MeanPairwiseSimilarity":
                    tag = {"sscd": "SSCD", "inception": "inception"}.get(f_type, f_type or "feat")
                    name = f"MSS ({tag} checkpoint)"
                else:
                    name = row[0]
                metrics[name] = (row[1], row[2])
    return metrics


def collect_metrics(metric_dir):
    """Collect all metrics from a metric directory."""
    result = OrderedDict()
    result.update(read_t2i(os.path.join(metric_dir, "t2i_metrics.csv")))
    result.update(read_prdc(os.path.join(metric_dir, "prdc_metrics.csv")))
    result.update(read_vendi(os.path.join(metric_dir, "vendi_metrics.csv")))
    return result


def write_single_method_csv(metric_dir, output_csv=None):
    """Collect t2i + prdc + vendi of a single method dir into one total_metrics.csv.

    Writes [metric, mean, std] rows. Missing CSVs are simply skipped.
    Default output: <metric_dir>/total_metrics.csv
    """
    metrics = collect_metrics(metric_dir)
    if not metrics:
        print(f"[WARN] no metrics found at {metric_dir}")
        return
    out_path = output_csv or os.path.join(metric_dir, "total_metrics.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        for m, (mean, std) in metrics.items():
            writer.writerow([m, mean, std if std else ""])
    print(f"[Saved] {out_path}  ({len(metrics)} metrics)")


# ──────────────────── Report ────────────────────

def build_report(methods_data):
    """Build comparison table.

    methods_data: OrderedDict {method_name: {metric: (mean, std)}}
    Returns: list of rows for CSV
    """
    # Collect all metric names in order
    all_metrics = []
    seen = set()
    for metrics in methods_data.values():
        for m in metrics:
            if m not in seen:
                all_metrics.append(m)
                seen.add(m)

    # Header
    header = ["Metric"]
    for method in methods_data:
        header.append(f"{method}_mean")
        header.append(f"{method}_std")

    rows = [header]
    for metric in all_metrics:
        row = [metric]
        for method, metrics in methods_data.items():
            val = metrics.get(metric, ("—", "—"))
            row.append(val[0])
            row.append(val[1])
        rows.append(row)

    return rows


def build_report_method_rows(methods_data):
    """Build comparison table with methods as rows.

    Transpose of build_report: one row per method, one column per metric (mean/std).
    Easier to compare methods at a glance.
    """
    all_metrics = []
    seen = set()
    for metrics in methods_data.values():
        for m in metrics:
            if m not in seen:
                all_metrics.append(m)
                seen.add(m)

    header = ["method"]
    for m in all_metrics:
        header.append(f"{m}_mean")
        header.append(f"{m}_std")

    rows = [header]
    for method, metrics in methods_data.items():
        row = [method]
        for m in all_metrics:
            val = metrics.get(m, ("", ""))
            row.append(val[0])
            row.append(val[1] if val[1] else "")
        rows.append(row)

    return rows


def print_table(methods_data):
    """Pretty-print comparison table to terminal."""
    all_metrics = []
    seen = set()
    for metrics in methods_data.values():
        for m in metrics:
            if m not in seen:
                all_metrics.append(m)
                seen.add(m)

    method_names = list(methods_data.keys())
    col_w = max(22, *(len(n) for n in method_names))
    metric_w = 25

    # Header
    hdr = f"{'Metric':<{metric_w}}"
    for name in method_names:
        hdr += f" {name:>{col_w}}"
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for metric in all_metrics:
        line = f"{metric:<{metric_w}}"
        for method, metrics in methods_data.items():
            val = metrics.get(metric, None)
            if val and val[0] not in ("—", ""):
                mean_s = val[0]
                std_s = val[1] if val[1] else ""
                if std_s:
                    line += f" {mean_s:>8s}±{std_s:<{col_w - 10}s}"
                else:
                    line += f" {mean_s:>{col_w}s}"
            else:
                line += f" {'—':>{col_w}s}"
        print(line)
    print("=" * len(hdr))


# ──────────────────── Main ────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark comparison report")
    parser.add_argument("--workdir", type=str, default="",
                        help="Benchmark root workdir (not needed with --collect_dir)")
    parser.add_argument("--methods", nargs="+", default=["DDIM", "CNO_infoNCE", "init_opti"])
    parser.add_argument("--ddim_dir", type=str, default="")
    parser.add_argument("--cno_dir", type=str, default="")
    parser.add_argument("--init_opti_dir", type=str, default="")
    parser.add_argument("--initnoise_dir", type=str, default="")
    parser.add_argument("--output", type=str, default="",
                        help="Output CSV path (default: workdir/comparison_report.csv)")
    # single-method collect mode: gather t2i+prdc+vendi of one dir into total_metrics.csv
    parser.add_argument("--collect_dir", type=str, default="",
                        help="Single method dir: collect its t2i/prdc/vendi into total_metrics.csv")
    parser.add_argument("--collect_output", type=str, default="",
                        help="Output path for collect mode (default: <collect_dir>/total_metrics.csv)")
    args = parser.parse_args()

    # ── single-method collect mode ──
    if args.collect_dir:
        write_single_method_csv(args.collect_dir, args.collect_output)
        return

    output_path = args.output or os.path.join(args.workdir, "comparison_report.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Map method name → metric_dir
    method_dirs = OrderedDict()
    for name, d in [
        ("DDIM", args.ddim_dir),
        ("CNO_infoNCE", args.cno_dir),
        ("init_opti", args.init_opti_dir),
        ("init_score_noise", args.initnoise_dir),
    ]:
        if d and os.path.isdir(d):
            method_dirs[name] = d   # eval이 {method_dir}/*.csv 에 직접 저장
        elif d:
            print(f"[WARN] {name} dir not found: {d}")

    # Collect
    methods_data = OrderedDict()
    for method, metric_dir in method_dirs.items():
        metrics = collect_metrics(metric_dir)
        if metrics:
            methods_data[method] = metrics
            print(f"[OK] {method}: {len(metrics)} metrics from {metric_dir}")
        else:
            print(f"[WARN] {method}: no metrics found at {metric_dir}")

    if not methods_data:
        print("[ERROR] No metrics collected. Check paths.")
        return

    # Build & write CSV
    rows = build_report(methods_data)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\n[Saved] {output_path}")

    # total_metrics.csv: methods as rows (one row per method) for at-a-glance comparison
    total_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "total_metrics.csv")
    rows_m = build_report_method_rows(methods_data)
    with open(total_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows_m)
    print(f"[Saved] {total_path}")

    # Terminal table
    print("\n")
    print_table(methods_data)


if __name__ == "__main__":
    main()
