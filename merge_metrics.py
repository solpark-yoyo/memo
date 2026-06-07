import argparse
import csv
import os


def read_prdc(path):
    """Read prdc_metrics.csv -> [(metric, value, '')]"""
    rows = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Precision, Recall, Density, Coverage
        values = next(reader)
        for h, v in zip(header, values):
            rows.append([h, v, ""])
    return rows


def read_t2i(path):
    """Read t2i_metrics.csv -> [(metric, mean, std)]"""
    rows = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 3 and row[0] in ("CLIPScore", "PickScore", "ImageReward"):
                rows.append([row[0], row[1], row[2]])
    return rows


def read_jepa(path):
    """Read jepa_metrics.csv -> [(JEPA_Score, mean, std)]"""
    mean_val, std_val = "", ""
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0].strip() == "mean":
                mean_val = row[1].strip()
            if len(row) >= 2 and row[0].strip() == "std":
                std_val = row[1].strip()
    return [["JEPA_Score", mean_val, std_val]]


def count_images(path):
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path) if f.endswith(exts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir",  type=str, required=True)
    parser.add_argument("--real_dir", type=str, default="")
    parser.add_argument("--fake_dir", type=str, default="")
    parser.add_argument("--num_real", type=int, default=0, help="Number of real images actually used")
    parser.add_argument("--num_fake", type=int, default=0, help="Number of fake images actually used")
    args = parser.parse_args()

    prdc_path = os.path.join(args.workdir, "prdc_metrics.csv")
    t2i_path  = os.path.join(args.workdir, "t2i_metrics.csv")
    jepa_path = os.path.join(args.workdir, "jepa_metrics.csv")
    out_path  = os.path.join(args.workdir, "result.csv")

    all_rows = []

    # 순서: T2I → PRDC → JEPA
    if os.path.exists(t2i_path):
        all_rows += read_t2i(t2i_path)
    else:
        print(f"[WARN] {t2i_path} not found, skipping T2I metrics")

    if os.path.exists(prdc_path):
        all_rows += read_prdc(prdc_path)
    else:
        print(f"[WARN] {prdc_path} not found, skipping PRDC")

    if os.path.exists(jepa_path):
        all_rows += read_jepa(jepa_path)
    else:
        print(f"[WARN] {jepa_path} not found, skipping JEPA Score")

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        writer.writerows(all_rows)
        writer.writerow([])
        if args.real_dir:
            total_real = count_images(args.real_dir)
            used_real  = args.num_real if args.num_real > 0 else total_real
            writer.writerow(["real_data", args.real_dir, used_real, total_real])
        if args.fake_dir:
            total_fake = count_images(args.fake_dir)
            used_fake  = args.num_fake if args.num_fake > 0 else total_fake
            writer.writerow(["fake_data", args.fake_dir, used_fake, total_fake])

    print(f"[Done] Saved result.csv to {out_path}")
    print("=" * 50)
    for row in all_rows:
        print(f"  {row[0]:<15}: {row[1]}  ±  {row[2]}")
    print("=" * 50)


if __name__ == "__main__":
    main()
