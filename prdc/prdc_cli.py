import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import tqdm
from .prdc import compute_prdc
from .Dataset import CustomDataset
from .Models import RandomX, TrainedX
from .utils import customtransforms

import os
import csv

# def main():
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog = 'PRDC CLI',
                        description = 'This is a CLI for PRDC which is ready to use and it has support for vgg16 (R64) case of original paper, hoever if you wish you can choose more or less than 64 output features from vgg16',
                        epilog = '')

    parser.add_argument('-r', '--real_dir', type = str, help = 'Path to real images directory', required = True)
    parser.add_argument('-f', '--fake_dir', type = str, help = 'Path to fake images directory', required = True)
    parser.add_argument('-o', '--out_feats', type = int, help = 'Number of output features from vgg16', default = 64)
    parser.add_argument('-t', '--type', type = str, help = 'Use pretrained (T) or random (R) vgg16', default = 'R')
    parser.add_argument('-b', '--batch_size', type = int, help = 'Batch size for dataloader', default = 64)
    parser.add_argument('-n', '--num_workers', type = int, help = 'Number of workers for dataloader', default = 4)
    parser.add_argument('-d', '--device', type = str, help = 'Device to use for extracting embeddings from vgg16', default = 'cpu')
    parser.add_argument('-k', '--nearest_k', type = int, help = 'Number of nearst neighbors', default = 5)
    parser.add_argument('--num_real', type = int, help = 'Number of real images to use (default: all)', default = None)
    parser.add_argument('--num_fake', type = int, help = 'Number of fake images to use (default: all)', default = None)
    parser.add_argument('--log_csv', type = str, help = 'Path to save detailed log CSV', default = None)
    parser.add_argument('--warn_log', type = str, help = 'Redirect Python warnings to this file', default = None)

    args = parser.parse_args()

    if args.warn_log:
        import logging
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.warn_log)), exist_ok=True)
        logging.captureWarnings(True)
        _wh = logging.FileHandler(args.warn_log, mode='a')
        logging.getLogger('py.warnings').addHandler(_wh)

    real_dir = args.real_dir
    fake_dir = args.fake_dir
    out_feats = args.out_feats
    batch_size = args.batch_size
    num_workers = args.num_workers
    k = args.nearest_k
    m_type = args.type
    device = torch.device(args.device)

    print(f"real_dir: {real_dir}")
    real_dataset = CustomDataset(real_dir, transforms=customtransforms, num_samples=args.num_real)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    fake_dataset = CustomDataset(fake_dir, transforms=customtransforms, num_samples=args.num_fake)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    assert m_type in ['R', 'T'], 'Model type must be R or T'
    if m_type == 'R':
        model = RandomX(out_feats=out_feats).to(device)
        model.eval()
    elif m_type == 'T':
        model = TrainedX(out_feats=out_feats).to(device)
        model.eval()

    real_embeddings = []
    fake_embeddings = []

    print("[INFO] Extracting real embeddings...")
    for real_batch in tqdm.tqdm(real_loader):
        real_batch = real_batch.to(device)
        real_embeddings_batch = model(real_batch).cpu().numpy()
        real_embeddings.append(real_embeddings_batch)

    print("[INFO] Extracting fake embeddings...")
    for fake_batch in tqdm.tqdm(fake_loader):
        fake_batch = fake_batch.to(device)
        fake_embeddings_batch = model(fake_batch).cpu().numpy()
        fake_embeddings.append(fake_embeddings_batch)

    real_embeddings = np.concatenate(real_embeddings, axis=0)
    fake_embeddings = np.concatenate(fake_embeddings, axis=0)

    metrics = compute_prdc(real_embeddings, fake_embeddings, nearest_k=k)

    precision = metrics['precision']
    recall = metrics['recall']
    density = metrics['density']
    coverage = metrics['coverage']

    # Save CSV in parent of fake_dir (same level as result/)
    output_dir = os.path.dirname(os.path.normpath(fake_dir))
    os.makedirs(output_dir, exist_ok=True)

    # Save as CSV with each metric as a column
    csv_file = os.path.join(output_dir, 'prdc_metrics.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['# Config'])
        writer.writerow(['vgg_type', m_type])
        writer.writerow(['vgg_feat_dim', out_feats])
        writer.writerow(['real_data_path', real_dir])
        writer.writerow(['fake_data_path', fake_dir])
        writer.writerow(['num_real', len(real_embeddings)])
        writer.writerow(['num_fake', len(fake_embeddings)])
        writer.writerow(['nearest_k', k])
        writer.writerow([])
        writer.writerow(['Precision', 'Recall', 'Density', 'Coverage'])
        writer.writerow([precision, recall, density, coverage])

    print(f"[INFO] PRDC metrics saved to: {csv_file}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Density: {density}")
    print(f"Coverage: {coverage}")

    if args.log_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_csv)), exist_ok=True)
        with open(args.log_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["# Config"])
            writer.writerow(["real_dir",     real_dir])
            writer.writerow(["fake_dir",     fake_dir])
            writer.writerow(["vgg_type",     m_type])
            writer.writerow(["vgg_feat_dim", out_feats])
            writer.writerow(["batch_size",   batch_size])
            writer.writerow(["nearest_k",    k])
            writer.writerow(["num_real",     len(real_embeddings)])
            writer.writerow(["num_fake",     len(fake_embeddings)])
            writer.writerow([])
            writer.writerow(["# Results"])
            writer.writerow(["metric", "value"])
            writer.writerow(["Precision", precision])
            writer.writerow(["Recall",    recall])
            writer.writerow(["Density",   density])
            writer.writerow(["Coverage",  coverage])
        print(f"[INFO] PRDC log saved to: {args.log_csv}")
