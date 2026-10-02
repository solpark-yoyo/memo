import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch

parser = argparse.ArgumentParser(description="diffusion memorization detection eval")
parser.add_argument("--path", default='./det_outputs/sd1_mem_gen1_modex,c|x_seed51.pt', type=str)
parser.add_argument("--npath", default='./det_outputs/sd1_nmem_gen1_modex,c|x_seed51.pt', type=str)
parser.add_argument("--path_cosine", default='./det_outputs/sd1_mem_cosine_gen1_modex,c|x_seed51.pt', type=str)
parser.add_argument("--npath_cosine", default='./det_outputs/sd1_nmem_cosine_gen1_modex,c|x_seed51.pt', type=str)
parser.add_argument("--sd_ver", default=1, type=int)
args = parser.parse_args()


def compute_tpr_at_thresholds(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    tpr_lst = []
    for thres in [0.01, 0.03]:
        closest_fpr_index = np.argmin(np.abs(fpr - thres))
        tpr_lst.append(tpr[closest_fpr_index])
    return tpr_lst


if args.sd_ver == 1:
    gamma_1 = 1.0
    gamma_2 = 2.0
elif args.sd_ver == 2:
    gamma_1 = 1.0
    gamma_2 = 0.1
else:
    gamma_1 = 2
    gamma_2 = 1

mem_data  = torch.load(args.path,        weights_only=True).mean(dim=1).float().numpy()
nmem_data = torch.load(args.npath,       weights_only=True).mean(dim=1).float().numpy()
mem_cosine  = torch.load(args.path_cosine,  weights_only=True).mean(dim=1).float().numpy()
nmem_cosine = torch.load(args.npath_cosine, weights_only=True).mean(dim=1).float().numpy()

combined_score = []
memorised_bool = []

for i in range(len(mem_data)):
    combined_score.append(gamma_1 * mem_data[i] + gamma_2 * mem_cosine[i])
    memorised_bool.append(1)

for i in range(len(nmem_data)):
    combined_score.append(gamma_1 * nmem_data[i] + gamma_2 * nmem_cosine[i])
    memorised_bool.append(0)

auc = roc_auc_score(memorised_bool, combined_score)
tpr_lst = compute_tpr_at_thresholds(memorised_bool, combined_score)
print(f'AUC: {auc:.3f} | TPR@1%FPR: {tpr_lst[0]:.3f} | TPR@3%FPR: {tpr_lst[1]:.3f}')
