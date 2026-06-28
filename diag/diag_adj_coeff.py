"""
Perspective-4 diagnostic: adjoint recursion signal dynamics.

Reproduces the EXACT DDIM 50-step schedule that run_ini_opti.py uses,
then computes A_j = sqrt(a_{j+1}/a_j) and B_j for every adjoint step,
plus the cumulative product prod(A_j).

KEY QUESTION (a): is A_j ~ 1 (norm preserved) or < 1 (vanishing)?
KEY QUESTION (b): how big is B_j, could B_j*(J^T g) rotate direction?
KEY QUESTION (c): cumulative prod(A_j) -> does |g| survive to x_T?
"""
import torch
from diffusers import DDIMScheduler

CKPT = "/home/geonsoo/Desktop/Datasets/Parksol/memo/ori_memo/ckpt/stable-diffusion-v1-5"

sched = DDIMScheduler.from_pretrained(CKPT, subfolder="scheduler", local_files_only=True)
total_alphas = sched.alphas_cumprod.clone()

NUM_SAMPLING = 50
sched2 = DDIMScheduler.from_pretrained(CKPT, subfolder="scheduler", local_files_only=True)
total_timesteps_pre = len(sched2.timesteps)
sched.set_timesteps(NUM_SAMPLING, device="cpu")
skip = total_timesteps_pre // NUM_SAMPLING

alphas_cumprod = torch.cat([torch.tensor([1.0]), sched.alphas_cumprod])

def alpha(t):
    if t >= 0:
        return alphas_cumprod[t]
    else:
        return sched.final_alpha_cumprod

timesteps = list(sched.timesteps)
print(f"num_sampling={NUM_SAMPLING}, total_timesteps_pre={total_timesteps_pre}, skip={skip}")
print(f"len(timesteps)={len(timesteps)}")
print(f"first 5 timesteps: {timesteps[:5]}")
print(f"last 5 timesteps: {timesteps[-5:]}")
print()

init_steps, num_steps, gap_steps = 10, 4, 3
update_indices = [init_steps + i * gap_steps for i in range(num_steps)]
update_indices = [i for i in update_indices if i < len(timesteps)]
print(f"update_indices (oi=10 default): {update_indices}")
print()

for t_idx in update_indices:
    print(f"=== Adjoint recursion for t_idx={t_idx} ({t_idx} folds: j={t_idx-1}..0) ===")
    print(f"{'j':>3} {'t_j':>6} {'a_j':>12} {'a_jp1':>12} {'A_j':>10} {'B_j':>12} {'|B|/|A|':>10}")
    print("-"*80)
    cumA = 1.0
    rows = []
    for k in range(t_idx, 0, -1):
        j = k - 1
        t_j = timesteps[j]
        a_j = alpha(t_j).item()
        a_jp1 = alpha(t_j - skip).item()
        A_j = (a_jp1 / a_j)**0.5
        B_j = (1 - a_jp1)**0.5 - (a_jp1 * (1 - a_j) / a_j)**0.5
        cumA *= A_j
        rows.append((j, t_j, a_j, a_jp1, A_j, B_j))
        print(f"{j:>3} {t_j:>6} {a_j:>12.6f} {a_jp1:>12.6f} {A_j:>10.5f} {B_j:>12.6f} {abs(B_j)/(abs(A_j)+1e-12):>10.5f}")
    growth = 1.0
    for r in rows:
        growth *= (abs(r[4]) + abs(r[5]))
    npos = sum(1 for r in rows if r[5] > 0)
    nneg = sum(1 for r in rows if r[5] < 0)
    print(f"  prod(A_j) = {cumA:.6f}   prod(|A_j|+|B_j|) = {growth:.6f}")
    print(f"  B_j>0: {npos}/{len(rows)}, B_j<0: {nneg}/{len(rows)}")
    print()
