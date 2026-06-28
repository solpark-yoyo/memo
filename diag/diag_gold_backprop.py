"""
Decisive test: is the adjoint recursion mathematically EXACT?
Compare adjoint g_xT against gold-standard full-backprop g_xT.
If cos ~ 1.0 -> adjoint correct (root cause is UPSTREAM in terminal head).
Run with empty_cache between the two paths to avoid OOM.
"""
import os, sys, torch, gc
SCRIPT_DIR = "/home/geonsoo/Desktop/Datasets/Parksol/memo/ori_memo"
sys.path.insert(0, SCRIPT_DIR)
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed

device = torch.device("cuda:0")
set_seed(42)
solver_config = munchify({"num_sampling": 50})
sd = StableDiffusion(solver_config=solver_config,
                     model_key=os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5"),
                     device=device, seed=42)
cfg = 7.5
uc, c = sd.get_text_embed(null_prompt="", prompt="An astronaut on the moon")
uc = uc.float(); c = c.float()
sd.unet.float(); sd.dtype = torch.float32
timesteps = list(sd.scheduler.timesteps)
init_steps = 2   # 2-step chain so full fp32 backprop fits in VRAM; tests adjoint exactness
t_idx = init_steps
s_idx = int(len(timesteps)*0.5); s_target = timesteps[s_idx]; alpha_s = sd.alpha(s_target)
x_T_init = torch.randn(1,4,64,64, device=device, dtype=torch.float32)
epsilon_ref = x_T_init.detach()
sd.unet.enable_gradient_checkpointing()  # fit gold backprop
def predict_noise_single(zt, t, emb):
    # call unet directly (no CFG double-batch) to halve activation memory
    t_in = t.unsqueeze(0) if len(t.shape)==0 else t
    return sd.unet(zt, t_in, encoder_hidden_states=emb)['sample']
def cfgc(nuc, nc): return nuc + cfg*(nc-nuc)

# reference x0_orig_ref
x0_orig_ref = None
with torch.no_grad():
    zt = x_T_init.to(sd.dtype)*sd.scheduler.init_noise_sigma
    for si, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        nuc, nc = sd.predict_noise(zt, t, uc, c); eps = cfgc(nuc, nc)
        x0h = (zt-(1-at).sqrt()*eps)/at.sqrt()
        zt = at_prev.sqrt()*x0h + (1-at_prev).sqrt()*eps
        if si == t_idx:
            x0_orig_ref = x0h.detach().clone().float()
            break

# GOLD: full backprop through the 10-UNet chain + head
x_T2 = x_T_init.clone().requires_grad_(True)
zt = x_T2.to(sd.dtype)*sd.scheduler.init_noise_sigma
with torch.enable_grad():
    for si, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        nuc, nc = sd.predict_noise(zt, t, uc, c); eps = cfgc(nuc, nc)
        x0h = (zt-(1-at).sqrt()*eps)/at.sqrt()
        zt = at_prev.sqrt()*x0h + (1-at_prev).sqrt()*eps
        if si == t_idx:
            break
    t_break = timesteps[t_idx]; at_break = sd.alpha(t_break)
    eps_t = cfgc(*sd.predict_noise(zt, t_break, uc, c))
    x0_hat = (zt-(1-at_break).sqrt()*eps_t)/at_break.sqrt()
    x_s = alpha_s.sqrt().to(sd.dtype)*x0_hat + (1-alpha_s).sqrt().to(sd.dtype)*x_T2.detach().to(sd.dtype)
    nuc_s, nc_s = sd.predict_noise(x_s, s_target, uc, c); eps_s = cfgc(nuc_s, nc_s)
    memo = (epsilon_ref.to(sd.dtype)-eps_s).reshape(1,-1).pow(2).mean(-1).mean()
    align = (x0_hat.float()-x0_orig_ref).reshape(1,-1).pow(2).mean(-1).mean()
    loss = memo + 0.1*align
    print(f"GOLD loss_memo={memo.item():.6f} align={align.item():.6f} total={loss.item():.6f}")
    g_xT_gold = torch.autograd.grad(loss, x_T2, retain_graph=False)[0]
print(f"||g_xT_gold|| = {g_xT_gold.flatten().norm().item():.6e}")

del x_T2, zt, x0h, eps_t, x0_hat, x_s, eps_s, nuc_s, nc_s, nuc, nc, loss, memo, align
gc.collect(); torch.cuda.empty_cache()

# ADJOINT: run the exact recursion from run_ini_opti.optimize_xT_adj
x_T = x_T_init.clone().requires_grad_(True)
# forward chain, cache
x_k = (x_T.detach()*sd.scheduler.init_noise_sigma).clone()
xs = [x_k.clone()]
with torch.no_grad():
    for si, t in enumerate(timesteps):
        at = sd.alpha(t); at_prev = sd.alpha(t - sd.skip)
        nuc, nc = sd.predict_noise(x_k, t, uc, c); eps = cfgc(nuc, nc)
        x0h = (x_k-(1-at).sqrt()*eps)/at.sqrt()
        x_k = at_prev.sqrt()*x0h + (1-at_prev).sqrt()*eps
        if si == t_idx: break
        if (si+1) <= t_idx: xs.append(x_k.clone())
x_end_state = xs[t_idx].clone()
with torch.enable_grad():
    x_end = x_end_state.detach().clone().requires_grad_(True)
    t_break = timesteps[t_idx]; at_break = sd.alpha(t_break)
    eps_t = cfgc(*sd.predict_noise(x_end, t_break, uc, c))
    x0_hat = (x_end-(1-at_break).sqrt()*eps_t)/at_break.sqrt()
    x_s = alpha_s.sqrt().to(sd.dtype)*x0_hat + (1-alpha_s).sqrt().to(sd.dtype)*x_T.detach().to(sd.dtype)
    nuc_s, nc_s = sd.predict_noise(x_s, s_target, uc, c); eps_s = cfgc(nuc_s, nc_s)
    memo = (epsilon_ref.to(sd.dtype)-eps_s).reshape(1,-1).pow(2).mean(-1).mean()
    align = (x0_hat.float()-x0_orig_ref).reshape(1,-1).pow(2).mean(-1).mean()
    loss = memo + 0.1*align
    print(f"ADJ  loss_memo={memo.item():.6f} align={align.item():.6f} total={loss.item():.6f}")
    g = torch.autograd.grad(loss, x_end, retain_graph=False)[0]
del x_end, eps_t, x0_hat, x_s, eps_s, nuc_s, nc_s, loss, memo, align
for k in range(t_idx, 0, -1):
    j = k-1; t_j = timesteps[j]
    a_j = sd.alpha(t_j); a_jp1 = sd.alpha(t_j - sd.skip)
    A_j = (a_jp1/a_j).sqrt()
    B_j = (1-a_jp1).sqrt() - (a_jp1*(1-a_j)/a_j).sqrt()
    x_j_local = xs[j].detach().clone().requires_grad_(True)
    eps_j = cfgc(*sd.predict_noise(x_j_local, t_j, uc, c))
    Jt_g = torch.autograd.grad(eps_j, x_j_local, grad_outputs=g, retain_graph=False)[0]
    g = A_j*g + B_j*Jt_g
    del x_j_local, eps_j, Jt_g
g_xT_adj = (g * sd.scheduler.init_noise_sigma).detach()
print(f"||g_xT_adj|| = {g_xT_adj.flatten().norm().item():.6e}")

cos = torch.nn.functional.cosine_similarity(g_xT_adj.flatten(), g_xT_gold.flatten(), dim=0).item()
rel_err = (g_xT_adj - g_xT_gold).flatten().norm().item() / g_xT_gold.flatten().norm().item()
print(f"\n{'='*70}")
print(f"DECISIVE: cos(adjoint g_xT, gold backprop g_xT) = {cos:.6f}")
print(f"          relative L2 error                      = {rel_err:.6f}")
print(f"{'='*70}")
print(f"If cos ~ 1.0 and rel_err < 1e-2: adjoint is EXACT.")
print(f"  => recursion is NOT the bug. lr-insensitivity root cause is UPSTREAM:")
print(f"     the terminal head g_terminal is itself prompt-blind (alpha_t=0.044,")
print(f"     x0_hat = noise-scaled garbage, eps_s near-identical across prompts).")
print(f"If cos < 0.9: adjoint has a recursion/cache/coeff bug.")
