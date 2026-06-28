import os, sys, torch
sys.path.insert(0, '/home/geonsoo/Desktop/Datasets/Parksol/memo/ori_memo')
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed
DEV = torch.device('cuda:0'); NFE=50; CFG=7.5; SEED=42
set_seed(SEED)
sd = StableDiffusion(solver_config=munchify({'num_sampling':NFE}),
    model_key='/home/geonsoo/Desktop/Datasets/Parksol/memo/ori_memo/ckpt/stable-diffusion-v1-5',
    device=DEV, seed=SEED)
try: del sd.safety_checker
except Exception: pass
torch.cuda.empty_cache()
if hasattr(sd.unet,'disable_gradient_checkpointing'): sd.unet.disable_gradient_checkpointing()
torch.cuda.empty_cache()
uc, c = sd.get_text_embed(null_prompt='', prompt='An astronaut on the moon')
uc=uc.float(); c=c.float()
CFG_=7.5; LAMBDA=0.1; BASE_S=0.5
def ccfg(nuc,nc): return nuc+CFG_*(nc-nuc)

for T_IDX in [1,2,3]:
    sd.unet.float(); sd.dtype=torch.float32
    timesteps=list(sd.scheduler.timesteps)
    s_idx=int(len(timesteps)*BASE_S); s_target=timesteps[s_idx]; alpha_s=sd.alpha(s_target)
    set_seed(SEED)
    x_T_init=torch.randn(1,4,16,16,device=DEV,dtype=torch.float32)
    # ref trajectory
    x0_orig_refs={}
    with torch.no_grad():
        zt=x_T_init*sd.scheduler.init_noise_sigma
        for si,t in enumerate(timesteps):
            at=sd.alpha(t); atp=sd.alpha(t-sd.skip)
            nuc,nc=sd.predict_noise(zt,t,uc,c); eps=ccfg(nuc,nc)
            x0h=(zt-(1-at).sqrt()*eps)/at.sqrt()
            zt=atp.sqrt()*x0h+(1-atp).sqrt()*eps
            if si==T_IDX: x0_orig_refs[si]=x0h.detach().clone().float(); break
    eref=x_T_init.detach()
    t_idx=T_IDX
    # adjoint path
    xk=(x_T_init.detach()*sd.scheduler.init_noise_sigma).clone(); xs=[xk.clone()]
    with torch.no_grad():
        for si,t in enumerate(timesteps):
            at=sd.alpha(t); atp=sd.alpha(t-sd.skip)
            nuc,nc=sd.predict_noise(xk,t,uc,c); eps=ccfg(nuc,nc)
            x0h=(xk-(1-at).sqrt()*eps)/at.sqrt()
            xk=atp.sqrt()*x0h+(1-atp).sqrt()*eps
            if si==t_idx: break
            if (si+1)<=t_idx: xs.append(xk.clone())
    x_end_state=xs[t_idx].clone()
    with torch.enable_grad():
        x_end=x_end_state.detach().clone().requires_grad_(True)
        tb=timesteps[t_idx]; atb=sd.alpha(tb)
        et=ccfg(*sd.predict_noise(x_end,tb,uc,c))
        x0h=(x_end-(1-atb).sqrt()*et)/atb.sqrt()
        xs_=alpha_s.sqrt()*x0h+(1-alpha_s).sqrt()*x_T_init.detach()
        ns=ccfg(*sd.predict_noise(xs_,s_target,uc,c))
        B=ns.shape[0]
        mp=(eref-ns).reshape(B,-1).pow(2).mean(-1).mean()
        la=((x0h.float()-x0_orig_refs[t_idx]).reshape(B,-1).pow(2).mean(-1).mean())
        loss=mp+LAMBDA*la
        g=torch.autograd.grad(loss,x_end,retain_graph=False)[0]
        gtn=g.flatten().norm().item()
    for k in range(t_idx,0,-1):
        j=k-1; tj=timesteps[j]; aj=sd.alpha(tj); ajp=sd.alpha(tj-sd.skip)
        A=(ajp/aj).sqrt(); Bc=(1-ajp).sqrt()-(ajp*(1-aj)/aj).sqrt()
        xl=xs[j].detach().clone().requires_grad_(True)
        ej=ccfg(*sd.predict_noise(xl,tj,uc,c))
        Jtg=torch.autograd.grad(ej,xl,grad_outputs=g,retain_graph=False)[0]
        g=A*g+Bc*Jtg
    g_adj=(g*sd.scheduler.init_noise_sigma).detach().reshape_as(x_T_init)
    # autograd path
    x_T2=x_T_init.clone().requires_grad_(True)
    xk2=x_T2*sd.scheduler.init_noise_sigma
    for si,t in enumerate(timesteps):
        at=sd.alpha(t); atp=sd.alpha(t-sd.skip)
        nuc,nc=sd.predict_noise(xk2,t,uc,c); eps=ccfg(nuc,nc)
        x0h=(xk2-(1-at).sqrt()*eps)/at.sqrt()
        xk2=atp.sqrt()*x0h+(1-atp).sqrt()*eps
        if si==t_idx: break
    x_end2=xk2
    et2=ccfg(*sd.predict_noise(x_end2,tb,uc,c))
    x0h2=(x_end2-(1-atb).sqrt()*et2)/atb.sqrt()
    xs2=alpha_s.sqrt()*x0h2+(1-alpha_s).sqrt()*x_T2.detach()
    ns2=ccfg(*sd.predict_noise(xs2,s_target,uc,c))
    mp2=(eref-ns2).reshape(B,-1).pow(2).mean(-1).mean()
    la2=((x0h2.float()-x0_orig_refs[t_idx]).reshape(B,-1).pow(2).mean(-1).mean())
    loss2=mp2+LAMBDA*la2
    g_auto=torch.autograd.grad(loss2,x_T2,retain_graph=False)[0].detach()
    def rel(a,b): return (a-b).flatten().norm().item()/(b.flatten().norm().item()+1e-12)
    e=rel(g_adj,g_auto)
    # cos sim
    cos=torch.nn.functional.cosine_similarity(g_adj.flatten().unsqueeze(0),g_auto.flatten().unsqueeze(0)).item()
    print(f"t_idx={T_IDX}: |g_adj|={g_adj.flatten().norm().item():.4e}  |g_auto|={g_auto.flatten().norm().item():.4e}  "
          f"rel_err={e:.3e}  cos_sim={cos:+.4f}  ratio_adj/auto={g_adj.flatten().norm().item()/g_auto.flatten().norm().item():.3f}  "
          f"{'EXACT' if e<1e-2 else 'MISMATCH'}")
    torch.cuda.empty_cache()
