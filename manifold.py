"""
manifold.py — memorization trajectory manifold 실험 (skill: manifold-dist)

실험 1 [off-manifold]  : --manifold {centroid_l2,knn,cosine,mmd,spectral,hausdorff,emd}
  on-manifold reference(ref_prompt, mscoco)를 기준으로 측정 풀 2개(memo·general)와의
  step별 거리를 각각 측정 → plot 2개 + 비교 오버레이 + 통합 csv
  (ref-general 곡선이 기준선 — d(ref,memo) > d(ref,general) 이면 off-manifold 지지)

실험 2 [fixation]      : --fixation {seed,prompt}  ("std" = z_t magnitude의 std)
  2-1 seed   : 고정 prompt 1개씩 × selected_seed list 별 trajectory → seed 수별 std table
  2-2 prompt : seed 고정 × num prompt list 별 trajectory(batch mean) → prompt 수별 std table

실험 3 [trajstd]        : --trajstd <tag> --sources "group:name=path" ... (group=general|memo)
  소스(prompt 파일)별 n_prompts개를 같은 shared noise에서 rollout(batch_mean으로 seed 축
  축소) → step별 prompt 간 ‖z_t‖ std 곡선을 general·memo 전체를 한 plot에 동시 오버레이
  (population 축 = prompt, 색 계열로 group 구분 — general=blue계, memo=red계)

실험 4 [seedstd]        : --seedstd <tag> --sources "group:name=path" ... (group=general|memo)
  소스별 프롬프트 각각에 대해 seed 축 std(‖z_t‖)를 구한 뒤(한 prompt에 따른 seed 민감도)
  prompt 축으로 평균 → 소스별 곡선을 general·memo 전체 한 plot에 동시 오버레이
  (2-1 seed fixation을 소스 내 전체 prompt로 일반화·평균한 버전)

출력 전부 memo/ori_memo/workdir/exp_main/manifold/ 아래 — experiment.md 설계 준수.
환경: conda div_DM, memo/ori_memo/ 에서 실행.
"""

import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse, csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from munch import munchify
from tqdm import tqdm
from latent_diffusion import StableDiffusion
from eps_trajectory import compact_spectral_block_energies

# 파일명 → 저장 폴더 태그 매핑 (run_manifold와 main 공용)
_TAG_OVERRIDES = {
    "memorized_prompts_membench": "membench",
    "wen2024_memorized_prompts": "wen2024",
    "cvpr2025_memo_prompt": "cvpr2025_memo",
    "sdv1_500_memorized": "wen500",
    "han_memorized": "han",
}

def _make_tag(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return _TAG_OVERRIDES.get(stem, stem.replace("_prompts", "").replace("_prompt", ""))


def _model_key_to_base(model_key_str):
    """model_key 문자열 → base 태그. 예: stable-diffusion-v1-4 → sd14_base"""
    mapping = {
        "stable-diffusion-v1-4": "sd14_base",
        "stable-diffusion-v1-5": "sd15_base",
        "stable-diffusion-2-1-base": "sd21_base",
        "stable-diffusion-2-base": "sd20_base",
        "stable-diffusion-xl-base-1.0": "sdxl_base",
        "stable-diffusion-3-medium-diffusers": "sd3_base",
        "FLUX.1-dev": "flux_dev",
        "FLUX.1-schnell": "flux_schnell",
    }
    return mapping.get(model_key_str, model_key_str)


# ref DB 소스 → off_manifold_exp/ 하위 실험 출력 폴더 (save 태그의 첫 성분 기준)
_REF_EXP_TAGS = {
    "laion_Aes_v2": "laion_Aes_v2",      # LAION-Aes v2 5+ (512x512 사전계산 풀)
    "ms_coco": "ms_coco/val2017",        # 기존 결과 위치 유지
}


def _ref_save_tag(ref_images):
    """--ref_images 경로 → save/ref/ 하위 태그 (ref DB 교체 실험의 핵심).

    save/ref/{src}/{subset}/results 형태: `ref/`와 마지막 `results` 사이 전체
      예) .../save/ref/laion_Aes_v2/512x512/results → laion_Aes_v2/512x512
    그 외(datasets/ 경로 직접 지정): [-3]/[-1]
      예) datasets/ms_coco/ori/val2017 → ms_coco/val2017
    """
    parts = os.path.abspath(ref_images).rstrip("/").split(os.sep)
    if "ref" in parts:
        rest = parts[parts.index("ref") + 1:]
        if rest and rest[-1] == "results":
            rest = rest[:-1]
        if rest:
            return os.path.join(*rest)
    if len(parts) >= 3:
        return os.path.join(parts[-3], parts[-1])
    return parts[-1]


def _ref_exp_tag(save_tag):
    """save ref 태그 → off_manifold_exp/ 하위 이름."""
    src = save_tag.split(os.sep)[0]
    return _REF_EXP_TAGS.get(src, save_tag)


def load_latent_pool(path, n=None, t_idx=None):
    """save/ latent.npz 로드 → **sample-major (n, T', 4, 64, 64)**.

    저장 포맷은 step-major (T, n, 4, 64, 64) — 프롬pt(샘플)별 trajectory 접근이
    쉽도록 축을 교환해 반환 (n=100 → (100, 50, 4, 64, 64)).
      n:     앞 n개 샘플 슬라이스 (대규모 사전계산 풀에서 추출)
      t_idx: 사용할 step 인덱스 리스트 (예: list(range(0, 50, 5))) — 미지정 시 전체
    """
    x = np.load(path)["x_t"]                                  # (T, n, 4, 64, 64)
    if t_idx is not None:
        x = x[list(t_idx)]
    if n is not None:
        x = x[:, :n]
    return torch.from_numpy(np.ascontiguousarray(x.transpose(1, 0, 2, 3, 4)))


def _parse_t_idx(spec, T):
    """--t_idx 문자열("0 5 10 ...") → 정렬된 step 인덱스 리스트. 미지정 시 전체."""
    if not spec:
        return list(range(T))
    sel = sorted(set(int(v) for v in spec.replace(",", " ").split()))
    bad = [s for s in sel if not (0 <= s < T)]
    if bad:
        raise SystemExit(f"[error] --t_idx 범위 오류: {bad} (T={T})")
    return sel


# ===================================================================
#  공통: 그룹 rollout — step별 latent pool 확보
# ===================================================================

def load_prompts(path, n):
    with open(path, "r") as f:
        prompts = [l.strip() for l in f.readlines() if l.strip()]
    if len(prompts) < n:
        print(f"[warn] {path}: 요청 {n}개 > 파일 {len(prompts)}개 — 있는 만큼 사용")
        n = len(prompts)
    return prompts[:n]


def seed_list(base_seed, count):
    return [base_seed + i * 1000 for i in range(count)]


@torch.no_grad()
def rollout_forward_noise(sd, image_dir, n_images, device, batch_mean=True, shared_noise=None):
    """mscoco 원본 이미지 → VAE encode → x_0 → forward noising으로 step별 latent pool.

    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    shared_noise가 제공되면: 각 image × batch_seeds개 noise를 만들어 batch_mean
    (일관된 variance + step 0에서 DDIM과 거리 0)

    Returns: pools [T, n, 4, 64, 64] fp16 · timesteps · imgs (원본 디코드 — FID용)
    """
    from PIL import Image
    timesteps = list(sd.scheduler.timesteps)
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg")))
    if len(files) < n_images:
        print(f"[warn] {image_dir}: 요청 {n_images} > 파일 {len(files)} — 있는 만큼 사용")
        n_images = len(files)
    files = files[:n_images]

    # 이미지 → [0,1] → [-1,1] → VAE encode → x_0
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    x0_list = []
    for fname in files:
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        x0_list.append(tf(img))
    x0 = torch.stack(x0_list).to(device, dtype=sd.dtype)
    x0_lat_list = []
    for i in range(0, len(x0), 8):
        x0_lat_list.append(sd.encode(x0[i:i + 8]))
    x0_lat = torch.cat(x0_lat_list, 0)
    print(f"[forward-noise] {len(files)} images → x_0 {tuple(x0_lat.shape)}")

    pools = []
    if shared_noise is not None:
        # shared_noise: (n, B, 4, 64, 64) — 각 image에 B개 noise → batch_mean
        B = shared_noise.shape[1]
        n = min(len(files), shared_noise.shape[0])
        for t in tqdm(timesteps, desc="[forward-noise] steps", unit="s"):
            at = sd.alpha(t)
            # (n, B, 4, 64, 64) → 각 image의 B개 noise 평균
            xt = at.sqrt() * x0_lat[:n, None] + (1 - at).sqrt() * shared_noise[:n].to(device, sd.dtype)
            xt = xt.mean(dim=1)                                # batch_mean → (n, 4, 64, 64)
            pools.append(xt.detach().to(torch.float16).cpu())
    else:
        for t in tqdm(timesteps, desc="[forward-noise] steps", unit="s"):
            at = sd.alpha(t)
            eps = torch.randn_like(x0_lat)
            xt = at.sqrt() * x0_lat + (1 - at).sqrt() * eps
            pools.append(xt.detach().to(torch.float16).cpu())

    # 원본 디코드 (FID용 — noising 안 한 x_0) · 배치 decode
    imgs = []
    for i in range(0, len(x0_lat), 8):
        imgs.append((sd.decode(x0_lat[i:i + 8]) / 2 + 0.5).clamp(0, 1).cpu())
    imgs = torch.cat(imgs, 0)

    stacked = torch.stack(pools)                              # [T, n, ...]
    if batch_mean:
        pass                                                  # 이미 이미지 단위 (batch 없음)
    return stacked, timesteps, imgs


@torch.no_grad()
def rollout_group(sd, prompts, seeds, cfg, device, batch_mean=True, shared_noise=None):
    """prompt별 x_T를 **batch로 초기화** — DDIM rollout.

    shared_noise (n, B, 4, 64, 64) 제공 시: prompt i의 시작점 = shared_noise[i]
    (ref forward noising과 같은 noise → step 0 거리 0)
    미제공 시: **per-seed 독립** 방식으로 x_T 생성
      (eps_trajectory.py analyze_multi_sample과 동일한 seeding):
      for each seed in seeds:
        torch.manual_seed(seed)
        x_T_si = torch.randn(1, 4, 64, 64, ...) * init_noise_sigma

    Returns:
        pools : list[T] of tensor · timesteps · imgs
    """
    timesteps = list(sd.scheduler.timesteps)
    B = len(seeds)
    if shared_noise is None:
        # ★ per-seed 독립 x_T (eps_trajectory.py analyze_multi_sample 방식)
        #   각 seed_i에 대해 torch.manual_seed(seed_i) 후 torch.randn(1, ...) → 정확히 재현
        x_T_list = []
        for seed in seeds:
            torch.manual_seed(int(seed))
            x_T_i = torch.randn(1, 4, 64, 64, device=device, dtype=sd.dtype) \
                    * sd.scheduler.init_noise_sigma
            x_T_list.append(x_T_i)
        x_T = torch.cat(x_T_list, dim=0)  # (B, 4, 64, 64)
    pools = [[] for _ in timesteps]
    final_x0 = []
    for pi, prompt in enumerate(tqdm(prompts, desc="[rollout] prompts", unit="p")):
        uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)
        uc_b = uc.repeat(B, 1, 1).to(sd.dtype)
        c_b = c.repeat(B, 1, 1).to(sd.dtype)
        if shared_noise is not None:
            zt = shared_noise[pi].to(device, sd.dtype) * sd.scheduler.init_noise_sigma
        else:
            zt = x_T.clone()
        for s_i, t in enumerate(timesteps):
            pools[s_i].append(zt.detach().to(torch.float16).cpu())
            at = sd.alpha(t)
            at_prev = sd.alpha(t - sd.skip)
            noise_uc, noise_c = sd.predict_noise(zt, t, uc_b, c_b)
            eps = noise_uc + cfg * (noise_c - noise_uc)             # [B,...]
            x0 = (zt - (1 - at).sqrt() * eps) / at.sqrt()          # Tweedie
            zt = at_prev.sqrt() * x0 + (1 - at_prev).sqrt() * eps  # DDIM (η=0)
        final_x0.append(x0.detach())
    # 최종 이미지 (step 50 = x_0) — FID용 · 배치 decode (한 번에 VAE 통과 시 OOM)
    _fx0 = torch.cat(final_x0, 0)
    imgs = []
    for i in range(0, len(_fx0), 8):
        imgs.append((sd.decode(_fx0[i:i + 8]) / 2 + 0.5).clamp(0, 1).cpu())
    imgs = torch.cat(imgs, 0)
    stacked = [torch.cat(p, dim=0) for p in pools]                  # [n*B, ...]
    if batch_mean:
        n = len(prompts)
        stacked = [pl.reshape(n, B, *pl.shape[1:]).float().mean(1).to(torch.float16)
                   for pl in stacked]
    return stacked, timesteps, imgs


# ===================================================================
#  거리 6종 (rules/dist-measure.md 정의 준수) — 입력: [N,D] float32 두 그룹
# ===================================================================

def _pairwise(A, B):
    """L2 거리 행렬 [N,M]"""
    a2 = (A ** 2).sum(1, keepdim=True)
    b2 = (B ** 2).sum(1, keepdim=True).T
    d2 = a2 + b2 - 2.0 * (A @ B.T)
    return d2.clamp_min(0).sqrt()


def dist_centroid_l2(G, M, k=None):
    return (G.mean(0) - M.mean(0)).norm().item()


def dist_knn(G, M, k=5):
    D = _pairwise(M, G)                       # memo 각 점 → general
    k = max(1, min(int(k), G.shape[0]))       # 풀 크기 초과 방지 (소규모 smoke)
    idx = D.topk(k, largest=False).indices    # 최근접 k
    return D.gather(1, idx).mean().item()


def dist_cosine(G, M, k=None):
    g, m = G.mean(0), M.mean(0)
    return (1 - torch.cosine_similarity(g, m, dim=0)).item()


def dist_mmd(G, M, k=None):
    def rbf(X, Y, bw):
        d2 = ((X[:, None] - Y[None]) ** 2).sum(-1)
        return torch.exp(-d2 / (2 * bw ** 2))
    Z = torch.cat([G, M])
    bw = _pairwise(Z, Z)[_pairwise(Z, Z) > 0].median()  # median heuristic
    kg, km = rbf(G, G, bw), rbf(M, M, bw)
    kgm = rbf(G, M, bw)
    return (kg.mean() + km.mean() - 2 * kgm.mean()).item()


def dist_spectral(G, M, k=None):
    def spec_center(X):
        E = torch.stack([torch.from_numpy(
            compact_spectral_block_energies(x[None].float(), block_size=16))
            for x in X])                       # (N, P)
        return E.mean(0)
    return (spec_center(G) - spec_center(M)).norm().item()


def dist_hausdorff(G, M, k=None):
    D = _pairwise(G, M)                        # knn 과 행렬 공유
    h_gm = D.min(dim=1).values.max().item()    # max_g min_m
    h_mg = D.min(dim=0).values.max().item()    # max_m min_g
    return max(h_gm, h_mg)                     # symmetric Hausdorff


def dist_emd(G, M, k=None):
    """W1 (Sinkhorn, POT) — 전역 분포 거리 표준.
    reg = 거리행렬 median × 0.05 (검증 2026-08-27: 동일 분포→0.001, 평행이동→정확,
    ×0.01은 수치 붕괴)."""
    import ot
    D = _pairwise(G, M)
    a = torch.full((len(G),), 1.0 / len(G), dtype=D.dtype)
    b = torch.full((len(M),), 1.0 / len(M), dtype=D.dtype)
    # median=0 폴백: 동일 seed 초기 step 등에서 D의 0이 절반 이상 → reg=0 발산 방지
    nz = D[D > 0]
    base = nz.median().item() if nz.numel() > 0 else 1.0
    reg = max(D.median().item(), base) * 0.05
    return float(ot.sinkhorn2(a, b, D, reg=reg, numItermax=5000))


def dist_frechet(G, M, k=None):
    """Fréchet distance (FID-style) — 가우시안 N(μ, diag σ²) 근사의 Wasserstein-2.

    diag 근사 폐형식: d² = ‖μ_G−μ_M‖² + ‖σ_G−σ_M‖² (평균 거리 + 폭 차이) → sqrt 반환
    (emd와 스케일 정렬). FID와 동일 이론 근거 — 표본 수에 강인 (가우시안 폐형식)."""
    mg, mm = G.mean(0), M.mean(0)
    sg, sm = G.std(0, unbiased=False), M.std(0, unbiased=False)
    return float(((mg - mm).norm() ** 2 + (sg - sm).norm() ** 2).sqrt())


DIST = {"centroid_l2": dist_centroid_l2, "knn": dist_knn, "cosine": dist_cosine,
        "mmd": dist_mmd, "spectral": dist_spectral, "hausdorff": dist_hausdorff,
        "emd": dist_emd, "frechet": dist_frechet}


# ===================================================================
#  최종 이미지 FID (Inception-V3, 전체 공분산 Fréchet)
# ===================================================================

_INCEPTION = None


def _inception_feat(imgs, device, bs=64):
    """imgs [N,3,512,512] in [0,1] → Inception-V3 pool3 feature [N,2048]"""
    global _INCEPTION
    if _INCEPTION is None:
        from torchvision.models import inception_v3, Inception_V3_Weights
        m = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        m.fc = torch.nn.Identity()          # avgpool 출력 2048-d
        _INCEPTION = m.to(device).eval()
    feats = []
    for i in range(0, len(imgs), bs):
        x = torch.nn.functional.interpolate(
            imgs[i:i + bs], size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5                 # Inception 정규화 [-1,1]
        with torch.no_grad():
            feats.append(_INCEPTION(x.to(device)).float().cpu())
    return torch.cat(feats)


def fid_from_images(imgsA, imgsB, device):
    """진짜 FID — 두 이미지 풀의 Inception feature Fréchet (전체 공분산)"""
    import numpy as np
    from scipy import linalg
    fA = _inception_feat(imgsA, device).numpy()
    fB = _inception_feat(imgsB, device).numpy()
    mu1, mu2 = fA.mean(0), fB.mean(0)
    s1, s2 = np.cov(fA, rowvar=False), np.cov(fB, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float((mu1 - mu2).dot(mu1 - mu2) + np.trace(s1 + s2 - 2.0 * covmean))


# ===================================================================
#  실험 1 — off-manifold
# ===================================================================

def run_manifold(sd, args, device):
    """실험 1 — on-manifold reference 고정 3-풀 구조 (experiment.md [1] 절차).

    a. ref 풀(mscoco 등 reference DB) = on-manifold trajectory (기준점 고정)
    b. 측정 풀 2개 — memo 풀 · general 풀(별도 소스, ref과 독립 seed)
    c/d. step별 d(ref, memo) · d(ref, general) 2곡선 → plot 2개 + 비교 오버레이 + 통합 csv
    (그룹 간 독립 seed: ref 42+ / general +150000 / memo +250000 — 결함 노트 준수)
    """
    ref_prompts = load_prompts(args.ref_prompt, args.n_prompts)
    general = load_prompts(args.general_prompt, args.n_prompts)
    memo = load_prompts(args.memo_prompt, args.n_prompts)

    # 3그룹 동일 seed 공유 — step 0의 x_T가 정확히 동일(거리 0)에서 출발,
    # 이후 곡선 = 프롬프트가 trajectory를 갈라놓는 순수 효과만 측정
    seeds_ref = seed_list(args.seed, args.batch)
    seeds_gen = seeds_ref
    seeds_memo = seeds_ref

    # ---- Latent 재사용: save/에 있으면 rollout 생략 ----
    _bname = os.path.basename(args.model_key.rstrip("/"))
    model_tag = _bname                                          # 경로용 (디렉토리명과 정합)
    model_disp = "sd" + _bname.split("v")[-1].replace("-", "") if "v" in _bname else _bname  # 표시용
    sampler_tag = type(sd.scheduler).__name__.replace("Scheduler", "").replace("Discrete", "").lower() if sd else "ddim"
    _base = os.path.join(args.out_root, model_tag, sampler_tag,
                         f"CFG={args.cfg}_NFE={args.NFE}", f"seed={args.seed}",
                         f"size=512x512", f"batch={args.batch}")

    def _tag(path):
        return _make_tag(path)  # module-level 매핑 사용

    r_tag, g_tag, m_tag = _tag(args.ref_prompt), _tag(args.general_prompt), _tag(args.memo_prompt)
    if args.ref_images:
        r_tag = _ref_save_tag(args.ref_images)

    def _latent_path(grp, tag):
        return os.path.join(_base, "save", grp, tag, "record", "latent", "latent.npz")

    _paths = [_latent_path("ref", r_tag), _latent_path("general", g_tag),
              _latent_path("memo", m_tag)]

    if all(os.path.exists(p) for p in _paths):
        print("[load-save] all 3 latent.npz exist — skip rollout, start experiment")
        # load_latent_pool: n_prompts 슬라이스 + sample-major 반환 → 기존 step-major로 교환
        R_pools = load_latent_pool(_paths[0], n=args.n_prompts).transpose(0, 1)
        G_pools = load_latent_pool(_paths[1], n=args.n_prompts).transpose(0, 1)
        M_pools = load_latent_pool(_paths[2], n=args.n_prompts).transpose(0, 1)
        timesteps = list(range(R_pools.shape[0]))         # T from shape (모델 불필요)
        R_imgs = G_imgs = M_imgs = None                   # 이미지 없음 (FID 생략)
    else:
        # ---- 공유 noise 생성: ref·general·memo가 같은 noise에서 출발 (step 0 거리 0) ----
        torch.manual_seed(args.seed)
        shared_noise = torch.randn(
            args.n_prompts, args.batch, 4, 64, 64,
            device=device, dtype=torch.float32)
        print(f"[shared-noise] {tuple(shared_noise.shape)} — 3 groups share (distance 0 at step 0)")

        if args.ref_images:
            print(f"[forward-noise ref] {args.ref_images} (on-manifold = real data forward noising)")
            R_pools, timesteps, R_imgs = rollout_forward_noise(
                sd, args.ref_images, args.n_prompts, device, shared_noise=shared_noise)
        else:
            print(f"[rollout] ref (on-manifold): {len(ref_prompts)} × {len(seeds_ref)}")
            R_pools, timesteps, R_imgs = rollout_group(sd, ref_prompts, seeds_ref, args.cfg,
                                                        device, shared_noise=shared_noise)
        print(f"[rollout] general (measurement pool): {len(general)} × {len(seeds_gen)}")
        G_pools, _, G_imgs = rollout_group(sd, general, seeds_gen, args.cfg,
                                            device, shared_noise=shared_noise)
        print(f"[rollout] memo (measurement pool): {len(memo)} × {len(seeds_memo)}")
        M_pools, _, M_imgs = rollout_group(sd, memo, seeds_memo, args.cfg,
                                            device, shared_noise=shared_noise)

    fn = DIST[args.manifold]
    k = args.knn_k if args.manifold == "knn" else None
    # Measurement step subset (--t_idx) — rollout full T, distance calc on selected steps
    sel = _parse_t_idx(args.t_idx, len(timesteps))
    if len(sel) < len(timesteps):
        print(f"[t_idx] measure {len(sel)} of {len(timesteps)} steps: {sel}")
    steps, d_ref_memo, d_ref_general = [], [], []
    for s_i in sel:
        R = R_pools[s_i].float().reshape(len(R_pools[s_i]), -1)
        G = G_pools[s_i].float().reshape(len(G_pools[s_i]), -1)
        M = M_pools[s_i].float().reshape(len(M_pools[s_i]), -1)
        d_ref_memo.append(fn(R, M, k))
        d_ref_general.append(fn(R, G, k))
        steps.append(s_i)
        if s_i % 10 == 0:
            print(f"  step {s_i}: d(ref,memo)={d_ref_memo[-1]:.4f}  "
                  f"d(ref,general)={d_ref_general[-1]:.4f}")

    _bname = os.path.basename(args.model_key.rstrip("/"))
    model_tag = _bname                                          # 경로용 (디렉토리명과 정합)
    model_disp = "sd" + _bname.split("v")[-1].replace("-", "") if "v" in _bname else _bname  # 표시용

    base = _base  # rollout 전에 계산한 경로 재사용
    save_root = os.path.join(base, "save")
    exp_root = os.path.join(base, "off_manifold_exp", _ref_exp_tag(r_tag), args.manifold)
    plot_dir = os.path.join(exp_root, "plot")
    csv_dir = os.path.join(exp_root, "csv")

    # save는 rollout 했을 때만 기록 (load-save 시 이미 저장돼 있음)
    if R_imgs is not None:
        from torchvision.utils import save_image
        for grp, pool_list, imgs, sub_tag in (
                ("ref", R_pools, R_imgs, r_tag),
                ("general", G_pools, G_imgs, g_tag),
                ("memo", M_pools, M_imgs, m_tag)):
            grp_results = os.path.join(save_root, grp, sub_tag, "results")
            grp_latent = os.path.join(save_root, grp, sub_tag, "record", "latent")
            grp_tweedie = os.path.join(save_root, grp, sub_tag, "record", "tweedie")
            for d in (grp_results, grp_latent, grp_tweedie):
                os.makedirs(d, exist_ok=True)
            for i in range(len(imgs)):
                save_image(imgs[i], os.path.join(grp_results, f"{grp}_{i // len(seeds_ref):04d}_"
                                                             f"{i % len(seeds_ref):02d}.png"))
            _xt = pool_list if isinstance(pool_list, torch.Tensor) else torch.stack(pool_list)
            np.savez_compressed(
                os.path.join(grp_latent, "latent.npz"),
                x_t=_xt.numpy(),
                prompts=[f"{grp}_{i}" for i in range(_xt.shape[1])],
            )

    os.makedirs(plot_dir, exist_ok=True); os.makedirs(csv_dir, exist_ok=True)

    # x축 값: step index(0=노이즈) → time(50=노이즈, 0=이미지) 로 변환
    times = [args.NFE - s for s in steps]                     # [50, 49, ..., 1]

    def _plot(ys, name, title):
        plt.figure(figsize=(9, 6))
        plt.plot(times, ys, "o-", linewidth=2.2, markersize=5)
        plt.xlabel(r"Time ($t$)", fontsize=13)
        plt.ylabel(rf"$d^{{\mathcal{{M}}}}_{{{args.manifold}}}$", fontsize=13)
        plt.title(title, fontsize=12)
        plt.xlim(args.NFE, 0)
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, name), dpi=150); plt.close()

    _plot(d_ref_memo, "distance_ref_memo.png",
          f"[Memo] {m_tag} — {model_disp}")
    _plot(d_ref_general, "distance_ref_general.png",
          f"[Gen] {g_tag} — {model_disp}")
    # 비교 오버레이 — plot 1(memo) > plot 2(general) 이면 off-manifold 지지
    plt.figure(figsize=(9, 6))
    plt.plot(times, d_ref_memo, "o-", color="tab:red", linewidth=2.2, markersize=5,
             label=f"[Memo] {m_tag}")
    plt.plot(times, d_ref_general, "s-", color="tab:blue", linewidth=2.2, markersize=5,
             label=f"[Gen] {g_tag}")
    plt.xlabel(r"Time ($t$)", fontsize=13)
    plt.ylabel(rf"$d^{{\mathcal{{M}}}}(\mathrm{{{args.manifold}}})$", fontsize=13)
    plt.title(f"Manifold Distance ({model_disp})  |  ref: {_ref_exp_tag(r_tag)}", fontsize=13)
    plt.xlim(args.NFE, 0)
    plt.grid(True, alpha=0.3); plt.legend(fontsize=11); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "distance_compare.png"), dpi=150); plt.close()

    with open(os.path.join(csv_dir, "manifold_distance.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "d_ref_memo", "d_ref_general"])
        for s, a, b in zip(steps, d_ref_memo, d_ref_general):
            w.writerow([s, f"{a:.6f}", f"{b:.6f}"])

    # ---- Final (step 50 = x_0) image FID — two pools w.r.t ref (real FID, Inception-V3) ----
    if not args.skip_fid and R_imgs is not None:
        fid_memo = fid_from_images(R_imgs, M_imgs, device)
        fid_gen = fid_from_images(R_imgs, G_imgs, device)
        print(f"\n[FID final] ref↔memo = {fid_memo:.2f} | ref↔general = {fid_gen:.2f}")
        with open(os.path.join(csv_dir, "fid_final.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pair", "fid"])
            w.writerow(["ref-memo", f"{fid_memo:.4f}"])
            w.writerow(["ref-general", f"{fid_gen:.4f}"])
    print(f"\n[Done] {args.manifold} -> {base}")


# ===================================================================
#  전체 소스 비교 — save/의 모든 general·memo latent를 한 plot에
# ===================================================================

def run_compare_all(sd, args, device):
    """save/에 존재하는 모든 general·memo latent를 로드 →
    d(ref, source)를 각각 계산 → **한 plot에 전부 오버레이**."""
    model_tag = os.path.basename(args.model_key.rstrip("/"))
    model_disp = "sd" + model_tag.split("v")[-1].replace("-", "") if "v" in model_tag else model_tag
    sampler_tag = type(sd.scheduler).__name__.replace("Scheduler", "").replace("Discrete", "").lower() if sd else "ddim"
    base = os.path.join(args.out_root, model_tag, sampler_tag,
                        f"CFG={args.cfg}_NFE={args.NFE}", f"seed={args.seed}",
                        f"size=512x512", f"batch={args.batch}")
    save_root = os.path.join(base, "save")

    # ref latent 로드 (ref DB 교체: --ref_images로 save/ref/{src}/{subset}/results 지정)
    r_tag = _ref_save_tag(args.ref_images) if args.ref_images else _make_tag(args.ref_prompt)
    ref_npz = os.path.join(save_root, "ref", r_tag, "record", "latent", "latent.npz")
    if not os.path.exists(ref_npz):
        print(f"[error] ref latent 없음: {ref_npz}"); return
    R_pools = load_latent_pool(ref_npz, n=args.n_prompts).transpose(0, 1)  # step-major (T, n, 4, 64, 64)
    T = R_pools.shape[0]
    sel = _parse_t_idx(args.t_idx, T)
    times = [args.NFE - s for s in sel]
    if len(sel) < T:
        print(f"[t_idx] {T} step 중 {len(sel)}개 측정: {sel}")
    print(f"[ref] {r_tag} — {R_pools.shape}")

    # 사용 가능한 general·memo latent 탐색
    sources = []
    for grp in ("general", "memo"):
        grp_dir = os.path.join(save_root, grp)
        if not os.path.isdir(grp_dir):
            continue
        for sub in sorted(os.listdir(grp_dir)):
            npz = os.path.join(grp_dir, sub, "record", "latent", "latent.npz")
            if os.path.exists(npz):
                sources.append((grp, sub, npz))
                print(f"[found] [{grp}] {sub}")
    if not sources:
        print("[error] 측정 대상 latent 없음"); return

    # 거리 계산
    fn = DIST[args.manifold]
    k = args.knn_k if args.manifold == "knn" else None
    curves = {}
    for grp, name, npz in sources:
        pools = torch.from_numpy(np.load(npz)["x_t"])
        ds = []
        for s_i in sel:
            R = R_pools[s_i].float().reshape(len(R_pools[s_i]), -1)
            S = pools[s_i].float().reshape(len(pools[s_i]), -1)
            ds.append(fn(R, S, k))
        curves[(grp, name)] = ds
        print(f"  [{grp}] {name}: step{sel[-1]}={ds[-1]:.2f}")

    # Plot — 전부 한 figure에
    gen_colors = ["tab:blue", "tab:cyan", "tab:purple"]
    memo_colors = ["tab:red", "tab:orange", "tab:pink"]
    fig, ax = plt.subplots(figsize=(10, 7))
    gi = mi = 0
    for (grp, name), ds in curves.items():
        if grp == "general":
            color = gen_colors[gi % len(gen_colors)]; gi += 1
            label = f"[Gen] {name}"
            ax.plot(times, ds, "s-", color=color, linewidth=2, markersize=5, label=label)
        else:
            color = memo_colors[mi % len(memo_colors)]; mi += 1
            label = f"[Memo] {name}"
            ax.plot(times, ds, "o-", color=color, linewidth=2, markersize=5, label=label)

    ax.set_xlabel(r"Time ($t$)", fontsize=13)
    ax.set_ylabel(rf"$d^{{\mathcal{{M}}}}_{{{args.manifold}}}$", fontsize=13)
    ax.set_title(f"Manifold Distance ({model_disp})  |  ref: {_ref_exp_tag(r_tag)}", fontsize=13)
    ax.set_xlim(args.NFE, 0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()

    out_dir = os.path.join(base, "off_manifold_exp", _ref_exp_tag(r_tag), f"compare_{args.manifold}")
    os.makedirs(os.path.join(out_dir, "plot"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "csv"), exist_ok=True)
    plt.savefig(os.path.join(out_dir, "plot", "distance_compare_all.png"), dpi=150)
    plt.close()

    # CSV 저장
    with open(os.path.join(out_dir, "csv", "distance_compare_all.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["time"] + [f"[{g}]{n}" for g, n in curves.keys()]
        w.writerow(header)
        for i in range(len(sel)):
            row = [times[i]] + [f"{curves[k][i]:.6f}" for k in curves.keys()]
            w.writerow(row)

    # ---- [1-1] Deviation graph: domain gap 제거 (general mean 대비 편차) ----
    # general 소스들의 mean을 각 time에서 계산 → 각 소스의 편차 = d - gen_mean
    gen_keys = [(g, n) for g, n in curves.keys() if g == "general"]
    memo_keys = [(g, n) for g, n in curves.keys() if g == "memo"]
    if gen_keys:
        gen_mean = [sum(curves[k][i] for k in gen_keys) / len(gen_keys) for i in range(len(sel))]

        fig, ax = plt.subplots(figsize=(10, 7))
        # general mean 기준선 (= 0)
        ax.axhline(y=0, color="gray", linewidth=1, linestyle="--", alpha=0.5)

        gi = mi = 0
        for (grp, name), ds in curves.items():
            dev = [ds[i] - gen_mean[i] for i in range(len(sel))]
            if grp == "general":
                color = gen_colors[gi % len(gen_colors)]; gi += 1
                ax.plot(times, dev, "s--", color=color, linewidth=1.5, markersize=4,
                        alpha=0.6, label=f"[Gen] {name}")
            else:
                color = memo_colors[mi % len(memo_colors)]; mi += 1
                ax.plot(times, dev, "o-", color=color, linewidth=2.2, markersize=5,
                        label=f"[Memo] {name}")

        ax.set_xlabel(r"Time ($t$)", fontsize=13)
        ax.set_ylabel(rf"$d^{{\prime\mathcal{{M}}}}_{{{args.manifold}}}$", fontsize=13)
        ax.set_title(f"Manifold Distance ({model_disp})  |  ref: {_ref_exp_tag(r_tag)}", fontsize=13)
        ax.set_xlim(args.NFE, 0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot", "deviation_compare_all.png"), dpi=150)
        plt.close()

        # CSV
        with open(os.path.join(out_dir, "csv", "deviation_compare_all.csv"), "w", newline="") as f:
            w = csv.writer(f)
            header = ["time", "gen_mean"] + [f"dev_[{g}]{n}" for g, n in curves.keys()]
            w.writerow(header)
            for i in range(len(sel)):
                row = [times[i], f"{gen_mean[i]:.6f}"] + \
                      [f"{curves[k][i] - gen_mean[i]:.6f}" for k in curves.keys()]
                w.writerow(row)

    print(f"\n[Done] → {out_dir}")


# ===================================================================
#  실험 2 — fixation ("std" = z_t magnitude의 std)
# ===================================================================

def _mag(pool):
    """pool [N,4,64,64] → magnitude [N]"""
    return pool.float().reshape(len(pool), -1).norm(dim=1)


def run_fixation(sd, args, device):
    general = load_prompts(args.general_prompt, args.n_prompts_max)
    memo = load_prompts(args.memo_prompt, args.n_prompts_max)
    _bname = os.path.basename(args.model_key.rstrip("/"))
    model_tag = _bname                                          # 경로용 (디렉토리명과 정합)
    model_disp = "sd" + _bname.split("v")[-1].replace("-", "") if "v" in _bname else _bname  # 표시용

    if args.fixation == "seed":
        # 2-1: 고정 prompt 1개씩 × selected_seed list 별 → seed 수별 std table
        g_fix, m_fix = general[args.fix_general_idx], memo[args.fix_memo_idx]
        counts = [int(x) for x in args.seed_list.split()]
        print(f"[2-1 seed] g=\"{g_fix[:40]}\" m=\"{m_fix[:40]}\" seeds={counts}")
        rows, curves = [], {}
        for s in counts:
            seeds = seed_list(args.seed, s)
            Gp, _, _ = rollout_group(sd, [g_fix], seeds, args.cfg, device, batch_mean=False)
            Mp, _, _ = rollout_group(sd, [m_fix], seeds, args.cfg, device, batch_mean=False)
            g_std = torch.stack([_mag(p).std(unbiased=False) for p in Gp])  # per-step std
            m_std = torch.stack([_mag(p).std(unbiased=False) for p in Mp])
            rows.append((s, g_std.mean().item(), m_std.mean().item()))
            curves[s] = (g_std.numpy(), m_std.numpy())
            print(f"  seeds={s}: std(general)={rows[-1][1]:.4f} std(memo)={rows[-1][2]:.4f}")
        out = os.path.join(args.out_root, model_tag, "fixation_seed",
                           f"gi={args.fix_general_idx}_mi={args.fix_memo_idx}")
        os.makedirs(os.path.join(out, "csv"), exist_ok=True)
        os.makedirs(os.path.join(out, "plot"), exist_ok=True)
        with open(os.path.join(out, "csv", "std_table.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["num_seeds", "std_general", "std_memo"])
            for s, g, m in rows: w.writerow([s, f"{g:.6f}", f"{m:.6f}"])
        fig, ax = plt.subplots(figsize=(9, 6))
        for s, (gs, ms) in curves.items():
            ax.plot(range(len(gs)), gs, "--", alpha=0.5, label=f"general n={s}")
            ax.plot(range(len(ms)), "-", alpha=0.8, label=f"memo n={s}")
        ax.set_xlabel("Denoising Step", fontsize=12)
        ax.set_ylabel(r"std of $\|z_t\|$ (across seeds)", fontsize=12)
        ax.set_title("[2-1 seed fixation] per-step ‖z_t‖ std", fontsize=12)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "plot", "std_curve.png"), dpi=150); plt.close()
        print(f"\n[Done] fixation(seed) -> {out}")

    else:
        # 2-2: fixed seed × num prompt list (batch mean z_t) -> per-prompt count std table
        counts = [int(x) for x in args.np_list.split()]
        seeds = seed_list(args.seed, args.batch)
        print(f"[2-2 prompt] seeds={seeds} np={counts}")
        rows = []
        for n in counts:
            Gp, _, _ = rollout_group(sd, general[:n], seeds, args.cfg, device, batch_mean=False)
            Mp, _, _ = rollout_group(sd, memo[:n], seeds, args.cfg, device, batch_mean=False)

            def rep_std(pools, n):
                """each step pool [n*seeds,...] -> per-prompt batch-mean z_t -> magnitude std ->
                   step average"""
                vals = []
                for pl in pools:
                    x = pl.reshape(n, len(seeds), -1).float().mean(1)  # per-prompt batch-mean
                    vals.append(x.norm(dim=1).std(unbiased=False))
                return torch.stack(vals).mean().item()

            g_std = rep_std(Gp, n); m_std = rep_std(Mp, n)
            rows.append((n, g_std, m_std))
            print(f"  np={n}: std(general)={g_std:.4f} std(memo)={m_std:.4f}")
        out = os.path.join(args.out_root, model_tag, "fixation_prompt",
                           f"seed={args.seed}_batch={args.batch}")
        os.makedirs(os.path.join(out, "csv"), exist_ok=True)
        with open(os.path.join(out, "csv", "std_table.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["num_prompts", "std_general", "std_memo"])
            for n, g, m in rows: w.writerow([n, f"{g:.6f}", f"{m:.6f}"])
        print(f"\n[Done] fixation(prompt) -> {out}")


# ===================================================================
#  실험 3 — trajstd (population 축 = 소스 내 서로 다른 prompt)
#  "std" = general·memo 소스들 각각에 대해, 해당 소스의 n개 prompt가 같은 shared
#  noise에서 출발했을 때 step별 ‖z_t‖가 prompt 간 얼마나 벌어지는지(collapse 여부)를
#  general vs memo를 **한 plot에 동시에** 겹쳐 그린다 (색 계열로 그룹 구분).
# ===================================================================

def _parse_sources(items):
    """--sources "group:name=path" 리스트 → [(group, name, path), ...].
    group 미지정("name=path")이면 group="general" 기본."""
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f'[error] --sources 형식 오류 (group:name=path 필요): "{item}"')
        head, path = item.split("=", 1)
        group, _, name = head.rpartition(":")
        out.append((group or "general", name, path))
    return out


# group별 색 계열 (run_compare_all과 동일 팔레트 — general=blue계, memo=red계)
_GROUP_COLORS = {
    "general": ["tab:blue", "tab:cyan", "tab:purple", "tab:green"],
    "memo": ["tab:red", "tab:orange", "tab:pink", "tab:brown"],
}


def _plot_group_curves(curves, out, title, ylabel, NFE):
    """{(group, name): np.ndarray[T]} → general·memo 색 계열로 한 plot에 오버레이 + csv 저장.
    (run_trajstd·run_seedstd 공용)"""
    os.makedirs(os.path.join(out, "csv"), exist_ok=True)
    os.makedirs(os.path.join(out, "plot"), exist_ok=True)

    T = len(next(iter(curves.values())))
    times = [NFE - s for s in range(T)]

    fig, ax = plt.subplots(figsize=(9, 6))
    color_idx = {}
    for (group, name), ys in curves.items():
        palette = _GROUP_COLORS.get(group, ["tab:gray", "tab:olive"])
        i = color_idx.get(group, 0); color_idx[group] = i + 1
        marker = "o-" if group == "general" else "s-"
        ax.plot(times, ys, marker, color=palette[i % len(palette)], linewidth=2, markersize=4,
                 label=f"[{group}] {name}")
    ax.set_xlabel(r"Time ($t$)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(NFE, 0)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "plot", "std_curve.png"), dpi=150)
    plt.close()

    with open(os.path.join(out, "csv", "std_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["step", "time"] + [f"[{g}]{n}" for g, n in curves.keys()]
        w.writerow(header)
        for s in range(T):
            w.writerow([s, times[s]] + [f"{curves[k][s]:.6f}" for k in curves.keys()])


def validate_trajstd_outputs(out, args):
    """Validate trajstd curve outputs exist.

    Checks:
      - csv/std_curve.csv
      - plot/std_curve.png
    """
    print(f"\n[validating outputs...]")
    csv_path = os.path.join(out, "csv", "std_curve.csv")
    plot_path = os.path.join(out, "plot", "std_curve.png")

    checks = {
        "csv/std_curve.csv": os.path.exists(csv_path),
        "plot/std_curve.png": os.path.exists(plot_path),
    }

    print(f"  Trajectory curves (prompts={args.n_prompts}):")
    validation_ok = True
    for check, ok in checks.items():
        marker = "  ✓" if ok else "  ✗"
        print(f"      {marker} {check}")
        if not ok:
            validation_ok = False

    if validation_ok:
        print(f"\n[✓ validation passed] All expected outputs exist")
    else:
        print(f"\n[✗ validation failed] Some outputs missing — check above")

    return validation_ok


def run_trajstd(sd, args, device):
    """Per-source: rollout n_prompts -> seed batch-mean -> per-step prompt-axis std
    -> overlay general·memo curves in one plot."""
    sources = _parse_sources(args.sources)
    seeds = seed_list(args.seed, args.batch)
    model_tag = os.path.basename(args.model_key.rstrip("/"))

    curves = {}   # (group, name) -> np.ndarray[T]
    for group, name, path in sources:
        prompts = load_prompts(path, args.n_prompts)
        print(f"[trajstd] [{group}] {name}: {len(prompts)} prompts × {len(seeds)} seeds(batch-mean)")
        pools, timesteps, _ = rollout_group(sd, prompts, seeds, args.cfg, device, batch_mean=True)
        # pools: list[T] of tensor (n_prompts, 4, 64, 64) — already batch-mean across seeds
        std_curve = torch.stack([_mag(p).std(unbiased=False) for p in pools])
        curves[(group, name)] = std_curve.numpy()

    out = os.path.join(args.out_root, model_tag, f"trajstd_{args.trajstd}", f"prompts={args.n_prompts}")
    _plot_group_curves(
        curves, out,
        title=f"[trajstd:{args.trajstd}] std($\\|z_t\\|$) across prompts, per step (general vs memo)",
        ylabel=r"std of $\|z_t\|$ (across prompts)", NFE=args.NFE)

    # validate outputs
    validate_trajstd_outputs(out, args)

    print(f"\n[Done] trajstd({args.trajstd}) -> {out}")


def analyze_cross_seed_deviation(source_pools_dict, curve_base, args):
    """Per-source seed deviation analysis: visualize how much each seed deviates from mean.

    Input:
      source_pools_dict: {(group, name): pools} where pools = list[T] of tensor(n*B, 4, 64, 64)

    Computation:
      at each step t:
        mag = reshape(n, B, -1).norm(dim=2) → (n, B)
        mean_per_prompt = mean(dim=1, keepdim=True) → (n, 1)
        deviation = mag - mean_per_prompt → (n, B)
        avg_deviation_per_seed = mean(dim=0) → (B,)  [avg over all prompts per seed]
      result: (T, B) — T steps × B seeds

    Output:
      csv: step, time, seed_0, seed_1, seed_2, seed_3, seed_4
      plot: B curve overlays (1 line per seed) — deviation from mean
    """
    if not source_pools_dict:
        return

    T = len(next(iter(source_pools_dict.values())))
    times = [args.NFE - s for s in range(T)]

    csv_dir = os.path.join(curve_base, "csv")
    plot_dir = os.path.join(curve_base, "plot")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # 소스별로 분석
    for (group, name), pools in source_pools_dict.items():
        # pools: list[T] of tensor(n*B, 4, 64, 64)
        n = len(pools[0]) // args.batch  # number of prompts
        B = args.batch

        deviations = []  # list[T] of (B,)
        for pl in pools:
            mag = pl.float().reshape(n, B, -1).norm(dim=2)  # (n, B)
            mean_per_prompt = mag.mean(dim=1, keepdim=True)  # (n, 1)
            dev = (mag - mean_per_prompt).abs()  # (n, B) — absolute deviation
            avg_dev_per_seed = dev.mean(dim=0)  # (B,) — average over prompts
            deviations.append(avg_dev_per_seed)

        deviations_arr = torch.stack(deviations).numpy()  # (T, B)

        # CSV 저장
        csv_path = os.path.join(csv_dir, f"cross_seed_deviation_{group}_{name}.csv")
        with open(csv_path, 'w', newline='') as f:
            header = ['step', 'time'] + [f'seed_{i}' for i in range(B)]
            writer = csv.writer(f)
            writer.writerow(header)
            for s in range(T):
                row = [s, times[s]] + [f"{deviations_arr[s, i]:.6f}" for i in range(B)]
                writer.writerow(row)
        print(f"[cross-seed-dev] [{group}] {name}: CSV saved -> {csv_path}")

        # Save plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 5 seeds
        for i in range(B):
            ax.plot(times, deviations_arr[:, i], color=colors[i % len(colors)],
                    linewidth=2, marker='o', markersize=4, label=f'seed_{i}', alpha=0.8)

        ax.set_xlabel(r"Time ($t$)", fontsize=13)
        ax.set_ylabel(r"avg deviation from prompt-mean of $\|z_t\|$", fontsize=13)
        ax.set_title(f"[cross-seed-deviation] [{group}] {name}", fontsize=12)
        ax.set_xlim(args.NFE, 0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        plt.tight_layout()

        plot_path = os.path.join(plot_dir, f"cross_seed_deviation_{group}_{name}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[cross-seed-dev] [{group}] {name}: Plot saved -> {plot_path}")


def analyze_cross_seed_twd_gap_deviation(source_gaps_dict, curve_base, args):
    """Per-source seed deviation analysis for tweedie gap: visualize how much each
    seed's gap deviates from prompt-mean of gap.

    Input:
      source_gaps_dict: {(group, name): raw_gaps} where raw_gaps shape=(T, n, B)
        raw_gaps[t, pi, si] = ||ε_ref_i - ε_s_{i,si}||² / D

    Computation:
      at each step t:
        gaps = raw_gaps[t]                        # (n, B)
        mean_per_prompt = gaps.mean(axis=1)       # (n, 1)
        deviation = |gaps - mean_per_prompt|      # (n, B)  absolute deviation
        avg_deviation_per_seed = deviation.mean(0) # (B,)   avg over prompts per seed
      result: (T, B) — T steps × B seeds

    Output:
      csv: step, time, seed_0, seed_1, ..., seed_{B-1}
      plot: B curve overlays (1 line per seed) — deviation from prompt-mean of gap
    """
    if not source_gaps_dict:
        return

    T = next(iter(source_gaps_dict.values())).shape[0]
    times = [args.NFE - s for s in range(T)]

    csv_dir = os.path.join(curve_base, "csv")
    plot_dir = os.path.join(curve_base, "plot")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # 소스별로 분석
    for (group, name), raw_gaps in source_gaps_dict.items():
        # raw_gaps: (T, n, B)
        _, n, B = raw_gaps.shape

        deviations = []  # list[T] of (B,)
        for t_idx in range(T):
            gaps = raw_gaps[t_idx]                              # (n, B)
            mean_per_prompt = gaps.mean(axis=1, keepdims=True)  # (n, 1)
            dev = np.abs(gaps - mean_per_prompt)                # (n, B) absolute deviation
            avg_dev_per_seed = dev.mean(axis=0)                 # (B,) avg over prompts
            deviations.append(avg_dev_per_seed)

        deviations_arr = np.stack(deviations)  # (T, B)

        # CSV 저장
        csv_path = os.path.join(csv_dir, f"cross_seed_twd_gap_deviation_{group}_{name}.csv")
        with open(csv_path, 'w', newline='') as f:
            header = ['step', 'time'] + [f'seed_{i}' for i in range(B)]
            writer = csv.writer(f)
            writer.writerow(header)
            for s in range(T):
                row = [s, times[s]] + [f"{deviations_arr[s, i]:.6f}" for i in range(B)]
                writer.writerow(row)
        print(f"[twd-gap-dev] [{group}] {name}: CSV saved -> {csv_path}")

        # Plot 저장
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i in range(B):
            ax.plot(times, deviations_arr[:, i], color=colors[i % len(colors)],
                    linewidth=2, marker='o', markersize=4, label=f'seed_{i}', alpha=0.8)

        ax.set_xlabel(r"Time ($t$)", fontsize=13)
        ax.set_ylabel(r"avg deviation from prompt-mean of $\|\varepsilon_{ref}-\varepsilon_s\|^2/D$",
                      fontsize=13)
        ax.set_title(f"[cross-seed-twd-gap-deviation] [{group}] {name}", fontsize=12)
        ax.set_xlim(args.NFE, 0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        plt.tight_layout()

        plot_path = os.path.join(plot_dir, f"cross_seed_twd_gap_deviation_{group}_{name}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[twd-gap-dev] [{group}] {name}: Plot saved -> {plot_path}")


def save_per_prompt_curves(pools, prompts, seeds, raw_gaps, group, name,
                           save_base, curve_base_latent_mag, curve_base_twd_gap, args):
    """Per-prompt 개별 plot + CSV 저장.

    latent_mag:
      plot: <curve_base_latent_mag>/plot/per_prompt/{group}/{name}/plot_{pi:02d}.png
      csv:  <curve_base_latent_mag>/csv/per_prompt/{group}/{name}/plot_{pi:02d}.csv
      내용: latent magnitude (B seeds) + twd_gap (B seeds, 있을 때)

    twd_gap (raw_gaps 있을 때만):
      csv:  <save_base>/{group}/{name}/record/twd_gap/per_prompt/plot_{pi:02d}.csv
      plot: <curve_base_twd_gap>/plot/per_prompt/{group}/{name}/plot_{pi:02d}.png

    CSV 공통 형식:
      행 1: prompt, <prompt text>
      행 2: (빈 행)
      행 3~: step, time, <data columns>

    Args:
        pools                : list[T] of tensor (n*B, 4, 64, 64) — batch_mean=False
        prompts              : list[str]
        seeds                : list[int]
        raw_gaps             : np.ndarray (T, n, B) or None
        group                : 'general' | 'memo'
        name                 : source name (e.g. 'coco_v2', 'webster')
        save_base            : workdir save root (save/{group}/{name} 상위)
        curve_base_latent_mag: curve_base/latent_mag 경로
        curve_base_twd_gap   : curve_base/twd_gap 경로
        args                 : Namespace (args.NFE 사용)
    """
    n = len(prompts)
    B = len(seeds)
    T = len(pools)

    has_twd = raw_gaps is not None  # (T, n, B)

    # latent_mag per_prompt 디렉토리
    lm_plot_dir = os.path.join(curve_base_latent_mag, "plot", "per_prompt", group, name)
    lm_csv_dir  = os.path.join(curve_base_latent_mag, "csv",  "per_prompt", group, name)
    os.makedirs(lm_plot_dir, exist_ok=True)
    os.makedirs(lm_csv_dir,  exist_ok=True)

    # twd_gap per_prompt 디렉토리 (twd 있을 때만)
    if has_twd:
        # CSV: save/{group}/{name}/record/twd_gap/per_prompt/
        twd_csv_dir  = os.path.join(save_base, group, name, "record", "twd_gap", "per_prompt")
        # Plot: curve/.../twd_gap/plot/per_prompt/{group}/{name}/
        twd_plot_dir = os.path.join(curve_base_twd_gap, "plot", "per_prompt", group, name)
        os.makedirs(twd_csv_dir,  exist_ok=True)
        os.makedirs(twd_plot_dir, exist_ok=True)

    # latent magnitude (T, n, B)
    mag_all = np.stack(
        [pl.float().reshape(n, B, -1).norm(dim=2).numpy() for pl in pools],
        axis=0
    )  # (T, n, B)

    times = [args.NFE - s for s in range(T)]

    def _prompt_header(w, prompt):
        w.writerow(['prompt', prompt])
        w.writerow([])

    for pi in range(n):
        prompt = prompts[pi]
        mag_pi = mag_all[:, pi, :]                         # (T, B)
        gap_pi = raw_gaps[:, pi, :] if has_twd else None  # (T, B)

        # ================================================================
        # 1) latent_mag CSV — mag + twd_gap (있을 때)
        # ================================================================
        with open(os.path.join(lm_csv_dir, f"plot_{pi:02d}.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            _prompt_header(w, prompt)
            header = ['step', 'time'] + [f'mag_seed_{si}' for si in range(B)]
            if has_twd:
                header += [f'twd_gap_seed_{si}' for si in range(B)]
            w.writerow(header)
            for s in range(T):
                row = [s, times[s]] + [f'{mag_pi[s, si]:.6f}' for si in range(B)]
                if has_twd:
                    row += [f'{gap_pi[s, si]:.6f}' for si in range(B)]
                w.writerow(row)

        # ================================================================
        # 2) latent_mag Plot — latent mag (상단) + twd_gap (하단, 있을 때)
        # ================================================================
        n_axes = 2 if has_twd else 1
        fig, axes = plt.subplots(n_axes, 1, figsize=(10, 5 * n_axes), squeeze=False)

        ax = axes[0, 0]
        for si in range(B):
            ax.plot(times, mag_pi[:, si], alpha=0.75, linewidth=1.5,
                    label=f'seed {seeds[si]}')
        ax.set_xlabel(r"Time ($t$)", fontsize=12)
        ax.set_ylabel(r"$\|z_t\|$", fontsize=12)
        ax.set_xlim(args.NFE, 0)
        ax.set_title(f"Latent Magnitude  [{group}] {name} — prompt {pi:02d}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.text(0.01, 0.97, prompt[:90], transform=ax.transAxes,
                fontsize=7, va='top', color='#555555',
                bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', alpha=0.5))

        if has_twd:
            ax2 = axes[1, 0]
            for si in range(B):
                ax2.plot(times, gap_pi[:, si], alpha=0.75, linewidth=1.5,
                         label=f'seed {seeds[si]}')
            ax2.set_xlabel(r"Time ($t$)", fontsize=12)
            ax2.set_ylabel(r"$\|x_T - \epsilon_s\|^2 / D$", fontsize=12)
            ax2.set_xlim(args.NFE, 0)
            ax2.set_title(f"Tweedie Gap  [{group}] {name} — prompt {pi:02d}", fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(lm_plot_dir, f"plot_{pi:02d}.png"), dpi=150)
        plt.close()

        if not has_twd:
            continue

        # ================================================================
        # 3) twd_gap CSV — save/{group}/{name}/record/twd_gap/per_prompt/
        # ================================================================
        with open(os.path.join(twd_csv_dir, f"plot_{pi:02d}.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            _prompt_header(w, prompt)
            header = ['step', 'time'] + [f'twd_gap_seed_{si}' for si in range(B)]
            w.writerow(header)
            for s in range(T):
                row = [s, times[s]] + [f'{gap_pi[s, si]:.6f}' for si in range(B)]
                w.writerow(row)

        # ================================================================
        # 4) twd_gap Plot — curve/.../twd_gap/plot/per_prompt/{group}/{name}/
        # ================================================================
        fig, ax = plt.subplots(figsize=(10, 5))
        for si in range(B):
            ax.plot(times, gap_pi[:, si], alpha=0.75, linewidth=1.5,
                    label=f'seed {seeds[si]}')
        ax.set_xlabel(r"Time ($t$)", fontsize=12)
        ax.set_ylabel(r"$\|x_T - \epsilon_s\|^2 / D$", fontsize=12)
        ax.set_xlim(args.NFE, 0)
        ax.set_title(f"Tweedie Gap  [{group}] {name} — prompt {pi:02d}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.text(0.01, 0.97, prompt[:90], transform=ax.transAxes,
                fontsize=7, va='top', color='#555555',
                bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow', alpha=0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(twd_plot_dir, f"plot_{pi:02d}.png"), dpi=150)
        plt.close()

    print(f"[per_prompt] [{group}] {name}: {n} prompts")
    print(f"  latent_mag plot → {lm_plot_dir}")
    print(f"  latent_mag csv  → {lm_csv_dir}")
    if has_twd:
        print(f"  twd_gap    csv  → {twd_csv_dir}")
        print(f"  twd_gap    plot → {twd_plot_dir}")


def incremental_rollout_seedstd(sd, prompts, seeds, cfg, device, latent_path,
                                  latents_dir, group, name, args):
    """Load existing latent cache, check shape, and generate missing prompts incrementally.

    If cache shape matches config: load and return.
    If cache shape is incomplete (n_cached < n_prompts): generate remaining prompts and append.

    Returns: (pools, n_prompts_generated, imgs, prompt_offset)
      pools: list[T] of tensor(n_prompts*batch, 4, 64, 64)
      n_prompts_generated: how many new prompts were generated (0 if all cached)
      imgs: generated images (None if all from cache)
      prompt_offset: starting prompt index for new imgs (for naming)
    """
    target_n = len(prompts)
    target_B = len(seeds)

    # Print the exact config path being used
    print(f"\n[seedstd] [{group}] {name}:")
    print(f"  Config path: batch={args.batch}, n_prompts={args.n_prompts}, seed={args.seed}")
    print(f"  Latent path: {latent_path}")

    # Try to load cache
    cached_x_t = None
    cached_n = 0
    cached_B = target_B
    T = None

    # force_rollout: skip cache entirely, regenerate all from scratch
    force_rollout = getattr(args, 'force_rollout', False)

    if force_rollout and os.path.exists(latent_path):
        print(f"  ⚡ FORCE ROLLOUT: 캐시 무시 — 강제로 새 rollout 수행")
        print(f"    (기존 latent.npz는 덮어씀)")
    elif os.path.exists(latent_path):
        # Load cache and check
        try:
            data = np.load(latent_path)
            cached_x_t = data["x_t"]  # (T, n_cached*B_cached, 4, 64, 64)
            cached_n = int(data["n_prompts"])
            cached_B = int(data["n_seeds"])
            T = cached_x_t.shape[0]

            print(f"  ✓ latent found at path")
            print(f"    cached shape:  (T={T}, n={cached_n}*B={cached_B}={cached_n*cached_B}, 4, 64, 64)")
            print(f"    target shape:  (T={T}, n={target_n}*B={target_B}={target_n*target_B}, 4, 64, 64)")
        except (OSError, ValueError, KeyError, Exception) as e:
            # Cache file corrupted or unreadable — treat as no cache
            print(f"  ✗ corrupted cache at path (error: {type(e).__name__})")
            print(f"    action: regenerate all prompts from scratch")
            cached_x_t = None
            cached_n = 0
            cached_B = target_B
            T = None

    # batch is fixed in path, so cached_B == target_B always
    # Only need to check n (number of prompts)

    if cached_x_t is not None and cached_n == target_n:
        # Perfect match
        print(f"  ✓ MATCH! cached n={cached_n}, target n={target_n} (B={cached_B} fixed by path)")
        print(f"    action: load cache as-is")
        pools = [torch.from_numpy(cached_x_t[t]) for t in range(T)]
        return pools, 0, None, 0

    elif cached_x_t is not None and cached_n < target_n:
        # Partial cache: not enough prompts
        # Continue from prompt cached_n onwards
        missing_count = target_n - cached_n
        print(f"  ℹ INCOMPLETE! cached n={cached_n}, target n={target_n}, need {missing_count} more")
        print(f"    batch fixed by path: B={cached_B}")
        print(f"    action: generate {missing_count} missing prompts and append to cache")

        # Generate only missing prompts
        missing_prompts = prompts[cached_n:]
        missing_pools, _, missing_imgs = rollout_group(sd, missing_prompts, seeds, cfg,
                                                        device, batch_mean=False)

        # Concatenate: cached + missing
        combined_x_t = []
        for t in range(T):
            cached_t = torch.from_numpy(cached_x_t[t])  # (cached_n*B, 4, 64, 64)
            missing_t = missing_pools[t]                 # (missing_count*B, 4, 64, 64)
            combined_t = torch.cat([cached_t, missing_t], dim=0)  # (target_n*B, 4, 64, 64)
            combined_x_t.append(combined_t)

        print(f"    combined: {cached_n*target_B} (cached) + {missing_count*target_B} (new) "
              f"= {target_n*target_B} total ✓")
        return combined_x_t, missing_count, missing_imgs, cached_n

    elif cached_x_t is not None and cached_n > target_n:
        # cached_n > target_n: cache has MORE prompts than needed
        excess_count = cached_n - target_n
        print(f"  ⚠ OVER-COMPLETE! cached n={cached_n}, target n={target_n}, excess={excess_count}")
        print(f"    batch fixed by path: B={cached_B}")
        print(f"    action: use first {target_n} prompts from cache, discard {excess_count} excess")

        # Slice: use only first target_n prompts
        sliced_x_t = []
        for t in range(T):
            cached_t = torch.from_numpy(cached_x_t[t])  # (cached_n*B, 4, 64, 64)
            # Keep only first target_n prompts worth of data
            sliced_t = cached_t[:target_n*target_B]    # (target_n*B, 4, 64, 64)
            sliced_x_t.append(sliced_t)

        print(f"    sliced: {target_n*target_B} from {cached_n*target_B} ✓")
        return sliced_x_t, 0, None, 0

    else:
        # No cache or corrupted cache — generate all from scratch
        print(f"  ✗ NO CACHE (or corrupted) at path")
        print(f"    target: shape=(T=?, n={target_n}*B={target_B}={target_n*target_B}, 4, 64, 64)")
        print(f"    action: generate all {target_n} prompts from scratch")
        pools, _, imgs = rollout_group(sd, prompts, seeds, cfg, device, batch_mean=False)
        return pools, target_n, imgs, 0


def validate_seedstd_outputs(save_base, curve_base_latent_mag, curve_base_twd_gap, sources, args):
    """Validate all expected outputs exist and match configuration for seedstd experiment.

    Checks:
      - latent.npz exists and shape=(T, n_prompts*batch, 4, 64, 64)
      - stored n_prompts and n_seeds match current config
      - results/ image count matches n_prompts*batch (if rollout done)
      - prompts.csv exists and has correct row count
      - curve/{csv,plot}/ exists with output files
    """
    print(f"\n[validating outputs...]")
    validation_ok = True
    seeds_count = args.batch

    # Check source-level outputs
    for group, name, _ in sources:
        source_path = os.path.join(save_base, group, name)
        latents_dir = os.path.join(source_path, "record", "latents")
        latent_path = os.path.join(latents_dir, "latent.npz")
        prompts_path = os.path.join(source_path, "prompts", "prompts.csv")
        results_dir = os.path.join(source_path, "results")

        print(f"  [{group}] {name}")
        source_ok = True

        # Check latent.npz existence and shape/metadata
        if os.path.exists(latent_path):
            try:
                data = np.load(latent_path)
                x_t = data["x_t"]  # shape: (T, n_prompts*batch, 4, 64, 64)
                stored_n_prompts = int(data["n_prompts"])
                stored_n_seeds = int(data["n_seeds"])

                # Verify shape matches config
                T, total_size, c, h, w = x_t.shape
                expected_total = args.n_prompts * seeds_count
                shape_match = (total_size == expected_total and c == 4 and h == 64 and w == 64)
                meta_match = (stored_n_prompts == args.n_prompts and stored_n_seeds == seeds_count)

                if shape_match and meta_match:
                    print(f"      ✓ latent.npz: shape={tuple(x_t.shape)}, "
                          f"n_prompts={stored_n_prompts}, n_seeds={stored_n_seeds} ✓")
                else:
                    print(f"      ✗ latent.npz shape/metadata mismatch!")
                    if not shape_match:
                        print(f"        Expected shape: (T, {expected_total}, 4, 64, 64), "
                              f"got: {tuple(x_t.shape)}")
                    if not meta_match:
                        print(f"        Expected n_prompts={args.n_prompts}, n_seeds={seeds_count}")
                        print(f"        Got n_prompts={stored_n_prompts}, n_seeds={stored_n_seeds}")
                    source_ok = False
            except Exception as e:
                print(f"      ✗ latent.npz load failed: {e}")
                source_ok = False
        else:
            print(f"      ✗ latent.npz not found")
            source_ok = False

        # Check prompts.csv
        if os.path.exists(prompts_path):
            try:
                with open(prompts_path, 'r') as f:
                    lines = f.readlines()
                # Header + n_prompts*batch data rows
                expected_rows = args.n_prompts * seeds_count + 1
                if len(lines) == expected_rows:
                    print(f"      ✓ prompts.csv: {len(lines)-1} rows (n_prompts={args.n_prompts} × batch={seeds_count})")
                else:
                    print(f"      ✗ prompts.csv row count mismatch: "
                          f"expected {expected_rows}, got {len(lines)}")
                    source_ok = False
            except Exception as e:
                print(f"      ✗ prompts.csv read failed: {e}")
                source_ok = False
        else:
            print(f"      ✗ prompts.csv not found")
            source_ok = False

        # Check results/ images (at least expected count, more is OK)
        if os.path.isdir(results_dir):
            img_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
            expected_imgs = args.n_prompts * seeds_count
            if len(img_files) >= expected_imgs:
                print(f"      ✓ results/: {len(img_files)} images (expected ≥{expected_imgs}) ✓")
            else:
                print(f"      ✗ results/ image count mismatch: "
                      f"expected ≥{expected_imgs}, got {len(img_files)}")
                source_ok = False
        else:
            print(f"      ℹ results/ not generated (cache mode)")

        if not source_ok:
            validation_ok = False

    # Check curve outputs — latent_mag metrics
    csv_dir_latent = os.path.join(curve_base_latent_mag, "csv")
    plot_dir_latent = os.path.join(curve_base_latent_mag, "plot")

    print(f"\n  Curve outputs (prompts={args.n_prompts}):")
    curve_checks = {
        "latent_mag/csv/seedstd_std.csv": os.path.exists(os.path.join(csv_dir_latent, "seedstd_std.csv")),
        "latent_mag/csv/seedstd_mean.csv": os.path.exists(os.path.join(csv_dir_latent, "seedstd_mean.csv")),
        "latent_mag/plot/seedstd_std.png": os.path.exists(os.path.join(plot_dir_latent, "seedstd_std.png")),
        "latent_mag/plot/seedstd_mean.png": os.path.exists(os.path.join(plot_dir_latent, "seedstd_mean.png")),
        "latent_mag/plot/cross_seed_deviation_*.png": len([f for f in os.listdir(plot_dir_latent)
                                                  if f.startswith("cross_seed_deviation_")]) > 0
                                           if os.path.isdir(plot_dir_latent) else False,
    }

    # Check twd_gap outputs
    csv_dir_twd = os.path.join(curve_base_twd_gap, "csv")
    plot_dir_twd = os.path.join(curve_base_twd_gap, "plot")

    twd_checks = {
        "twd_gap/csv/twd_gap_std.csv": os.path.exists(os.path.join(csv_dir_twd, "twd_gap_std.csv")),
        "twd_gap/csv/twd_gap_mean.csv": os.path.exists(os.path.join(csv_dir_twd, "twd_gap_mean.csv")),
        "twd_gap/plot/twd_gap_std.png": os.path.exists(os.path.join(plot_dir_twd, "twd_gap_std.png")),
        "twd_gap/plot/twd_gap_mean.png": os.path.exists(os.path.join(plot_dir_twd, "twd_gap_mean.png")),
        "twd_gap/plot/cross_seed_twd_gap_deviation_*.png": (
            len([f for f in os.listdir(plot_dir_twd)
                 if f.startswith("cross_seed_twd_gap_deviation_")]) > 0
            if os.path.isdir(plot_dir_twd) else False
        ),
    }

    curve_checks.update(twd_checks)

    for check, ok in curve_checks.items():
        marker = "  ✓" if ok else "  ✗"
        print(f"      {marker} {check}")
        if not ok:
            validation_ok = False

    if validation_ok:
        print(f"\n[✓ validation passed] All outputs valid with correct shapes/counts")
    else:
        print(f"\n[✗ validation failed] Shape or count mismatch — check above")

    return validation_ok


@torch.no_grad()
def compute_twd_gap_from_pools(sd, pools, prompts, seeds, cfg, timesteps, device,
                                target_step_ratio=0.5):
    """Compute Tweedie domain gap via the full Tweedie chain (matches eps_trajectory.py analyze_single).

    Chain per (prompt, seed, step t):
      1) ε_θ = ε_θ(z_t, t)                              — current step CFG-guided noise
      2) x̂_0|t = (z_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t             — Tweedie denoised estimate
      3) x_s = √ᾱ_s·x̂_0|t + √(1-ᾱ_s)·ε_original        — re-forward to FIXED target step s
                                                          (ε_original = z_T = pool[0], per-seed)
      4) ε_s = ε_θ(x_s, s)                              — CFG-guided noise at forwarded point
      5) gap = ‖ε_ref − ε_s‖² / D                        — ε_ref = fresh random noise (per seed)

    Aggregation:
      raw_gaps[t, pi, si] — per-(step, prompt, seed) gap
      std curve — std across seeds → mean across prompts
      mean curve — mean across seeds → mean across prompts

    Returns: {
        "std":       np.array[T],
        "mean":      np.array[T],
        "raw_gaps":  np.array[T, n, B],
    }
    """
    T = len(pools)
    n = len(prompts)
    B = len(seeds)
    D = 4 * 64 * 64

    twd_std_curve = []
    twd_mean_curve = []
    raw_gaps = np.zeros((T, n, B), dtype=np.float32)

    # Precompute text embeddings (batched per prompt)
    text_embeds = {}
    for pi, prompt in enumerate(prompts):
        uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)
        uc_b = uc.repeat(B, 1, 1).to(sd.dtype)
        c_b = c.repeat(B, 1, 1).to(sd.dtype)
        text_embeds[pi] = (uc_b, c_b)

    # Fixed target step s (eps_trajectory.py convention: target_step_ratio = 0.5 → middle)
    target_idx = int(T * target_step_ratio)
    target_idx = max(0, min(target_idx, T - 1))
    s = timesteps[target_idx]
    as_ = sd.alpha(s)
    s_val = s.item() if torch.is_tensor(s) else int(s)
    as_val = as_.item() if torch.is_tensor(as_) else float(as_)
    print(f"[twd_gap] tweedie chain: fixed target s = {s_val} (idx={target_idx}, "
          f"ᾱ_s = {as_val:.4f}, ratio={target_step_ratio})")

    # ε_original from pool[0] (= x_T, initial noise) — per (prompt, seed) in fp32
    z_T = pools[0].to(device=device, dtype=torch.float32).reshape(n, B, 4, 64, 64)

    # ε_ref = x_T (self-referential) — eps_trajectory.py와 동일 방식
    #   forward noise와 동일한 x_T를 reference로 사용 → late step에서 offset 없음

    with torch.no_grad():
        for t_idx, pool in enumerate(tqdm(pools, desc="[twd_gap] steps", unit="t")):
            z_t_all = pool.to(device=device, dtype=sd.dtype).reshape(n, B, 4, 64, 64)
            t = timesteps[t_idx]
            at = sd.alpha(t)

            prompt_gaps_std = []
            prompt_gaps_mean = []

            for pi in range(n):
                uc_b, c_b = text_embeds[pi]  # (B, 77, 768)
                z_b = z_t_all[pi]            # (B, 4, 64, 64) fp16

                # 1) ε_θ(z_t, t) — CFG-guided (fp32 for numerical stability)
                noise_uc, noise_c = sd.predict_noise(z_b, t, uc_b, c_b)
                eps_theta = noise_uc.float() + cfg * (noise_c.float() - noise_uc.float())

                # 2) Tweedie x̂_0|t = (z_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t
                z_b_f = z_b.float()
                x0_hat = (z_b_f - (1 - at).sqrt() * eps_theta) / at.sqrt()  # (B, 4, 64, 64)

                # 3) Forward to FIXED step s using ε_original (=z_T[pi], per-seed)
                eps_original = z_T[pi]  # (B, 4, 64, 64) fp32
                x_s = as_.sqrt() * x0_hat + (1 - as_).sqrt() * eps_original

                # 4) ε_s = ε_θ(x_s, s) — CFG-guided
                noise_uc_s, noise_c_s = sd.predict_noise(x_s.to(sd.dtype), s, uc_b, c_b)
                eps_s = (noise_uc_s.float()
                         + cfg * (noise_c_s.float() - noise_uc_s.float()))

                # 5) gap = ‖ε_ref − ε_s‖² / D  (ε_ref = x_T, self-referential)
                eps_ref = eps_original  # (B, 4, 64, 64) = z_T[pi]
                diff = eps_ref - eps_s
                distances = diff.pow(2).reshape(B, -1).sum(dim=1) / D  # (B,)

                distances_np = distances.cpu().numpy()
                raw_gaps[t_idx, pi, :] = distances_np
                prompt_gaps_std.append(float(np.std(distances_np)))
                prompt_gaps_mean.append(float(np.mean(distances_np)))

            twd_std_curve.append(float(np.mean(prompt_gaps_std)))
            twd_mean_curve.append(float(np.mean(prompt_gaps_mean)))

    return {
        "std": np.array(twd_std_curve),
        "mean": np.array(twd_mean_curve),
        "raw_gaps": raw_gaps,
    }


# ===================================================================
#  twd_gap CSV caching (incremental, per-source)
# ===================================================================

def load_twd_gap_csv(csv_path, T, B):
    """Load twd_gap raw CSV. Returns (raw_gaps (T, n_cached, B), n_cached) or (None, 0).

    CSV format:
      header: prompt_idx, seed_idx, step_0, step_1, ..., step_{T-1}
      rows:   n_cached * B rows (one per (prompt, seed) combination)
    """
    if not os.path.exists(csv_path):
        return None, 0
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return None, 0
        header = rows[0]
        data_rows = rows[1:]
        step_cols = header[2:]
        T_cached = len(step_cols)
        if T_cached != T:
            print(f"[twd_gap cache] T mismatch: cached={T_cached}, target={T} → invalidate")
            return None, 0
        if len(data_rows) % B != 0:
            print(f"[twd_gap cache] row count {len(data_rows)} not divisible by B={B} → invalidate")
            return None, 0
        n_cached = len(data_rows) // B

        raw = np.zeros((n_cached, B, T), dtype=np.float32)
        for row in data_rows:
            pi = int(row[0]); si = int(row[1])
            if pi >= n_cached or si >= B:
                continue
            for t in range(T):
                raw[pi, si, t] = float(row[2 + t])
        raw_gaps = raw.transpose(2, 0, 1)  # (T, n_cached, B)
        return raw_gaps, n_cached
    except Exception as e:
        print(f"[twd_gap cache] load failed ({type(e).__name__}: {e})")
        return None, 0


def save_twd_gap_csv(csv_path, raw_gaps):
    """Save twd_gap raw values as CSV.

    raw_gaps: (T, n, B) np.ndarray
    CSV: header=[prompt_idx, seed_idx, step_0, ..., step_{T-1}], n*B rows.
    """
    T, n, B = raw_gaps.shape
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['prompt_idx', 'seed_idx'] + [f'step_{t}' for t in range(T)]
        writer.writerow(header)
        for pi in range(n):
            for si in range(B):
                row = [pi, si] + [f"{float(raw_gaps[t, pi, si]):.6f}" for t in range(T)]
                writer.writerow(row)


def incremental_twd_gap_compute(sd, pools, prompts, seeds, cfg, timesteps, device,
                                 twd_gap_csv_path, target_step_ratio=0.5,
                                 cached_raw=None, cached_n=0):
    """Incremental twd_gap computation with CSV caching.

    - cached_n < target_n: compute missing prompts only, append to cache
    - no cache: compute all
    (case cached_n >= target_n handled by caller — this function is only called when compute needed)

    Returns:
      raw_gaps: (T, target_n, B)
      n_new: number of newly computed prompts
    """
    target_n = len(prompts)
    B = len(seeds)
    T = len(pools)

    if sd is None:
        raise RuntimeError("incremental_twd_gap_compute: sd is None but computation required")

    if cached_raw is not None and cached_n < target_n:
        # Incremental: compute missing prompts only
        n_missing = target_n - cached_n
        print(f"  ℹ INCREMENTAL: cached_n={cached_n}, target_n={target_n}, "
              f"compute {n_missing} missing prompts")

        # Slice pools for missing prompts (pool layout: prompt-major)
        missing_prompts = prompts[cached_n:]
        missing_pools = []
        for pl in pools:
            pl_reshaped = pl.reshape(target_n, B, 4, 64, 64)
            missing_pools.append(pl_reshaped[cached_n:].reshape(-1, 4, 64, 64))

        new_metrics = compute_twd_gap_from_pools(
            sd, missing_pools, missing_prompts, seeds, cfg, timesteps, device,
            target_step_ratio=target_step_ratio)
        new_raw = new_metrics["raw_gaps"]  # (T, n_missing, B)

        # Concatenate along prompt axis
        raw_gaps = np.concatenate([cached_raw, new_raw], axis=1)  # (T, target_n, B)
        n_new = n_missing
    else:
        # No cache — compute all
        print(f"  ✗ NO CACHE: compute all {target_n} prompts")
        new_metrics = compute_twd_gap_from_pools(
            sd, pools, prompts, seeds, cfg, timesteps, device,
            target_step_ratio=target_step_ratio)
        raw_gaps = new_metrics["raw_gaps"]  # (T, target_n, B)
        n_new = target_n

    # Save updated CSV
    save_twd_gap_csv(twd_gap_csv_path, raw_gaps)
    print(f"  saved -> {twd_gap_csv_path}  (T={T}, n={target_n}, B={B})")

    return raw_gaps, n_new


def run_seedstd(sd, args, device):
    """Per-source: rollout n_prompts prompts × seed(batch) -> compute per-prompt step-wise
    **seed-axis** std(‖z_t‖) -> average across **prompt axis** -> overlay curves for all
    general·memo sources in one plot.
    (aka "latent std per text prompt -> prompt average" — experiment 4)

    Save structure:
      workdir/exp_main/manifold/latent_std/{model_key}/{sampler}/{cfg_tag}/{batch_tag}/{seed_tag}/
      ├── save/{group}/{name}/
      │   ├── results/           (img_XXXX_YY.png: prompt_idx_seed_idx)
      │   ├── record/latents/    (latent.npz)
      │   └── prompts/           (prompts.csv)
      └── curve/prompts={n_prompts}/
          ├── csv/   (std_curve.csv)
          └── plot/  (std_curve.png, ...)

    Skip rollout if cache exists — can call with sd=None (main skips model load)."""
    sources = _parse_sources(args.sources)
    seeds = seed_list(args.seed, args.batch)

    model_key_basename = os.path.basename(args.model_key.rstrip("/"))
    sampler_tag = "ddim"
    cfg_tag = f"CFG={args.cfg}_NFE={args.NFE}"
    batch_tag = f"batch={args.batch}"
    seed_tag = f"seed={args.seed}"

    base_path = os.path.join(
        "workdir", "exp_main", "manifold", "latent_std",
        model_key_basename, sampler_tag, cfg_tag, batch_tag)
    seed_path = os.path.join(base_path, seed_tag)
    save_base = os.path.join(seed_path, "save")
    curve_base_parent = os.path.join(seed_path, "curve", f"prompts={args.n_prompts}")
    curve_base_latent_mag = os.path.join(curve_base_parent, "latent_mag")  # latent magnitude metrics
    curve_base_twd_gap = os.path.join(curve_base_parent, "twd_gap")       # tweedie gap metrics

    curves = {}   # (group, name) -> {"std": np.ndarray[T], "mean": np.ndarray[T]}
    for group, name, path in sources:
        source_path = os.path.join(save_base, group, name)
        results_dir = os.path.join(source_path, "results")
        latents_dir = os.path.join(source_path, "record", "latents")
        latent_path = os.path.join(latents_dir, "latent.npz")

        prompts = load_prompts(path, args.n_prompts)
        n = len(prompts)
        B = len(seeds)

        # Try incremental rollout: load cache + generate missing prompts
        pools, n_new, imgs, prompt_offset = incremental_rollout_seedstd(
            sd, prompts, seeds, args.cfg, device, latent_path,
            latents_dir, group, name, args)

        # ── Pretty-print returned values ─────────────────────────────────
        _pool_shape = tuple(pools[0].shape) if len(pools) > 0 else None
        _pool_dtype = str(pools[0].dtype).replace("torch.", "") if len(pools) > 0 else "n/a"
        _pool_device = str(pools[0].device) if len(pools) > 0 else "n/a"
        if imgs is None:
            _imgs_info = "None (no new images — cache hit or over-complete)"
        else:
            _imgs_info = (f"Tensor shape={tuple(imgs.shape)}, "
                          f"dtype={str(imgs.dtype).replace('torch.', '')}, "
                          f"device={imgs.device}, "
                          f"range=[{float(imgs.min()):.3f}, {float(imgs.max()):.3f}]")
        if n_new == 0:
            _scenario = "Full cache hit — rollout skipped"
        elif n_new == len(prompts):
            _scenario = "No cache — full rollout"
        else:
            _scenario = (f"Partial cache — {len(prompts) - n_new} cached + "
                         f"{n_new} newly generated")

        print(f"  ┌─ incremental_rollout_seedstd() returned ─────────────────────")
        print(f"  │ pools          : list[T={len(pools)}] of Tensor{_pool_shape}  "
              f"[{_pool_dtype}, {_pool_device}]")
        print(f"  │ n_new          : {n_new}  (# newly rolled-out prompts)")
        print(f"  │ imgs           : {_imgs_info}")
        print(f"  │ prompt_offset  : {prompt_offset}  "
              f"(new imgs start from img_{prompt_offset:04d}_YY.png)")
        print(f"  │ scenario       : {_scenario}")
        print(f"  └──────────────────────────────────────────────────────────────")

        # Save latent (update or initial save)
        os.makedirs(latents_dir, exist_ok=True)
        pools_arr = np.stack([p.to(torch.float16).numpy() for p in pools])
        np.savez_compressed(latent_path, x_t=pools_arr,
                            n_prompts=np.array(n), n_seeds=np.array(B))

        # Verify saved latent shape matches config
        T_actual = pools_arr.shape[0]
        n_actual = n
        B_actual = B
        expected_shape = (T_actual, n * B, 4, 64, 64)
        actual_shape = pools_arr.shape
        shape_match = (actual_shape == expected_shape)

        if n_new > 0:
            print(f"[seedstd] latent updated -> {latent_path}")
            print(f"        saved: shape={actual_shape}, n={n_actual}, B={B_actual}")
        else:
            print(f"[seedstd] latent confirmed -> {latent_path}")
            print(f"        saved: shape={actual_shape}, n={n_actual}, B={B_actual}")

        if not shape_match:
            print(f"        ✗ SHAPE MISMATCH!")
            print(f"          expected: {expected_shape}")
            print(f"          actual:   {actual_shape}")
            raise RuntimeError(f"Saved latent shape mismatch for [{group}] {name}")
        else:
            print(f"        ✓ shape verified")

        # save prompts CSV (always save, consolidate full list)
        prompts_dir = os.path.join(source_path, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        csv_path = os.path.join(prompts_dir, "prompts.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['img_name', 'text_prompt'])
            for pi, prompt in enumerate(prompts):
                for si in range(B):
                    img_name = f"img_{pi:04d}_{si:02d}.png"
                    writer.writerow([img_name, prompt])
        print(f"[seedstd] prompts CSV saved -> {csv_path} ({len(prompts)} prompts × {B} seeds)")

        # save images: img_XXXX_YY.png (only for newly generated, append mode)
        if imgs is not None and n_new > 0:
            os.makedirs(results_dir, exist_ok=True)
            from torchvision.utils import save_image
            saved_count = 0
            for flat_idx in range(len(imgs)):
                pi = flat_idx // B + prompt_offset  # prompt index (offset for incremental)
                si = flat_idx % B                   # seed index
                fname = os.path.join(results_dir, f"img_{pi:04d}_{si:02d}.png")
                save_image(imgs[flat_idx], fname)
                saved_count += 1
            print(f"[seedstd] images saved -> {results_dir}/ ({saved_count} images, prompts {prompt_offset}-{prompt_offset+n_new-1})")

        # pools: list[T] of tensor (n*B, 4, 64, 64) — [prompt0_seed0..B-1, prompt1_seed0..B-1, ...]
        # Compute two metrics:
        # 1. std: std of magnitude per prompt, then average across prompts
        # 2. mean: mean of magnitude per prompt, then average across prompts
        step_curve_std = []
        step_curve_mean = []
        for pl in pools:
            mag = pl.float().reshape(n, B, -1).norm(dim=2)            # (n, B)
            # Metric 1: std across seeds
            std_per_prompt = mag.std(dim=1, unbiased=False)            # (n,)
            step_curve_std.append(std_per_prompt.mean().item())        # mean across prompts
            # Metric 2: mean across seeds
            mean_per_prompt = mag.mean(dim=1)                          # (n,)
            step_curve_mean.append(mean_per_prompt.mean().item())      # mean across prompts
        curves[(group, name)] = {
            "std": np.array(step_curve_std),
            "mean": np.array(step_curve_mean)
        }

    # save curve (csv + plot) — two metrics: std and mean
    T = len(next(iter(curves.values()))["std"])
    times = [args.NFE - s for s in range(T)]

    csv_dir = os.path.join(curve_base_latent_mag, "csv")
    plot_dir = os.path.join(curve_base_latent_mag, "plot")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # ---- Metric 1: STD ----
    # CSV: step, time, + all source std values
    csv_path_std = os.path.join(csv_dir, "seedstd_std.csv")
    with open(csv_path_std, 'w', newline='') as f:
        # Header: step, time, source1, source2, ...
        header = ['step', 'time'] + [f"{g}({n})" for g, n in curves.keys()]
        writer = csv.writer(f)
        writer.writerow(header)
        # Data rows
        for s in range(T):
            row = [s, times[s]] + [f"{curves[k]['std'][s]:.6f}" for k in curves.keys()]
            writer.writerow(row)
    print(f"[seedstd] std CSV saved -> {csv_path_std}")

    # Plot: all sources overlay (std metric)
    fig, ax = plt.subplots(figsize=(9, 6))
    color_idx = {}
    for (g, n), metrics in curves.items():
        palette = _GROUP_COLORS.get(g, ["tab:gray", "tab:olive"])
        i = color_idx.get(g, 0); color_idx[g] = i + 1
        marker = "o-" if g == "general" else "s-"
        ax.plot(times, metrics["std"], marker, color=palette[i % len(palette)],
                linewidth=2, markersize=4, label=f"[{g}] {n}")
    ax.set_xlabel(r"Time ($t$)", fontsize=13)
    ax.set_ylabel(r"mean over prompts of std($\|z_t\|$) across seeds", fontsize=13)
    ax.set_title(f"[seedstd] Seed Std by Source", fontsize=12)
    ax.set_xlim(args.NFE, 0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "seedstd_std.png"), dpi=150)
    plt.close()
    print(f"[seedstd] std plot saved -> {os.path.join(plot_dir, 'seedstd_std.png')}")

    # ---- Metric 2: MEAN ----
    # CSV: step, time, + all source mean values
    csv_path_mean = os.path.join(csv_dir, "seedstd_mean.csv")
    with open(csv_path_mean, 'w', newline='') as f:
        # Header: step, time, source1, source2, ...
        header = ['step', 'time'] + [f"{g}({n})" for g, n in curves.keys()]
        writer = csv.writer(f)
        writer.writerow(header)
        # Data rows
        for s in range(T):
            row = [s, times[s]] + [f"{curves[k]['mean'][s]:.6f}" for k in curves.keys()]
            writer.writerow(row)
    print(f"[seedstd] mean CSV saved -> {csv_path_mean}")

    # Plot: all sources overlay (mean metric)
    fig, ax = plt.subplots(figsize=(9, 6))
    color_idx = {}
    for (g, n), metrics in curves.items():
        palette = _GROUP_COLORS.get(g, ["tab:gray", "tab:olive"])
        i = color_idx.get(g, 0); color_idx[g] = i + 1
        marker = "o-" if g == "general" else "s-"
        ax.plot(times, metrics["mean"], marker, color=palette[i % len(palette)],
                linewidth=2, markersize=4, label=f"[{g}] {n}")
    ax.set_xlabel(r"Time ($t$)", fontsize=13)
    ax.set_ylabel(r"mean over prompts of mean($\|z_t\|$) across seeds", fontsize=13)
    ax.set_title(f"[seedstd] Mean Magnitude by Source", fontsize=12)
    ax.set_xlim(args.NFE, 0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "seedstd_mean.png"), dpi=150)
    plt.close()
    print(f"[seedstd] mean plot saved -> {os.path.join(plot_dir, 'seedstd_mean.png')}")

    # cross-seed deviation analysis (how much each seed deviates from mean)
    print(f"\n[analyzing cross-seed-deviation...]")
    source_pools_dict = {}  # {(group, name): pools}
    for group, name, path in sources:
        source_path = os.path.join(save_base, group, name)
        latents_dir = os.path.join(source_path, "record", "latents")
        latent_path = os.path.join(latents_dir, "latent.npz")
        if os.path.exists(latent_path):
            data = np.load(latent_path)
            pools_arr = data["x_t"]  # (T, n*B, 4, 64, 64)
            pools = [torch.from_numpy(pools_arr[t]) for t in range(len(pools_arr))]
            source_pools_dict[(group, name)] = pools

    if source_pools_dict:
        analyze_cross_seed_deviation(source_pools_dict, curve_base_latent_mag, args)
    print(f"[Done] cross-seed-deviation")

    # ---- Tweedie Gap Computation (with CSV caching) ----
    print(f"\n[computing tweedie domain gap...]")
    twd_curves = {}  # {(group, name): {"std": ..., "mean": ..., "raw_gaps": ...}}
    T_target = args.NFE
    timesteps = list(sd.scheduler.timesteps) if sd is not None else None

    for group, name, path in sources:
        source_path = os.path.join(save_base, group, name)
        latents_dir = os.path.join(source_path, "record", "latents")
        latent_path = os.path.join(latents_dir, "latent.npz")
        twd_gap_dir = os.path.join(source_path, "record", "twd_gap")
        twd_gap_csv_path = os.path.join(twd_gap_dir, "twd_gap_raw.csv")

        prompts = load_prompts(path, args.n_prompts)
        target_n = len(prompts)
        B = args.batch

        # 1) Try cached CSV first
        cached_raw, cached_n = load_twd_gap_csv(twd_gap_csv_path, T_target, B)

        if cached_raw is not None and cached_n >= target_n: # CSV가 있다
            # Cache sufficient — no computation needed
            print(f"  [{group}] {name}: ✓ CACHE HIT (n_cached={cached_n} >= target_n={target_n}) "
                  f"— read CSV only")
            raw_gaps = cached_raw[:, :target_n, :]  # (T, target_n, B)
        elif sd is not None and os.path.exists(latent_path):
            # Load pools and compute (full or incremental)
            print(f"  [{group}] {name}: cache incomplete or missing — compute via Tweedie chain")
            data = np.load(latent_path)
            pools_arr = data["x_t"]  # (T, n*B, 4, 64, 64)
            pools = [torch.from_numpy(pools_arr[t]) for t in range(len(pools_arr))]

            raw_gaps, n_new = incremental_twd_gap_compute(
                sd, pools, prompts, seeds, args.cfg, timesteps, device,
                twd_gap_csv_path, target_step_ratio=args.target_step_ratio,
                cached_raw=cached_raw, cached_n=cached_n)
        else:
            # No cache and no model — skip this source
            print(f"  [{group}] {name}: ✗ SKIP (no cache and no model available)")
            continue

        # Derive std/mean curves from raw_gaps
        twd_curves[(group, name)] = {
            "std":       raw_gaps.std(axis=2).mean(axis=1),   # (T,)
            "mean":      raw_gaps.mean(axis=2).mean(axis=1),  # (T,)
            "raw_gaps":  raw_gaps,
        }
        print(f"  [{group}] {name}: tweedie gap ready (n={target_n}, B={B}, T={T_target})")

    # Save curves (csv + plot) — from twd_curves derived above (cache or fresh)
    if True:  # scope wrapper for the else branch below
        if twd_curves:
            csv_dir_twd = os.path.join(curve_base_twd_gap, "csv")
            plot_dir_twd = os.path.join(curve_base_twd_gap, "plot")
            os.makedirs(csv_dir_twd, exist_ok=True)
            os.makedirs(plot_dir_twd, exist_ok=True)

            # ---- Metric 1: TWD STD ----
            csv_path_twd_std = os.path.join(csv_dir_twd, "twd_gap_std.csv")
            with open(csv_path_twd_std, 'w', newline='') as f:
                header = ['step', 'time'] + [f"{g}({n})" for g, n in twd_curves.keys()]
                writer = csv.writer(f)
                writer.writerow(header)
                for s in range(T):
                    row = [s, times[s]] + [f"{twd_curves[k]['std'][s]:.6f}" for k in twd_curves.keys()]
                    writer.writerow(row)
            print(f"[seedstd] twd_gap_std.csv saved -> {csv_path_twd_std}")

            fig, ax = plt.subplots(figsize=(9, 6))
            color_idx = {}
            for (g, n), metrics in twd_curves.items():
                palette = _GROUP_COLORS.get(g, ["tab:gray", "tab:olive"])
                i = color_idx.get(g, 0); color_idx[g] = i + 1
                marker = "o-" if g == "general" else "s-"
                ax.plot(times, metrics["std"], marker, color=palette[i % len(palette)],
                        linewidth=2, markersize=4, label=f"[{g}] {n}")
            ax.set_xlabel(r"Time ($t$)", fontsize=13)
            ax.set_ylabel(r"Tweedie domain gap (std)", fontsize=13)
            ax.set_title(f"[seedstd] Tweedie Gap - Std by Source", fontsize=12)
            ax.set_xlim(args.NFE, 0)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir_twd, "twd_gap_std.png"), dpi=150)
            plt.close()
            print(f"[seedstd] twd_gap_std.png saved")

            # ---- Metric 2: TWD MEAN ----
            csv_path_twd_mean = os.path.join(csv_dir_twd, "twd_gap_mean.csv")
            with open(csv_path_twd_mean, 'w', newline='') as f:
                header = ['step', 'time'] + [f"{g}({n})" for g, n in twd_curves.keys()]
                writer = csv.writer(f)
                writer.writerow(header)
                for s in range(T):
                    row = [s, times[s]] + [f"{twd_curves[k]['mean'][s]:.6f}" for k in twd_curves.keys()]
                    writer.writerow(row)
            print(f"[seedstd] twd_gap_mean.csv saved -> {csv_path_twd_mean}")

            fig, ax = plt.subplots(figsize=(9, 6))
            color_idx = {}
            for (g, n), metrics in twd_curves.items():
                palette = _GROUP_COLORS.get(g, ["tab:gray", "tab:olive"])
                i = color_idx.get(g, 0); color_idx[g] = i + 1
                marker = "o-" if g == "general" else "s-"
                ax.plot(times, metrics["mean"], marker, color=palette[i % len(palette)],
                        linewidth=2, markersize=4, label=f"[{g}] {n}")
            ax.set_xlabel(r"Time ($t$)", fontsize=13)
            ax.set_ylabel(r"Tweedie domain gap (mean)", fontsize=13)
            ax.set_title(f"[seedstd] Tweedie Gap - Mean by Source", fontsize=12)
            ax.set_xlim(args.NFE, 0)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir_twd, "twd_gap_mean.png"), dpi=150)
            plt.close()
            print(f"[seedstd] twd_gap_mean.png saved")

            # ---- Cross-seed twd_gap deviation analysis ----
            print(f"\n[analyzing cross-seed twd_gap deviation...]")
            source_gaps_dict = {k: m["raw_gaps"] for k, m in twd_curves.items()}
            analyze_cross_seed_twd_gap_deviation(source_gaps_dict, curve_base_twd_gap, args)
            print(f"[Done] cross-seed twd_gap deviation")
        else:
            print("  [skip] no twd_gap data (cache missing and no model)")
    print(f"[Done] tweedie gap")

    # ---- Per-prompt curves (latent_mag + twd_gap per seed) ----
    print(f"\n[saving per-prompt curves...]")
    for group, name, path in sources:
        source_path = os.path.join(save_base, group, name)
        latent_path = os.path.join(source_path, "record", "latents", "latent.npz")
        if not os.path.exists(latent_path):
            print(f"  [{group}] {name}: latent.npz not found, skip per-prompt")
            continue
        prompts_src = load_prompts(path, args.n_prompts)
        n_src = len(prompts_src)
        data = np.load(latent_path)
        pools_arr = data["x_t"]            # (T, n*B, 4, 64, 64)
        pools_src = [torch.from_numpy(pools_arr[t]) for t in range(len(pools_arr))]
        raw_gaps_src = twd_curves.get((group, name), {}).get("raw_gaps", None)
        if raw_gaps_src is not None:
            raw_gaps_src = raw_gaps_src[:, :n_src, :]   # (T, n_src, B)
        save_per_prompt_curves(pools_src, prompts_src, seeds, raw_gaps_src,
                               group, name, save_base,
                               curve_base_latent_mag, curve_base_twd_gap, args)
    print(f"[Done] per-prompt curves")

    # print save info per source
    for group, name, path in sources:
        source_path = os.path.join(save_base, group, name)
        print(f"[seedstd] [{group}] {name}:")
        print(f"    save: {source_path}/")

    # validate all outputs exist
    validate_seedstd_outputs(save_base, curve_base_latent_mag, curve_base_twd_gap, sources, args)

    print(f"\n[Done] seedstd({args.seedstd})")


# ===================================================================
#  Main
# ===================================================================

def main():
    p = argparse.ArgumentParser(description="manifold-dist 실험 (off-manifold / fixation)")
    # 공통
    p.add_argument("--model_key", type=str, default=os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-4"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--NFE", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=5, help="num seeds per prompt")
    p.add_argument("--ref_prompt", type=str,
                   default=os.path.join(SCRIPT_DIR, "examples", "assets", "coco_v2.txt"),
                   help="실험 1 — on-manifold reference DB prompt (기준 풀, mscoco)")
    p.add_argument("--ref_images", type=str, default=None,
                   help="on-manifold ref를 원본 이미지 forward noising으로 정의 — ref DB 교체 실험: "
                        "save/ref/{src}/{subset}/results 형태 권장 (예: .../save/ref/laion_Aes_v2/512x512/results). "
                        "결과는 off_manifold_exp/{src}/ 아래 저장")
    p.add_argument("--general_prompt", type=str,
                   default=os.path.join(SCRIPT_DIR, "examples", "assets", "geneval_prompts.txt"),
                   help="실험 1 — 측정 대상 general 풀 (ref과 별개 소스)")
    p.add_argument("--memo_prompt", type=str,
                   default=os.path.join(SCRIPT_DIR, "examples", "assets", "cvpr2025_memo_prompt.txt"))
    p.add_argument("--out_root", type=str,
                   default=os.path.join(SCRIPT_DIR, "workdir", "exp_main", "manifold"))
    p.add_argument("--n_prompts_max", type=int, default=25,
                   help="fixation 에서 불러둘 prompt 상한 (np_list 최대보다 크게)")
    # 실험 선택
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifold", type=str, choices=list(DIST.keys()),
                   help="실험 1 — off-manifold 거리 방법")
    g.add_argument("--fixation", type=str, choices=["seed", "prompt"],
                   help="실험 2 — fixation (std = ‖z_t‖ std)")
    g.add_argument("--trajstd", type=str,
                   help="실험 3 — general·memo 소스 간 population std(‖z_t‖) 곡선을 한 plot에 "
                        "동시 오버레이 (population 축 = 소스 내 서로 다른 prompt). 값은 출력 "
                        "폴더 태그(예: general_memo). --sources \"group:name=path\" 필수 "
                        "(group=general|memo)")
    g.add_argument("--seedstd", type=str,
                   help="실험 4 — 소스별 프롬프트마다 seed 축 std(‖z_t‖)를 구한 뒤 prompt 축으로 "
                        "평균한 곡선을 general·memo 전체에 대해 한 plot에 동시 오버레이 "
                        "(한 prompt에 따른 latent std → prompt 평균). 값은 출력 폴더 태그. "
                        "--sources \"group:name=path\" 필수 (group=general|memo)")
    p.add_argument("--compare_all", action="store_true",
                   help="save/의 모든 general·memo latent를 한 plot에 비교")
    # 실험 1 전용
    p.add_argument("--n_prompts", type=int, default=100)
    p.add_argument("--t_idx", type=str, default=None,
                   help='측정할 step 인덱스 — 공백/콤마 구분 (예: "0 10 20 30 40 49"). '
                        '미지정 시 전체 T step 측정. rollout은 항상 전체 진행')
    p.add_argument("--knn_k", type=int, default=5)
    p.add_argument("--skip_fid", action="store_true",
                   help="최종 이미지 FID 계산 생략 (rollout만)")
    # 실험 2 전용
    p.add_argument("--seed_list", type=str, default="5 10 50 100",
                   help="2-1 selected seed list")
    p.add_argument("--np_list", type=str, default="1 5 10 20",
                   help="2-2 num prompt list")
    p.add_argument("--fix_general_idx", type=int, default=0)
    p.add_argument("--fix_memo_idx", type=int, default=0)
    # 실험 3 전용
    p.add_argument("--sources", type=str, nargs="+", default=None,
                   help='실험 3 전용 — "group:name=path" 형식 소스 리스트 (group=general|memo) '
                        '(예: general:gen_eval=examples/assets/gen_eval.txt '
                        'memo:webster=examples/assets/sdv1_500_mem.txt ...)')
    # 실험 4 전용 (seedstd) — cache 강제 무시
    p.add_argument("--force_rollout", action="store_true",
                   help="캐시된 latent.npz가 있어도 무시하고 강제로 새로 rollout 수행 "
                        "(기존 latent.npz는 덮어씀). seedstd 전용.")
    # 실험 4 twd_gap — Tweedie chain의 fixed target step 비율
    p.add_argument("--target_step_ratio", type=float, default=0.5,
                   help="Tweedie gap 계산 시 x̂_0|t를 forward할 fixed target step s의 "
                        "비율 (0=x_T, 1=x_0). 기본 0.5 (mid-noise level). "
                        "eps_trajectory.py analyze_single 방식.")
    args = p.parse_args()
    if (args.trajstd or args.seedstd) and not args.sources:
        raise SystemExit("[error] --sources required when using --trajstd/--seedstd")
    device = torch.device(args.device)

    # ---- save-load 모드 감지: latent.npz 3개 존재 시 모델 로드 생략 ----
    _mt = os.path.basename(args.model_key.rstrip("/"))
    _st = "ddim"  # sampler 태그 (save에 저장된 이름과 정합 필요)
    _base = os.path.join(args.out_root, _mt, _st,
                         f"CFG={args.cfg}_NFE={args.NFE}", f"seed={args.seed}",
                         f"size=512x512", f"batch={args.batch}")
    _tags = {"ref": _ref_save_tag(args.ref_images) if args.ref_images
             else _make_tag(args.ref_prompt),
             "general": _make_tag(args.general_prompt),
             "memo": _make_tag(args.memo_prompt)}
    _has_save = all(os.path.exists(
        os.path.join(_base, "save", g, _tags[g], "record", "latent", "latent.npz")
    ) for g in ("ref", "general", "memo"))

    _seedstd_all_pools = False
    if args.seedstd and args.sources and not args.force_rollout:
        _sampler = "ddim"
        _cfg_tag = f"CFG={args.cfg}_NFE={args.NFE}"
        _batch_tag = f"batch={args.batch}"
        _seed_tag = f"seed={args.seed}"
        _base_seed = os.path.join("workdir", "exp_main", "manifold", "latent_std",
                                  _mt, _sampler, _cfg_tag, _batch_tag, _seed_tag, "save")
        # Check if ALL pools exist AND have correct shape (n_prompts and batch)
        _seedstd_all_pools = True
        for g, n, _ in _parse_sources(args.sources):
            latent_file = os.path.join(_base_seed, g, n, "record", "latents", "latent.npz")
            if not os.path.exists(latent_file):
                _seedstd_all_pools = False
                break
            # Load metadata to check shape correctness
            try:
                data = np.load(latent_file)
                cached_n = int(data["n_prompts"])
                cached_B = int(data["n_seeds"])
                # Check if cached shape matches current config
                if cached_n != args.n_prompts or cached_B != args.batch:
                    _seedstd_all_pools = False
                    break
            except Exception:
                _seedstd_all_pools = False
                break
    elif args.force_rollout and args.seedstd:
        print("[force-rollout] 캐시 무시 — 모든 소스 강제 rollout")

    # ---- twd_gap CSV 캐시 감지: 각 source의 record/twd_gap/twd_gap_raw.csv 확인 ----
    _seedstd_all_twd_cached = False
    if args.seedstd and args.sources and not args.force_rollout:
        _sampler = "ddim"
        _cfg_tag = f"CFG={args.cfg}_NFE={args.NFE}"
        _batch_tag = f"batch={args.batch}"
        _seed_tag = f"seed={args.seed}"
        _base_seed = os.path.join("workdir", "exp_main", "manifold", "latent_std",
                                  _mt, _sampler, _cfg_tag, _batch_tag, _seed_tag, "save")
        _seedstd_all_twd_cached = True
        for g, n, _ in _parse_sources(args.sources):
            twd_csv = os.path.join(_base_seed, g, n, "record", "twd_gap", "twd_gap_raw.csv")
            if not os.path.exists(twd_csv):
                _seedstd_all_twd_cached = False
                break
            # Check n_cached >= args.n_prompts
            try:
                with open(twd_csv, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                if not rows:
                    _seedstd_all_twd_cached = False
                    break
                n_data_rows = len(rows) - 1  # exclude header
                n_cached_here = n_data_rows // args.batch
                # T check
                header_cols = len(rows[0]) - 2  # exclude prompt_idx, seed_idx
                if header_cols != args.NFE or n_cached_here < args.n_prompts:
                    _seedstd_all_twd_cached = False
                    break
            except Exception:
                _seedstd_all_twd_cached = False
                break

    if args.compare_all:
        print("[compare-all] load from save/ — skip model load (GPU 0GB)")
        sd = None
    elif args.manifold and _has_save:
        print("[save-load] all 3 latent.npz exist — skip model load (GPU 0GB)")
        sd = None  # only distance calc without model
    elif args.seedstd and _seedstd_all_pools and _seedstd_all_twd_cached:
        print(f"[pool-load] latent AND twd_gap CSV caches complete "
              f"(n_prompts>={args.n_prompts}, batch={args.batch}) → skip model load (GPU 0GB)")
        sd = None
    elif args.seedstd and _seedstd_all_pools:
        print(f"[pool-load] latent pools complete but twd_gap CSV incomplete "
              f"(n_prompts={args.n_prompts}, batch={args.batch})")
        print(f"  → load model for tweedie gap computation")
        solver_config = munchify({"num_sampling": args.NFE})
        sd = StableDiffusion(solver_config=solver_config, model_key=args.model_key,
                             device=device, seed=args.seed)
    else:
        solver_config = munchify({"num_sampling": args.NFE})
        sd = StableDiffusion(solver_config=solver_config, model_key=args.model_key,
                             device=device, seed=args.seed)

    if args.compare_all and args.manifold:
        run_compare_all(sd, args, device)     # save/의 전체 소스 비교
    elif args.manifold:
        run_manifold(sd, args, device)
    elif args.trajstd:
        run_trajstd(sd, args, device)
    elif args.seedstd:
        run_seedstd(sd, args, device)
    else:
        if sd is None:
            print("[error] fixation 실험은 모델 필요 — save-load 불가")
            sys.exit(1)
        run_fixation(sd, args, device)


if __name__ == "__main__":
    main()
