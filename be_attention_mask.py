"""BE (Bright Ending) attention mask — Chen et al. ICLR 2025 기반 local memorization 평가용.

마지막 denoising step의 cross-attention에서 "final text token" 에 대한 attention이
비정상적으로 큰 patch = memorized local region (Bright Ending anomaly).
이를 Otsu threshold 로 binary spatial mask m 을 만든다.

기존 AttendExciteAttnProcessor(ptp_utils)는 requires_grad 가드 때문에 no_grad inference
중엔 attention 을 저장하지 않음 → BE 평가용으로 가드를 제거한 전용 processor 사용.

Usage:
    from be_attention_mask import generate_be_mask
    mask, img = generate_be_mask(sd, prompt, cfg=7.5, device='cuda')
    # mask: [H, W] binary (image 해상도, 기본 512x512)
    # img:  생성 이미지 [3, 512, 512] (0~1)
"""
import os
import sys
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils_local.ptp_utils import AttentionStore, AttendExciteAttnProcessor
from utils_local.attn_utils import fn_get_otsu_mask


class BEAttnProcessor(AttendExciteAttnProcessor):
    """BE 평가용: requires_grad 가드 없이 항상 cross-attention 을 저장."""
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if encoder_hidden_states is None:
            is_cross = False
            encoder_hidden_states = hidden_states
        else:
            is_cross = True

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if is_cross and (self.attn_where == "all" or self.place_in_unet == self.attn_where):
            # cross-attention 만 full probs 계산 + 저장 (BE mask 용)
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            self.attnstore(attention_probs, is_cross, self.place_in_unet)
            hidden_states = torch.bmm(attention_probs, value)
        else:
            # self-attention 또는 비캡처 cross: efficient sdpa (full probs X → 메모리 절약)
            hidden_states = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask
            )
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        return hidden_states


def register_be_attention(sd, attn_res=(16, 16)):
    """SD UNet 에 BE 전용 attention processor + store 등록.
    attn_res: 캡처할 attention 해상도 (latent 공간, 보통 16x16 또는 32x32).
    """
    sd.attention_store = AttentionStore(attn_res, num_heads=sd.unet.config.attention_head_dim)
    attn_procs = {}
    for name in sd.unet.attn_processors.keys():
        place = "mid" if name.startswith("mid_block") else \
                "up" if name.startswith("up_blocks") else \
                "down" if name.startswith("down_blocks") else None
        if place is None:
            attn_procs[name] = sd.unet.attn_processors[name]
            continue
        attn_procs[name] = BEAttnProcessor(attnstore=sd.attention_store, place_in_unet=place,
                                           attn_where="all", attn_type="cross")
    sd.unet.set_attn_processor(attn_procs)
    return sd.attention_store


def build_be_mask_from_store(attention_store, final_token_idx=-2,
                             from_where=("up", "down", "mid"),
                             image_size=512, otsu=True):
    """저장된 cross-attention 에서 BE mask 생성.

    final_token_idx: "final text token" 방향 (기본 -2 = EOS 직전 마지막 실제 토큰).
    from_where: 어느 UNet block 들을 합칠지.
    image_size: mask 를 upsample 할 image 해상도 (512 → image metric 용, 64 → latent 용).
    """
    # 1) cross-attention map aggregate: (B, H', W', n_tokens)
    cross = attention_store.aggregate_attention(from_where=from_where, is_cross=True)
    # B, H', W', T = cross.shape
    # 2) final-token 방향 추출 → (B, H', W')
    attn_final = cross[..., final_token_idx]  # (B, H', W')
    # batch 첫 장 (cond) 사용
    attn_map = attn_final[0].detach().cpu()  # (H', W')

    # 3) normalize (0~1) 후 Otsu (또는 단순 threshold)
    am = attn_map.numpy()
    am = (am - am.min()) / (am.max() - am.min() + 1e-8)
    am_t = torch.from_numpy(am).float()
    if otsu:
        import cv2
        am_u8 = (am * 255).astype(np.uint8)
        thr, _ = cv2.threshold(am_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (am > (thr / 255.0)).astype(np.float32)
    else:
        mask = (am > am.mean()).astype(np.float32)
    mask_t = torch.from_numpy(mask).float()  # (H', W')

    # 4) upsample → image_size (또는 latent)
    mask_up = torch.nn.functional.interpolate(
        mask_t[None, None], size=(image_size, image_size), mode='nearest'
    )[0, 0]
    return mask_up  # (image_size, image_size)


def generate_be_mask(sd, prompt, null_prompt="", cfg=7.5,
                     num_inference_steps=50, device='cuda',
                     final_token_idx=-2, image_size=512,
                     attn_res=(16, 16)):
    """SD 로 prompt 생성하며 마지막 step BE mask 생성.

    Returns:
        mask: [image_size, image_size] binary
        img:  [3, image_size, image_size] (0~1) 생성 이미지
    """
    uc, c = sd.get_text_embed(null_prompt=null_prompt, prompt=prompt)
    uc, c = uc.float(), c.float()                  # fp32 (사용자 요청)
    register_be_attention(sd, attn_res=attn_res)
    sd.unet.float()                                 # fp32 통일
    sd.vae.float()                                  # VAE 도 fp32 (decode dtype match)
    try:
        sd.unet.enable_gradient_checkpointing()     # 메모리 절약 (activation 재계산)
    except Exception:
        pass
    store = sd.attention_store

    # 직접 DDIM loop (sample() 의 popt/etc_kwargs 의존 회피)
    zt = sd.initialize_latent()    # fp32 (unet.float 와 맞춤)
    for step, t in enumerate(sd.scheduler.timesteps):
        noise_uc, noise_c = sd.predict_noise(zt, t, uc, c)
        eps = noise_uc + cfg * (noise_c - noise_uc)
        at = sd.alpha(t)
        at_prev = sd.alpha(t - sd.skip)
        x0 = (zt - (1 - at).sqrt() * eps) / at.sqrt()
        zt = at_prev.sqrt() * x0 + (1 - at_prev).sqrt() * eps
        # 매 step 끝: step_store → attention_store commit (마지막 step 것이 남음)
        store.between_steps()

    img = (sd.decode(x0) / 2 + 0.5).clamp(0, 1)
    mask = build_be_mask_from_store(store, final_token_idx=final_token_idx,
                                    image_size=image_size)
    return mask, img


if __name__ == "__main__":
    # 간단 smoke test (실제 실행은 별도 스크립트에서 SD 로드 후)
    print("be_attention_mask module loaded.")
    print("  generate_be_mask(sd, prompt, ...) → mask[H,W], img[3,H,W]")
