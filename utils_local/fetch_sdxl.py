"""
stable-diffusion-xl-base-1.0 을 diffusers 0.29.x 호환 형태로 다운로드합니다.

저장 경로: jepa_sd_1/ckpt/stable-diffusion-xl-base-1.0/

diffusers 0.29.2 에서 from_pretrained 로드에 필요한 구조:
    model_index.json
    scheduler/scheduler_config.json          ← 0.29.x 는 이 파일명 사용
    unet/config.json + diffusion_pytorch_model.safetensors
    vae/config.json  + diffusion_pytorch_model.safetensors
    text_encoder/config.json  + model.safetensors
    text_encoder_2/config.json + model.safetensors
    tokenizer/{tokenizer_config.json, vocab.json, merges.txt, special_tokens_map.json}
    tokenizer_2/  (same)

제외 파일 (불필요 / 구조 혼란 야기):
    *.onnx, *.onnx_data, *.xml, openvino*  ← OpenVINO 포맷
    flax_model*, tf_model*, *.h5, *.msgpack ← 비 PyTorch 포맷
    *.fp16.safetensors                       ← fp32 로 충분, 필요 시 제거 가능

사용법:
    # 기본 (이미 존재하면 스킵)
    python -m utils_local.fetch_sdxl

    # 강제 재다운로드
    python -m utils_local.fetch_sdxl --force

    # 저장 경로 직접 지정
    python -m utils_local.fetch_sdxl --local_dir /path/to/ckpt/stable-diffusion-xl-base-1.0
"""

import argparse
import os
from pathlib import Path

REPO_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# jepa_sd_1/ckpt/stable-diffusion-xl-base-1.0
_DEFAULT_DIR = Path(__file__).parent.parent / "ckpt" / "stable-diffusion-xl-base-1.0"

# diffusers 0.29.2 로드에 필요한 최소 파일 목록 (검증용)
_REQUIRED_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "tokenizer_2/merges.txt",
]

# 불필요한 파일 패턴 (OpenVINO, 비PyTorch 포맷)
_IGNORE_PATTERNS = [
    "*.onnx",
    "*.onnx_data",
    "*.xml",
    "openvino*",
    "flax_model*",
    "tf_model*",
    "*.h5",
    "*.msgpack",
    # fp16 safetensors 는 from_pretrained 에서 torch_dtype=float16 으로 자동 처리되므로 제외
    # 필요하면 아래 주석 해제
    # "*.fp16.safetensors",
]


def verify_structure(local_dir: Path) -> bool:
    missing = []
    for rel in _REQUIRED_FILES:
        if not (local_dir / rel).exists():
            missing.append(rel)
    if missing:
        print("\n[WARN] 아래 파일이 없습니다:")
        for f in missing:
            print(f"       missing: {f}")
        return False
    print("[OK] 필수 파일 구조 확인 완료")
    return True


def download(local_dir: Path, force: bool = False):
    from huggingface_hub import snapshot_download

    if local_dir.exists() and any(local_dir.iterdir()) and not force:
        print(f"[SKIP] 이미 존재: {local_dir}")
        print("       강제 재다운로드: --force 옵션 사용")
        verify_structure(local_dir)
        return

    print(f"[HF] {REPO_ID} 다운로드 시작")
    print(f"     → {local_dir}")
    print(f"     제외 패턴: {_IGNORE_PATTERNS}\n")

    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(local_dir),
        ignore_patterns=_IGNORE_PATTERNS,
    )

    print(f"\n[OK] 다운로드 완료: {local_dir}")
    verify_structure(local_dir)


def main():
    parser = argparse.ArgumentParser(
        description="stable-diffusion-xl-base-1.0 다운로드 (diffusers 0.29.x 호환)"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=str(_DEFAULT_DIR),
        help=f"저장 경로 (기본값: {_DEFAULT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 존재해도 강제 재다운로드",
    )
    args = parser.parse_args()

    download(Path(args.local_dir), force=args.force)


if __name__ == "__main__":
    main()
