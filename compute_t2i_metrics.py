import blobfile as bf
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel
from PIL import Image
import torch
import torch.nn.functional as F
import os
import csv
from tqdm import tqdm
import os
import ImageReward as RM

from torchvision.io import read_image

import argparse

_CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")

def main():
    parser = argparse.ArgumentParser(description="T2I metrics computation")
    parser.add_argument("--eval_dir", type=str, default="", required=True)
    parser.add_argument("--prompt_dir", type=str, default="/home/usb/CFGpp/examples/assets/coco_v2.txt")
    parser.add_argument("--num_prompts", type=int, default=1000)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path (default: metrics/<basedir>_t2i.csv)")
    parser.add_argument("--log_csv", type=str, default=None, help="Path to save detailed log CSV")
    parser.add_argument("--cs_only", action='store_true')
    parser.add_argument("--warn_log", type=str, default=None, help="Redirect Python warnings to this file")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.warn_log:
        import logging
        os.makedirs(os.path.dirname(os.path.abspath(args.warn_log)), exist_ok=True)
        logging.captureWarnings(True)
        _wh = logging.FileHandler(args.warn_log, mode='a')
        logging.getLogger('py.warnings').addHandler(_wh)

    evalfolder = args.eval_dir
    num_prompts = args.num_prompts
    prompt_dir = args.prompt_dir

    # CSV 경로: metrics/<basedir>_t2i.csv
    if args.output_csv:
        output_file = args.output_csv
    else:
        project_root = os.path.dirname(os.path.abspath(__file__))
        metrics_dir = os.path.join(project_root, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        basedir = os.path.basename(os.path.normpath(evalfolder))
        output_file = os.path.join(metrics_dir, f"{basedir}_t2i.csv")

    text_list = []
    with open(prompt_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # Only add non-empty lines
                text_list.append(stripped_line)
    prompts = text_list[:num_prompts]
    if args.num_images_per_prompt > 1:
        prompts = [text for text in prompts for _ in range(args.num_images_per_prompt)]
    
    num_samples = num_prompts * args.num_images_per_prompt

    def _list_image_files_recursively(data_dir):
        results = []
        for entry in sorted(bf.listdir(data_dir)):
            full_path = bf.join(data_dir, entry)
            ext = entry.split(".")[-1]
            if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
                results.append(full_path)
            elif bf.isdir(full_path):
                results.extend(_list_image_files_recursively(full_path))
        return results

    device = args.device
    if device in "cuda":
        torch.cuda.set_device(args.device)
        
    clip_local_path = os.path.join(_CKPT_DIR, "clip-vit-base-patch16")
    print(f"[CKPT] Loading CLIP (local): {clip_local_path}")
    clip_model = CLIPModel.from_pretrained(clip_local_path, local_files_only=True).eval().to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_local_path, local_files_only=True)

    def clip_score_fn(img_tensor, prompt):
        pil_img = Image.fromarray(img_tensor.permute(1, 2, 0).cpu().numpy())
        inputs = clip_processor(text=[prompt], images=[pil_img], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            img_emb = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_emb = clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        return (img_emb @ txt_emb.T).squeeze() * 100


    processor_name_or_path = os.path.join(_CKPT_DIR, "CLIP-ViT-H-14-laion2B-s32B-b79K")
    model_pretrained_name_or_path = os.path.join(_CKPT_DIR, "PickScore_v1")
    print(f"[CKPT] Loading PickScore processor (local): {processor_name_or_path}")
    print(f"[CKPT] Loading PickScore model (local): {model_pretrained_name_or_path}")
    processor = AutoProcessor.from_pretrained(processor_name_or_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path, local_files_only=True).eval().to(device)

    def calc_probs(prompt, images):
        # preprocess
        image_inputs = processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        
        text_inputs = processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)


        with torch.no_grad():
            # embed
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
            # get probabilities if you have multiple images to choose from
            probs = torch.softmax(scores, dim=-1)
        
        # return probs.cpu().tolist()
        return scores.cpu()

    # Monkey-patch BLIP's init_tokenizer to use local bert-base-uncased.
    # blip_pretrain.py does `from .blip import init_tokenizer`, so we must patch
    # blip_pretrain's namespace (not blip's) to override the already-imported binding.
    _bert_local = os.path.join(_CKPT_DIR, "bert-base-uncased")
    print(f"[CKPT] Loading BERT tokenizer (local): {_bert_local}")
    import ImageReward.models.BLIP.blip as _blip_module
    import ImageReward.models.BLIP.blip_pretrain as _blip_pretrain_module
    from transformers import BertTokenizer as _BertTokenizer
    def _local_init_tokenizer():
        tokenizer = _BertTokenizer.from_pretrained(_bert_local)
        tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        return tokenizer
    _blip_module.init_tokenizer = _local_init_tokenizer
    _blip_pretrain_module.init_tokenizer = _local_init_tokenizer

    _ir_pt = os.path.join(_CKPT_DIR, "ImageReward", "ImageReward.pt")
    print(f"[CKPT] Loading ImageReward (local): {_ir_pt}")
    model_rm = RM.load(
        _ir_pt,
        med_config=os.path.join(_CKPT_DIR, "ImageReward", "med_config.json"),
    )

    eval_img_path = _list_image_files_recursively(evalfolder)[:num_samples]

    assert len(eval_img_path) == num_samples and len(eval_img_path) == len(prompts), "Number of samples should be equal to the length of image files and text files."

    cs_list = []

    for idx, prompt in enumerate(tqdm(prompts, desc="CLIPScore")):
        img = read_image(eval_img_path[idx]).to(device)
        clipscore = clip_score_fn(img, prompt).detach()
        cs_list.append(clipscore.item())

    cs_tensor = torch.tensor(cs_list)

    ps_list = []
    ir_list = []

    if not args.cs_only:
        for idx, prompt in enumerate(tqdm(prompts, desc="PickScore")):
            pickscore = calc_probs(prompt, [Image.open(eval_img_path[idx])])
            ps_list.append(pickscore.item())
        ps_tensor = torch.tensor(ps_list)

        for idx, prompt in enumerate(tqdm(prompts, desc="ImageReward")):
            ir = model_rm.score(prompt, [eval_img_path[idx]])
            ir_list.append(ir)
        ir_tensor = torch.tensor(ir_list)

    # CSV 저장
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["# Config"])
        writer.writerow(["clip_ckpt", clip_local_path])
        writer.writerow(["pickscore_ckpt", model_pretrained_name_or_path])
        writer.writerow(["imagereward_ckpt", _ir_pt])
        writer.writerow(["fake_data_path", evalfolder])
        writer.writerow(["prompt_dir", prompt_dir])
        writer.writerow(["num_prompts", num_prompts])
        writer.writerow(["num_images_per_prompt", args.num_images_per_prompt])
        writer.writerow(["total_images_loaded", len(eval_img_path)])
        writer.writerow([])

        if args.cs_only:
            writer.writerow(["filename", "CLIPScore"])
            for idx, img_path in enumerate(eval_img_path):
                filename = os.path.basename(img_path)
                writer.writerow([filename, f"{cs_list[idx]:.6f}"])
            writer.writerow([])
            writer.writerow(["metric", "mean", "std"])
            writer.writerow(["CLIPScore", f"{cs_tensor.mean().item():.6f}", f"{cs_tensor.std().item():.6f}"])
        else:
            writer.writerow(["filename", "CLIPScore", "PickScore", "ImageReward"])
            for idx, img_path in enumerate(eval_img_path):
                filename = os.path.basename(img_path)
                writer.writerow([filename, f"{cs_list[idx]:.6f}", f"{ps_list[idx]:.6f}", f"{ir_list[idx]:.6f}"])
            writer.writerow([])
            writer.writerow(["metric", "mean", "std"])
            writer.writerow(["CLIPScore", f"{cs_tensor.mean().item():.6f}", f"{cs_tensor.std().item():.6f}"])
            writer.writerow(["PickScore", f"{ps_tensor.mean().item():.6f}", f"{ps_tensor.std().item():.6f}"])
            writer.writerow(["ImageReward", f"{ir_tensor.mean().item():.6f}", f"{ir_tensor.std().item():.6f}"])

    print(f"[Done] Saved metrics to {output_file}")
    print("=" * 50)
    print(f"  CLIPScore  : {cs_tensor.mean().item():.4f} ± {cs_tensor.std().item():.4f}")
    if not args.cs_only:
        print(f"  PickScore  : {ps_tensor.mean().item():.4f} ± {ps_tensor.std().item():.4f}")
        print(f"  ImageReward: {ir_tensor.mean().item():.4f} ± {ir_tensor.std().item():.4f}")
    print("=" * 50)

    if args.log_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_csv)), exist_ok=True)
        with open(args.log_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["# Config"])
            writer.writerow(["eval_dir",      evalfolder])
            writer.writerow(["prompt_dir",    prompt_dir])
            writer.writerow(["clip_model",    os.path.join(_CKPT_DIR, "clip-vit-base-patch16")])
            writer.writerow(["pickscore_model", os.path.join(_CKPT_DIR, "PickScore_v1")])
            writer.writerow(["ir_model",      os.path.join(_CKPT_DIR, "ImageReward")])
            writer.writerow(["num_prompts",   num_prompts])
            writer.writerow([])
            writer.writerow(["# Results"])
            writer.writerow(["metric", "mean", "std"])
            writer.writerow(["CLIPScore", f"{cs_tensor.mean().item():.6f}", f"{cs_tensor.std().item():.6f}"])
            if not args.cs_only:
                writer.writerow(["PickScore",   f"{ps_tensor.mean().item():.6f}", f"{ps_tensor.std().item():.6f}"])
                writer.writerow(["ImageReward", f"{ir_tensor.mean().item():.6f}", f"{ir_tensor.std().item():.6f}"])
        print(f"[Done] T2I log saved to: {args.log_csv}")

if __name__ == '__main__':
    main()