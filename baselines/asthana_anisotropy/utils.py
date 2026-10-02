import torch
import torchvision.transforms as transforms
import torch.nn as nn


def arnoldi_iteration_jvp(b, n: int, jvp_func):
    eps = 1e-12
    h = torch.zeros((n + 1, n), dtype=torch.float64, device=b.device)
    Q = torch.zeros((b.numel(), n + 1), dtype=torch.complex128, device=b.device)

    Q[:, 0] = b / torch.norm(b, 2)

    for k in range(1, n + 1):
        vec = Q[:, k - 1]
        v = jvp_func(vec)
        for j in range(k):
            h[j, k - 1] = torch.real(Q[:, j].conj().dot(v))
            v = v - h[j, k - 1] * Q[:, j]

        h[k, k - 1] = torch.norm(v, 2)

        if h[k, k - 1] > eps:
            Q[:, k] = v / h[k, k - 1]
        else:
            return Q[:, :k], h[:k+1, :k]

    h = h[:-1, :]
    return h


def measure_CLIP_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)


def measure_SSCD_similarity(gt_images, images, model, device):
    ret_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    gt_images = torch.stack([ret_transform(x.convert("RGB")) for x in gt_images]).to(device)
    images    = torch.stack([ret_transform(x.convert("RGB")) for x in images]).to(device)

    gt_images = gt_images.type(torch.float32)
    images    = images.type(torch.float32)
    with torch.no_grad():
        feat_1 = nn.functional.normalize(model(gt_images).clone(), dim=1, p=2)
        feat_2 = nn.functional.normalize(model(images).clone(),    dim=1, p=2)
        return torch.max(torch.mm(feat_1, feat_2.T), dim=0).values
