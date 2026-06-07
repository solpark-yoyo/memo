import os
import torch
import torch.nn as nn
from torchvision.models import vgg16

_CKPT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ckpt")
_VGG16_PTH = os.path.join(_CKPT_DIR, "vgg16-397923af.pth")

class RandomX(nn.Module):
    def __init__(self, out_feats=64):
        super(RandomX, self).__init__()
        self.vgg16 = vgg16(pretrained=False)
        self.features = self.vgg16.features.requires_grad_(False)
        self.fc1 = nn.Linear(25088, out_feats).requires_grad_(False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class TrainedX(nn.Module):
    def __init__(self, out_feats=64):
        super(TrainedX, self).__init__()
        print(f"[CKPT] Loading VGG16 weights (local): {_VGG16_PTH}")
        self.vgg16 = vgg16(pretrained=False)
        self.vgg16.load_state_dict(torch.load(_VGG16_PTH, map_location='cpu'))
        self.features = self.vgg16.features.requires_grad_(False)
        self.fc1 = self.vgg16.classifier[0].requires_grad_(False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x