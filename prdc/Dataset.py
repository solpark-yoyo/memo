import os
import glob as glob_module
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_root, transforms=None, num_samples=None):
        self.root = data_root
        self.transforms = transforms
        img_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            img_paths += glob_module.glob(os.path.join(data_root, "**", ext), recursive=True)
            img_paths += glob_module.glob(os.path.join(data_root, ext))
        self.imgs = sorted(set(img_paths))
        if num_samples is not None:
            self.imgs = self.imgs[:num_samples]
        print("Total images found:", len(self.imgs))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        try:
            img = Image.open(self.imgs[index]).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
        except:
            print(f"Error in image {self.imgs[index]}")
            img = torch.zeros(3, 224, 224)
        return img