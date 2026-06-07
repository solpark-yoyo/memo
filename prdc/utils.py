from torchvision import transforms
from PIL import Image

def resize(img, w, h):
    return img.resize((w, h), Image.Resampling.BILINEAR)
customtransforms = transforms.Compose([
        transforms.Lambda(lambda img: resize(img, 224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])