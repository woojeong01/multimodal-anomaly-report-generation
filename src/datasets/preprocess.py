import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

# ImageNet 정규화 값
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(image_size: tuple[int, int] = (1024, 1024)) -> Compose:
    return Compose([
        Resize(image_size),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def prepare_input(image_bgr: np.ndarray, transform: Compose) -> torch.Tensor:
    image_rgb = image_bgr[:, :, ::-1]
    pil_image = Image.fromarray(image_rgb)
    return transform(pil_image).unsqueeze(0)