from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
import torch


def resize(image):
    """Shared resize + pad logic used by both train and val transforms."""
    width, height = image.size
    new_width = int(width * (64 / height))
    image = image.resize((new_width, 64))
    return image,new_width

def pad(image,new_width):
    img_tensor = TF.to_tensor(image)
    padded = torch.nn.functional.pad(img_tensor, (0, 1400 - img_tensor.shape[-1]), value=1.0)
    seq_len = new_width // 8
    return padded, seq_len

def transform(image):
    """Clean transform — no augmentation. Use for validation/test."""
    image = image.convert('L')
    image,new_width=resize(image)
    return pad(image,new_width)


def train_transform(image):
    """
    Augmented transform — use for training only.
    Applies resizing then mild rotation, elastic distortion, and brightness/contrast jitter
    BEFORE padding, so augmentation acts on the real image content,
    not on the padded canvas.
    """
    image = image.convert('L')
    image,new_width=resize(image)

    # 1. Slight random rotation (±0.5 degrees) — simulates natural handwriting slant variation.
    #    fillcolor=255 keeps the background white (matches your padding value=1.0 convention).
    angle = random.uniform(-0.5, 0.5)
    image = TF.rotate(image, angle, fill=255)

    # 2. Elastic distortion — simulates natural stroke/pen variation between writers.
    #    alpha controls displacement magnitude, sigma controls smoothness of the warp.
    if random.random() < 0.5:  # apply only half the time, to keep some clean samples too
        elastic = T.ElasticTransform(alpha=15.0, sigma=5.0, fill=255)
        image = elastic(image)

    # 3. Brightness/contrast jitter — simulates scan/photo lighting differences.
    jitter = T.ColorJitter(brightness=0.3, contrast=0.3)
    image = jitter(image)

    return pad(image,new_width)