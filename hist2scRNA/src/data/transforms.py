"""
Data augmentation and preprocessing transforms

Provides transformations for histopathology images and spatial data.
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import random


class HistoNormalize:
    """
    Normalize histopathology images using standard H&E staining statistics
    """
    def __init__(self, mean=(0.7, 0.6, 0.7), std=(0.15, 0.15, 0.15)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return F.normalize(img, mean=self.mean, std=self.std)


class RandomHistoAugmentation:
    """
    Random augmentation for histopathology images
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        # Random horizontal flip
        if random.random() < self.p:
            img = F.hflip(img)

        # Random vertical flip
        if random.random() < self.p:
            img = F.vflip(img)

        # Random rotation (0, 90, 180, 270 degrees)
        if random.random() < self.p:
            angle = random.choice([0, 90, 180, 270])
            img = F.rotate(img, angle)

        # Random color jitter
        if random.random() < self.p:
            color_jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
            img = color_jitter(img)

        return img


def get_default_transforms(augment=False, img_size=224):
    """
    Get default transforms for histopathology images

    Args:
        augment: whether to apply data augmentation
        img_size: target image size

    Returns:
        transform: composed transform
    """
    transforms = []

    # Resize if needed
    transforms.append(T.Resize((img_size, img_size)))

    # Convert to tensor
    transforms.append(T.ToTensor())

    # Apply augmentation if requested
    if augment:
        transforms.append(RandomHistoAugmentation(p=0.5))

    # Normalize
    transforms.append(HistoNormalize())

    return T.Compose(transforms)
