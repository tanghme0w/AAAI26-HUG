"""
Data Transforms for HUG

This module provides image preprocessing and augmentation transforms
compatible with BLIP-2 preprocessing requirements.
"""

import torch
from torchvision import transforms
from PIL import Image
from typing import Optional


class BLIPImageTransform:
    """
    Standard image transform for BLIP-2 models.

    BLIP-2 expects images to be:
    - Resized to 224x224 (or model-specific size)
    - Normalized with ImageNet statistics
    """

    def __init__(
        self,
        image_size: int = 224,
        is_train: bool = True
    ):
        """
        Args:
            image_size: Target image size (default: 224)
            is_train: If True, apply training augmentations
        """
        self.image_size = image_size

        if is_train:
            # Training transforms with augmentation
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP/BLIP normalization
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        else:
            # Evaluation transforms (no augmentation)
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Apply transform to image.

        Args:
            image: PIL Image

        Returns:
            Transformed image tensor [3, H, W]
        """
        return self.transform(image)


class SquarePad:
    """
    Pad image to square before resizing.
    Useful for maintaining aspect ratio.
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Pad image to square.

        Args:
            image: PIL Image

        Returns:
            Padded PIL Image
        """
        width, height = image.size
        max_side = max(width, height)

        # Create new square image with padding
        new_image = Image.new(image.mode, (max_side, max_side), (255, 255, 255))

        # Paste original image in center
        left = (max_side - width) // 2
        top = (max_side - height) // 2
        new_image.paste(image, (left, top))

        return new_image


def get_transform(image_size: int = 224, is_train: bool = True) -> BLIPImageTransform:
    """
    Get the standard transform for HUG training/evaluation.

    Args:
        image_size: Target image size
        is_train: Whether to use training augmentations

    Returns:
        Transform function
    """
    return BLIPImageTransform(image_size=image_size, is_train=is_train)
