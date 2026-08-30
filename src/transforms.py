import random as rn

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class ClassificationTransform:
    """
    Single source of truth for preprocessing used by both training
    (ImageClassificationDataset) and inference (predict_tumor_class).

    Args:
        image_size (tuple): target size (H, W)
        grayscale (bool): if True, convert to "L" and normalize with
            mean=[0.5], std=[0.5]; if False, convert to "RGB" and use
            ImageNet mean/std.
        augment (bool): enable/disable augmentation (train split only).
            Should always be False for inference.
    """

    def __init__(
            self, image_size: list[int] | tuple,
            grayscale=False, augment=False):
        self.image_size = image_size
        self.grayscale = grayscale
        self.augment = augment

        if grayscale:
            self.mean = [0.5]
            self.std = [0.5]
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

    def __call__(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        # --- Mode normalization ---
        image = image.convert("L") if self.grayscale else image.convert("RGB")

        # --- Resize ---
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)

        # --- Augmentation (train only) ---
        if self.augment:
            if rn.random() > 0.5:
                image = TF.hflip(image)
            if rn.random() > 0.5:
                angle = rn.uniform(-15, 15)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            if rn.random() > 0.5:
                brightness = rn.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness)

        # --- Convert to tensor ---
        image = TF.to_tensor(image)  # [C, H, W], values in [0, 1]

        # --- Normalize ---
        image = TF.normalize(image, mean=self.mean, std=self.std)

        return image


class SegmentationTransforms:
    """
    Test-time preprocessing for a single MRI image, mirroring
    BrainMRISegDataset's preprocessing but WITHOUT augmentation.

    Note: unlike BrainMRISegDataset (aspect-ratio-preserving resize +
    zero-pad), this performs a plain resize to image_size — matches
    the original predict_mask implementation as-is.

    Pipeline:
        grayscale -> bilinear resize -> to_tensor -> normalize

    Args:
        image_size (tuple): target (H, W) size — must match training size.
    """

    def __init__(self, image_size: tuple[int, int] = (384, 384)):
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Args:
            image: a PIL Image (any mode/size).

        Returns:
            input_tensor: [1, H, W] normalized tensor (no batch dim).
        """
        image = image.convert("L")  # single-channel, same as training
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)

        input_tensor = TF.to_tensor(image)  # [1, H, W], values in [0, 1]
        input_tensor = TF.normalize(input_tensor, mean=[0.5], std=[0.5])

        return input_tensor
