import io
import os
import shutil
from pathlib import Path
from typing import Optional

import kagglehub
import torch
import torch.nn as nn
from PIL import Image

from src.transforms import SegmentationTransforms, ClassificationTransform


def _resolve_device(device: str | None) -> str:
    """
    Normalize a requested device string, falling back to CPU with a
    warning (instead of raising) when CUDA was requested/expected but
    isn't available. This keeps the app usable on machines without a
    GPU (e.g. a demo laptop or a free-tier deployment).
    """
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Warning: CUDA was requested but is not available.")

    if device == "cpu":
        raise RuntimeError(
            "CUDA is not available.\n"
            "This model is intended to run on an NVIDIA GPU.\n"
            "Please run the code on a machine with CUDA support."
        )


def _to_pil_image(image_or_path: "str | Path | bytes | Image.Image") -> Image.Image:
    """
    Accepts a file path, raw bytes (e.g. from an uploaded file), or an
    already-open PIL Image, and returns a PIL Image in all cases.
    """
    if isinstance(image_or_path, Image.Image):
        return image_or_path
    if isinstance(image_or_path, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_or_path))
    return Image.open(image_or_path)


def load_model(model_class: type,
               weights_path: str | Path,
               model_kwargs: dict | None = None,
               device: str | None = None) -> nn.Module:
    """
    Load a trained PyTorch model from a saved state_dict and prepare it
    for inference.

    Args:
        model_class: The nn.Module subclass defining the model architecture
            (not an instance, the class itself).
        weights_path: Path to the saved state_dict file (.pt or .pth).
        model_kwargs: Keyword arguments needed to instantiate model_class
            (e.g. {"num_classes": 5}). Pass None if the class takes no args.
        device: Target device, e.g. "cuda", "cpu". If None, automatically
            picks "cuda" when available, otherwise "cpu".

    Returns:
        The model instance, loaded with trained weights, moved to the
        target device, and set to evaluation mode (ready for prediction).
    """
    model_kwargs = model_kwargs or {}
    device = _resolve_device(device)

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at '{weights_path}'. Place the trained "
            f"checkpoint there (see README.md) before running inference."
        )

    try:
        # Instantiate the architecture
        model = model_class(**model_kwargs)

        # Load the trained weights (weights_only=True is the safe default
        # from PyTorch 2.6+; state dicts of plain tensors satisfy it).
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)

        # Some checkpoints were saved from a DataParallel-wrapped model and
        # carry a "module." prefix on every key — strip it if present so
        # load_state_dict works regardless of how the model was trained.
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

        model.load_state_dict(state_dict)

        # Move to device and switch to inference mode
        model.to(device)
        model.eval()

        print(f"Model loaded from '{weights_path}' and ready for inference on '{device}'.")
        return model
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading model from '{weights_path}': {e}") from e


def download_data(dataset: Optional[str],
                  dest_dir: Optional[str] = "data") -> str:
    """
    Download the specified dataset from Kaggle and store it in the
    destination directory.

    Args:
        dataset: Kaggle dataset identifier in the form 'owner/dataset-name'.
        dest_dir: Path to the local folder where the data will be saved.

    Returns:
        The path to the local directory containing the downloaded data.
    """
    os.makedirs(dest_dir, exist_ok=True)

    try:
        # Download the dataset (kagglehub caches it and returns the cache path)
        cached_path = kagglehub.dataset_download(dataset)

        # Copy the downloaded contents into the destination directory
        for item in os.listdir(cached_path):
            src = os.path.join(cached_path, item)
            dst = os.path.join(dest_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        print(f"Dataset saved to '{dest_dir}'.")
        return dest_dir

    except Exception as e:
        print(f"Failed to download dataset '{dataset}': {e}")
        raise


def predict_mask(
        model: nn.Module,
        image_or_path: str | bytes,
        device: str | None = None,
        image_size: tuple[int, int] = (512, 512),
) -> dict:
    """
    Run binary segmentation inference on a single MRI image.

    Model output is expected to have shape:
        (B, 1, H, W)

    The single output channel represents the tumor probability.
    """

    device = _resolve_device(device)

    image = _to_pil_image(image_or_path)
    original_size = image.size  # (width, height)

    transform = SegmentationTransforms(image_size)

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

        # Binary segmentation:
        # output shape -> (1, 1, H, W)
        tumor_probabilities = torch.sigmoid(output).squeeze(0).squeeze(0)

        # Convert probability map to binary mask
        predicted_mask = (tumor_probabilities >= 0.5).long()

    tumor_pixels = (predicted_mask == 1).sum().item()
    total_pixels = predicted_mask.numel()

    tumor_coverage = (tumor_pixels / total_pixels) * 100

    if tumor_pixels > 0:
        confidence = (
                float(
                    tumor_probabilities[predicted_mask == 1]
                        .mean()
                        .item()
                )
                * 100
        )
    else:
        confidence = 0.0

    return {
        "predicted_mask": predicted_mask.cpu(),
        "probabilities": tumor_probabilities.cpu(),
        "tumor_coverage": round(tumor_coverage, 2),
        "confidence": round(confidence, 2),
        "original_size": original_size,
    }


def predict_tumor_class(model: nn.Module,
                        image_or_path: str | bytes,
                        class_names: list[str] | None = None,
                        device: str | None = None,
                        resize: tuple[int, int] = (256, 256),
                        ) -> dict:
    """
    Run classification inference on a single brain MRI image using a
    loaded model.

    Args:
        model: A model already loaded and set to eval() (see load_model).
        image_or_path: Path to the image file to classify.
        class_names: Optional list mapping class index -> class name.
            If None, the raw class index is returned instead of a name.
            For BRISC-style datasets this is typically something like
            ["glioma", "meningioma", "no_tumor", "pituitary"].
        device: "cuda" or "cpu". If None, auto-detects (falls back to
            requiring GPU, same as predict_image).
        resize: Spatial size the image is resized to before inference.
            Should match the size the model was trained on.

    Returns:
        A dict with:
            - "predicted_class": class name (or index if class_names is None)
            - "predicted_index": the raw predicted class index
            - "confidence": confidence of the predicted class, as a percentage (0-100)
            - "probabilities": dict mapping class name -> probability (0-100),
                or a raw tensor if class_names is None
            - "has_tumor": bool, False only if predicted class is "no_tumor"
                (best-effort check based on class_names; None if it can't
                be determined)
    """
    device = _resolve_device(device)

    # Reuse the exact same preprocessing used at training time (see
    # transforms.py) instead of a separate ad-hoc pipeline, so inference
    # always matches training. grayscale=True matches ClassificationModel's
    # default in_channels=1 (brain MRI scans are single-channel, same
    # convention as the segmentation model).
    transform = ClassificationTransform(
        image_size=resize,
        grayscale=True,
        augment=False,
    )

    image = _to_pil_image(image_or_path)
    input_tensor = transform(image).unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze(0)
        predicted_index = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted_index].item()) * 100

    if class_names is not None:
        predicted_class = class_names[predicted_index]
        probs_out = {
            name: round(float(probabilities[i].item()) * 100, 2)
            for i, name in enumerate(class_names)
        }
        has_tumor = predicted_class.lower() != "no_tumor"
    else:
        predicted_class = predicted_index
        probs_out = probabilities.cpu()
        has_tumor = None

    return {
        "predicted_class": predicted_class,
        "predicted_index": predicted_index,
        "confidence": round(confidence, 2),
        "probabilities": probs_out,
        "has_tumor": has_tumor,
    }
