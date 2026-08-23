import os
import shutil
from typing import Optional

import kagglehub
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms


def load_model(model_class: type,
               weights_path: str,
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
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        raise RuntimeError(
            "No GPU available (or 'cpu' was explicitly requested), but this "
            "model requires a GPU device. Make sure CUDA is available and "
            "torch.cuda.is_available() returns True."
        )
    try:

        # Instantiate the architecture
        model = model_class(**model_kwargs)

        # Load the trained weights
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

        # Move to device and switch to inference mode
        model.to(device)
        model.eval()

        print(f"Model loaded from '{weights_path}' and ready for inference on '{device}'.")
        return model
    except Exception as E:
        raise RuntimeError("Error loading model")


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


def predict_mask(model: nn.Module,
                 image_or_path: str | bytes,
                 device: str | None = None,
                 image_size: tuple[int, int] = (512, 512),
                 ) -> dict:
    """
    Run segmentation inference on a single MRI image using a loaded model.
    Mirrors the exact preprocessing used in BrainMRISegDataset (test-time,
    no augmentation): grayscale -> bilinear resize -> to_tensor -> normalize.

    Args:
        model: A model already loaded and set to eval() (see load_model).
        image_or_path: Path to the image file to segment.
        device: "cuda" or "cpu". If None, auto-detects (falls back to
            requiring GPU, same as predict_image).
        image_size: Target (H, W) size — must match training size (512x512).

    Returns:
        A dict with:
            - "predicted_mask": tensor (H, W), long, 0=background 1=tumor
            - "probabilities": tensor (H, W) of softmax probability for the
                tumor class (class index 1)
            - "tumor_coverage": percentage of pixels predicted as tumor (0-100)
            - "confidence": mean tumor-class probability within the predicted
                tumor region (0-100), or 0.0 if no tumor pixels were predicted
            - "original_size": (width, height) of the input image, for
                resizing the mask back for overlay on the original image
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        raise RuntimeError(
            "No GPU available (or 'cpu' was explicitly requested), but this "
            "model requires a GPU device. Make sure CUDA is available and "
            "torch.cuda.is_available() returns True."
        )

    with Image.open(image_or_path) as img:
        image = img.copy()

    original_size = image.size  # (width, height), before any resizing

    # --- Match BrainMRISegDataset._joint_transform exactly (no augment) ---
    image = image.convert("L")  # single-channel, same as training
    image = TF.resize(image, image_size, interpolation=TF.InterpolationMode.BILINEAR)

    input_tensor = TF.to_tensor(image)  # [1, H, W], values in [0, 1]
    input_tensor = TF.normalize(input_tensor, mean=[0.5], std=[0.5])
    input_tensor = input_tensor.unsqueeze(0).to(device)  # add batch dim -> [1, 1, H, W]

    with torch.no_grad():
        output = model(input_tensor)  # expected shape: (1, 2, H, W)
        probabilities = torch.softmax(output, dim=1).squeeze(0)  # -> (2, H, W)
        predicted_mask = probabilities.argmax(dim=0)  # -> (H, W), long

        tumor_probabilities = probabilities[1]  # prob of tumor class -> (H, W)

    tumor_pixels = (predicted_mask == 1).sum().item()
    total_pixels = predicted_mask.numel()
    tumor_coverage = (tumor_pixels / total_pixels) * 100

    if tumor_pixels > 0:
        confidence = float(tumor_probabilities[predicted_mask == 1].mean().item()) * 100
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
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        raise RuntimeError(
            "No GPU available (or 'cpu' was explicitly requested), but this "
            "model requires a GPU device. Make sure CUDA is available and "
            "torch.cuda.is_available() returns True."
        )

    image = Image.open(image_or_path).convert("RGB")
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
