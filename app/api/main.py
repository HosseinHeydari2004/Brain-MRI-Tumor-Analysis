"""
Brain MRI Analysis API — FastAPI backend.

Wraps the segmentation (ResNet34 U-Net3+) and classification (ResNet34)
models behind a small REST API so any client (the bundled Streamlit app,
a curl request, another service) can request tumor classification and/or
segmentation for a single MRI slice.

Run with:
    uvicorn app.api.main:app --reload --port 8000

Model weights are expected at:
    models/best_model_seg.pth
    models/best_model_classif.pth
(see README.md for details). The API starts up fine even if the weights
are missing — endpoints that need a missing model return HTTP 503 with a
clear message instead of crashing the whole server.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.api.schemas import (
    ClassificationResponse,
    ErrorResponse,
    FullAnalysisResponse,
    HealthResponse,
    SegmentationResponse,
)
from src.model import ClassificationModel, SegmentationModel
from src.utils import load_model, predict_mask, predict_tumor_class
from src.visualization import draw_tumor_bbox, draw_tumor_overlay

logger = logging.getLogger("uvicorn.error")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

SEGMENTATION_WEIGHTS = MODELS_DIR / "best_model_seg.pth"
CLASSIFICATION_WEIGHTS = MODELS_DIR / "best_model_classif.pth"

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}

app = FastAPI(
    title="Brain MRI Analysis API",
    description="Tumor classification + segmentation for brain MRI scans (BRISC 2025).",
    version="1.0.0",
)

# Allow the Streamlit app (and local dev tools) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy, cached model loading — the server must start even without weights.
# ---------------------------------------------------------------------------
_segmentation_model: torch.nn.Module | None = None
_classification_model: torch.nn.Module | None = None
_segmentation_error: str | None = None
_classification_error: str | None = None


def get_segmentation_model() -> torch.nn.Module:
    global _segmentation_model, _segmentation_error
    if _segmentation_model is None:
        if _segmentation_error is not None:
            raise HTTPException(status_code=503, detail=_segmentation_error)
        try:
            _segmentation_model = load_model(
                model_class=SegmentationModel,
                weights_path=SEGMENTATION_WEIGHTS,
                model_kwargs={"in_channels": 1, "out_channels": 1},
                device=DEVICE,
            )
        except Exception as e:
            _segmentation_error = str(e)
            logger.warning("Segmentation model unavailable: %s", e)
            raise HTTPException(status_code=503, detail=_segmentation_error) from e
    return _segmentation_model


def get_classification_model() -> torch.nn.Module:
    global _classification_model, _classification_error
    if _classification_model is None:
        if _classification_error is not None:
            raise HTTPException(status_code=503, detail=_classification_error)
        try:
            _classification_model = load_model(
                model_class=ClassificationModel,
                weights_path=CLASSIFICATION_WEIGHTS,
                model_kwargs={"num_classes": len(CLASSES)},
                device=DEVICE,
            )
        except Exception as e:
            _classification_error = str(e)
            logger.warning("Classification model unavailable: %s", e)
            raise HTTPException(status_code=503, detail=_classification_error) from e
    return _classification_model


async def _read_upload_as_image(file: UploadFile) -> Image.Image:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload a PNG/JPEG/BMP/TIFF image.",
        )
    raw_bytes = await file.read()
    try:
        return Image.open(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.") from e


def _encode_png_base64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _run_segmentation(image: Image.Image) -> SegmentationResponse:
    model = get_segmentation_model()
    result = predict_mask(model=model, image_or_path=image, device=DEVICE)

    mask = result["predicted_mask"].numpy().astype(np.uint8)  # (H, W), model resolution
    original_size = result["original_size"]  # (W, H)

    mask_image = Image.fromarray(mask * 255).resize(original_size, resample=Image.NEAREST)
    mask_resized = (np.array(mask_image) > 0).astype(np.uint8)

    image_rgb = np.array(image.convert("RGB"))
    bbox_result = draw_tumor_bbox(image_rgb, mask_resized)
    overlay = draw_tumor_overlay(bbox_result["image_with_boxes"], mask_resized, alpha=0.35)

    return SegmentationResponse(
        tumor_coverage=result["tumor_coverage"],
        confidence=result["confidence"],
        num_regions=bbox_result["num_regions"],
        mask_png_base64=_encode_png_base64(mask_resized * 255),
        overlay_png_base64=_encode_png_base64(overlay),
    )


def _run_classification(image: Image.Image) -> ClassificationResponse:
    model = get_classification_model()
    result = predict_tumor_class(model=model, image_or_path=image, class_names=CLASSES, device=DEVICE)
    return ClassificationResponse(**result)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {"message": "Brain MRI Analysis API is running. See /docs for usage."}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        segmentation_model_loaded=SEGMENTATION_WEIGHTS.exists(),
        classification_model_loaded=CLASSIFICATION_WEIGHTS.exists(),
        device=DEVICE,
    )


@app.post(
    "/predict/classification",
    response_model=ClassificationResponse,
    responses={503: {"model": ErrorResponse}},
)
async def predict_classification(file: UploadFile = File(...)) -> ClassificationResponse:
    image = await _read_upload_as_image(file)
    return _run_classification(image)


@app.post(
    "/predict/segmentation",
    response_model=SegmentationResponse,
    responses={503: {"model": ErrorResponse}},
)
async def predict_segmentation(file: UploadFile = File(...)) -> SegmentationResponse:
    image = await _read_upload_as_image(file)
    return _run_segmentation(image)


@app.post(
    "/predict/full-analysis",
    response_model=FullAnalysisResponse,
    responses={503: {"model": ErrorResponse}},
)
async def predict_full_analysis(file: UploadFile = File(...)) -> FullAnalysisResponse:
    """
    Runs classification first; segmentation is only run (and returned)
    when a tumor class is predicted, avoiding a wasted forward pass on
    clearly healthy scans.
    """
    image = await _read_upload_as_image(file)
    classification = _run_classification(image)

    segmentation = None
    if classification.has_tumor:
        segmentation = _run_segmentation(image)

    return FullAnalysisResponse(classification=classification, segmentation=segmentation)
