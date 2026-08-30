"""
Pydantic schemas for the Brain MRI Analysis API.

These define the exact shape of every JSON response the API returns,
so the Streamlit frontend (or any other client) can rely on a stable
contract instead of a raw dict.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' if the API is reachable.")
    segmentation_model_loaded: bool
    classification_model_loaded: bool
    device: str


class ClassificationResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: float = Field(..., description="Confidence of the predicted class, 0-100.")
    probabilities: dict[str, float] = Field(..., description="Class name -> probability (0-100).")
    has_tumor: bool


class SegmentationResponse(BaseModel):
    tumor_coverage: float = Field(..., description="Percent of pixels predicted as tumor, 0-100.")
    confidence: float = Field(..., description="Mean tumor-class probability in the predicted region, 0-100.")
    num_regions: int = Field(..., description="Number of distinct tumor regions detected.")
    mask_png_base64: str = Field(..., description="Predicted binary mask, PNG-encoded, base64 string.")
    overlay_png_base64: str = Field(..., description="Original image with tumor overlay + bounding boxes, PNG-encoded, base64 string.")


class FullAnalysisResponse(BaseModel):
    classification: ClassificationResponse
    segmentation: SegmentationResponse | None = Field(
        None, description="Omitted when the classifier predicts no tumor."
    )


class ErrorResponse(BaseModel):
    detail: str
