"""Public API models (also the source of the generated OpenAPI document)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mask_detection.detector import Face
from mask_detection.jobs import JobStatus
from mask_detection.mask import LABELS, MaskEstimate


class Box(BaseModel):
    x: float = Field(description="Left edge in pixels of the (EXIF-upright) image")
    y: float = Field(description="Top edge in pixels")
    width: float
    height: float


class Point(BaseModel):
    x: float
    y: float


class Landmarks(BaseModel):
    right_eye: Point
    left_eye: Point
    nose: Point
    right_mouth: Point
    left_mouth: Point


class MaskOut(BaseModel):
    """Mask state of one face. `mask_weared_incorrect` is the weakest class (see docs/EVALUATION.md)."""

    label: str = Field(description="with_mask, without_mask or mask_weared_incorrect")
    confidence: float = Field(
        ge=0, le=1, description="Probability of the label (temperature fitted on a validation split)"
    )
    uncertain: bool = Field(description="confidence is below the configured threshold: treat the label as unreliable")
    low_resolution: bool = Field(
        description="the face is smaller than the configured size: detection and labels are less reliable"
    )
    probabilities: dict[str, float]

    @classmethod
    def from_estimate(cls, e: MaskEstimate, *, low_resolution: bool) -> MaskOut:
        return cls(
            label=e.label,
            confidence=round(e.confidence, 4),
            uncertain=e.uncertain,
            low_resolution=low_resolution,
            probabilities={n: round(p, 4) for n, p in zip(LABELS, e.probabilities, strict=True)},
        )


class FaceOut(BaseModel):
    box: Box
    score: float = Field(ge=0, le=1, description="Face detection confidence")
    landmarks: Landmarks
    mask: MaskOut

    @classmethod
    def from_face(cls, face: Face, estimate: MaskEstimate, *, low_resolution_px: int) -> FaceOut:
        re, le, nose, rm, lm = (Point(x=x, y=y) for x, y in face.landmarks)
        return cls(
            box=Box(x=face.x, y=face.y, width=face.width, height=face.height),
            score=face.score,
            landmarks=Landmarks(right_eye=re, left_eye=le, nose=nose, right_mouth=rm, left_mouth=lm),
            mask=MaskOut.from_estimate(estimate, low_resolution=min(face.width, face.height) < low_resolution_px),
        )


class ImageInfo(BaseModel):
    width: int
    height: int


class ModelInfo(BaseModel):
    name: str
    sha256: str
    execution_providers: list[str]


class ModelsInfo(BaseModel):
    face_detector: ModelInfo
    mask_classifier: ModelInfo


class Summary(BaseModel):
    """Counts over the detected faces. Not a compliance verdict: detection misses faces (see docs/EVALUATION.md)."""

    faces: int
    with_mask: int
    without_mask: int
    mask_weared_incorrect: int
    uncertain: int


class AnalysisResult(BaseModel):
    image: ImageInfo
    faces: list[FaceOut]
    summary: Summary
    inference_ms: float = Field(description="Server-side time for detection and mask analysis (no network/queueing)")
    models: ModelsInfo


class AnalysisResponse(AnalysisResult):
    request_id: str


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float
    attempts: int
    result: AnalysisResult | None = None
    error: str | None = None


class ProblemDetails(BaseModel):
    """RFC 9457 problem document."""

    type: str
    title: str
    status: int
    detail: str
    instance: str | None = None
    request_id: str | None = None
    errors: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, bool] = Field(default_factory=dict)
    version: str | None = None
