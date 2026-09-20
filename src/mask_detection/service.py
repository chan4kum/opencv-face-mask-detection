"""Inference service shared by the HTTP API and the queue workers."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import anyio
import numpy as np

from mask_detection.detector import Face, YuNetDetector
from mask_detection.errors import AppError, OverloadedError
from mask_detection.imaging import validate_and_decode
from mask_detection.mask import MaskClassifier, MaskEstimate
from mask_detection.metrics import (
    FACES_PER_IMAGE,
    IMAGES_REJECTED,
    INFERENCE_DURATION,
    INFERENCE_REJECTED,
    MASK_PREDICTIONS,
)
from mask_detection.schemas import AnalysisResult, FaceOut, ImageInfo, ModelInfo, ModelsInfo, Summary
from mask_detection.telemetry import get_tracer

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mask_detection.config import Settings

FACE_MODEL_NAME = "yunet-2026may"
MASK_MODEL_NAME = "mobilenetv3-small-mask-cc0"


class InferenceService:
    """Bounded-concurrency wrapper: validate -> decode -> detect faces -> classify mask state, off the event loop."""

    def __init__(self, detector: YuNetDetector, classifier: MaskClassifier, settings: Settings) -> None:
        self._detector = detector
        self._classifier = classifier
        self._settings = settings
        self._slots = asyncio.Semaphore(settings.max_concurrent_inference)
        self._models = ModelsInfo(
            face_detector=ModelInfo(
                name=FACE_MODEL_NAME, sha256=settings.model_sha256, execution_providers=list(detector.providers)
            ),
            mask_classifier=ModelInfo(
                name=MASK_MODEL_NAME, sha256=settings.mask_model_sha256, execution_providers=list(classifier.providers)
            ),
        )

    @property
    def models(self) -> ModelsInfo:
        return self._models

    async def analyze_bytes(
        self, data: bytes, *, source: str
    ) -> tuple[AnalysisResult, NDArray[np.uint8], list[tuple[Face, MaskEstimate]]]:
        """Validate, detect faces and classify mask state. Sheds load (503) instead of queueing without bound."""
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=self._settings.inference_queue_timeout_s)
        except TimeoutError:
            INFERENCE_REJECTED.labels(reason="queue_timeout").inc()
            raise OverloadedError("inference capacity exhausted, retry shortly", headers={"Retry-After": "1"}) from None
        try:
            with get_tracer().start_as_current_span("analyze") as span:
                try:
                    image, results, elapsed = await anyio.to_thread.run_sync(self._process, data)
                except AppError as exc:
                    IMAGES_REJECTED.labels(reason=exc.code).inc()
                    raise
                span.set_attribute("faces.count", len(results))
                span.set_attribute("image.width", image.shape[1])
                span.set_attribute("image.height", image.shape[0])
        finally:
            self._slots.release()

        INFERENCE_DURATION.labels(source=source).observe(elapsed)
        FACES_PER_IMAGE.observe(len(results))
        for _, est in results:
            MASK_PREDICTIONS.labels(label=est.label, uncertain=str(est.uncertain).lower()).inc()
        result = AnalysisResult(
            image=ImageInfo(width=image.shape[1], height=image.shape[0]),
            faces=[FaceOut.from_face(f, est, low_resolution_px=self._settings.low_resolution_px) for f, est in results],
            summary=Summary(
                faces=len(results),
                with_mask=sum(e.label == "with_mask" for _, e in results),
                without_mask=sum(e.label == "without_mask" for _, e in results),
                mask_weared_incorrect=sum(e.label == "mask_weared_incorrect" for _, e in results),
                uncertain=sum(e.uncertain for _, e in results),
            ),
            inference_ms=round(elapsed * 1000, 3),
            models=self._models,
        )
        return result, image, results

    def _process(self, data: bytes) -> tuple[NDArray[np.uint8], list[tuple[Face, MaskEstimate]], float]:
        image = validate_and_decode(data, max_pixels=self._settings.max_image_pixels)
        start = time.perf_counter()
        faces = self._detector.detect(image)
        results = [(f, self._classifier.classify_face(image, (f.x, f.y, f.width, f.height))) for f in faces]
        return image, results, time.perf_counter() - start
