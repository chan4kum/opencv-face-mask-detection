"""Face-mask state classification on face crops.

A small MobileNetV3 classifier (trained in ``training/`` on a CC0 dataset, exported to ONNX with the ImageNet
normalisation and a fitted softmax temperature *inside* the graph) labels each detected face as
``with_mask``, ``without_mask`` or ``mask_weared_incorrect``. The crop and preprocessing functions here are
imported by the training code, so training and serving cannot drift apart.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

LABELS: Final = ("with_mask", "without_mask", "mask_weared_incorrect")
INPUT_SIZE: Final = 128
DEFAULT_CROP_MARGIN: Final = 1.3


class ModelIntegrityError(RuntimeError):
    """The model file is missing or does not match the pinned checksum."""


@dataclass(frozen=True, slots=True)
class MaskEstimate:
    label: str
    confidence: float  # calibrated probability of ``label`` (temperature fitted on a held-out validation split)
    uncertain: bool  # confidence below the configured threshold
    probabilities: tuple[float, ...]  # one per LABELS entry, sums to 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def square_crop(
    image_bgr: NDArray[np.uint8], box_xywh: tuple[float, float, float, float], margin: float, size: int = INPUT_SIZE
) -> NDArray[np.uint8]:
    """Square crop of side ``max(w, h) * margin`` around the box (edge-replicated outside the image), resized."""
    x, y, w, h = box_xywh
    side = max(w, h) * margin
    cx, cy = x + w / 2.0, y + h / 2.0
    x0, y0 = round(cx - side / 2.0), round(cy - side / 2.0)
    s = max(2, round(side))
    ih, iw = image_bgr.shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x0 + s - iw), max(0, y0 + s - ih)
    if pad_l or pad_t or pad_r or pad_b:
        image_bgr = cast(
            "NDArray[np.uint8]", cv2.copyMakeBorder(image_bgr, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        )
        x0, y0 = x0 + pad_l, y0 + pad_t
    crop = image_bgr[y0 : y0 + s, x0 : x0 + s]
    interp = cv2.INTER_AREA if s > size else cv2.INTER_LINEAR
    return cast("NDArray[np.uint8]", cv2.resize(crop, (size, size), interpolation=interp))


def preprocess(crop_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
    """BGR uint8 -> RGB float32 in [0, 1], NCHW. (ImageNet mean/std normalisation happens inside the ONNX graph.)"""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)


class MaskClassifier:
    """Thread-safe (``InferenceSession.run`` is re-entrant) mask-state classifier."""

    def __init__(
        self,
        model_path: Path,
        *,
        expected_sha256: str | None,
        min_confidence: float = 0.7,
        crop_margin: float = DEFAULT_CROP_MARGIN,
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
        providers: list[str] | None = None,
    ) -> None:
        if not model_path.is_file():
            raise ModelIntegrityError(f"mask model not found: {model_path}")
        if expected_sha256 is not None:
            actual = sha256_file(model_path)
            if actual != expected_sha256:
                raise ModelIntegrityError(
                    f"mask model checksum mismatch for {model_path}: expected {expected_sha256}, got {actual}"
                )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = inter_op_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=providers or ["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self.min_confidence = min_confidence
        self.crop_margin = crop_margin
        self.providers = self._session.get_providers()

    def warmup(self) -> None:
        self.classify_crop(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))

    def classify_crop(self, crop_bgr: NDArray[np.uint8]) -> MaskEstimate:
        probs = self._session.run(None, {self._input: preprocess(crop_bgr)})[0][0].astype(np.float64)
        probs = probs / probs.sum()
        top = int(probs.argmax())
        conf = float(probs[top])
        return MaskEstimate(
            label=LABELS[top],
            confidence=conf,
            uncertain=conf < self.min_confidence,
            probabilities=tuple(float(p) for p in probs),
        )

    def classify_face(self, image_bgr: NDArray[np.uint8], box_xywh: tuple[float, float, float, float]) -> MaskEstimate:
        return self.classify_crop(square_crop(image_bgr, box_xywh, self.crop_margin))
