"""YuNet face detector running on ONNX Runtime.

The model outputs raw per-anchor predictions on three feature-map strides (8, 16, 32). This module
implements the decoding and non-maximum suppression itself so inference has no dependency on
``cv2.dnn`` and can run on any ONNX Runtime execution provider.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

STRIDES: Final = (8, 16, 32)
PAD_MULTIPLE: Final = 32  # the largest stride; padded dims must divide evenly


class ModelIntegrityError(RuntimeError):
    """The model file is missing or does not match the pinned checksum."""


@dataclass(frozen=True, slots=True)
class Face:
    """One detection in original-image pixel coordinates."""

    x: float
    y: float
    width: float
    height: float
    score: float
    # Five landmarks: right eye, left eye, nose tip, right mouth corner, left mouth corner.
    landmarks: tuple[tuple[float, float], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nms(boxes_xywh: NDArray[np.float32], scores: NDArray[np.float32], iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices ordered by descending score."""
    if len(boxes_xywh) == 0:
        return []
    x1 = boxes_xywh[:, 0]
    y1 = boxes_xywh[:, 1]
    x2 = x1 + boxes_xywh[:, 2]
    y2 = y1 + boxes_xywh[:, 3]
    areas = boxes_xywh[:, 2] * boxes_xywh[:, 3]
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        iw = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        ih = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = iw * ih
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]
    return keep


class YuNetDetector:
    """Thread-safe (``InferenceSession.run`` is re-entrant) YuNet face detector."""

    def __init__(
        self,
        model_path: Path,
        *,
        expected_sha256: str | None,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        max_faces: int = 500,
        max_side: int = 960,
        upscale_to: int = 0,
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
        providers: list[str] | None = None,
    ) -> None:
        if not model_path.is_file():
            raise ModelIntegrityError(f"model file not found: {model_path}")
        if expected_sha256 is not None:
            actual = sha256_file(model_path)
            if actual != expected_sha256:
                raise ModelIntegrityError(
                    f"model checksum mismatch for {model_path}: expected {expected_sha256}, got {actual}"
                )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = inter_op_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # errors only; we log through structlog
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=providers or ["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_faces = max_faces
        self.max_side = max_side
        self.upscale_to = upscale_to  # images whose longest side is below this are upscaled to it (0 = never)
        self.providers = self._session.get_providers()

    def warmup(self) -> None:
        """Run one dummy inference so the first real request does not pay initialisation cost."""
        self.detect(np.zeros((self.max_side, self.max_side, 3), dtype=np.uint8))

    def detect(self, image_bgr: NDArray[np.uint8]) -> list[Face]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
            raise ValueError("expected an HxWx3 uint8 BGR image")
        h0, w0 = image_bgr.shape[:2]
        if h0 == 0 or w0 == 0:
            raise ValueError("empty image")

        longest = max(h0, w0)
        scale = min(1.0, self.max_side / longest)
        if scale == 1.0 and 0 < longest < self.upscale_to:
            scale = min(
                self.upscale_to / longest, 4.0
            )  # small images: enlarge so small faces reach the detector's sweet spot
        if scale != 1.0:
            new_w, new_h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
            resized = cv2.resize(
                image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            )
        else:
            resized = image_bgr
        h, w = resized.shape[:2]
        pad_h = -(-h // PAD_MULTIPLE) * PAD_MULTIPLE
        pad_w = -(-w // PAD_MULTIPLE) * PAD_MULTIPLE
        padded = cv2.copyMakeBorder(resized, 0, pad_h - h, 0, pad_w - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # YuNet expects raw 0..255 BGR pixels (no mean/std normalisation), NCHW float32.
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        outputs = self._session.run(None, {self._input_name: blob})
        boxes, scores, kps = self._decode(outputs, pad_w)

        # Discard detections that fall entirely in the padding, then map back to the original size.
        keep_score = scores >= self.score_threshold
        boxes, scores, kps = boxes[keep_score], scores[keep_score], kps[keep_score]
        keep = nms(boxes, scores, self.nms_threshold)[: self.max_faces]

        inv = 1.0 / scale
        faces: list[Face] = []
        for i in keep:
            x, y, bw, bh = (boxes[i] * inv).tolist()
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(w0), x + bw), min(float(h0), y + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            pts = tuple((float(px * inv), float(py * inv)) for px, py in kps[i].reshape(5, 2))
            faces.append(Face(x1, y1, x2 - x1, y2 - y1, float(scores[i]), pts))
        return faces

    @staticmethod
    def _decode(
        outputs: list[NDArray[np.float32]], pad_w: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Decode the 12 raw outputs (cls, obj, bbox, kps for each stride) to pixel coordinates."""
        cls_all, obj_all, bbox_all, kps_all = outputs[0:3], outputs[3:6], outputs[6:9], outputs[9:12]
        boxes, scores, kps = [], [], []
        for idx, stride in enumerate(STRIDES):
            cls = np.clip(cls_all[idx][0, :, 0], 0.0, 1.0)
            obj = np.clip(obj_all[idx][0, :, 0], 0.0, 1.0)
            bbox = bbox_all[idx][0]
            kp = kps_all[idx][0]
            cols = pad_w // stride
            anchor = np.arange(cls.shape[0])
            col = (anchor % cols).astype(np.float32)
            row = (anchor // cols).astype(np.float32)

            cx = (col + bbox[:, 0]) * stride
            cy = (row + bbox[:, 1]) * stride
            bw = np.exp(bbox[:, 2]) * stride
            bh = np.exp(bbox[:, 3]) * stride
            boxes.append(np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1))
            scores.append(np.sqrt(cls * obj))

            pts = np.empty((cls.shape[0], 10), dtype=np.float32)
            pts[:, 0::2] = (kp[:, 0::2] + col[:, None]) * stride
            pts[:, 1::2] = (kp[:, 1::2] + row[:, None]) * stride
            kps.append(pts)
        return (
            np.concatenate(boxes).astype(np.float32),
            np.concatenate(scores).astype(np.float32),
            np.concatenate(kps).astype(np.float32),
        )
