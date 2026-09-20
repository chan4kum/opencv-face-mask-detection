"""Draw detections onto an image (used by the CLI and the ``/v1/detect/annotated`` endpoint)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mask_detection.detector import Face
    from mask_detection.mask import MaskEstimate

_COLOURS = {"with_mask": (0, 200, 0), "without_mask": (0, 0, 255), "mask_weared_incorrect": (0, 165, 255)}
_LANDMARK_COLORS = ((255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255))


def draw_faces(image_bgr: NDArray[np.uint8], faces: Sequence[tuple[Face, MaskEstimate]]) -> NDArray[np.uint8]:
    """Return a copy of the image with boxes and age-group labels drawn (`?` marks an uncertain estimate)."""
    out = image_bgr.copy()
    thickness = max(1, round(min(out.shape[:2]) / 300))
    for face, est in faces:
        colour = (128, 128, 128) if est.uncertain else _COLOURS[est.label]
        x1, y1 = round(face.x), round(face.y)
        x2, y2 = round(face.x + face.width), round(face.y + face.height)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        cv2.putText(
            out,
            f"{est.label.replace('mask_weared_incorrect', 'incorrect')}{' ?' if est.uncertain else ''}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * thickness,
            colour,
            thickness,
            cv2.LINE_AA,
        )
        for (px, py), color in zip(face.landmarks, _LANDMARK_COLORS, strict=True):
            cv2.circle(out, (round(px), round(py)), thickness + 1, color, -1, cv2.LINE_AA)
    return out


def encode_jpeg(image_bgr: NDArray[np.uint8], quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:  # pragma: no cover - encoder failure on a valid array is not expected
        raise RuntimeError("JPEG encoding failed")
    return bytes(buf.tobytes())
