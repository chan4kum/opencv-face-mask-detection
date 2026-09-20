from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mask_detection.config import DEFAULT_MODEL_SHA256
from mask_detection.detector import ModelIntegrityError, YuNetDetector, nms, sha256_file
from tests.conftest import DATA, MODEL


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(a[0], b[0])), max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def load(name: str) -> np.ndarray:
    img = cv2.imread(str(DATA / name))
    assert img is not None
    return img


def test_detects_single_face(detector: YuNetDetector) -> None:
    faces = detector.detect(load("astronaut.jpg"))
    assert len(faces) == 1
    f = faces[0]
    assert f.score > 0.9
    assert 150 < f.x < 210 and 40 < f.y < 90  # known position of the face in the fixture
    assert len(f.landmarks) == 5


def test_detects_two_faces_with_correct_offsets(detector: YuNetDetector) -> None:
    faces = sorted(detector.detect(load("two_faces.jpg")), key=lambda f: f.x)
    assert len(faces) == 2
    assert faces[1].x - faces[0].x == pytest.approx(512, abs=3)  # second copy is 512px to the right


def test_no_face_in_noise(detector: YuNetDetector) -> None:
    assert detector.detect(load("noise.png")) == []


def test_matches_opencv_reference_implementation(detector: YuNetDetector) -> None:
    """Our ONNX Runtime decoder must agree with OpenCV's own YuNet implementation."""
    if not hasattr(cv2, "FaceDetectorYN"):
        pytest.skip("cv2.FaceDetectorYN not available")
    img = load("two_faces.jpg")
    ref = cv2.FaceDetectorYN.create(str(MODEL), "", (img.shape[1], img.shape[0]), 0.7, 0.3, 5000)
    _, ref_faces = ref.detect(img)
    ours = detector.detect(img)
    assert ref_faces is not None and len(ref_faces) == len(ours)
    for r in ref_faces:
        best = max(iou(tuple(r[:4]), (f.x, f.y, f.width, f.height)) for f in ours)  # type: ignore[arg-type]
        assert best > 0.95
    assert max(f.score for f in ours) == pytest.approx(float(max(r[14] for r in ref_faces)), abs=0.01)


@pytest.mark.parametrize("size", [(256, 256), (300, 220), (1023, 511), (2048, 2048)])
def test_scale_invariance_of_coordinates(detector: YuNetDetector, size: tuple[int, int]) -> None:
    """Boxes are returned in original-image pixels regardless of internal resizing/padding."""
    base = load("astronaut.jpg")
    ref = detector.detect(base)[0]
    resized = cv2.resize(base, size)
    sx, sy = size[0] / base.shape[1], size[1] / base.shape[0]
    faces = detector.detect(resized)
    assert faces, f"no face at {size}"
    expected = (ref.x * sx, ref.y * sy, ref.width * sx, ref.height * sy)
    assert max(iou(expected, (f.x, f.y, f.width, f.height)) for f in faces) > 0.7


def test_downscaling_for_large_inputs_keeps_original_coordinates() -> None:
    small_side = YuNetDetector(MODEL, expected_sha256=DEFAULT_MODEL_SHA256, max_side=256)
    base = load("astronaut.jpg")
    big = cv2.resize(base, (2048, 2048))
    faces = small_side.detect(big)
    assert faces
    f = faces[0]
    assert f.x >= 0 and f.x + f.width <= 2048 and f.y >= 0 and f.y + f.height <= 2048
    assert f.x == pytest.approx(178 * 4, rel=0.12)


def test_boxes_are_clipped_to_image(detector: YuNetDetector) -> None:
    cropped = load("astronaut.jpg")[:, 190:]  # cuts the face in half
    for f in detector.detect(cropped):
        assert f.x >= 0 and f.y >= 0
        assert f.x + f.width <= cropped.shape[1] + 1e-3
        assert f.y + f.height <= cropped.shape[0] + 1e-3


def test_threshold_and_max_faces_are_respected() -> None:
    strict = YuNetDetector(MODEL, expected_sha256=DEFAULT_MODEL_SHA256, score_threshold=0.9999)
    assert strict.detect(load("astronaut.jpg")) == []
    capped = YuNetDetector(MODEL, expected_sha256=DEFAULT_MODEL_SHA256, max_faces=1)
    assert len(capped.detect(load("two_faces.jpg"))) == 1


@pytest.mark.parametrize(
    "bad",
    [np.zeros((10, 10), np.uint8), np.zeros((10, 10, 4), np.uint8), np.zeros((10, 10, 3), np.float32)],
)
def test_rejects_wrong_array_shapes(detector: YuNetDetector, bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="HxWx3 uint8"):
        detector.detect(bad)


def test_tiny_image_does_not_crash(detector: YuNetDetector) -> None:
    assert detector.detect(np.zeros((1, 1, 3), np.uint8)) == []


def test_checksum_mismatch_refuses_to_load() -> None:
    with pytest.raises(ModelIntegrityError, match="checksum mismatch"):
        YuNetDetector(MODEL, expected_sha256="0" * 64)


def test_missing_model_refuses_to_load(tmp_path: Path) -> None:
    with pytest.raises(ModelIntegrityError, match="not found"):
        YuNetDetector(tmp_path / "nope.onnx", expected_sha256=None)


def test_pinned_checksum_matches_vendored_model() -> None:
    assert sha256_file(MODEL) == DEFAULT_MODEL_SHA256


class TestNms:
    def test_empty(self) -> None:
        assert nms(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0.3) == []

    def test_suppresses_overlap_keeps_highest(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 10, 10]], np.float32)
        scores = np.array([0.8, 0.9, 0.7], np.float32)
        assert nms(boxes, scores, 0.3) == [1, 2]

    def test_keeps_all_when_disjoint(self) -> None:
        boxes = np.array([[0, 0, 5, 5], [10, 10, 5, 5]], np.float32)
        assert sorted(nms(boxes, np.array([0.5, 0.6], np.float32), 0.3)) == [0, 1]

    def test_zero_area_boxes_do_not_divide_by_zero(self) -> None:
        boxes = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], np.float32)
        assert len(nms(boxes, np.array([0.5, 0.4], np.float32), 0.3)) >= 1
