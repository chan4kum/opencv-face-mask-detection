from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mask_detection.config import DEFAULT_MASK_MODEL_SHA256
from mask_detection.mask import (
    DEFAULT_CROP_MARGIN,
    INPUT_SIZE,
    LABELS,
    MaskClassifier,
    ModelIntegrityError,
    preprocess,
    sha256_file,
    square_crop,
)
from tests.conftest import DATA, MASK_MODEL, ROOT, SAMPLES


@pytest.fixture(scope="module")
def classifier() -> MaskClassifier:
    return MaskClassifier(MASK_MODEL, expected_sha256=DEFAULT_MASK_MODEL_SHA256)


def sample(name: str) -> tuple[np.ndarray, tuple[float, float, float, float], str]:
    face = SAMPLES[name]["faces"][0]
    return cv2.imread(str(DATA / name)), tuple(float(v) for v in face["box"]), face["label"]  # type: ignore[return-value]


class TestSquareCrop:
    def test_output_is_always_input_size(self) -> None:
        img = np.zeros((300, 400, 3), np.uint8)
        for box in [(100, 100, 50, 80), (0, 0, 40, 40), (350, 250, 60, 60), (-20, -20, 60, 60), (10, 10, 500, 500)]:
            assert square_crop(img, box, 1.3).shape == (INPUT_SIZE, INPUT_SIZE, 3)

    def test_crop_is_centred_on_the_face(self) -> None:
        img = np.zeros((400, 400, 3), np.uint8)
        img[180:220, 180:220] = 255
        crop = square_crop(img, (180, 180, 40, 40), 2.0)
        ys, xs = np.where(crop[:, :, 0] == 255)
        assert abs(xs.mean() - INPUT_SIZE / 2) < 4 and abs(ys.mean() - INPUT_SIZE / 2) < 4

    def test_out_of_image_area_is_edge_replicated_not_black(self) -> None:
        assert square_crop(np.full((100, 100, 3), 200, np.uint8), (0, 0, 40, 40), 2.0).min() == 200


def test_preprocess_is_rgb_unit_range_nchw() -> None:
    bgr = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
    bgr[:, :, 0], bgr[:, :, 2] = 255, 0  # pure blue in BGR
    x = preprocess(bgr)
    assert x.shape == (1, 3, INPUT_SIZE, INPUT_SIZE) and x.dtype == np.float32
    assert float(x[0, 2].mean()) == pytest.approx(1.0) and float(x[0, 0].mean()) == 0.0  # blue is the LAST RGB channel


def test_model_metadata_matches_serving_constants() -> None:
    """Guard against train/serve skew: the exported model's recorded settings must equal what the service uses."""
    meta = json.loads((ROOT / "models" / "mask_classifier.json").read_text())
    assert tuple(meta["labels"]) == LABELS and meta["input_size"] == INPUT_SIZE
    assert meta["crop_margin"] == DEFAULT_CROP_MARGIN
    assert meta["onnx_sha256"] == DEFAULT_MASK_MODEL_SHA256 == sha256_file(MASK_MODEL)
    assert meta["onnx_parity_max_abs_diff"] < 1e-4


class TestClassifier:
    @pytest.mark.parametrize(
        "name", [n for n, v in SAMPLES.items() if v["faces"][0]["label"] != "mask_weared_incorrect"]
    )
    def test_ground_truth_samples_from_the_held_out_split(self, classifier: MaskClassifier, name: str) -> None:
        img, box, truth = sample(name)
        est = classifier.classify_face(img, box)
        assert est.label == truth and est.confidence > 0.7 and not est.uncertain

    def test_incorrect_sample_returns_a_valid_estimate(self, classifier: MaskClassifier) -> None:
        """`mask_weared_incorrect` has ~37% recall (docs/EVALUATION.md), so only the output contract is asserted."""
        name = next(n for n, v in SAMPLES.items() if v["faces"][0]["label"] == "mask_weared_incorrect")
        img, box, _ = sample(name)
        est = classifier.classify_face(img, box)
        assert est.label in LABELS and sum(est.probabilities) == pytest.approx(1.0, abs=1e-4)

    def test_probabilities_are_a_distribution_over_the_labels(self, classifier: MaskClassifier) -> None:
        img, box, _ = sample(next(iter(SAMPLES)))
        est = classifier.classify_face(img, box)
        assert len(est.probabilities) == len(LABELS) and est.confidence == pytest.approx(max(est.probabilities))

    def test_is_deterministic(self, classifier: MaskClassifier) -> None:
        img, box, _ = sample(next(iter(SAMPLES)))
        assert classifier.classify_face(img, box) == classifier.classify_face(img, box)

    def test_uncertain_flag_follows_the_threshold(self) -> None:
        img, box, _ = sample(next(iter(SAMPLES)))
        assert (
            MaskClassifier(MASK_MODEL, expected_sha256=None, min_confidence=0.0).classify_face(img, box).uncertain
            is False
        )
        assert (
            MaskClassifier(MASK_MODEL, expected_sha256=None, min_confidence=1.01).classify_face(img, box).uncertain
            is True
        )

    def test_noise_does_not_crash(self, classifier: MaskClassifier) -> None:
        noise = np.random.default_rng(0).integers(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        assert classifier.classify_crop(noise).label in LABELS


class TestModelIntegrity:
    def test_missing_model(self, tmp_path: Path) -> None:
        with pytest.raises(ModelIntegrityError, match="not found"):
            MaskClassifier(tmp_path / "nope.onnx", expected_sha256=None)

    def test_tampered_model_is_refused(self, tmp_path: Path) -> None:
        bad = tmp_path / "m.onnx"
        bad.write_bytes(MASK_MODEL.read_bytes() + b"\x00")
        with pytest.raises(ModelIntegrityError, match="checksum mismatch"):
            MaskClassifier(bad, expected_sha256=DEFAULT_MASK_MODEL_SHA256)
