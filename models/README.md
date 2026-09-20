# Models

| File | What | License | SHA-256 |
|---|---|---|---|
| `mask_classifier.onnx` | MobileNetV3-Small mask-state classifier, trained by `training/train.py` | MIT (this repo); data CC0 (declared); see docs/MODEL_CARD.md | `55da2cf41fc4f2f92445be7c37ea260f64fe1e6e64f03d65af9ad2a7d94210c7` |
| `mask_classifier.json` | Everything needed to audit the model: labels, input size, crop margin, temperature, dataset revision, seed, metrics, ONNX parity | | |
| `face_detection_yunet_2026may.onnx` | YuNet face detector, vendored from OpenCV Zoo (see `YUNET_README.md`) | MIT (`LICENSE`) | `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0` |

The service refuses to start if a file's SHA-256 differs from the pinned value (`MD_MODEL_SHA256`, `MD_MASK_MODEL_SHA256`). Verify with `uv run mask-detection verify-models`.
To replace the classifier: retrain (`training/README.md`), then update `DEFAULT_MASK_MODEL_SHA256` in `src/mask_detection/config.py`; the test-suite fails if the metadata and constants disagree.
