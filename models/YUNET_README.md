# Model provenance

| | |
|---|---|
| File | `face_detection_yunet_2026may.onnx` |
| Model | YuNet, a light-weight face detector (Wu et al., *Machine Intelligence Research*, 2023) |
| Source | https://github.com/opencv/opencv_zoo/tree/26cc381e4d2594bb9f47a26eb8fd96c94a13660d/models/mask_detection_yunet |
| Download | `https://github.com/opencv/opencv_zoo/raw/26cc381e4d2594bb9f47a26eb8fd96c94a13660d/models/mask_detection_yunet/face_detection_yunet_2026may.onnx` (Git LFS: `raw.githubusercontent.com` URLs return a pointer file, not the model) |
| Upstream commit | `26cc381e4d2594bb9f47a26eb8fd96c94a13660d` (2026-05-22) |
| SHA-256 | `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0` (matches the `oid` in upstream's LFS pointer) |
| License | MIT (see `LICENSE`, Copyright (c) 2020 Shiqi Yu) |
| Input | `input` `[1, 3, H, W]` float32, BGR, raw 0-255 values, H and W multiples of 32 (dynamic) |
| Outputs | 12 tensors: `cls`, `obj`, `bbox`, `kps` for strides 8, 16, 32 |

The service refuses to start if the file's SHA-256 differs from `MD_MODEL_SHA256`
(default: the value above). Verify manually with `uv run mask-detection verify-model`.

To upgrade the model: replace the file, update the checksum here and in
`src/mask_detection/config.py`, and re-run the test-suite (including the OpenCV parity test).
