# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/), versioning: [SemVer](https://semver.org/).

## [Unreleased]

## [1.0.0]

### Changed
- Rebuilt from a Keras/OpenCV webcam script (with no included model) into a production-grade, reproducible service.

### Added
- Reproducible training pipeline (`training/`): pinned CC0 dataset with per-file manifest, image-level splits, validation-only selection and calibration, ONNX export with parity check, generated evaluation report.
- MobileNetV3-Small mask classifier (`with_mask`, `without_mask`, `mask_weared_incorrect`) with `uncertain` and `low_resolution` flags; YuNet detection with small-image upscaling chosen on validation images.
- `POST /v1/analyze`, `/v1/analyze/annotated`, async `/v1/jobs` on NATS JetStream + S3; hashed API keys, request limits, load shedding.
- Prometheus metrics, OpenTelemetry tracing, Grafana dashboard, alerts; hardened Docker image, Compose stack, Helm chart, CI/CD.
- Model card and evaluation report with held-out results, per-class and per-size breakdowns and end-to-end detection recall.
- Test guard against train/serve skew (model metadata vs serving constants) and end-to-end tests on CC0 ground-truth images.
