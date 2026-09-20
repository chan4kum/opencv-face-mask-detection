# ADR 0001: Train a small classifier on a public-domain dataset instead of reusing a third-party mask model

**Status:** accepted

**Context.** The candidates found were `prithivMLmods/Face-Mask-Detection` (Apache-2.0, but a SigLIP2-base checkpoint of several hundred MB whose training dataset has **no stated license**) and assorted GitHub/Keras projects with unclear data terms. The Kaggle *Face Mask Detection* dataset (853 images, 4,072 faces) is declared CC0 1.0.

**Decision.** Train a MobileNetV3-Small on the CC0 dataset (`training/`), with image-level splits, a shared crop function, validation-only model selection and calibration, and a single pass over the held-out test split. Export to ONNX with normalisation and temperature inside the graph and commit the 6 MB weights under MIT.

**Consequences.**
- (+) Clear provenance and reproducibility (pinned dataset revision, per-file manifest, seed, metadata JSON).
- (+) Small and fast; honest, fully documented held-out metrics; no license ambiguity in the weights themselves.
- (-) Small data: only 84 `mask_weared_incorrect` training faces, so that class is weak (37.5% recall).
- (-) The dataset's photos are scraped web images; CC0 is as declared by the uploader.
- (-) Retraining needs PyTorch; training on the Apple GPU is not bit-for-bit deterministic.
