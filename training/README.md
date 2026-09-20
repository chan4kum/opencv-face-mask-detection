# Training

Reproduces `models/mask_classifier.onnx` from a public-domain dataset. Not part of the service package; needs PyTorch.

```bash
uv venv --python 3.12 .train-venv && source .train-venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install huggingface_hub onnx onnxscript onnxruntime opencv-python-headless pillow numpy scikit-learn
uv pip install -e .                                   # the service package (shared crop code)
python training/fetch_dataset.py /tmp/maskdata        # pinned revision, verified against training/dataset_manifest.json
python training/train.py --data /tmp/maskdata --out models --report docs/EVALUATION.md
```

* **Data:** Kaggle "Face Mask Detection" (853 images, 4,072 faces, CC0 1.0), mirrored at `hmnshudhmn24/face-mask-detection` (pinned revision in `fetch_dataset.py`).
* **Splits:** by *image* (70/15/15, seed 42), never by face, so faces from one photo cannot leak between train and test.
* **Crops:** the exact `mask_detection.mask.square_crop` used at serving time, so training and serving cannot drift.
* **Model:** ImageNet-pretrained MobileNetV3-Small (torchvision, BSD-3), 128 px input, 3 classes.
* **Calibration:** a softmax temperature is fitted on the validation split only and baked into the ONNX graph.
* **Reporting:** the held-out test split is touched once, at the end: overall and per-class metrics, metrics by face size, expected calibration error, and an end-to-end evaluation with the YuNet detector.
