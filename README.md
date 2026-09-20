# Face Mask Detection

Trains a MobileNetV2 classifier on a with-mask / without-mask dataset, then detects mask usage on live video.

Part of a series of beginner-friendly OpenCV projects.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# dataset/ must contain with_mask/ and without_mask/ image folders
python train.py
python main.py
```

Press `q` to quit any live window. Omit input arguments to use your webcam where supported.

## License

MIT
