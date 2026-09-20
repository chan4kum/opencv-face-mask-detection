"""Production-grade face mask detection service built on YuNet, ONNX Runtime and OpenCV."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("opencv-face-mask-detection")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
