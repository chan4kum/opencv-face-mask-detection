from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from mask_detection.errors import InvalidImageError, UnsupportedMediaTypeError
from mask_detection.imaging import probe_image, validate_and_decode

LIMIT = 40_000_000


def encode(img: Image.Image, fmt: str, **kw: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "BMP"])
def test_supported_formats_decode_to_bgr(fmt: str) -> None:
    data = encode(Image.new("RGB", (64, 48), (255, 0, 0)), fmt)
    out = validate_and_decode(data, max_pixels=LIMIT)
    assert out.shape == (48, 64, 3) and out.dtype == np.uint8
    assert out[24, 32, 2] > 200 and out[24, 32, 0] < 60  # red in BGR order


def test_grayscale_and_palette_and_rgba_become_3_channel() -> None:
    for mode in ("L", "P", "RGBA"):
        out = validate_and_decode(encode(Image.new(mode, (20, 20)), "PNG"), max_pixels=LIMIT)
        assert out.shape == (20, 20, 3)


def test_exif_orientation_is_applied() -> None:
    img = Image.new("RGB", (40, 20), (0, 255, 0))  # landscape
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display -> portrait
    out = validate_and_decode(encode(img, "JPEG", exif=exif), max_pixels=LIMIT)
    assert out.shape[:2] == (40, 20)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\xff\xd8\xff\xe0" + b"\x00" * 10, b"GIF89a" + b"\x00" * 20])
def test_garbage_is_rejected(data: bytes) -> None:
    with pytest.raises((InvalidImageError, UnsupportedMediaTypeError)):
        validate_and_decode(data, max_pixels=LIMIT)


def test_unsupported_but_valid_format_is_415() -> None:
    with pytest.raises(UnsupportedMediaTypeError) as exc:
        probe_image(encode(Image.new("RGB", (8, 8)), "GIF"), max_pixels=LIMIT)
    assert exc.value.status == 415


def test_pixel_budget_checked_from_header_before_decoding() -> None:
    # A 20000x20000 all-black PNG is tiny on the wire but 400 MP decoded: must be refused from the header.
    data = encode(Image.new("1", (20000, 20000)), "PNG", optimize=False)
    assert len(data) < 5_000_000
    with pytest.raises(InvalidImageError, match="limit"):
        probe_image(data, max_pixels=LIMIT)


def test_truncated_file_is_rejected_or_decoded_safely() -> None:
    data = encode(Image.effect_noise((256, 256), 60).convert("RGB"), "PNG")
    with pytest.raises(InvalidImageError):
        validate_and_decode(data[: len(data) // 3], max_pixels=LIMIT)
