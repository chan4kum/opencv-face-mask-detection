from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from mask_detection.api.app import create_app
from mask_detection.config import DEFAULT_MODEL_SHA256, Settings
from mask_detection.detector import YuNetDetector

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "face_detection_yunet_2026may.onnx"
DATA = Path(__file__).resolve().parent / "data"
MASK_MODEL = ROOT / "models" / "mask_classifier.onnx"
SAMPLES = json.loads((DATA / "samples.json").read_text())  # CC0 images with ground truth (held-out test split)


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "environment": "test",
        "log_json": False,
        "log_level": "WARNING",
        "model_path": MODEL,
    }
    return Settings(**{**base, **overrides})


@pytest.fixture(scope="session")
def detector() -> YuNetDetector:
    return YuNetDetector(MODEL, expected_sha256=DEFAULT_MODEL_SHA256, max_side=4096)


@pytest.fixture
def astronaut_bytes() -> bytes:
    return (DATA / "astronaut.jpg").read_bytes()


@pytest.fixture
def two_faces_bytes() -> bytes:
    return (DATA / "two_faces.jpg").read_bytes()


@pytest.fixture
def noise_bytes() -> bytes:
    return (DATA / "noise.png").read_bytes()


@pytest.fixture
def make_client() -> Callable[..., Any]:
    """Factory returning an async context manager yielding an httpx client bound to a fresh app."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _make(**overrides: Any) -> AsyncIterator[httpx.AsyncClient]:
        app = create_app(make_settings(**overrides))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
        ):
            client.app = app  # type: ignore[attr-defined]
            yield client

    return _make


@pytest.fixture
async def client(make_client: Callable[..., Any]) -> AsyncIterator[httpx.AsyncClient]:
    async with make_client() as c:
        yield c
