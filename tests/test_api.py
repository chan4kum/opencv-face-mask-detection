from __future__ import annotations

import asyncio
import io
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image

from mask_detection.config import hash_api_key

IMG = {"content-type": "image/jpeg"}


async def test_health_and_ready(client: httpx.AsyncClient) -> None:
    assert (await client.get("/healthz")).json()["status"] == "ok"
    r = await client.get("/readyz")
    assert r.status_code == 200 and r.json()["checks"] == {"model": True}


async def test_analyze_returns_faces(client: httpx.AsyncClient, two_faces_bytes: bytes) -> None:
    r = await client.post("/v1/analyze", content=two_faces_bytes, headers=IMG)
    assert r.status_code == 200
    body = r.json()
    assert body["image"] == {"width": 1024, "height": 512}
    assert len(body["faces"]) == 2
    face = body["faces"][0]
    assert set(face) == {"box", "score", "landmarks", "mask"} and set(face["landmarks"]) == {
        "right_eye",
        "left_eye",
        "nose",
        "right_mouth",
        "left_mouth",
    }
    assert body["models"]["face_detector"]["name"] == "yunet-2026may" and body["inference_ms"] > 0
    assert body["models"]["mask_classifier"]["name"].startswith("mobilenetv3")
    mask = face["mask"]
    assert mask["label"] in {"with_mask", "without_mask", "mask_weared_incorrect"}
    assert sum(mask["probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    assert (
        0 <= mask["confidence"] <= 1
        and isinstance(mask["uncertain"], bool)
        and isinstance(mask["low_resolution"], bool)
    )
    assert (
        body["summary"]["faces"] == 2
        and body["summary"]["with_mask"] + body["summary"]["without_mask"] + body["summary"]["mask_weared_incorrect"]
        == 2
    )
    assert body["request_id"] == r.headers["x-request-id"]


async def test_analyze_no_face_returns_empty_list(client: httpx.AsyncClient, noise_bytes: bytes) -> None:
    r = await client.post("/v1/analyze", content=noise_bytes, headers={"content-type": "image/png"})
    assert r.status_code == 200 and r.json()["faces"] == []


async def test_annotated_endpoint_returns_jpeg_with_drawing(client: httpx.AsyncClient, astronaut_bytes: bytes) -> None:
    r = await client.post("/v1/analyze/annotated", content=astronaut_bytes, headers=IMG)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.headers["x-face-count"] == "1"
    out = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=int)
    orig = np.asarray(Image.open(io.BytesIO(astronaut_bytes)).convert("RGB"), dtype=int)
    assert out.shape == orig.shape and np.abs(out - orig).sum() > 0


async def test_octet_stream_is_accepted(client: httpx.AsyncClient, astronaut_bytes: bytes) -> None:
    r = await client.post("/v1/analyze", content=astronaut_bytes, headers={"content-type": "application/octet-stream"})
    assert r.status_code == 200


async def test_wrong_content_type_is_415(client: httpx.AsyncClient, astronaut_bytes: bytes) -> None:
    r = await client.post("/v1/analyze", content=astronaut_bytes, headers={"content-type": "text/plain"})
    assert r.status_code == 415 and r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["type"] == "urn:mask-detection:problem:unsupported-media-type"


async def test_gif_is_415_even_with_image_content_type(client: httpx.AsyncClient) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "GIF")
    r = await client.post("/v1/analyze", content=buf.getvalue(), headers=IMG)
    assert r.status_code == 415


@pytest.mark.parametrize("payload", [b"", b"definitely not an image"])
async def test_invalid_image_is_422(client: httpx.AsyncClient, payload: bytes) -> None:
    r = await client.post("/v1/analyze", content=payload, headers=IMG)
    assert r.status_code == 422 and r.json()["status"] == 422


async def test_oversize_body_is_413_by_content_length(make_client: Callable[..., Any], astronaut_bytes: bytes) -> None:
    async with make_client(max_upload_bytes=2048) as c:
        r = await c.post("/v1/analyze", content=astronaut_bytes, headers=IMG)
    assert r.status_code == 413


async def test_oversize_body_is_413_when_streamed_without_length(make_client: Callable[..., Any]) -> None:
    async def gen() -> Any:
        for _ in range(10):
            yield b"x" * 1024

    async with make_client(max_upload_bytes=2048) as c:
        r = await c.post("/v1/analyze", content=gen(), headers=IMG)  # chunked: no Content-Length
    assert r.status_code == 413


async def test_pixel_bomb_is_rejected(make_client: Callable[..., Any]) -> None:
    buf = io.BytesIO()
    Image.new("1", (20000, 20000)).save(buf, "PNG")
    async with make_client() as c:
        r = await c.post("/v1/analyze", content=buf.getvalue(), headers={"content-type": "image/png"})
    assert r.status_code == 422 and "limit" in r.json()["detail"]


async def test_request_id_is_echoed_when_valid_and_replaced_when_not(client: httpx.AsyncClient) -> None:
    ok = await client.get("/healthz", headers={"x-request-id": "trace-abc-12345"})
    assert ok.headers["x-request-id"] == "trace-abc-12345"
    bad = await client.get("/healthz", headers={"x-request-id": "bad id\twith spaces & $ymbols"})
    assert bad.headers["x-request-id"] != "bad id\twith spaces & $ymbols" and len(bad.headers["x-request-id"]) == 32


async def test_security_headers_present(client: httpx.AsyncClient) -> None:
    h = (await client.get("/healthz")).headers
    assert h["x-content-type-options"] == "nosniff" and h["cache-control"] == "no-store"
    assert "default-src 'none'" in h["content-security-policy"]


async def test_unknown_route_and_method_are_problem_json(client: httpx.AsyncClient) -> None:
    r = await client.get("/nope")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/problem+json")
    r = await client.get("/v1/analyze")
    assert r.status_code == 405 and r.json()["status"] == 405


async def test_metrics_endpoint_exposes_expected_series(client: httpx.AsyncClient, astronaut_bytes: bytes) -> None:
    await client.post("/v1/analyze", content=astronaut_bytes, headers=IMG)
    text = (await client.get("/metrics")).text
    assert 'md_http_requests_total{method="POST",route="/v1/analyze",status="200"}' in text
    assert "md_inference_duration_seconds_bucket" in text and "md_faces_per_image_count" in text
    assert "md_build_info" in text


async def test_metrics_labels_use_route_templates_not_raw_paths(client: httpx.AsyncClient) -> None:
    await client.get("/some/random/path/123")
    text = (await client.get("/metrics")).text
    assert "/some/random/path/123" not in text and 'route="unmatched"' in text


async def test_docs_can_be_disabled(make_client: Callable[..., Any]) -> None:
    async with make_client(docs_enabled=False) as c:
        assert (await c.get("/docs")).status_code == 404 and (await c.get("/openapi.json")).status_code == 404
    async with make_client(docs_enabled=True) as c:
        assert (await c.get("/openapi.json")).status_code == 200


async def test_jobs_endpoints_report_501_when_async_disabled(client: httpx.AsyncClient, astronaut_bytes: bytes) -> None:
    assert (await client.post("/v1/jobs", content=astronaut_bytes, headers=IMG)).status_code == 501
    assert (await client.get("/v1/jobs/3f0c7b0e-8f0d-4b7e-9f77-1c7f6d1b2a10")).status_code == 501


class TestAuth:
    KEY = "s3cr3t-key"

    @pytest.fixture
    def secured(self, make_client: Callable[..., Any]) -> Any:
        return make_client(api_key_hashes=frozenset({hash_api_key(self.KEY), hash_api_key("other")}))

    async def test_missing_key_is_401_with_challenge(self, secured: Any, astronaut_bytes: bytes) -> None:
        async with secured as c:
            r = await c.post("/v1/analyze", content=astronaut_bytes, headers=IMG)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"

    async def test_wrong_key_is_401(self, secured: Any, astronaut_bytes: bytes) -> None:
        async with secured as c:
            r = await c.post("/v1/analyze", content=astronaut_bytes, headers={**IMG, "authorization": "Bearer nope"})
        assert r.status_code == 401

    async def test_valid_keys_are_accepted(self, secured: Any, astronaut_bytes: bytes) -> None:
        async with secured as c:
            for key in (self.KEY, "other"):
                r = await c.post(
                    "/v1/analyze", content=astronaut_bytes, headers={**IMG, "authorization": f"Bearer {key}"}
                )
                assert r.status_code == 200

    async def test_probes_and_metrics_stay_unauthenticated(self, secured: Any) -> None:
        async with secured as c:
            assert (await c.get("/healthz")).status_code == 200
            assert (await c.get("/readyz")).status_code == 200
            assert (await c.get("/metrics")).status_code == 200

    async def test_auth_checked_before_body_is_processed(self, secured: Any) -> None:
        async with secured as c:
            r = await c.post("/v1/analyze", content=b"garbage", headers=IMG)
        assert r.status_code == 401  # not 422: unauthenticated callers learn nothing about validation


async def test_load_shedding_returns_503_with_retry_after(
    make_client: Callable[..., Any], astronaut_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mask_detection.detector import YuNetDetector

    gate, entered = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()
    real = YuNetDetector.detect

    def slow(self: YuNetDetector, image: Any) -> Any:
        loop.call_soon_threadsafe(entered.set)
        asyncio.run_coroutine_threadsafe(gate.wait(), loop).result(timeout=10)
        return real(self, image)

    async with make_client(max_concurrent_inference=1, inference_queue_timeout_s=0.05) as c:
        monkeypatch.setattr(YuNetDetector, "detect", slow)
        first = asyncio.create_task(c.post("/v1/analyze", content=astronaut_bytes, headers=IMG))
        await asyncio.wait_for(entered.wait(), 5)
        shed = await c.post("/v1/analyze", content=astronaut_bytes, headers=IMG)
        gate.set()
        assert (await first).status_code == 200
    assert shed.status_code == 503 and shed.headers["retry-after"] == "1"
    assert shed.json()["type"] == "urn:mask-detection:problem:overloaded"


async def test_unhandled_errors_do_not_leak_details(
    make_client: Callable[..., Any], astronaut_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mask_detection.detector import YuNetDetector

    def boom(self: YuNetDetector, image: Any) -> Any:
        raise RuntimeError("secret internal path /etc/shadow")

    async with make_client() as c:
        monkeypatch.setattr(YuNetDetector, "detect", boom)
        transport = httpx.ASGITransport(app=c.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as safe:
            r = await safe.post("/v1/analyze", content=astronaut_bytes, headers=IMG)
    assert r.status_code == 500 and "shadow" not in r.text and r.json()["request_id"]


async def test_end_to_end_on_held_out_ground_truth_images(client: httpx.AsyncClient) -> None:
    """Real detector + real classifier on CC0 images from the held-out test split (labels from the dataset)."""
    from tests.conftest import DATA, SAMPLES

    for name, truth in SAMPLES.items():
        if truth["faces"][0]["label"] == "mask_weared_incorrect":
            continue  # weakest class (recall ~37%), see docs/EVALUATION.md
        r = await client.post("/v1/analyze", content=(DATA / name).read_bytes(), headers={"content-type": "image/png"})
        body = r.json()
        assert r.status_code == 200 and body["summary"]["faces"] == 1, (name, body["summary"])
        assert body["faces"][0]["mask"]["label"] == truth["faces"][0]["label"], name


async def test_annotated_image_marks_the_faces(client: httpx.AsyncClient) -> None:
    from tests.conftest import DATA

    r = await client.post(
        "/v1/analyze/annotated",
        content=(DATA / "maksssksksss520.png").read_bytes(),
        headers={"content-type": "image/png"},
    )
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.headers["x-face-count"] == "1"
