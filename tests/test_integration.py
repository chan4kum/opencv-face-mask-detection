"""End-to-end tests against real NATS JetStream and an S3-compatible store.

Enabled by exporting:
    MD_TEST_NATS_URL=nats://127.0.0.1:4222
    MD_TEST_S3_ENDPOINT=http://127.0.0.1:8333
    MD_TEST_S3_ACCESS_KEY=...  MD_TEST_S3_SECRET_KEY=...
(``make up`` publishes both services on localhost.) Every test uses uniquely named
stream / KV bucket / S3 bucket and removes them afterwards.
"""

from __future__ import annotations

import asyncio
import io
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from PIL import Image

from mask_detection.api.app import create_app, load_models
from mask_detection.config import Settings, hash_api_key
from mask_detection.jobs import JobBus, JobMessage
from mask_detection.service import InferenceService
from mask_detection.storage import ObjectStore
from mask_detection.worker import Worker
from tests.conftest import make_settings

NATS_URL = os.environ.get("MD_TEST_NATS_URL")
S3_ENDPOINT = os.environ.get("MD_TEST_S3_ENDPOINT")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (NATS_URL and S3_ENDPOINT), reason="MD_TEST_NATS_URL / MD_TEST_S3_ENDPOINT not set"),
]

IMG = {"content-type": "image/jpeg"}
KEY_A, KEY_B = "integration-key-a", "integration-key-b"


@dataclass
class Rig:
    settings: Settings
    client: httpx.AsyncClient
    store: ObjectStore
    start_worker: Callable[[], Any]


def object_exists(rig: Rig, key: str) -> bool:
    try:
        rig.store._client.head_object(Bucket=rig.settings.s3_bucket, Key=key)
    except Exception as exc:
        if any(marker in str(exc) for marker in ("NoSuchKey", "Not Found", "404")):
            return False
        raise
    return True


async def wait_terminal(
    client: httpx.AsyncClient, job_id: str, headers: dict[str, str] | None = None, timeout: float = 30
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = await client.get(f"/v1/jobs/{job_id}", headers=headers)
        assert r.status_code == 200, r.text
        if r.json()["status"] in ("succeeded", "failed"):
            return dict(r.json())
        await asyncio.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


@pytest.fixture
async def rig() -> AsyncIterator[Rig]:
    suffix = uuid.uuid4().hex[:8]
    settings = make_settings(
        async_enabled=True,
        nats_url=NATS_URL,
        nats_stream=f"FDT_{suffix}",
        nats_subject=f"fdt.{suffix}.detect",
        nats_consumer=f"w-{suffix}",
        nats_kv_bucket=f"fdt_{suffix}",
        s3_bucket=f"fdt-{suffix}",
        s3_endpoint_url=S3_ENDPOINT,
        s3_access_key_id=os.environ.get("MD_TEST_S3_ACCESS_KEY", "devaccesskey"),
        s3_secret_access_key=os.environ.get("MD_TEST_S3_SECRET_KEY", "devsecretkey-change-me"),
        worker_concurrency=3,
        job_retry_backoff_s=0.1,
        job_ack_wait_s=5,
        api_key_hashes=frozenset({hash_api_key(KEY_A), hash_api_key(KEY_B)}),
    )
    store = ObjectStore(settings)
    await store.ensure_bucket()
    app = create_app(settings)
    workers: list[tuple[Worker, JobBus]] = []

    async def start_worker() -> Worker:
        # Separate bus/store/inference instances: behaves like a separate process.
        bus, wstore = JobBus(settings), ObjectStore(settings)
        await bus.connect()
        worker = Worker(settings, bus, wstore, InferenceService(*load_models(settings), settings))
        worker.start()
        workers.append((worker, bus))
        return worker

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        try:
            yield Rig(settings, client, store, start_worker)
        finally:
            for worker, bus in workers:
                await worker.stop()
                await bus.close()
            bus = JobBus(settings)
            await bus.connect()
            with suppress(Exception):
                await bus.js.delete_stream(settings.nats_stream)
                await bus.js.delete_key_value(settings.nats_kv_bucket)
            await bus.close()
            raw = store._client
            with suppress(Exception):
                for obj in raw.list_objects_v2(Bucket=settings.s3_bucket).get("Contents", []):
                    raw.delete_object(Bucket=settings.s3_bucket, Key=obj["Key"])
                raw.delete_bucket(Bucket=settings.s3_bucket)


AUTH_A = {"authorization": f"Bearer {KEY_A}"}
AUTH_B = {"authorization": f"Bearer {KEY_B}"}


async def test_readiness_reflects_backing_services(rig: Rig) -> None:
    r = await rig.client.get("/readyz")
    assert r.status_code == 200 and r.json()["checks"] == {"model": True, "nats": True, "object_storage": True}


async def test_job_lifecycle_and_input_deleted(rig: Rig, two_faces_bytes: bytes) -> None:
    await rig.start_worker()
    r = await rig.client.post("/v1/jobs", content=two_faces_bytes, headers={**IMG, **AUTH_A})
    assert r.status_code == 202
    job = r.json()
    done = await wait_terminal(rig.client, job["job_id"], AUTH_A)
    assert done["status"] == "succeeded" and len(done["result"]["faces"]) == 2 and done["attempts"] == 1
    # The worker deletes the input *after* it persists the result and acks, so poll instead of asserting immediately.
    deadline = time.monotonic() + 10
    while object_exists(rig, f"inputs/{job['job_id']}"):
        assert time.monotonic() < deadline, "input object was not deleted after processing"
        await asyncio.sleep(0.1)


async def test_api_and_worker_are_decoupled(rig: Rig, astronaut_bytes: bytes) -> None:
    """Jobs submitted while no worker is running are queued, then processed once one appears."""
    r = await rig.client.post("/v1/jobs", content=astronaut_bytes, headers={**IMG, **AUTH_A})
    job_id = r.json()["job_id"]
    await asyncio.sleep(0.5)
    queued = (await rig.client.get(f"/v1/jobs/{job_id}", headers=AUTH_A)).json()
    assert queued["status"] == "queued" and queued["result"] is None
    await rig.start_worker()
    assert (await wait_terminal(rig.client, job_id, AUTH_A))["status"] == "succeeded"


async def test_corrupt_image_fails_permanently_after_one_attempt(rig: Rig) -> None:
    await rig.start_worker()
    buf = io.BytesIO()
    Image.effect_noise((256, 256), 60).convert("RGB").save(buf, "PNG")
    truncated = buf.getvalue()[: len(buf.getvalue()) // 3]  # header intact, so submission validation passes
    r = await rig.client.post("/v1/jobs", content=truncated, headers={"content-type": "image/png", **AUTH_A})
    assert r.status_code == 202
    done = await wait_terminal(rig.client, r.json()["job_id"], AUTH_A)
    assert done["status"] == "failed" and done["attempts"] == 1 and done["error"]


async def test_missing_input_object_fails_with_clear_error(rig: Rig) -> None:
    await rig.start_worker()
    job_id = str(uuid.uuid4())
    bus = JobBus(rig.settings)
    await bus.connect()
    await bus.submit(
        JobMessage(job_id=job_id, owner=hash_api_key(KEY_A)[:16], object_key="inputs/ghost", created_at=time.time())
    )
    await bus.close()
    done = await wait_terminal(rig.client, job_id, AUTH_A)
    assert done["status"] == "failed" and "not found" in done["error"]


async def test_jobs_are_isolated_between_api_keys(rig: Rig, astronaut_bytes: bytes) -> None:
    await rig.start_worker()
    job_id = (await rig.client.post("/v1/jobs", content=astronaut_bytes, headers={**IMG, **AUTH_A})).json()["job_id"]
    assert (await rig.client.get(f"/v1/jobs/{job_id}", headers=AUTH_B)).status_code == 404
    assert (await rig.client.get(f"/v1/jobs/{job_id}", headers=AUTH_A)).status_code == 200
    assert (await rig.client.get(f"/v1/jobs/{job_id}")).status_code == 401


async def test_unknown_and_malformed_job_ids_are_404(rig: Rig) -> None:
    for job_id in (str(uuid.uuid4()), "not-a-uuid", "../../etc/passwd"):
        assert (await rig.client.get(f"/v1/jobs/{job_id}", headers=AUTH_A)).status_code == 404


async def test_many_jobs_are_processed_by_multiple_workers(rig: Rig, two_faces_bytes: bytes) -> None:
    await rig.start_worker()
    await rig.start_worker()  # two "replicas" sharing the durable consumer
    ids = []
    for _ in range(24):
        r = await rig.client.post("/v1/jobs", content=two_faces_bytes, headers={**IMG, **AUTH_A})
        ids.append(r.json()["job_id"])
    results = await asyncio.gather(*(wait_terminal(rig.client, i, AUTH_A) for i in ids))
    assert all(r["status"] == "succeeded" and len(r["result"]["faces"]) == 2 for r in results)
    assert len(set(ids)) == 24


async def test_invalid_upload_is_rejected_before_queueing(rig: Rig) -> None:
    r = await rig.client.post("/v1/jobs", content=b"nope", headers={**IMG, **AUTH_A})
    assert r.status_code == 422
    listing = rig.store._client.list_objects_v2(Bucket=rig.settings.s3_bucket)
    assert listing.get("KeyCount", 0) == 0  # nothing was uploaded to storage


async def test_worker_stops_promptly_when_idle(rig: Rig) -> None:
    worker = await rig.start_worker()
    await asyncio.sleep(0.3)
    started = time.monotonic()
    await worker.stop()
    assert time.monotonic() - started < 4


async def test_worker_app_serves_probes_and_metrics(rig: Rig) -> None:
    """The worker process (uvicorn app) starts its consumers, reports readiness and exposes metrics."""
    from mask_detection.worker import create_app as create_worker_app

    app = create_worker_app(rig.settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as c,
    ):
        assert (await c.get("/healthz")).status_code == 200
        ready = await c.get("/readyz")
        assert ready.status_code == 200 and ready.json()["checks"] == {"worker": True}
        assert "md_worker_jobs_in_flight" in (await c.get("/metrics")).text


async def test_bus_shutdown_is_prompt_even_with_active_pull_subscription(rig: Rig) -> None:
    """Regression: nc.drain() blocked ~30 s on pull subscriptions, exceeding the pod grace period."""
    bus = JobBus(rig.settings)
    await bus.connect()
    await bus.pull_subscription()
    started = time.monotonic()
    await bus.close()
    assert time.monotonic() - started < 5
