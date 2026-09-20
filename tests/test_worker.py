"""Worker delivery semantics, tested against in-memory fakes (no NATS/S3 needed)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from mask_detection.api.app import load_models
from mask_detection.errors import DependencyUnavailableError, OverloadedError
from mask_detection.jobs import JobMessage, JobRecord, JobStatus
from mask_detection.service import InferenceService
from mask_detection.storage import ObjectNotFoundError
from mask_detection.worker import Worker
from tests.conftest import make_settings


@dataclass
class FakeMeta:
    num_delivered: int = 1


@dataclass
class FakeMsg:
    data: bytes
    metadata: FakeMeta = field(default_factory=FakeMeta)
    headers: dict[str, str] | None = None
    calls: list[tuple[str, Any]] = field(default_factory=list)

    async def ack(self) -> None:
        self.calls.append(("ack", None))

    async def nak(self, delay: float | None = None) -> None:
        self.calls.append(("nak", delay))

    async def term(self) -> None:
        self.calls.append(("term", None))


class FakeBus:
    def __init__(self) -> None:
        self.records: dict[str, JobRecord] = {}
        self.fail_put = False
        self.fail_on_call: int | None = None  # 1-based index of the put_record call that should fail
        self.put_calls = 0

    def is_ready(self) -> bool:
        return True

    async def get_record(self, job_id: str) -> JobRecord | None:
        return self.records.get(job_id)

    async def put_record(self, record: JobRecord) -> None:
        self.put_calls += 1
        if self.fail_put or self.put_calls == self.fail_on_call:
            raise RuntimeError("kv down")
        self.records[record.job_id] = record.model_copy(deep=True)


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.error: Exception | None = None

    async def get(self, key: str) -> bytes:
        if self.error:
            raise self.error
        if key not in self.objects:
            raise ObjectNotFoundError(key)
        return self.objects[key]

    async def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.objects.pop(key, None)


@pytest.fixture
def rig() -> tuple[Worker, FakeBus, FakeStore]:
    settings = make_settings(job_max_deliver=3, job_retry_backoff_s=0.5, async_enabled=True)
    bus, store = FakeBus(), FakeStore()
    worker = Worker(settings, bus, store, InferenceService(*load_models(settings), settings))  # type: ignore[arg-type]
    return worker, bus, store


def job_msg(job_id: str = "job-1", key: str = "inputs/job-1", attempt: int = 1, raw: bytes | None = None) -> FakeMsg:
    payload = JobMessage(job_id=job_id, owner="o", object_key=key, created_at=time.time() - 0.2)
    return FakeMsg(raw if raw is not None else payload.model_dump_json().encode(), FakeMeta(attempt))


async def test_success_path_persists_result_then_acks_and_deletes_input(
    rig: tuple[Worker, FakeBus, FakeStore], two_faces_bytes: bytes
) -> None:
    worker, bus, store = rig
    store.objects["inputs/job-1"] = two_faces_bytes
    msg = job_msg()
    await worker._process(msg)  # type: ignore[arg-type]
    rec = bus.records["job-1"]
    assert rec.status is JobStatus.SUCCEEDED and rec.result is not None and len(rec.result["faces"]) == 2
    assert msg.calls == [("ack", None)] and store.deleted == ["inputs/job-1"]


async def test_invalid_image_fails_permanently_without_retry(rig: tuple[Worker, FakeBus, FakeStore]) -> None:
    worker, bus, store = rig
    store.objects["inputs/job-1"] = b"not an image"
    msg = job_msg()
    await worker._process(msg)  # type: ignore[arg-type]
    assert bus.records["job-1"].status is JobStatus.FAILED and "image" in (bus.records["job-1"].error or "")
    assert msg.calls == [("ack", None)]


async def test_missing_input_fails_permanently(rig: tuple[Worker, FakeBus, FakeStore]) -> None:
    worker, bus, _ = rig
    msg = job_msg()
    await worker._process(msg)  # type: ignore[arg-type]
    assert bus.records["job-1"].status is JobStatus.FAILED and "not found" in (bus.records["job-1"].error or "")
    assert msg.calls == [("ack", None)]


async def test_transient_error_is_naked_with_growing_backoff(rig: tuple[Worker, FakeBus, FakeStore]) -> None:
    worker, bus, store = rig
    store.error = ConnectionError("s3 flaked")
    for attempt, expected_delay in ((1, 0.5), (2, 1.0)):
        msg = job_msg(attempt=attempt)
        await worker._process(msg)  # type: ignore[arg-type]
        assert msg.calls == [("nak", expected_delay)]
        assert bus.records["job-1"].status is JobStatus.PROCESSING


async def test_transient_error_on_last_attempt_fails_the_job(rig: tuple[Worker, FakeBus, FakeStore]) -> None:
    worker, bus, store = rig
    store.error = ConnectionError("still down")
    msg = job_msg(attempt=3)  # == job_max_deliver
    await worker._process(msg)  # type: ignore[arg-type]
    rec = bus.records["job-1"]
    assert rec.status is JobStatus.FAILED and rec.error == "processing failed after retries"
    assert msg.calls == [("ack", None)]
    assert "still down" not in (rec.error or "")  # internals are not exposed to API clients


async def test_redelivery_of_finished_job_is_idempotent(
    rig: tuple[Worker, FakeBus, FakeStore], astronaut_bytes: bytes
) -> None:
    worker, bus, store = rig
    store.objects["inputs/job-1"] = astronaut_bytes
    await worker._process(job_msg())  # type: ignore[arg-type]
    first = bus.records["job-1"].model_copy(deep=True)
    store.error = RuntimeError("must not be touched again")
    msg = job_msg(attempt=2)
    await worker._process(msg)  # type: ignore[arg-type]
    assert msg.calls == [("ack", None)] and bus.records["job-1"] == first


async def test_poison_message_is_terminated(rig: tuple[Worker, FakeBus, FakeStore]) -> None:
    worker, bus, _ = rig
    msg = job_msg(raw=b'{"unexpected": true}')
    await worker._process(msg)  # type: ignore[arg-type]
    assert msg.calls == [("term", None)] and bus.records == {}


async def test_expired_kv_record_is_recreated(rig: tuple[Worker, FakeBus, FakeStore], astronaut_bytes: bytes) -> None:
    worker, bus, store = rig
    store.objects["inputs/job-1"] = astronaut_bytes
    assert "job-1" not in bus.records
    await worker._process(job_msg())  # type: ignore[arg-type]
    assert bus.records["job-1"].status is JobStatus.SUCCEEDED


async def test_outcome_is_persisted_before_ack(rig: tuple[Worker, FakeBus, FakeStore], astronaut_bytes: bytes) -> None:
    """If persisting the final result fails, the message must NOT be acked (so it is redelivered)."""
    worker, bus, store = rig
    store.objects["inputs/job-1"] = astronaut_bytes
    bus.fail_on_call = 2  # call 1 = "processing" marker succeeds, call 2 = final result write fails
    msg = job_msg()
    await worker._handle(msg)  # type: ignore[arg-type]  # _handle is the last-resort guard
    assert ("ack", None) not in msg.calls and [c[0] for c in msg.calls] == ["nak"]
    assert bus.records["job-1"].status is JobStatus.PROCESSING  # result was never marked done
    assert store.deleted == []  # and the input is kept for the retry


async def test_delete_input_can_be_disabled(two_faces_bytes: bytes) -> None:
    settings = make_settings(delete_input_after_processing=False)
    bus, store = FakeBus(), FakeStore()
    worker = Worker(settings, bus, store, InferenceService(*load_models(settings), settings))  # type: ignore[arg-type]
    store.objects["inputs/job-1"] = two_faces_bytes
    await worker._process(job_msg())  # type: ignore[arg-type]
    assert store.deleted == [] and "inputs/job-1" in store.objects


@pytest.mark.parametrize("error", [DependencyUnavailableError("s3 down"), OverloadedError("busy")])
async def test_5xx_app_errors_are_retried_not_failed(rig: tuple[Worker, FakeBus, FakeStore], error: Exception) -> None:
    """Regression: a storage outage (503) must be retried, not recorded as a permanent job failure."""
    worker, bus, store = rig
    store.error = error
    msg = job_msg(attempt=1)
    await worker._process(msg)  # type: ignore[arg-type]
    assert msg.calls == [("nak", 0.5)]
    assert bus.records["job-1"].status is JobStatus.PROCESSING
