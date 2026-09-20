"""HTTP routes."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from mask_detection import __version__
from mask_detection.annotate import draw_faces, encode_jpeg
from mask_detection.api.security import CurrentPrincipal
from mask_detection.api.state import AppState, get_state
from mask_detection.errors import (
    AsyncDisabledError,
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from mask_detection.imaging import probe_image
from mask_detection.jobs import JobMessage, JobStatus, is_valid_job_id, new_job_id
from mask_detection.metrics import IMAGES_REJECTED, JOBS_SUBMITTED
from mask_detection.schemas import (
    AnalysisResponse,
    HealthResponse,
    JobAccepted,
    JobResponse,
    ModelsInfo,
    ProblemDetails,
)

State = Annotated[AppState, Depends(get_state)]

ALLOWED_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/bmp", "application/octet-stream"})

_PROBLEMS: dict[int | str, dict[str, object]] = {
    401: {"model": ProblemDetails, "description": "Missing or invalid API key"},
    413: {"model": ProblemDetails, "description": "Payload too large"},
    415: {"model": ProblemDetails, "description": "Unsupported image type"},
    422: {"model": ProblemDetails, "description": "Not a valid image"},
    503: {"model": ProblemDetails, "description": "At capacity or dependency unavailable"},
}

ops = APIRouter(tags=["operations"])
v1 = APIRouter(prefix="/v1", tags=["detection"], responses=_PROBLEMS)


async def read_image_body(request: Request, limit: int) -> bytes:
    """Read the raw request body, enforcing a hard byte limit while streaming."""
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        IMAGES_REJECTED.labels(reason="unsupported-media-type").inc()
        raise UnsupportedMediaTypeError(
            f"Content-Type {content_type or '(none)'!r} not accepted; send the raw image bytes with one of: "
            + ", ".join(sorted(ALLOWED_CONTENT_TYPES))
        )
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        IMAGES_REJECTED.labels(reason="payload-too-large").inc()
        raise PayloadTooLargeError(f"body exceeds the {limit}-byte limit")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            IMAGES_REJECTED.labels(reason="payload-too-large").inc()
            raise PayloadTooLargeError(f"body exceeds the {limit}-byte limit")
    return bytes(body)


# --------------------------------------------------------------------------------------
# operations
# --------------------------------------------------------------------------------------
@ops.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@ops.get(
    "/readyz",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
    summary="Readiness probe",
)
async def readyz(state: State) -> Response:
    checks = {"model": True}
    if state.bus is not None:
        checks["nats"] = state.bus.is_ready()
        checks["object_storage"] = await state.storage_healthy()
    ready = all(checks.values()) and not state.shutting_down
    body = HealthResponse(status="ready" if ready else "not_ready", checks=checks, version=__version__)
    return JSONResponse(body.model_dump(), status_code=200 if ready else 503)


# --------------------------------------------------------------------------------------
# synchronous detection
# --------------------------------------------------------------------------------------
@v1.get("/models", response_model=ModelsInfo, summary="Loaded model metadata")
async def model_info(state: State, _: CurrentPrincipal) -> ModelsInfo:
    return state.inference.models


@v1.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Detect faces and classify their mask state (synchronous)",
    description="Send the raw image bytes as the request body with an `image/*` Content-Type.",
)
async def analyze(request: Request, state: State, _: CurrentPrincipal) -> AnalysisResponse:
    data = await read_image_body(request, state.settings.max_upload_bytes)
    result, _image, _faces = await state.inference.analyze_bytes(data, source="api")
    return AnalysisResponse(request_id=request.state.request_id, **result.model_dump())


@v1.post(
    "/analyze/annotated",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}, "description": "JPEG with boxes and landmarks drawn"}},
    summary="Classify mask state and return the annotated image",
)
async def detect_annotated(request: Request, state: State, _: CurrentPrincipal) -> Response:
    data = await read_image_body(request, state.settings.max_upload_bytes)
    result, image, faces = await state.inference.analyze_bytes(data, source="api")
    jpeg = await anyio.to_thread.run_sync(lambda: encode_jpeg(draw_faces(image, faces)))
    return Response(jpeg, media_type="image/jpeg", headers={"X-Face-Count": str(len(result.faces))})


# --------------------------------------------------------------------------------------
# asynchronous jobs
# --------------------------------------------------------------------------------------
@v1.post(
    "/jobs",
    status_code=202,
    response_model=JobAccepted,
    summary="Submit an image for asynchronous detection",
    responses={501: {"model": ProblemDetails, "description": "Async pipeline disabled"}},
)
async def submit_job(request: Request, state: State, principal: CurrentPrincipal) -> JobAccepted:
    if state.bus is None or state.store is None:
        raise AsyncDisabledError
    data = await read_image_body(request, state.settings.max_upload_bytes)
    probe_image(data, max_pixels=state.settings.max_image_pixels)  # cheap header validation up front

    job_id = new_job_id()
    key = f"inputs/{job_id}"
    await state.store.put(key, data)
    try:
        await state.bus.submit(JobMessage(job_id=job_id, owner=principal.id, object_key=key, created_at=time.time()))
    except Exception:
        await asyncio.shield(_best_effort_delete(state, key))  # do not orphan the upload
        raise
    JOBS_SUBMITTED.inc()
    return JobAccepted(job_id=job_id, status=JobStatus.QUEUED, status_url=f"/v1/jobs/{job_id}")


@v1.get("/jobs/{job_id}", response_model=JobResponse, summary="Get the status/result of a job")
async def get_job(job_id: str, state: State, principal: CurrentPrincipal) -> JobResponse:
    if state.bus is None:
        raise AsyncDisabledError
    # Owner mismatch is reported as 404 so job IDs of other tenants are not disclosed.
    record = await state.bus.get_record(job_id) if is_valid_job_id(job_id) else None
    if record is None or record.owner != principal.id:
        raise NotFoundError("no such job (it may have expired)")
    return JobResponse.model_validate(record.model_dump())


async def _best_effort_delete(state: AppState, key: str) -> None:
    if state.store is None:
        return
    with contextlib.suppress(Exception):  # cleanup must never mask the original error
        await state.store.delete(key)
