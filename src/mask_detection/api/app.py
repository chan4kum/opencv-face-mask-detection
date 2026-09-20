"""FastAPI application factory."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from mask_detection import __version__
from mask_detection.api.middleware import ObservabilityMiddleware
from mask_detection.api.routes import ops, v1
from mask_detection.api.state import AppState
from mask_detection.config import Settings, get_settings
from mask_detection.detector import YuNetDetector
from mask_detection.errors import AppError
from mask_detection.jobs import JobBus
from mask_detection.logging_config import configure_logging, get_logger
from mask_detection.mask import MaskClassifier
from mask_detection.metrics import BUILD_INFO
from mask_detection.service import InferenceService
from mask_detection.storage import ObjectStore
from mask_detection.telemetry import setup_tracing

log = get_logger(__name__)

PROBLEM_JSON = "application/problem+json"


def load_models(settings: Settings) -> tuple[YuNetDetector, MaskClassifier]:
    """Verify (checksums) and load both models, then warm them up. Blocking CPU work."""
    detector = YuNetDetector(
        settings.model_path,
        expected_sha256=settings.model_sha256,
        score_threshold=settings.score_threshold,
        nms_threshold=settings.nms_threshold,
        max_faces=settings.max_faces,
        max_side=settings.inference_max_side,
        upscale_to=settings.detector_upscale_to,
        intra_op_threads=settings.ort_intra_op_threads,
        inter_op_threads=settings.ort_inter_op_threads,
    )
    classifier = MaskClassifier(
        settings.mask_model_path,
        expected_sha256=settings.mask_model_sha256,
        min_confidence=settings.mask_min_confidence,
        crop_margin=settings.mask_crop_margin,
        intra_op_threads=settings.ort_intra_op_threads,
        inter_op_threads=settings.ort_inter_op_threads,
    )
    detector.warmup()
    classifier.warmup()
    return detector, classifier


def problem(
    request: Request,
    *,
    status: int,
    code: str,
    title: str,
    detail: str,
    headers: dict[str, str] | None = None,
    extra: dict[str, object] | None = None,
) -> JSONResponse:
    body: dict[str, object] = {
        "type": f"urn:mask-detection:problem:{code}",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": request.url.path,
        "request_id": getattr(request.state, "request_id", None),
    }
    if extra:
        body.update(extra)
    return JSONResponse(body, status_code=status, media_type=PROBLEM_JSON, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        return problem(
            request, status=exc.status, code=exc.code, title=exc.title, detail=exc.detail, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return problem(
            request,
            status=422,
            code="validation-error",
            title="Request validation failed",
            detail="one or more request parameters are invalid",
            extra={"errors": errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return problem(
            request,
            status=exc.status_code,
            code=f"http-{exc.status_code}",
            title=str(exc.detail),
            detail=str(exc.detail),
            headers=dict(exc.headers or {}),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_exception", path=request.url.path, exc_info=exc)
        return problem(
            request,
            status=500,
            code="internal-error",
            title="Internal server error",
            detail="an unexpected error occurred; quote the request_id when reporting it",
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        tracer_provider = setup_tracing(settings)
        # Model load + warm-up is blocking CPU work: keep it off the event loop.
        detector, classifier = await anyio.to_thread.run_sync(load_models, settings)
        state = AppState(settings=settings, inference=InferenceService(detector, classifier, settings))
        BUILD_INFO.labels(
            version=__version__, model_sha256=settings.model_sha256[:12], provider=detector.providers[0]
        ).set(1)
        if settings.async_enabled:
            state.store = ObjectStore(settings)
            state.bus = JobBus(settings)
            await state.bus.connect()
        if not settings.api_key_hashes:
            log.warning("authentication_disabled", environment=settings.environment)
        app.state.fd = state
        app.state.settings = settings
        log.info(
            "started",
            version=__version__,
            environment=settings.environment,
            providers=detector.providers,
            async_enabled=settings.async_enabled,
        )
        try:
            yield
        finally:
            state.shutting_down = True  # readiness flips to 503 while in-flight requests drain
            if state.bus is not None:
                await state.bus.close()
            if tracer_provider is not None:
                await asyncio.to_thread(tracer_provider.shutdown)
            log.info("stopped")

    app = FastAPI(
        title="Face Mask Detection Service",
        version=__version__,
        description="Face detection plus mask-state classification: send raw image bytes, get per-face mask labels.",
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.add_middleware(ObservabilityMiddleware)
    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allow_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )
    install_error_handlers(app)
    app.include_router(ops)
    app.include_router(v1)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    if settings.environment != "test":
        FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")
    return app
