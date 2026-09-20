"""Command-line interface: local mask classification, webcam demo, API-key generation, model verification."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import anyio
import cv2

from mask_detection.annotate import draw_faces
from mask_detection.config import DEFAULT_MASK_MODEL_SHA256, DEFAULT_MODEL_SHA256, Settings, hash_api_key
from mask_detection.detector import ModelIntegrityError, YuNetDetector, sha256_file
from mask_detection.errors import AppError
from mask_detection.imaging import validate_and_decode
from mask_detection.mask import MaskClassifier
from mask_detection.mask import ModelIntegrityError as MaskModelIntegrityError
from mask_detection.mask import sha256_file as mask_sha256_file
from mask_detection.schemas import FaceOut
from mask_detection.storage import ObjectStore

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def _models(score: float | None) -> tuple[YuNetDetector, MaskClassifier]:
    s = Settings(environment="dev")
    detector = YuNetDetector(
        s.model_path,
        expected_sha256=s.model_sha256,
        score_threshold=score if score is not None else s.score_threshold,
        nms_threshold=s.nms_threshold,
        max_faces=s.max_faces,
        max_side=s.inference_max_side,
        upscale_to=s.detector_upscale_to,
    )
    classifier = MaskClassifier(
        s.mask_model_path,
        expected_sha256=s.mask_model_sha256,
        min_confidence=s.mask_min_confidence,
        crop_margin=s.mask_crop_margin,
    )
    return detector, classifier


def _cmd_analyze(args: argparse.Namespace) -> int:
    settings = Settings(environment="dev")
    try:
        image = validate_and_decode(args.image.read_bytes(), max_pixels=settings.max_image_pixels)
    except OSError as exc:
        print(f"error: cannot read {args.image}: {exc}", file=sys.stderr)
        return 2
    except AppError as exc:
        print(f"error: {exc.detail}", file=sys.stderr)
        return 2
    detector, classifier = _models(args.score)
    results = [(f, classifier.classify_face(image, (f.x, f.y, f.width, f.height))) for f in detector.detect(image)]
    payload = {
        "image": {"width": image.shape[1], "height": image.shape[0]},
        "faces": [
            FaceOut.from_face(f, e, low_resolution_px=settings.low_resolution_px).model_dump() for f, e in results
        ],
    }
    if args.output and not cv2.imwrite(str(args.output), draw_faces(image, results)):
        print(f"error: could not write {args.output}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2 if sys.stdout.isatty() else None))
    return 0


def _cmd_webcam(args: argparse.Namespace) -> int:
    detector, classifier = _models(args.score)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"error: cannot open camera {args.camera}", file=sys.stderr)
        return 2
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            frame = cast("NDArray[np.uint8]", raw)
            results = [
                (f, classifier.classify_face(frame, (f.x, f.y, f.width, f.height))) for f in detector.detect(frame)
            ]
            cv2.imshow("mask-detection (q to quit)", draw_faces(frame, results))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except cv2.error as exc:
        print(
            "error: this OpenCV build has no GUI support (headless). For the webcam demo run:\n"
            "  uv pip uninstall opencv-python-headless && uv pip install opencv-python\n"
            f"({exc})",
            file=sys.stderr,
        )
        return 2
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def _cmd_keygen(_: argparse.Namespace) -> int:
    key = secrets.token_urlsafe(32)
    print(f"API key (give to the client, shown once): {key}")
    print(f"MD_API_KEY_HASHES entry (put on the server):  {hash_api_key(key)}")
    return 0


def _cmd_init_storage(args: argparse.Namespace) -> int:
    settings = Settings(environment="dev")
    store = ObjectStore(settings)
    try:
        notes = anyio.run(lambda: store.ensure_bucket(expire_days=args.expire_days, region=settings.s3_region))
    except AppError as exc:
        print(f"error: {exc.detail}", file=sys.stderr)
        return 1
    for note in notes:
        print(f"warning: {note}", file=sys.stderr)
    print(f"bucket {settings.s3_bucket!r} ready (inputs/ expire after {args.expire_days} day(s))")
    return 0


def _cmd_verify_models(_: argparse.Namespace) -> int:
    s = Settings(environment="dev")
    status = 0
    for label, path, expected, digest in (
        ("face detector", s.model_path, DEFAULT_MODEL_SHA256, sha256_file),
        ("mask classifier", s.mask_model_path, DEFAULT_MASK_MODEL_SHA256, mask_sha256_file),
    ):
        try:
            actual = digest(path)
        except OSError as exc:
            print(f"{label}: {exc}", file=sys.stderr)
            status = max(status, 2)
            continue
        ok = actual == expected
        print(f"{label}: {path}: sha256={actual} {'OK' if ok else 'MISMATCH (expected ' + expected + ')'}")
        status = max(status, 0 if ok else 1)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mask-detection", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    est = sub.add_parser("analyze", help="detect faces and classify their mask state in an image; prints JSON")
    est.add_argument("image", type=Path)
    est.add_argument("-o", "--output", type=Path, help="write an annotated copy here")
    est.add_argument("--score", type=float, help="face score threshold override (0-1)")
    est.set_defaults(func=_cmd_analyze)

    cam = sub.add_parser("webcam", help="live webcam demo (requires GUI-enabled OpenCV)")
    cam.add_argument("--camera", type=int, default=0)
    cam.add_argument("--score", type=float)
    cam.set_defaults(func=_cmd_webcam)

    sub.add_parser("keygen", help="generate an API key and its SHA-256 hash").set_defaults(func=_cmd_keygen)

    ini = sub.add_parser("init-storage", help="create the S3 bucket and its input-expiry lifecycle rule")
    ini.add_argument("--expire-days", type=int, default=1)
    ini.set_defaults(func=_cmd_init_storage)

    sub.add_parser("verify-models", help="check both model files against their pinned checksums").set_defaults(
        func=_cmd_verify_models
    )

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ModelIntegrityError, MaskModelIntegrityError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
