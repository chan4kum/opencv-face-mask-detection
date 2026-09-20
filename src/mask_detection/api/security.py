"""API-key authentication. Only SHA-256 digests of keys are ever configured or compared."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from mask_detection.config import Settings
from mask_detection.errors import UnauthorizedError

_bearer = HTTPBearer(auto_error=False, description="API key, sent as `Authorization: Bearer <key>`")


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller. ``id`` scopes async-job ownership."""

    id: str


ANONYMOUS = Principal(id="anonymous")


def authenticate(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    settings: Settings = request.app.state.settings
    if not settings.api_key_hashes:
        return ANONYMOUS  # auth explicitly disabled (dev/test, or enforced upstream)
    if credentials is None:
        raise UnauthorizedError("missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    digest = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    matched: str | None = None
    for known in settings.api_key_hashes:  # no early exit: constant work per known key
        if hmac.compare_digest(digest, known):
            matched = known
    if matched is None:
        raise UnauthorizedError("invalid API key", headers={"WWW-Authenticate": "Bearer"})
    return Principal(id=matched[:16])


CurrentPrincipal = Annotated[Principal, Depends(authenticate)]
