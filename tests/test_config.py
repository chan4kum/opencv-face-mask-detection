from __future__ import annotations

import pytest
from pydantic import ValidationError

from mask_detection.config import Settings, hash_api_key

H = hash_api_key("k")


def test_prod_requires_api_keys_or_explicit_opt_out() -> None:
    with pytest.raises(ValidationError, match="requires MD_API_KEY_HASHES"):
        Settings(environment="prod")
    Settings(environment="prod", api_key_hashes=frozenset({H}))
    Settings(environment="prod", auth_disabled=True)


def test_rejects_malformed_hashes() -> None:
    with pytest.raises(ValidationError, match="SHA-256"):
        Settings(api_key_hashes=frozenset({"plaintext-key"}))


def test_s3_credentials_must_be_paired() -> None:
    with pytest.raises(ValidationError, match="set together"):
        Settings(s3_access_key_id="a")  # type: ignore[arg-type]


def test_env_parsing_of_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MD_API_KEY_HASHES", f" {H.upper()} , {hash_api_key('b')} ")
    monkeypatch.setenv("MD_CORS_ALLOW_ORIGINS", "https://a.example, https://b.example")
    monkeypatch.setenv("MD_SCORE_THRESHOLD", "0.5")
    s = Settings()
    assert H in s.api_key_hashes and len(s.api_key_hashes) == 2
    assert s.cors_allow_origins == ("https://a.example", "https://b.example")
    assert s.score_threshold == 0.5


@pytest.mark.parametrize(("field", "value"), [("score_threshold", 1.5), ("max_upload_bytes", 1), ("log_level", "LOUD")])
def test_out_of_range_values_fail_fast(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_secrets_are_not_leaked_in_repr() -> None:
    s = Settings(s3_access_key_id="AKIAEXAMPLE", s3_secret_access_key="topsecret")  # type: ignore[arg-type]
    assert "topsecret" not in repr(s) and "AKIAEXAMPLE" not in repr(s)


def test_settings_are_immutable() -> None:
    with pytest.raises(ValidationError):
        Settings().score_threshold = 0.1  # type: ignore[misc]
