"""The one place that talks to the API.

The previous frontend put the backend URL in an editable sidebar text box and
told the user, in the interface, to go and start a server. That is a developer's
mental model leaking into a product: an end user cannot act on it and should
never have been asked to. The address is configuration, so it comes from the
environment, and an unreachable service is presented as a service state rather
than as a field the user got wrong.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

DEFAULT_BASE_URL = "http://localhost:8000"

CONNECT_TIMEOUT = 3.0
READ_TIMEOUT = 30.0
BATCH_READ_TIMEOUT = 120.0


class ServiceUnavailable(RuntimeError):
    """The analysis service could not be reached at all."""


class ServiceError(RuntimeError):
    """The service answered, but not with what was asked for."""


def resolve_base_url() -> str:
    """Where the API lives.

    Accepts the older ``.../analyze`` form that run_app.ps1 used to export, so
    an environment left over from a previous version still points somewhere
    sensible instead of 404ing on ``/analyze/analyze``.
    """
    raw = (
        os.getenv("MEDICAL_NLP_API_BASE_URL")
        or os.getenv("MEDICAL_NLP_API_URL")
        or os.getenv("API_URL")
        or DEFAULT_BASE_URL
    ).strip()

    if "://" not in raw:
        raw = "http://" + raw

    parsed = urlparse(raw)
    if not parsed.netloc:
        return DEFAULT_BASE_URL

    path = parsed.path.rstrip("/")
    for legacy_suffix in ("/analyze", "/engines", "/models"):
        if path.endswith(legacy_suffix):
            path = path[: -len(legacy_suffix)]
            break

    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _headers() -> dict[str, str]:
    key = os.getenv("MEDICAL_NLP_API_KEY", "").strip()
    return {"X-API-Key": key} if key else {}


@dataclass(frozen=True)
class ServiceStatus:
    reachable: bool
    version: str | None = None
    database_ready: bool = False
    auth_required: bool = False
    engines_available: dict[str, bool] | None = None
    detail: str = ""

    @property
    def engine_count(self) -> int:
        return sum(1 for ok in (self.engines_available or {}).values() if ok)


def _request(method: str, path: str, *, timeout: float, **kwargs: Any) -> requests.Response:
    url = f"{resolve_base_url()}{path}"
    try:
        response = requests.request(
            method, url, headers=_headers(), timeout=(CONNECT_TIMEOUT, timeout), **kwargs
        )
    except requests.RequestException as exc:
        raise ServiceUnavailable(str(exc)) from exc

    if response.status_code == 401:
        raise ServiceError(
            "The analysis service requires an API key. Set MEDICAL_NLP_API_KEY in the "
            "environment this application runs in."
        )
    if response.status_code >= 400:
        raise ServiceError(_error_detail(response))
    return response


def _error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"The service returned HTTP {response.status_code}."
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, list) and detail:
        first = detail[0]
        if isinstance(first, dict):
            detail = first.get("msg")
    return str(detail) if detail else f"The service returned HTTP {response.status_code}."


def _json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise ServiceError("The service returned a response this application cannot read.") from exc


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

def status() -> ServiceStatus:
    try:
        payload = _json(_request("GET", "/health", timeout=5.0))
    except (ServiceUnavailable, ServiceError) as exc:
        return ServiceStatus(reachable=False, detail=str(exc))
    return ServiceStatus(
        reachable=payload.get("status") == "ok",
        version=payload.get("version"),
        database_ready=bool(payload.get("database_ready")),
        auth_required=bool(payload.get("auth_required")),
        engines_available=payload.get("engines_available") or {},
    )


def engines() -> dict:
    """Every engine with its own metrics, plus the arbitration policy.

    There is deliberately no hardcoded fallback. The previous frontend answered
    an unreachable service with an invented engine carrying a hardcoded accuracy
    -- a figure this project has withdrawn, and one that never described the
    shipped artifact anyway. An application that cannot reach the service does
    not know what the engines are, and now says so.
    """
    return _json(_request("GET", "/engines", timeout=10.0))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(
    text: str,
    *,
    engine_key: str,
    session_id: str | None = None,
    patient_reference: str | None = None,
    persist: bool = True,
) -> dict:
    payload = {
        "text": text,
        "model_key": engine_key,
        "session_id": session_id,
        "patient_reference": patient_reference or None,
        "persist": persist,
    }
    return _json(_request("POST", "/analyze", timeout=READ_TIMEOUT, json=payload))


def analyze_batch(
    notes: list[str], *, engine_key: str, session_id: str | None = None, persist: bool = True
) -> dict:
    payload = {
        "notes": notes,
        "model_key": engine_key,
        "session_id": session_id,
        "persist": persist,
    }
    return _json(_request("POST", "/analyze/batch", timeout=BATCH_READ_TIMEOUT, json=payload))


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def _history_params(
    *,
    session_id: str | None = None,
    disease: str | None = None,
    engine_key: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    params: dict[str, Any] = {k: v for k, v in extra.items() if v is not None}
    if session_id:
        params["session_id"] = session_id
    if disease:
        params["disease"] = disease
    if engine_key:
        params["engine_key"] = engine_key
    return params


def consultations(
    *,
    limit: int = 50,
    offset: int = 0,
    session_id: str | None = None,
    disease: str | None = None,
    engine_key: str | None = None,
) -> dict:
    params = _history_params(
        session_id=session_id, disease=disease, engine_key=engine_key, limit=limit, offset=offset
    )
    return _json(_request("GET", "/consultations", timeout=15.0, params=params))


def stats() -> dict:
    return _json(_request("GET", "/consultations/stats", timeout=15.0))


def export_csv(
    *,
    limit: int = 1000,
    session_id: str | None = None,
    disease: str | None = None,
    engine_key: str | None = None,
) -> bytes:
    params = _history_params(
        session_id=session_id, disease=disease, engine_key=engine_key, limit=limit
    )
    return _request("GET", "/consultations/export", timeout=30.0, params=params).content


def consultation(consultation_id: str) -> dict:
    return _json(_request("GET", f"/consultations/{consultation_id}", timeout=15.0))


def audit_trail(consultation_id: str) -> dict:
    return _json(_request("GET", f"/consultations/{consultation_id}/audit", timeout=15.0))


def report(consultation_id: str) -> dict:
    return _json(_request("GET", f"/consultations/{consultation_id}/report", timeout=15.0))
