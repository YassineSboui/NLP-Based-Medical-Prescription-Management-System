"""Shared request dependencies."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce ``X-API-Key`` when, and only when, a key is configured.

    The endpoint that hands out medication information used to have no auth at
    all and ``allow_origins=["*"]``. Demanding a key unconditionally would break
    the offline demo, and baking in a default key would be worse than no key, so
    this is opt-in through ``MEDICAL_NLP_API_KEY`` and ``/health`` advertises
    whether it is on.
    """
    settings = get_settings()
    if not settings.auth_required:
        return

    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid X-API-Key header is required.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )
