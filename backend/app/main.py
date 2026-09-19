"""Application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import API_VERSION, router
from app.core.config import get_settings
from app.db.session import init_database

DESCRIPTION = """
Educational NLP API for symptom triage.

**This is not a diagnostic tool.** It suggests one of 14 common conditions from
free-text symptoms and returns educational medication information. Every
response carries a disclaimer and per-medication warnings, and every prediction
states which engine produced it along with that engine's measured accuracy.

Runs entirely offline. No API keys, no external services, no network calls at
request time.
"""


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="NLP-Based Medical Prescription Management API",
        description=DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
    )

    # Not `["*"]`. The previous configuration paired a wildcard origin with
    # allow_credentials on an unauthenticated endpoint that returns medication
    # information. Origins are now explicit and configurable through
    # MEDICAL_NLP_CORS_ORIGINS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    app.include_router(router)
    return app


app = create_app()
