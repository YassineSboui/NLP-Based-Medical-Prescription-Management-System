"""Runtime configuration, all of it from environment variables.

Two defaults are deliberate and worth knowing about.

**CORS is not ``*``.** The previous setting was ``allow_origins=["*"]`` together
with ``allow_credentials=True`` on an unauthenticated endpoint that returns
medication suggestions. The default is now the local Streamlit and dev-server
origins, and anything else has to be named in ``MEDICAL_NLP_CORS_ORIGINS``.

**Authentication is opt-in, and says so.** Requiring a key by default would
break the offline single-machine demo this project is built for, and shipping a
default key would be worse than none. So ``MEDICAL_NLP_API_KEY`` is unset by
default, ``/health`` reports ``auth_required: false`` so nobody is misled about
it, and setting the variable turns key checking on for every endpoint that
reads or writes consultation data.

Everything here works offline. No configuration value points at an external
service, and there is nothing to sign up for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from app.core.paths import PROJECT_DIR

DEFAULT_CORS_ORIGINS = (
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


def _split_env_list(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    database_url: str
    cors_origins: list[str] = field(default_factory=list)
    api_key: str | None = None
    max_batch_items: int = 50
    max_text_length: int = 4000

    @property
    def auth_required(self) -> bool:
        return bool(self.api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    default_db_path = PROJECT_DIR / "data" / "medical_nlp.db"
    default_db_path.parent.mkdir(parents=True, exist_ok=True)

    return Settings(
        database_url=os.getenv("MEDICAL_NLP_DB_URL", f"sqlite:///{default_db_path.as_posix()}"),
        cors_origins=_split_env_list("MEDICAL_NLP_CORS_ORIGINS", DEFAULT_CORS_ORIGINS),
        api_key=os.getenv("MEDICAL_NLP_API_KEY") or None,
        max_batch_items=int(os.getenv("MEDICAL_NLP_MAX_BATCH", "50")),
        max_text_length=int(os.getenv("MEDICAL_NLP_MAX_TEXT_LENGTH", "4000")),
    )
