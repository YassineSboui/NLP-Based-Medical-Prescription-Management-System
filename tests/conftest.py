"""Shared fixtures.

Every test runs against a throwaway SQLite file so a test run can never touch
the developer's real consultation history, and so the suite starts from a known
empty database.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(scope="session", autouse=True)
def _isolated_database(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point the app at a temporary database before anything imports it."""
    database_path = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["MEDICAL_NLP_DB_URL"] = f"sqlite:///{database_path.as_posix()}"
    os.environ.pop("MEDICAL_NLP_API_KEY", None)

    from app.core.config import get_settings
    from app.db.session import reset_state

    get_settings.cache_clear()
    reset_state()


@pytest.fixture(scope="session")
def client(_isolated_database: None):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def dataset():
    import pandas as pd

    from app.core.paths import DATASET_PATH

    return pd.read_csv(DATASET_PATH)


@pytest.fixture(scope="session")
def labels(dataset) -> set[str]:
    return set(dataset["disease"].unique())


@pytest.fixture(scope="session")
def use_cases(labels: set[str]):
    from app.core.use_cases import load_use_cases

    return load_use_cases(labels=labels)
