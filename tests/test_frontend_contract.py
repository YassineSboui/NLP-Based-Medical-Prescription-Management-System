"""The two things the console can get wrong without anyone noticing.

The console is not unit-testable in any interesting way -- it is layout over a
service that already has 105 tests. These two are the exceptions, because both
fail silently.

The first is the service address. It moved out of an editable text box and into
the environment, and `run_app.ps1` used to export the `.../analyze` form. An
environment left over from a previous version, or a developer who copies the
old value out of habit, must not end up with a console quietly requesting
`/analyze/analyze` and reporting the service as unreachable.

The second is that the console must never invent an engine. The previous
frontend answered an unreachable service with a hardcoded engine carrying a
hardcoded accuracy, which is how a withdrawn number stayed on screen long after
it had been withdrawn everywhere else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

from ui import api  # noqa: E402
from ui.components import title_case  # noqa: E402

ADDRESS_VARIABLES = ("MEDICAL_NLP_API_BASE_URL", "MEDICAL_NLP_API_URL", "API_URL")


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ADDRESS_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_the_default_address_is_the_documented_one() -> None:
    assert api.resolve_base_url() == "http://localhost:8000"


@pytest.mark.parametrize("variable", ADDRESS_VARIABLES)
def test_every_supported_variable_is_read(variable: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(variable, "http://127.0.0.1:8010")
    assert api.resolve_base_url() == "http://127.0.0.1:8010"


@pytest.mark.parametrize(
    "configured",
    [
        "http://localhost:8000/analyze",
        "http://localhost:8000/analyze/",
        "http://localhost:8000/engines",
        "http://localhost:8000/models",
    ],
)
def test_a_legacy_endpoint_value_resolves_to_the_base(
    configured: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_app.ps1 used to export the /analyze form. It must still work."""
    monkeypatch.setenv("MEDICAL_NLP_API_URL", configured)
    assert api.resolve_base_url() == "http://localhost:8000"


def test_a_bare_host_and_port_is_assumed_to_be_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDICAL_NLP_API_URL", "localhost:8000")
    assert api.resolve_base_url() == "http://localhost:8000"


def test_nonsense_falls_back_to_the_default_rather_than_to_an_unusable_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDICAL_NLP_API_URL", "   ")
    assert api.resolve_base_url() == "http://localhost:8000"


def test_an_unreachable_service_yields_no_engines_rather_than_invented_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression that let a withdrawn accuracy survive on screen.

    There is no fallback engine list. Asking an unreachable service for its
    engines raises; it does not return a plausible-looking fiction.
    """
    monkeypatch.setenv("MEDICAL_NLP_API_URL", "http://127.0.0.1:1")

    with pytest.raises(api.ServiceUnavailable):
        api.engines()

    status = api.status()
    assert status.reachable is False
    assert status.engine_count == 0


def test_stored_labels_are_displayed_the_way_a_clinician_writes_them() -> None:
    assert title_case("hepatitis b") == "Hepatitis B"
    assert title_case("hiv") == "HIV"
    assert title_case("covid-like illness") == "COVID-like illness"
    assert title_case("malaria") == "Malaria"
    assert title_case("unknown") == "Unknown"
    assert title_case("") == "Unknown"
