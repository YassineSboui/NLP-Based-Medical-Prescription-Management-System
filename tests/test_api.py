"""End-to-end API contract tests."""

from __future__ import annotations

import csv
import io

import pytest

SAMPLE = "I have fever, headache, chills, sweating and body pain for 3 days."


def test_health_reports_the_truth_about_itself(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database_ready"] is True
    # Nobody should have to guess whether the endpoint is protected.
    assert body["auth_required"] is False
    assert body["engines_available"]["classical"] is True


def test_engines_endpoint_documents_the_policy(client):
    body = client.get("/engines").json()
    assert body["ensemble_policy"]
    assert body["policy_description"]
    assert 0.0 <= body["model_confidence_floor"] <= 1.0

    keys = {engine["key"] for engine in body["engines"]}
    assert {"classical", "advanced", "symptom_profile"} <= keys

    for engine in body["engines"]:
        assert engine["confidence_meaning"], f"{engine['key']} does not say what its confidence means"
        assert engine["metrics"]["basis"]


def test_models_alias_still_works_for_existing_clients(client):
    """A deprecated alias that changed shape is just a broken endpoint."""
    legacy = client.get("/models").json()
    current = client.get("/engines").json()

    assert legacy["models"] == current["engines"]
    assert legacy["default_model"] == current["default_engine"]
    # The new keys are present too, so a client can migrate without switching path.
    assert legacy["engines"] == current["engines"]


def test_analyze_returns_the_full_contract(client):
    body = client.post("/analyze", json={"text": SAMPLE}).json()

    for field in (
        "consultation_id",
        "session_id",
        "created_at",
        "original_text",
        "cleaned_text",
        "extracted_entities",
        "predicted_disease",
        "confidence",
        "decision",
        "engine_opinions",
        "model_used",
        "recommended_actions",
        "recommended_medicines",
        "disclaimer",
    ):
        assert field in body, f"Missing contract field {field!r}"

    assert body["disclaimer"], "Every response must carry the safety disclaimer"
    for medication in body["recommended_medicines"]:
        assert medication["warnings"], f"{medication['name']} ships with no warnings"


def test_analyze_rejects_text_that_is_too_short(client):
    assert client.post("/analyze", json={"text": "hi"}).status_code == 422


def test_analyze_can_skip_persistence(client):
    body = client.post("/analyze", json={"text": SAMPLE, "persist": False}).json()
    assert body["consultation_id"] is None


def test_session_id_is_echoed_and_groups_records(client):
    session_id = "test-session-grouping"
    for _ in range(2):
        client.post("/analyze", json={"text": SAMPLE, "session_id": session_id})

    body = client.get("/consultations", params={"session_id": session_id}).json()
    assert body["total"] == 2
    assert all(item["session_id"] == session_id for item in body["items"])


def test_consultation_detail_and_audit_trail(client):
    created = client.post(
        "/analyze", json={"text": SAMPLE, "patient_reference": "case-42"}
    ).json()
    consultation_id = created["consultation_id"]

    detail = client.get(f"/consultations/{consultation_id}").json()
    assert detail["patient_reference"] == "case-42"
    assert detail["predicted_disease"] == created["predicted_disease"]
    assert detail["engine_key"] == created["decision"]["decided_by"]["key"]
    assert detail["policy_reason"]

    audit = client.get(f"/consultations/{consultation_id}/audit").json()
    assert len(audit["events"]) >= 1

    event = audit["events"][0]
    assert event["engine_key"] == detail["engine_key"]
    assert event["predicted_disease"] == detail["predicted_disease"]
    # The snapshot is what makes the trail reconstructable after a retrain.
    assert event["engine_metrics_basis"]
    assert event["suggested_medications"] == [
        item["name"] for item in detail["recommended_medicines"]
    ]


def test_missing_consultation_is_a_404(client):
    assert client.get("/consultations/does-not-exist").status_code == 404
    assert client.get("/consultations/does-not-exist/audit").status_code == 404
    assert client.get("/consultations/does-not-exist/report").status_code == 404


def test_report_leads_with_attribution_and_caveats(client):
    consultation_id = client.post("/analyze", json={"text": SAMPLE}).json()["consultation_id"]
    report = client.get(f"/consultations/{consultation_id}/report").json()

    assert report["format"] == "markdown"
    content = report["content"]
    assert "Not a diagnosis" in content
    assert "Which engine answered" in content
    # The caveat has to appear before the result, or a printout reads as a diagnosis.
    assert content.index("Not a diagnosis") < content.index("## Result")
    assert "Audit trail" in content


def test_batch_processes_several_notes(client):
    notes = [
        SAMPLE,
        "I have severe watery diarrhea, vomiting, thirst and dehydration after drinking unsafe water.",
        "I have runny nose, sneezing, sore throat, nasal congestion and mild cough.",
    ]
    body = client.post("/analyze/batch", json={"notes": notes}).json()

    assert body["submitted"] == 3
    assert body["succeeded"] == 3
    assert body["failed"] == 0
    assert len({item["result"]["consultation_id"] for item in body["results"]}) == 3
    assert all(item["result"]["session_id"] == body["session_id"] for item in body["results"])


def test_one_bad_note_does_not_fail_the_whole_batch(client):
    body = client.post("/analyze/batch", json={"notes": [SAMPLE, "x"]}).json()

    assert body["succeeded"] == 1
    assert body["failed"] == 1
    failed = next(item for item in body["results"] if not item["ok"])
    assert failed["error"]
    assert failed["result"] is None


def test_batch_size_is_capped(client):
    response = client.post("/analyze/batch", json={"notes": [SAMPLE] * 200})
    assert response.status_code == 422


def test_csv_export_carries_the_attribution(client):
    session_id = "test-session-export"
    client.post("/analyze", json={"text": SAMPLE, "session_id": session_id})

    response = client.get("/consultations/export", params={"session_id": session_id})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]

    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 1
    row = rows[0]
    assert row["decided_by_engine"]
    assert row["engine_accuracy"]
    assert row["engine_metrics_basis"]
    assert row["predicted_disease"]


def test_history_can_be_filtered_and_paged(client):
    body = client.get("/consultations", params={"limit": 2, "offset": 0}).json()
    assert body["limit"] == 2
    assert len(body["items"]) <= 2
    assert body["total"] >= len(body["items"])

    filtered = client.get("/consultations", params={"engine_key": "classical"}).json()
    assert all(item["engine_key"] == "classical" for item in filtered["items"])


def test_stats_summarise_the_history(client):
    body = client.get("/consultations/stats").json()
    assert body["total_consultations"] > 0
    assert body["total_sessions"] > 0
    assert sum(body["by_engine"].values()) == body["total_consultations"]
    assert sum(body["by_disease"].values()) == body["total_consultations"]


def test_cors_is_not_a_wildcard():
    """The old config paired `*` with credentials on an unauthenticated endpoint."""
    from app.core.config import get_settings

    origins = get_settings().cors_origins
    assert "*" not in origins
    assert origins, "CORS origins must be configured, not empty"


def test_api_key_is_enforced_when_configured(monkeypatch, tmp_path):
    """Auth is off by default, but must actually work once a key is set."""
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.db.session import reset_state

    monkeypatch.setenv("MEDICAL_NLP_API_KEY", "secret-key")
    monkeypatch.setenv("MEDICAL_NLP_DB_URL", f"sqlite:///{(tmp_path / 'auth.db').as_posix()}")
    get_settings.cache_clear()
    reset_state()

    try:
        from app.main import create_app

        with TestClient(create_app()) as secured:
            assert secured.get("/health").json()["auth_required"] is True
            assert secured.post("/analyze", json={"text": SAMPLE}).status_code == 401
            assert secured.get("/consultations").status_code == 401

            allowed = secured.post(
                "/analyze", json={"text": SAMPLE}, headers={"X-API-Key": "secret-key"}
            )
            assert allowed.status_code == 200
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
        reset_state()


@pytest.mark.parametrize("model_key", ["classical", "advanced", "nonsense"])
def test_model_key_is_always_resolved_to_something_real(client, model_key):
    body = client.post("/analyze", json={"text": SAMPLE, "model_key": model_key}).json()
    assert body["decision"]["decided_by"]["key"] in {"classical", "advanced", "symptom_profile"}
