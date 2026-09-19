"""Regression tests for the silent-override attribution bug.

The bug: a hidden rule engine decided the answer whenever its synthesised
confidence beat the classifier's probability, while the response kept reporting
``model_used: complement_naive_bayes`` with ``accuracy: 0.906``. The UI showed
the ML model's metrics for a prediction the ML model did not make.

What makes that bug catchable is the invariant it violated. The engine named in
the response must be the engine whose opinion the response is repeating, and the
metrics attached must be that engine's own. Every test in this file checks a
face of that invariant, so the bug cannot come back in any of its forms:
rewritten arbitration, an added engine, or a refactor that reintroduces a
default attribution.
"""

from __future__ import annotations

import json

import pytest

from app.core.paths import ENGINE_METRICS_PATH
from app.services.disease_prediction_service import (
    MODEL_CONFIDENCE_FLOOR,
    DiseasePredictionService,
)
from app.services.prediction_engines import (
    CLASSICAL_ENGINE,
    SYMPTOM_PROFILE_ENGINE,
    EngineOutcome,
)


@pytest.fixture(scope="module")
def service() -> DiseasePredictionService:
    return DiseasePredictionService()


def test_decision_matches_the_opinion_of_the_engine_it_credits(service, use_cases):
    """The headline invariant: credited engine == engine that produced the answer.

    This is the assertion the old code could not have passed. There, the rule
    engine supplied `predicted_disease` while `model_used` described the
    classifier, so the credited engine's opinion and the returned answer came
    apart on exactly the inputs where the rule engine won.
    """
    for case in use_cases:
        decision = service.predict(case.text)
        credited = decision.decided_by.key

        if decision.abstained:
            continue

        opinion = next(item for item in decision.opinions if item.engine == credited)
        assert opinion.disease == decision.predicted_disease, (
            f"Response credits {credited!r} but that engine said {opinion.disease!r} "
            f"while the response says {decision.predicted_disease!r}: {case.text!r}"
        )
        assert opinion.confidence == pytest.approx(decision.confidence), (
            f"Response credits {credited!r} but reports a confidence that engine did not give."
        )


def test_reported_metrics_belong_to_the_deciding_engine(service, use_cases, monkeypatch):
    """The metrics shipped with an answer must be the deciding engine's own.

    Forces the rule engine to decide by making the classical engine report a
    confidence under the floor, then checks that what comes back is the rule
    engine's accuracy and not the classifier's.
    """
    measured = json.loads(ENGINE_METRICS_PATH.read_text(encoding="utf-8"))["engines"]
    classical_accuracy = measured[CLASSICAL_ENGINE]["accuracy"]
    rule_accuracy = measured[SYMPTOM_PROFILE_ENGINE]["accuracy"]
    assert classical_accuracy != rule_accuracy, "Test cannot distinguish the engines"

    def unsure(_cleaned_text: str) -> EngineOutcome:
        return EngineOutcome(
            engine_key=CLASSICAL_ENGINE,
            disease="malaria",
            confidence=MODEL_CONFIDENCE_FLOOR - 0.01,
            available=True,
            detail="forced low confidence",
        )

    monkeypatch.setattr(service.engines[CLASSICAL_ENGINE], "predict", unsure)

    decision = service.predict("I have severe watery diarrhea, vomiting and dehydration.")

    assert decision.decided_by.key == SYMPTOM_PROFILE_ENGINE
    assert decision.decided_by.metrics.accuracy == rule_accuracy
    assert decision.decided_by.metrics.accuracy != classical_accuracy, (
        "The rule engine decided but the classifier's accuracy was returned. "
        "This is the exact bug this suite exists to prevent."
    )
    assert "symptom-profile" in decision.policy_reason.lower()


def test_rule_engine_confidence_is_not_presented_as_a_probability(service):
    """It is a match share. Saying otherwise is what made `>=` look reasonable."""
    info = service.engine_info(SYMPTOM_PROFILE_ENGINE)
    assert "not a probability" in info.confidence_meaning.lower()

    classical = service.engine_info(CLASSICAL_ENGINE)
    assert "probability" in classical.confidence_meaning.lower()


def test_every_engine_carries_its_own_distinct_metrics(service):
    """No engine may inherit another's numbers."""
    accuracies = {}
    for key in service.engines:
        info = service.engine_info(key)
        if info.is_available and info.metrics.accuracy is not None:
            accuracies[key] = info.metrics.accuracy

    assert len(accuracies) >= 2, "Need at least two evaluated engines to test this"
    assert len(set(accuracies.values())) == len(accuracies), (
        f"Two engines report identical accuracy, which suggests shared metrics: {accuracies}"
    )


def test_api_response_states_which_engine_decided(client):
    response = client.post(
        "/analyze",
        json={"text": "I have fever, headache, chills, sweating and body pain for 3 days."},
    )
    assert response.status_code == 200
    body = response.json()

    credited = body["decision"]["decided_by"]["key"]
    assert credited in {item["engine"] for item in body["engine_opinions"]}

    # model_used is a compatibility alias and must not drift from decided_by.
    assert body["model_used"]["key"] == credited
    assert body["model_used"]["metrics"]["accuracy"] == body["decision"]["decided_by"]["metrics"]["accuracy"]

    opinion = next(item for item in body["engine_opinions"] if item["engine"] == credited)
    assert opinion["disease"] == body["predicted_disease"]

    assert body["decision"]["policy_reason"], "A decision must explain why that engine decided"


def test_metrics_say_what_they_were_measured_on(service):
    for key in service.engines:
        info = service.engine_info(key)
        assert info.metrics.basis, f"{key} publishes metrics with no stated basis"


def test_abstention_is_reported_rather_than_guessed(service, monkeypatch):
    """When nothing is confident, the API says so instead of naming a disease."""

    def unsure(_cleaned_text: str) -> EngineOutcome:
        return EngineOutcome(
            engine_key=CLASSICAL_ENGINE,
            disease="malaria",
            confidence=0.01,
            available=True,
            detail="forced low confidence",
        )

    monkeypatch.setattr(service.engines[CLASSICAL_ENGINE], "predict", unsure)

    # Text with no recognisable symptom, so the rule engine has no opinion either.
    decision = service.predict("the quick brown fox jumped over the lazy dog")

    assert decision.abstained is True
    assert decision.predicted_disease == "unknown"
    assert "abstain" in decision.policy_reason.lower()


def test_requesting_an_unavailable_engine_falls_back_and_says_so(service):
    """A bogus model_key must not silently produce a mislabelled answer."""
    decision = service.predict("I have fever and chills and a headache.", model_key="does-not-exist")
    assert decision.decided_by.key in {CLASSICAL_ENGINE, SYMPTOM_PROFILE_ENGINE}
    assert service.resolve_primary("does-not-exist") == CLASSICAL_ENGINE
