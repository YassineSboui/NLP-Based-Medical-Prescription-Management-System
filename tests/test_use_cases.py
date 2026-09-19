"""The scenarios in ``tests/use_case_tests.txt``, executed rather than read.

That file held 14 labelled scenarios plus a negation case, and nothing ran
them -- pytest was not even a dependency. They are the only evaluation material
in the repository that was not derived from the source captures, which makes
them the closest thing to an external check this project has.
"""

from __future__ import annotations

import pytest

from app.services.nlp_service import NLPService
from app.utils.text_cleaning import clean_text


@pytest.fixture(scope="module")
def nlp() -> NLPService:
    return NLPService()


def test_the_scenario_file_actually_parses(use_cases, labels):
    assert len(use_cases) >= 14, f"Expected at least 14 labelled scenarios, parsed {len(use_cases)}"
    for case in use_cases:
        assert case.expected in labels
        assert len(case.text) > 10


def test_every_documented_label_has_a_scenario(use_cases, labels):
    covered = {case.expected for case in use_cases}
    assert covered == labels, f"Labels with no scenario: {sorted(labels - covered)}"


def test_documented_scenarios_still_produce_their_documented_answer(nlp, use_cases):
    """If this fails, either the model regressed or the docs are now lying.

    Kept as a hard assertion rather than a tolerance, because these examples are
    published in the README and in tests/use_case_tests.txt as what the system
    does. A drift here has to be either fixed or documented, not averaged away.
    """
    failures = []
    for case in use_cases:
        analysis = nlp.analyze(case.text)
        if analysis.decision.predicted_disease != case.expected:
            failures.append(
                f"{case.text!r}\n    expected {case.expected!r}, "
                f"got {analysis.decision.predicted_disease!r} "
                f"from {analysis.decision.decided_by.key!r}"
            )

    assert not failures, "Documented scenarios no longer hold:\n  " + "\n  ".join(failures)


def test_negation_case_from_the_scenario_file(nlp):
    """The lookbehind in _has_positive_match is crude but load-bearing.

    "I have cough and chest pain but no fever and no chills" must not extract
    fever or chills. Without this, every sentence that rules a symptom out would
    count as reporting it.
    """
    entities = nlp.extract_entities(clean_text("I have cough and chest pain but no fever and no chills."))

    assert "cough" in entities.symptoms
    assert "chest pain" in entities.symptoms
    assert "fever" not in entities.symptoms
    assert "chills" not in entities.symptoms


@pytest.mark.parametrize(
    "text,expected_absent",
    [
        ("I have a rash without fever", "fever"),
        ("patient denies vomiting but has diarrhea", "vomiting"),
        ("no cough, just a sore throat", "cough"),
        ("not experiencing headache, mainly nausea", "headache"),
        ("denies having chest pain, reports fatigue", "chest pain"),
        ("negative for jaundice, has abdominal pain", "jaundice"),
    ],
)
def test_negation_holds_for_other_phrasings(nlp, text, expected_absent):
    entities = nlp.extract_entities(clean_text(text))
    assert expected_absent not in entities.symptoms, f"{expected_absent!r} extracted from {text!r}"


@pytest.mark.parametrize(
    "text,expected_present",
    [
        # The symptom after the negated one must survive. A negation window that
        # is too wide suppresses real symptoms, which is worse than missing a
        # negation.
        ("no fever but severe headache", "headache"),
        ("no fever, severe headache and a rash", "rash"),
        ("without cough, but vomiting and nausea", "vomiting"),
        ("no cough yesterday, cough today", "cough"),
    ],
)
def test_negation_does_not_suppress_the_symptoms_that_follow(nlp, text, expected_present):
    entities = nlp.extract_entities(clean_text(text))
    assert expected_present in entities.symptoms, f"{expected_present!r} lost from {text!r}"


def test_dosage_mentions_are_extracted(nlp):
    entities = nlp.extract_entities(clean_text("I took paracetamol 500mg twice daily for 3 days."))
    assert "paracetamol" in entities.medications
    assert any("500" in mention for mention in entities.dosage_mentions)


def test_mentioned_disease_triggers_a_comparison_note(nlp):
    """Naming a disease in the text must not be silently ignored."""
    analysis = nlp.analyze("I was told I have typhoid but I have watery diarrhea and dehydration.")
    assert "typhoid fever" in analysis.entities.diseases
    if analysis.decision.predicted_disease != "typhoid fever":
        assert any("typhoid" in action.lower() for action in analysis.recommended_actions)
