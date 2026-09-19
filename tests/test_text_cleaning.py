"""Tests for the text cleaner and the shared symptom lexicon.

The protected stop-word list and the medical phrase normalisations are the most
genuinely domain-specific code in the project. They are also exactly the sort of
thing a well-meaning refactor deletes -- "why is `no` in a protected list?" --
so the reason each one exists is asserted here rather than only commented.
"""

from __future__ import annotations

import pytest

from app.core.lexicon import canonical_symptoms, cleaned_forms, source_forms
from app.utils.text_cleaning import PHRASE_NORMALIZATIONS, PROTECTED_WORDS, clean_text


def test_negation_words_survive_cleaning():
    """If the cleaner strips "no", "no fever" becomes "fever"."""
    for word in ("no", "not", "never", "without"):
        assert word in PROTECTED_WORDS
        assert word in clean_text(f"I have {word} fever").split()


@pytest.mark.parametrize(
    "word",
    ["blood", "breath", "chest", "cough", "fever", "pain", "rash", "weight", "loss", "smell", "taste"],
)
def test_clinically_loaded_words_are_never_stripped(word):
    """These are all in sklearn's English stop-word list and all carry meaning here."""
    assert word in PROTECTED_WORDS
    assert word in clean_text(f"the {word} is a problem").split()


@pytest.mark.parametrize(
    "raw,expected_fragment",
    [
        ("COVID-19", "covid"),
        ("coronavirus", "covid"),
        ("SARS-CoV-2", "covid"),
        ("diarrhoea", "diarrhea"),
        ("myalgia", "muscle pain"),
        ("photophobia", "light sensitivity"),
        ("conjunctivitis", "red watery eyes"),
        ("shortness of breath", "shortness breath"),
        ("stomach pain", "abdominal pain"),
        ("stuffy nose", "nasal congestion"),
        ("yellow eyes", "jaundice"),
        ("high temperature", "fever"),
    ],
)
def test_medical_phrases_are_normalised(raw, expected_fragment):
    assert expected_fragment in clean_text(raw)


def test_the_phrase_normalisation_table_has_not_been_gutted():
    assert len(PHRASE_NORMALIZATIONS) >= 27, (
        "The medical phrase normalisations are real domain work; "
        f"only {len(PHRASE_NORMALIZATIONS)} remain."
    )


def test_the_protected_word_list_has_not_been_gutted():
    assert len(PROTECTED_WORDS) >= 25, f"Only {len(PROTECTED_WORDS)} protected words remain."


def test_cleaning_is_idempotent():
    once = clean_text("I have had a HIGH TEMPERATURE and diarrhoea since Monday!")
    assert clean_text(once) == once


def test_cleaning_strips_accents_and_punctuation():
    assert clean_text("fièvre, céphalée!") == clean_text("fievre cephalee")


def test_cleaning_handles_empty_and_noise_input():
    assert clean_text("") == ""
    assert clean_text("!!! ???") == ""


def test_lexicon_views_agree_on_their_canonical_names():
    """The whole point of the shared file is that these cannot drift apart."""
    canonicals = set(canonical_symptoms())
    assert set(source_forms()) <= canonicals
    assert set(cleaned_forms()) <= canonicals
    assert canonicals, "The lexicon is empty"


def test_cleaned_forms_survive_the_cleaner():
    """A cleaned_form that clean_text would rewrite could never match anything."""
    broken = []
    for canonical, forms in cleaned_forms().items():
        for form in forms:
            if clean_text(form) != form:
                broken.append(f"{canonical}: {form!r} becomes {clean_text(form)!r}")

    assert not broken, "cleaned_forms entries that clean_text rewrites:\n" + "\n".join(broken)
