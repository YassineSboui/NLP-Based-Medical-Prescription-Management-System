"""Access to the shared symptom vocabulary.

Before this module there were two independent symptom vocabularies: one inside
the dataset builder for matching CDC/WHO prose, and one inside ``nlp_service``
for matching user input. They disagreed -- "pain behind eyes" against "pain
behind the eyes" -- which meant the symptom the dataset recorded and the symptom
the API extracted could be different strings for the same thing, and the rule
engine's hand-written profiles agreed with neither.

Both views now come from ``app/data/symptom_lexicon.json``:

``source_forms``
    surface spellings as they appear in published source prose, used when
    deciding what a fetched passage attests.

``cleaned_forms``
    surface spellings as they survive :func:`app.utils.text_cleaning.clean_text`,
    used when deciding what a user's sentence mentions.
"""

from __future__ import annotations

import json
from functools import lru_cache

from app.core.paths import SYMPTOM_LEXICON_PATH


@lru_cache(maxsize=1)
def _document() -> dict:
    return json.loads(SYMPTOM_LEXICON_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def source_forms() -> dict[str, tuple[str, ...]]:
    """canonical symptom -> spellings to look for in published source prose."""
    return {
        canonical: tuple(entry["source_forms"])
        for canonical, entry in _document()["symptoms"].items()
        if entry["source_forms"]
    }


@lru_cache(maxsize=1)
def cleaned_forms() -> dict[str, tuple[str, ...]]:
    """canonical symptom -> spellings to look for in cleaned user text."""
    return {
        canonical: tuple(entry["cleaned_forms"])
        for canonical, entry in _document()["symptoms"].items()
        if entry["cleaned_forms"]
    }


@lru_cache(maxsize=1)
def canonical_symptoms() -> tuple[str, ...]:
    return tuple(_document()["symptoms"])
