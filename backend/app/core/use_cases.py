"""Loader for ``tests/use_case_tests.txt``.

That file is a set of scenarios written by hand, independently of the dataset
builder and of any source page. Nothing in it was generated from the CDC/WHO
captures, so it is the only genuinely external evaluation signal this project
has. It is small (14 labelled sentences), which is why every number derived
from it is reported with a confidence interval rather than on its own.

The file stays plain text because it doubles as the demo script. This module is
the single parser for it, shared by the training pipeline and the test suite, so
the two can never disagree about what the expected answer is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.paths import USE_CASES_PATH

INPUT_PATTERN = re.compile(r"^Input:\s*(.+)$")
EXPECTED_PATTERN = re.compile(r"^Expected:\s*(.+)$")


@dataclass(frozen=True)
class UseCase:
    """One scenario: free text in, expected label out."""

    text: str
    expected: str


def load_use_cases(labels: set[str] | None = None) -> list[UseCase]:
    """Parse the scenario file into (text, expected) pairs.

    ``labels`` filters to scenarios whose expectation is an actual class name.
    The file also contains prose expectations -- the negation case expects
    "extract cough and chest pain, but not fever or chills" -- which are
    meaningful to a human and to the entity-extraction tests, but are not
    classification targets. Passing the label set drops those.
    """
    lines = USE_CASES_PATH.read_text(encoding="utf-8").splitlines()
    cases: list[UseCase] = []
    pending_text: str | None = None

    for line in lines:
        stripped = line.strip()

        input_match = INPUT_PATTERN.match(stripped)
        if input_match:
            pending_text = input_match.group(1).strip()
            continue

        expected_match = EXPECTED_PATTERN.match(stripped)
        if expected_match and pending_text is not None:
            expected = expected_match.group(1).strip().rstrip(".")
            if labels is None or expected in labels:
                cases.append(UseCase(text=pending_text, expected=expected))
            pending_text = None

    return cases
