"""Rebuild ``symptoms_dataset.csv`` from the fetched CDC/WHO evidence base.

Why this script exists
----------------------
The dataset used to be hand-curated. Hand-curation is fine right up until
somebody asks "where does this row come from?", and for medical training data
that question has to have an answer for *every* row. Inventing a clinical
description and attributing it to CDC or WHO would be worse than having a small
dataset, so this builder is deliberately unable to do it.

Every row emitted here is one of exactly three kinds, recorded in the
``provenance`` column:

``source``
    The full set of symptom terms a single quoted passage attests for this
    disease, written out in the order the passage mentions them. Nothing added.

``source_combination``
    A subset of the terms attested by one quoted passage. The clinical content
    is the source's; only the choice of which listed symptoms co-occur in this
    example is synthetic. The quote that attests every term is stored on the row.

``paraphrase``
    A wording variant of one specific other row, named in ``derived_from``.
    Only the surface form changes -- templates and a synonym table that are both
    defined in this file and unit-tested. No symptom is ever added, removed or
    upgraded in severity, so the clinical claim is identical to the parent's.

The hard guarantee, enforced here and re-checked by ``tests/test_dataset.py``:
for every ``source`` and ``source_combination`` row, each surface term in the
row text occurs verbatim in that row's own ``source_quote``. A row whose terms
cannot be found in fetched text is dropped, not repaired.

``group_id`` exists for the evaluation pipeline. A paraphrase and its parent are
near-duplicates; letting them straddle a train/test boundary would inflate the
score. ``group_id`` keeps a row and all of its paraphrases on the same side of
every split.

Usage:
    python backend/scripts/build_dataset.py            # rebuild the CSV
    python backend/scripts/build_dataset.py --report   # print coverage, write nothing
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BACKEND_DIR))

from app.core.lexicon import source_forms  # noqa: E402
from app.core.paths import DATASET_PATH, SCRAPED_SOURCES_PATH  # noqa: E402

# Deterministic output: the same evidence base must always produce the same CSV.
RANDOM_SEED = 20240517

# Per-label caps. They keep the label distribution flat even though WHO fact
# sheets are far wordier than CDC symptom pages, which would otherwise hand
# whichever disease has the chattiest source several times more rows.
MAX_COMBINATIONS_PER_LABEL = 16
MAX_PARAPHRASES_PER_LABEL = 26
MIN_TERMS_PER_ROW = 3
MAX_TERMS_PER_ROW = 6

DATASET_FIELDNAMES = [
    "row_id",
    "text",
    "disease",
    "provenance",
    "derived_from",
    "group_id",
    "symptom_terms",
    "source_name",
    "source_url",
    "source_section",
    "source_quote",
    "source_fetched_at",
    "source_note",
]

# Headings whose text we accept as attesting symptoms. Treatment and prevention
# prose mentions symptom words too ("seek care if you have prolonged cough"),
# but it is describing what to do, not what the presentation looks like, so it
# is not used as symptom evidence.
SYMPTOM_HEADING_PATTERN = re.compile(r"symptom|signs", re.IGNORECASE)

# A sentence anywhere on the page that explicitly enumerates symptoms is also
# accepted, because WHO routinely puts the real symptom list in "Overview".
SYMPTOM_SENTENCE_PATTERN = re.compile(
    r"symptoms?\s+(?:include|are|of\s+\w[\w\s-]*\s+(?:include|are)|may\s+include|usually\s+include|can\s+include)",
    re.IGNORECASE,
)

# The symptom vocabulary lives in app/data/symptom_lexicon.json so that the
# dataset builder, the entity extractor and the rule engine cannot drift apart.
# Matching always prefers the longest surface form, so "severe acute watery
# diarrhoea" wins over "diarrhoea".
SYMPTOM_LEXICON: dict[str, tuple[str, ...]] = source_forms()

# Lexical variants used only by paraphrase rows.
#
# The bar for an entry here is strict: it must be a different way of saying the
# same thing, with no change in what is being claimed. Intensifiers and
# specifiers are the failure mode to watch for, because they quietly relabel the
# example. Rendering a source's plain "cough" as "a cough that will not stop"
# pushes the row toward tuberculosis; rendering a source's plain "diarrhea" as
# "watery stools" pushes it toward cholera. Both were caught in review and are
# deliberately absent, along with "constant sneezing", "heavy sweating" and
# "bad headache".
PARAPHRASE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "fever": ("a fever", "a high temperature", "feeling feverish"),
    "chills": ("chills", "the chills", "shivering"),
    "headache": ("a headache", "head pain", "pain in my head"),
    "muscle pain": ("muscle aches", "aching muscles", "sore muscles"),
    "joint pain": ("joint pain", "aching joints", "pain in my joints"),
    "fatigue": ("fatigue", "tiredness", "feeling tired", "no energy"),
    "nausea": ("nausea", "feeling nauseous", "feeling sick to my stomach"),
    "vomiting": ("vomiting", "throwing up", "being sick"),
    "diarrhea": ("diarrhea", "loose stools"),
    "abdominal pain": ("abdominal pain", "stomach pain", "pain in my belly"),
    "constipation": ("constipation", "trouble passing stool"),
    "cough": ("a cough", "coughing"),
    "chest pain": ("chest pain", "pain in my chest"),
    "shortness of breath": ("shortness of breath", "difficulty breathing", "trouble catching my breath"),
    "night sweats": ("night sweats", "sweating at night", "waking up sweating at night"),
    "weight loss": ("weight loss", "losing weight"),
    "loss of appetite": ("loss of appetite", "no appetite", "not wanting to eat"),
    "sore throat": ("a sore throat", "throat pain"),
    "runny nose": ("a runny nose", "a nose that keeps running"),
    "nasal congestion": ("nasal congestion", "a blocked nose", "a stuffy nose"),
    "sneezing": ("sneezing", "sneezing a lot"),
    "loss of taste": ("loss of taste", "no sense of taste"),
    "loss of smell": ("loss of smell", "no sense of smell"),
    "rash": ("a rash", "a skin rash"),
    "red watery eyes": ("red watery eyes", "red eyes that water"),
    "koplik spots": ("koplik spots", "white spots inside the mouth"),
    "stiff neck": ("a stiff neck", "neck stiffness"),
    "light sensitivity": ("sensitivity to light", "light hurting my eyes"),
    "confusion": ("confusion", "feeling confused", "altered mental status"),
    "jaundice": ("jaundice", "yellowing of the skin and eyes", "yellow eyes"),
    "dark urine": ("dark urine", "urine that looks dark"),
    "bleeding": ("bleeding", "bleeding gums"),
    "blood in sputum": ("coughing up blood", "blood in my sputum"),
    "dehydration": ("dehydration", "feeling dehydrated"),
    "swollen glands": ("swollen glands", "swollen lymph nodes"),
    "pain behind the eyes": ("pain behind the eyes", "pain behind my eyes"),
    "mouth ulcers": ("mouth ulcers", "sores in my mouth"),
    "sweating": ("sweating", "sweats"),
    "seizures": ("seizures", "fits"),
    "restlessness": ("restlessness", "feeling restless"),
    "ear infection": ("an ear infection", "a painful ear"),
    "fast heart rate": ("a fast heart rate", "a racing heartbeat"),
    "body pain": ("body aches", "aching all over"),
    "muscle cramps": ("cramps", "muscle cramps", "leg cramps"),
    "dry mouth": ("a dry mouth",),
    "low blood pressure": ("low blood pressure",),
    "irritability": ("irritability", "feeling irritable"),
}

# Patient-voice frames. ``{terms}`` is filled with the synonym-rendered symptom
# list; ``{duration}`` with a neutral duration phrase. Frames never assert
# anything clinical beyond the parent row's symptom set.
#
# Templates are split by how they join a duration so the result stays
# grammatical: "for {duration}" plus "since yesterday" produced "for since
# yesterday" in the first cut of this file.
PARAPHRASE_TEMPLATES: tuple[str, ...] = (
    "i have {terms}",
    "i am experiencing {terms}",
    "patient reports {terms}",
    "my symptoms are {terms}",
    "came in complaining of {terms}",
    "he has {terms}",
    "my child has {terms}",
)

PARAPHRASE_TEMPLATES_WITH_DURATION: tuple[str, ...] = (
    "i have been having {terms} {duration}",
    "i have had {terms} {duration}",
    "i started with {terms} {duration}",
    "she has had {terms} {duration}",
    "{terms} {duration}",
)

DURATION_PHRASES: tuple[str, ...] = (
    "for the past two days",
    "for three days",
    "for about a week",
    "since yesterday",
    "for several days now",
    "for the last few days",
)


@dataclass
class Evidence:
    """One quoted passage that attests a set of symptoms for one disease."""

    disease: str
    source_name: str
    source_url: str
    section: str
    quote: str
    fetched_at: str
    terms: list[str] = field(default_factory=list)  # surface forms, order of appearance
    canonicals: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.disease}|{self.source_name}|{self.section}"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def build_matcher() -> list[tuple[str, re.Pattern[str]]]:
    """Surface form -> compiled pattern, longest first so specifics win."""
    pairs: list[tuple[str, str]] = []
    for canonical, surfaces in SYMPTOM_LEXICON.items():
        for surface in surfaces:
            pairs.append((canonical, surface))
    pairs.sort(key=lambda pair: len(pair[1]), reverse=True)
    return [(canonical, re.compile(rf"\b{re.escape(surface)}\b", re.IGNORECASE)) for canonical, surface in pairs]


MATCHER = build_matcher()


def find_attested_terms(quote: str) -> tuple[list[str], list[str]]:
    """Return (surface terms, canonical names) attested by ``quote``.

    Matches never overlap: once a span is claimed by a longer surface form, a
    shorter one cannot re-claim it. Results are ordered by position so the row
    text reads in the same order as the source passage.
    """
    claimed: list[tuple[int, int]] = []
    hits: list[tuple[int, str, str]] = []
    seen_canonicals: set[str] = set()

    for canonical, pattern in MATCHER:
        if canonical in seen_canonicals:
            continue
        for match in pattern.finditer(quote):
            start, end = match.span()
            if any(start < claimed_end and end > claimed_start for claimed_start, claimed_end in claimed):
                continue
            claimed.append((start, end))
            hits.append((start, canonical, match.group(0).lower()))
            seen_canonicals.add(canonical)
            break

    hits.sort(key=lambda hit: hit[0])
    return [hit[2] for hit in hits], [hit[1] for hit in hits]


def collect_evidence(records: list[dict]) -> list[Evidence]:
    """Turn fetched pages into quoted passages that attest symptoms."""
    evidence: list[Evidence] = []

    for record in records:
        if not record.get("extracted_sections"):
            continue

        disease = record["disease"]
        fetched_at = record.get("fetched_at") or ""

        for section in record["extracted_sections"]:
            heading = str(section["heading"])
            items = [str(item) for item in section["items"]]
            heading_is_symptoms = bool(SYMPTOM_HEADING_PATTERN.search(heading))

            if heading_is_symptoms:
                # The whole section is a symptom list; quote it entire so the
                # attestation for every term is visible on the row.
                usable = items
            else:
                usable = [item for item in items if SYMPTOM_SENTENCE_PATTERN.search(item)]

            if not usable:
                continue

            quote = " ".join(usable)
            terms, canonicals = find_attested_terms(quote)
            if len(terms) < MIN_TERMS_PER_ROW:
                continue

            evidence.append(
                Evidence(
                    disease=disease,
                    source_name=record["source_name"],
                    source_url=record["source_url"],
                    section=heading,
                    quote=quote,
                    fetched_at=fetched_at,
                    terms=terms,
                    canonicals=canonicals,
                )
            )

    return evidence


def make_row(
    row_id: str,
    text: str,
    disease: str,
    provenance: str,
    group_id: str,
    evidence: Evidence,
    note: str,
    canonicals: list[str],
    derived_from: str = "",
) -> dict[str, str]:
    return {
        "row_id": row_id,
        "text": text,
        "disease": disease,
        "provenance": provenance,
        "derived_from": derived_from,
        "group_id": group_id,
        "symptom_terms": "|".join(sorted(canonicals)),
        "source_name": evidence.source_name,
        "source_url": evidence.source_url,
        "source_section": evidence.section,
        "source_quote": evidence.quote,
        "source_fetched_at": evidence.fetched_at,
        "source_note": note,
    }


def build_source_rows(evidence_by_disease: dict[str, list[Evidence]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for disease in sorted(evidence_by_disease):
        for index, evidence in enumerate(evidence_by_disease[disease], start=1):
            row_id = f"{slug(disease)}-src-{index:03d}"
            rows.append(
                make_row(
                    row_id=row_id,
                    text=" ".join(evidence.terms),
                    disease=disease,
                    provenance="source",
                    group_id=row_id,
                    evidence=evidence,
                    canonicals=list(evidence.canonicals),
                    note=(
                        f"Full symptom set attested by the '{evidence.section}' section of "
                        f"{evidence.source_name}, in the order the passage lists them."
                    ),
                )
            )

    return rows


def build_combination_rows(
    evidence_by_disease: dict[str, list[Evidence]],
    rng: random.Random,
) -> list[dict[str, str]]:
    """Subsets of one passage's attested symptoms -- never across passages.

    Mixing terms from two passages would produce a row no single source
    supports, so combinations are always drawn from one quote.
    """
    rows: list[dict[str, str]] = []

    for disease in sorted(evidence_by_disease):
        evidence_list = evidence_by_disease[disease]
        seen_texts: set[str] = set()
        candidates: list[tuple[Evidence, tuple[str, ...]]] = []

        for evidence in evidence_list:
            term_count = len(evidence.terms)
            for size in range(MIN_TERMS_PER_ROW, min(term_count, MAX_TERMS_PER_ROW) + 1):
                if size >= term_count:
                    continue  # that is the `source` row, already emitted
                for _ in range(12):
                    subset = tuple(sorted(rng.sample(evidence.terms, size), key=evidence.terms.index))
                    candidates.append((evidence, subset))

        rng.shuffle(candidates)
        index = 0
        for evidence, subset in candidates:
            if index >= MAX_COMBINATIONS_PER_LABEL:
                break
            text = " ".join(subset)
            if text in seen_texts:
                continue
            seen_texts.add(text)
            index += 1
            row_id = f"{slug(disease)}-comb-{index:03d}"
            rows.append(
                make_row(
                    row_id=row_id,
                    text=text,
                    disease=disease,
                    provenance="source_combination",
                    group_id=row_id,
                    evidence=evidence,
                    canonicals=[evidence.canonicals[evidence.terms.index(term)] for term in subset],
                    note=(
                        f"Subset of the symptoms attested by the '{evidence.section}' section of "
                        f"{evidence.source_name}. Every term appears in source_quote; the "
                        f"co-occurrence shown here is synthetic."
                    ),
                )
            )

    return rows


def render_terms(canonicals: list[str], rng: random.Random) -> str:
    """Render a symptom list in patient voice using synonym variants only."""
    rendered = [rng.choice(PARAPHRASE_SYNONYMS[canonical]) for canonical in canonicals]
    if len(rendered) == 1:
        return rendered[0]
    return ", ".join(rendered[:-1]) + " and " + rendered[-1]


def build_paraphrase_rows(
    parents: list[dict[str, str]],
    evidence_by_key: dict[str, Evidence],
    rng: random.Random,
) -> list[dict[str, str]]:
    """One wording variant per call, always tied to exactly one parent row.

    The symptom set comes from the parent's recorded ``symptom_terms`` rather
    than from re-parsing its text, so a paraphrase cannot drift away from what
    its parent claims even if the surface-form splitter were wrong.
    """
    by_disease: dict[str, list[dict[str, str]]] = defaultdict(list)
    for parent in parents:
        by_disease[parent["disease"]].append(parent)

    rows: list[dict[str, str]] = []

    for disease in sorted(by_disease):
        parent_rows = by_disease[disease]
        seen_texts = {parent["text"] for parent in parent_rows}
        index = 0
        attempts = 0

        while index < MAX_PARAPHRASES_PER_LABEL and attempts < MAX_PARAPHRASES_PER_LABEL * 20:
            attempts += 1
            parent = parent_rows[attempts % len(parent_rows)]
            evidence = evidence_by_key[f"{parent['disease']}|{parent['source_name']}|{parent['source_section']}"]

            canonicals = parent["symptom_terms"].split("|")
            if len(canonicals) < MIN_TERMS_PER_ROW:
                continue
            rng.shuffle(canonicals)

            if rng.random() < 0.5:
                text = rng.choice(PARAPHRASE_TEMPLATES).format(terms=render_terms(canonicals, rng))
            else:
                text = rng.choice(PARAPHRASE_TEMPLATES_WITH_DURATION).format(
                    terms=render_terms(canonicals, rng),
                    duration=rng.choice(DURATION_PHRASES),
                )

            if text in seen_texts:
                continue
            seen_texts.add(text)
            index += 1

            rows.append(
                make_row(
                    row_id=f"{slug(disease)}-para-{index:03d}",
                    text=text,
                    disease=disease,
                    provenance="paraphrase",
                    group_id=parent["group_id"],
                    evidence=evidence,
                    canonicals=canonicals,
                    note=(
                        f"Wording variant of {parent['row_id']}. Same symptom set, patient-voice "
                        f"phrasing and synonyms only; no clinical content added."
                    ),
                    derived_from=parent["row_id"],
                )
            )

    return rows


def verify(rows: list[dict[str, str]]) -> list[str]:
    """Re-derive the integrity guarantee. Returns a list of problems."""
    problems: list[str] = []
    by_id = {row["row_id"]: row for row in rows}

    for row in rows:
        if row["provenance"] in {"source", "source_combination"}:
            quote = row["source_quote"].lower()
            for term in _split_surface_terms(row["text"]):
                if term not in quote:
                    problems.append(f"{row['row_id']}: term {term!r} not in source_quote")
        elif row["provenance"] == "paraphrase":
            parent = by_id.get(row["derived_from"])
            if parent is None:
                problems.append(f"{row['row_id']}: derived_from {row['derived_from']!r} does not exist")
            elif parent["disease"] != row["disease"]:
                problems.append(f"{row['row_id']}: label differs from parent {row['derived_from']}")
            elif parent["group_id"] != row["group_id"]:
                problems.append(f"{row['row_id']}: group differs from parent {row['derived_from']}")
            elif parent["symptom_terms"] != row["symptom_terms"]:
                problems.append(
                    f"{row['row_id']}: symptom set drifted from parent {row['derived_from']} "
                    f"({row['symptom_terms']} != {parent['symptom_terms']})"
                )
        else:
            problems.append(f"{row['row_id']}: unknown provenance {row['provenance']!r}")

        if not row["source_url"].startswith("https://"):
            problems.append(f"{row['row_id']}: source_url is not an https URL")

    return problems


def _split_surface_terms(text: str) -> list[str]:
    """Greedily recover the surface terms that were concatenated into ``text``.

    Row text for ``source`` / ``source_combination`` rows is a space-joined list
    of multi-word surface forms, so it cannot be split on whitespace: "muscle
    aches" is one term, not two. Longest-first matching against the lexicon
    recovers the original terms so each can be checked against the quote.
    """
    terms: list[str] = []
    remaining = text
    known = sorted({surface for surfaces in SYMPTOM_LEXICON.values() for surface in surfaces}, key=len, reverse=True)

    while remaining:
        for surface in known:
            if remaining == surface or remaining.startswith(surface + " "):
                terms.append(surface)
                remaining = remaining[len(surface):].strip()
                break
        else:
            head, _, remaining = remaining.partition(" ")
            terms.append(head)
    return terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="Print coverage and write nothing.")
    args = parser.parse_args()

    records = json.loads(SCRAPED_SOURCES_PATH.read_text(encoding="utf-8"))
    evidence = collect_evidence(records)

    evidence_by_disease: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence:
        evidence_by_disease[item.disease].append(item)
    evidence_by_key = {item.key: item for item in evidence}

    rng = random.Random(RANDOM_SEED)
    source_rows = build_source_rows(evidence_by_disease)
    combination_rows = build_combination_rows(evidence_by_disease, rng)
    parents = source_rows + combination_rows
    paraphrase_rows = build_paraphrase_rows(parents, evidence_by_key, rng)

    rows = parents + paraphrase_rows
    rows.sort(key=lambda row: (row["disease"], row["provenance"], row["row_id"]))

    problems = verify(rows)
    if problems:
        for problem in problems[:25]:
            print(f"INTEGRITY FAILURE: {problem}")
        raise SystemExit(f"{len(problems)} integrity problems; refusing to write the dataset.")

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["disease"]][row["provenance"]] += 1

    print(f"Evidence passages: {len(evidence)} across {len(evidence_by_disease)} labels")
    print(f"{'label':22s} {'source':>7s} {'comb':>6s} {'para':>6s} {'total':>6s}")
    for disease in sorted(counts):
        breakdown = counts[disease]
        total = sum(breakdown.values())
        print(
            f"{disease:22s} {breakdown['source']:7d} {breakdown['source_combination']:6d} "
            f"{breakdown['paraphrase']:6d} {total:6d}"
        )
    print(f"{'TOTAL':22s} {len(source_rows):7d} {len(combination_rows):6d} {len(paraphrase_rows):6d} {len(rows):6d}")

    if args.report:
        return

    with DATASET_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {DATASET_PATH}")


if __name__ == "__main__":
    main()
