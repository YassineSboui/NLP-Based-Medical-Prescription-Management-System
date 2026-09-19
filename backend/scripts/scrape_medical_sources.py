"""Fetch the CDC/WHO source pages listed in ``dataset_sources.json``.

The output of this script is the *evidence base* for the training dataset:
``build_dataset.py`` refuses to emit a row whose symptom terms cannot be found
in the text captured here. Two properties therefore matter more than coverage:

1. Nothing is ever invented. Only text returned by a real HTTP 200 is stored.
2. A failed refetch never destroys a previous successful capture. Some
   publishers (CDC at the time of writing) block non-browser clients from
   certain networks with HTTP 403. Losing a good capture to a transient block
   would silently shrink the evidence base, so failures are recorded next to
   the preserved capture instead of overwriting it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup, Tag


BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "app" / "data"
SOURCE_REGISTRY_PATH = DATA_DIR / "dataset_sources.json"
SCRAPED_JSON_PATH = DATA_DIR / "scraped_medical_sources.json"
SCRAPED_CSV_PATH = DATA_DIR / "scraped_medical_sources.csv"

PROJECT_USER_AGENT = "Academic NLP Medical Prescription Project/2.0 (+educational data collection)"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Tried in order. The honest project agent goes first so publishers can identify
# the client; the browser agent is only a fallback for edges that reject it.
USER_AGENTS = (PROJECT_USER_AGENT, BROWSER_USER_AGENT)

BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT_SECONDS = 30
RETRY_DELAY_SECONDS = 2.0

RELEVANT_HEADINGS = {
    "about",
    "overview",
    "key facts",
    "key points",
    "symptoms",
    "signs and symptoms",
    "signs & symptoms",
    "clinical signs",
    "diagnosis",
    "testing",
    "treatment",
    "prevention",
}

MEDICAL_KEYWORDS = {
    "fever",
    "headache",
    "cough",
    "rash",
    "pain",
    "diarrhea",
    "diarrhoea",
    "vomiting",
    "nausea",
    "fatigue",
    "chills",
    "sweats",
    "breathing",
    "dehydration",
    "jaundice",
    "blood",
    "treatment",
    "testing",
    "diagnosis",
    "antibiotics",
    "medicine",
    "vaccine",
}


@dataclass(frozen=True)
class Source:
    disease: str
    name: str
    url: str


def load_sources() -> list[Source]:
    """One entry per (disease, url) pair.

    A single page can legitimately be evidence for more than one label -- the CDC
    flu page, for example, is cited both for ``flu`` and for the flu-vs-cold
    contrast under ``common cold``. Deduplicating by URL alone silently dropped
    one of those labels, so dedup is per (disease, url) and the HTTP fetch is
    cached by URL instead.
    """
    registry = json.loads(SOURCE_REGISTRY_PATH.read_text(encoding="utf-8"))
    unique: dict[tuple[str, str], Source] = {}

    for disease, sources in registry.items():
        for source in sources:
            url = source["url"]
            unique[(disease, url)] = Source(disease=disease, name=source["name"], url=url)

    return sorted(unique.values(), key=lambda item: (item.disease, item.name))


def clean_text(value: str) -> str:
    return " ".join(value.split())


def is_relevant_heading(text: str) -> bool:
    normalized = clean_text(text).lower()
    return any(heading in normalized for heading in RELEVANT_HEADINGS)


def is_symptom_heading(text: str) -> bool:
    """True for headings that announce a symptom list, however they are worded.

    Publishers do not stick to a fixed set of headings: CDC uses "Signs and
    symptoms", "Early symptoms", "Later symptoms", "Symptoms of active TB
    disease include:" and "7-14 days after a measles infection: first symptoms
    show". A substring match on symptom/sign catches all of them.
    """
    normalized = clean_text(text).lower()
    return "symptom" in normalized or "signs" in normalized


def contains_medical_keyword(text: str) -> bool:
    normalized = clean_text(text).lower()
    return any(keyword in normalized for keyword in MEDICAL_KEYWORDS)


def remove_noise(soup: BeautifulSoup) -> None:
    for tag_name in ["script", "style", "noscript", "svg", "header", "footer", "nav"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()


def iter_section_nodes(heading: Tag) -> Iterable[Tag]:
    for sibling in heading.find_next_siblings():
        if not isinstance(sibling, Tag):
            continue
        if sibling.name in {"h1", "h2", "h3"}:
            break
        yield sibling


def is_wanted_item(text: str, keep_all: bool) -> bool:
    """Decide whether one <p>/<li> is worth capturing.

    The keyword filter was silently discarding the most valuable text on the
    page. CDC symptom pages are bullet lists of bare symptom names -- "Runny
    nose", "Red, watery eyes", "Koplik spots", "Shortness of breath" -- none of
    which contain a word from MEDICAL_KEYWORDS, so the entire symptom list was
    dropped and only the prose around it survived. Under a heading that already
    announces itself as a symptom list, every bullet is kept.

    Elsewhere on the page the keyword filter still earns its place: it is what
    keeps navigation, funding statements and vaccine-coverage percentages out.
    """
    if len(text) < 3:
        return False
    if keep_all:
        return True
    return contains_medical_keyword(text)


def extract_text_items(nodes: Iterable[Tag], keep_all: bool = False) -> list[str]:
    items: list[str] = []

    for node in nodes:
        for element in node.find_all(["p", "li"], recursive=True):
            text = clean_text(element.get_text(" ", strip=True))
            if text and is_wanted_item(text, keep_all):
                items.append(text)

        if node.name in {"p", "li"}:
            text = clean_text(node.get_text(" ", strip=True))
            if text and is_wanted_item(text, keep_all):
                items.append(text)

    return deduplicate(items)


def deduplicate(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def extract_sections(html: str) -> tuple[str, list[dict[str, object]]]:
    soup = BeautifulSoup(html, "html.parser")
    remove_noise(soup)

    title_tag = soup.find("h1") or soup.find("title")
    title = clean_text(title_tag.get_text(" ", strip=True)) if title_tag else "Untitled source"

    sections: list[dict[str, object]] = []
    for heading in soup.find_all(["h1", "h2", "h3"]):
        heading_text = clean_text(heading.get_text(" ", strip=True))
        if not heading_text or not is_relevant_heading(heading_text):
            continue

        items = extract_text_items(iter_section_nodes(heading), keep_all=is_symptom_heading(heading_text))
        if items:
            sections.append({"heading": heading_text, "items": items})

    if not sections:
        fallback_items = []
        for element in soup.find_all(["p", "li"]):
            text = clean_text(element.get_text(" ", strip=True))
            if text and contains_medical_keyword(text):
                fallback_items.append(text)
        if fallback_items:
            sections.append({"heading": "Keyword-based fallback extraction", "items": deduplicate(fallback_items)})

    return title, sections


_HTML_CACHE: dict[str, tuple[str, int]] = {}


def fetch_html(url: str) -> tuple[str, int]:
    """Fetch ``url`` once per run, retrying once per user agent.

    Raises the last error if every attempt fails.
    """
    if url in _HTML_CACHE:
        return _HTML_CACHE[url]

    last_error: Exception | None = None

    for user_agent in USER_AGENTS:
        headers = {**BASE_HEADERS, "User-Agent": user_agent}
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            _HTML_CACHE[url] = (response.text, response.status_code)
            return _HTML_CACHE[url]
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(RETRY_DELAY_SECONDS)

    raise last_error if last_error else RuntimeError(f"Could not fetch {url}")


def scrape_source(source: Source) -> dict[str, object]:
    html, status_code = fetch_html(source.url)
    title, sections = extract_sections(html)
    now = datetime.now(timezone.utc).isoformat()

    return {
        "disease": source.disease,
        "source_name": source.name,
        "source_url": source.url,
        "fetch_status": "ok",
        "status_code": status_code,
        "title": title,
        "fetched_at": now,
        "attempted_at": now,
        "content_sha256": hashlib.sha256(html.encode("utf-8", "ignore")).hexdigest(),
        "extracted_sections": sections,
    }


def preserved_record(source: Source, previous: dict[str, object], error: str) -> dict[str, object]:
    """Keep a previous successful capture and record the failed refetch beside it.

    This is what stops a transient 403 from quietly deleting real evidence.
    """
    record = dict(previous)
    record.update(
        {
            "disease": source.disease,
            "source_name": source.name,
            "source_url": source.url,
            "fetch_status": "preserved_previous_capture",
            "attempted_at": datetime.now(timezone.utc).isoformat(),
            "last_error": error,
            "note": (
                "Refetch failed. The text below is the previous successful capture of this "
                "URL and is unchanged; 'fetched_at' is when that capture was made."
            ),
        }
    )
    return record


def failed_record(source: Source, error: str) -> dict[str, object]:
    return {
        "disease": source.disease,
        "source_name": source.name,
        "source_url": source.url,
        "fetch_status": "failed",
        "status_code": None,
        "title": "Fetch failed",
        "fetched_at": None,
        "attempted_at": datetime.now(timezone.utc).isoformat(),
        "last_error": error,
        "extracted_sections": [],
    }


def load_previous_records() -> dict[tuple[str, str], dict[str, object]]:
    if not SCRAPED_JSON_PATH.exists():
        return {}
    try:
        records = json.loads(SCRAPED_JSON_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        (record["disease"], record["source_url"]): record
        for record in records
        if record.get("extracted_sections")
    }


def write_csv(records: list[dict[str, object]]) -> None:
    with SCRAPED_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["disease", "source_name", "source_url", "fetch_status", "fetched_at", "heading", "text"],
        )
        writer.writeheader()

        for record in records:
            for section in record["extracted_sections"]:
                for item in section["items"]:
                    writer.writerow(
                        {
                            "disease": record["disease"],
                            "source_name": record["source_name"],
                            "source_url": record["source_url"],
                            "fetch_status": record.get("fetch_status", "ok"),
                            "fetched_at": record.get("fetched_at"),
                            "heading": section["heading"],
                            "text": item,
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not make network calls; just rewrite the CSV from the stored JSON capture.",
    )
    args = parser.parse_args()

    previous = load_previous_records()
    records: list[dict[str, object]] = []
    stats = {"ok": 0, "preserved_previous_capture": 0, "failed": 0}

    for source in load_sources():
        if args.offline:
            stored = previous.get((source.disease, source.url))
            if stored:
                records.append(stored)
                stats[str(stored.get("fetch_status", "ok"))] = stats.get(str(stored.get("fetch_status", "ok")), 0) + 1
            continue

        print(f"Scraping {source.name}: {source.url}")
        try:
            record = scrape_source(source)
        except (requests.RequestException, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            stored = previous.get((source.disease, source.url))
            if stored:
                print(f"  -> failed ({error}); keeping previous capture")
                record = preserved_record(source, stored, error)
            else:
                print(f"  -> failed ({error}); no previous capture to keep")
                record = failed_record(source, error)
        records.append(record)
        stats[str(record["fetch_status"])] = stats.get(str(record["fetch_status"]), 0) + 1

    SCRAPED_JSON_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(records)

    print(f"Saved JSON extraction to {SCRAPED_JSON_PATH}")
    print(f"Saved CSV extraction to {SCRAPED_CSV_PATH}")
    print(f"Sources: {len(records)} total -> {stats}")


if __name__ == "__main__":
    main()
