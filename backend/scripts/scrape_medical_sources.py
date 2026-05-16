from __future__ import annotations

import csv
import json
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

REQUEST_HEADERS = {
    "User-Agent": "Academic NLP Medical Prescription Project/1.0 (+educational data collection)"
}

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
    registry = json.loads(SOURCE_REGISTRY_PATH.read_text(encoding="utf-8"))
    unique: dict[str, Source] = {}

    for disease, sources in registry.items():
        for source in sources:
            url = source["url"]
            unique[url] = Source(disease=disease, name=source["name"], url=url)

    return sorted(unique.values(), key=lambda item: (item.disease, item.name))


def clean_text(value: str) -> str:
    return " ".join(value.split())


def is_relevant_heading(text: str) -> bool:
    normalized = clean_text(text).lower()
    return any(heading in normalized for heading in RELEVANT_HEADINGS)


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


def extract_text_items(nodes: Iterable[Tag]) -> list[str]:
    items: list[str] = []

    for node in nodes:
        for element in node.find_all(["p", "li"], recursive=True):
            text = clean_text(element.get_text(" ", strip=True))
            if text and contains_medical_keyword(text):
                items.append(text)

        if node.name in {"p", "li"}:
            text = clean_text(node.get_text(" ", strip=True))
            if text and contains_medical_keyword(text):
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

        items = extract_text_items(iter_section_nodes(heading))
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


def scrape_source(source: Source) -> dict[str, object]:
    response = requests.get(source.url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    title, sections = extract_sections(response.text)

    return {
        "disease": source.disease,
        "source_name": source.name,
        "source_url": source.url,
        "status_code": response.status_code,
        "title": title,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "extracted_sections": sections,
    }


def write_csv(records: list[dict[str, object]]) -> None:
    with SCRAPED_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["disease", "source_name", "source_url", "heading", "text"],
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
                            "heading": section["heading"],
                            "text": item,
                        }
                    )


def main() -> None:
    records: list[dict[str, object]] = []

    for source in load_sources():
        print(f"Scraping {source.name}: {source.url}")
        try:
            records.append(scrape_source(source))
        except requests.RequestException as exc:
            records.append(
                {
                    "disease": source.disease,
                    "source_name": source.name,
                    "source_url": source.url,
                    "status_code": None,
                    "title": "Fetch failed",
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                    "extracted_sections": [],
                }
            )

    SCRAPED_JSON_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(records)

    print(f"Saved JSON extraction to {SCRAPED_JSON_PATH}")
    print(f"Saved CSV extraction to {SCRAPED_CSV_PATH}")
    print(f"Scraped {len(records)} unique sources")


if __name__ == "__main__":
    main()
