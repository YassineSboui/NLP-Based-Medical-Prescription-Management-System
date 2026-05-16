import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PROTECTED_WORDS = {
    "blood",
    "breath",
    "breathing",
    "chest",
    "cough",
    "diarrhea",
    "fatigue",
    "fever",
    "headache",
    "loss",
    "muscle",
    "never",
    "night",
    "no",
    "not",
    "pain",
    "rash",
    "shortness",
    "smell",
    "sweats",
    "taste",
    "vomiting",
    "weakness",
    "weight",
    "without",
}

STOP_WORDS = set(ENGLISH_STOP_WORDS) - PROTECTED_WORDS

PHRASE_NORMALIZATIONS = {
    r"\bcovid[\s-]?19\b": "covid",
    r"\bcorona\s+virus\b": "covid",
    r"\bcoronavirus\b": "covid",
    r"\bsars[\s-]?cov[\s-]?2\b": "covid",
    r"\bdiarrhoea\b": "diarrhea",
    r"\bcoryza\b": "runny nose",
    r"\bconjunctivitis\b": "red watery eyes",
    r"\bphotophobia\b": "light sensitivity",
    r"\bmyalgia\b": "muscle pain",
    r"\bfebrile\b": "fever",
    r"\bshortness\s+of\s+breath\b": "shortness breath",
    r"\bdifficulty\s+breathing\b": "shortness breath",
    r"\bloss\s+of\s+taste\b": "loss taste",
    r"\bloss\s+of\s+smell\b": "loss smell",
    r"\bnight\s+sweats\b": "night sweats",
    r"\bweight\s+loss\b": "weight loss",
    r"\bstomach\s+pain\b": "abdominal pain",
    r"\bbelly\s+pain\b": "abdominal pain",
    r"\bbody\s+aches\b": "body pain",
    r"\bbody\s+ache\b": "body pain",
    r"\bmuscle\s+aches\b": "muscle pain",
    r"\bstuffy\s+nose\b": "nasal congestion",
    r"\bblocked\s+nose\b": "nasal congestion",
    r"\byellowing\s+of\s+the\s+skin\s+and\s+eyes\b": "jaundice",
    r"\byellow\s+eyes\b": "jaundice",
    r"\bstiff\s+neck\b": "stiff neck",
    r"\bhigh\s+temperature\b": "fever",
}


def clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    for pattern, replacement in PHRASE_NORMALIZATIONS.items():
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"[^a-z0-9\s\-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    tokens = [token for token in normalized.split() if token not in STOP_WORDS]
    return " ".join(tokens)
