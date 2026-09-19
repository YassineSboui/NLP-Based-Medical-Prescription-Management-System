# NLP-Based Medical Prescription Management System

Symptom-triage API. Takes free-text symptoms, extracts medical entities, suggests one of 14 common conditions, and returns educational medication information with safety warnings. Every prediction states which engine produced it and what that engine's measured accuracy is. Every suggestion is recorded with an audit trail.

Runs entirely offline. No API keys, no external services, no network calls at request time.

## Safety disclaimer

This project is for academic demonstration only. It does not provide a medical diagnosis, does not replace a doctor, and must not be used for self-medication. All medicines, dosages, diagnoses, and tests must be confirmed by a qualified healthcare professional.

## Honest performance summary

| Measure | Classical (default) | Sequence (optional) | Rule engine |
| --- | --- | --- | --- |
| Held-out test accuracy | **0.894** (95% CI 0.828–0.937) | 0.846 | 0.707 |
| Held-out test macro F1 | 0.888 | 0.836 | 0.642 |
| 5-fold grouped CV accuracy | 0.861 (95% CI 0.797–0.924) | — | — |
| External hand-written scenarios (n=14) | 14/14 (95% CI 0.785–1.000) | 11/14 | 12/14 |
| **Cross-source transfer recall** | **0.524** | — | — |

**Read the last row before you read the first one.** Cross-source transfer holds out an entire CDC or WHO passage from training and then tests on it. At 0.524 it says the model is substantially learning how one particular source page words things, not the underlying condition. Cholera collapses to 0.015 on that measure. The 0.894 is a real number, honestly measured on data the shipped artifact never saw, but it describes performance on *wording it has seen the style of*. Treat 0.894 as the ceiling and 0.524 as the floor.

An earlier version of this README claimed 0.906. That number came from ranking 72 pipelines on the same 32-row set that was then reported, with the shipped model refit on that same set. It is not comparable to anything here and has been withdrawn.

Full methodology, per-class intervals and limitations: `models/classical/metrics.txt` and the `evaluation` block of `models/classical/model_metadata.json`.

## Features

- Free-text symptom analysis with rule-based extraction of symptoms, diseases, medication names, and dosage mentions, including negation handling ("no fever" does not extract fever).
- Three documented prediction engines behind one arbitration policy, with honest attribution of which one answered.
- Abstention: when no engine is confident, the API returns `unknown` instead of guessing.
- Medication and dosage guidance from a JSON knowledge base, with per-medication warnings.
- Consultation records, queryable history, audit trail, CSV export, and Markdown reports.
- Batch processing of multiple notes.
- SQLite persistence via SQLAlchemy — one file, no database server.
- Optional API-key authentication and an explicit CORS origin list.

## Disease scope

14 source-backed labels, plus `unknown` for abstention:

`malaria`, `typhoid fever`, `tuberculosis`, `hiv`, `flu`, `common cold`, `gastroenteritis`, `covid-like illness`, `dengue`, `cholera`, `pneumonia`, `meningitis`, `hepatitis b`, `measles`

Anything outside these is answered as the nearest of the 14, or abstained on. This is a triage demonstrator, not a differential diagnosis system.

## The prediction engines

| Engine | Key | What it is | Confidence means |
| --- | --- | --- | --- |
| Classical | `classical` | TF-IDF + logistic regression. The default. | `predict_proba` — a probability |
| Sequence | `advanced` | Bidirectional GRU (Keras). Optional, needs TensorFlow. | softmax — a probability |
| Symptom profile | `symptom_profile` | Weighted matching against per-disease symptom profiles derived from the dataset | share of profile-match weight — **not** a probability |

### Arbitration policy `primary_model_with_rule_fallback_v1`

1. The requested statistical engine runs. So does the rule engine, always, so its opinion is on the record.
2. If the statistical engine's probability is at or above `0.20`, it decides.
3. Otherwise the rule engine decides, if it has an opinion.
4. Otherwise the ensemble abstains and returns `unknown`.

The rule engine is a fallback, never an override. It only speaks where the primary has already admitted it does not know, so the two confidence numbers — which are on different scales — are never compared with each other.

The `0.20` floor was chosen by sweeping candidate floors on the validation split with a model fit on the training split only. It is the only floor that improves on model-only for both accuracy (0.895 → 0.903) and macro F1 (0.869 → 0.875). The table is in `models/classical/metrics.txt`.

`decision.decided_by` always names the engine that actually answered and carries that engine's own metrics. `engine_opinions` lists what every engine thought, including the ones that were overruled.

## Dataset

616 rows across 14 balanced labels, built from 28 CDC/WHO source captures (15 CDC pages, 13 WHO fact sheets), all fetched live at HTTP 200.

Every row declares its provenance:

| `provenance` | Rows | Meaning |
| --- | --- | --- |
| `source` | 28 | The full symptom set one quoted source passage attests |
| `source_combination` | 224 | A subset of one passage's symptoms. The terms are the source's; the co-occurrence is synthetic |
| `paraphrase` | 364 | A wording variant of one named row (`derived_from`). Synonyms and patient-voice phrasing only |

The builder refuses to write the file unless every `source` and `source_combination` row has each of its terms present verbatim in that row's own `source_quote`, and every `paraphrase` carries exactly its parent's symptom set. `tests/test_dataset_integrity.py` re-checks this from the CSV alone.

**No clinical description in this dataset was authored by hand and attributed to CDC or WHO.** Paraphrases change wording only: intensifiers and specifiers are banned from the synonym table, because rendering a source's plain "cough" as "a cough that will not stop" would quietly relabel the example toward tuberculosis.

Columns: `row_id`, `text`, `disease`, `provenance`, `derived_from`, `group_id`, `symptom_terms`, `source_name`, `source_url`, `source_section`, `source_quote`, `source_fetched_at`, `source_note`.

`group_id` ties a row to its paraphrases so the evaluation split cannot put near-duplicates on both sides.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

Optional, for the sequence model only:

```bash
pip install -r backend/requirements-advanced.txt
```

## Run the backend

```bash
cd backend
uvicorn app.main:app --reload
```

Interactive docs at `http://localhost:8000/docs`. The SQLite store is created on first start at `data/medical_nlp.db` (gitignored).

To launch backend and Streamlit frontend together on Windows:

```bash
run_app.bat
```

## The console

`frontend/streamlit_app.py` is the entry point; it routes and nothing else. The markup helpers live in `frontend/ui/components.py`, the API calls in `frontend/ui/api.py`, one module per screen under `frontend/ui/views/`, and the whole stylesheet in `frontend/assets/app.css` — one file, loaded once, with no webfont import so the interface makes no network call of its own.

| Screen | What it is for |
| --- | --- |
| Analysis | One note in, one attributed answer out, with the arbitration policy and the transfer number standing beside the input |
| History | Every recorded consultation, filterable, with per-consultation audit trail, CSV export and Markdown report |
| Batch | Many notes in one pass, per-item success and failure |
| Model evaluation | What the model actually does, cross-source transfer first |

Every result names the engine that produced it and shows **that engine's** measured accuracy. When the rule engine answers, the panel reports the rule engine's 0.707, not the classifier's 0.894 — showing the classifier's number for an answer the rules produced was the central dishonesty of the previous interface, and `tests/test_engine_attribution.py` exists to keep it fixed.

The cross-source transfer recall of 0.524 appears in four places: a chip in the masthead, a standing card on the analysis input screen, a callout beside every result — scoped to the predicted label, so a prediction of cholera carries cholera's 0.015 — and as the headline of the evaluation screen.

## Configuration

All optional. Every default works offline.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEDICAL_NLP_DB_URL` | `sqlite:///data/medical_nlp.db` | Consultation store |
| `MEDICAL_NLP_CORS_ORIGINS` | localhost:8501, localhost:3000 | Comma-separated allowed origins |
| `MEDICAL_NLP_API_KEY` | unset | When set, every analysis and history endpoint requires `X-API-Key` |
| `MEDICAL_NLP_MAX_BATCH` | `50` | Maximum notes per batch |
| `MEDICAL_NLP_MAX_TEXT_LENGTH` | `4000` | Maximum characters per note |
| `MEDICAL_NLP_API_BASE_URL` | `http://localhost:8000` | Frontend only: where the console looks for the API |

The frontend reads the service address from the environment and never exposes it as a field. An address is configuration; an end user cannot act on it, and a console that asks them to is admitting it is two processes held together by hand. When the service cannot be reached the console says so as a service state, offers a retry, and shows no metrics at all rather than inventing any.

Authentication is off by default because requiring a key would break the offline single-machine demo, and shipping a default key would be worse than none. `/health` reports `auth_required` so nobody has to guess.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Status, version, auth flag, database readiness, per-engine availability |
| `GET` | `/engines` | Every engine with its own metrics, plus the arbitration policy |
| `GET` | `/models` | Deprecated alias of `/engines` |
| `POST` | `/analyze` | Analyse one note, persist a consultation |
| `POST` | `/analyze/batch` | Analyse many notes; per-item success/failure |
| `GET` | `/consultations` | History, filterable and paged |
| `GET` | `/consultations/stats` | Aggregate counts by disease and engine |
| `GET` | `/consultations/export` | CSV export of filtered history |
| `GET` | `/consultations/{id}` | One consultation in full |
| `GET` | `/consultations/{id}/audit` | Audit trail for one consultation |
| `GET` | `/consultations/{id}/report` | Markdown report |

### `POST /analyze`

```json
{
  "text": "I have fever, headache, chills, sweating and body pain for 3 days.",
  "model_key": "classical",
  "session_id": null,
  "patient_reference": null,
  "persist": true
}
```

Response (abridged):

```json
{
  "consultation_id": "208ff007-e6d8-4441-990f-9e88ec80476e",
  "session_id": "0e2f...",
  "created_at": "2026-09-19T02:18:15.100337Z",
  "original_text": "...",
  "cleaned_text": "fever headache chills sweating body pain 3 days",
  "extracted_entities": {
    "symptoms": ["fever", "chills", "headache", "sweating", "body pain"],
    "diseases": [], "medications": [], "dosage_mentions": []
  },
  "predicted_disease": "malaria",
  "confidence": 0.321,
  "decision": {
    "predicted_disease": "malaria",
    "confidence": 0.321,
    "decided_by": {
      "key": "classical",
      "name": "logistic_regression",
      "family": "TF-IDF + linear classifier",
      "confidence_meaning": "predict_proba of the winning class; a probability over the 14 known labels.",
      "is_available": true,
      "is_default": true,
      "accuracy": 0.894,
      "macro_f1": 0.888,
      "metrics": {
        "accuracy": 0.894,
        "accuracy_ci_95": [0.828, 0.937],
        "macro_f1": 0.888,
        "external_use_case_accuracy": 1.0,
        "cross_source_transfer_recall": 0.524,
        "basis": "held-out test set of 123 rows, group-aware split, engine run standalone with no arbitration",
        "evaluated_at": "2026-09-19T02:16:33Z",
        "source_file": "engine_metrics.json"
      }
    },
    "policy": "primary_model_with_rule_fallback_v1",
    "policy_reason": "The classical engine reported 0.321, at or above the 0.2 confidence floor, so it decided.",
    "abstained": false
  },
  "engine_opinions": [
    {"engine": "classical", "disease": "malaria", "confidence": 0.321, "available": true, "detail": "...", "matched_symptoms": []},
    {"engine": "symptom_profile", "disease": "gastroenteritis", "confidence": 0.173, "available": true, "detail": "Matched 3 profile symptoms: ...", "matched_symptoms": ["fever", "headache", "..."]}
  ],
  "model_used": { "...": "alias of decision.decided_by" },
  "recommended_actions": ["..."],
  "recommended_medicines": [
    {"name": "Artemether-lumefantrine", "standard_dosage": "...", "administration": "...", "warnings": ["..."]}
  ],
  "disclaimer": "..."
}
```

## Reproducing the pipeline

```bash
python backend/scripts/scrape_medical_sources.py   # fetch CDC/WHO pages
python backend/scripts/build_dataset.py            # rebuild the dataset from those captures
python backend/scripts/train_model.py              # train + evaluate the classical model
python backend/scripts/evaluate_engines.py         # measure every engine on one shared split
python backend/scripts/train_deep_learning_model.py  # optional, needs TensorFlow
```

`build_dataset.py --report` prints coverage without writing. `scrape_medical_sources.py --offline` rebuilds the CSV from the stored capture with no network.

The builder is deterministic: the same evidence base always produces the same CSV.

## Tests

```bash
pip install -r backend/requirements.txt
python -m pytest
```

117 tests. They cover dataset provenance (verified from the CSV, not by re-running the builder), the group-aware split and published intervals, text cleaning and negation, the API contract, persistence and audit, and — the important one — engine attribution: `tests/test_engine_attribution.py` asserts that the engine named in a response is the engine whose opinion the response is repeating, and that the metrics returned are that engine's own.

`tests/test_frontend_contract.py` covers the two things the console can get wrong silently: resolving the service address, including the older `.../analyze` form a stale environment may still hold, and never inventing an engine when the service is unreachable.

## Model artifacts

```text
models/
  classical/
    trained_model.joblib        # the shipped model, fit on train+validation only
    vectorizer.joblib
    metrics.txt                 # full evaluation report incl. methodology and limitations
    model_metadata.json
    model_comparison.png, overall_metrics.png, prediction_outcomes.png
  advanced/
    sequence_model.keras        # one artifact, not three
    tokenizer.joblib, label_encoder.joblib
    deep_learning_metrics.txt, deep_learning_metadata.json
    training_history.png, advanced_overall_metrics.png
  evaluation/
    engine_metrics.json         # every engine, one shared split
```

The shipped classical model is fit on train+validation and never sees the test set, so the published test number describes that exact artifact rather than a differently-fit sibling of it.

## Documentation

- `docs/full_project_documentation.md` — architecture, dataset, setup, demo guide, evaluation
- `docs/technical_report.md` — formal technical report
- `docs/teacher_q_and_a.md` — prepared answers to likely questions
- `tests/use_case_tests.txt` — the 14 scenarios, executed by `tests/test_use_cases.py`

## Dataset sources

All fetched live. CDC pages: malaria, typhoid, tuberculosis (about + signs), HIV, flu, common cold, norovirus, COVID-19, dengue, cholera, pneumonia, meningitis, measles, hepatitis B. WHO fact sheets: malaria, typhoid, tuberculosis, HIV/AIDS, influenza (seasonal), COVID-19, diarrhoeal disease, dengue, cholera, pneumonia, meningitis, measles, hepatitis B.

The full registry with per-source symptom lists is `backend/app/data/dataset_sources.json`; the captured text with fetch status and content hashes is `backend/app/data/scraped_medical_sources.json`.

This is not a real patient-record dataset and it is not clinically validated.

## Known limitations

- Every row for a label descends from one or two source passages, so evaluation folds are not clinically independent. Cross-source transfer (0.524) is the honest generalisation measure; the test accuracy is an upper bound.
- Five labels (`common cold`, `covid-like illness`, `gastroenteritis`, `hiv`, `pneumonia`) are backed by a single source passage each and cannot be transfer-tested at all. Of the nine that can be, transfer recall ranges from 0.925 (hepatitis B) down to 0.015 (cholera).
- The external scenario set has 14 items. Its interval is correspondingly wide.
- 14 labels only. Anything outside them is answered as the nearest of the 14, or abstained on.
- Negation handling is a narrow lookbehind, not a parser. It catches "no fever" and "not experiencing headache"; it will not catch arbitrary negation scope.
- `patient_reference` is stored verbatim. It is a case label, not a place for identifying patient data.

## Future improvements

- Broaden the source base per label so cross-source transfer becomes meaningful for all 14.
- spaCy NER trained on annotated medical text, in place of dictionary matching.
- Fine-tune a clinical transformer for symptom classification.
- Multilingual support for English and French symptom descriptions.
- Validate all medical content with licensed healthcare professionals.
