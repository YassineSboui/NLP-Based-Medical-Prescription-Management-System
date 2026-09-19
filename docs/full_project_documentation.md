# Full Project Documentation

## 1. Project Overview

This project is an academic NLP-based medical prescription management system. It allows a user to enter symptoms in natural language, extracts medical entities, predicts a likely disease/condition, and returns educational treatment information with strict safety warnings.

The system is not a real medical diagnosis tool. It is designed for demonstration, learning, and academic evaluation.

## 2. Main Objective

The goal is to demonstrate how Natural Language Processing and machine learning can help structure patient-reported symptoms and support preliminary medical decision support.

The application performs four main tasks:

- Analyze free-text symptoms.
- Extract medical entities such as symptoms, diseases, medications, and dosage mentions.
- Predict a likely disease/condition using a trained NLP model.
- Return safe educational guidance from a medical knowledge base.

## 3. Supported Diseases

The current version supports 14 disease labels:

- Malaria
- Typhoid fever
- Tuberculosis
- HIV
- Flu
- Common cold
- Gastroenteritis
- COVID-like illness
- Dengue
- Cholera
- Pneumonia
- Meningitis
- Hepatitis B
- Measles

The first four diseases come directly from the original project context. The other diseases were added to make the demo more complete and relevant for common infectious and respiratory/gastrointestinal conditions.

## 4. Project Architecture

The project is divided into three main parts:

- Backend API using FastAPI.
- NLP and machine learning services in Python.
- Frontend interface using Streamlit.

Project structure:

```text
backend/
  app/
    core/
      paths.py
    main.py
    api/routes.py
    services/
      nlp_service.py
      recommendation_service.py
      disease_prediction_service.py
    models/schemas.py
    data/
      symptoms_dataset.csv
      dataset_sources.json
      medication_knowledge_base.json
    utils/text_cleaning.py
  scripts/train_model.py
  scripts/train_deep_learning_model.py
  requirements.txt
  requirements-advanced.txt
frontend/
  streamlit_app.py
models/
  classical/
    trained_model.joblib
    vectorizer.joblib
    metrics.txt
    model_metadata.json
    model_comparison.png
    prediction_outcomes.png
    overall_metrics.png
  advanced/
tests/
  use_case_tests.txt
docs/
```

## 5. Backend Explanation

The backend is built with FastAPI. It exposes REST endpoints that the frontend can call.

Main file:

```text
backend/app/main.py
```

This file creates the FastAPI app, configures CORS, and includes the API routes.

Important route file:

```text
backend/app/api/routes.py
```

Available endpoints:

- `GET /health`: status, version, whether auth is required, database readiness, per-engine availability.
- `GET /engines`: every prediction engine with its own measured metrics, plus the arbitration policy and the confidence floor.
- `GET /models`: deprecated alias of `/engines`, kept so existing clients keep working.
- `POST /analyze`: analyses symptom text, records a consultation, returns the prediction with full engine attribution.
- `POST /analyze/batch`: analyses several notes; one bad note does not fail the batch.
- `GET /consultations`: history, filterable by session, disease, engine and date, paged.
- `GET /consultations/stats`: aggregate counts by disease and engine.
- `GET /consultations/export`: CSV export of the filtered history.
- `GET /consultations/{id}`: one consultation in full.
- `GET /consultations/{id}/audit`: the audit trail for that consultation.
- `GET /consultations/{id}/report`: a Markdown report of that consultation.

Example request:

```json
{
  "text": "I have fever, headache, chills and body pain.",
  "model_key": "classical"
}
```

Example response contains:

- original text
- cleaned text
- extracted entities
- predicted disease
- confidence score
- recommended actions
- medication information
- safety disclaimer

## 6. Data Layer

The dataset is stored in:

```text
backend/app/data/symptoms_dataset.csv
```

This is a source-backed curated dataset. It is not a real patient-record dataset. Each row contains a symptom phrase, disease label, and source information.

Dataset columns:

- `text`: symptom phrase used for training.
- `disease`: target label.
- `source_name`: name of the public health source.
- `source_url`: URL of the source.
- `source_note`: explanation of how the row was derived.

The source registry is stored in:

```text
backend/app/data/dataset_sources.json
```

This file lists the official sources used for each disease. The main sources are CDC and WHO public health pages.

### 6.1 Controlled Scraping Script

A controlled scraping script is available in:

```text
backend/scripts/scrape_medical_sources.py
```

The script reads `dataset_sources.json`, fetches the official CDC/WHO pages, extracts symptom/treatment-related sections using BeautifulSoup, and saves raw extracted content to:

```text
backend/app/data/scraped_medical_sources.json
backend/app/data/scraped_medical_sources.csv
```

The scraped output is used for traceability and data collection support. The final training dataset remains curated manually because medical content must be reviewed and cleaned carefully before being used for model training.

## 7. Text Cleaning

Text cleaning is implemented in:

```text
backend/app/utils/text_cleaning.py
```

The cleaning function does the following:

- Converts Unicode text to ASCII where possible.
- Converts text to lowercase.
- Normalizes medical phrases.
- Removes unnecessary punctuation.
- Normalizes spaces.
- Removes general English stop words while preserving important medical and negation words such as `no`, `not`, `without`, `fever`, `cough`, `pain`, and `blood`.

Examples of phrase normalization:

- `diarrhoea` becomes `diarrhea`.
- `coryza` becomes `runny nose`.
- `photophobia` becomes `light sensitivity`.
- `yellow eyes` becomes `jaundice`.
- `difficulty breathing` becomes `shortness breath`.

This improves model consistency because different words can describe the same symptom.

## 8. Entity Extraction

Entity extraction is implemented in:

```text
backend/app/services/nlp_service.py
```

The system uses a rule-based entity extractor. It uses dictionaries of known medical terms and aliases.

It extracts:

- symptoms
- disease names
- medication names
- dosage mentions

Example:

Input:

```text
I have high fever, severe headache and pain behind the eyes.
```

Extracted symptoms:

```text
fever, headache, pain behind eyes
```

The extractor also has simple negation handling. For example:

```text
I have cough but no fever.
```

The system should extract `cough` but should not extract `fever` as a positive symptom.

## 9. Disease Prediction

Disease prediction is implemented in:

```text
backend/app/services/disease_prediction_service.py
```

The model uses TF-IDF vectorization and a classical machine learning classifier.

The training script compares multiple models:

- Logistic Regression
- Calibrated Linear SVC
- Complement Naive Bayes

The best model is selected based on validation performance.

Current best model:

```text
logistic_regression (C=4.0, TF-IDF unigrams, sublinear_tf)
```

Current metrics:

```text
Held-out test accuracy : 0.894   95% CI [0.828, 0.937]
Held-out test macro F1 : 0.888
Grouped 5-fold CV      : 0.861   95% CI [0.797, 0.924]
External scenarios     : 14/14   95% CI [0.785, 1.000]
Cross-source transfer  : 0.524   <- the honest generalisation number
```

The training script fits 60 pipelines on the training split and ranks them on the
*validation* split only. The test split is opened once, at the end, by one model.
The shipped artifact is fit on train+validation and never sees the test set, so
the accuracy above describes that exact artifact.

Read the cross-source transfer figure before the headline. It retrains with one
entire CDC/WHO passage held out and then tests on it, and at 0.524 it says the
model is substantially learning how a particular source page words things rather
than the condition itself. 0.894 is the ceiling; 0.524 is the floor.

An earlier version of this document reported 0.906 for Complement Naive Bayes.
That figure came from ranking 72 pipelines on the same 32-row set that was then
published, with the shipped model refit on that same set, so it measured nothing
and has been withdrawn.

These metrics are stored in:

```text
models/classical/metrics.txt
```

The training script also generates visual evaluation files:

```text
models/classical/model_comparison.png
models/classical/prediction_outcomes.png
models/classical/overall_metrics.png
```

The overall metrics image shows the final best model accuracy and macro F1. The model comparison chart uses a zoomed y-axis so close model scores are easier to see. The prediction outcomes chart shows TP, FP, TN, and FN for each model using one-vs-rest aggregation, because this is a multi-class classification task. Per-class matrix images were removed to keep the evaluation simple for presentation.

### 9.1 Optional Advanced Deep-Learning Model

An optional advanced model script is available in:

```text
backend/scripts/train_deep_learning_model.py
```

It trains bidirectional LSTM and GRU sequence classifiers using tokenized symptom text, an embedding layer, dropout, and early stopping. This script is included for academic comparison because the teacher requested an advanced model option.

Install optional dependencies with:

```bash
pip install -r backend/requirements-advanced.txt
```

Run it with:

```bash
python backend/scripts/train_deep_learning_model.py
```

It saves outputs to:

```text
models/advanced/
```

Generated advanced artifacts include:

```text
models/advanced/sequence_model.keras
models/advanced/tokenizer.joblib
models/advanced/label_encoder.joblib
models/advanced/deep_learning_metrics.txt
models/advanced/deep_learning_metadata.json
models/advanced/training_history.png
models/advanced/advanced_overall_metrics.png
```

One artifact ships, not three. `best_sequence_model.keras` used to be a
byte-for-byte copy of `lstm_model.keras`, and `gru_model.keras` was a losing
candidate that nothing loaded.

Latest advanced result:

```text
Best advanced model: bigru_64
Test accuracy: 0.846
Test macro F1: 0.836
External scenarios: 11/14
```

An earlier version reported 0.844 for an LSTM. That run passed
`validation_data=(x_test, y_test)` to `fit()` under
`EarlyStopping(restore_best_weights=True)`, so the test set was literally
selecting the model's weights. The figure has been withdrawn. Early stopping now
watches the validation split, the tokenizer is fit on the training split only,
and every candidate architecture is listed in the metadata rather than only the
winner.

Important explanation: the TF-IDF model remains the main application model because the dataset is small. LSTM/GRU models are advanced sequence models, but they normally require much more labeled data. In this project, they are used as a comparative experiment, not as a claim of clinical superiority.

Model metadata is stored in:

```text
models/classical/model_metadata.json
```

## 10. Why TF-IDF Instead Of BERT

The project uses TF-IDF with classical ML for the MVP because:

- It is lightweight.
- It runs quickly on a normal laptop.
- It is easier to explain during a presentation.
- It is transparent and suitable for a small dataset.
- It avoids the complexity of training or fine-tuning a large transformer model.

BERT or DistilBERT can be added later if a larger validated dataset is collected.

## 11. The Symptom-Profile Engine and the Arbitration Policy

The symptom-profile rules are a first-class engine, not a hidden fallback.

Earlier, they were hidden: 14 hand-written symptom sets with a bespoke cholera
bonus and a confidence invented as `min(0.92, 0.35 + score/profile_size)` took
over the answer whenever that invented number came out at least as high as the
classifier's probability -- while the response still reported the classifier's
name and the classifier's accuracy. The interface showed the ML model's metrics
for a prediction the ML model had not made.

What it is now:

- Profiles are derived from the dataset's `symptom_terms`, so they are backed by
  the same CDC/WHO passages as every training row. No hand-written sets.
- Each symptom carries an inverse-frequency weight, so "koplik spots" (one label)
  outweighs "fever" (nearly every label). This replaces the cholera bonus.
- Its confidence is a profile-match *share*, and is labelled as not a probability
  everywhere it is published.
- It has its own measured metrics: 0.707 accuracy on the same held-out test set
  the other engines are measured on, and 12/14 on the external scenarios.

The policy, `primary_model_with_rule_fallback_v1`:

1. The requested statistical engine runs, and so does the rule engine.
2. If the statistical engine's probability is at or above 0.20, it decides.
3. Otherwise the rule engine decides, if it has an opinion.
4. Otherwise the ensemble abstains and returns `unknown`.

The rule engine is a fallback, never an override. It speaks only where the
primary has already admitted it does not know, so the two confidence numbers --
which are on different scales -- are never compared with each other.

The 0.20 floor was measured, not guessed: it is the only floor in the validation
sweep that improves on model-only for both accuracy and macro F1.

`decision.decided_by` always names the engine that answered and carries that
engine's own metrics. `engine_opinions` shows what every engine thought,
including the ones that were overruled.

## 12. Recommendation System

Medication and action recommendations are stored in:

```text
backend/app/data/medication_knowledge_base.json
```

The recommendation service is implemented in:

```text
backend/app/services/recommendation_service.py
```

For each disease, the knowledge base contains:

- recommended clinical actions
- medication or treatment information
- standard dosage information when safe to mention
- administration notes
- warnings

The system avoids giving dangerous direct prescriptions. It uses educational wording such as:

- consult a doctor
- perform diagnostic testing
- dose depends on age, weight, pregnancy status, and local guidelines
- do not self-medicate

## 13. Safety Strategy

Medical safety is very important in this project.

The system includes:

- A disclaimer in every API response.
- A visible warning in the frontend.
- No final diagnosis wording.
- No encouragement of self-medication.
- Warnings for severe diseases such as meningitis, cholera, tuberculosis, and dengue.
- Clinician validation reminders.

Example disclaimer:

```text
This application is for academic and educational purposes only. It does not provide a medical diagnosis, does not replace a licensed healthcare professional, and must not be used for self-medication.
```

## 14. Frontend Explanation

The frontend is implemented in:

```text
frontend/streamlit_app.py
```

It provides:

- A medical-themed interface.
- A symptom text area.
- Demo examples in the sidebar.
- Backend URL configuration.
- Model selector with validation metrics.
- Analysis button.
- Prediction result.
- Extracted entities.
- Recommended actions.
- Medication and safety information.

The frontend calls the backend endpoint:

```text
POST http://localhost:8000/analyze
```

It validates that the response contains expected fields before showing the result.

## 15. How The Full Request Flow Works

1. The user types symptoms in Streamlit.
2. Streamlit sends the symptom text to FastAPI.
3. FastAPI validates the request using Pydantic.
4. The NLP service cleans the text.
5. The entity extractor identifies symptoms, diseases, medications, and dosage mentions.
6. The prediction service predicts the disease with the selected model, either `classical` or `advanced`.
7. The recommendation service loads guidance from the knowledge base.
8. FastAPI returns a structured JSON response.
9. Streamlit displays the result in the interface.

## 16. Model Training

The training script is:

```text
backend/scripts/train_model.py
```

To train:

```bash
python backend/scripts/train_model.py
```

The script:

- loads the dataset
- cleans the text
- splits data into train/test sets
- trains multiple candidate models
- evaluates accuracy and macro F1-score
- selects the best model
- saves the model and vectorizer
- saves metrics and metadata

Generated files:

```text
models/classical/trained_model.joblib
models/classical/vectorizer.joblib
models/classical/metrics.txt
models/classical/model_metadata.json
models/classical/model_comparison.png
models/classical/overall_metrics.png
```

## 17. Why Accuracy Is Not Enough

Accuracy alone can be misleading, especially with multiple disease classes.

The project also uses Macro F1-score because it gives a more balanced view across all disease labels.

Macro F1 is useful because every disease matters, not only the most common label.

## 18. Limitations

The system has important limitations:

- The dataset is curated from public symptom pages, not real patient records.
- It is not clinically validated.
- It cannot diagnose patients.
- It cannot replace laboratory tests.
- It cannot replace a doctor or pharmacist.
- It supports only the diseases included in the dataset.
- Similar diseases may share symptoms, causing possible confusion.

## 19. Future Improvements

Possible improvements include:

- Add more validated medical datasets.
- Add real annotated clinical text if legally and ethically available.
- Add spaCy NER trained on medical entities.
- Add multilingual English/French support.
- Add DistilBERT or BioClinicalBERT after dataset expansion.
- Add authentication for real users.
- Add user feedback collection.
- Add explainability visualizations.
- Add probability ranking for top 3 diseases.
- Add doctor/pharmacist review of the knowledge base.

## 20. Setup, Run, And Demo Guide

Install dependencies from the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

Train the main model and generate evaluation visuals:

```bash
python backend/scripts/train_model.py
```

Run the backend:

```bash
cd backend
python -m uvicorn app.main:app --reload
```

Run the frontend from another terminal at the project root:

```bash
python -m streamlit run frontend\streamlit_app.py
```

Open the frontend:

```text
http://localhost:8501
```

Open the API documentation:

```text
http://localhost:8000/docs
```

Recommended live demo cases:

| Case | Input Summary | Expected Result |
|---|---|---|
| Malaria | fever, headache, chills, sweating, body pain | malaria |
| Dengue | high fever, headache, pain behind eyes, joint pain, rash | dengue |
| Cholera | watery diarrhea, vomiting, thirst, dehydration | cholera |
| Meningitis | fever, severe headache, stiff neck, vomiting, light sensitivity | meningitis |

Example malaria input:

```text
I have fever, headache, chills, sweating and body pain for 3 days.
```

Example dengue input:

```text
I have high fever, severe headache, pain behind the eyes, joint pain and rash.
```

Example cholera input:

```text
I have severe watery diarrhea, vomiting, thirst and dehydration after drinking unsafe water.
```

Example meningitis input:

```text
I have fever, severe headache, stiff neck, vomiting and light sensitivity.
```

## 21. Presentation Plan

Recommended duration: 10 to 15 minutes.

Suggested slide order:

| Slide | Topic | Key Message |
|---|---|---|
| 1 | Title | NLP-Based Medical Prescription Management System |
| 2 | Problem Context | Symptoms are unstructured and unsafe self-medication is risky |
| 3 | Objectives | Analyze text, extract entities, predict disease, return safe guidance |
| 4 | Supported Diseases | 14 labels, including the original malaria, typhoid, tuberculosis, and HIV scope |
| 5 | Architecture | Streamlit frontend, FastAPI backend, NLP service, ML model, knowledge base |
| 6 | Dataset | 616 rows from 28 CDC/WHO captures, every row carrying its provenance; not real patient data |
| 7 | NLP Pipeline | Cleaning, phrase normalization, entity extraction, TF-IDF, classification |
| 8 | Entity Extraction | Symptoms, diseases, medications, dosage mentions, simple negation handling |
| 9 | ML Model | 60 pipelines selected on a validation split; best is logistic regression C=4.0. Test accuracy 0.894, cross-source transfer 0.524 |
| 10 | Evaluation Visuals | Overall metrics chart and model comparison chart |
| 11 | Advanced Model | Optional LSTM/GRU experiment, kept separate because the dataset is small |
| 12 | Recommendation System | JSON knowledge base with educational actions, medicines, warnings |
| 13 | Web Demo | Show malaria, dengue, cholera, and meningitis examples |
| 14 | Limitations | Not clinically validated, not diagnosis, no self-medication |
| 15 | Future Work | Larger datasets, medical NER, multilingual support, BERT, clinician validation |

Final message to say:

```text
This project is not a replacement for doctors. It is an academic demonstration of how NLP and machine learning can structure symptom text and provide safe educational guidance.
```

## 22. Important Files Checklist

| File | Purpose |
|---|---|
| `backend/app/main.py` | FastAPI application setup |
| `backend/app/core/paths.py` | Centralized project paths for data and model artifacts |
| `backend/app/api/routes.py` | All HTTP endpoints |
| `backend/app/api/dependencies.py` | Optional API-key check |
| `backend/app/core/config.py` | Environment configuration (database, CORS, auth limits) |
| `backend/app/core/lexicon.py` | Shared symptom vocabulary access |
| `backend/app/core/evaluation.py` | The one split and interval helpers every script uses |
| `backend/app/core/use_cases.py` | Parser for the scenario file |
| `backend/app/db/models.py` | Consultation and audit tables |
| `backend/app/db/session.py` | SQLite engine and session handling |
| `backend/app/services/prediction_engines.py` | The three prediction engines |
| `backend/app/services/consultation_service.py` | Persistence, history, export, reports |
| `backend/app/services/nlp_service.py` | Main NLP orchestration and entity extraction |
| `backend/app/services/disease_prediction_service.py` | Loads model and predicts disease |
| `backend/app/services/recommendation_service.py` | Loads recommendation knowledge base |
| `backend/app/utils/text_cleaning.py` | Text preprocessing and phrase normalization |
| `backend/app/data/symptoms_dataset.csv` | Training dataset, every row carrying its provenance |
| `backend/app/data/symptom_lexicon.json` | Shared symptom vocabulary (source and cleaned views) |
| `backend/app/data/dataset_sources.json` | CDC/WHO source registry |
| `backend/app/data/medication_knowledge_base.json` | Educational medication and action guidance |
| `backend/scripts/scrape_medical_sources.py` | Controlled CDC/WHO scraping script |
| `backend/scripts/build_dataset.py` | Rebuilds the dataset from fetched sources, with provenance |
| `backend/scripts/evaluate_engines.py` | Measures every engine on one shared split |
| `tests/` | 105 pytest tests, including the engine-attribution regression suite |
| `backend/scripts/train_model.py` | Main TF-IDF model training and evaluation visuals |
| `backend/scripts/train_deep_learning_model.py` | Optional LSTM/GRU advanced experiment |
| `frontend/streamlit_app.py` | Streamlit user interface |
| `models/classical/metrics.txt` | Latest model metrics |
| `models/classical/model_comparison.png` | Candidate model comparison visual |
| `models/classical/prediction_outcomes.png` | TP, FP, TN, and FN by model using one-vs-rest aggregation |
| `models/classical/overall_metrics.png` | Overall accuracy and macro F1 visual |
| `tests/use_case_tests.txt` | Manual demo and validation scenarios |

## 23. How To Explain The Project In One Minute

This project is a web-based NLP medical assistant for academic demonstration. The user enters symptoms in natural language. The backend cleans the text, extracts medical entities using rule-based dictionaries, predicts a likely disease using a TF-IDF machine learning model, then returns safe educational treatment information from a JSON knowledge base. The system supports 14 common diseases and uses source-backed symptom examples from CDC and WHO. It is not a diagnosis system and always recommends clinical validation.
