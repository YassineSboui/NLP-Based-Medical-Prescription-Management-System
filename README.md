# NLP-Based Medical Prescription Management System

Academic MVP for analyzing patient-reported symptoms, extracting medical entities, suggesting likely common diseases, and returning educational medication information with safety warnings.

## Safety Disclaimer

This project is for academic demonstration only. It does not provide a real medical diagnosis, does not replace a doctor, and must not be used for self-medication. All medicines, dosages, diagnoses, and tests must be confirmed by a qualified healthcare professional.

## Features

- Symptom analysis from free-text input.
- Rule-based extraction of symptoms, diseases, medication names, and dosage mentions.
- Disease prediction using TF-IDF and automatic model selection across Logistic Regression, calibrated Linear SVC, and Complement Naive Bayes.
- Medication and dosage guidance from a JSON knowledge base.
- Clear medical warnings and recommended clinical actions.
- FastAPI backend and Streamlit frontend.
- Medical-themed Streamlit interface with teal/blue visual design, cards, example scenarios, and clear safety warnings.

## Current Disease Scope

The expanded demo currently supports 14 source-backed labels:

- `malaria`
- `typhoid fever`
- `tuberculosis`
- `hiv`
- `flu`
- `common cold`
- `gastroenteritis`
- `covid-like illness`
- `dengue`
- `cholera`
- `pneumonia`
- `meningitis`
- `hepatitis b`
- `measles`

## Project Structure

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
  advanced/
tests/
  use_case_tests.txt
docs/
  full_project_documentation.md
  technical_report.md
  teacher_q_and_a.md
```

## Documentation

Detailed documentation is available in:

- `docs/full_project_documentation.md`: complete project documentation, architecture, dataset, setup, demo guide, presentation plan, model evaluation, advanced LSTM/GRU explanation, and file checklist.
- `docs/technical_report.md`: formal English technical report draft for submission.
- `docs/teacher_q_and_a.md`: likely teacher questions with prepared answers.
- `tests/use_case_tests.txt`: ready-to-paste test cases with expected results.

## Setup

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

## Train the Model

Training is optional because the API can train a model in memory at startup if no saved model exists. To save a repeatable model and metrics:

```bash
python backend/scripts/train_model.py
```

This creates:

- `models/classical/trained_model.joblib`
- `models/classical/vectorizer.joblib`
- `models/classical/metrics.txt`
- `models/classical/model_metadata.json`
- `models/classical/model_comparison.png`
- `models/classical/prediction_outcomes.png`
- `models/classical/overall_metrics.png`

## Optional Advanced LSTM/GRU Experiment

The production/demo model remains the TF-IDF model because the dataset is small and classical text classifiers usually generalize better in this situation. An optional advanced neural-network experiment is included for academic comparison:

```bash
pip install -r backend/requirements-advanced.txt
python backend/scripts/train_deep_learning_model.py
```

This trains bidirectional LSTM and GRU sequence classifiers and saves their comparison artifacts in:

```text
models/advanced/
```

Generated advanced artifacts include:

- `models/advanced/lstm_model.keras`
- `models/advanced/gru_model.keras`
- `models/advanced/best_sequence_model.keras`
- `models/advanced/tokenizer.joblib`
- `models/advanced/label_encoder.joblib`
- `models/advanced/deep_learning_metrics.txt`
- `models/advanced/deep_learning_metadata.json`
- `models/advanced/training_history.png`
- `models/advanced/advanced_overall_metrics.png`
- `models/advanced/advanced_prediction_outcomes.png`

Latest advanced result: the best advanced model was `lstm_u64_e96_s48_b8_pool_lr7e4_seed7` with 0.844 accuracy and 0.838 macro F1. The optimized classical model remains better with 0.906 accuracy and 0.907 macro F1. Important: the advanced model is included to demonstrate sequence-model experimentation, not because it is guaranteed to outperform TF-IDF on only 128 rows.

## Optional: Scrape Source Pages

The project includes a controlled scraper for the CDC/WHO source pages listed in `backend/app/data/dataset_sources.json`:

```bash
python backend/scripts/scrape_medical_sources.py
```

This creates:

- `backend/app/data/scraped_medical_sources.json`
- `backend/app/data/scraped_medical_sources.csv`

The scraped files are for traceability and raw data collection. The training dataset remains curated manually for medical safety and data quality.

## Run the Backend

To launch the complete application on Windows, run one of these from the project root:

```bash
run_app.bat
```

or:

```bash
powershell -ExecutionPolicy Bypass -File .\run_app.ps1
```

This starts both FastAPI and Streamlit. The frontend will be available at:

```text
http://localhost:8501
```

If you prefer to run services manually, start the backend first:

```bash
cd backend
uvicorn app.main:app --reload
```

API documentation will be available at:

```text
http://localhost:8000/docs
```

Main endpoint:

```text
POST http://localhost:8000/analyze
```

Available model list:

```text
GET http://localhost:8000/models
```

Example JSON body:

```json
{
  "text": "I have fever, headache, chills, sweating and body pain for 3 days.",
  "model_key": "classical"
}
```

Supported `model_key` values are `classical` and `advanced`.

## Run the Frontend

Open a second terminal from the project root:

```bash
streamlit run frontend/streamlit_app.py
```

If the backend runs on another URL, set `API_URL` before starting Streamlit.

## Demo Scenario

Input:

```text
I have fever, headache, chills, sweating and body pain for 3 days.
```

Expected output:

- Extracted symptoms: fever, headache, chills, sweating, body pain.
- Possible disease: malaria.
- Recommended action: consult a doctor and perform diagnostic testing.
- Medication information: educational antimalarial and fever-relief information from the knowledge base.
- Safety disclaimer: not a diagnosis and not a prescription.

## Dataset Sources

The current dataset is curated from official public health symptom references, mainly CDC and WHO pages. Each training row includes `source_name`, `source_url`, and `source_note` columns.

Important: this is not a real patient-record dataset and it is not clinically validated. It is suitable for an academic MVP and demonstration of the NLP pipeline.

Main references include:

- CDC Malaria: `https://www.cdc.gov/malaria/about/index.html`
- WHO Malaria: `https://www.who.int/news-room/fact-sheets/detail/malaria`
- WHO Typhoid: `https://www.who.int/news-room/fact-sheets/detail/typhoid`
- CDC Typhoid: `https://www.cdc.gov/typhoid-fever/signs-symptoms/index.html`
- CDC Tuberculosis: `https://www.cdc.gov/tb/signs-symptoms/index.html`
- WHO Tuberculosis: `https://www.who.int/news-room/fact-sheets/detail/tuberculosis`
- WHO HIV/AIDS: `https://www.who.int/news-room/fact-sheets/detail/hiv-aids`
- CDC HIV: `https://www.cdc.gov/hiv/about/index.html`
- CDC Flu: `https://www.cdc.gov/flu/signs-symptoms/index.html`
- CDC Common Cold: `https://www.cdc.gov/common-cold/about/index.html`
- CDC Norovirus: `https://www.cdc.gov/norovirus/about/index.html`
- WHO Diarrhoeal Disease: `https://www.who.int/news-room/fact-sheets/detail/diarrhoeal-disease`
- CDC COVID-19: `https://www.cdc.gov/covid/signs-symptoms/index.html`
- WHO Dengue: `https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue`
- WHO Cholera: `https://www.who.int/news-room/fact-sheets/detail/cholera`
- CDC Pneumonia: `https://www.cdc.gov/pneumonia/about/index.html`
- CDC Meningitis: `https://www.cdc.gov/meningitis/about/index.html`
- WHO Hepatitis B: `https://www.who.int/news-room/fact-sheets/detail/hepatitis-b`
- CDC Measles: `https://www.cdc.gov/measles/signs-symptoms/index.html`

## API Response Shape

```json
{
  "original_text": "...",
  "cleaned_text": "...",
  "extracted_entities": {
    "symptoms": [],
    "diseases": [],
    "medications": [],
    "dosage_mentions": []
  },
  "predicted_disease": "malaria",
  "confidence": 0.82,
  "model_used": {
    "key": "classical",
    "name": "complement_naive_bayes",
    "accuracy": 0.906,
    "macro_f1": 0.907
  },
  "recommended_actions": [],
  "recommended_medicines": [],
  "disclaimer": "..."
}
```

## Technical Choices

- FastAPI for a simple REST backend.
- Streamlit for a fast demo-ready frontend.
- TF-IDF with validation-based model selection for a transparent NLP baseline.
- User-selectable prediction model: classical TF-IDF model or advanced LSTM model.
- Symptom-profile fallback to make common demo cases more reliable and explainable.
- JSON knowledge base for easy editing and presentation.
- Rule-based medical entity extraction for reliability and explainability.

## Future Improvements

- Expand the dataset with verified medical sources.
- Add spaCy NER trained on annotated medical text.
- Fine-tune DistilBERT or BioClinicalBERT for symptom classification.
- Add multilingual support for English and French symptom descriptions.
- Add authentication and audit logs for real clinical workflows.
- Validate all medical content with licensed healthcare professionals.
"# NLP-Based-Medical-Prescription-Management-System" 
