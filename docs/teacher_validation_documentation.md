# Teacher Validation Documentation

## 1. General Project Presentation

### Project Title

NLP-Based Medical Prescription Management System

### Project Objective

The objective of this project is to build a web application that uses Natural Language Processing to analyze symptoms written by a user in natural language, extract medical entities, predict a possible disease, and provide educational medication and treatment guidance with safety warnings.

The project is designed for academic demonstration. It is not a real diagnosis system and must not be used for self-medication.

### Main Problem Addressed

In many regions, especially where access to reliable medical information is limited, patients may misunderstand symptoms or misuse medication. This project demonstrates how NLP can support a preliminary symptom analysis workflow by organizing symptom text and linking it to disease and medication information.

### Main Features

- User enters symptoms in natural language.
- The app cleans and normalizes the text.
- The app extracts medical entities such as symptoms, diseases, medications, and dosage mentions.
- The app predicts a likely disease/condition.
- The app returns recommended clinical actions.
- The app displays medication information and warnings.
- The app clearly shows a medical disclaimer.

### Supported Disease Labels

The current application supports 14 labels:

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

The first core diseases match the project subject, and the additional diseases were added to improve the demo and show that the architecture can scale.

## 2. Technologies Used

### Backend

- Python
- FastAPI
- Pydantic
- Uvicorn

### Frontend

- Streamlit
- HTML/CSS embedded inside Streamlit for custom medical design

### NLP and Machine Learning

- scikit-learn
- TF-IDF vectorization
- Logistic Regression
- Calibrated Linear SVC
- Complement Naive Bayes
- joblib for model persistence

### Data Format

- CSV for the symptom dataset
- JSON for dataset sources
- JSON for the medication knowledge base

## 3. Global Architecture

The application follows a simple client-server architecture.

```text
User
  |
  v
Streamlit Frontend
  |
  v
FastAPI Backend
  |
  v
NLP Service
  |
  |-- Text Cleaning
  |-- Entity Extraction
  |-- Disease Prediction
  |-- Recommendation System
  v
Structured JSON Response
  |
  v
Displayed Results In Frontend
```

The frontend and backend run separately:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`

## 4. Folder-by-Folder Explanation

## 4.1 Root Folder

Root folder:

```text
NLP Project/
```

This is the main project directory. It contains the backend, frontend, documentation, trained models, and testing files.

Important root files:

```text
README.md
.gitignore
use_case_tests.txt
Prompt.txt
```

### `README.md`

This file explains how to install, run, train, and test the project. It also documents the supported diseases, dataset sources, and main technical choices.

### `.gitignore`

This file prevents unnecessary files from being committed to Git. For example:

- `.venv/`
- `__pycache__/`
- `*.pyc`
- temporary logs

This keeps the GitHub repository clean.

### `use_case_tests.txt`

This file contains ready-to-use test scenarios for the teacher or demo. Each use case includes:

- symptom input
- expected predicted disease
- expected extracted symptoms

Example:

```text
Input: I have high fever, severe headache, pain behind the eyes, joint pain and rash.
Expected: dengue
```

### `Prompt.txt`

This file contains the original project prompt and requirements used to guide the implementation.

## 4.2 Backend Folder

Backend folder:

```text
backend/
```

This folder contains the FastAPI backend and all NLP/ML logic.

Main backend structure:

```text
backend/
  app/
  scripts/
  requirements.txt
```

### `backend/requirements.txt`

This file lists the Python dependencies required to run the backend and frontend.

Important dependencies include:

- `fastapi`
- `uvicorn`
- `streamlit`
- `pandas`
- `scikit-learn`
- `joblib`
- `requests`

Install command:

```bash
pip install -r backend/requirements.txt
```

## 4.3 Backend App Folder

Folder:

```text
backend/app/
```

This is the main Python package of the backend application.

It contains:

```text
main.py
api/
models/
services/
utils/
data/
```

### `backend/app/main.py`

This file creates the FastAPI application.

Responsibilities:

- Create the FastAPI app.
- Configure CORS.
- Include API routes.
- Define app title, description, and version.

Important code concept:

```python
app = FastAPI(...)
app.include_router(router)
```

The backend is launched with:

```bash
cd backend
python -m uvicorn app.main:app --reload
```

## 4.4 API Folder

Folder:

```text
backend/app/api/
```

This folder contains API route definitions.

Important file:

```text
backend/app/api/routes.py
```

### `routes.py`

This file defines the backend endpoints.

Endpoints:

```text
GET /health
POST /analyze
```

### `GET /health`

Used to verify that the backend is running.

Example response:

```json
{
  "status": "ok",
  "service": "medical-prescription-nlp"
}
```

### `POST /analyze`

This is the main endpoint.

It receives symptom text and returns:

- cleaned text
- extracted entities
- predicted disease
- confidence score
- recommended actions
- medication information
- disclaimer

Example request:

```json
{
  "text": "I have fever, headache, chills and body pain."
}
```

## 4.5 Models Folder

Folder:

```text
backend/app/models/
```

This folder contains Pydantic schemas.

Important file:

```text
backend/app/models/schemas.py
```

### `schemas.py`

This file defines the structure of API input and output data.

Main schemas:

- `AnalyzeRequest`
- `ExtractedEntities`
- `MedicationRecommendation`
- `AnalyzeResponse`
- `HealthResponse`

Why it is useful:

- It validates user input.
- It makes API responses consistent.
- It improves FastAPI documentation automatically.

## 4.6 Services Folder

Folder:

```text
backend/app/services/
```

This folder contains the main business logic of the app.

Files:

```text
nlp_service.py
disease_prediction_service.py
recommendation_service.py
```

### `nlp_service.py`

This is the orchestration service.

Responsibilities:

- Clean the input text.
- Extract entities.
- Call the prediction model.
- Call the recommendation service.
- Build the final response.

It contains dictionaries for:

- symptom terms
- disease terms
- medication terms

It also includes dosage extraction patterns.

Example extracted entities:

```json
{
  "symptoms": ["fever", "headache", "chills"],
  "diseases": [],
  "medications": [],
  "dosage_mentions": []
}
```

### `disease_prediction_service.py`

This service loads the trained model and predicts the disease.

Responsibilities:

- Load `models/trained_model.joblib`.
- Clean incoming text.
- Generate prediction probabilities.
- Return the best disease and confidence score.
- Apply symptom-profile fallback rules for strong disease patterns.

The fallback helps with strong symptom combinations such as:

- watery diarrhea + dehydration = cholera
- stiff neck + headache + light sensitivity = meningitis
- fever + chills + headache = malaria

### `recommendation_service.py`

This service loads the medication knowledge base.

Responsibilities:

- Read `medication_knowledge_base.json`.
- Find recommendations for the predicted disease.
- Return clinical actions and medication information.

The recommendations are educational and safety-focused.

## 4.7 Utils Folder

Folder:

```text
backend/app/utils/
```

Important file:

```text
backend/app/utils/text_cleaning.py
```

### `text_cleaning.py`

This file handles text preprocessing.

Steps:

- Unicode normalization.
- Lowercasing.
- Medical phrase normalization.
- Punctuation removal.
- Whitespace normalization.
- Basic stop-word removal.

Examples:

```text
diarrhoea -> diarrhea
photophobia -> light sensitivity
coryza -> runny nose
```

This improves prediction quality because symptoms can be written in different ways.

## 4.8 Data Folder

Folder:

```text
backend/app/data/
```

This folder stores all source-backed data and knowledge base files.

Files:

```text
symptoms_dataset.csv
dataset_sources.json
medication_knowledge_base.json
```

### `symptoms_dataset.csv`

This is the model training dataset.

Columns:

- `text`
- `disease`
- `source_name`
- `source_url`
- `source_note`

The dataset is curated from CDC and WHO symptom descriptions.

Important clarification:

```text
This is not real patient data. It is a curated educational dataset based on official public health sources.
```

### `dataset_sources.json`

This file lists the sources used for each disease.

Sources include:

- CDC malaria page
- WHO malaria page
- CDC tuberculosis page
- WHO HIV page
- CDC COVID page
- WHO dengue page
- WHO cholera page
- CDC pneumonia page
- CDC meningitis page
- WHO hepatitis B page
- CDC measles page

### Controlled scraping script

The scraping implementation is located in:

```text
backend/scripts/scrape_medical_sources.py
```

It reads `dataset_sources.json`, downloads each CDC/WHO page, extracts relevant sections such as symptoms, signs, diagnosis, treatment, and prevention, then saves the raw extracted data in:

```text
backend/app/data/scraped_medical_sources.json
backend/app/data/scraped_medical_sources.csv
```

This script documents the automated data collection method. The final dataset remains curated because medical content needs human review before training.

### `medication_knowledge_base.json`

This file maps diseases to treatment guidance.

For each disease it contains:

- recommended actions
- medication information
- dosage information or explanation
- administration notes
- warnings

Example diseases with safety-focused guidance:

- malaria
- dengue
- cholera
- meningitis
- tuberculosis

## 4.9 Scripts Folder

Folder:

```text
backend/scripts/
```

Important file:

```text
backend/scripts/train_model.py
```

### `train_model.py`

This script trains and evaluates the machine learning model.

It performs:

- dataset loading
- text cleaning
- train/test splitting
- TF-IDF vectorization
- training multiple models
- comparing metrics
- saving the best model

Models compared:

- Logistic Regression
- Calibrated Linear SVC
- Complement Naive Bayes

Generated files:

```text
models/trained_model.joblib
models/vectorizer.joblib
models/metrics.txt
models/model_metadata.json
```

Run command:

```bash
python backend/scripts/train_model.py
```

## 4.10 Frontend Folder

Folder:

```text
frontend/
```

Important file:

```text
frontend/streamlit_app.py
```

### `streamlit_app.py`

This file contains the complete user interface.

Frontend responsibilities:

- Display medical-themed interface.
- Show demo scenarios.
- Allow user to enter symptoms.
- Send symptoms to the backend.
- Validate backend response.
- Display prediction and recommendations.
- Display safety warnings.

The frontend communicates with:

```text
http://localhost:8000/analyze
```

Run command:

```bash
python -m streamlit run frontend/streamlit_app.py
```

## 4.11 Models Folder

Folder:

```text
models/
```

This folder stores trained ML artifacts.

Files:

```text
trained_model.joblib
vectorizer.joblib
metrics.txt
model_metadata.json
```

### `trained_model.joblib`

The saved best machine learning pipeline.

### `vectorizer.joblib`

The saved TF-IDF vectorizer.

### `metrics.txt`

Contains model evaluation results.

Current results:

```text
Best model: linear_svc_calibrated
Accuracy: 0.844
Macro F1: 0.837
```

### `model_metadata.json`

Contains:

- best model name
- number of dataset rows
- number of labels
- label list
- candidate model results

## 4.12 Docs Folder

Folder:

```text
docs/
```

This folder contains documentation for understanding and defending the project.

Files:

```text
architecture.md
dataset_description.md
project_summary.md
full_project_documentation.md
teacher_q_and_a.md
demo_script.md
teacher_validation_documentation.md
```

### `architecture.md`

Explains the technical architecture and request flow.

### `dataset_description.md`

Explains the dataset, labels, preprocessing, and sources.

### `project_summary.md`

Short summary of the project.

### `full_project_documentation.md`

Detailed explanation of the whole project.

### `teacher_q_and_a.md`

Prepared answers to likely teacher questions.

### `demo_script.md`

Step-by-step presentation/demo script.

### `teacher_validation_documentation.md`

This document. It explains the general project and every folder/file for validation.

## 5. End-to-End Example

Input:

```text
I have high fever, severe headache, pain behind the eyes, joint pain and rash.
```

Processing:

1. Text is cleaned.
2. Symptoms are extracted.
3. TF-IDF vector is generated.
4. Model predicts disease.
5. Recommendation service retrieves guidance.

Expected output:

```text
Predicted disease: dengue
Extracted symptoms: fever, headache, pain behind eyes, muscle pain, rash
Recommended action: consult a healthcare professional and monitor warning signs
Medication information: paracetamol and hydration guidance, avoid aspirin/ibuprofen unless clinician says otherwise
```

## 6. How To Launch The App

### Step 1: Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
```

### Step 2: Start Backend

```bash
cd backend
python -m uvicorn app.main:app --reload
```

Backend docs:

```text
http://localhost:8000/docs
```

### Step 3: Start Frontend

Open another terminal from the root folder:

```bash
.venv\Scripts\activate
python -m streamlit run frontend\streamlit_app.py
```

Frontend:

```text
http://localhost:8501
```

## 7. How To Train The Model

From the root folder:

```bash
python backend/scripts/train_model.py
```

This regenerates:

```text
models/trained_model.joblib
models/vectorizer.joblib
models/metrics.txt
models/model_metadata.json
```

## 8. Validation And Testing

Use:

```text
use_case_tests.txt
```

This file contains test cases for all 14 labels.

Example:

```text
Input: I have fever, severe headache, stiff neck, vomiting and light sensitivity.
Expected: meningitis
```

## 9. Safety And Ethics

The system is intentionally designed with safety limitations.

It does not say:

```text
You have disease X.
```

Instead, it says:

```text
Possible disease: X
```

It always includes medical disclaimers and encourages professional validation.

## 10. What To Say To The Teacher

If asked to summarize:

```text
This project is an academic NLP medical assistant. The user writes symptoms in natural language. The backend cleans the text, extracts medical entities, predicts a possible disease using a TF-IDF machine learning model, and retrieves educational treatment guidance from a JSON knowledge base. The system supports 14 diseases using a source-backed dataset curated from CDC and WHO. It is not a real diagnosis system and includes strong medical safety disclaimers.
```

## 11. Strong Points Of The Project

- Complete backend and frontend.
- Source-backed dataset.
- 14 disease labels.
- Rule-based entity extraction.
- Multiple ML models compared.
- Saved model artifacts.
- Medical knowledge base.
- Safety-first design.
- Documentation and demo test cases.

## 12. Known Limitations

- Dataset is not real patient data.
- Model is not clinically validated.
- Similar diseases can share symptoms.
- Only supports known labels.
- Does not replace medical testing.
- Medication information must be validated by professionals.

## 13. Future Work

- Add real anonymized clinical data with ethical approval.
- Add French and Arabic support.
- Add trained spaCy NER.
- Add DistilBERT or BioClinicalBERT.
- Add top-3 disease suggestions.
- Add explainability for model predictions.
- Add doctor/pharmacist validation of the knowledge base.
- Add deployment with Docker.
