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
  requirements.txt
frontend/
  streamlit_app.py
models/
  trained_model.joblib
  vectorizer.joblib
  metrics.txt
  model_metadata.json
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

- `GET /health`: checks if the backend is running.
- `POST /analyze`: analyzes symptom text and returns predictions and recommendations.

Example request:

```json
{
  "text": "I have fever, headache, chills and body pain."
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
- Removes basic stop words.

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
linear_svc_calibrated
```

Current metrics:

```text
Accuracy: 0.844
Macro F1: 0.837
```

These metrics are stored in:

```text
models/metrics.txt
```

Model metadata is stored in:

```text
models/model_metadata.json
```

## 10. Why TF-IDF Instead Of BERT

The project uses TF-IDF with classical ML for the MVP because:

- It is lightweight.
- It runs quickly on a normal laptop.
- It is easier to explain during a presentation.
- It is transparent and suitable for a small dataset.
- It avoids the complexity of training or fine-tuning a large transformer model.

BERT or DistilBERT can be added later if a larger validated dataset is collected.

## 11. Symptom-Profile Fallback

The system also includes a transparent symptom-profile fallback.

This means that for very strong symptom combinations, rules can support the ML model.

Examples:

- Fever + chills + headache can strongly suggest malaria.
- Watery diarrhea + dehydration can strongly suggest cholera.
- Stiff neck + headache + light sensitivity can strongly suggest meningitis.

This improves demo reliability and makes some decisions easier to explain.

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
6. The prediction service predicts the disease.
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
models/trained_model.joblib
models/vectorizer.joblib
models/metrics.txt
models/model_metadata.json
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

## 20. How To Explain The Project In One Minute

This project is a web-based NLP medical assistant for academic demonstration. The user enters symptoms in natural language. The backend cleans the text, extracts medical entities using rule-based dictionaries, predicts a likely disease using a TF-IDF machine learning model, then returns safe educational treatment information from a JSON knowledge base. The system supports 14 common diseases and uses source-backed symptom examples from CDC and WHO. It is not a diagnosis system and always recommends clinical validation.
