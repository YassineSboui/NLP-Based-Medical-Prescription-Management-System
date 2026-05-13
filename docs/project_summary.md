# Project Summary

This project is an academic MVP for an NLP-based medical prescription management system. It accepts patient-reported symptoms in natural language, extracts simple medical entities, suggests a likely common disease, and returns educational medication information with safety warnings.

The application focuses on common conditions included in the project brief:

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

The system is designed for demonstration and learning. Its dataset is curated from public CDC and WHO symptom references, not from real patient records. It is not a diagnostic device and must not be used for self-medication.

## Main Features

- Natural-language symptom input.
- Rule-based entity extraction for symptoms, diseases, medications, and dosage mentions.
- Disease prediction using TF-IDF and validation-based model selection.
- Medication information from a JSON knowledge base.
- Safety-first recommendations emphasizing tests, doctor consultation, and contraindication warnings.
- FastAPI backend and medical-themed Streamlit frontend.

## Current MVP Limits

- The dataset is small and curated from official public symptom pages, but it is not a validated clinical dataset.
- Predictions are based on symptom text only and cannot replace medical testing.
- Medication dosages are educational summaries, not prescriptions.
- Future versions should use validated clinical datasets and medical review.
