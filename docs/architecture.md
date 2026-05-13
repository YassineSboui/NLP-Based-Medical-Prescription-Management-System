# Architecture

## Components

- `frontend/streamlit_app.py`: Streamlit user interface for demos.
- `backend/app/main.py`: FastAPI application entry point.
- `backend/app/api/routes.py`: API routes including `/health` and `/analyze`.
- `backend/app/services/nlp_service.py`: Orchestrates cleaning, entity extraction, prediction, and recommendations.
- `backend/app/services/disease_prediction_service.py`: Loads the saved TF-IDF classifier and applies symptom-profile fallback rules for high-signal disease patterns.
- `backend/app/services/recommendation_service.py`: Reads medication and action guidance from the JSON knowledge base.
- `backend/app/utils/text_cleaning.py`: Text normalization and stop-word removal.
- `backend/app/data/symptoms_dataset.csv`: Source-backed curated training dataset with CDC/WHO source metadata.
- `backend/app/data/dataset_sources.json`: Source registry for diseases and symptom references.
- `backend/app/data/medication_knowledge_base.json`: Educational medication information.
- `backend/scripts/train_model.py`: Repeatable model training script that compares Logistic Regression, calibrated Linear SVC, and Complement Naive Bayes.

## Request Flow

1. The user enters symptoms in the Streamlit frontend.
2. Streamlit sends a POST request to `POST /analyze`.
3. FastAPI validates the request with Pydantic schemas.
4. The NLP service cleans the text and extracts entities using dictionaries.
5. The prediction service classifies the text using the best saved TF-IDF model, then compares it with a transparent symptom-profile fallback for common high-signal cases.
6. The recommendation service maps the predicted disease to actions and medication information.
7. The API returns extracted entities, predicted disease, confidence, recommended actions, medication information, and a safety disclaimer.

## Model Choice

The MVP uses classical NLP baselines instead of BERT because they are lightweight, transparent, fast to run locally, and easier to explain during a 10-15 minute academic demonstration. The training script compares multiple models and saves the best performer with metrics and metadata. The symptom-profile fallback improves reliability on obvious combinations such as watery diarrhea and dehydration for cholera, stiff neck and light sensitivity for meningitis, or fever, chills, and headache for malaria. A future version can add spaCy NER or fine-tuned DistilBERT after collecting a larger validated dataset.
