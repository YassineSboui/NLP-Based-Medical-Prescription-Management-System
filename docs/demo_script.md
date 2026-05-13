# Demo Script

## 1. Before The Demo

Start the backend:

```bash
cd backend
python -m uvicorn app.main:app --reload
```

Start the frontend from the project root:

```bash
python -m streamlit run frontend\streamlit_app.py
```

Open:

```text
http://localhost:8501
```

## 2. Short Introduction

Say:

```text
This project is an academic NLP-based medical prescription management system. It analyzes patient symptoms written in natural language, extracts medical entities, predicts a possible disease, and returns educational medication guidance with strict safety warnings.
```

## 3. Explain Architecture

Say:

```text
The application has a FastAPI backend, a Streamlit frontend, a machine learning prediction service, a rule-based entity extractor, and a JSON medical knowledge base.
```

## 4. Demo Case 1: Malaria

Paste:

```text
I have fever, headache, chills, sweating and body pain for 3 days.
```

Expected:

```text
malaria
```

Explain:

```text
The system extracts symptoms such as fever, headache, chills, sweating, and body pain. The model predicts malaria and returns actions such as diagnostic testing and doctor consultation.
```

## 5. Demo Case 2: Dengue

Paste:

```text
I have high fever, severe headache, pain behind the eyes, joint pain and rash.
```

Expected:

```text
dengue
```

Explain:

```text
Dengue has symptoms such as high fever, pain behind the eyes, joint pain, and rash. The app also warns against unsafe self-medication.
```

## 6. Demo Case 3: Cholera

Paste:

```text
I have severe watery diarrhea, vomiting, thirst and dehydration after drinking unsafe water.
```

Expected:

```text
cholera
```

Explain:

```text
The system detects watery diarrhea and dehydration. It recommends urgent rehydration and medical care because cholera can become severe quickly.
```

## 7. Demo Case 4: Meningitis

Paste:

```text
I have fever, severe headache, stiff neck, vomiting and light sensitivity.
```

Expected:

```text
meningitis
```

Explain:

```text
Meningitis is treated as an emergency. The app emphasizes urgent medical care and does not suggest self-treatment.
```

## 8. Explain Safety

Say:

```text
Because this is medical content, the system never claims to diagnose the patient. It provides educational information and always recommends validation by a healthcare professional.
```

## 9. Explain Model

Say:

```text
The model uses TF-IDF text vectorization. The training script compares Logistic Regression, calibrated Linear SVC, and Complement Naive Bayes, then saves the best model.
```

## 10. Closing

Say:

```text
The current version is a strong academic MVP. Future work would include larger validated datasets, trained medical NER, multilingual support, and medical expert validation.
```
