# NLP-Based Medical Prescription Management System

## Technical Report

**Project Type:** Academic NLP and Web Development Project  
**Duration:** 5 weeks  
**Main Technologies:** Python, FastAPI, Streamlit, scikit-learn, TF-IDF, JSON, CSV, optional TensorFlow/Keras

---

## Abstract

This project presents an academic Natural Language Processing system for medical prescription management and symptom analysis. The application allows users to enter symptoms in natural language, extracts medical entities, predicts a possible disease or condition, and returns educational medication and treatment guidance with clear safety warnings. The system focuses on common infectious and respiratory/gastrointestinal diseases, including malaria, typhoid fever, tuberculosis, HIV, dengue, cholera, pneumonia, meningitis, hepatitis B, measles, flu, common cold, gastroenteritis, and COVID-like illness.

The implemented solution uses a FastAPI backend, a Streamlit frontend, a rule-based entity extraction module, TF-IDF vectorization, classical machine learning classifiers, and a SQLite consultation store with an audit trail. The dataset is derived from official public health symptom descriptions published by CDC and WHO, with each row recording whether it is a direct source rendering, a recombination of one source passage's symptoms, or a marked paraphrase of another row. Model selection is performed on a validation split and the held-out test split is read once; the shipped artifact is fit on train+validation and never sees the test set. The best model is logistic regression (C=4.0) over TF-IDF unigrams, with a test accuracy of 0.894 (95% CI 0.828-0.937) and a macro F1 of 0.888. A cross-source transfer diagnostic, which retrains with an entire source passage held out, gives a pooled recall of 0.524 and is the more honest indicator of generalisation. The system is designed strictly for academic demonstration and does not provide real medical diagnosis or prescriptions.

---

## 1. Introduction

Medical prescription management is an important public health topic, especially in regions where access to reliable healthcare information may be limited. Inadequate prescription management can lead to self-medication, inappropriate drug use, delayed diagnosis, antimicrobial resistance, and avoidable complications.

Natural Language Processing can help transform unstructured patient-reported symptoms into structured information. By extracting symptoms and mapping them to possible diseases, an NLP-based system can support educational triage and help users understand when professional care is necessary.

This project aims to develop a functional web application that demonstrates how NLP and machine learning can be used to analyze symptoms and provide safety-first medication information.

---

## 2. Problem Context

In many regions, including parts of Sub-Saharan Africa, common infectious diseases such as malaria, typhoid fever, tuberculosis, and HIV remain major health concerns. Patients may describe symptoms informally, and these descriptions are often ambiguous. A digital assistant can help organize this information and provide preliminary educational guidance.

However, medical systems must be designed carefully because wrong diagnosis or unsafe medication advice can harm patients. For this reason, this project does not claim to diagnose users. Instead, it provides possible disease suggestions, recommended actions, and warnings that emphasize consultation with qualified healthcare professionals.

---

## 3. Project Objectives

The main objectives of this project are:

- Build a functional web application for symptom analysis.
- Create a source-backed medical dataset from reliable public health references.
- Implement an NLP pipeline for cleaning, normalization, and entity extraction.
- Train and evaluate machine learning models for disease prediction.
- Implement a recommendation system that maps diseases to safe educational guidance.
- Provide a user-friendly interface for demonstration.
- Include clear safety disclaimers and avoid unsafe medical advice.

---

## 4. System Scope

The system currently supports 14 disease labels:

| Disease / Condition | Reason For Inclusion |
|---|---|
| Malaria | Core project disease |
| Typhoid fever | Core project disease |
| Tuberculosis | Core project disease |
| HIV | Core project disease |
| Flu | Common respiratory illness |
| Common cold | Common respiratory illness |
| Gastroenteritis | Common gastrointestinal condition |
| COVID-like illness | Common respiratory/infectious condition |
| Dengue | Important mosquito-borne disease |
| Cholera | Important diarrheal disease |
| Pneumonia | Common respiratory infection |
| Meningitis | Serious emergency condition |
| Hepatitis B | Infectious liver disease |
| Measles | Vaccine-preventable infectious disease |

---

## 5. System Architecture

The application follows a client-server architecture.

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
  |-- Text Cleaning
  |-- Entity Extraction
  |-- Disease Prediction
  |-- Recommendation Lookup
  v
JSON Response
  |
  v
Frontend Result Display
```

### 5.1 Main Components

| Component | Role |
|---|---|
| Streamlit frontend | User interface for symptom input and results |
| FastAPI backend | REST API for analysis requests |
| NLP service | Coordinates cleaning, extraction, prediction, and recommendations |
| Disease prediction service | Loads the selected ML model and predicts disease labels |
| Recommendation service | Retrieves disease-specific guidance from JSON |
| Dataset | Source-backed symptom examples |
| Knowledge base | Medication and action guidance |

---

## 6. Dataset Description

The dataset is stored in:

```text
backend/app/data/symptoms_dataset.csv
```

It contains 616 rows and 14 disease labels, derived from 28 CDC/WHO page captures. Each row records the symptom text, the target label, its `provenance` (`source`, `source_combination` or `paraphrase`), a `derived_from` pointer for paraphrases, a `group_id` that keeps near-duplicates together during splitting, the canonical `symptom_terms`, and the source name, URL, section, verbatim quote and fetch timestamp. The dataset builder refuses to emit a source-backed row unless every one of its terms appears verbatim in that row's own quote.

### 6.1 Dataset Columns

| Column | Description |
|---|---|
| `text` | Symptom phrase used for training |
| `disease` | Target disease label |
| `source_name` | Public health source name |
| `source_url` | Source URL |
| `source_note` | Explanation of how the row was derived |

### 6.2 Data Sources

The dataset was curated from official public health references, mainly CDC and WHO.

Examples of sources:

- CDC Malaria
- WHO Malaria
- WHO Typhoid
- CDC Tuberculosis
- WHO Tuberculosis
- WHO HIV/AIDS
- CDC Influenza
- CDC Common Cold
- CDC COVID-19
- WHO Dengue
- WHO Cholera
- CDC Pneumonia
- CDC Meningitis
- WHO Hepatitis B
- CDC Measles

The source registry is stored in:

```text
backend/app/data/dataset_sources.json
```

### 6.3 Dataset Limitation

The dataset is source-backed but not clinical patient data. It is suitable for academic prototyping but not for clinical validation.

---

## 7. NLP Pipeline

The NLP pipeline transforms raw symptom text into structured features and predictions.

```text
Raw symptom text
  -> text cleaning
  -> phrase normalization
  -> entity extraction
  -> TF-IDF vectorization
  -> disease classification
  -> recommendation lookup
```

### 7.1 Text Cleaning

Text cleaning is implemented in:

```text
backend/app/utils/text_cleaning.py
```

The cleaning process includes:

- Unicode normalization.
- Lowercasing.
- Medical phrase normalization.
- Punctuation removal.
- Whitespace normalization.
- Basic stop-word removal.

Examples of normalization:

| Original expression | Normalized expression |
|---|---|
| diarrhoea | diarrhea |
| photophobia | light sensitivity |
| coryza | runny nose |
| yellow eyes | jaundice |
| difficulty breathing | shortness breath |
| stomach pain | abdominal pain |

### 7.2 Entity Extraction

Entity extraction is implemented in:

```text
backend/app/services/nlp_service.py
```

The application uses a rule-based entity extractor. It relies on dictionaries of known symptoms, disease names, and medication aliases.

Extracted entity types:

- Symptoms
- Diseases
- Medications
- Dosage mentions

Example input:

```text
I have high fever, severe headache, pain behind the eyes, joint pain and rash.
```

Extracted symptoms:

```text
fever, headache, pain behind eyes, muscle pain, rash
```

### 7.3 Negation Handling

The system includes simple negation handling. If a user writes `no fever`, the system tries not to extract fever as a positive symptom.

Example:

```text
I have cough and chest pain but no fever.
```

Expected extraction:

```text
cough, chest pain
```

### 7.4 Dosage Extraction

Dosage mentions are detected using regular expressions.

Examples:

- `500mg`
- `2 tablets`
- `once daily`
- `twice daily`
- `every 8 hours`

---

## 8. Machine Learning Model

The disease prediction model is trained in:

```text
backend/scripts/train_model.py
```

The model uses TF-IDF vectorization with unigrams, bigrams, and trigrams.

```python
TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True)
```

### 8.1 Candidate Models

The training script compares three models:

| Model | Description |
|---|---|
| Logistic Regression | Linear classifier suitable for text classification |
| Calibrated Linear SVC | Linear SVM with probability calibration |
| Complement Naive Bayes | Naive Bayes variant effective for text classification |

### 8.2 Model Results

Sixty pipelines were fit on the training split and ranked on the validation
split. Best configuration per model family, on validation:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.895 | 0.869 |
| RBF SVC | 0.871 | 0.857 |
| Calibrated Linear SVC | 0.855 | 0.844 |
| Complement Naive Bayes | 0.774 | 0.725 |

Best model:

```text
Logistic Regression, C=4.0, TF-IDF unigrams with sublinear term frequency
```

The test split was then opened once, by that model alone:

| Measure | Value | 95% interval |
|---|---:|---|
| Test accuracy | 0.894 | 0.828 - 0.937 |
| Test macro F1 | 0.888 | - |
| Grouped 5-fold CV accuracy | 0.861 | 0.797 - 0.924 |
| External scenarios (n=14) | 1.000 | 0.785 - 1.000 |
| Cross-source transfer recall | 0.524 | - |

An earlier version of this report gave 0.906 for Complement Naive Bayes. That
figure was produced by ranking 72 pipelines on the same 32-row split that was
then reported, and the shipped model was refit on that split, so the number
described neither a held-out estimate nor the shipped artifact. It is withdrawn.

Cross-source transfer is the figure to quote when asked whether the model
generalises. It holds out an entire CDC/WHO passage from training and tests on
it, and at 0.524 it shows that a large part of the apparent accuracy comes from
recognising how one source page words things. Five of the fourteen labels are
backed by a single passage and cannot be transfer-tested at all.

### 8.3 Evaluation Visualizations

The training script also generates visual evaluation artifacts for the report and presentation:

| Artifact | Purpose |
|---|---|
| `models/classical/model_comparison.png` | Compares candidate models using accuracy and macro F1-score |
| `models/classical/prediction_outcomes.png` | Shows TP, FP, TN, and FN for each model using one-vs-rest aggregation |
| `models/classical/overall_metrics.png` | Shows final best model accuracy and macro F1-score |

The final report uses overall metrics only to keep evaluation simple for presentation.

### 8.4 Optional Advanced Sequence Models

An optional advanced training script is included in:

```text
backend/scripts/train_deep_learning_model.py
```

This script trains bidirectional LSTM and GRU sequence classifiers using tokenized symptom text, an embedding layer, dropout regularization, and early stopping against the validation split. It is not the default production model: the dataset is still small at 616 rows, all of it derived from 28 source pages, and the classical model outperforms it on the same held-out test split. It is included as an academic comparison.

Advanced artifacts are saved in:

```text
models/advanced/
```

Examples:

- `models/advanced/sequence_model.keras`
- `models/advanced/tokenizer.joblib`
- `models/advanced/label_encoder.joblib`
- `models/advanced/deep_learning_metrics.txt`
- `models/advanced/deep_learning_metadata.json`
- `models/advanced/training_history.png`
- `models/advanced/advanced_overall_metrics.png`

Only the selected model is saved. Previously three `.keras` files were committed,
of which `best_sequence_model.keras` was a byte-for-byte copy of
`lstm_model.keras` and `gru_model.keras` was a losing candidate nothing loaded.

The best advanced model in the latest run was `bigru_64`, with a test accuracy of 0.846 and a macro F1 of 0.836, and 11 of 14 external scenarios correct. An earlier run reported 0.844 for an LSTM, but that run passed the test set to `fit()` as `validation_data` while `EarlyStopping(restore_best_weights=True)` was active, so the test set was selecting the model's weights; that figure is withdrawn. The conclusion stands on the corrected numbers: classical TF-IDF is retained for the main application because it is more appropriate for a small source-backed dataset and, measured properly, it wins.

### 8.5 Saved Artifacts

The trained model and metadata are saved in:

```text
models/classical/trained_model.joblib
models/classical/vectorizer.joblib
models/classical/metrics.txt
models/classical/model_metadata.json
models/classical/model_comparison.png
models/classical/overall_metrics.png
```

---

## 9. Symptom-Profile Fallback

In addition to ML prediction, the project includes transparent fallback rules for strong symptom patterns.

Examples:

| Symptom Pattern | Disease |
|---|---|
| fever + chills + headache | malaria |
| watery diarrhea + dehydration | cholera |
| stiff neck + headache + light sensitivity | meningitis |
| jaundice + dark urine + abdominal pain | hepatitis B |

This fallback improves reliability for clear symptom combinations and makes decisions easier to explain during demonstration.

---

## 10. Recommendation System

The recommendation knowledge base is stored in:

```text
backend/app/data/medication_knowledge_base.json
```

The recommendation service is implemented in:

```text
backend/app/services/recommendation_service.py
```

For each disease, the knowledge base contains:

- Recommended clinical actions
- Medication or treatment information
- Standard dosage information when safe
- Administration notes
- Warnings

The system avoids dangerous medical advice and repeatedly emphasizes professional validation.

Example for dengue:

- Seek medical assessment.
- Drink fluids.
- Monitor bleeding or severe abdominal pain.
- Avoid aspirin and ibuprofen unless a clinician says otherwise.

---

## 11. Web Application

### 11.1 Backend

The backend is implemented using FastAPI.

Main endpoint:

```text
POST /analyze
```

The backend receives:

```json
{
  "text": "I have fever and chills."
}
```

It returns:

- cleaned text
- extracted entities
- predicted disease
- confidence score
- selected model information
- recommended actions
- medication information
- disclaimer

### 11.2 Frontend

The frontend is implemented with Streamlit in:

```text
frontend/streamlit_app.py
```

It provides:

- symptom text area
- demo scenario selector
- backend URL configuration
- analysis button
- result dashboard
- safety warnings

---

## 12. Testing

Testing scenarios are documented in:

```text
tests/use_case_tests.txt
```

Example test cases:

| Input Summary | Expected Prediction |
|---|---|
| fever, chills, headache, sweating | malaria |
| high fever, pain behind eyes, rash | dengue |
| watery diarrhea, dehydration | cholera |
| stiff neck, headache, photophobia | meningitis |
| jaundice, dark urine, abdominal pain | hepatitis B |

---

## 13. Screenshots And Diagrams To Include

The final report should include screenshots such as:

- Home page of the Streamlit app.
- Symptom input example.
- Prediction result for malaria.
- Prediction result for dengue.
- Medication safety information section.
- FastAPI Swagger documentation page.
- Model comparison chart from `models/classical/model_comparison.png`.
- TP/FP/TN/FN prediction chart from `models/classical/prediction_outcomes.png`.
- Overall metrics chart from `models/classical/overall_metrics.png`.

Suggested diagrams:

- System architecture diagram.
- NLP pipeline diagram.
- Dataset structure diagram.
- Model comparison table/bar chart.
- Optional LSTM/GRU training history chart.

---

## 14. Discussion

The project successfully demonstrates a complete NLP pipeline integrated into a working web application. It uses source-backed data, transparent preprocessing, rule-based entity extraction, machine learning classification, and a safety-focused recommendation knowledge base.

The use of TF-IDF and classical ML models is appropriate for this project because the dataset is small and the goal is explainability. The system is easy to run locally and easy to demonstrate.

---

## 15. Limitations

The project has the following limitations:

- The dataset is not real patient data.
- The system is not clinically validated.
- The model only predicts known labels.
- Similar diseases can share symptoms.
- The system cannot replace laboratory tests.
- Medication information is educational and not a prescription.
- The knowledge base should be reviewed by clinicians before real-world use.

---

## 16. Future Work

Future improvements include:

- Add more validated medical datasets.
- Add multilingual support for English and French.
- Train a medical NER model with spaCy.
- Fine-tune DistilBERT or BioClinicalBERT.
- Add top-3 disease predictions.
- Add explainability for predictions.
- Add doctor/pharmacist validation.
- Deploy the application online.
- Add Docker support.

---

## 17. Conclusion

This project implements a functional NLP-based medical prescription management prototype. It demonstrates how natural language symptoms can be cleaned, structured, classified, and connected to safe educational guidance. The system includes a FastAPI backend, Streamlit frontend, source-backed dataset, trained ML model, entity extraction, and recommendation knowledge base.

Although the application is not clinically validated, it provides a strong academic demonstration of NLP, machine learning, and web development applied to medical symptom analysis.

---

## References

- CDC Malaria: https://www.cdc.gov/malaria/about/index.html
- WHO Malaria: https://www.who.int/news-room/fact-sheets/detail/malaria
- WHO Typhoid: https://www.who.int/news-room/fact-sheets/detail/typhoid
- CDC Typhoid: https://www.cdc.gov/typhoid-fever/signs-symptoms/index.html
- CDC Tuberculosis: https://www.cdc.gov/tb/signs-symptoms/index.html
- WHO Tuberculosis: https://www.who.int/news-room/fact-sheets/detail/tuberculosis
- WHO HIV/AIDS: https://www.who.int/news-room/fact-sheets/detail/hiv-aids
- CDC Influenza: https://www.cdc.gov/flu/signs-symptoms/index.html
- CDC Common Cold: https://www.cdc.gov/common-cold/about/index.html
- CDC COVID-19: https://www.cdc.gov/covid/signs-symptoms/index.html
- WHO Dengue: https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue
- WHO Cholera: https://www.who.int/news-room/fact-sheets/detail/cholera
- CDC Pneumonia: https://www.cdc.gov/pneumonia/about/index.html
- CDC Meningitis: https://www.cdc.gov/meningitis/about/index.html
- WHO Hepatitis B: https://www.who.int/news-room/fact-sheets/detail/hepatitis-b
- CDC Measles: https://www.cdc.gov/measles/signs-symptoms/index.html
