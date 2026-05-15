# Team Presentation Plan

## Presentation Duration

Recommended duration: 10 to 15 minutes.

## Slide 1 - Title

Title:

```text
NLP-Based Medical Prescription Management System
```

Content:

- Team members
- Academic year
- Project duration
- Technologies: Python, FastAPI, Streamlit, scikit-learn

Speaker notes:

```text
Today we present our NLP-based medical prescription management system. The goal is to analyze symptoms written in natural language and provide educational medical guidance with safety warnings.
```

## Slide 2 - Problem Context

Content:

- Prescription misuse can be dangerous.
- Patients often describe symptoms informally.
- Access to reliable information can be limited.
- Common diseases need early awareness and professional validation.

Speaker notes:

```text
The main problem is that symptoms are usually written in unstructured language. NLP can help transform this text into structured information that supports preliminary understanding.
```

## Slide 3 - Objectives

Content:

- Analyze symptom text.
- Extract symptoms, diseases, medications, and dosage mentions.
- Predict possible disease.
- Recommend safe educational actions.
- Build a web application.

Speaker notes:

```text
Our objective was not to replace doctors, but to build an educational prototype that demonstrates NLP and machine learning applied to medical symptom analysis.
```

## Slide 4 - Supported Diseases

Content:

Core diseases:

- Malaria
- Typhoid fever
- Tuberculosis
- HIV

Expanded diseases:

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

Speaker notes:

```text
The original project focused on four main diseases. We expanded the system to 14 labels to make the application more complete and realistic for demonstration.
```

## Slide 5 - System Architecture

Content diagram:

```text
User -> Streamlit Frontend -> FastAPI Backend -> NLP Service -> ML Model -> Knowledge Base -> Result
```

Speaker notes:

```text
The user interacts with the Streamlit frontend. The frontend sends symptoms to the FastAPI backend. The backend runs the NLP pipeline, predicts the disease, retrieves recommendations, and returns the result.
```

## Slide 6 - Dataset

Content:

- Dataset file: `symptoms_dataset.csv`
- 128 rows
- 14 labels
- Source-backed from CDC and WHO
- Not real patient data

Speaker notes:

```text
The dataset was curated from official CDC and WHO symptom descriptions. Each row includes a symptom phrase, disease label, source name, source URL, and note.
```

## Slide 7 - NLP Pipeline

Content diagram:

```text
Raw text
 -> Cleaning
 -> Phrase normalization
 -> Entity extraction
 -> TF-IDF vectorization
 -> Classification
 -> Recommendation lookup
```

Speaker notes:

```text
The NLP pipeline first cleans the text and normalizes medical expressions. Then it extracts entities using dictionaries and converts the text into TF-IDF features for classification.
```

## Slide 8 - Entity Extraction

Content:

Extracted entities:

- Symptoms
- Diseases
- Medications
- Dosage mentions

Example:

```text
Input: high fever, headache, pain behind the eyes, rash
Extracted: fever, headache, pain behind eyes, rash
```

Speaker notes:

```text
We used a rule-based entity extractor because it is transparent and easy to explain. It also includes simple negation handling, for example no fever.
```

## Slide 9 - Machine Learning Model

Content:

Models compared:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 0.812 | 0.776 |
| Calibrated Linear SVC | 0.844 | 0.837 |
| Complement Naive Bayes | 0.844 | 0.826 |

Best model:

```text
Calibrated Linear SVC
```

Speaker notes:

```text
We used TF-IDF because it is lightweight and explainable. The best validation result was obtained with calibrated Linear SVC.
```

## Slide 10 - Recommendation System

Content:

- Knowledge base file: `medication_knowledge_base.json`
- Maps predicted disease to actions and medication information
- Includes warnings
- Avoids unsafe prescription advice

Speaker notes:

```text
The recommendation system is safety-first. It gives educational guidance and always reminds the user to consult a qualified healthcare professional.
```

## Slide 11 - Web Application Demo

Demo inputs:

```text
I have fever, headache, chills, sweating and body pain for 3 days.
```

Expected:

```text
malaria
```

Second input:

```text
I have high fever, severe headache, pain behind the eyes, joint pain and rash.
```

Expected:

```text
dengue
```

Speaker notes:

```text
Now we demonstrate the application. We enter symptoms in natural language and the system returns extracted entities, possible disease, actions, and safety information.
```

## Slide 12 - Testing And Results

Content:

- Tested 14 disease scenarios.
- Representative examples are stored in `use_case_tests.txt`.
- Best model accuracy: 0.844.
- Best macro F1: 0.837.

Speaker notes:

```text
We tested the application using prepared use cases for each disease. The results show that the system can correctly identify strong symptom patterns.
```

## Slide 13 - Challenges

Content:

- Medical symptoms overlap across diseases.
- Real clinical data is sensitive and difficult to access.
- Medication advice must be safe.
- Small datasets can limit model performance.
- UI integration with backend required validation.

Speaker notes:

```text
One of the main challenges was safety. We avoided direct diagnosis and prescription wording, and we clearly show medical disclaimers.
```

## Slide 14 - Limitations And Future Work

Content:

Limitations:

- Not clinically validated
- Not real patient data
- Limited to 14 labels
- Cannot replace medical tests

Future work:

- Add validated datasets
- Add multilingual support
- Add spaCy NER
- Add BERT/DistilBERT
- Add top-3 predictions
- Deploy online

Speaker notes:

```text
The current system is a strong academic MVP. For real-world usage, it would require medical expert validation and larger clinical datasets.
```

## Slide 15 - Conclusion

Content:

- Functional NLP web application
- Source-backed dataset
- Entity extraction and disease prediction
- Safety-first recommendation system
- FastAPI and Streamlit integration

Speaker notes:

```text
In conclusion, the project demonstrates a complete NLP pipeline integrated into a functional web application for educational symptom analysis and medical guidance.
```

## Recommended Live Demo Order

1. Malaria
2. Dengue
3. Cholera
4. Meningitis

## Final Message To Say

```text
This project is not a replacement for doctors. It is an academic demonstration of how NLP and machine learning can structure symptom text and provide safe educational guidance.
```
