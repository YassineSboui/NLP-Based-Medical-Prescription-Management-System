# Teacher Q&A Preparation

## 1. What is the main goal of this project?

The goal is to build a web application that uses NLP to analyze patient-reported symptoms, extract medical entities, predict a possible disease, and return educational treatment information with safety warnings.

## 2. Is this a real diagnosis system?

No. It is an academic prototype. It does not replace a doctor, laboratory tests, or a pharmacist. It only gives an educational symptom-based suggestion.

## 3. Why did you choose these diseases?

The original project asked for malaria, typhoid fever, tuberculosis, and HIV. I expanded the scope to include flu, common cold, gastroenteritis, COVID-like illness, dengue, cholera, pneumonia, meningitis, hepatitis B, and measles because they are common infectious or symptom-relevant conditions and have official CDC/WHO symptom references.

## 4. How many diseases are supported?

The current version supports 14 labels.

## 5. Where did the dataset come from?

The dataset is curated from official CDC and WHO public health pages. Each row includes the source name, URL, and a note explaining how the symptom phrase was derived.

## 6. Is the dataset real patient data?

No. It is not real patient data. It is a source-backed educational dataset built from official symptom descriptions. This avoids privacy issues but also means the model is not clinically validated.

## 7. Why not use real hospital data?

Real hospital data contains sensitive personal health information. It requires ethical approval, anonymization, legal permissions, and medical supervision. For an academic MVP, a source-backed public dataset is safer and more realistic.

## 7.1 Did you implement web scraping?

Yes, a controlled scraping script was added in `backend/scripts/scrape_medical_sources.py`. It reads the official CDC/WHO source registry, fetches each page, extracts relevant symptom and treatment sections using BeautifulSoup, and saves the extracted raw content as JSON and CSV. However, the final training dataset is still curated manually from these reliable sources because medical data must be reviewed carefully before being used for modeling.

## 8. What NLP techniques are used?

The project uses text normalization, phrase normalization, stop-word removal, rule-based entity extraction, TF-IDF vectorization, and supervised classification.

## 9. What machine learning models did you use?

The training script compares Logistic Regression, calibrated Linear SVC, and Complement Naive Bayes. The best model is selected based on validation performance.

## 10. What is the current best model?

The current best model is calibrated Linear SVC.

## 11. Why did calibrated Linear SVC perform well?

Linear SVC is strong for text classification because TF-IDF creates sparse high-dimensional vectors. Calibration allows it to provide probability-like confidence scores.

## 12. What is TF-IDF?

TF-IDF means Term Frequency-Inverse Document Frequency. It converts text into numerical vectors by giving more importance to words that are frequent in a document but not common across all documents.

## 13. Why did you use TF-IDF instead of word embeddings?

TF-IDF is simple, fast, transparent, and effective for small datasets. It is easier to explain in an academic presentation. Embeddings or BERT require more data and more computation.

## 14. Why did you not use BERT?

BERT is powerful but heavy. For a small source-backed dataset, fine-tuning BERT may overfit and complicate the project. The current solution is more reliable for a 5-week MVP.

## 15. What is entity extraction?

Entity extraction identifies important medical terms in the user's text, such as symptoms, diseases, medication names, and dosage mentions.

## 16. Is the entity extractor trained?

No. It is rule-based. It uses dictionaries of symptoms, diseases, and medications. This makes it transparent and easy to extend.

## 17. Why use rule-based NER instead of spaCy NER?

Rule-based extraction is simpler, explainable, and reliable for the MVP. A trained spaCy NER model would require annotated medical text, which was not available.

## 18. What preprocessing is done?

The text is lowercased, normalized, cleaned of punctuation, phrase-normalized, tokenized by splitting, and filtered with basic stop-word removal.

## 19. What are examples of phrase normalization?

Examples include `diarrhoea` to `diarrhea`, `photophobia` to `light sensitivity`, `coryza` to `runny nose`, and `yellow eyes` to `jaundice`.

## 20. Does the system handle negation?

It has simple negation handling. For example, if the user says `no fever`, the system tries not to extract fever as a positive symptom.

## 21. What is the recommendation system?

The recommendation system maps the predicted disease to educational actions and medication information stored in `medication_knowledge_base.json`.

## 22. Does the app prescribe medication?

No. It gives educational medication information and warnings. It always says that treatment must be validated by a qualified clinician.

## 23. Why are dosages not exact prescriptions?

Dosage depends on age, weight, pregnancy status, allergies, liver/kidney function, severity, and local medical guidelines. Giving exact prescriptions would be unsafe.

## 24. How does the frontend communicate with the backend?

The Streamlit frontend sends a POST request to the FastAPI `/analyze` endpoint with the symptom text as JSON.

## 25. What does the backend return?

It returns cleaned text, extracted entities, predicted disease, confidence score, recommended actions, medication information, and a disclaimer.

## 26. What is Pydantic used for?

Pydantic defines and validates the request and response schemas in FastAPI.

## 27. What is CORS and why did you add it?

CORS allows the frontend to call the backend from another origin or port. It is useful because Streamlit and FastAPI run separately.

## 28. What is the confidence score?

It is the model's estimated confidence for the predicted class, combined with the symptom-profile fallback in some strong cases. It should not be interpreted as clinical certainty.

## 29. What happens if the symptoms are ambiguous?

The model still chooses the closest label, but the result should be interpreted carefully. The app includes disclaimers and recommends clinical validation.

## 30. What are the limitations of this system?

The dataset is small, not clinical patient data, and not clinically validated. The model only supports known labels and cannot replace medical diagnosis.

## 31. How can the project be improved?

It can be improved by adding validated clinical datasets, multilingual support, trained medical NER, transformer models, top-3 predictions, and clinician-reviewed medication rules.

## 32. Why is medical safety important here?

Because wrong medication or diagnosis can harm patients. The app avoids final diagnosis wording and always recommends consulting a healthcare professional.

## 33. How do you test the app?

I test the backend with FastAPI Swagger and the frontend with prepared use cases in `use_case_tests.txt`. I also run `python backend/scripts/train_model.py` to verify model training.

## 34. What is the role of `use_case_tests.txt`?

It contains ready-to-use symptom scenarios, expected disease predictions, and expected extracted symptoms for demonstration and testing.

## 35. What would you say if the teacher asks if this is medically reliable?

I would say it is not medically reliable for real-world diagnosis. It is an academic prototype that demonstrates the NLP pipeline. Clinical reliability would require validated datasets, medical review, and real clinical evaluation.
