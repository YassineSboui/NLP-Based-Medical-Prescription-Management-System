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

The main training script compares Logistic Regression, calibrated Linear SVC, and Complement Naive Bayes. The best model is selected based on validation performance. I also added an optional advanced deep-learning script that trains bidirectional LSTM and GRU sequence models for academic comparison.

## 10. What is the current best model?

The current best model is logistic regression with C=4.0 over TF-IDF unigrams, chosen from 60 candidate pipelines ranked on a validation split. On the held-out test split it reaches 0.894 accuracy (95% CI 0.828-0.937) and 0.888 macro F1.

I should be clear about one thing if asked: an earlier version of this project reported 0.906. That number was not trustworthy. All 72 candidate pipelines were scored on the same 32-row test set that was then published as the result, which reports the maximum of 72 noisy estimates, and the shipped model was refit on that same 32 rows. The current pipeline selects on validation, reads the test set once, and ships the model that was never trained on it. The honest number is slightly lower and it means something.

## 11. Why did Complement Naive Bayes perform well?

Complement Naive Bayes often performs well on text classification with TF-IDF features, especially when the dataset is small and classes have overlapping words. In the optimized run, alpha 0.2 with TF-IDF unigrams/bigrams/trigrams gave the best validation result.

## 12. What is TF-IDF?

TF-IDF means Term Frequency-Inverse Document Frequency. It converts text into numerical vectors by giving more importance to words that are frequent in a document but not common across all documents.

## 13. Why did you use TF-IDF instead of word embeddings?

TF-IDF is simple, fast, transparent, and effective for small datasets. It is easier to explain in an academic presentation. Embeddings or BERT require more data and more computation.

## 14. Why did you not use BERT?

BERT is powerful but heavy. For a small source-backed dataset, fine-tuning BERT may overfit and complicate the project. The current solution is more reliable for a 5-week MVP.

## 14.1 Did you add an advanced model?

Yes. The file `backend/scripts/train_deep_learning_model.py` trains optional bidirectional LSTM and GRU models using token sequences, embeddings, dropout, and early stopping against the validation split. It saves the selected model, tokenizer, label encoder, metrics, and plots in `models/advanced/`. The best advanced model is `bigru_64`, with 0.846 test accuracy and 0.836 macro F1.

An earlier run reported 0.844, but it passed the test set to `fit()` as `validation_data` while early stopping restored the best weights, so the test set was choosing the weights. That figure is withdrawn. I keep TF-IDF as the main production model because the dataset is still small and, measured properly, the classical model wins on the same held-out split.

## 14.2 What evaluation visuals are included?

The main training script now generates `models/classical/overall_metrics.png`, `models/classical/model_comparison.png`, and `models/classical/prediction_outcomes.png`. These show the final overall accuracy/macro F1, compare the optimized classical model families on a zoomed scale, and show TP, FP, TN, and FN for each model using one-vs-rest aggregation.

The advanced model also has `models/advanced/advanced_overall_metrics.png` and `models/advanced/advanced_prediction_outcomes.png`. The advanced training history chart was simplified so it no longer has a long legend hiding the plot.

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

The Streamlit frontend calls `GET /models` to list available models, then sends a POST request to `/analyze` with the symptom text and selected `model_key` as JSON.

## 25. What does the backend return?

It returns cleaned text, extracted entities, predicted disease, confidence score, model used, recommended actions, medication information, and a disclaimer.

## 25.1 Can the user choose the prediction model?

Yes. The sidebar includes a model selector. The user can choose the default classical TF-IDF model or the advanced LSTM model. The backend receives this choice through `model_key`.

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

I test the backend with FastAPI Swagger and the frontend with prepared use cases in `tests/use_case_tests.txt`. I also run `python backend/scripts/train_model.py` to verify model training and generate metrics/visuals. For the optional advanced experiment, I run `python backend/scripts/train_deep_learning_model.py` after installing `backend/requirements-advanced.txt`.

## 34. What is the role of `tests/use_case_tests.txt`?

It contains ready-to-use symptom scenarios, expected disease predictions, and expected extracted symptoms for demonstration and testing.

## 35. What would you say if the teacher asks if this is medically reliable?

I would say it is not medically reliable for real-world diagnosis. It is an academic prototype that demonstrates the NLP pipeline. Clinical reliability would require validated datasets, medical review, and real clinical evaluation.

## 36. Where is the training code exactly?

The training code is in `backend/scripts/train_model.py`. The actual training happens with `pipeline.fit(train_text, train_labels)`. After selecting the best model, the script retrains it on the full dataset using `best_pipeline.fit(dataset["cleaned_text"], dataset["disease"])`, then saves it with `joblib.dump`.

## 37. Where are the evaluation images generated?

They are generated in `backend/scripts/train_model.py` using matplotlib and seaborn. The final project keeps only overall evaluation images: `models/classical/overall_metrics.png`, `models/classical/model_comparison.png`, and `models/classical/prediction_outcomes.png`.

## 38. Why do you have JSON and CSV scraped files?

JSON keeps the structured scraped data, including source metadata, sections, items, and errors. CSV is easier to inspect manually in Excel, Google Sheets, or pandas. Both are raw traceability files. The model does not train directly on the scraped files; it trains on the manually curated `backend/app/data/symptoms_dataset.csv` file.

## 39. Why is manual curation needed after scraping?

Medical web pages contain navigation text, repeated content, explanations, warnings, and non-symptom information. Manual curation ensures that only relevant, clean, source-backed symptom examples are used for training. This is safer for a medical NLP project.

## 40. What is the difference between entity extraction and disease prediction?

Entity extraction identifies terms inside the user input, such as `fever`, `cough`, `malaria`, `paracetamol`, or `500mg`. Disease prediction uses the full cleaned symptom text and the trained ML model to choose the most likely disease label.

## 41. What should you show in the presentation?

Show the Streamlit interface, one or two live symptom examples, the FastAPI Swagger page, the overall metrics chart, the model comparison chart, and the safety disclaimer. Recommended demo cases are malaria, dengue, cholera, and meningitis.

## 42. What is the short presentation structure?

Use this order: problem context, objectives, supported diseases, architecture, dataset, NLP pipeline, ML training, evaluation visuals, optional LSTM/GRU advanced model, recommendation system, live demo, limitations, and future work.

## 43. What do you say about LSTM/GRU if results are not better?

I would say that LSTM and GRU were added as optional advanced sequence-model experiments. On the held-out test split the best sequence model (`bigru_64`) reaches 0.846 accuracy and 0.836 macro F1, while the classical TF-IDF model reaches 0.894 and 0.888. Both numbers come from the same group-aware split, with selection done on validation and the test set read once, so they are directly comparable. The classical model is stronger on this dataset, so it remains the main model: it is simpler, explainable, and more appropriate for small data.

## 44. How do you run the project for demo?

First run the backend with `cd backend` then `python -m uvicorn app.main:app --reload`. Then open another terminal from the project root and run `python -m streamlit run frontend\streamlit_app.py`. The frontend opens at `http://localhost:8501`, and the backend API docs are at `http://localhost:8000/docs`.

## 45. What are good demo inputs?

For malaria: `I have fever, headache, chills, sweating and body pain for 3 days.` For dengue: `I have high fever, severe headache, pain behind the eyes, joint pain and rash.` For cholera: `I have severe watery diarrhea, vomiting, thirst and dehydration after drinking unsafe water.` For meningitis: `I have fever, severe headache, stiff neck, vomiting and light sensitivity.`

## 46. Why does the metadata show only one final model?

Because the metadata files were cleaned for final project submission. Hyperparameter and architecture variants were tested internally during training, but the final metadata shows only the selected best model, its score, and its best parameters/configuration. This keeps the project easy to validate and avoids confusing model variants such as `X_1`, `X_2`, and `X_3`.
