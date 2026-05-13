# Dataset Description

## Source

The current MVP uses a small source-backed curated dataset stored in `backend/app/data/symptoms_dataset.csv`. The rows are derived from public symptom descriptions published by CDC and WHO. Each row includes source metadata so the dataset can be cited in the technical report.

This is not a downloaded clinical patient dataset. It is a curated educational dataset built from official public health symptom pages. The symptoms are source-backed, but the examples are still not real patient records and are not clinically validated for diagnostic performance.

The source registry is stored in `backend/app/data/dataset_sources.json`.

## Schema

- `text`: Symptom description in natural language.
- `disease`: Target disease label.
- `source_name`: Name of the public source used for the row.
- `source_url`: URL of the source.
- `source_note`: Short explanation of how the row was derived from the source.

## Labels

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

## Preprocessing

- Unicode normalization.
- Lowercasing.
- Punctuation removal.
- Whitespace normalization.
- Basic stop-word removal.
- Phrase normalization for common medical aliases such as diarrhoea/diarrhea, coryza/runny nose, photophobia/light sensitivity, and jaundice/yellow eyes.
- Basic negation-aware entity extraction for phrases such as `no fever` or `denies cough`.
- TF-IDF vectorization with unigrams, bigrams, and trigrams.
- Validation-based model selection across Logistic Regression, calibrated Linear SVC, and Complement Naive Bayes.

## Source References

- CDC Malaria: `https://www.cdc.gov/malaria/about/index.html`
- WHO Malaria: `https://www.who.int/news-room/fact-sheets/detail/malaria`
- WHO Typhoid: `https://www.who.int/news-room/fact-sheets/detail/typhoid`
- CDC Typhoid: `https://www.cdc.gov/typhoid-fever/signs-symptoms/index.html`
- CDC Tuberculosis: `https://www.cdc.gov/tb/signs-symptoms/index.html`
- WHO Tuberculosis: `https://www.who.int/news-room/fact-sheets/detail/tuberculosis`
- WHO HIV/AIDS: `https://www.who.int/news-room/fact-sheets/detail/hiv-aids`
- CDC HIV: `https://www.cdc.gov/hiv/about/index.html`
- CDC Influenza: `https://www.cdc.gov/flu/signs-symptoms/index.html`
- CDC Common Cold: `https://www.cdc.gov/common-cold/about/index.html`
- CDC Norovirus/Gastroenteritis: `https://www.cdc.gov/norovirus/about/index.html`
- WHO Diarrhoeal Disease: `https://www.who.int/news-room/fact-sheets/detail/diarrhoeal-disease`
- CDC COVID-19: `https://www.cdc.gov/covid/signs-symptoms/index.html`
- WHO Dengue: `https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue`
- WHO Cholera: `https://www.who.int/news-room/fact-sheets/detail/cholera`
- CDC Pneumonia: `https://www.cdc.gov/pneumonia/about/index.html`
- CDC Meningitis: `https://www.cdc.gov/meningitis/about/index.html`
- WHO Hepatitis B: `https://www.who.int/news-room/fact-sheets/detail/hepatitis-b`
- CDC Measles: `https://www.cdc.gov/measles/signs-symptoms/index.html`

## Validation Warning

The dataset is source-backed but not clinically validated. It should be presented as an academic curated dataset, not as a medical diagnostic dataset. Model metrics from this dataset are useful for checking the software pipeline, not for proving medical accuracy.

For a stronger final project, the dataset should be expanded with medically reviewed records or validated public corpora, then reviewed by a clinician or pharmacist before any real-world use.
