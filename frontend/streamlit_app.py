import html
import os
from textwrap import dedent
from urllib.parse import urlparse

import requests
import streamlit as st
import streamlit.components.v1 as components


DEFAULT_API_URL = "http://localhost:8000/analyze"


def get_default_api_url() -> str:
    configured_url = os.getenv("MEDICAL_NLP_API_URL") or os.getenv("API_URL") or DEFAULT_API_URL
    parsed = urlparse(configured_url)
    if parsed.hostname in {"localhost", "127.0.0.1"} and parsed.path.endswith("/analyze"):
        return configured_url
    return DEFAULT_API_URL


EXAMPLES = {
    "Malaria": "I have fever, headache, chills, sweating and body pain for 3 days.",
    "Typhoid fever": "I have prolonged high fever, fatigue, headache, nausea and abdominal pain.",
    "Tuberculosis": "I have had a cough for more than three weeks, night sweats, fever and weight loss.",
    "Dengue": "I have high fever, severe headache, pain behind the eyes, joint pain and rash.",
    "Cholera": "I have severe watery diarrhea, vomiting, thirst and signs of dehydration after unsafe water.",
    "Pneumonia": "I have cough, fever, chills, chest pain when breathing and shortness of breath.",
    "Meningitis": "I have fever, severe headache, stiff neck, vomiting and light sensitivity.",
    "Hepatitis B": "I have yellow eyes, dark urine, nausea, vomiting and abdominal pain.",
    "Measles": "My child has high fever, cough, runny nose, red watery eyes and a rash spreading from the face.",
    "COVID-like": "I have fever, dry cough, shortness of breath and loss of taste.",
}

REQUIRED_RESPONSE_FIELDS = {
    "cleaned_text",
    "extracted_entities",
    "predicted_disease",
    "confidence",
    "recommended_actions",
    "recommended_medicines",
    "disclaimer",
}


def esc(value: object) -> str:
    return html.escape(str(value))


def pills(items: list[str], empty_text: str = "None detected") -> str:
    if not items:
        return f'<span class="empty-text">{esc(empty_text)}</span>'
    return "".join(f'<span class="pill">{esc(item)}</span>' for item in items)


def validate_api_response(payload: dict) -> tuple[bool, str]:
    missing = sorted(REQUIRED_RESPONSE_FIELDS - set(payload.keys()))
    if missing:
        return False, f"The API response is not from this project. Missing fields: {', '.join(missing)}"
    return True, ""


def render_results(result: dict) -> None:
    entities = result["extracted_entities"]
    confidence = float(result["confidence"])
    confidence_pct = int(round(confidence * 100))

    actions_html = "".join(f"<li>{esc(action)}</li>" for action in result["recommended_actions"])

    meds_html = ""
    for medication in result["recommended_medicines"]:
        warnings_html = "".join(f"<li>{esc(warning)}</li>" for warning in medication["warnings"])
        meds_html += dedent(
            f"""
            <article class="medication-card">
              <div class="medication-header">
                <span class="med-tag">Medication Info</span>
                <h4>{esc(medication['name'])}</h4>
              </div>
              <div class="medication-grid">
                <div>
                  <span class="micro-label">Standard dosage / information</span>
                  <p>{esc(medication['standard_dosage'])}</p>
                </div>
                <div>
                  <span class="micro-label">Administration</span>
                  <p>{esc(medication['administration'])}</p>
                </div>
              </div>
              <div class="warning-strip">
                <span class="micro-label">Warnings</span>
                <ul>{warnings_html}</ul>
              </div>
            </article>
            """
        ).strip()

    if not meds_html:
        meds_html = '<div class="empty-panel">No medication information available for this prediction.</div>'

    result_styles = dedent(
        """
        <style>
          :root {
            --navy: #062b3b;
            --ink: #113946;
            --muted: #5a7580;
            --blue: #0077b6;
            --teal: #00a896;
            --cyan: #48cae4;
            --line: #bfe8e1;
            --danger: #b42318;
          }
          * {
            box-sizing: border-box;
          }
          body {
            margin: 0;
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--ink);
            background: transparent;
          }
          .results-shell {
            padding: 0.2rem 0 1rem 0;
          }
          .results-title-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: end;
            margin: 0.4rem 0 0.8rem 0;
          }
          .eyebrow,
          .micro-label {
            display: inline-block;
            color: var(--teal);
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-weight: 900;
            font-size: 0.72rem;
          }
          .results-title-row h2 {
            margin: 0.1rem 0 0 0;
            color: var(--navy);
            font-size: 2rem;
            letter-spacing: -0.04em;
          }
          .confidence-badge {
            padding: 0.7rem 0.95rem;
            border-radius: 999px;
            background: var(--navy);
            color: white;
            font-weight: 900;
            box-shadow: 0 14px 32px rgba(6, 43, 59, 0.18);
            white-space: nowrap;
          }
          .result-grid {
            display: grid;
            gap: 1rem;
          }
          .no-code-grid,
          .detail-grid,
          .medication-grid {
            grid-template-columns: 1fr 1fr;
          }
          .detail-grid {
            margin-top: 1rem;
          }
          .result-card,
          .medication-card,
          .disclaimer-box,
          .empty-panel {
            border-radius: 28px;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(0, 119, 182, 0.12);
            box-shadow: 0 18px 48px rgba(0, 95, 115, 0.08);
            padding: 1.25rem;
          }
          .result-card h3,
          .medication-card h4 {
            color: var(--navy);
            margin: 0.45rem 0 0.5rem 0;
            font-size: 1.45rem;
            letter-spacing: -0.03em;
          }
          .diagnosis-card {
            background: linear-gradient(145deg, #ffffff, #e9fffa);
          }
          .diagnosis-card h3 {
            font-size: 2rem;
          }
          .confidence-track {
            height: 11px;
            border-radius: 999px;
            background: #d9f4ef;
            overflow: hidden;
            margin: 0.85rem 0;
          }
          .confidence-track div {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--blue), var(--teal));
          }
          .subtle,
          .empty-text {
            color: var(--muted);
          }
          .red-card {
            background: linear-gradient(145deg, #fff5f5, #ffffff);
            border-color: #f1b8b4;
          }
          .red-card p {
            color: #6c1c16;
            font-weight: 750;
            line-height: 1.58;
          }
          .entity-block {
            margin-top: 1rem;
          }
          .pill {
            display: inline-block;
            margin: 0.22rem 0.18rem 0 0;
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            color: #005f73;
            font-weight: 800;
            background: #dffaf4;
            border: 1px solid #a5e4db;
          }
          .action-list,
          .warning-strip ul {
            margin: 0.7rem 0 0 1.1rem;
            padding: 0;
          }
          .action-list li,
          .warning-strip li {
            margin: 0.55rem 0;
            color: var(--ink);
            line-height: 1.5;
          }
          .medication-section {
            margin-top: 1.1rem;
          }
          .section-title {
            color: var(--navy);
            font-weight: 900;
            font-size: 1.45rem;
            letter-spacing: -0.03em;
            margin: 0 0 0.75rem 0;
          }
          .medication-card {
            margin-bottom: 1rem;
          }
          .med-tag {
            display: inline-block;
            padding: 0.32rem 0.58rem;
            border-radius: 999px;
            background: #e7f7ff;
            color: var(--blue);
            font-weight: 900;
            font-size: 0.72rem;
          }
          .medication-grid {
            display: grid;
            gap: 1rem;
          }
          .medication-grid p {
            color: var(--ink);
            line-height: 1.55;
          }
          .warning-strip {
            margin-top: 0.7rem;
            padding: 0.9rem 1rem;
            border-radius: 20px;
            background: #fff8e6;
            border: 1px solid #f0d48b;
          }
          .disclaimer-box {
            margin-top: 1rem;
            background: #fff1f0;
            border-color: #f0b8b4;
            color: #6c1c16;
            font-weight: 800;
            line-height: 1.55;
          }
          @media (max-width: 900px) {
            .results-title-row {
              align-items: start;
              flex-direction: column;
            }
            .no-code-grid,
            .detail-grid,
            .medication-grid {
              grid-template-columns: 1fr;
            }
          }
        </style>
        """
    ).strip()

    results_html = dedent(
        f"""
        {result_styles}
        <section class="results-shell">
          <div class="results-title-row">
            <div>
              <span class="eyebrow">Analysis Complete</span>
              <h2>Clinical NLP Result</h2>
            </div>
            <div class="confidence-badge">{confidence_pct}% confidence</div>
          </div>

          <div class="result-grid top-grid no-code-grid">
            <article class="result-card diagnosis-card">
              <span class="micro-label">Possible disease</span>
              <h3>{esc(result['predicted_disease']).title()}</h3>
              <div class="confidence-track"><div style="width: {confidence_pct}%"></div></div>
              <p class="subtle">This is a symptom-based educational suggestion, not a diagnosis.</p>
            </article>

            <article class="result-card red-card">
              <span class="micro-label">Safety reminder</span>
              <p>Do not self-medicate. Confirm diagnosis, tests, medicines, and dosages with a qualified clinician.</p>
            </article>
          </div>

          <div class="result-grid detail-grid">
            <article class="result-card entities-card">
              <h3>Extracted Medical Entities</h3>
              <div class="entity-block">
                <span class="micro-label">Symptoms</span>
                <div>{pills(entities['symptoms'])}</div>
              </div>
              <div class="entity-block">
                <span class="micro-label">Diseases mentioned</span>
                <div>{pills(entities['diseases'], 'No disease names mentioned')}</div>
              </div>
              <div class="entity-block">
                <span class="micro-label">Medications mentioned</span>
                <div>{pills(entities['medications'], 'No medication names mentioned')}</div>
              </div>
              <div class="entity-block">
                <span class="micro-label">Dosage mentions</span>
                <div>{pills(entities['dosage_mentions'], 'No dosage mentions detected')}</div>
              </div>
            </article>

            <article class="result-card actions-card">
              <h3>Recommended Actions</h3>
              <ul class="action-list">{actions_html}</ul>
            </article>
          </div>

          <div class="medication-section">
            <div class="section-title">Medication And Safety Information</div>
            {meds_html}
          </div>

          <div class="disclaimer-box">{esc(result['disclaimer'])}</div>
        </section>
        """
    ).strip()

    component_height = 1150 + (len(result["recommended_medicines"]) * 360)
    components.html(results_html, height=component_height, scrolling=True)


st.set_page_config(page_title="Medical Prescription NLP", layout="wide")

st.markdown(
    dedent(
        """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

      :root {
        --navy: #062b3b;
        --ink: #113946;
        --muted: #5a7580;
        --blue: #0077b6;
        --teal: #00a896;
        --cyan: #48cae4;
        --mint: #dffaf4;
        --soft: #f5fffc;
        --line: #bfe8e1;
        --danger: #b42318;
        --warning: #b7791f;
        --shadow: 0 24px 70px rgba(0, 95, 115, 0.16);
      }

      html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }

      .stApp {
        background:
          radial-gradient(circle at 8% 5%, rgba(72, 202, 228, 0.24), transparent 30rem),
          radial-gradient(circle at 90% 20%, rgba(0, 168, 150, 0.16), transparent 24rem),
          linear-gradient(135deg, #eefdf9 0%, #f7fbff 48%, #ffffff 100%);
      }

      .block-container {
        max-width: 1280px;
        padding: 3.4rem 2.4rem 2.8rem 2.4rem;
      }

      .main p,
      .main li,
      .main label,
      .main span,
      .main div,
      .main h1,
      .main h2,
      .main h3,
      .main h4,
      .main strong {
        color: var(--ink) !important;
      }

      section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #062b3b 0%, #005f73 100%);
      }

      section[data-testid="stSidebar"] * {
        color: #effffd !important;
      }

      section[data-testid="stSidebar"] input,
      section[data-testid="stSidebar"] textarea {
        color: #0b2530 !important;
        -webkit-text-fill-color: #0b2530 !important;
        background: #ffffff !important;
      }

      .hero {
        position: relative;
        display: grid;
        grid-template-columns: minmax(0, 1.24fr) minmax(280px, 0.76fr);
        gap: 1.5rem;
        align-items: stretch;
        padding: 1.55rem;
        border-radius: 34px;
        background: linear-gradient(135deg, #05384b 0%, #007b83 48%, #22c1c3 100%);
        box-shadow: var(--shadow);
        overflow: hidden;
        margin-bottom: 1.35rem;
      }

      .hero:before {
        content: '';
        position: absolute;
        width: 22rem;
        height: 22rem;
        right: -7rem;
        top: -9rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.13);
      }

      .hero-copy {
        position: relative;
        z-index: 1;
        padding: 1rem 0.7rem 0.75rem 0.7rem;
      }

      .hero-copy .kicker {
        display: inline-flex;
        gap: 0.45rem;
        align-items: center;
        padding: 0.42rem 0.68rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.25);
        color: #ffffff !important;
        font-weight: 800;
        font-size: 0.84rem;
        letter-spacing: 0.02em;
      }

      .hero-copy h1 {
        margin: 1rem 0 0.65rem 0;
        color: #ffffff !important;
        font-size: clamp(2rem, 4.5vw, 3.45rem);
        line-height: 0.98;
        letter-spacing: -0.055em;
      }

      .hero-copy p {
        max-width: 680px;
        color: #e9fffc !important;
        font-size: 1.05rem;
        line-height: 1.65;
        margin: 0;
      }

      .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1.15rem;
      }

      .badge {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 0.84rem;
        padding: 0.48rem 0.72rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.22);
      }

      .hero-panel {
        position: relative;
        z-index: 1;
        min-height: 220px;
        border-radius: 28px;
        background:
          linear-gradient(145deg, rgba(255,255,255,0.18), rgba(255,255,255,0.08));
        border: 1px solid rgba(255, 255, 255, 0.22);
        overflow: hidden;
      }

      .orb {
        position: absolute;
        width: 106px;
        height: 106px;
        top: 28px;
        right: 30px;
        border-radius: 30px;
        display: grid;
        place-items: center;
        background: #ffffff;
        box-shadow: 0 20px 35px rgba(0,0,0,0.16);
      }

      .orb:before,
      .orb:after {
        content: '';
        position: absolute;
        background: var(--teal);
        border-radius: 999px;
      }

      .orb:before { width: 58px; height: 16px; }
      .orb:after { width: 16px; height: 58px; }

      .ecg {
        position: absolute;
        left: 30px;
        right: 30px;
        bottom: 34px;
      }

      .ecg path {
        stroke-dasharray: 560;
        stroke-dashoffset: 560;
        animation: ecg 2.8s ease-in-out infinite;
      }

      @keyframes ecg {
        0% { stroke-dashoffset: 560; opacity: 0.45; }
        50% { stroke-dashoffset: 0; opacity: 1; }
        100% { stroke-dashoffset: -560; opacity: 0.45; }
      }

      .input-zone {
        display: grid;
        grid-template-columns: minmax(0, 1.55fr) minmax(280px, 0.85fr);
        gap: 1rem;
        align-items: stretch;
        margin: 1.1rem 0 1.35rem 0;
      }

      .helper-card {
        height: 100%;
        padding: 1.3rem;
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.68);
        border: 1px solid rgba(0, 119, 182, 0.12);
        box-shadow: 0 18px 48px rgba(0, 95, 115, 0.08);
      }

      .helper-card h3 {
        margin: 0 0 0.55rem 0;
        color: var(--navy) !important;
        font-size: 1.2rem;
      }

      .helper-card p {
        color: var(--muted) !important;
        line-height: 1.55;
      }

      .step-row {
        display: grid;
        grid-template-columns: 34px 1fr;
        gap: 0.7rem;
        align-items: start;
        padding: 0.55rem 0;
      }

      .step-dot {
        width: 30px;
        height: 30px;
        border-radius: 10px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, var(--blue), var(--teal));
        color: white !important;
        font-weight: 900;
      }

      .step-row span:last-child {
        color: var(--ink) !important;
        font-weight: 650;
      }

      .input-title {
        color: var(--navy) !important;
        font-size: 1.28rem;
        font-weight: 900;
        margin: 0 0 0.35rem 0;
      }

      .input-subtitle {
        color: var(--muted) !important;
        margin-bottom: 0.75rem;
      }

      div[data-testid="stTextArea"] textarea {
        min-height: 190px !important;
        border-radius: 26px !important;
        border: 1px solid #9edfd7 !important;
        background: rgba(255,255,255,0.86) !important;
        color: #0b2530 !important;
        -webkit-text-fill-color: #0b2530 !important;
        box-shadow: 0 18px 48px rgba(0, 95, 115, 0.10) !important;
        padding: 1.2rem !important;
        font-size: 1rem !important;
      }

      div[data-testid="stTextArea"] label,
      div[data-testid="stTextInput"] label {
        color: var(--ink) !important;
        font-weight: 800 !important;
      }

      div.stButton > button:first-child {
        height: 3.25rem;
        border: 0;
        border-radius: 18px;
        color: white !important;
        font-weight: 900;
        letter-spacing: 0.01em;
        background: linear-gradient(135deg, #0077b6 0%, #00a896 100%);
        box-shadow: 0 18px 34px rgba(0, 119, 182, 0.24);
      }

      div.stButton > button:first-child:hover {
        border: 0;
        filter: brightness(1.04);
      }

      .safety-ribbon {
        border-radius: 22px;
        padding: 1rem 1.1rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #fff8e7, #ffffff);
        border: 1px solid #f1d492;
        color: #6a4300 !important;
        font-weight: 750;
        box-shadow: 0 14px 36px rgba(183, 121, 31, 0.08);
      }

      .loading-card {
        border-radius: 28px;
        padding: 1.2rem 1.35rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(226,255,249,0.88));
        border: 1px solid var(--line);
        box-shadow: 0 18px 48px rgba(0, 95, 115, 0.10);
        margin: 1rem 0;
      }

      .loading-card strong {
        color: var(--navy) !important;
        font-size: 1.02rem;
      }

      .loading-card p {
        color: var(--muted) !important;
        margin: 0.2rem 0 0.7rem 0;
      }

      .loading-line {
        height: 10px;
        border-radius: 99px;
        overflow: hidden;
        background: #d5f4ef;
      }

      .loading-line div {
        height: 100%;
        width: 36%;
        border-radius: 99px;
        background: linear-gradient(90deg, var(--blue), var(--teal), var(--cyan));
        animation: slide 1.05s infinite ease-in-out;
      }

      @keyframes slide {
        0% { transform: translateX(-105%); }
        100% { transform: translateX(300%); }
      }

      .results-shell {
        margin-top: 1.35rem;
      }

      .results-title-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: end;
        margin: 1.2rem 0 0.8rem 0;
      }

      .eyebrow,
      .micro-label {
        display: inline-block;
        color: var(--teal) !important;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-weight: 900;
        font-size: 0.72rem;
      }

      .results-title-row h2 {
        margin: 0.1rem 0 0 0;
        color: var(--navy) !important;
        font-size: 2rem;
        letter-spacing: -0.04em;
      }

      .confidence-badge {
        padding: 0.7rem 0.95rem;
        border-radius: 999px;
        background: var(--navy);
        color: white !important;
        font-weight: 900;
        box-shadow: 0 14px 32px rgba(6, 43, 59, 0.18);
      }

      .result-grid {
        display: grid;
        gap: 1rem;
      }

      .top-grid {
        grid-template-columns: 1fr 1.15fr 0.95fr;
      }

      .no-code-grid {
        grid-template-columns: 1fr 1fr;
      }

      .detail-grid {
        grid-template-columns: 1fr 1.15fr;
        margin-top: 1rem;
      }

      .result-card,
      .medication-card,
      .disclaimer-box,
      .empty-panel {
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(0, 119, 182, 0.12);
        box-shadow: 0 18px 48px rgba(0, 95, 115, 0.08);
        padding: 1.25rem;
      }

      .native-card {
        min-height: 100%;
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(0, 119, 182, 0.12);
        box-shadow: 0 18px 48px rgba(0, 95, 115, 0.08);
        padding: 1.25rem;
        margin: 0.55rem 0 1rem 0;
      }

      .native-card,
      .native-card * {
        color: var(--ink) !important;
      }

      .native-card h3,
      .native-card h4 {
        color: var(--navy) !important;
        margin-top: 0.55rem;
      }

      .danger-native {
        background: linear-gradient(145deg, #fff5f5, #ffffff);
        border-color: #f1b8b4;
      }

      .danger-native p,
      .danger-native div {
        color: #6c1c16 !important;
        font-weight: 750;
      }

      .tall-card {
        min-height: 330px;
      }

      .medication-native {
        margin-top: 0.6rem;
      }

      .spaced-label {
        margin-top: 1rem;
      }

      .native-card div[data-testid="stCodeBlock"] pre {
        background: #062b3b !important;
        color: #dffaf4 !important;
        border-radius: 18px !important;
      }

      .native-card div[data-testid="stCodeBlock"] code,
      .native-card div[data-testid="stCodeBlock"] code * {
        color: #dffaf4 !important;
      }

      .result-card h3,
      .medication-card h4 {
        color: var(--navy) !important;
        margin: 0.45rem 0 0.5rem 0;
        font-size: 1.45rem;
        letter-spacing: -0.03em;
      }

      .diagnosis-card {
        background: linear-gradient(145deg, #ffffff, #e9fffa);
      }

      .diagnosis-card h3 {
        font-size: 2rem;
      }

      .confidence-track {
        height: 11px;
        border-radius: 999px;
        background: #d9f4ef;
        overflow: hidden;
        margin: 0.85rem 0;
      }

      .confidence-track div {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--blue), var(--teal));
      }

      .subtle,
      .empty-text {
        color: var(--muted) !important;
      }

      .code-card code {
        display: block;
        margin-top: 0.55rem;
        padding: 1rem;
        border-radius: 18px;
        background: #062b3b;
        color: #dffaf4 !important;
        line-height: 1.55;
        white-space: pre-wrap;
      }

      .red-card {
        background: linear-gradient(145deg, #fff5f5, #ffffff);
        border-color: #f1b8b4;
      }

      .red-card p {
        color: #6c1c16 !important;
        font-weight: 750;
        line-height: 1.58;
      }

      .entity-block {
        margin-top: 1rem;
      }

      .pill {
        display: inline-block;
        margin: 0.22rem 0.18rem 0 0;
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        color: #005f73 !important;
        font-weight: 800;
        background: #dffaf4;
        border: 1px solid #a5e4db;
      }

      .action-list,
      .warning-strip ul {
        margin: 0.7rem 0 0 1.1rem;
        padding: 0;
      }

      .action-list li,
      .warning-strip li {
        margin: 0.55rem 0;
        color: var(--ink) !important;
        line-height: 1.5;
      }

      .medication-section {
        margin-top: 1.1rem;
      }

      .section-title {
        color: var(--navy) !important;
        font-weight: 900;
        font-size: 1.45rem;
        letter-spacing: -0.03em;
        margin: 0 0 0.75rem 0;
      }

      .medication-card {
        margin-bottom: 1rem;
      }

      .med-tag {
        display: inline-block;
        padding: 0.32rem 0.58rem;
        border-radius: 999px;
        background: #e7f7ff;
        color: var(--blue) !important;
        font-weight: 900;
        font-size: 0.72rem;
      }

      .medication-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
      }

      .medication-grid p {
        color: var(--ink) !important;
        line-height: 1.55;
      }

      .warning-strip {
        margin-top: 0.7rem;
        padding: 0.9rem 1rem;
        border-radius: 20px;
        background: #fff8e6;
        border: 1px solid #f0d48b;
      }

      .disclaimer-box {
        margin-top: 1rem;
        background: #fff1f0;
        border-color: #f0b8b4;
        color: #6c1c16 !important;
        font-weight: 800;
        line-height: 1.55;
      }

      div[data-testid="stAlert"] {
        border-radius: 18px;
      }

      div[data-testid="stAlert"] * {
        color: #102a36 !important;
      }

      @media (max-width: 980px) {
        .block-container { padding: 2.2rem 1rem 2rem 1rem; }
        .hero,
        .input-zone,
        .top-grid,
        .detail-grid,
        .medication-grid {
          grid-template-columns: 1fr;
        }
        .hero-panel { min-height: 150px; }
        .results-title-row { align-items: start; flex-direction: column; }
      }
    </style>
    """
    ).strip(),
    unsafe_allow_html=True,
)

st.markdown(
    dedent(
        """
    <section class="hero">
      <div class="hero-copy">
        <span class="kicker">Medical NLP prototype</span>
        <h1>Prescription Intelligence For Symptom Triage</h1>
        <p>Enter patient-reported symptoms, extract medical entities, predict a likely condition, and display safe educational medication guidance with clear warnings.</p>
        <div class="hero-badges">
          <span class="badge">14 disease labels</span>
          <span class="badge">CDC / WHO sourced</span>
          <span class="badge">FastAPI backend</span>
          <span class="badge">Streamlit interface</span>
        </div>
      </div>
      <div class="hero-panel" aria-hidden="true">
        <div class="orb"></div>
        <div class="ecg">
          <svg viewBox="0 0 520 120" width="100%" height="120">
            <path d="M6 67 H92 L116 67 L142 22 L174 104 L205 67 H285 L312 42 L342 86 L372 67 H514"
              fill="none" stroke="white" stroke-width="9" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </div>
      </div>
    </section>
    <div class="safety-ribbon">Medical safety: academic demonstration only. This application is not a diagnosis tool, not a prescription tool, and must not be used for self-medication.</div>
    """
    ).strip(),
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Demo Center")
    selected_example = st.selectbox("Choose a scenario", ["Custom"] + list(EXAMPLES.keys()))
    st.caption("Pick a scenario, then click Analyze Symptoms.")
    st.divider()
    st.subheader("Backend URL")
    api_url = st.text_input("FastAPI analyze endpoint", value=get_default_api_url())
    st.caption("Use localhost unless your backend is deployed elsewhere.")
    if api_url != DEFAULT_API_URL and "webhooks.fivetran.com" in api_url:
        st.error("This URL points to Fivetran, not your FastAPI backend. Use http://localhost:8000/analyze")
    st.divider()
    st.subheader("Scope")
    st.write("14 source-backed disease labels")
    st.write("CDC/WHO symptom references")
    st.write("TF-IDF model selection")

default_text = "" if selected_example == "Custom" else EXAMPLES[selected_example]

st.markdown('<div class="input-zone">', unsafe_allow_html=True)
input_col, guide_col = st.columns([1.55, 0.85])

with input_col:
    st.markdown('<div class="input-title">Patient Scenario Description</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-subtitle">Write symptoms naturally, as a patient would describe them.</div>', unsafe_allow_html=True)
    symptoms_text = st.text_area(
        "Describe symptoms in natural language",
        value=default_text,
        height=190,
        placeholder="Example: I have high fever, severe headache, pain behind the eyes and joint pain.",
        label_visibility="collapsed",
    )
    analyze = st.button("Analyze Symptoms", type="primary", use_container_width=True)

with guide_col:
    st.markdown(
        dedent(
            """
        <aside class="helper-card">
          <h3>Demo Flow</h3>
          <p>Use this panel for a clean classroom demonstration.</p>
          <div class="step-row"><span class="step-dot">1</span><span>Start the FastAPI backend.</span></div>
          <div class="step-row"><span class="step-dot">2</span><span>Select or type a symptom scenario.</span></div>
          <div class="step-row"><span class="step-dot">3</span><span>Run the analysis.</span></div>
          <div class="step-row"><span class="step-dot">4</span><span>Explain entities, prediction, and safety guidance.</span></div>
        </aside>
        """
        ).strip(),
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)

if analyze:
    if len(symptoms_text.strip()) < 3:
        st.warning("Please enter a symptom description before analysis.")
    elif not api_url.strip().endswith("/analyze"):
        st.error("The backend URL must point to the FastAPI /analyze endpoint, for example http://localhost:8000/analyze")
    else:
        try:
            loading_placeholder = st.empty()
            loading_placeholder.markdown(
                dedent(
                    """
                <div class="loading-card">
                  <strong>Analyzing symptoms...</strong>
                  <p>Cleaning text, extracting entities, running the model, and preparing safety guidance.</p>
                  <div class="loading-line"><div></div></div>
                </div>
                """
                ).strip(),
                unsafe_allow_html=True,
            )
            try:
                response = requests.post(api_url.strip(), json={"text": symptoms_text}, timeout=20)
                response.raise_for_status()
                result = response.json()
            finally:
                loading_placeholder.empty()
        except requests.RequestException as exc:
            st.error(f"Could not reach the FastAPI backend. Start it with: cd backend && python -m uvicorn app.main:app --reload. Details: {exc}")
        except ValueError:
            st.error("The backend returned a non-JSON response. Check that the URL points to FastAPI, not another service.")
        else:
            valid, message = validate_api_response(result)
            if not valid:
                st.error(message)
                st.code(result)
            else:
                render_results(result)

st.caption(
    "Academic MVP: source-backed curated dataset, rule-based entity extraction, classical NLP model selection, and safety-first prescription knowledge base."
)
