"""Symptom Triage Console — the frontend entry point.

Four screens behind one masthead:

    Analysis    one note, one attributed answer
    History     every consultation the service recorded, with CSV export
    Batch       many notes in one pass, per-item success and failure
    Evaluation  what this model actually does, including the parts that look bad

This file does the routing and nothing else. The markup helpers are in
``ui/components.py``, the API calls in ``ui/api.py``, the stylesheet in
``assets/app.css``, and one screen per module under ``ui/views/``.

Three things the previous version did that this one does not:

* It carried a hardcoded engine with a hardcoded accuracy as an offline
  fallback -- a figure this project has since withdrawn. An application that
  cannot reach the service does not know what the engines are, and now says so.
* It exposed the backend URL as an editable text box and instructed the end
  user, in the interface, to go and start a server. The address is
  configuration; an unreachable service is a service state.
* It carried its stylesheet inline, twice.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui import api, chrome  # noqa: E402
from ui.views import analysis, batch, evaluation, history  # noqa: E402

st.set_page_config(
    page_title="Symptom Triage Console",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"Get help": None, "Report a Bug": None, "About": None},
)

chrome.apply_page_style()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
st.session_state.setdefault("session_log", [])
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("last_batch", None)


@st.cache_data(ttl=20, show_spinner=False)
def _service_status() -> api.ServiceStatus:
    return api.status()


@st.cache_data(ttl=120, show_spinner=False)
def _engines() -> dict:
    return api.engines()


status = _service_status()
engines_payload: dict | None = None
if status.reachable:
    try:
        engines_payload = _engines()
    except (api.ServiceUnavailable, api.ServiceError):
        status = api.ServiceStatus(reachable=False, detail="The engine registry is unavailable.")

chrome.masthead(status, engines_payload)
chrome.caution_ribbon()

if not status.reachable or engines_payload is None:
    chrome.service_unavailable(status)
else:
    analysis_tab, history_tab, batch_tab, evaluation_tab = st.tabs(
        ["Analysis", "History", "Batch", "Model evaluation"]
    )

    with analysis_tab:
        analysis.render(engines_payload)
    with history_tab:
        history.render(engines_payload)
    with batch_tab:
        batch.render(engines_payload)
    with evaluation_tab:
        evaluation.render(engines_payload)
