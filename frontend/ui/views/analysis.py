"""Analysis: one note in, one attributed answer out."""

from __future__ import annotations

import streamlit as st

from .. import api, evaluation_data
from ..components import (
    esc,
    fmt3,
    html,
    result_body,
    section,
    title_case,
)

SCENARIOS: dict[str, str] = {
    "Malaria": "I have fever, headache, chills, sweating and body pain for 3 days.",
    "Typhoid fever": "I have prolonged high fever, fatigue, headache, nausea and abdominal pain.",
    "Tuberculosis": "I have had a cough for more than three weeks, night sweats, fever and weight loss.",
    "Dengue": "I have high fever, severe headache, pain behind the eyes, joint pain and rash.",
    "Cholera": "I have severe watery diarrhea, vomiting, thirst and signs of dehydration after unsafe water.",
    "Pneumonia": "I have cough, fever, chills, chest pain when breathing and shortness of breath.",
    "Meningitis": "I have fever, severe headache, stiff neck, vomiting and light sensitivity.",
    "Hepatitis B": "I have yellow eyes, dark urine, nausea, vomiting and abdominal pain.",
    "Measles": "My child has high fever, cough, runny nose, red watery eyes and a rash spreading from the face.",
    "COVID-like illness": "I have fever, dry cough, shortness of breath and loss of taste.",
    "Vague note (rule engine answers)": (
        "hmm okay so uh there is like a fever thing and also headache maybe I dunno it comes and "
        "goes randomly sometimes not really sure honestly"
    ),
    "Out of scope (system abstains)": "recurring fevers that come every two days",
}

PRIMARY_ENGINE_KEYS = ("classical", "advanced")


def _selectable_engines(engines_payload: dict) -> list[dict]:
    engines = [
        engine
        for engine in engines_payload.get("engines", [])
        if engine.get("key") in PRIMARY_ENGINE_KEYS and engine.get("is_available", True)
    ]
    engines.sort(key=lambda engine: (not engine.get("is_default"), engine.get("key", "")))
    return engines


def _engine_label(engine: dict) -> str:
    metrics = engine.get("metrics") or {}
    accuracy = metrics.get("accuracy")
    accuracy_text = f"accuracy {float(accuracy):.3f}" if accuracy is not None else "not evaluated"
    role = "default" if engine.get("is_default") else "optional"
    return f"{engine.get('name')} — {engine.get('key')} · {accuracy_text} · {role}"


def _policy_card(engines_payload: dict) -> None:
    floor = engines_payload.get("model_confidence_floor")
    policy = engines_payload.get("ensemble_policy", "")
    html(
        '<div class="card">'
        '<span class="label">Arbitration policy</span>'
        f'<p class="prose" style="margin-bottom:0.8rem"><span class="mono" '
        f'style="font-size:0.78rem;color:var(--accent)">{esc(policy)}</span></p>'
        '<ol class="action-list">'
        "<li>The selected statistical engine runs. So does the rule engine, always, so its "
        "opinion is on the record even when it is not used.</li>"
        f"<li>If the statistical engine's probability reaches <strong>{fmt3(floor)}</strong>, "
        "it decides.</li>"
        "<li>Below that floor the model is treated as not knowing, and the rule engine decides "
        "instead — as a fallback, never as an override.</li>"
        "<li>If neither has an answer, the system returns <strong>unknown</strong> rather than "
        "guessing.</li>"
        "</ol></div>"
    )


def _generalisation_card() -> None:
    """The 0.524, standing, on the screen where notes are actually entered.

    A caveat that only appears next to a result is a caveat someone can miss by
    not scrolling. This one is on the entry screen, before anything is asked of
    the model.
    """
    pooled = evaluation_data.pooled_transfer_recall()
    if pooled is None:
        return
    html(
        '<div class="card" style="margin-top:0.7rem">'
        '<span class="label">Before you trust a result</span>'
        f'<p class="prose">Cross-source transfer recall is <strong>{fmt3(pooled)}</strong>. '
        "Hold an entire CDC or WHO source passage out of training, test on it, and the model "
        "recovers about half of what it should — cholera almost none of it. It is "
        "substantially learning one source page's phrasing rather than the condition.</p>"
        '<p class="quiet" style="margin-top:0.5rem">Held-out accuracy is the ceiling; this is '
        "the floor. The full evaluation, per label, is on the Model evaluation tab.</p></div>"
    )


def _running_card(placeholder, note: str) -> None:
    placeholder.markdown(
        "\n".join(
            line.lstrip()
            for line in f"""
            <div class="progress">
              <div class="progress-head">
                <span class="t">Analysing note</span>
                <span class="sub">{len(note)} characters</span>
              </div>
              <div class="sweep"><i></i></div>
              <div class="pipeline">
                <span>Normalise text</span><span>Extract entities</span>
                <span>Run every engine</span><span>Arbitrate</span><span>Record consultation</span>
              </div>
            </div>
            """.splitlines()
        ),
        unsafe_allow_html=True,
    )


def render(engines_payload: dict) -> None:
    engines = _selectable_engines(engines_payload)
    if not engines:
        st.warning("No statistical engine is available. The rule engine alone cannot be selected.")
        return

    left, right = st.columns([1.5, 1])

    with left:
        section("Patient note", "free text, as the patient describes it")
        scenario_names = ["Blank"] + list(SCENARIOS)
        scenario = st.selectbox(
            "Scenario library", scenario_names, key="scenario", label_visibility="visible"
        )
        note = st.text_area(
            "Symptom description",
            value="" if scenario == "Blank" else SCENARIOS[scenario],
            height=190,
            key=f"note::{scenario}",
            placeholder=(
                "Example: I have high fever, severe headache, pain behind the eyes and joint pain."
            ),
        )

        controls_left, controls_right = st.columns([1.35, 1])
        with controls_left:
            engine = st.selectbox(
                "Primary engine",
                engines,
                format_func=_engine_label,
                key="engine_choice",
            )
        with controls_right:
            patient_reference = st.text_input(
                "Case label (optional)", key="patient_reference", placeholder="e.g. ward-b-14"
            )
        st.caption(
            "A case label is stored verbatim as written. It is a local reference, not a place "
            "for identifying patient data."
        )
        run = st.button("Run analysis", type="primary", use_container_width=True)

    with right:
        section("How this decides", "published policy")
        _policy_card(engines_payload)
        _generalisation_card()

    progress = st.empty()

    if run:
        if len(note.strip()) < 3:
            st.warning("Enter a symptom description of at least three characters.")
        else:
            _running_card(progress, note)
            try:
                result = api.analyze(
                    note,
                    engine_key=engine["key"],
                    session_id=st.session_state.session_id,
                    patient_reference=patient_reference,
                )
            except api.ServiceUnavailable:
                progress.empty()
                st.error(
                    "The analysis service went away mid-request. Nothing was recorded. "
                    "Your note is still in the box — try again."
                )
                return
            except api.ServiceError as exc:
                progress.empty()
                st.error(str(exc))
                return

            progress.empty()
            st.session_state.last_result = result
            st.session_state.session_log = [result] + st.session_state.get("session_log", [])[:49]
            _render_result(result, animate=True)
            return

    if st.session_state.get("last_result"):
        _render_result(st.session_state["last_result"], animate=False)


def _render_result(result: dict, *, animate: bool) -> None:
    consultation_id = result.get("consultation_id")
    section(
        "Result",
        f"consultation {consultation_id[:8]}" if consultation_id else "not recorded",
    )
    result_body(result, animate=animate)

    if consultation_id:
        _record_actions(result, consultation_id)


def _record_actions(result: dict, consultation_id: str) -> None:
    section("Record", "every suggestion leaves an audit trail")
    left, right = st.columns([1, 1])

    with left:
        try:
            report = api.report(consultation_id)
            st.download_button(
                "Download Markdown report",
                data=report.get("content", ""),
                file_name=f"consultation-{consultation_id[:8]}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        except (api.ServiceUnavailable, api.ServiceError):
            st.caption("The report for this consultation is not available right now.")

    with right:
        st.caption(
            f"Consultation `{consultation_id}` · session "
            f"`{result.get('session_id', '')[:8]}` · recorded "
            f"{result.get('created_at', '')[:19].replace('T', ' ')}"
        )

    with st.expander("Audit trail"):
        try:
            events = api.audit_trail(consultation_id).get("events", [])
        except (api.ServiceUnavailable, api.ServiceError):
            st.caption("The audit trail is not available right now.")
            return
        rows = [
            '<div class="row audit-row head"><span>Recorded</span><span>Event</span><span>Engine</span>'
            "<span>Its accuracy</span><span>Outcome</span><span>Suggested</span></div>"
        ]
        for event in events:
            rows.append(
                '<div class="row audit-row">'
                f'<span class="t">{esc(str(event.get("created_at", ""))[:19].replace("T", " "))}</span>'
                f'<span class="mono" style="font-size:0.75rem">{esc(event.get("event_type", ""))}</span>'
                f'<span class="eng">{esc(event.get("engine_key", ""))}</span>'
                f'<span class="c">{fmt3(event.get("engine_accuracy"))}</span>'
                f'<span class="dx">{esc(title_case(event.get("predicted_disease")))}</span>'
                f'<span class="note-text">{esc(", ".join(event.get("suggested_medications", [])) or "none")}</span>'
                "</div>"
            )
        html(f'<div class="dtable">{"".join(rows)}</div>')
