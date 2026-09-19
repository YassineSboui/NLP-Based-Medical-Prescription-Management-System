"""Batch: many notes in one pass, with per-note success and failure."""

from __future__ import annotations

import csv
import io

import streamlit as st

from .. import api
from ..components import (
    esc,
    fmt3,
    html,
    result_body,
    section,
    short,
    stat_strip,
    title_case,
)

SAMPLE = """I have fever, headache, chills, sweating and body pain for 3 days.
I have had a cough for more than three weeks, night sweats, fever and weight loss.
My child has high fever, cough, runny nose, red watery eyes and a rash spreading from the face.
I have severe watery diarrhea, vomiting, thirst and dehydration after drinking unsafe water.
recurring fevers that come every two days
ok"""


def _rows_to_csv(results: list[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "index",
            "ok",
            "note",
            "predicted_disease",
            "confidence",
            "abstained",
            "decided_by_engine",
            "engine_name",
            "engine_accuracy",
            "policy_reason",
            "consultation_id",
            "error",
        ]
    )
    for item in results:
        result = item.get("result") or {}
        decision = result.get("decision", {})
        engine = decision.get("decided_by", {}) or {}
        writer.writerow(
            [
                item.get("index"),
                item.get("ok"),
                result.get("original_text", ""),
                result.get("predicted_disease", ""),
                result.get("confidence", ""),
                decision.get("abstained", ""),
                engine.get("key", ""),
                engine.get("name", ""),
                (engine.get("metrics") or {}).get("accuracy", ""),
                decision.get("policy_reason", ""),
                result.get("consultation_id", "") or "",
                item.get("error") or "",
            ]
        )
    return buffer.getvalue().encode("utf-8")


def render(engines_payload: dict) -> None:
    engines = [
        engine
        for engine in engines_payload.get("engines", [])
        if engine.get("key") in ("classical", "advanced") and engine.get("is_available", True)
    ]
    engines.sort(key=lambda engine: not engine.get("is_default"))

    left, right = st.columns([1.5, 1])

    with left:
        section("Notes", "one note per line")
        text = st.text_area(
            "Notes to analyse",
            value=SAMPLE,
            height=208,
            key="batch_text",
            label_visibility="collapsed",
        )
        notes = [line.strip() for line in text.splitlines() if line.strip()]
        controls_left, controls_right = st.columns([1.4, 1])
        with controls_left:
            engine = st.selectbox(
                "Primary engine",
                engines,
                format_func=lambda item: f"{item.get('name')} — {item.get('key')}",
                key="batch_engine",
            )
        with controls_right:
            persist = st.selectbox(
                "Record consultations", ["Yes", "No"], key="batch_persist"
            )
        run = st.button(
            f"Analyse {len(notes)} note{'s' if len(notes) != 1 else ''}",
            type="primary",
            use_container_width=True,
            disabled=not notes,
        )

    with right:
        section("How a batch behaves", "per-item outcomes")
        html(
            '<div class="card"><ol class="action-list">'
            "<li>Every note is analysed independently under the same arbitration policy.</li>"
            "<li>A note that fails validation reports its own error. It does not fail the "
            "batch, so a day's worth of notes still returns partial results.</li>"
            "<li>Recorded notes join the consultation store and the audit trail exactly as "
            "single analyses do.</li>"
            "</ol></div>"
        )
        st.caption(
            "The sample includes one note that is deliberately too short, so the per-item "
            "failure path is visible rather than hypothetical."
        )

    progress = st.empty()

    if run and notes:
        progress.markdown(
            "\n".join(
                line.lstrip()
                for line in f"""
                <div class="progress">
                  <div class="progress-head">
                    <span class="t">Analysing {len(notes)} notes</span>
                    <span class="sub">{esc(engine.get('key', ''))}</span>
                  </div>
                  <div class="sweep"><i></i></div>
                </div>
                """.splitlines()
            ),
            unsafe_allow_html=True,
        )
        try:
            batch = api.analyze_batch(
                notes,
                engine_key=engine["key"],
                session_id=st.session_state.session_id,
                persist=persist == "Yes",
            )
        except api.ServiceUnavailable:
            progress.empty()
            st.error("The analysis service went away mid-batch. Nothing further was recorded.")
            return
        except api.ServiceError as exc:
            progress.empty()
            st.error(str(exc))
            return
        progress.empty()
        st.session_state.last_batch = batch

    batch = st.session_state.get("last_batch")
    if not batch:
        return

    results = batch.get("results", [])
    engine_counts: dict[str, int] = {}
    abstentions = 0
    for item in results:
        if not item.get("ok"):
            continue
        decision = (item.get("result") or {}).get("decision", {})
        key = (decision.get("decided_by") or {}).get("key", "?")
        engine_counts[key] = engine_counts.get(key, 0) + 1
        abstentions += bool(decision.get("abstained"))

    section("Outcome", f"batch {batch.get('batch_id', '')[:8]}")
    stat_strip(
        [
            ("Submitted", str(batch.get("submitted", 0)), "notes in this batch"),
            ("Succeeded", str(batch.get("succeeded", 0)), "analysed and attributed"),
            ("Failed", str(batch.get("failed", 0)), "rejected with a reason"),
            (
                "Abstained",
                str(abstentions),
                "returned unknown rather than guess",
            ),
        ],
        columns=4,
        flag_index=3 if abstentions else -1,
    )

    rows = [
        '<div class="row batch-row head"><span>#</span><span>Note</span><span>Outcome</span>'
        "<span>Confidence</span><span>Decided by</span></div>"
    ]
    for item in results:
        index = item.get("index", 0) + 1
        if not item.get("ok"):
            source_note = notes[item["index"]] if item.get("index", 0) < len(notes) else ""
            rows.append(
                '<div class="row batch-row">'
                f'<span class="c">{index}</span>'
                f'<span><span class="note-text">{esc(short(source_note, 110))}</span>'
                f'<span class="row-error">{esc(item.get("error", ""))}</span></span>'
                '<span class="dx abstain">Rejected</span>'
                '<span class="c">—</span>'
                '<span class="eng">—</span>'
                "</div>"
            )
            continue
        result = item.get("result") or {}
        decision = result.get("decision", {})
        abstained = bool(decision.get("abstained"))
        rows.append(
            '<div class="row batch-row">'
            f'<span class="c">{index}</span>'
            f'<span class="note-text">{esc(short(result.get("original_text"), 110))}</span>'
            f'<span class="dx{" abstain" if abstained else ""}">'
            f'{esc(title_case(result.get("predicted_disease")))}</span>'
            f'<span class="c">{fmt3(result.get("confidence"))}</span>'
            f'<span class="eng">{esc((decision.get("decided_by") or {}).get("key", ""))}</span>'
            "</div>"
        )
    html(f'<div class="dtable">{"".join(rows)}</div>')

    footer_left, footer_right = st.columns([1.6, 1])
    with footer_left:
        attribution = ", ".join(f"{key} × {count}" for key, count in sorted(engine_counts.items()))
        st.caption(f"Decided by: {attribution or 'nothing succeeded'}")
    with footer_right:
        st.download_button(
            "Export batch results (CSV)",
            data=_rows_to_csv(results),
            file_name=f"batch-{batch.get('batch_id', 'results')[:8]}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    succeeded = [item for item in results if item.get("ok")]
    if not succeeded:
        return

    section("Open one result", "full attribution for a single note")
    labels = {
        f"{item['index'] + 1}. {title_case((item['result'] or {}).get('predicted_disease'))} "
        f"· {short((item['result'] or {}).get('original_text'), 52)}": item["index"]
        for item in succeeded
    }
    chosen = st.selectbox("Note", list(labels), key="batch_selected")
    selected = next(item for item in succeeded if item["index"] == labels[chosen])
    result_body(selected["result"], animate=False)
