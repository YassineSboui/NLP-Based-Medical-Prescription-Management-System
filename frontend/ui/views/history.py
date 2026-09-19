"""History: every consultation the service recorded, filterable and exportable."""

from __future__ import annotations

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
    when,
)

ALL = "All"


def _summary_row(item: dict) -> str:
    abstained = bool(item.get("abstained"))
    return (
        '<div class="row history-row">'
        f'<span class="t">{esc(when(item.get("created_at")))}</span>'
        f'<span class="note-text">{esc(short(item.get("original_text"), 110))}</span>'
        f'<span class="dx{" abstain" if abstained else ""}">'
        f'{esc(title_case(item.get("predicted_disease")))}</span>'
        f'<span class="c">{fmt3(item.get("confidence"))}</span>'
        f'<span class="eng">{esc(item.get("engine_key", ""))}</span>'
        f'<span><span class="src">{esc(item.get("source", ""))}</span></span>'
        "</div>"
    )


def render(engines_payload: dict) -> None:
    try:
        stats = api.stats()
    except api.ServiceError as exc:
        st.error(str(exc))
        return

    section("Recorded activity", "SQLite consultation store")
    abstentions = stats.get("abstentions", 0)
    total = stats.get("total_consultations", 0)
    top_disease = max(stats.get("by_disease", {}).items(), key=lambda kv: kv[1], default=("—", 0))
    engine_counts = stats.get("by_engine", {})
    rule_answers = engine_counts.get("symptom_profile", 0)

    stat_strip(
        [
            ("Consultations", str(total), f"across {stats.get('total_sessions', 0)} sessions"),
            (
                "Answered by rule engine",
                str(rule_answers),
                f"{(rule_answers / total * 100):.0f}% of all answers" if total else "no records yet",
            ),
            (
                "Abstentions",
                str(abstentions),
                f"{(abstentions / total * 100):.0f}% returned unknown" if total else "no records yet",
            ),
            ("Most frequent", title_case(top_disease[0]), f"{top_disease[1]} consultations"),
            (
                "Last recorded",
                when(stats.get("last_consultation_at")) or "—",
                "first " + (when(stats.get("first_consultation_at")) or "—"),
            ),
        ],
        columns=5,
        flag_index=1 if rule_answers else -1,
    )

    section("Browse", "filter, inspect, export")

    disease_options = [ALL] + sorted(stats.get("by_disease", {}))
    engine_options = [ALL] + sorted(
        {engine.get("key") for engine in engines_payload.get("engines", []) if engine.get("key")}
        | set(engine_counts)
    )

    filter_cols = st.columns([1, 1, 1, 1])
    with filter_cols[0]:
        disease = st.selectbox("Condition", disease_options, key="history_disease")
    with filter_cols[1]:
        engine_key = st.selectbox("Deciding engine", engine_options, key="history_engine")
    with filter_cols[2]:
        scope = st.selectbox("Scope", ["All sessions", "This session"], key="history_scope")
    with filter_cols[3]:
        limit = st.selectbox("Rows", [25, 50, 100, 200], index=1, key="history_limit")

    filters = {
        "disease": None if disease == ALL else disease,
        "engine_key": None if engine_key == ALL else engine_key,
        "session_id": st.session_state.session_id if scope == "This session" else None,
    }

    try:
        page = api.consultations(limit=limit, **filters)
    except api.ServiceError as exc:
        st.error(str(exc))
        return

    items = page.get("items", [])
    if not items:
        html(
            '<div class="card"><p class="quiet">No consultations match these filters. Run an '
            "analysis, or widen the filters.</p></div>"
        )
        return

    rows = [
        '<div class="row history-row head"><span>Recorded</span><span>Note</span>'
        "<span>Outcome</span><span>Confidence</span><span>Decided by</span><span>Source</span></div>"
    ]
    rows += [_summary_row(item) for item in items]
    html(f'<div class="dtable">{"".join(rows)}</div>')

    footer_left, footer_right = st.columns([1.6, 1])
    with footer_left:
        st.caption(
            f"Showing {len(items)} of {page.get('total', 0)} recorded consultations"
            + (f" · condition {disease}" if disease != ALL else "")
            + (f" · engine {engine_key}" if engine_key != ALL else "")
        )
    with footer_right:
        try:
            csv_bytes = api.export_csv(limit=1000, **filters)
            st.download_button(
                "Export filtered history (CSV)",
                data=csv_bytes,
                file_name="consultations.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except (api.ServiceUnavailable, api.ServiceError):
            st.caption("Export is not available right now.")

    section("Inspect a consultation", "full record, as it was written")
    labels = {
        f"{when(item.get('created_at'))} · {title_case(item.get('predicted_disease'))} "
        f"· {item.get('engine_key')} · {item.get('consultation_id', '')[:8]}": item[
            "consultation_id"
        ]
        for item in items
    }
    chosen_label = st.selectbox("Consultation", list(labels), key="history_selected")
    chosen_id = labels[chosen_label]

    try:
        detail = api.consultation(chosen_id)
    except (api.ServiceUnavailable, api.ServiceError) as exc:
        st.error(str(exc))
        return

    html(
        '<div class="card"><span class="label">Recorded note</span>'
        f'<p class="prose">{esc(detail.get("original_text", ""))}</p></div>'
    )
    result_body(_detail_as_result(detail, engines_payload), animate=False)

    try:
        report = api.report(chosen_id)
        st.download_button(
            "Download Markdown report",
            data=report.get("content", ""),
            file_name=f"consultation-{chosen_id[:8]}.md",
            mime="text/markdown",
        )
    except (api.ServiceUnavailable, api.ServiceError):
        st.caption("The report for this consultation is not available right now.")


def _detail_as_result(detail: dict, engines_payload: dict) -> dict:
    """Reshape a stored consultation into the shape a live result has.

    The stored record keeps the deciding engine's key, name and metrics, which
    is what the attribution panel needs. Family and confidence meaning are not
    stored per record; they are looked up from the live engine registry by key,
    so the panel never describes one engine with another engine's words.
    """
    metrics = detail.get("engine_metrics", {}) or {}
    engine_key = detail.get("engine_key", "")
    registry = {
        engine.get("key"): engine for engine in engines_payload.get("engines", []) if engine.get("key")
    }
    described = registry.get(engine_key, {})
    return {
        "predicted_disease": detail.get("predicted_disease"),
        "confidence": detail.get("confidence"),
        "cleaned_text": detail.get("cleaned_text", ""),
        "extracted_entities": detail.get("extracted_entities", {}),
        "engine_opinions": detail.get("engine_opinions", []),
        "recommended_actions": detail.get("recommended_actions", []),
        "recommended_medicines": detail.get("recommended_medicines", []),
        "disclaimer": detail.get("disclaimer", ""),
        "decision": {
            "predicted_disease": detail.get("predicted_disease"),
            "confidence": detail.get("confidence"),
            "abstained": detail.get("abstained", False),
            "policy": detail.get("policy", ""),
            "policy_reason": detail.get("policy_reason", ""),
            "decided_by": {
                "key": engine_key,
                "name": detail.get("engine_name", ""),
                "family": described.get("family", ""),
                "confidence_meaning": described.get("confidence_meaning", ""),
                "metrics": metrics,
            },
        },
    }
