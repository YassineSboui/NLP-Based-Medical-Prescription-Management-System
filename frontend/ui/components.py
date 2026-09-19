"""Markup helpers shared by every screen.

Streamlit's Markdown renderer treats a four-space indent as a code block, so
every block of HTML is flattened before it is handed over. That one rule is why
`html()` exists and why nothing here calls `st.markdown` directly.
"""

from __future__ import annotations

import html as _html
from typing import Iterable, Sequence

import streamlit as st

from . import evaluation_data

DASH = "—"


def esc(value: object) -> str:
    return _html.escape(str(value if value is not None else ""))


def html(markup: str) -> None:
    st.markdown("\n".join(line.lstrip() for line in markup.splitlines()), unsafe_allow_html=True)


def fmt3(value: object, dash: str = "—") -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return dash


def fmt_pct(value: object, dash: str = "—") -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return dash


def fmt_ci(interval: Sequence[float] | None) -> str:
    if not interval or len(interval) != 2:
        return ""
    return f"95% CI {float(interval[0]):.3f}–{float(interval[1]):.3f}"


def title_case(label: object) -> str:
    """Display a dataset label the way a clinician writes it.

    The labels are stored lowercase, so naive capitalisation produces
    "Hepatitis b" and "Hiv". These are the cases where that is wrong.
    """
    text = str(label or "").strip()
    if not text:
        return "Unknown"
    special = {
        "hiv": "HIV",
        "covid-like illness": "COVID-like illness",
        "hepatitis b": "Hepatitis B",
    }
    return special.get(text.lower(), text[0].upper() + text[1:])


def short(text: object, limit: int = 96) -> str:
    value = str(text or "").replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def when(timestamp: object) -> str:
    value = str(timestamp or "")
    if "T" not in value:
        return value
    date_part, _, time_part = value.partition("T")
    return f"{date_part} {time_part[:5]}"


def pills(items: Iterable[str], empty_text: str) -> str:
    rendered = "".join(f'<span class="pill">{esc(item)}</span>' for item in items or [])
    return rendered or f'<span class="empty">{esc(empty_text)}</span>'


# ---------------------------------------------------------------------------
# Section furniture
# ---------------------------------------------------------------------------

def section(title: str, note: str = "") -> None:
    note_html = f'<span class="note">{esc(note)}</span>' if note else ""
    html(f'<div class="section-head"><h2>{esc(title)}</h2>{note_html}</div>')


def stat_strip(items: Sequence[tuple[str, str, str]], columns: int = 4, flag_index: int = -1) -> None:
    cells = []
    for index, (key, value, sub) in enumerate(items):
        classes = "stat flag" if index == flag_index else "stat"
        sub_html = f'<span class="sub">{esc(sub)}</span>' if sub else ""
        cells.append(
            f'<div class="{classes}"><span class="k">{esc(key)}</span>'
            f'<span class="v">{esc(value)}</span>{sub_html}</div>'
        )
    html(f'<div class="stats c{columns}">{"".join(cells)}</div>')


def disclaimer(text: str) -> None:
    html(
        '<div class="disclaimer"><span class="label">Not a diagnosis</span>'
        f"{esc(text)}</div>"
    )


# ---------------------------------------------------------------------------
# The 0.524
# ---------------------------------------------------------------------------

def transfer_callout(predicted_label: str | None = None) -> None:
    """The cross-source transfer number, with the label-specific figure when there is one.

    Published deliberately. A prediction of cholera carries cholera's 0.015 next
    to it, because the pooled figure alone would understate how badly that label
    travels.
    """
    pooled = evaluation_data.pooled_transfer_recall()
    if pooled is None:
        return

    label_entry = evaluation_data.transfer_for_label(predicted_label or "")
    body = (
        "<p>Hold an entire CDC or WHO source passage out of training and test the model on it, "
        "and pooled recall across the nine labels that can be transfer-tested is "
        f"<strong>{fmt3(pooled)}</strong>. The model is substantially learning how one source "
        "page words a condition, not the condition. Held-out accuracy is the ceiling; this is "
        "the floor.</p>"
    )

    if label_entry and label_entry.get("evaluable"):
        recall = float(label_entry["mean_recall_on_held_out_passage"])
        passages = label_entry.get("passages")
        headline, headline_sub = fmt3(recall), f"{esc(title_case(predicted_label))} transfer"
        body += (
            f"<p>For <strong>{esc(title_case(predicted_label))}</strong> specifically, transfer "
            f"recall is <strong>{fmt3(recall)}</strong> across {esc(passages)} source passages"
            + (
                ". Phrased by a source this model never trained on, this label is largely missed."
                if recall < 0.4
                else ". This label holds up comparatively well across sources."
            )
            + "</p>"
        )
    elif label_entry is not None:
        headline, headline_sub = fmt3(pooled), "Pooled transfer"
        body += (
            f"<p><strong>{esc(title_case(predicted_label))}</strong> cannot be transfer-tested at "
            "all: only one source passage backs it, so there is nothing to hold out. Its "
            "generalisation is unmeasured, not good.</p>"
        )
    else:
        headline, headline_sub = fmt3(pooled), "Pooled transfer"

    html(
        '<div class="transfer label-scope">'
        f'<div><span class="big">{headline}<span class="big-sub">{headline_sub}</span></span></div>'
        '<div><div class="ttl" role="heading" aria-level="3">Cross-source transfer recall</div>'
        f"{body}</div></div>"
    )


# ---------------------------------------------------------------------------
# A result
# ---------------------------------------------------------------------------

CONFIDENCE_FLOOR = 0.20


def verdict(result: dict, *, animate: bool = False) -> None:
    """Headline plus attribution, side by side, because they belong together.

    The old interface printed a disease and, underneath it, the classical
    model's accuracy — whichever engine had actually produced the answer. The
    attribution panel here always describes the engine named in
    `decision.decided_by`, and the metrics next to it are that engine's own.
    """
    decision = result.get("decision", {})
    engine = decision.get("decided_by") or result.get("model_used") or {}
    metrics = engine.get("metrics") or {}
    abstained = bool(decision.get("abstained"))
    confidence = float(result.get("confidence") or 0.0)
    pct = max(0.0, min(100.0, confidence * 100.0))

    transfer_recall = metrics.get("cross_source_transfer_recall")
    transfer_value = fmt3(transfer_recall)
    transfer_sub = "pooled, source held out" if transfer_recall is not None else "not measured here"

    enter = ' enter' if animate else ""
    caveat = (
        "No engine was confident enough to name a condition. The system abstained instead of "
        "guessing — that is the designed behaviour, not a failure."
        if abstained
        else "A symptom-based educational suggestion over 14 labels. Not a diagnosis, and not a "
        "differential."
    )

    html(
        f'<div class="verdict{" abstained" if abstained else ""}{enter}">'
        '<div class="verdict-main">'
        f'<span class="label">{"Outcome" if abstained else "Suggested condition"}</span>'
        '<div class="dx-title" role="heading" aria-level="2">'
        f"{esc(title_case(result.get('predicted_disease')))}</div>"
        f'<p class="caveat">{esc(caveat)}</p>'
        '<div class="confidence">'
        '<div class="confidence-row">'
        f'<span class="n">{fmt3(confidence)}</span>'
        f'<span class="meaning">{esc(engine.get("confidence_meaning", ""))}</span>'
        "</div>"
        f'<div class="track"><div class="fill" style="width:{pct:.1f}%"></div>'
        f'<div class="floor" style="left:{CONFIDENCE_FLOOR * 100:.0f}%" data-label="0.20 floor"></div></div>'
        '<div class="track-legend"></div>'
        "</div></div>"
        '<div class="attribution">'
        # On an abstention nothing decided anything. The engine named here is the
        # one whose number the confidence bar is showing, and calling it the
        # decider would be the same class of mistake this rework set out to fix.
        f'<span class="label">{"Primary engine" if abstained else "Decided by"}</span>'
        f'<p class="engine-name">{esc(engine.get("name", "unknown"))}'
        f'<span class="engine-key">{esc(engine.get("key", "?"))}</span></p>'
        f'<p class="engine-family">{esc(engine.get("family", ""))}</p>'
        '<div class="metric-pair">'
        '<div class="col"><span class="k">Its accuracy</span></div>'
        '<div class="col floorward"><span class="k">Its transfer</span></div>'
        f'<div class="col"><span class="v">{fmt3(metrics.get("accuracy"))}</span>'
        f'<span class="sub">{esc(fmt_ci(metrics.get("accuracy_ci_95")) or "held-out test set")}</span></div>'
        f'<div class="col floorward"><span class="v">{transfer_value}</span>'
        f'<span class="sub">{esc(transfer_sub)}</span></div>'
        "</div></div></div>"
    )

    html(
        f'<div class="policy-note{enter}{" d1" if animate else ""}">'
        '<span class="label">Why this engine answered</span>'
        f"{esc(decision.get('policy_reason', ''))} "
        f"<code>{esc(decision.get('policy', ''))}</code></div>"
    )


ENGINE_ROLE = {
    "classical": "Statistical model",
    "advanced": "Statistical model",
    "symptom_profile": "Rule engine",
}


def opinions_table(result: dict, *, animate: bool = False) -> None:
    """What every engine thought, including the ones that were overruled."""
    decision = result.get("decision", {})
    decided_key = (decision.get("decided_by") or {}).get("key")
    abstained = bool(decision.get("abstained"))
    rows = [
        '<div class="opinion head"><span>Engine</span><span>Its answer</span>'
        "<span>Confidence</span><span>Reasoning</span></div>"
    ]

    for opinion in result.get("engine_opinions", []) or []:
        key = opinion.get("engine", "")
        used = key == decided_key and not abstained
        if not opinion.get("available", True):
            tag = '<span class="tag unavailable">unavailable</span>'
        elif key == decided_key and abstained:
            # Its answer was discarded along with everyone else's.
            tag = '<span class="tag overruled">below floor</span>'
        elif used:
            tag = '<span class="tag decided">decided</span>'
        else:
            tag = '<span class="tag overruled">not used</span>'
        answer = title_case(opinion.get("disease")) if opinion.get("disease") else "No opinion"
        rows.append(
            f'<div class="opinion{" used" if used else ""}">'
            f'<span class="engine">{esc(key)}{tag}</span>'
            f'<span class="verdict-cell">{esc(answer)}</span>'
            f'<span class="conf">{fmt3(opinion.get("confidence"))}</span>'
            f'<span class="detail">{esc(opinion.get("detail", ""))}</span>'
            "</div>"
        )

    enter = ' enter d2' if animate else ""
    html(f'<div class="opinions{enter}">{"".join(rows)}</div>')
    html(
        '<p class="quiet" style="margin-top:0.55rem;max-width:104ch">The rule engine reports a '
        "share of profile-match weight, not a probability. The two confidence columns are on "
        "different scales and the arbitration policy never compares them with each other "
        "— the rule engine only speaks where the model has already said it does not "
        "know.</p>"
    )


def entities_card(entities: dict, cleaned_text: str, *, animate: bool = False) -> None:
    enter = " enter d1" if animate else ""
    html(
        f'<div class="card{enter}">'
        '<div class="entity-row"><span class="k">Symptoms</span>'
        f'<span>{pills(entities.get("symptoms"), "None detected")}</span></div>'
        '<div class="entity-row"><span class="k">Diseases named</span>'
        f'<span>{pills(entities.get("diseases"), "None mentioned")}</span></div>'
        '<div class="entity-row"><span class="k">Medications named</span>'
        f'<span>{pills(entities.get("medications"), "None mentioned")}</span></div>'
        '<div class="entity-row"><span class="k">Dosage mentions</span>'
        f'<span>{pills(entities.get("dosage_mentions"), "None detected")}</span></div>'
        '<div class="entity-row"><span class="k">Normalised text</span>'
        f'<span class="mono" style="font-size:0.78rem;color:var(--muted)">{esc(cleaned_text)}</span></div>'
        "</div>"
    )


def actions_card(actions: Sequence[str], *, animate: bool = False) -> None:
    enter = " enter d2" if animate else ""
    items = "".join(f"<li>{esc(action)}</li>" for action in actions or [])
    if not items:
        items = '<li class="empty">No actions available for this outcome.</li>'
    html(f'<div class="card{enter}"><ol class="action-list">{items}</ol></div>')


def medications(medicines: Sequence[dict], *, animate: bool = False) -> None:
    """Educational medication information. The warnings block is not optional."""
    if not medicines:
        html(
            '<div class="card"><p class="quiet">No medication information is attached to this '
            "outcome. Where the system abstains, it offers no medication guidance at all.</p></div>"
        )
        return

    enter = " enter d3" if animate else ""
    for medication in medicines:
        warnings = "".join(f"<li>{esc(w)}</li>" for w in medication.get("warnings", []) or [])
        if not warnings:
            warnings = "<li>Confirm every dose with a qualified clinician before use.</li>"
        html(
            f'<div class="med{enter}">'
            '<div class="med-head"><span class="kind">Educational information</span>'
            '<div class="med-title" role="heading" aria-level="3">'
            f"{esc(medication.get('name', ''))}</div></div>"
            '<div class="med-body">'
            f'<div><span class="label">Typical dosage</span><p>{esc(medication.get("standard_dosage", ""))}</p></div>'
            f'<div><span class="label">Administration</span><p>{esc(medication.get("administration", ""))}</p></div>'
            "</div>"
            '<div class="med-warnings"><span class="label">Warnings</span>'
            f"<ul>{warnings}</ul></div></div>"
        )


def result_body(result: dict, *, animate: bool = False) -> None:
    """One full result, in the order the pipeline produced it."""
    verdict(result, animate=animate)

    section("Cross-source generalisation", "models/classical/model_metadata.json")
    transfer_callout(result.get("predicted_disease"))

    section("Every engine's opinion", f"policy {result.get('decision', {}).get('policy', '')}")
    opinions_table(result, animate=animate)

    left, right = st.columns([1, 1])
    with left:
        section("Extracted entities", "rule-based, negation-aware")
        entities_card(result.get("extracted_entities", {}), result.get("cleaned_text", ""), animate=animate)
    with right:
        section("Recommended next steps", "educational guidance")
        actions_card(result.get("recommended_actions", []), animate=animate)

    section("Medication information", "every entry carries its warnings")
    medications(result.get("recommended_medicines", []), animate=animate)
    disclaimer(result.get("disclaimer", ""))


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------

def bar_rows(rows: Sequence[dict], *, low: float = 0.40, mid: float = 0.70) -> None:
    rendered = []
    for row in rows:
        value = row.get("value")
        if value is None:
            rendered.append(
                f'<div class="bar na"><span class="k">{esc(row.get("label", ""))}</span>'
                f'<span class="t"></span><span class="v">n/a</span></div>'
            )
            continue
        value = float(value)
        tone = "low" if value < low else ("mid" if value < mid else "")
        rendered.append(
            f'<div class="bar {tone}"><span class="k">{esc(row.get("label", ""))}</span>'
            f'<span class="t"><i style="width:{max(1.0, value * 100):.1f}%"></i></span>'
            f'<span class="v">{value:.3f}</span></div>'
        )
    axis = (
        '<div class="bar-axis"><div class="scale"><span>0.0</span><span>0.25</span>'
        "<span>0.50</span><span>0.75</span><span>1.0</span></div></div>"
    )
    html(f'<div class="bars">{"".join(rendered)}{axis}</div>')
