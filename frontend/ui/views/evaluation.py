"""Evaluation: what this model actually does, including the parts that look bad.

This screen exists because the interesting number in this project is not the
accuracy. It is the cross-source transfer recall of 0.524, and it is published
on purpose. A model demo that reports only its best figure has not been
evaluated; it has been advertised.
"""

from __future__ import annotations

import streamlit as st

from .. import evaluation_data
from ..components import (
    DASH,
    bar_rows,
    esc,
    fmt3,
    fmt_ci,
    html,
    section,
    stat_strip,
    title_case,
)

WITHDRAWN = (
    "An earlier version of this project reported 0.906 as its headline accuracy. That figure "
    "came from ranking candidate pipelines on the same 32 rows it was then reported on, with "
    "the shipped model refit on those rows. It described nothing, it is not comparable to any "
    "number on this screen, and it has been withdrawn."
)


def render(engines_payload: dict) -> None:
    pooled = evaluation_data.pooled_transfer_recall()
    transfer = evaluation_data.transfer()
    cross_validation = evaluation_data.cross_validation()
    external = evaluation_data.external_use_cases()
    dataset = evaluation_data.dataset_facts()
    split = evaluation_data.split_facts()
    classical = _classical_metrics(engines_payload)

    section("Read this first", "the honest generalisation measure")
    _transfer_headline(pooled, transfer)

    section("Per-label transfer", "mean recall when that label's own source passage is held out")
    rows = evaluation_data.transfer_rows()
    bar_rows(
        [
            {
                "label": title_case(row["label"]),
                "value": row["recall"] if row["evaluable"] else None,
            }
            for row in rows
        ]
    )
    blocked = [row for row in rows if not row["evaluable"]]
    if blocked:
        st.caption(
            "Not evaluable: "
            + ", ".join(title_case(row["label"]) for row in blocked)
            + ". Each is backed by a single source passage, so there is nothing to hold out. "
            "Their generalisation is unmeasured, which is not the same as good."
        )

    section("Held-out performance", "group-aware split, the artifact that ships")
    stat_strip(
        [
            (
                "Test accuracy",
                fmt3(classical.get("accuracy")),
                fmt_ci(classical.get("accuracy_ci_95")),
            ),
            ("Test macro F1", fmt3(classical.get("macro_f1")), "14 labels, unweighted"),
            (
                "Grouped 5-fold CV",
                fmt3(cross_validation.get("accuracy_mean")),
                fmt_ci(cross_validation.get("accuracy_ci")),
            ),
            (
                "External scenarios",
                f"{external.get('correct', DASH)}/{external.get('count', DASH)}",
                fmt_ci(external.get("accuracy_ci")),
            ),
            (
                "Cross-source transfer",
                fmt3(pooled),
                f"{transfer.get('rows_evaluated', DASH)} rows, source held out",
            ),
        ],
        columns=5,
        flag_index=4,
    )
    st.caption(
        "The first four describe performance on wording whose style the model has seen. The "
        "fifth describes performance on a source it has not. Treat the accuracy as the ceiling "
        "and the transfer recall as the floor."
    )

    section("Every engine on one split", "models/evaluation/engine_metrics.json")
    _engine_table(engines_payload)

    left, right = st.columns([1, 1])
    with left:
        section("Weakest labels on the test split", "per-class recall, worst first")
        bar_rows(
            [
                {"label": title_case(row.get("label")), "value": row.get("recall")}
                for row in evaluation_data.per_class_rows()[:7]
            ],
            low=0.75,
            mid=0.9,
        )
        st.caption(
            "Recall on the held-out test split, with 95% Wilson intervals in "
            "models/classical/metrics.txt. Supports are small; the intervals are wide."
        )
    with right:
        section("Cross-validation folds", "StratifiedGroupKFold, n=5")
        folds = cross_validation.get("accuracy_per_fold", []) or []
        # One tone. Fold-to-fold variation is spread, not quality, and colouring
        # it good-to-bad would read as a judgement the number does not make.
        bar_rows(
            [{"label": f"Fold {index + 1}", "value": value} for index, value in enumerate(folds)],
            low=0.0,
            mid=0.0,
        )
        st.caption(
            "Folds are group-aware: a note and its paraphrases never land on both sides of a "
            "split. The spread across folds is why the interval is reported."
        )

    section("How it was measured", "methodology and its limits")
    method_left, method_right = st.columns([1, 1])
    with method_left:
        sizes = split.get("sizes", {})
        groups = split.get("groups", {})
        html(
            '<div class="card"><div class="card-title" role="heading" aria-level="3">Split</div>'
            '<div class="entity-row"><span class="k">Train</span>'
            f'<span class="mono">{esc(sizes.get("train", DASH))} rows · '
            f'{esc(groups.get("train", DASH))} groups</span></div>'
            '<div class="entity-row"><span class="k">Validation</span>'
            f'<span class="mono">{esc(sizes.get("validation", DASH))} rows · '
            f'{esc(groups.get("validation", DASH))} groups</span></div>'
            '<div class="entity-row"><span class="k">Test</span>'
            f'<span class="mono">{esc(sizes.get("test", DASH))} rows · '
            f'{esc(groups.get("test", DASH))} groups</span></div>'
            '<div class="entity-row"><span class="k">Selection</span>'
            f'<span>{esc(split.get("model_selection", ""))}</span></div>'
            '<div class="entity-row"><span class="k">Shipped artifact</span>'
            f'<span>{esc(split.get("shipped_model_fit_on", ""))}</span></div>'
            "</div>"
        )
    with method_right:
        limitations = "".join(
            f"<li>{esc(item)}</li>" for item in split.get("known_limitations", [])
        )
        html(
            '<div class="card">'
            '<div class="card-title" role="heading" aria-level="3">Known limitations</div>'
            f'<ol class="action-list">{limitations}</ol></div>'
        )

    section("Dataset", "616 rows, every one attributable")
    provenance = dataset.get("provenance", {})
    stat_strip(
        [
            ("Rows", str(dataset.get("rows", DASH)), f"{dataset.get('labels', DASH)} labels"),
            ("Source rows", str(provenance.get("source", DASH)), "one quoted passage each"),
            (
                "Source combinations",
                str(provenance.get("source_combination", DASH)),
                "subset of one passage",
            ),
            ("Paraphrases", str(provenance.get("paraphrase", DASH)), "wording variants only"),
        ],
        columns=4,
    )
    st.caption(
        f"Built from {dataset.get('source_count', DASH)} live-fetched CDC and WHO captures. "
        "No clinical description in the dataset was written by hand and attributed to a source. "
        "Because every row for a label descends from one or two passages, the evaluation folds "
        "are not clinically independent — which is exactly what the transfer measure above "
        "is there to expose."
    )

    section("Withdrawn", "what this project no longer claims")
    html(f'<div class="card"><p class="prose">{esc(WITHDRAWN)}</p></div>')


def _transfer_headline(pooled: float | None, transfer: dict) -> None:
    if pooled is None:
        html(
            '<div class="card"><p class="quiet">The cross-source transfer evaluation is not '
            "available on this machine.</p></div>"
        )
        return
    html(
        '<div class="transfer">'
        f'<div><span class="big">{fmt3(pooled)}'
        '<span class="big-sub">Pooled transfer recall</span></span></div>'
        '<div><div class="ttl" role="heading" aria-level="3">The model is substantially '
        "learning one source page's phrasing</div>"
        "<p>Hold an entire CDC or WHO source passage out of training, then test the model on "
        f"that passage. Across the nine labels that can be tested this way, pooled recall over "
        f"{esc(transfer.get('rows_evaluated', DASH))} rows is <strong>{fmt3(pooled)}</strong>. "
        "Cholera falls to 0.015 and flu to 0.189: rephrased by a source it never trained on, the "
        "model misses them almost entirely.</p>"
        "<p>This is published deliberately. It is the number that says what the system would do "
        "against wording it has not met, and it is the reason this is a triage demonstrator "
        "rather than anything a clinic should rely on.</p></div></div>"
    )


def _classical_metrics(engines_payload: dict) -> dict:
    for engine in engines_payload.get("engines", []):
        if engine.get("key") == "classical":
            return engine.get("metrics") or {}
    return evaluation_data.engine_metrics_file().get("engines", {}).get("classical", {})


def _engine_table(engines_payload: dict) -> None:
    rows = [
        '<div class="row head"><span>Engine</span><span>Accuracy</span><span>95% interval</span>'
        "<span>Macro F1</span><span>Scenarios</span><span>Transfer recall</span></div>"
    ]
    for engine in engines_payload.get("engines", []):
        metrics = engine.get("metrics") or {}
        transfer_recall = metrics.get("cross_source_transfer_recall")
        rows.append(
            '<div class="row">'
            f'<span class="name">{esc(engine.get("name", ""))}'
            f'<small>{esc(engine.get("key", ""))} · {esc(engine.get("family", ""))}</small></span>'
            f'<span class="n">{fmt3(metrics.get("accuracy"))}</span>'
            f'<span class="ci">{esc(fmt_ci(metrics.get("accuracy_ci_95")) or DASH)}</span>'
            f'<span class="n">{fmt3(metrics.get("macro_f1"))}</span>'
            f'<span class="n">{fmt3(metrics.get("external_use_case_accuracy"))}</span>'
            + (
                f'<span class="n">{fmt3(transfer_recall)}</span>'
                if transfer_recall is not None
                else '<span class="n dim">not measured</span>'
            )
            + "</div>"
        )
    html(f'<div class="dtable etable">{"".join(rows)}</div>')
    st.caption(
        "Every engine run standalone over the same group-aware held-out split of 123 rows, with "
        "no arbitration. An engine that declines to answer is scored wrong, not skipped. The "
        "rule engine's 0.707 is the number that matters whenever the rule engine is the one that "
        "answered — which is what the attribution panel on a result reports."
    )
