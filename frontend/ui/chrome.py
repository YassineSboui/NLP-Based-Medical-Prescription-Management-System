"""Page shell: the stylesheet, the masthead, the caution ribbon, the offline state."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import streamlit as st

from . import api, evaluation_data
from .components import esc, fmt3, html

STYLESHEET = Path(__file__).resolve().parent.parent / "assets" / "app.css"

CAUTION_TEXT = (
    "Academic demonstration only. This application does not diagnose, does not prescribe, and "
    "must not be used for self-medication. Every condition, test, medicine and dose shown here "
    "requires confirmation by a qualified healthcare professional."
)


@lru_cache(maxsize=4)
def _stylesheet_at(_mtime: float) -> str:
    try:
        return STYLESHEET.read_text(encoding="utf-8")
    except OSError:
        return ""


def _stylesheet() -> str:
    """The stylesheet, cached on its modification time.

    Cached because it is re-read on every Streamlit rerun, keyed on mtime
    because a plain cache means editing the CSS does nothing until the server
    is restarted.
    """
    try:
        mtime = STYLESHEET.stat().st_mtime
    except OSError:
        return ""
    return _stylesheet_at(mtime)


def apply_page_style() -> None:
    """Load the stylesheet. Once, from one file.

    The previous frontend inlined roughly 830 lines of CSS into the Python
    module across two separate blocks — a global one and a near-copy inside the
    results renderer. They had already diverged.
    """
    st.markdown(f"<style>{_stylesheet()}</style>", unsafe_allow_html=True)


def masthead(status: api.ServiceStatus, engines_payload: dict | None) -> None:
    chips = []

    if status.reachable:
        chips.append(
            '<span class="chip ok"><span class="dot"></span>'
            f'<span class="k">Service</span><span class="v">v{esc(status.version or "?")}</span></span>'
        )
        chips.append(
            f'<span class="chip ok"><span class="dot"></span><span class="k">Engines</span>'
            f'<span class="v">{status.engine_count}/{len(status.engines_available or {})}</span></span>'
        )
        chips.append(
            '<span class="chip'
            + (" ok" if status.database_ready else " down")
            + '"><span class="dot"></span><span class="k">Record store</span>'
            + f'<span class="v">{"ready" if status.database_ready else "unavailable"}</span></span>'
        )
    else:
        chips.append(
            '<span class="chip down"><span class="dot"></span>'
            '<span class="k">Service</span><span class="v">offline</span></span>'
        )

    pooled = evaluation_data.pooled_transfer_recall()
    if pooled is not None:
        chips.append(
            '<span class="chip warn"><span class="dot"></span>'
            f'<span class="k">Transfer recall</span><span class="v">{fmt3(pooled)}</span></span>'
        )

    floor = (engines_payload or {}).get("model_confidence_floor")
    if floor is not None:
        chips.append(
            '<span class="chip"><span class="dot"></span>'
            f'<span class="k">Confidence floor</span><span class="v">{fmt3(floor)}</span></span>'
        )

    html(
        '<div class="masthead"><div class="masthead-id">'
        '<div class="mark">NLP</div>'
        "<div><h1>Symptom Triage Console</h1>"
        '<p class="tagline">Free-text symptom analysis over 14 source-backed conditions. Every '
        "answer names the engine that produced it and reports that engine's own measured "
        "performance.</p></div></div>"
        f'<div class="masthead-meta">{"".join(chips)}</div></div>'
    )


def caution_ribbon() -> None:
    html(
        '<div class="caution-ribbon"><span class="label">Medical safety</span>'
        f"<p>{esc(CAUTION_TEXT)}</p></div>"
    )


def service_unavailable(status: api.ServiceStatus) -> None:
    """What the user sees when the analysis service cannot be reached.

    A product state, not a developer instruction. The previous interface printed
    a shell command here and asked the end user to run it.
    """
    html(
        '<div class="unavailable"><div class="glyph">!</div>'
        '<div class="ttl" role="heading" aria-level="2">Analysis service unavailable</div>'
        "<p>This console cannot reach the analysis service, so it has nothing to show and will "
        "not guess. No consultation has been recorded. Nothing you type will be lost — retry "
        "once the service is back.</p></div>"
    )
    _, middle, _ = st.columns([1, 0.9, 1])
    with middle:
        if st.button("Retry connection", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        if status.detail:
            with st.expander("Technical detail"):
                st.code(status.detail, language="text")
