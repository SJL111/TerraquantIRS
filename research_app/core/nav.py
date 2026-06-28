"""Page navigation helpers."""
from __future__ import annotations

import streamlit as st


def open_company(ticker: str) -> None:
    st.session_state["overview_ticker"] = ticker.upper()
    st.switch_page("pages/1_公司概览.py")


def handle_map_click(
    clicked: str | None,
    *,
    skip: str | None = None,
) -> None:
    """Navigate to company overview when the map reports a click."""
    if not clicked or (skip and clicked.upper() == skip.upper()):
        return
    open_company(clicked)
