"""View 0: frosted-glass login portal with a 1-click evaluator bypass."""
from __future__ import annotations

import streamlit as st

from . import styles as ui

# Demo-only credentials (replace with st.secrets / real auth for production)
CREDENTIALS = {"admin": "aquapulse", "evaluator": "yamuna2026"}


def _login(user: str) -> None:
    st.session_state["authenticated"] = True
    st.session_state["user"] = user
    st.session_state["page"] = "Command Center"
    st.rerun()


def render() -> None:
    ui.inject_css(login=True)
    _, mid, _ = st.columns([1, 1.15, 1])
    with mid:
        st.write("")
        st.markdown(ui._flat("""
            <div class="glass" style="text-align:center;padding:2rem 1.5rem;margin-bottom:1rem">
              <div style="font-size:3rem">🌊</div>
              <div class="brand" style="font-size:2rem">Aqua<span>Pulse</span></div>
              <div style="font-family:'Space Grotesk';color:#8ab4ff;font-weight:600">Yamuna AI</div>
              <p class="muted" style="margin-top:.6rem">Predictive Water Quality Forecasting Engine<br>
              LSTM-powered intelligence for the Delhi Yamuna corridor</p>
            </div>"""), unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            user = st.text_input("Username", placeholder="admin")
            pwd = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Sign in", **ui.STRETCH)
        if submitted:
            if CREDENTIALS.get(user.strip().lower()) == pwd and pwd:
                _login(user.strip().lower())
            else:
                st.error("Invalid credentials. Use the evaluator access below.")

        st.markdown('<p class="muted" style="text-align:center;margin:.6rem 0">— or —</p>', unsafe_allow_html=True)
        if st.button("⚡ 1-Click Evaluator Access", key="bypass", **ui.STRETCH):
            _login("evaluator")
        st.markdown('<p class="muted" style="text-align:center;margin-top:1rem">'
                    'Mid-term build · CPCB-aligned synthetic baseline with live hot-swap to real data</p>',
                    unsafe_allow_html=True)
