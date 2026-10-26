"""AquaPulse: Yamuna AI — entry point, session_state router and sidebar controller."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(page_title="AquaPulse · Yamuna AI", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

from backend.data_generator import get_data_signature, load_dataset
from backend.lstm_engine import tf_available, train_station_model
from frontend import ai_time_machine, command_center, login_view, model_diagnostics, styles, what_if_lab

PAGES = {
    "🛰️  Command Center": command_center.render,
    "🔮  AI Time Machine": ai_time_machine.render,
    "🧪  What-If Lab": what_if_lab.render,
    "🩺  Model Diagnostics": model_diagnostics.render,
}


# ------------------------------- cached backend ------------------------------- #
@st.cache_data(show_spinner="Loading dataset…")
def _load(_sig):
    return load_dataset()


@st.cache_resource(show_spinner=False)
def _tf_ok() -> bool:
    return tf_available()


@st.cache_resource(show_spinner=False)
def _train(station, _sig, use_lstm, epochs, _df):
    bar = st.progress(0.0, text=f"Training {station} forecaster…")

    def cb(ep, total, logs):
        vl = logs.get("val_loss")
        bar.progress(min(ep / total, 1.0), text=f"Epoch {ep}/{total}" + (f" · val_loss {vl:.4f}" if vl is not None else ""))

    try:
        return train_station_model(_df, station, epochs=epochs, use_lstm=use_lstm, progress_cb=cb)
    finally:
        bar.empty()


def main() -> None:
    st.session_state.setdefault("authenticated", False)
    if not st.session_state["authenticated"]:
        login_view.render()
        return

    styles.inject_css()
    sig = get_data_signature()
    df, meta = _load(sig)
    tf_ok = _tf_ok()
    st.session_state.setdefault("use_lstm", tf_ok)
    st.session_state.setdefault("epochs", 30)
    st.session_state.setdefault("page", list(PAGES)[0])

    def get_bundle(station: str):
        use_lstm = bool(st.session_state["use_lstm"] and tf_ok)
        with st.spinner(f"Preparing {station} model (trained once, then cached)…"):
            return _train(station, sig, use_lstm, int(st.session_state["epochs"]), df)

    ctx = {"df": df, "meta": meta, "get_bundle": get_bundle, "tf_ok": tf_ok}

    with st.sidebar:
        st.markdown('<div class="brand">🌊 Aqua<span>Pulse</span></div><div class="muted">Yamuna AI · v1.0</div>', unsafe_allow_html=True)
        st.write("")
        st.radio("Navigate", list(PAGES), key="page", label_visibility="collapsed")
        st.divider()
        model_diagnostics.render_source_manager(meta, key="side", detailed=False)
        with st.expander("⚙️ Model engine"):
            st.toggle("Use LSTM (TensorFlow)", key="use_lstm", disabled=not tf_ok,
                      help="Turn off for an instant Ridge baseline." if tf_ok else "TensorFlow not installed: using fallback model.")
            st.slider("Max epochs", 5, 100, key="epochs", step=5)
            st.caption("Active: " + ("LSTM" if st.session_state["use_lstm"] and tf_ok else "Ridge fallback"))
        st.divider()
        st.markdown(f'<div class="muted">Signed in as <b>{st.session_state.get("user", "evaluator")}</b></div>', unsafe_allow_html=True)
        if st.button("Sign out", key="logout", **styles.STRETCH):
            st.session_state["authenticated"] = False
            st.rerun()

    PAGES[st.session_state["page"]](ctx)


main()
