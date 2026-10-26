"""View 2: LSTM dual-line forecast chart, historical back-casting and automated spike alerts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.data_generator import wqi_category
from backend.lstm_engine import detect_spike_alert, forecast, prepare_station_frame

from . import styles as ui

NEON, PINK = "#5eead4", "#ff5c7a"


def _vline(fig: go.Figure, x, label: str) -> None:
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=dict(color="rgba(255,255,255,.55)", dash="dot", width=1.5))
    fig.add_annotation(x=x, y=1.02, yref="paper", text=label, showarrow=False, font=dict(size=11, color="#cfe3ff"))


def _series_chart(hist: pd.DataFrame, fc: pd.DataFrame, actual: pd.DataFrame, col: str, title: str,
                  fc_col: str | None = None, band: bool = False, lookback: int = 90) -> go.Figure:
    fc_col = fc_col or col
    h = hist.tail(lookback)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["Date"], y=h[col], name="Observed", line=dict(color="#8ab4ff", width=2.5)))
    if not actual.empty:
        fig.add_trace(go.Scatter(x=actual["Date"], y=actual[col], name="Actual (what really happened)",
                                 line=dict(color="#ffd166", width=2.5)))
    x = pd.concat([h["Date"].tail(1), fc["Date"]])
    y = pd.concat([h[col].tail(1), fc[fc_col]])
    if band:
        fig.add_trace(go.Scatter(x=fc["Date"], y=fc["WQI_Upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc["Date"], y=fc["WQI_Lower"], fill="tonexty", fillcolor=ui.hex_to_rgba(NEON, .15),
                                 line=dict(width=0), name="90% band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, name="LSTM forecast", line=dict(color=NEON, width=3.2, dash="dash"),
                             mode="lines+markers", marker=dict(size=[0] + [5] * len(fc))))
    _vline(fig, h["Date"].iloc[-1], "forecast origin")
    fig.update_yaxes(title_text=title)
    return ui.style_fig(fig, 420)


def render(ctx: dict) -> None:
    df, meta = ctx["df"], ctx["meta"]
    ui.section_header("AI Time Machine", "LSTM forecasting · back-cast any past date and compare against reality", "🔮")

    stations = meta["stations"]
    c1, c2, c3 = st.columns([1.3, 1, 1.8])
    station = c1.selectbox("Station", stations, index=stations.index("Okhla Barrage") if "Okhla Barrage" in stations else 0, key="tm_station")
    horizon = c2.slider("Horizon (days)", 7, 60, 30, key="tm_h")
    mode = c3.radio("Mode", ["🔮 Forecast the future", "⏪ Time machine (back-cast)"], horizontal=True, key="tm_mode")

    bundle = ctx["get_bundle"](station)
    full = prepare_station_frame(df, station)

    cutoff = None
    if mode.startswith("⏪"):
        lo = full["Date"].iloc[bundle.window + 30].date()
        hi = (full["Date"].iloc[-1] - pd.Timedelta(days=horizon)).date()
        if hi <= lo:
            st.warning("Not enough history for back-casting with this horizon.")
            return
        default = max(lo, hi - pd.Timedelta(days=60))
        cutoff = st.slider("Travel back to…", min_value=lo, max_value=hi, value=default, key="tm_cut")
        cutoff = pd.Timestamp(cutoff)

    fc = forecast(bundle, df, horizon, cutoff)
    hist = full if cutoff is None else full[full["Date"] <= cutoff]
    actual = full[full["Date"].isin(fc["Date"])] if cutoff is not None else full.iloc[0:0]

    alert = detect_spike_alert(fc, hist)
    ui.alert_banner(alert["level"], alert["headline"], alert["detail"])
    if cutoff is not None and bool(fc["In_Sample"].iloc[0]):
        st.info("This date falls inside the model's training period (in-sample). Pick a later date for an honest out-of-sample test.")

    k = st.columns(4)
    with k[0]:
        ui.kpi_card("Forecast avg WQI", f"{fc['WQI'].mean():.1f}", f"vs {hist['WQI'].tail(7).mean():.1f} last observed week",
                    ui.category_color(wqi_category(fc['WQI'].mean())))
    with k[1]:
        ui.kpi_card("Worst day", f"{fc['WQI'].min():.1f}", f"{fc.loc[fc['WQI'].idxmin(), 'Date']:%d %b %Y}", PINK)
    with k[2]:
        ui.kpi_card("Peak BOD", f"{fc['BOD_mgL'].max():.1f}", f"mg/L · {alert['bod_surge_pct']:+.0f}% vs last week", "#ffd166", unit="")
    with k[3]:
        if cutoff is not None and not fc["Actual_WQI"].isna().all():
            mae = float(np.nanmean(np.abs(fc["Actual_WQI"] - fc["WQI"])))
            ui.kpi_card("Back-cast MAE", f"{mae:.2f}", "WQI points vs actual", NEON)
        else:
            ui.kpi_card("Model", bundle.model_kind, f"hold-out WQI MAE {bundle.metrics['WQI']['MAE']:.2f} · R² {bundle.metrics['WQI']['R2']:.2f}", NEON)

    st.markdown("#### WQI: observed vs LSTM forecast")
    ui.show(_series_chart(hist, fc, actual, "WQI", "WQI", band=True))

    tabs = st.tabs(["BOD", "Dissolved Oxygen", "Ammonia"])
    for tab, (col, title) in zip(tabs, (("BOD_mgL", "BOD (mg/L)"), ("DO_mgL", "DO (mg/L)"), ("Ammonia_mgL", "Ammonia (mg/L)"))):
        with tab:
            ui.show(_series_chart(hist, fc, actual, col, title, lookback=60))

    with st.expander("Forecast table"):
        st.dataframe(fc.drop(columns=["In_Sample"]), hide_index=True, **ui.STRETCH)
