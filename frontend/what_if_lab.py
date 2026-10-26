"""View 3: interactive policy sliders with real-time WQI recalculation."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from backend.data_generator import PARAMS
from backend.lstm_engine import (LEVER_SPECS, POLICY_PRESETS, apply_policy_to_forecast, forecast,
                                 prepare_station_frame, simulate_policy)

from . import styles as ui


def _apply_preset(name: str) -> None:
    for k in LEVER_SPECS:
        st.session_state[f"lv_{k}"] = POLICY_PRESETS[name].get(k, 0)


def render(ctx: dict) -> None:
    df, meta = ctx["df"], ctx["meta"]
    ui.section_header("What-If Lab", "Stress-test Yamuna cleanup policies · WQI recalculates instantly", "🧪")

    stations = meta["stations"]
    top = st.columns([1.3, 1.5, 2])
    station = top[0].selectbox("Station", stations, index=stations.index("Okhla Barrage") if "Okhla Barrage" in stations else 0, key="wi_station")
    basis = top[1].radio("Baseline", ["Latest 7-day mean", "Worst day (last 90 d)"], key="wi_basis")
    project = top[2].toggle("Overlay 30-day LSTM projection", value=False, key="wi_proj",
                            help="Runs the trained model forward and applies the policy to every forecast day.")

    sf = prepare_station_frame(df, station)
    if basis.startswith("Latest"):
        base = sf.tail(7)[PARAMS].mean().to_dict()
    else:
        tail = sf.tail(90)
        base = tail.loc[tail["WQI"].idxmin(), PARAMS].to_dict()

    for k, spec in LEVER_SPECS.items():
        st.session_state.setdefault(f"lv_{k}", spec["default"])

    st.markdown("**Scenario presets**")
    pc = st.columns(len(POLICY_PRESETS))
    for col, name in zip(pc, POLICY_PRESETS):
        col.button(name, key=f"preset_{name}", on_click=_apply_preset, args=(name,), **ui.STRETCH)

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown("#### Policy levers")
        levers = {}
        for k, spec in LEVER_SPECS.items():
            levers[k] = st.slider(f"{spec['label']} · {spec['unit']}", spec["min"], spec["max"], step=spec["step"],
                                  key=f"lv_{k}", help=spec["help"])

    res = simulate_policy(base, levers)
    after, before = res["after"], res["before"]
    with right:
        st.markdown("#### Projected impact")
        color = ui.category_color(res["category_after"])
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=res["wqi_after"], number=dict(font=dict(size=54, color="#fff")),
            delta=dict(reference=res["wqi_before"], increasing=dict(color="#34f5c5"), decreasing=dict(color="#ff5c7a")),
            title=dict(text=f"WQI · {res['category_before']} → {res['category_after']}", font=dict(size=15)),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=color, thickness=.3), bgcolor="rgba(255,255,255,.04)",
                       borderwidth=0, threshold=dict(line=dict(color="#fff", width=3), value=res["wqi_before"]),
                       steps=[dict(range=[0, 20], color="rgba(255,92,122,.28)"), dict(range=[20, 40], color="rgba(255,159,90,.24)"),
                              dict(range=[40, 60], color="rgba(255,209,102,.22)"), dict(range=[60, 80], color="rgba(110,231,255,.2)"),
                              dict(range=[80, 100], color="rgba(52,245,197,.22)")])))
        ui.show(ui.style_fig(fig, 300, legend=False))
        m = st.columns(3)
        m[0].metric("BOD (mg/L)", f"{after['BOD_mgL']:.1f}", f"{after['BOD_mgL'] - before['BOD_mgL']:.1f}", delta_color="inverse")
        m[1].metric("DO (mg/L)", f"{after['DO_mgL']:.2f}", f"{after['DO_mgL'] - before['DO_mgL']:+.2f}")
        m[2].metric("Ammonia (mg/L)", f"{after['Ammonia_mgL']:.2f}", f"{after['Ammonia_mgL'] - before['Ammonia_mgL']:.2f}", delta_color="inverse")

    names = [LEVER_SPECS[k]["label"] for k in LEVER_SPECS]
    wf = go.Figure(go.Waterfall(
        x=["Baseline WQI", *names, "Policy WQI"], measure=["absolute", *["relative"] * len(names), "total"],
        y=[res["wqi_before"], *res["lever_contributions"].values(), res["wqi_after"]],
        increasing=dict(marker=dict(color="#34f5c5")), decreasing=dict(marker=dict(color="#ff5c7a")),
        totals=dict(marker=dict(color="#8ab4ff")), connector=dict(line=dict(color="rgba(255,255,255,.3)")),
        texttemplate="%{y:.1f}", textposition="outside"))
    wf.update_yaxes(title_text="WQI", range=[0, 105])
    st.markdown("#### Where the improvement comes from")
    ui.show(ui.style_fig(wf, 380, legend=False))
    st.caption("Levers are applied cumulatively in the order shown, so contributions add up exactly to the total gain.")

    if project:
        bundle = ctx["get_bundle"](station)
        traj = apply_policy_to_forecast(forecast(bundle, df, 30), levers)
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=traj["Date"], y=traj["WQI_Baseline"], name="Baseline forecast", line=dict(color="#ff9f5a", width=3, dash="dash")))
        f2.add_trace(go.Scatter(x=traj["Date"], y=traj["WQI_Policy"], name="With policy", line=dict(color="#34f5c5", width=3.5),
                                fill="tonexty", fillcolor=ui.hex_to_rgba("#34f5c5", .12)))
        f2.update_yaxes(title_text="WQI", range=[0, 100])
        st.markdown("#### 30-day trajectory: business-as-usual vs policy")
        ui.show(ui.style_fig(f2, 360))
