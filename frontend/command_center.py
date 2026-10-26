"""View 1: live station KPIs + Yamuna corridor escalation analytics."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backend.data_generator import STATION_META, latest_snapshot, order_stations, wqi_category

from . import styles as ui


def _snapshot(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    sub = df[df["Date"] <= ts]
    return latest_snapshot(sub) if not sub.empty else sub


def render(ctx: dict) -> None:
    df, meta = ctx["df"], ctx["meta"]
    ui.section_header("Command Center", "Live Yamuna corridor intelligence · Palla → Asgarpur", "🛰️")

    dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
    c_top = st.columns([3, 1.2])
    with c_top[1]:
        snap_date = st.date_input("Snapshot date", value=dmax, min_value=dmin, max_value=dmax, key="cc_date")
    if isinstance(snap_date, (list, tuple)):
        snap_date = snap_date[0] if snap_date else dmax
    with c_top[0]:
        color = "#34f5c5" if meta["is_real"] else "#ffd166"
        st.markdown(f'Data source: {ui.badge_html(meta["mode"], color)} <span class="muted">&nbsp;{meta["source"]} · '
                    f'{meta["rows"]:,} records · {dmin:%d %b %Y} → {dmax:%d %b %Y}</span>', unsafe_allow_html=True)

    ts = pd.Timestamp(snap_date)
    snap = _snapshot(df, ts)
    if snap.empty:
        st.warning("No data on or before the selected date.")
        return
    prev = _snapshot(df, ts - pd.Timedelta(days=7))
    prev_map = prev.set_index("Station")["WQI"] if not prev.empty else pd.Series(dtype=float)
    snap["dWQI"] = snap["WQI"] - snap["Station"].map(prev_map)

    # ---- headline KPIs ----
    avg = snap["WQI"].mean()
    best, worst = snap.loc[snap["WQI"].idxmax()], snap.loc[snap["WQI"].idxmin()]
    n_bad = int((snap["WQI"] < 40).sum())
    k = st.columns(4)
    with k[0]:
        ui.kpi_card("Corridor Avg WQI", f"{avg:.1f}", f"{wqi_category(avg)} · week-on-week",
                    ui.category_color(wqi_category(avg)), delta=snap["dWQI"].mean())
    with k[1]:
        ui.kpi_card("Cleanest Station", best["Station"], f"WQI {best['WQI']:.1f} · BOD {best['BOD_mgL']:.1f} mg/L",
                    ui.category_color(wqi_category(best["WQI"])))
    with k[2]:
        ui.kpi_card("Most Polluted", worst["Station"], f"WQI {worst['WQI']:.1f} · DO {worst['DO_mgL']:.2f} mg/L",
                    ui.category_color(wqi_category(worst["WQI"])))
    with k[3]:
        ui.kpi_card("Stations Poor / Critical", f"{n_bad}<small> / {len(snap)}</small>", "WQI below 40 (CPCB-aligned bands)",
                    "#ff5c7a" if n_bad else "#34f5c5")

    # ---- station cards ----
    st.markdown("#### Live station readings")
    rows = [snap.iloc[i:i + 3] for i in range(0, len(snap), 3)]
    for chunk in rows:
        cols = st.columns(3)
        for col, (_, r) in zip(cols, chunk.iterrows()):
            cat = wqi_category(r["WQI"])
            role = STATION_META.get(r["Station"], {}).get("role", "Monitoring station")
            with col:
                ui.kpi_card(f"{r['Station']}", f"{r['WQI']:.1f}",
                            f"{ui.badge_html(cat, ui.category_color(cat))}<br>BOD {r['BOD_mgL']:.1f} · DO {r['DO_mgL']:.2f} · "
                            f"NH₃ {r['Ammonia_mgL']:.2f} mg/L<br><span class='muted'>{role}</span>",
                            ui.category_color(cat), delta=r["dWQI"])

    # ---- corridor escalation chart ----
    st.markdown("#### Corridor escalation: how pollution compounds downstream")
    order = snap["Station"].tolist()
    km = [STATION_META.get(s, {}).get("km") for s in order]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=order, y=snap["WQI"], name="WQI", opacity=.85, customdata=km,
                         marker=dict(color=[ui.category_color(wqi_category(v)) for v in snap["WQI"]],
                                     line=dict(color="rgba(255,255,255,.4)", width=1)),
                         text=snap["WQI"].round(1), textposition="outside",
                         hovertemplate="%{x}<br>WQI %{y:.1f}<br>~%{customdata} km from Palla<extra></extra>"),
                  secondary_y=False)
    for col, name, colr in (("BOD_mgL", "BOD (mg/L)", "#ff5c7a"), ("DO_mgL", "DO (mg/L)", "#6ee7ff"),
                            ("Ammonia_mgL", "Ammonia (mg/L)", "#ffd166")):
        fig.add_trace(go.Scatter(x=order, y=snap[col], name=name, mode="lines+markers",
                                 line=dict(color=colr, width=3, shape="spline"), marker=dict(size=9)), secondary_y=True)
    fig.update_yaxes(title_text="WQI (higher = cleaner)", range=[0, 115], secondary_y=False)
    fig.update_yaxes(title_text="Concentration (mg/L)", secondary_y=True, showgrid=False)
    ui.show(ui.style_fig(fig, 440))

    # ---- trends + heatmap ----
    t1, t2 = st.tabs(["📈 Station trends", "🗓️ Seasonal heatmap"])
    with t1:
        default = [s for s in ("Palla", "Okhla Barrage") if s in meta["stations"]] or meta["stations"][:2]
        picks = st.multiselect("Stations", meta["stations"], default=default, key="cc_pick")
        window = st.select_slider("Look-back", ["90 days", "180 days", "365 days", "All"], value="365 days", key="cc_win")
        sub = df[df["Date"] <= ts]
        if window != "All":
            sub = sub[sub["Date"] > ts - pd.Timedelta(days=int(window.split()[0]))]
        fig2 = go.Figure()
        for s in order_stations(picks):
            g = sub[sub["Station"] == s]
            fig2.add_trace(go.Scatter(x=g["Date"], y=g["WQI"].rolling(7, min_periods=1).mean(), name=s, mode="lines", line=dict(width=2.5)))
        fig2.update_yaxes(title_text="WQI (7-day mean)", range=[0, 100])
        ui.show(ui.style_fig(fig2, 380))
    with t2:
        m = df[df["Date"] <= ts].assign(M=lambda d: d["Date"].dt.to_period("M").dt.to_timestamp())
        piv = m.pivot_table(index="Station", columns="M", values="WQI", aggfunc="mean")
        piv = piv.reindex(order_stations(piv.index)).iloc[:, -24:]
        fig3 = go.Figure(go.Heatmap(z=piv.values, x=[c.strftime("%b %y") for c in piv.columns], y=piv.index, zmin=0, zmax=100,
                                    colorscale=[[0, "#ff5c7a"], [.4, "#ffd166"], [.7, "#6ee7ff"], [1, "#34f5c5"]],
                                    colorbar=dict(title="WQI"), hovertemplate="%{y}<br>%{x}: %{z:.1f}<extra></extra>"))
        fig3.update_yaxes(autorange="reversed")
        ui.show(ui.style_fig(fig3, 360, legend=False))
        st.caption("Monsoon months (Jul–Sep) visibly lift downstream WQI through dilution; lean-flow months (Mar–Jun) are worst.")
