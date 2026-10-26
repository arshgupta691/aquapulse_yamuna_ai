"""View 4: dataset source manager (hot-swap), model diagnostics, correlation heatmap and exports."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from backend.data_generator import PARAMS, REAL_PATH, remove_real_dataset, save_uploaded_real
from backend.lstm_engine import ALL_TARGETS, forecast, training_history_frame

from . import styles as ui


def _clear_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


def render_source_manager(meta: dict, key: str = "main", detailed: bool = True) -> None:
    """Dataset Source Manager: shows active mode + drag-and-drop hot-swap. Used in sidebar and Diagnostics."""
    color = "#34f5c5" if meta["is_real"] else "#ffd166"
    st.markdown(ui._flat(f'<div class="glass" style="padding:.8rem 1rem"><div class="kpi-label">Active dataset</div>'
                         f'<div style="margin:.35rem 0">{ui.badge_html(meta["mode"], color)}</div>'
                         f'<div class="muted">{meta["source"]}<br>{meta["rows"]:,} rows · {len(meta["stations"])} stations<br>'
                         f'{meta["date_min"]:%d %b %Y} → {meta["date_max"]:%d %b %Y}</div></div>'), unsafe_allow_html=True)

    up = st.file_uploader("Drop real CPCB CSV to hot-swap", type=["csv"], key=f"uploader_{key}")
    if up is not None:
        uid = f"{up.name}-{up.size}"
        if st.session_state.get("_last_upload") != uid:
            st.session_state["_last_upload"] = uid
            try:
                rep = save_uploaded_real(up)
                st.session_state["_upload_msg"] = ("success", f"Activated **{up.name}**: {rep['rows_out']:,} clean rows across {len(rep['stations'])} station(s).")
                _clear_caches()
                st.rerun()
            except ValueError as exc:
                st.session_state["_upload_msg"] = ("error", f"Could not use {up.name}: {exc}")
    msg = st.session_state.get("_upload_msg")
    if msg:
        (st.success if msg[0] == "success" else st.error)(msg[1])
    for w in meta["warnings"]:
        st.warning(w)

    if REAL_PATH.exists() and st.button("↩ Revert to synthetic baseline", key=f"revert_{key}", **ui.STRETCH):
        remove_real_dataset()
        st.session_state.pop("_last_upload", None)
        st.session_state["_upload_msg"] = ("success", "Reverted to the Synthetic CPCB Baseline.")
        _clear_caches()
        st.rerun()

    if detailed and meta["report"]:
        rep = meta["report"]
        with st.expander("Sanitizer report (real dataset)", expanded=True):
            c = st.columns(4)
            c[0].metric("Rows in → out", f"{rep['rows_in']:,} → {rep['rows_out']:,}")
            c[1].metric("BDL converted", rep["bdl_converted"])
            c[2].metric("Cells interpolated", f"{rep['cells_interpolated']:,}")
            c[3].metric("Out-of-range removed", rep["out_of_range_nullified"])
            st.caption(f"WQI: {rep['wqi_source']} · invalid dates dropped: {rep['invalid_dates_dropped']} · "
                       f"missing parameters: {', '.join(rep['missing_parameters']) or 'none'}")
            st.json(rep["column_map"], expanded=False)


def render(ctx: dict) -> None:
    df, meta = ctx["df"], ctx["meta"]
    ui.section_header("Model Diagnostics", "Dataset hot-swap · LSTM validation · statistical health · exports", "🩺")
    t1, t2, t3, t4 = st.tabs(["🔁 Dataset Source", "🎯 Model Performance", "🔗 Correlation & Health", "📦 Export"])

    with t1:
        render_source_manager(meta, key="diag", detailed=True)
        st.markdown("#### Data health by station")
        health = (df.groupby("Station").agg(Records=("Date", "count"), From=("Date", "min"), To=("Date", "max"),
                                            WQI_mean=("WQI", "mean"), BOD_mean=("BOD_mgL", "mean"), DO_mean=("DO_mgL", "mean"))
                  .reindex(meta["stations"]).reset_index())
        health[["WQI_mean", "BOD_mean", "DO_mean"]] = health[["WQI_mean", "BOD_mean", "DO_mean"]].round(2)
        st.dataframe(health, hide_index=True, **ui.STRETCH)

    with t2:
        station = st.selectbox("Station model", meta["stations"], key="dg_station")
        b = ctx["get_bundle"](station)
        k = st.columns(5)
        k[0].metric("Engine", b.model_kind)
        k[1].metric("WQI MAE", f"{b.metrics['WQI']['MAE']:.2f}")
        k[2].metric("WQI RMSE", f"{b.metrics['WQI']['RMSE']:.2f}")
        k[3].metric("WQI R²", f"{b.metrics['WQI']['R2']:.3f}")
        k[4].metric("vs persistence", f"{b.metrics['baseline']['skill_vs_persistence_pct']:+.1f}%",
                    help="Skill relative to the naive 'tomorrow = today' baseline. Positive = model adds value.")
        st.caption(f"Train windows: {b.n_train:,} · hold-out windows: {b.n_test:,} · training ends {b.train_end_date:%d %b %Y}")

        tf = b.test_frame
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tf["Date"], y=tf["Actual_WQI"], name="Actual", line=dict(color="#ffd166", width=2.5)))
        fig.add_trace(go.Scatter(x=tf["Date"], y=tf["Pred_WQI"], name="LSTM (1-day ahead)", line=dict(color="#5eead4", width=2.5, dash="dash")))
        fig.update_yaxes(title_text="WQI")
        st.markdown("#### Hold-out validation (unseen period)")
        ui.show(ui.style_fig(fig, 360))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Training loss per epoch")
            hist = training_history_frame(b)
            if hist.empty:
                st.info("Loss curves are available when TensorFlow is installed and the LSTM engine is active "
                        f"(current engine: {b.model_kind}).")
            else:
                f2 = go.Figure()
                f2.add_trace(go.Scatter(x=hist["Epoch"], y=hist["loss"], name="Train loss", line=dict(color="#8ab4ff", width=3)))
                if "val_loss" in hist:
                    f2.add_trace(go.Scatter(x=hist["Epoch"], y=hist["val_loss"], name="Validation loss", line=dict(color="#ff5c7a", width=3)))
                f2.update_yaxes(title_text="MSE (scaled)", type="log")
                f2.update_xaxes(title_text="Epoch")
                ui.show(ui.style_fig(f2, 340))
        with c2:
            st.markdown("#### Per-parameter error")
            rows = [{"Target": t, "MAE": round(b.metrics[t]["MAE"], 3), "RMSE": round(b.metrics[t]["RMSE"], 3),
                     "R²": round(b.metrics[t]["R2"], 3)} for t in ALL_TARGETS]
            st.dataframe(rows, hide_index=True, **ui.STRETCH)

    with t3:
        scope = st.selectbox("Scope", ["All stations", *meta["stations"]], key="dg_scope")
        sub = df if scope == "All stations" else df[df["Station"] == scope]
        corr = sub[[*PARAMS, "WQI"]].corr().round(2)
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, text=corr.values, texttemplate="%{text}",
                                   colorscale=[[0, "#ff5c7a"], [.5, "rgba(20,28,48,1)"], [1, "#5eead4"]], colorbar=dict(title="r")))
        fig.update_yaxes(autorange="reversed")
        st.markdown("#### Parameter correlation matrix")
        ui.show(ui.style_fig(fig, 480, legend=False))
        st.caption("Expect strong BOD↔Ammonia coupling and BOD↔DO anti-correlation: the signature of organic loading.")
        box = go.Figure()
        for s in meta["stations"]:
            box.add_trace(go.Box(y=df.loc[df["Station"] == s, "WQI"], name=s, boxmean=True))
        box.update_yaxes(title_text="WQI")
        ui.show(ui.style_fig(box, 340, legend=False))

    with t4:
        station = st.selectbox("Station for model exports", meta["stations"], key="dg_export_station")
        b = ctx["get_bundle"](station)
        fc = forecast(b, df, 30)
        c = st.columns(3)
        c[0].download_button("⬇ Active dataset (CSV)", df.to_csv(index=False).encode(), "aquapulse_active_dataset.csv", "text/csv", **ui.STRETCH)
        c[1].download_button("⬇ 30-day forecast (CSV)", fc.to_csv(index=False).encode(), f"forecast_{station.replace(' ', '_')}.csv", "text/csv", **ui.STRETCH)
        c[2].download_button("⬇ Hold-out predictions (CSV)", b.test_frame.to_csv(index=False).encode(), f"holdout_{station.replace(' ', '_')}.csv", "text/csv", **ui.STRETCH)
        st.caption(f"Exports use the active source: {meta['mode']}.")
