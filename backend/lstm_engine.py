"""
AquaPulse backend :: lstm_engine.py
===================================
Deep-learning forecasting + policy simulation maths.

* Min-Max scaling (fit on the training period only -> no leakage)
* 30-day sliding windows over 6 water parameters + seasonal sin/cos calendar features
* Stacked LSTM (TensorFlow/Keras, imported lazily) predicting next-day parameters;
  WQI is derived from the predicted parameters so it is always self-consistent
* Automatic Ridge fallback if TensorFlow is unavailable (demo never hard-fails)
* Recursive multi-day forecasting, historical "time machine" back-casts, spike alerts
* Policy simulator (STP / drains / industry / e-flow / aeration) for the What-If Lab
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from .data_generator import (DO_BOD_TAU, PARAMS, PHYSICAL_RANGES, compute_wqi,
                             do_saturation, wqi_category)

WINDOW = 30
N_OUT = len(PARAMS)
ALL_TARGETS = PARAMS + ["WQI"]


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def prepare_station_frame(df: pd.DataFrame, station: str) -> pd.DataFrame:
    """Continuous daily frame for one station (gaps interpolated) with a derived WQI."""
    s = df.loc[df["Station"] == station, ["Date", *PARAMS]].drop_duplicates("Date")
    if s.empty:
        raise ValueError(f"No data for station '{station}'.")
    s = s.sort_values("Date").set_index("Date").asfreq("D")
    s[PARAMS] = s[PARAMS].interpolate(method="time", limit_direction="both").ffill().bfill()
    s.index.name = "Date"
    s = s.reset_index()
    s["WQI"] = compute_wqi(s["pH"], s["DO_mgL"], s["BOD_mgL"], s["Ammonia_mgL"])
    return s


def _calendar_features(dates) -> np.ndarray:
    """Seasonality encoded as sin/cos of day-of-year, rescaled to [0, 1]."""
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy().astype(float)
    ang = 2 * np.pi * doy / 365.25
    return (np.column_stack([np.sin(ang), np.cos(ang)]) + 1.0) / 2.0


def create_sliding_windows(data: np.ndarray, window: int = WINDOW,
                           n_targets: int = N_OUT) -> tuple[np.ndarray, np.ndarray]:
    """(T, F) -> X:(T-window, window, F) and y:(T-window, n_targets) = next-day first n_targets cols."""
    view = sliding_window_view(data, window, axis=0)[:-1]          # (T-window, F, window)
    X = np.ascontiguousarray(view.transpose(0, 2, 1), dtype=np.float32)
    y = data[window:, :n_targets].astype(np.float32)
    return X, y


def _inverse(scaler: MinMaxScaler, arr: np.ndarray) -> np.ndarray:
    inv = scaler.inverse_transform(np.asarray(arr, dtype=float))
    for j, p in enumerate(PARAMS):
        lo, hi = PHYSICAL_RANGES[p]
        inv[:, j] = np.clip(inv[:, j], lo, hi)
    return inv


def _wqi_from(arr: np.ndarray) -> np.ndarray:
    return compute_wqi(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class _RidgeSeqModel:
    """Flattened-window ridge regression; same predict() contract as the Keras model."""
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X.reshape(len(X), -1), y)
        return self

    def predict(self, X):
        return self.model.predict(X.reshape(len(X), -1))


def _build_keras_model(window: int, n_in: int, n_out: int, lr: float = 1e-3):
    from tensorflow.keras import layers, models, optimizers
    model = models.Sequential([
        layers.Input(shape=(window, n_in)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_out),
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


def _predict(model, X: np.ndarray) -> np.ndarray:
    if isinstance(model, _RidgeSeqModel):
        return model.predict(X)
    if len(X) <= 4:                                   # fast path for recursive forecasting
        return np.asarray(model(X, training=False))
    return model.predict(X, batch_size=256, verbose=0)


@dataclass
class ModelBundle:
    station: str
    model: Any
    model_kind: str                    # "LSTM" or "Ridge (fallback)"
    scaler: MinMaxScaler
    window: int
    history: dict                      # Keras loss curves (empty for fallback)
    metrics: dict                      # per-target MAE / RMSE / R2 + persistence baseline
    test_frame: pd.DataFrame           # one-step-ahead actual vs predicted on hold-out period
    train_end_date: pd.Timestamp
    n_train: int
    n_test: int
    epochs_run: int
    extras: dict = field(default_factory=dict)


def _score(actual: np.ndarray, pred: np.ndarray, names: list[str]) -> dict:
    return {nm: {"MAE": float(mean_absolute_error(actual[:, j], pred[:, j])),
                 "RMSE": float(np.sqrt(mean_squared_error(actual[:, j], pred[:, j]))),
                 "R2": float(r2_score(actual[:, j], pred[:, j]))}
            for j, nm in enumerate(names)}


def train_station_model(df: pd.DataFrame, station: str, *, window: int = WINDOW,
                        epochs: int = 40, batch_size: int = 32, test_frac: float = 0.15,
                        use_lstm: bool = True, seed: int = 42,
                        progress_cb: Callable[[int, int, dict], None] | None = None) -> ModelBundle:
    """Train on the chronological head, evaluate one-step-ahead on the hold-out tail."""
    s = prepare_station_frame(df, station)
    if len(s) < window + 120:
        raise ValueError(f"Need at least {window + 120} daily records for '{station}', found {len(s)}.")

    values = s[PARAMS].to_numpy(dtype=float)
    n_split = int(len(s) * (1 - test_frac))
    scaler = MinMaxScaler().fit(values[:n_split])                  # train-only fit
    full = np.hstack([scaler.transform(values), _calendar_features(s["Date"])])
    X, y = create_sliding_windows(full, window)

    k = n_split - window                                           # windows whose target is in train
    X_tr, y_tr, X_te, y_te = X[:k], y[:k], X[k:], y[k:]

    history: dict = {}
    epochs_run = 0
    if use_lstm and tf_available():
        import tensorflow as tf
        from tensorflow.keras import callbacks
        tf.keras.utils.set_random_seed(seed)
        model = _build_keras_model(window, X.shape[2], N_OUT)
        cbs = [callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
               callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)]
        if progress_cb is not None:
            class _Progress(callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress_cb(epoch + 1, epochs, dict(logs or {}))
            cbs.append(_Progress())
        hist = model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                         validation_split=0.1, shuffle=True, callbacks=cbs, verbose=0)
        history = {k_: [float(v) for v in vals] for k_, vals in hist.history.items()}
        epochs_run, kind = len(history.get("loss", [])), "LSTM"
    else:
        model = _RidgeSeqModel().fit(X_tr, y_tr)
        kind = "Ridge (fallback)"
        if progress_cb is not None:
            progress_cb(1, 1, {})

    pred_te = _inverse(scaler, _predict(model, X_te))
    act_te = _inverse(scaler, y_te)
    pred_all = np.column_stack([pred_te, _wqi_from(pred_te)])
    act_all = np.column_stack([act_te, _wqi_from(act_te)])

    metrics = _score(act_all, pred_all, ALL_TARGETS)
    wqi_series = s["WQI"].to_numpy()
    persistence = np.abs(wqi_series[n_split:] - wqi_series[n_split - 1:-1]).mean()
    metrics["baseline"] = {
        "persistence_MAE_WQI": float(persistence),
        "skill_vs_persistence_pct": float((1 - metrics["WQI"]["MAE"] / persistence) * 100) if persistence > 0 else 0.0,
    }

    test_frame = pd.DataFrame({"Date": s["Date"].iloc[n_split:].to_numpy()})
    for j, nm in enumerate(ALL_TARGETS):
        test_frame[f"Actual_{nm}"] = act_all[:, j]
        test_frame[f"Pred_{nm}"] = pred_all[:, j]

    return ModelBundle(station=station, model=model, model_kind=kind, scaler=scaler, window=window,
                       history=history, metrics=metrics, test_frame=test_frame,
                       train_end_date=s["Date"].iloc[n_split - 1], n_train=int(k), n_test=int(len(X_te)),
                       epochs_run=epochs_run)


def training_history_frame(bundle: ModelBundle) -> pd.DataFrame:
    """Epoch loss curves ready for plotting (empty frame for the fallback model)."""
    if not bundle.history:
        return pd.DataFrame(columns=["Epoch", "loss", "val_loss"])
    out = pd.DataFrame(bundle.history)
    out.insert(0, "Epoch", np.arange(1, len(out) + 1))
    return out


# --------------------------------------------------------------------------- #
# Forecasting ("AI Time Machine")
# --------------------------------------------------------------------------- #
def forecast(bundle: ModelBundle, df: pd.DataFrame, horizon: int = 30,
             cutoff_date=None) -> pd.DataFrame:
    """Recursive multi-day forecast.

    cutoff_date=None -> forecast beyond the latest observation (true future).
    cutoff_date=<past date> -> "time machine" back-cast using only data up to that date;
    Actual_* columns then show what really happened. ``In_Sample`` flags cut-offs the
    model was trained on (so evaluators can see honest out-of-sample cases too).
    """
    full = prepare_station_frame(df, bundle.station)
    s = full if cutoff_date is None else full[full["Date"] <= pd.Timestamp(cutoff_date)]
    if len(s) < bundle.window:
        raise ValueError(f"Need at least {bundle.window} days of history before the cut-off date.")

    scaled = np.clip(bundle.scaler.transform(s[PARAMS].to_numpy(dtype=float)), -0.25, 1.25)
    win = np.hstack([scaled, _calendar_features(s["Date"])])[-bundle.window:]
    last_date = s["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    preds = []
    cal_future = _calendar_features(future_dates)
    for i in range(horizon):
        p = np.clip(_predict(bundle.model, win[None, ...].astype(np.float32))[0], -0.25, 1.25)
        preds.append(p)
        win = np.vstack([win[1:], np.concatenate([p, cal_future[i]])])

    inv = _inverse(bundle.scaler, np.asarray(preds))
    out = pd.DataFrame(inv, columns=PARAMS)
    out.insert(0, "Date", future_dates)
    out["WQI"] = _wqi_from(inv)

    rmse = bundle.metrics["WQI"]["RMSE"]                           # heuristic 90% band, widening with horizon
    sigma = rmse * (1.0 + 0.35 * np.log1p(np.arange(1, horizon + 1)))
    out["WQI_Lower"] = np.clip(out["WQI"] - 1.645 * sigma, 0, 100)
    out["WQI_Upper"] = np.clip(out["WQI"] + 1.645 * sigma, 0, 100)
    out["Horizon_Day"] = np.arange(1, horizon + 1)

    actual = full.set_index("Date")
    out["Actual_WQI"] = out["Date"].map(actual["WQI"])
    out["Actual_BOD_mgL"] = out["Date"].map(actual["BOD_mgL"])
    out["In_Sample"] = bool(last_date <= bundle.train_end_date)
    return out.round({"pH": 2, "DO_mgL": 2, "BOD_mgL": 2, "Ammonia_mgL": 3, "Water_Temp_C": 1,
                      "Conductivity": 0, "WQI": 2, "WQI_Lower": 2, "WQI_Upper": 2})


# --------------------------------------------------------------------------- #
# Automated spike alert
# --------------------------------------------------------------------------- #
ALERT_LIMITS = {"BOD_mgL": 3.0, "DO_mgL": 5.0, "Ammonia_mgL": 1.2}   # CPCB bathing-class style limits


def detect_spike_alert(forecast_df: pd.DataFrame, station_frame: pd.DataFrame,
                       recent_days: int = 7) -> dict:
    """Compare the forecast against the trailing week of observations.

    station_frame must end at the same cut-off the forecast started from
    (use prepare_station_frame(...) and slice by date for time-machine mode).
    """
    base = station_frame.tail(recent_days)
    base_wqi, base_bod, base_nh3 = base["WQI"].mean(), base["BOD_mgL"].mean(), base["Ammonia_mgL"].mean()

    worst = forecast_df.loc[forecast_df["WQI"].idxmin()]
    bod_peak = forecast_df.loc[forecast_df["BOD_mgL"].idxmax()]
    nh3_peak = forecast_df["Ammonia_mgL"].max()
    do_min = forecast_df["DO_mgL"].min()

    wqi_drop = float(base_wqi - worst["WQI"])
    bod_surge = float((bod_peak["BOD_mgL"] / max(base_bod, 0.1) - 1) * 100)
    nh3_surge = float((nh3_peak / max(base_nh3, 0.02) - 1) * 100)

    if wqi_drop >= 15 or bod_surge >= 50:
        level = "CRITICAL"
    elif wqi_drop >= 7 or bod_surge >= 25 or nh3_surge >= 40:
        level = "WARNING"
    else:
        level = "STABLE"

    breaches = {
        "BOD_mgL": int((forecast_df["BOD_mgL"] > ALERT_LIMITS["BOD_mgL"]).sum()),
        "DO_mgL": int((forecast_df["DO_mgL"] < ALERT_LIMITS["DO_mgL"]).sum()),
        "Ammonia_mgL": int((forecast_df["Ammonia_mgL"] > ALERT_LIMITS["Ammonia_mgL"]).sum()),
    }
    cat = wqi_category(float(worst["WQI"]))
    if level == "STABLE":
        headline = "No significant deterioration forecast"
        detail = (f"WQI is expected to stay near {forecast_df['WQI'].mean():.1f} "
                  f"(trailing week: {base_wqi:.1f}); BOD peaks at {bod_peak['BOD_mgL']:.1f} mg/L.")
    else:
        headline = "Predicted pollution spike" if level == "CRITICAL" else "Early-warning: water quality declining"
        detail = (f"WQI forecast to fall {wqi_drop:.1f} pts to {worst['WQI']:.1f} ({cat}) around "
                  f"{pd.Timestamp(worst['Date']):%d %b %Y}. BOD peaks at {bod_peak['BOD_mgL']:.1f} mg/L "
                  f"({bod_surge:+.0f}% vs last week), DO may dip to {do_min:.1f} mg/L.")
    return {"level": level, "headline": headline, "detail": detail, "wqi_drop": wqi_drop,
            "worst_date": pd.Timestamp(worst["Date"]), "worst_wqi": float(worst["WQI"]),
            "bod_peak": float(bod_peak["BOD_mgL"]), "bod_surge_pct": bod_surge,
            "nh3_surge_pct": nh3_surge, "do_min": float(do_min), "limit_breach_days": breaches}


# --------------------------------------------------------------------------- #
# Policy simulation ("What-If Lab")
# --------------------------------------------------------------------------- #
LEVER_SPECS: dict[str, dict[str, Any]] = {
    "stp_pct":        dict(label="Sewage Treatment Upgrade", unit="% extra sewage treated", min=0, max=100, default=0, step=5,
                           help="Additional share of domestic sewage routed through functioning STPs."),
    "drain_pct":      dict(label="Drain Interception & Diversion", unit="% of drain flow tapped", min=0, max=100, default=0, step=5,
                           help="Najafgarh / Shahdara-type drains intercepted and sent to treatment."),
    "industrial_pct": dict(label="Industrial Effluent Cut", unit="% discharge reduction", min=0, max=100, default=0, step=5,
                           help="CETP/ETP compliance and zero-liquid-discharge enforcement."),
    "flow_pct":       dict(label="Environmental Flow Release", unit="% extra fresh flow", min=0, max=200, default=0, step=10,
                           help="Additional upstream freshwater released for dilution."),
    "aeration_pct":   dict(label="In-stream Aeration & Wetlands", unit="% intensity", min=0, max=100, default=0, step=5,
                           help="Mechanical aeration and constructed-wetland polishing."),
}

POLICY_PRESETS: dict[str, dict[str, float]] = {
    "Business as Usual":  {k: 0 for k in LEVER_SPECS},
    "Moderate Action":    dict(stp_pct=40, drain_pct=30, industrial_pct=30, flow_pct=20, aeration_pct=20),
    "Aggressive Cleanup": dict(stp_pct=80, drain_pct=70, industrial_pct=70, flow_pct=60, aeration_pct=50),
    "Mission Yamuna":     dict(stp_pct=100, drain_pct=100, industrial_pct=100, flow_pct=150, aeration_pct=80),
}

# Pollution-load composition and removal efficiency per source
_BOD_SHARES = dict(sewage=0.55, drains=0.30, industrial=0.15)
_BOD_EFF = dict(sewage=0.90, drains=0.85, industrial=0.90)
_NH3_SHARES = dict(sewage=0.70, drains=0.20, industrial=0.10)
_NH3_EFF = dict(sewage=0.75, drains=0.70, industrial=0.85)       # ammonia is harder to strip than BOD


def _load_factor(shares: Mapping[str, float], eff: Mapping[str, float], s: float, d: float, i: float) -> float:
    return (shares["sewage"] * (1 - eff["sewage"] * s)
            + shares["drains"] * (1 - eff["drains"] * d)
            + shares["industrial"] * (1 - eff["industrial"] * i))


def _simulate_core(base: Mapping[str, float], lv: Mapping[str, float]) -> dict[str, float]:
    s, d, i = (lv.get(k, 0) / 100.0 for k in ("stp_pct", "drain_pct", "industrial_pct"))
    a = lv.get("aeration_pct", 0) / 100.0
    flow = 1.0 + lv.get("flow_pct", 0) / 100.0
    dil = flow ** -0.75                                           # imperfect mixing -> exponent < 1

    bod_lf = _load_factor(_BOD_SHARES, _BOD_EFF, s, d, i)
    nh3_lf = _load_factor(_NH3_SHARES, _NH3_EFF, s, d, i)
    bod1 = base["BOD_mgL"] * bod_lf * dil * (1 - 0.15 * a)        # aeration also oxidises some BOD
    nh31 = base["Ammonia_mgL"] * nh3_lf * dil

    dsat = do_saturation(base["Water_Temp_C"])
    eq = lambda b: dsat * (0.04 + 0.96 * np.exp(-b / DO_BOD_TAU))  # noqa: E731  same law as the generator
    do1 = float(np.clip(base["DO_mgL"] + eq(bod1) - eq(base["BOD_mgL"]), 0, dsat))
    do1 += a * 0.35 * (dsat - do1)                                 # aeration closes up to 35% of the DO deficit

    ph1 = base["pH"] + (7.6 - base["pH"]) * 0.35 * (1 - bod_lf)
    cond1 = base["Conductivity"] * (0.75 + 0.25 * bod_lf) * flow ** -0.35
    return {"pH": float(ph1), "DO_mgL": float(do1), "BOD_mgL": float(bod1), "Ammonia_mgL": float(nh31),
            "Water_Temp_C": float(base["Water_Temp_C"]), "Conductivity": float(cond1)}


def _wqi_of(p: Mapping[str, float]) -> float:
    return float(compute_wqi(p["pH"], p["DO_mgL"], p["BOD_mgL"], p["Ammonia_mgL"]))


def simulate_policy(baseline: Mapping[str, float], levers: Mapping[str, float]) -> dict:
    """Apply policy levers to a baseline reading and recompute WQI.

    Returns before/after parameters, WQI, category, % reductions and a per-lever
    WQI contribution breakdown. Levers are applied cumulatively in LEVER_SPECS order and each
    lever's marginal WQI gain is recorded, so the contributions sum exactly to wqi_delta
    (ideal for a waterfall chart; WQI curves are non-linear so isolated gains would not add up).
    """
    base = {k: float(baseline[k]) for k in PARAMS}
    after = _simulate_core(base, levers)
    w0, w1 = _wqi_of(base), _wqi_of(after)
    contrib, active, prev = {}, {}, w0
    for k in LEVER_SPECS:
        active[k] = levers.get(k, 0)
        cur = _wqi_of(_simulate_core(base, active))
        contrib[k], prev = cur - prev, cur
    return {
        "before": base, "after": after,
        "wqi_before": w0, "wqi_after": w1, "wqi_delta": w1 - w0,
        "category_before": wqi_category(w0), "category_after": wqi_category(w1),
        "reduction_pct": {k: (1 - after[k] / base[k]) * 100 if base[k] else 0.0 for k in ("BOD_mgL", "Ammonia_mgL")},
        "lever_contributions": contrib,
    }


def apply_policy_to_forecast(forecast_df: pd.DataFrame, levers: Mapping[str, float]) -> pd.DataFrame:
    """Re-run every forecast day through the simulator -> baseline vs policy WQI trajectories."""
    rows = [_simulate_core({p: float(r[p]) for p in PARAMS}, levers) for _, r in forecast_df.iterrows()]
    scen = pd.DataFrame(rows)
    out = forecast_df[["Date", "WQI"]].rename(columns={"WQI": "WQI_Baseline"}).reset_index(drop=True)
    out["WQI_Policy"] = compute_wqi(scen["pH"], scen["DO_mgL"], scen["BOD_mgL"], scen["Ammonia_mgL"])
    for p in ("BOD_mgL", "DO_mgL", "Ammonia_mgL"):
        out[f"{p}_Policy"] = scen[p].round(3)
    return out
