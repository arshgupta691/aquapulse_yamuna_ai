"""
AquaPulse backend :: data_generator.py
======================================
Hot-swap data pipeline for the Yamuna corridor.

Priority order used by ``load_dataset``:
    1. A CSV uploaded through the UI (validated + sanitized in memory)
    2. data/yamuna_cpcb_real.csv           -> "Verified CPCB Real Dataset"
    3. data/yamuna_cpcb_synthetic.csv      -> "Synthetic CPCB Baseline"
       (auto-generated on first run if missing)

Every source is normalised to the same canonical schema:
    Date, Station, pH, DO_mgL, BOD_mgL, Ammonia_mgL, Water_Temp_C, Conductivity, WQI
so the LSTM engine and all UI views never care which source is active.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SYNTHETIC_PATH = DATA_DIR / "yamuna_cpcb_synthetic.csv"
REAL_PATH = DATA_DIR / "yamuna_cpcb_real.csv"

MODE_SYNTHETIC = "Synthetic CPCB Baseline"
MODE_REAL = "Verified CPCB Real Dataset"

PARAMS = ["pH", "DO_mgL", "BOD_mgL", "Ammonia_mgL", "Water_Temp_C", "Conductivity"]
CANONICAL_COLUMNS = ["Date", "Station", *PARAMS, "WQI"]

# Values outside these ranges are physically implausible -> treated as sensor faults (NaN)
PHYSICAL_RANGES = {
    "pH": (0.0, 14.0),
    "DO_mgL": (0.0, 20.0),
    "BOD_mgL": (0.0, 500.0),
    "Ammonia_mgL": (0.0, 100.0),
    "Water_Temp_C": (0.0, 45.0),
    "Conductivity": (0.0, 20000.0),
    "WQI": (0.0, 100.0),
}

# "BDL" (Below Detection Limit) -> conventional half-detection-limit substitution
BDL_VALUES = {
    "pH": np.nan,
    "DO_mgL": 0.1,
    "BOD_mgL": 1.0,
    "Ammonia_mgL": 0.05,
    "Water_Temp_C": np.nan,
    "Conductivity": 10.0,
    "WQI": np.nan,
}
# Last-resort defaults if a parameter is missing from the *entire* real file
DEFAULT_FALLBACKS = {"pH": 7.5, "DO_mgL": 5.0, "BOD_mgL": 5.0,
                     "Ammonia_mgL": 0.5, "Water_Temp_C": 25.0, "Conductivity": 800.0}

DO_BOD_TAU = 12.0  # e-folding BOD scale linking BOD -> equilibrium DO (shared with simulator)

# Delhi stretch of the Yamuna, upstream -> downstream (km values are approximate)
STATION_META: dict[str, dict[str, Any]] = {
    "Palla":             dict(order=0, km=0.0,  bod=2.5,  nh3=0.30, ph=7.90, cond=420,  spike_p=0.002, role="Delhi entry point (upstream reference)"),
    "Wazirabad":         dict(order=1, km=25.0, bod=4.0,  nh3=0.80, ph=7.75, cond=470,  spike_p=0.003, role="Water-supply intake barrage"),
    "ISBT Bridge":       dict(order=2, km=31.0, bod=22.0, nh3=8.0,  ph=7.45, cond=900,  spike_p=0.0045, role="Post Najafgarh-drain confluence"),
    "Nizamuddin Bridge": dict(order=3, km=40.0, bod=32.0, nh3=12.0, ph=7.30, cond=1050, spike_p=0.0045, role="Dense urban discharge zone"),
    "Okhla Barrage":     dict(order=4, km=46.0, bod=38.0, nh3=15.0, ph=7.20, cond=1150, spike_p=0.0045, role="Delhi exit barrage (worst case)"),
    "Asgarpur":          dict(order=5, km=56.0, bod=42.0, nh3=17.0, ph=7.15, cond=1200, spike_p=0.0045, role="Downstream of Delhi"),
}


def order_stations(names: Iterable[str]) -> list[str]:
    """Sort station names upstream -> downstream; unknown stations go last (A-Z)."""
    names = list(dict.fromkeys(names))
    known = sorted((n for n in names if n in STATION_META), key=lambda n: STATION_META[n]["order"])
    other = sorted(n for n in names if n not in STATION_META)
    return known + other


# --------------------------------------------------------------------------- #
# Water-quality maths (single source of truth, reused by LSTM + What-If lab)
# --------------------------------------------------------------------------- #
# Sub-index curves (higher = better, 0-100), shaped around CPCB designated-use limits
_WQI_CURVES = {
    "pH":          ([4, 5.5, 6.5, 7.0, 8.5, 9.0, 10, 11], [0, 30, 80, 100, 100, 80, 30, 0]),
    "DO_mgL":      ([0, 2, 4, 5, 6, 8],                   [0, 15, 40, 65, 85, 100]),
    "BOD_mgL":     ([0, 2, 3, 6, 15, 30],                 [100, 90, 70, 40, 10, 0]),
    "Ammonia_mgL": ([0, 0.5, 1.2, 3, 10, 25],             [100, 90, 70, 40, 10, 0]),
}
_WQI_WEIGHTS = {"DO_mgL": 0.30, "BOD_mgL": 0.30, "Ammonia_mgL": 0.25, "pH": 0.15}

WQI_BANDS = [(80, "Excellent"), (60, "Good"), (40, "Moderate"), (20, "Poor"), (0, "Critical")]


def compute_wqi(pH, DO_mgL, BOD_mgL, Ammonia_mgL):
    """Weighted-arithmetic WQI on a 0-100 scale (100 = pristine). Scalar or array-like."""
    inputs = {"pH": pH, "DO_mgL": DO_mgL, "BOD_mgL": BOD_mgL, "Ammonia_mgL": Ammonia_mgL}
    total = 0.0
    for name, val in inputs.items():
        xp, fp = _WQI_CURVES[name]
        total = total + _WQI_WEIGHTS[name] * np.interp(np.asarray(val, dtype=float), xp, fp)
    out = np.round(total, 2)
    return float(out) if np.ndim(out) == 0 else out


def wqi_category(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "Unknown"
    for threshold, label in WQI_BANDS:
        if value >= threshold:
            return label
    return "Critical"


def do_saturation(temp_c):
    """Dissolved-oxygen saturation (mg/L) vs water temperature (Benson-Krause style fit)."""
    t = np.asarray(temp_c, dtype=float)
    out = 14.652 - 0.41022 * t + 0.007991 * t ** 2 - 0.000077774 * t ** 3
    return float(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------- #
# Synthetic CPCB-style generator (mid-term demo baseline)
# --------------------------------------------------------------------------- #
def generate_synthetic_dataset(start: str = "2021-01-01", end: str = "2024-12-31",
                               seed: int = 42) -> pd.DataFrame:
    """Domain-faithful daily data: monsoon dilution, lean-season peaks, drain-driven
    downstream escalation, BOD->DO coupling, AR(1) noise and industrial spike events."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    doy = dates.dayofyear.to_numpy().astype(float)

    dilution = np.exp(-0.5 * ((doy - 228.0) / 32.0) ** 2)               # monsoon peak ~mid-Aug
    lean = 0.5 * (1.0 + np.cos(2 * np.pi * (doy - 135.0) / 365.0))      # lean flow peak ~mid-May
    seasonal = np.clip(1.0 + 0.30 * lean - 0.55 * dilution, 0.25, None)

    frames = []
    for name, m in STATION_META.items():
        temp = np.clip(21.0 + 9.5 * np.sin(2 * np.pi * (doy - 80.0) / 365.0) - 1.5 * dilution
                       + 0.15 * m["order"] + rng.normal(0, 0.8, n), 4, 36)

        x = np.zeros(n)
        eps = rng.normal(0, 0.08, n)
        for t in range(1, n):                                            # AR(1) persistence
            x[t] = 0.9 * x[t - 1] + eps[t]

        spike = np.zeros(n)                                              # decaying discharge events
        level, draws, mags = 0.0, rng.random(n), rng.uniform(0.5, 1.4, n)
        for t in range(n):
            level *= 0.72
            if draws[t] < m["spike_p"]:
                level += mags[t]
            spike[t] = level

        bod = np.clip(m["bod"] * seasonal * np.exp(x) * (1.0 + spike), 0.5, 90.0)
        nh3 = np.clip(bod * (m["nh3"] / m["bod"]) * np.exp(rng.normal(0, 0.10, n)), 0.02, 40.0)
        dsat = do_saturation(temp)
        do_eq = dsat * (0.04 + 0.96 * np.exp(-bod / DO_BOD_TAU))
        do = np.clip(do_eq + rng.normal(0, 1, n) * (0.15 + 0.03 * do_eq), 0.0, dsat)
        ph = np.clip(m["ph"] + 0.18 * np.sin(2 * np.pi * (doy - 60.0) / 365.0)
                     - 0.10 * dilution + rng.normal(0, 0.10, n), 6.4, 9.0)
        cond = m["cond"] * (1 - 0.35 * dilution) * (1 + 0.10 * lean) * np.exp(rng.normal(0, 0.04, n))

        frames.append(pd.DataFrame({
            "Date": dates, "Station": name,
            "pH": ph.round(2), "DO_mgL": do.round(2), "BOD_mgL": bod.round(2),
            "Ammonia_mgL": nh3.round(3), "Water_Temp_C": temp.round(1), "Conductivity": cond.round(0),
        }))

    df = pd.concat(frames, ignore_index=True)
    df["WQI"] = compute_wqi(df["pH"], df["DO_mgL"], df["BOD_mgL"], df["Ammonia_mgL"])
    df["Station"] = pd.Categorical(df["Station"], categories=list(STATION_META), ordered=True)
    df = df.sort_values(["Date", "Station"]).reset_index(drop=True)
    df["Station"] = df["Station"].astype(str)
    return df[CANONICAL_COLUMNS]


def ensure_synthetic_csv(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force or not SYNTHETIC_PATH.exists():
        generate_synthetic_dataset().to_csv(SYNTHETIC_PATH, index=False)
    return SYNTHETIC_PATH


# --------------------------------------------------------------------------- #
# Real CPCB / DPCC sanitizer & schema mapper
# --------------------------------------------------------------------------- #
# (canonical name, regex on the *normalised* header: lowercase, alphanumerics only)
_COLUMN_RULES = [
    ("Date",         r"^date|date$|^timestamp$|^day$"),
    ("Station",      r"station|location|site|samplingpoint|monitoringpoint"),
    ("pH",           r"^ph(value|level|units?)?$"),
    ("DO_mgL",       r"^do($|mg)|dissolvedoxygen"),
    ("BOD_mgL",      r"^bod|biochemicaloxygen"),
    ("Ammonia_mgL",  r"ammonia|^nh3|^nh4"),
    ("Water_Temp_C", r"^(water)?temp"),
    ("Conductivity", r"^cond|^ec($|\w*cm)|conductivity"),
    ("WQI",          r"^wqi|waterqualityindex"),
]

_STATION_ALIASES = [
    (r"palla", "Palla"),
    (r"wazirabad|wazirabd", "Wazirabad"),
    (r"isbt|kashmer|kashmir", "ISBT Bridge"),
    (r"nizam", "Nizamuddin Bridge"),
    (r"okhla", "Okhla Barrage"),
    (r"asgar", "Asgarpur"),
]

_NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "—", "ns", "nr", "nm",
              "not available", "not measured", "na.", "#n/a"}
_BDL_TOKENS = {"bdl", "below detection limit", "below detectable limit", "nd", "not detected", "<dl", "trace"}


def _norm(col: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


def _is_bdl(v: Any) -> bool:
    s = str(v).strip().lower()
    return s in _BDL_TOKENS or s.startswith("bdl")


def _parse_value(v: Any, bdl_value: float) -> float:
    """Robust string->float: handles BDL, NA, '<2', '1,234', '7.8 (approx)' etc."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip().lower().replace(",", "")
    if s in _NA_TOKENS:
        return np.nan
    if _is_bdl(s):
        return bdl_value
    if s == "nil":
        return 0.0
    m = re.match(r"^([<>]=?)\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        val = float(m.group(2))
        return val / 2.0 if m.group(1).startswith("<") else val
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan


def _parse_dates(series: pd.Series) -> pd.Series:
    """ISO strings are parsed year-first; everything else day-first (Indian DD-MM-YYYY convention)."""
    txt = series.astype(str).str.strip()
    iso = txt.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
    parts = []
    for mask, dayfirst in ((iso, False), (~iso, True)):
        if mask.any():
            parts.append(pd.to_datetime(txt[mask], errors="coerce", dayfirst=dayfirst, format="mixed"))
    return pd.concat(parts).reindex(txt.index)


def _map_columns(columns: list[str]) -> dict[str, str]:
    """canonical name -> original column name (first match wins; Station prefers '...name')."""
    norm = {c: _norm(c) for c in columns}
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for canon, pattern in _COLUMN_RULES:
        cands = [c for c in columns if c not in used and re.search(pattern, norm[c])]
        if canon == "Station":
            cands.sort(key=lambda c: "name" not in norm[c])
        if cands:
            mapping[canon] = cands[0]
            used.add(cands[0])
    return mapping


def _standardize_station(name: Any) -> str:
    s = re.sub(r"\s+", " ", str(name)).strip()
    if not s or s.lower() == "nan":
        return "Unknown Station"
    low = s.lower()
    for pattern, canon in _STATION_ALIASES:
        if re.search(pattern, low):
            return canon
    return s.title()


def sanitize_real_cpcb(raw: pd.DataFrame, *, max_gap_days: int = 90,
                       prefer_reported_wqi: bool = False) -> tuple[pd.DataFrame, dict]:
    """Map + clean a raw CPCB/DPCC export into the canonical schema.

    Steps: header mapping -> BDL/NA parsing -> date parsing (day-first) -> physical range
    checks -> station standardisation -> per-station daily regularisation with
    time-based interpolation -> WQI (recomputed by default so it matches the What-If maths).
    Returns (clean_df, report).
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    report: dict[str, Any] = {"rows_in": int(len(df))}

    mapping = _map_columns(list(df.columns))
    report["column_map"] = {orig: canon for canon, orig in mapping.items()}

    # ---- dates (supports Year+Month exports as a fallback) ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "Date" in mapping:
            dates = _parse_dates(df[mapping["Date"]])
        else:
            lowered = {_norm(c): c for c in df.columns}
            if "year" in lowered and "month" in lowered:
                combo = df[lowered["year"]].astype(str) + " " + df[lowered["month"]].astype(str) + " 1"
                dates = pd.to_datetime(combo, errors="coerce")
            else:
                raise ValueError("No date column found (expected e.g. 'Date', 'Sampling Date' or Year+Month).")

    present = [p for p in PARAMS if p in mapping]
    if len(present) < 2:
        raise ValueError(f"Could not recognise enough water-quality columns (found: {present or 'none'}).")
    report["missing_parameters"] = [p for p in PARAMS if p not in mapping]

    out = pd.DataFrame({"Date": dates})
    out["Station"] = (df[mapping["Station"]].map(_standardize_station)
                      if "Station" in mapping else "Yamuna Station")

    bdl_count = 0
    for p in PARAMS + ["WQI"]:
        if p in mapping:
            series = df[mapping[p]]
            bdl_count += int(series.map(_is_bdl).sum())
            out[p] = series.map(lambda v, b=BDL_VALUES.get(p, np.nan): _parse_value(v, b))
        else:
            out[p] = np.nan
    report["bdl_converted"] = bdl_count

    # ---- physical plausibility ----
    nullified = 0
    for p, (lo, hi) in PHYSICAL_RANGES.items():
        bad = out[p].notna() & ((out[p] < lo) | (out[p] > hi))
        nullified += int(bad.sum())
        out.loc[bad, p] = np.nan
    report["out_of_range_nullified"] = nullified

    report["invalid_dates_dropped"] = int(out["Date"].isna().sum())
    out = out.dropna(subset=["Date"])
    if out.empty:
        raise ValueError("No rows with a valid date remained after parsing.")
    out["Date"] = out["Date"].dt.normalize()

    # ---- per-station daily regularisation + time interpolation ----
    cols = PARAMS + ["WQI"]
    pieces, interpolated = [], 0
    for station, g in out.groupby("Station", sort=False):
        g = g.groupby("Date")[cols].mean()                      # merge duplicate sample dates
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
        interpolated += int(g[PARAMS].isna().sum().sum())
        g[cols] = g[cols].interpolate(method="time", limit=max_gap_days, limit_direction="both")
        g[cols] = g[cols].fillna(g[cols].median())              # very long gaps -> station median
        g.index.name = "Date"
        g = g.reset_index()
        g.insert(1, "Station", station)
        pieces.append(g)
    clean = pd.concat(pieces, ignore_index=True)
    report["cells_interpolated"] = interpolated

    for p in PARAMS:                                            # parameter absent for a station/file
        if clean[p].isna().any():
            fill = clean[p].median() if clean[p].notna().any() else DEFAULT_FALLBACKS[p]
            clean[p] = clean[p].fillna(fill)

    computed = pd.Series(compute_wqi(clean["pH"], clean["DO_mgL"], clean["BOD_mgL"], clean["Ammonia_mgL"]),
                         index=clean.index)
    if prefer_reported_wqi and "WQI" in mapping and clean["WQI"].notna().mean() > 0.5:
        clean["WQI"] = clean["WQI"].fillna(computed).clip(0, 100)
        report["wqi_source"] = "reported (gaps filled with computed)"
    else:
        clean["WQI"] = computed
        report["wqi_source"] = "recomputed from parameters"

    clean["Station"] = pd.Categorical(clean["Station"], categories=order_stations(clean["Station"].unique()), ordered=True)
    clean = clean.sort_values(["Date", "Station"]).reset_index(drop=True)
    clean["Station"] = clean["Station"].astype(str)
    clean = clean[CANONICAL_COLUMNS].round({"pH": 2, "DO_mgL": 2, "BOD_mgL": 2, "Ammonia_mgL": 3,
                                            "Water_Temp_C": 1, "Conductivity": 0, "WQI": 2})

    report["rows_out"] = int(len(clean))
    report["stations"] = order_stations(clean["Station"].unique())
    report["date_range"] = (clean["Date"].min().date().isoformat(), clean["Date"].max().date().isoformat())
    return clean, report


# --------------------------------------------------------------------------- #
# Hot-swap loader
# --------------------------------------------------------------------------- #
def _read_csv_any(source) -> pd.DataFrame:
    """Read a path or Streamlit UploadedFile as all-strings (so 'BDL'/'NA' survive)."""
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, dtype=str, keep_default_na=False, sep=None,
                               engine="python", encoding=enc)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise ValueError(f"Unable to read CSV: {last_err}")


def _build_meta(mode: str, source: str, df: pd.DataFrame, report: dict | None,
                warns: list[str]) -> dict:
    return {
        "mode": mode,
        "is_real": mode == MODE_REAL,
        "source": source,
        "rows": int(len(df)),
        "stations": order_stations(df["Station"].unique()),
        "date_min": df["Date"].min(),
        "date_max": df["Date"].max(),
        "report": report or {},
        "warnings": warns,
    }


def load_dataset(uploaded_file=None) -> tuple[pd.DataFrame, dict]:
    """Priority loader. Never raises for bad *real* data: it falls back to the synthetic
    baseline and records why in meta['warnings'] so the UI can surface it."""
    warns: list[str] = []

    if uploaded_file is not None:
        try:
            df, rep = sanitize_real_cpcb(_read_csv_any(uploaded_file))
            return df, _build_meta(MODE_REAL, f"Uploaded: {getattr(uploaded_file, 'name', 'file.csv')}", df, rep, warns)
        except Exception as exc:  # noqa: BLE001
            warns.append(f"Uploaded file rejected ({exc}). Falling back.")

    if REAL_PATH.exists():
        try:
            df, rep = sanitize_real_cpcb(_read_csv_any(str(REAL_PATH)))
            return df, _build_meta(MODE_REAL, f"data/{REAL_PATH.name}", df, rep, warns)
        except Exception as exc:  # noqa: BLE001
            warns.append(f"data/{REAL_PATH.name} could not be processed ({exc}). Using synthetic baseline.")

    ensure_synthetic_csv()
    try:
        df = pd.read_csv(SYNTHETIC_PATH, parse_dates=["Date"])
        if list(df.columns) != CANONICAL_COLUMNS:
            raise ValueError("schema mismatch")
    except Exception:  # noqa: BLE001  (corrupted/old file -> rebuild)
        ensure_synthetic_csv(force=True)
        df = pd.read_csv(SYNTHETIC_PATH, parse_dates=["Date"])
    return df, _build_meta(MODE_SYNTHETIC, f"data/{SYNTHETIC_PATH.name}", df, None, warns)


def save_uploaded_real(uploaded_file) -> dict:
    """Validate an uploaded CSV, then persist it as data/yamuna_cpcb_real.csv (the hot swap).
    Raises ValueError if the file can't be mapped. Returns the sanitizer report."""
    _, report = sanitize_real_cpcb(_read_csv_any(uploaded_file))
    uploaded_file.seek(0)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REAL_PATH.write_bytes(uploaded_file.read())
    return report


def remove_real_dataset() -> bool:
    """Revert to the synthetic baseline. True if a real file was removed."""
    if REAL_PATH.exists():
        REAL_PATH.unlink()
        return True
    return False


def get_data_signature() -> tuple[float, float]:
    """Cheap fingerprint (file mtimes) - use as a cache key so Streamlit refreshes on hot-swap."""
    mt = lambda p: p.stat().st_mtime if p.exists() else 0.0  # noqa: E731
    return mt(REAL_PATH), mt(SYNTHETIC_PATH)


# --------------------------------------------------------------------------- #
# Small helpers for the UI layer
# --------------------------------------------------------------------------- #
def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Most recent reading per station, ordered upstream -> downstream."""
    snap = df.loc[df.groupby("Station")["Date"].idxmax()].copy()
    ordered = order_stations(snap["Station"].tolist())
    return snap.set_index("Station").loc[ordered].reset_index()


def get_station_frame(df: pd.DataFrame, station: str) -> pd.DataFrame:
    return df[df["Station"] == station].sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":  # python -m backend.data_generator
    path = ensure_synthetic_csv(force=True)
    print(f"Synthetic CPCB baseline written to {path}")
