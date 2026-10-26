"""Dark Liquid Glass design system: CSS injection, HTML components, Plotly theming."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Streamlit >=1.50 replaced use_container_width with width="stretch"
try:
    _NEW_API = tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 50)
except Exception:  # noqa: BLE001
    _NEW_API = False
STRETCH: dict = {"width": "stretch"} if _NEW_API else {"use_container_width": True}

CATEGORY_COLORS = {"Excellent": "#34f5c5", "Good": "#6ee7ff", "Moderate": "#ffd166",
                   "Poor": "#ff9f5a", "Critical": "#ff5c7a", "Unknown": "#94a3b8"}
LEVEL_STYLE = {"CRITICAL": ("#ff5c7a", "🚨"), "WARNING": ("#ffb84d", "⚠️"), "STABLE": ("#34f5c5", "✅")}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; color: #dbe9ff; }
.stApp { background: #050914; }
.stApp::before { content:""; position:fixed; inset:-20%; z-index:0; pointer-events:none;
  background: radial-gradient(40% 35% at 18% 20%, rgba(45,212,191,.28), transparent 70%),
              radial-gradient(35% 40% at 82% 15%, rgba(99,102,241,.30), transparent 70%),
              radial-gradient(45% 40% at 70% 85%, rgba(14,165,233,.24), transparent 70%),
              radial-gradient(30% 30% at 10% 85%, rgba(236,72,153,.14), transparent 70%);
  animation: drift 26s ease-in-out infinite alternate; }
@keyframes drift { from { transform: translate3d(-2%,-1%,0) scale(1);} to { transform: translate3d(3%,2%,0) scale(1.08);} }
[data-testid="stAppViewContainer"], [data-testid="stSidebar"] { position: relative; z-index: 1; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
h1,h2,h3,h4 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -.01em; color:#f2f8ff; }

[data-testid="stSidebar"] { background: rgba(10,16,32,.55); backdrop-filter: blur(26px) saturate(170%);
  -webkit-backdrop-filter: blur(26px) saturate(170%); border-right: 1px solid rgba(255,255,255,.08); }
[data-testid="stSidebar"] .stRadio > div { gap: .35rem; }
[data-testid="stSidebar"] .stRadio label { background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.07);
  border-radius: 14px; padding: .55rem .8rem; width:100%; transition: all .2s; }
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(94,234,212,.12); border-color: rgba(94,234,212,.35); }
[data-testid="stSidebar"] .stRadio label:has(input:checked) { background: linear-gradient(135deg, rgba(94,234,212,.25), rgba(99,102,241,.22));
  border-color: rgba(94,234,212,.55); box-shadow: 0 0 18px rgba(94,234,212,.18); }
[data-testid="stSidebar"] .stRadio label > div:first-child { display:none; }

.glass { background: linear-gradient(135deg, rgba(255,255,255,.10), rgba(255,255,255,.03));
  border: 1px solid rgba(255,255,255,.14); border-radius: 22px; backdrop-filter: blur(20px) saturate(160%);
  -webkit-backdrop-filter: blur(20px) saturate(160%);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.22), 0 18px 44px rgba(0,0,0,.38); padding: 1.1rem 1.25rem; }
.kpi { position:relative; overflow:hidden; margin-bottom:.9rem; min-height: 128px; }
.kpi::after { content:""; position:absolute; right:-30px; top:-30px; width:120px; height:120px; border-radius:50%;
  background: radial-gradient(circle, var(--accent) 0%, transparent 70%); opacity:.28; }
.kpi-label { font-size:.74rem; text-transform:uppercase; letter-spacing:.12em; color:#9db4d6; font-weight:600; }
.kpi-value { font-family:'Space Grotesk',sans-serif; font-size:2.15rem; font-weight:700; color:#fff; line-height:1.15; margin:.25rem 0; }
.kpi-value small { font-size:.9rem; color:#9db4d6; font-weight:500; }
.kpi-sub { font-size:.8rem; color:#a9bdd9; }
.chip { display:inline-block; font-size:.72rem; font-weight:600; padding:.12rem .55rem; border-radius:999px; margin-left:.4rem;
  border:1px solid rgba(255,255,255,.15); vertical-align:middle; }
.chip.up { color:#34f5c5; background:rgba(52,245,197,.12);} .chip.down { color:#ff7a93; background:rgba(255,92,122,.12);}
.badge { display:inline-block; padding:.2rem .7rem; border-radius:999px; font-size:.75rem; font-weight:600; letter-spacing:.04em;
  border:1px solid; backdrop-filter: blur(8px); }

.hero { margin-bottom: 1.2rem; }
.hero h1 { font-size: 2.1rem; margin:0; background: linear-gradient(90deg,#fff,#7ff3e0 55%,#8ab4ff); -webkit-background-clip:text;
  -webkit-text-fill-color: transparent; }
.hero p { color:#9db4d6; margin:.2rem 0 0; }
.alert { display:flex; gap:1rem; align-items:flex-start; border-left: 4px solid var(--c); margin: .6rem 0 1rem; }
.alert .t { font-family:'Space Grotesk'; font-weight:700; font-size:1.1rem; color:#fff; }
.alert .d { color:#c4d4ec; font-size:.92rem; margin-top:.15rem; }
.alert .i { font-size:1.8rem; }

.stButton > button, .stDownloadButton > button { background: linear-gradient(135deg, rgba(94,234,212,.22), rgba(99,102,241,.22));
  color:#eaf6ff; border:1px solid rgba(255,255,255,.2); border-radius:14px; font-weight:600; backdrop-filter: blur(12px);
  transition: all .2s; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: rgba(94,234,212,.7); box-shadow: 0 0 22px rgba(94,234,212,.3);
  transform: translateY(-1px); color:#fff; }
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, .stDateInput > div > div { background: rgba(255,255,255,.06) !important;
  border: 1px solid rgba(255,255,255,.14) !important; border-radius: 14px !important; }
[data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,.04); border: 1.5px dashed rgba(94,234,212,.45); border-radius: 18px; }
[data-testid="stFileUploaderDropzone"]:hover { background: rgba(94,234,212,.08); }
[data-testid="stMetric"] { background: linear-gradient(135deg, rgba(255,255,255,.08), rgba(255,255,255,.02)); border:1px solid rgba(255,255,255,.12);
  border-radius: 18px; padding: .8rem 1rem; backdrop-filter: blur(14px); }
[data-testid="stMetricLabel"] { color:#9db4d6; }
[data-testid="stForm"] { background: linear-gradient(135deg, rgba(255,255,255,.08), rgba(255,255,255,.02)); border:1px solid rgba(255,255,255,.14);
  border-radius: 22px; backdrop-filter: blur(20px); }
.stTabs [data-baseweb="tab-list"] { gap:.4rem; background: rgba(255,255,255,.04); padding:.3rem; border-radius:16px; border:1px solid rgba(255,255,255,.08); }
.stTabs [data-baseweb="tab"] { border-radius: 12px; padding: .4rem 1rem; color:#9db4d6; }
.stTabs [aria-selected="true"] { background: rgba(94,234,212,.18); color:#fff; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
[data-testid="stExpander"] { background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.1); border-radius:16px; }
[data-testid="stDataFrame"] { border-radius: 16px; overflow:hidden; border:1px solid rgba(255,255,255,.1); }
.js-plotly-plot { border-radius: 18px; }
.brand { font-family:'Space Grotesk'; font-weight:700; font-size:1.35rem; color:#fff; }
.brand span { color:#5eead4; }
.muted { color:#8da3c4; font-size:.8rem; }
"""

LOGIN_CSS = """
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display:none !important; }
"""


def inject_css(login: bool = False) -> None:
    st.markdown(f"<style>{CSS}{LOGIN_CSS if login else ''}</style>", unsafe_allow_html=True)


def _flat(html: str) -> str:
    """Single-line HTML (Streamlit's markdown turns indented/blank-line HTML into code blocks)."""
    return "".join(line.strip() for line in html.splitlines())


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def category_color(category: str) -> str:
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["Unknown"])


def badge_html(text: str, color: str) -> str:
    return _flat(f'<span class="badge" style="color:{color};border-color:{hex_to_rgba(color, .5)};'
                 f'background:{hex_to_rgba(color, .12)}">{text}</span>')


def kpi_html(label: str, value: str, sub: str = "", accent: str = "#5eead4", delta: float | None = None,
             unit: str = "", up_is_good: bool = True) -> str:
    chip = ""
    if delta is not None and delta == delta:  # not NaN
        good = (delta >= 0) == up_is_good
        chip = f'<span class="chip {"up" if good else "down"}">{"▲" if delta >= 0 else "▼"} {abs(delta):.1f}</span>'
    unit_html = f" <small>{unit}</small>" if unit else ""
    return _flat(f'<div class="glass kpi" style="--accent:{accent}"><div class="kpi-label">{label}</div>'
                 f'<div class="kpi-value">{value}{unit_html}{chip}</div><div class="kpi-sub">{sub}</div></div>')


def kpi_card(*args, **kwargs) -> None:
    st.markdown(kpi_html(*args, **kwargs), unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "", icon: str = "") -> None:
    st.markdown(_flat(f'<div class="hero"><h1>{icon} {title}</h1><p>{subtitle}</p></div>'), unsafe_allow_html=True)


def alert_banner(level: str, headline: str, detail: str) -> None:
    color, icon = LEVEL_STYLE.get(level, LEVEL_STYLE["STABLE"])
    st.markdown(_flat(f'<div class="glass alert" style="--c:{color}"><div class="i">{icon}</div><div>'
                      f'<div class="t">{headline} {badge_html(level, color)}</div><div class="d">{detail}</div></div></div>'),
                unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 420, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(family="Inter, sans-serif", color="#cfe3ff"), margin=dict(l=10, r=10, t=40, b=10),
        showlegend=legend, legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="rgba(10,16,32,.92)", font_color="#fff", bordercolor="rgba(94,234,212,.5)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zeroline=False)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, **STRETCH)
