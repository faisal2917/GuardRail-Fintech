"""
GuardRail Fintech — Credit Card Fraud Intelligence (Streamlit + PyTorch)
========================================================================

PROJECT STRUCTURE
-----------------
    GuardRail Fintech/
    ├── app.py                  # this Streamlit dashboard
    ├── fraud_model.py          # shared PyTorch model + scaler + explainer (imported by app and trainer)
    ├── data_pipeline.py        # KaggleDataCollector (download) + DataPreparer (clean + split)
    ├── train.py                # local training script (CPU / CUDA / Apple-silicon MPS)
    ├── requirements.txt
    ├── .streamlit/config.toml  # theme
    ├── data/
    │   ├── raw/creditcard.csv  # Kaggle dataset — NOT committed (150 MB, see .gitignore)
    │   └── processed/          # train/val/test.csv + data_report.json from data_pipeline.py
    └── artifacts/              # produced by train.py — commit these for deployment
        ├── fraud_model.pt
        ├── metadata.json
        ├── background.npy
        ├── presets.json
        └── sample_batch.csv

RUN LOCALLY
-----------
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python data_pipeline.py all     # download from Kaggle, clean, split
    python train.py                 # or: python train.py --synthetic   (no dataset download needed)
    streamlit run app.py

DEPLOY TO STREAMLIT COMMUNITY CLOUD
-----------------------------------
    1. Train locally, then commit the `artifacts/` folder (a few hundred KB) — never the raw CSV.
    2. Push the repo to GitHub.
    3. On https://share.streamlit.io → "Create app" → pick the repo/branch, main file `app.py`.
    4. Under "Advanced settings" choose Python 3.12. Cloud installs from requirements.txt
       (it pulls the CPU-only PyTorch wheel, which keeps the build small and fast).
"""
from __future__ import annotations

import io
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from fraud_model import FEATURES, TARGET, FraudScorer, explain

st.set_page_config(
    page_title="GuardRail · Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MAX_STYLED_ROWS = 1500  # pandas Styler is slow on large frames; the download always contains every row

NAVY, SLATE, EMERALD, AMBER, CRIMSON = "#0B1F3A", "#475569", "#10B981", "#F59E0B", "#DC2626"
INK, MUTED, GRID = "#0F172A", "#64748B", "#E2E8F0"

# --------------------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
#MainMenu, footer {{ visibility: hidden; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {NAVY} 0%, #12294A 100%); }}
[data-testid="stSidebar"] * {{ color: #E2E8F0; }}
[data-testid="stSidebar"] input {{ color: {INK} !important; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.12); }}

/* Hero */
.gr-hero {{ background: linear-gradient(120deg, {NAVY} 0%, #173B6C 60%, #0E5E57 100%); color: #fff;
           border-radius: 18px; padding: 26px 30px; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(11,31,58,.18); }}
.gr-hero h1 {{ font-size: 1.75rem; font-weight: 800; margin: 0; letter-spacing: -.02em; color: #fff; }}
.gr-hero p {{ margin: 6px 0 0; color: #CBD5E1; font-size: .95rem; }}
.gr-chip {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .72rem; font-weight: 600;
           background: rgba(255,255,255,.12); color: #E2E8F0; margin-right: 6px; margin-top: 10px; }}

/* Cards */
.gr-card {{ background: #fff; border: 1px solid {GRID}; border-radius: 14px; padding: 18px 20px;
           box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 4px 14px rgba(15,23,42,.05); height: 100%; }}
.gr-card .lbl {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: {MUTED}; font-weight: 600; }}
.gr-card .val {{ font-size: 1.6rem; font-weight: 800; color: {INK}; margin-top: 4px; letter-spacing: -.02em; }}
.gr-card .sub {{ font-size: .8rem; color: {MUTED}; margin-top: 2px; }}
.gr-accent-emerald {{ border-left: 4px solid {EMERALD}; }}
.gr-accent-amber   {{ border-left: 4px solid {AMBER}; }}
.gr-accent-crimson {{ border-left: 4px solid {CRIMSON}; }}
.gr-accent-navy    {{ border-left: 4px solid {NAVY}; }}

.gr-decision {{ border-radius: 14px; padding: 16px 20px; font-weight: 700; display: flex; align-items: center;
               justify-content: space-between; margin-bottom: 12px; }}
.gr-decision .big {{ font-size: 1.15rem; letter-spacing: .02em; }}
.gr-decision .small {{ font-size: .82rem; font-weight: 500; opacity: .85; }}

.gr-section {{ font-size: 1.02rem; font-weight: 700; color: {INK}; margin: 6px 0 10px; }}
.gr-driver {{ background: #FFF5F5; border: 1px solid #FECACA; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; }}
.gr-driver.safe {{ background: #F0FDF4; border-color: #BBF7D0; }}
.gr-driver b {{ color: {INK}; }}

/* Tab bar: sticky segmented control */
.stTabs [role="tablist"] {{ position: sticky; top: 3.9rem; z-index: 50; margin-bottom: 14px;
                           display: flex; gap: 6px; background: #fff; border: 1px solid {GRID}; border-radius: 14px; padding: 6px;
                           box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 6px 18px rgba(15,23,42,.06); overflow-x: auto; scrollbar-width: none; }}
.stTabs [role="tablist"]::-webkit-scrollbar {{ display: none; }}
.stTabs [data-testid="stTab"] {{ flex: 1 1 0; justify-content: center; height: 44px; padding: 0 18px; border-radius: 10px; color: {SLATE};
                                white-space: nowrap; cursor: pointer; transition: background .15s, color .15s, box-shadow .15s; }}
.stTabs [data-testid="stTab"] p {{ font-size: .92rem; font-weight: 600; color: inherit; display: flex; align-items: center; gap: 8px; }}
.stTabs [data-testid="stTab"] [data-testid="stIconMaterial"] {{ font-size: 1.2rem; color: inherit; }}
.stTabs [data-testid="stTab"]:hover {{ background: #F1F5F9; color: {NAVY}; }}
.stTabs [data-testid="stTab"][aria-selected="true"] {{ background: linear-gradient(135deg, {NAVY} 0%, #1E3A66 100%); color: #fff;
                                                      box-shadow: 0 4px 12px rgba(11,31,58,.22); }}
.stTabs [data-testid="stTab"][data-focus-visible="true"] {{ outline: 2px solid {EMERALD}; outline-offset: 2px; }}
.stTabs .react-aria-SelectionIndicator {{ display: none; }}

/* Batch scanner: stepper, panels, dropzone */
.gr-steps {{ display: flex; align-items: center; gap: 12px; margin: 2px 0 16px; }}
.gr-step {{ display: flex; align-items: center; gap: 10px; color: {MUTED}; }}
.gr-step .dot {{ width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center; font-weight: 700; font-size: .85rem;
                background: #E2E8F0; color: {SLATE}; flex-shrink: 0; }}
.gr-step b {{ display: block; font-size: .88rem; color: {INK}; }}
.gr-step small {{ font-size: .75rem; }}
.gr-step.active .dot {{ background: {NAVY}; color: #fff; box-shadow: 0 0 0 4px rgba(11,31,58,.12); }}
.gr-step.done .dot {{ background: {EMERALD}; color: #fff; }}
.gr-step.error .dot {{ background: {CRIMSON}; color: #fff; }}
.gr-step-line {{ flex: 1; height: 2px; background: {GRID}; border-radius: 2px; min-width: 24px; }}
.gr-step-line.done {{ background: {EMERALD}; }}
@media (max-width: 640px) {{
  .gr-steps {{ gap: 8px; }}
  .gr-step {{ gap: 6px; }}
  .gr-step small {{ display: none; }}
  .gr-step b {{ font-size: .78rem; white-space: nowrap; }}
  .gr-step-line {{ min-width: 10px; }}
  .stTabs [data-testid="stTab"] {{ flex: 0 0 auto; }}
}}

.st-key-upload_panel, .st-key-sample_panel {{ background: #fff; border: 1px solid {GRID}; border-radius: 16px; padding: 20px 22px;
                                             box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 4px 14px rgba(15,23,42,.05); }}
.gr-panel-title {{ font-weight: 700; font-size: 1rem; color: {INK}; }}
.gr-panel-sub {{ font-size: .82rem; color: {MUTED}; margin: 2px 0 6px; }}

section[data-testid="stFileUploaderDropzone"] {{ display: flex !important; flex-direction: column !important; align-items: center !important;
    justify-content: center; gap: 12px;
    min-height: 200px; padding: 28px 20px; border: 2px dashed #CBD5E1; border-radius: 14px; text-align: center;
    background: radial-gradient(circle at 50% 0%, #EEF7F4 0%, #F8FAFC 60%); transition: border-color .15s, background .15s; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: {EMERALD}; background: #F0FDF9; }}
[data-testid="stFileUploaderDropzone"]::before {{ content: ""; order: 0; width: 54px; height: 54px; border-radius: 16px;
    background: #DDF5EC url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23059669' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 14.9A7 7 0 1 1 15.7 8h1.8a4.5 4.5 0 0 1 2.5 8.2'/%3E%3Cpath d='M12 12v9'/%3E%3Cpath d='m16 16-4-4-4 4'/%3E%3C/svg%3E") center / 28px no-repeat; }}
[data-testid="stFileUploaderDropzoneInstructions"] {{ order: 1; }}
[data-testid="stFileUploaderDropzoneInstructions"] > div {{ display: flex; flex-direction: column; align-items: center; gap: 2px; }}
[data-testid="stFileUploaderDropzoneInstructions"] > div::before {{ content: "Drag & drop your transactions CSV"; font-weight: 700; font-size: 1rem; color: {INK}; text-align: center; }}
[data-testid="stFileUploaderDropzoneInstructions"] span {{ color: {MUTED}; font-size: .8rem; }}
[data-testid="stFileUploaderDropzone"] > span {{ order: 2; }}
[data-testid="stFileUploaderDropzone"] button {{ background: {NAVY}; border: 0; border-radius: 10px; padding: 8px 20px; }}
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button * {{ color: #fff !important; }}
[data-testid="stFileUploaderDropzone"] button:hover {{ background: #1E3A66; }}
[data-testid="stFileUploaderFile"] {{ border: 1px solid {GRID}; border-radius: 12px; padding: 8px 12px; background: #F8FAFC; margin-top: 10px; }}

.gr-chips, .gr-pills {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }}
.gr-chips span {{ font-size: .75rem; font-weight: 600; padding: 3px 10px; border-radius: 999px; font-family: ui-monospace, Menlo, monospace; }}
.gr-chips .req {{ background: #EEF2F7; color: {NAVY}; }}
.gr-chips .opt {{ background: #fff; color: {SLATE}; border: 1px dashed #CBD5E1; }}
.gr-pills {{ margin: 16px 0 14px; }}
.gr-pill {{ font-size: .8rem; padding: 6px 12px; border-radius: 10px; border: 1px solid {GRID}; background: #fff; color: {SLATE}; }}
.gr-pill b {{ color: {INK}; font-weight: 600; }}
.gr-pill.ok {{ background: #ECFDF5; border-color: #A7F3D0; color: #047857; }}
.gr-pill.warn {{ background: #FFFBEB; border-color: #FDE68A; color: #B45309; }}
.gr-pill.err {{ background: #FEF2F2; border-color: #FECACA; color: #B91C1C; }}

/* Native widgets */
div.stButton > button, div.stDownloadButton > button {{ border-radius: 10px; font-weight: 600; border: 1px solid {GRID}; }}
[data-testid="stMetric"] {{ background: #fff; border: 1px solid {GRID}; border-radius: 12px; padding: 12px 16px; }}
[data-testid="stExpander"] {{ border-radius: 12px; }}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Loading (cached)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading fraud model…")
def load_scorer() -> FraudScorer | None:
    path = ARTIFACTS / "fraud_model.pt"
    if not path.exists():
        return None
    return FraudScorer.load(path)


@st.cache_resource
def load_background() -> np.ndarray | None:
    path = ARTIFACTS / "background.npy"
    return np.load(path) if path.exists() else None


@st.cache_data
def load_json(name: str) -> dict:
    path = ARTIFACTS / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def baseline_logodds(_scorer: FraudScorer, background: np.ndarray) -> float:
    with torch.no_grad():
        return float(_scorer.model(torch.from_numpy(background)).mean())


@st.cache_data(show_spinner=False)
def explain_row(_scorer: FraudScorer, background: np.ndarray, row: tuple[float, ...]) -> tuple[np.ndarray, str]:
    z = _scorer.transform(np.asarray([row]))
    values, method = explain(_scorer.model, background, z)
    return values[0], method


@st.cache_data(show_spinner=False)
def read_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


@st.cache_data(show_spinner=False)
def score_frame(_scorer: FraudScorer, df: pd.DataFrame) -> np.ndarray:
    return _scorer.predict_proba(df[FEATURES].to_numpy(np.float32))


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def money(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def risk_band(p: float, threshold: float) -> tuple[str, str]:
    if p >= threshold:
        return "High", CRIMSON
    if p >= threshold / 2:
        return "Moderate", AMBER
    return "Low", EMERALD


def stepper_html(stage: int, failed: bool = False) -> str:
    """Three-step progress header for the batch scanner. `stage` is the index of the current step."""
    steps = [("Upload", "CSV file or sample batch"), ("Validate", "Schema and data checks"), ("Review & export", "Filter, inspect, download")]
    parts = []
    for i, (title, sub) in enumerate(steps):
        if i < stage:
            state, mark = "done", "✓"
        elif i == stage:
            state, mark = ("error", "!") if failed else ("active", str(i + 1))
        else:
            state, mark = "", str(i + 1)
        if i:
            parts.append(f'<div class="gr-step-line {"done" if i <= stage and not (failed and i == stage) else ""}"></div>')
        parts.append(f'<div class="gr-step {state}"><span class="dot">{mark}</span><div><b>{title}</b><small>{sub}</small></div></div>')
    return f'<div class="gr-steps">{"".join(parts)}</div>'


def card(label: str, value: str, sub: str = "", accent: str = "navy") -> str:
    return f'<div class="gr-card gr-accent-{accent}"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def style_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        title_font=dict(size=14, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def gauge(p: float, threshold: float) -> go.Figure:
    _, color = risk_band(p, threshold)
    t = threshold * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%", "font": {"size": 46, "color": color}, "valueformat": ".1f"},
            title={"text": "Fraud Probability", "font": {"size": 14, "color": MUTED}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": GRID},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, t / 2], "color": "#D1FAE5"},
                    {"range": [t / 2, t], "color": "#FEF3C7"},
                    {"range": [t, 100], "color": "#FEE2E2"},
                ],
                "threshold": {"line": {"color": NAVY, "width": 3}, "thickness": 0.85, "value": t},
            },
        )
    )
    return style_fig(fig, height=290)


def describe(feature: str, raw: float, z: float) -> str:
    direction = "above" if z >= 0 else "below"
    if feature == "Amount":
        return f"transaction amount of <b>{money(raw)}</b> is {abs(z):.1f}σ {direction} the average ticket"
    if feature == "Time":
        hh, mm = divmod(int(raw % 86_400) // 60, 60)
        return f"transaction timing (≈ {hh:02d}:{mm:02d} on the statement clock) is {abs(z):.1f}σ {direction} the typical time"
    return f"behavioural component <b>{feature}</b> is {abs(z):.1f}σ {direction} the training baseline"


# --------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------
scorer = load_scorer()
background = load_background()
meta = load_json("metadata.json")
presets = load_json("presets.json")

# Sidebar -------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ GuardRail")
    st.caption("Fraud Intelligence Console")
    st.divider()

    if scorer is None:
        st.error("Model not loaded")
    else:
        st.markdown("**Model status**  \n🟢 PyTorch FraudNet online")
        if meta:
            st.caption(f"Trained {meta.get('trained_at', '—')[:10]} · {meta.get('data_source', '—')}")
        if meta.get("data_source") == "synthetic":
            st.warning("Demo model trained on synthetic data. Retrain on `creditcard.csv` for real results.")

    st.divider()
    st.markdown("**Decision policy**")
    default_thr = float(scorer.threshold) if scorer else 0.5
    threshold = st.slider(
        "Block threshold",
        0.01,
        0.99,
        float(np.clip(round(default_thr, 2), 0.01, 0.99)),
        0.01,
        help=f"Transactions at or above this probability are flagged. Tuned value (max F1 on validation): {default_thr:.3f}",
    )
    review_cost = st.number_input("Manual review cost / case ($)", 0.0, 500.0, 12.0, 1.0)
    chargeback_fee = st.number_input("Chargeback fee / fraud ($)", 0.0, 500.0, 25.0, 1.0)
    st.divider()
    st.caption("Built with Streamlit · PyTorch · SHAP · Plotly")

# Hero ------------------------------------------------------------------------------------
chips = ""
if meta:
    m = meta.get("test_metrics", {})
    chips = (
        f'<span class="gr-chip">ROC-AUC {m.get("roc_auc", 0):.3f}</span>'
        f'<span class="gr-chip">PR-AUC {m.get("pr_auc", 0):.3f}</span>'
        f'<span class="gr-chip">Recall {m.get("recall", 0):.1%}</span>'
        f'<span class="gr-chip">Threshold {threshold:.2f}</span>'
    )
st.markdown(
    f"""<div class="gr-hero"><h1>Credit Card Fraud Intelligence</h1>
    <p>Real-time risk scoring, explainable decisions and batch screening powered by a PyTorch neural network.</p>{chips}</div>""",
    unsafe_allow_html=True,
)

if scorer is None:
    st.warning(
        "**No trained model found** in `artifacts/fraud_model.pt`.\n\n"
        "Train one locally first:\n\n"
        "```bash\npython train.py              # uses data/creditcard.csv\npython train.py --synthetic  # quick demo, no download\n```"
    )
    st.stop()

# Single-transaction inputs live in session state so presets can overwrite them.
baseline_values = dict(zip(scorer.features, scorer.mean.astype(float).tolist()))
default_values = presets.get("legitimate", baseline_values)
for f in scorer.features:
    st.session_state.setdefault(f"in_{f}", float(default_values.get(f, baseline_values[f])))


def apply_values(values: dict) -> None:
    for f in scorer.features:
        st.session_state[f"in_{f}"] = float(values.get(f, baseline_values[f]))


tab_rt, tab_xai, tab_batch, tab_ins = st.tabs(
    [
        ":material/bolt: Real-Time Risk Analyzer",
        ":material/psychology: Explainable AI",
        ":material/upload_file: Batch CSV Scanner",
        ":material/insights: Model Insights",
    ]
)

# --------------------------------------------------------------------------------------
# TAB 1 — Real-time analyzer
# --------------------------------------------------------------------------------------
with tab_rt:
    b1, b2, b3, _ = st.columns([1.2, 1.3, 1, 2.5])
    b1.button("✅ Load Legitimate Sample", on_click=apply_values, args=(presets.get("legitimate", baseline_values),),
              disabled="legitimate" not in presets, width="stretch")
    b2.button("🚨 Load High-Risk Fraud Sample", on_click=apply_values, args=(presets.get("fraud", baseline_values),),
              disabled="fraud" not in presets, width="stretch")
    b3.button("↺ Reset to Baseline", on_click=apply_values, args=(baseline_values,), width="stretch")

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown('<div class="gr-section">Transaction parameters</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.number_input("Transaction amount ($)", min_value=0.0, max_value=100_000.0, step=10.0, key="in_Amount", format="%.2f")
        c2.number_input("Time (seconds since first txn)", min_value=0.0, max_value=200_000.0, step=60.0, key="in_Time", format="%.0f",
                        help="Kaggle dataset convention: seconds elapsed since the first transaction in the 48h window.")

        importance = meta.get("importance", {}).get("values", {})
        v_features = [f for f in scorer.features if f.startswith("V")]
        key_drivers = [f for f in importance if f.startswith("V")][:6] or v_features[:6]

        st.markdown('<div class="gr-section">Key risk drivers <span style="color:#64748B;font-weight:500;font-size:.8rem">· most influential PCA components</span></div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        for i, f in enumerate(key_drivers):
            cur = st.session_state[f"in_{f}"]
            lo, hi = min(-30.0, float(np.floor(cur))), max(30.0, float(np.ceil(cur)))
            (s1 if i % 2 == 0 else s2).slider(f, lo, hi, step=0.01, key=f"in_{f}")

        with st.expander("All other PCA components (V1–V28)"):
            cols = st.columns(4)
            for i, f in enumerate([f for f in v_features if f not in key_drivers]):
                cols[i % 4].number_input(f, step=0.1, key=f"in_{f}", format="%.4f")

    row = np.array([[st.session_state[f"in_{f}"] for f in scorer.features]], dtype=np.float32)
    try:
        p = float(scorer.predict_proba(row)[0])
    except Exception as exc:  # malformed input should never take the page down
        st.warning(f"Could not score this transaction: {exc}")
        st.stop()
    st.session_state["scored_row"] = tuple(row[0].tolist())
    st.session_state["scored_p"] = p

    amount = float(st.session_state["in_Amount"])
    band, color = risk_band(p, threshold)
    flagged = p >= threshold
    exposure = p * (amount + chargeback_fee)
    net_benefit = exposure - review_cost

    with right:
        st.plotly_chart(gauge(p, threshold), width="stretch", config={"displayModeBar": False})
        if flagged:
            decision, note, bg = "⛔ BLOCK & REVIEW", "Probability exceeds block threshold", "#FEE2E2"
        elif band == "Moderate":
            decision, note, bg = "⚠️ STEP-UP AUTHENTICATION", "Elevated risk — request 3-DS / OTP", "#FEF3C7"
        else:
            decision, note, bg = "✅ APPROVE", "Within normal behavioural range", "#D1FAE5"
        st.markdown(
            f'<div class="gr-decision" style="background:{bg};color:{color}"><span class="big">{decision}</span>'
            f'<span class="small">{band} risk · {note}</span></div>',
            unsafe_allow_html=True,
        )

        k1, k2 = st.columns(2)
        if flagged:
            k1.markdown(card("Potential Loss Prevented", money(exposure), f"p × (amount + {money(chargeback_fee)} chargeback)", "emerald"), unsafe_allow_html=True)
        else:
            k1.markdown(card("Expected Exposure Accepted", money(exposure), "Residual risk if approved", "amber" if band == "Moderate" else "navy"), unsafe_allow_html=True)
        k2.markdown(card("Net Benefit of Blocking", money(net_benefit), f"after {money(review_cost)} review cost", "emerald" if net_benefit > 0 else "crimson"), unsafe_allow_html=True)
        st.write("")
        k3, k4 = st.columns(2)
        k3.markdown(card("Worst-case Loss", money(amount + chargeback_fee), "If this transaction is fraud", "crimson"), unsafe_allow_html=True)
        k4.markdown(card("Break-even Probability", f"{review_cost / max(amount + chargeback_fee, 1e-9):.1%}", "Blocking pays off above this", "navy"), unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# TAB 2 — Explainable AI
# --------------------------------------------------------------------------------------
with tab_xai:
    if background is None:
        st.warning("`artifacts/background.npy` is missing, so explanations are unavailable. Re-run `python train.py`.")
    elif "scored_row" not in st.session_state:
        st.info("Score a transaction in **Real-Time Risk Analyzer** to see its explanation.")
    else:
        row_t = st.session_state["scored_row"]
        p = st.session_state["scored_p"]
        with st.spinner("Computing SHAP attributions…"):
            values, method = explain_row(scorer, background, row_t)
        z = scorer.transform(np.asarray([row_t]))[0]
        base = baseline_logodds(scorer, background)
        logit = float(np.log(max(p, 1e-9) / max(1 - p, 1e-9)))

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(card("Fraud Probability", f"{p:.1%}", f"{risk_band(p, threshold)[0]} risk", "crimson" if p >= threshold else "emerald"), unsafe_allow_html=True)
        m2.markdown(card("Baseline Log-odds", f"{base:+.2f}", "Average over reference sample", "navy"), unsafe_allow_html=True)
        m3.markdown(card("Model Log-odds", f"{logit:+.2f}", f"Δ {logit - base:+.2f} from baseline", "navy"), unsafe_allow_html=True)
        m4.markdown(card("Attribution Method", method.split("·")[0].strip(), method.split("·")[-1].strip(), "navy"), unsafe_allow_html=True)
        st.write("")

        top_k = st.slider("Number of features to display", 5, len(scorer.features), 12)
        order = np.argsort(-np.abs(values))[:top_k][::-1]
        names = [scorer.features[i] for i in order]
        raw_vals = [row_t[i] for i in order]
        fig = go.Figure(
            go.Bar(
                x=values[order],
                y=names,
                orientation="h",
                marker_color=[CRIMSON if v > 0 else EMERALD for v in values[order]],
                customdata=np.stack([raw_vals, z[order]], axis=1),
                hovertemplate="<b>%{y}</b><br>Contribution: %{x:+.3f} log-odds<br>Value: %{customdata[0]:.3f}<br>z-score: %{customdata[1]:+.2f}σ<extra></extra>",
                text=[f"{v:+.2f}" for v in values[order]],
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig.update_layout(title="Feature contributions to fraud score (red ↑ risk · green ↓ risk)", xaxis_title="SHAP value (log-odds)")
        fig.add_vline(x=0, line_color=SLATE, line_width=1)

        left, right = st.columns([1.5, 1], gap="large")
        left.plotly_chart(style_fig(fig, height=max(360, 32 * top_k)), width="stretch", config={"displayModeBar": False})

        with right:
            st.markdown('<div class="gr-section">Decision narrative</div>', unsafe_allow_html=True)
            risk_up = [i for i in np.argsort(-values) if values[i] > 0][:3]
            risk_down = [i for i in np.argsort(values) if values[i] < 0][:2]
            if risk_up:
                i = risk_up[0]
                st.markdown(
                    f'<div class="gr-driver"><b>Primary Risk Driver:</b> the {describe(scorer.features[i], row_t[i], z[i])}, '
                    f'adding <b>{values[i]:+.2f}</b> log-odds of fraud.</div>',
                    unsafe_allow_html=True,
                )
                for i in risk_up[1:]:
                    st.markdown(f"- **Secondary driver** — {describe(scorer.features[i], row_t[i], z[i])} ({values[i]:+.2f})", unsafe_allow_html=True)
            for i in risk_down:
                st.markdown(
                    f'<div class="gr-driver safe"><b>Mitigating factor:</b> the {describe(scorer.features[i], row_t[i], z[i])}, '
                    f'reducing risk by <b>{abs(values[i]):.2f}</b> log-odds.</div>',
                    unsafe_allow_html=True,
                )
            st.caption(
                "V1–V28 are PCA components released by the card issuer to protect cardholder privacy, so their business meaning "
                "is anonymised. Contributions are in log-odds relative to the baseline; together they account for the gap "
                "between baseline and model log-odds."
            )

# --------------------------------------------------------------------------------------
# TAB 3 — Batch CSV scanner
# --------------------------------------------------------------------------------------
with tab_batch:
    sample_path = ARTIFACTS / "sample_batch.csv"
    stepper = st.empty()

    up_col, side_col = st.columns([1.7, 1], gap="large")
    with up_col, st.container(key="upload_panel"):
        st.markdown(
            '<div class="gr-panel-title">Upload transactions</div>'
            '<div class="gr-panel-sub">Drop a CSV export from your card processor. It is processed in memory and never written to disk.</div>',
            unsafe_allow_html=True,
        )
        upload = st.file_uploader("Transactions CSV", type=["csv"], label_visibility="collapsed")
    with side_col:
        with st.container(key="sample_panel"):
            st.markdown(
                '<div class="gr-panel-title">No file handy?</div>'
                '<div class="gr-panel-sub">Score held-out test-set transactions with known fraud labels.</div>',
                unsafe_allow_html=True,
            )
            use_sample = st.toggle("Use sample batch", disabled=not sample_path.exists(), help="Ignored while a file is uploaded.")
            if sample_path.exists():
                st.download_button("Download sample CSV", sample_path.read_bytes(), "sample_batch.csv", "text/csv", icon=":material/download:", width="stretch")
        st.markdown(
            '<div class="gr-card" style="margin-top:14px"><div class="lbl">Expected schema</div>'
            '<div class="gr-chips"><span class="req">Time</span><span class="req">V1 … V28</span><span class="req">Amount</span>'
            f'<span class="opt">{TARGET} · optional</span></div>'
            f'<div class="sub">Extra columns such as IDs are passed through to the export. Include <code>{TARGET}</code> (0/1) to measure recall and precision on the batch.</div></div>',
            unsafe_allow_html=True,
        )

    if upload is not None:
        raw, source_name = upload.getvalue(), upload.name
    elif use_sample and sample_path.exists():
        raw, source_name = sample_path.read_bytes(), sample_path.name
    else:
        raw, source_name = None, None

    df, error, skipped = None, None, 0
    if raw is not None:
        try:
            df = read_csv(raw)
        except Exception as exc:
            error = f"Could not read the file as CSV ({exc})."
        if df is not None:
            missing = [f for f in FEATURES if f not in df.columns]
            if df.empty:
                error = "The file has no rows."
            elif missing:
                error = f"Missing required columns: {', '.join(missing)}"
            else:
                df = df.copy()
                df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
                bad = df[FEATURES].isna().any(axis=1)
                skipped = int(bad.sum())
                df = df.loc[~bad].reset_index(drop=True)
                if df.empty:
                    error = "No valid rows left after removing rows with missing or non-numeric values."

    stage = 0 if raw is None else (1 if error else 2)
    stepper.markdown(stepper_html(stage, failed=error is not None), unsafe_allow_html=True)

    if raw is not None:
        pills = [("", f"<b>{escape(source_name)}</b> · {len(raw) / 1e6:.2f} MB")]
        if df is not None and not error:
            labelled = TARGET in df.columns
            pills += [
                ("ok", f"✓ <b>{len(df):,}</b> valid rows"),
                ("ok", f"✓ All {len(FEATURES)} model features present"),
                ("ok" if labelled else "", f"✓ <b>{TARGET}</b> labels found" if labelled else f"No <b>{TARGET}</b> labels · scoring only"),
            ]
            if skipped:
                pills.append(("warn", f"⚠ {skipped:,} row(s) skipped · missing or non-numeric values"))
        else:
            pills.append(("err", f"✕ {escape(error)}"))
        st.markdown(
            '<div class="gr-pills">' + "".join(f'<span class="gr-pill {k}">{t}</span>' for k, t in pills) + "</div>",
            unsafe_allow_html=True,
        )

    if raw is not None and not error:
        with st.spinner(f"Scoring {len(df):,} transactions…"):
            probs = score_frame(scorer, df)
        result = df.copy()
        result.insert(0, "Fraud Risk Score", (probs * 100).round(2))
        result.insert(1, "Risk Level", [risk_band(p, threshold)[0] for p in probs])
        result.insert(2, "Flagged", np.where(probs >= threshold, "Yes", "No"))

        flagged_mask = result["Flagged"] == "Yes"
        flagged_amount = float(result.loc[flagged_mask, "Amount"].sum())
        expected = float((probs * (result["Amount"] + chargeback_fee)).sum())
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(card("Transactions Scanned", f"{len(result):,}", "valid rows", "navy"), unsafe_allow_html=True)
        k2.markdown(card("Flagged", f"{int(flagged_mask.sum()):,}", f"{flagged_mask.mean():.2%} of batch", "crimson"), unsafe_allow_html=True)
        k3.markdown(card("Value Held for Review", money(flagged_amount), f"review cost {money(review_cost * flagged_mask.sum())}", "amber"), unsafe_allow_html=True)
        if TARGET in result.columns and result[TARGET].isin([0, 1]).all():
            truth = result[TARGET].astype(int).to_numpy()
            tp = int(((truth == 1) & flagged_mask).sum())
            rec = tp / max(int(truth.sum()), 1)
            prec = tp / max(int(flagged_mask.sum()), 1)
            k4.markdown(card("Batch Recall / Precision", f"{rec:.0%} / {prec:.0%}", f"{int(truth.sum())} labelled frauds", "emerald"), unsafe_allow_html=True)
        else:
            k4.markdown(card("Expected Fraud Exposure", money(expected), "Σ p × (amount + chargeback)", "emerald"), unsafe_allow_html=True)
        st.write("")

        hist = go.Figure(go.Histogram(x=result["Fraud Risk Score"], nbinsx=50, marker_color=SLATE))
        hist.add_vline(x=threshold * 100, line_color=CRIMSON, line_dash="dash", annotation_text="block threshold")
        hist.update_layout(title="Risk score distribution", xaxis_title="Fraud risk score (%)", yaxis_title="Transactions", yaxis_type="log", bargap=0.05)
        st.plotly_chart(style_fig(hist, height=260), width="stretch", config={"displayModeBar": False})

        f1, f2, f3 = st.columns([1, 1.4, 1])
        show = f1.multiselect("Flagged", ["Yes", "No"], default=["Yes", "No"])
        min_risk = f2.slider("Minimum risk score (%)", 0.0, 100.0, 0.0, 0.5)
        sort_desc = f3.toggle("Sort by risk (highest first)", value=True)

        view = result[result["Flagged"].isin(show) & (result["Fraud Risk Score"] >= min_risk)]
        if sort_desc:
            view = view.sort_values("Fraud Risk Score", ascending=False)

        def highlight(r: pd.Series) -> list[str]:
            return ["background-color: #FDECEC; color: #7F1D1D" if r["Flagged"] == "Yes" else ""] * len(r)

        shown = view.head(MAX_STYLED_ROWS)
        st.dataframe(
            shown.style.apply(highlight, axis=1).format(
                {f: "{:.4f}" for f in FEATURES} | {"Time": "{:,.0f}", "Amount": "${:,.2f}", "Fraud Risk Score": "{:.2f}%"}
            ),
            width="stretch",
            height=440,
            hide_index=True,
        )
        if len(view) > MAX_STYLED_ROWS:
            st.caption(f"Showing the first {MAX_STYLED_ROWS:,} of {len(view):,} matching rows. Downloads include everything.")

        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("Download full results", result.to_csv(index=False).encode(), "guardrail_scored.csv", "text/csv", type="primary", icon=":material/download:", width="stretch")
        d2.download_button("Download flagged only", result[flagged_mask].to_csv(index=False).encode(), "guardrail_flagged.csv", "text/csv", icon=":material/flag:", width="stretch")

# --------------------------------------------------------------------------------------
# TAB 4 — Model insights
# --------------------------------------------------------------------------------------
with tab_ins:
    if not meta:
        st.warning("`artifacts/metadata.json` is missing. Re-run `python train.py` to generate model analytics.")
    else:
        m = meta["test_metrics"]
        cols = st.columns(5)
        for col, (lbl, key, sub) in zip(
            cols,
            [("ROC-AUC", "roc_auc", "ranking quality"), ("PR-AUC", "pr_auc", "imbalance-aware"), ("Precision", "precision", "flags that are fraud"),
             ("Recall", "recall", "fraud caught"), ("F1 Score", "f1", f"@ threshold {meta['threshold']:.3f}")],
        ):
            col.markdown(card(lbl, f"{m[key]:.3f}", sub, "navy"), unsafe_allow_html=True)
        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.markdown(card("Fraud $ Caught (test)", money(m["fraud_amount_caught"]), "at tuned threshold", "emerald"), unsafe_allow_html=True)
        c2.markdown(card("Fraud $ Missed (test)", money(m["fraud_amount_missed"]), "false negatives", "crimson"), unsafe_allow_html=True)
        c3.markdown(card("Training Data", f"{meta['n_rows']:,} rows", f"{meta['fraud_rate']:.3%} fraud · {meta['data_source']}", "navy"), unsafe_allow_html=True)
        st.write("")

        g1, g2 = st.columns(2, gap="large")
        pr, roc = meta["curves"]["pr"], meta["curves"]["roc"]
        fig_pr = go.Figure(go.Scatter(x=pr["x"], y=pr["y"], mode="lines", line=dict(color=CRIMSON, width=2.5), fill="tozeroy", fillcolor="rgba(220,38,38,.08)", name="Model"))
        fig_pr.add_hline(y=meta["fraud_rate"], line_dash="dot", line_color=SLATE, annotation_text="random")
        fig_pr.update_layout(title=f"Precision–Recall curve (AP = {m['pr_auc']:.3f})", xaxis_title="Recall", yaxis_title="Precision", showlegend=False)
        g1.plotly_chart(style_fig(fig_pr), width="stretch", config={"displayModeBar": False})

        fig_roc = go.Figure(go.Scatter(x=roc["x"], y=roc["y"], mode="lines", line=dict(color=NAVY, width=2.5), fill="tozeroy", fillcolor="rgba(11,31,58,.06)"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=SLATE, dash="dot")))
        fig_roc.update_layout(title=f"ROC curve (AUC = {m['roc_auc']:.3f})", xaxis_title="False positive rate", yaxis_title="True positive rate", showlegend=False)
        g2.plotly_chart(style_fig(fig_roc), width="stretch", config={"displayModeBar": False})

        g3, g4 = st.columns(2, gap="large")
        cm = m["confusion"]
        z_cm = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        fig_cm = go.Figure(go.Heatmap(z=z_cm, x=["Pred: Legit", "Pred: Fraud"], y=["Actual: Legit", "Actual: Fraud"], colorscale=[[0, "#F1F5F9"], [1, NAVY]],
                                      showscale=False, text=[[f"{v:,}" for v in r] for r in z_cm], texttemplate="%{text}", textfont={"size": 18}))
        fig_cm.update_layout(title="Confusion matrix (held-out test set)")
        fig_cm.update_yaxes(autorange="reversed")
        g3.plotly_chart(style_fig(fig_cm), width="stretch", config={"displayModeBar": False})

        hist = pd.DataFrame(meta["history"])
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=hist["epoch"], y=hist["train_loss"], name="Train loss", line=dict(color=SLATE, width=2)))
        fig_h.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_loss"], name="Val loss", line=dict(color=AMBER, width=2)))
        fig_h.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_pr_auc"], name="Val PR-AUC", yaxis="y2", line=dict(color=EMERALD, width=2.5)))
        fig_h.update_layout(title="Training history", xaxis_title="Epoch", yaxis_title="Loss",
                            yaxis2=dict(title="PR-AUC", overlaying="y", side="right", range=[0, 1], showgrid=False))
        style_fig(fig_h).update_layout(legend=dict(orientation="h", yanchor="top", y=-0.22, x=0), margin=dict(b=40))
        g4.plotly_chart(fig_h, width="stretch", config={"displayModeBar": False})

        imp = meta.get("importance", {})
        if imp.get("values"):
            top = list(imp["values"].items())[:15][::-1]
            fig_i = go.Figure(go.Bar(x=[v for _, v in top], y=[k for k, _ in top], orientation="h", marker_color=NAVY))
            fig_i.update_layout(title=f"Global feature importance · mean |attribution| ({imp.get('method', '')})", xaxis_title="Mean |log-odds contribution|")
            st.plotly_chart(style_fig(fig_i, height=460), width="stretch", config={"displayModeBar": False})

        with st.expander("Model card"):
            p_ = meta["params"]
            st.markdown(
                f"""
| Property | Value |
|---|---|
| Architecture | MLP {len(FEATURES)} → {' → '.join(map(str, p_['hidden']))} → 1 (BatchNorm · SiLU · Dropout {p_['dropout']}) |
| Loss | BCEWithLogits, pos_weight {p_['pos_weight']:.1f} |
| Optimiser | AdamW, lr {p_['lr']}, batch {p_['batch_size']} |
| Epochs run | {p_['epochs_run']} (early stopping on val PR-AUC, best {p_['best_val_pr_auc']:.4f}) |
| Split | train {meta['split']['train']:,} · val {meta['split']['val']:,} · test {meta['split']['test']:,} (stratified) |
| Threshold | {meta['threshold']:.4f} (max F1 on validation) |
| Trained on | {meta['device']} · PyTorch {meta['torch_version']} · {meta['trained_at']} |
"""
            )
