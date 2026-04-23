"""
Real-Time Drilling Advisory System
====================================
Streamlit dashboard for wiper trip risk prediction.
Uses ML ensemble (Gradient Boosted Trees + Isolation Forest).
Trained on real labels from daily drilling reports.
Industrial dark-themed UI with live parameter streaming.

CSS   → style.css
HTML  → templates.py
Model → model.py
Logic → engine.py
Reports → report_parser.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os

from engine import (
    load_data, compute_risk_score, generate_advisory,
    generate_events, get_risk_level, trend_arrow, trend_color,
    UNITS, DISPLAY_LABELS,
)
from model import WiperTripPredictor
import templates as T

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drilling Advisory System — 16A(78)-32",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Load CSS from external file
# ---------------------------------------------------------------------------
CSS_PATH = os.path.join(os.path.dirname(__file__), "style.css")
with open(CSS_PATH) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Data & Train Model
# ---------------------------------------------------------------------------
# Use the full 36-column CSV for richer features; subsample=10 → ~60K rows
FULL_DATA = os.path.join(
    os.path.dirname(__file__),
    "16A(78)-32_time_data_10s_intervals.csv",
)
SIMPLIFIED_DATA = os.path.join(
    os.path.dirname(__file__),
    "16A(78)-32_time_data_10s_intervals_simplified.csv",
)
DATA_FILE = FULL_DATA if os.path.exists(FULL_DATA) else SIMPLIFIED_DATA


@st.cache_data
def get_data():
    sub = 10 if "simplified" not in DATA_FILE else 6
    return load_data(DATA_FILE, subsample=sub)


@st.cache_resource
def get_trained_model(_df):
    predictor = WiperTripPredictor()
    metrics = predictor.train(_df)
    return predictor, metrics


df = get_data()
with st.spinner("Training ML models on 30-min windows (temporal split, calibrated)..."):
    ml_model, training_metrics = get_trained_model(df)

# ---------------------------------------------------------------------------
# Feature-name mapping for the importance chart
# ---------------------------------------------------------------------------
FEAT_NAME_MAP = {
    "MSE_mean": "MSE (avg)", "MSE_max": "MSE (max)", "MSE_std": "MSE (vol)",
    "MSE_trend": "MSE (trend)",
    "TRQ_mean": "Torque (avg)", "TRQ_std": "Torque (vol)", "TRQ_trend": "Torque (trend)",
    "TRQ_min": "Torque (min)", "TRQ_max": "Torque (max)", "TRQ_range": "Torque (range)",
    "SPP_mean": "Pressure (avg)", "SPP_std": "Pressure (vol)", "SPP_trend": "Pressure (trend)",
    "SPP_min": "Press (min)", "SPP_max": "Press (max)", "SPP_range": "Press (range)",
    "ROP_mean": "ROP (avg)", "ROP_std": "ROP (vol)", "ROP_trend": "ROP (trend)",
    "ROP_min": "ROP (min)", "ROP_max": "ROP (max)", "ROP_range": "ROP (range)",
    "WOB_mean": "WOB (avg)", "WOB_std": "WOB (vol)",
    "FLOW_IN_mean": "Flow (avg)", "FLOW_IN_trend": "Flow (trend)",
    "DH_TRQ_mean": "DH Torque (avg)", "DH_TRQ_trend": "DH Torque (trend)",
    "DIFF_P_mean": "Diff Press (avg)",
    "TRQ_ROP_ratio": "Torque/ROP", "MSE_x_RPM": "MSE×RPM",
    "DH_TRQ_diff": "DH Torque Diff", "Flow_pressure_ratio": "Flow/Press",
    "WOB_TRQ_ratio": "WOB/Torque",
    "HOOKLOAD_mean": "Hookload (avg)", "HOOKLOAD_trend": "Hookload (trend)",
    "HOOKLOAD_cv": "Hookload CV",
    "MWD_INC_mean": "Inclination (avg)", "MWD_INC_max": "Inclination (max)",
    "INC_HIGH_ANGLE_frac": "High Angle %", "INC_CRITICAL_frac": "Critical Angle %",
    "INC_x_TRQ": "Inc × Torque", "INC_x_MSE": "Inc × MSE",
    "PIT_GL_abs_sum": "Pit G/L (sum)", "PIT_GL_abs_max": "Pit G/L (max)",
    "FLOW_RATIO_mean": "Flow Ratio", "FLOW_IMBALANCE_mean": "Flow Imbalance",
    "GAS_max": "Gas (max)", "GAS_trend": "Gas (trend)",
    "BLOCK_VEL_mean": "Block Vel (avg)", "BLOCK_VEL_abs_mean": "Block |Vel|",
    "ON_BOTTOM_frac": "On-Bottom %", "RPM_mean": "RPM (avg)",
}

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
DEFAULTS = {
    "idx": 50, "running": False, "event_log": [],
    "risk_history": {}, "risk_scores": [], "mode": "Auto",
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Controls — Top Row (never re-rendered during streaming)
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 1, 1])
with c1:
    speed = st.slider("Refresh Interval (s)", 0.1, 2.0, 0.5, 0.1, key="speed")
with c2:
    window_size = st.slider("Chart Window (pts)", 50, 300, 100, 10, key="window")
with c3:
    mode = st.radio("Mode", ["Auto", "Manual"], horizontal=True, key="mode_select")
    st.session_state.mode = mode
with c4:
    if st.button("▶ START", use_container_width=True, type="primary"):
        st.session_state.running = True
with c5:
    if st.button("■ STOP", use_container_width=True):
        st.session_state.running = False

# Manual stepping (only shown when relevant)
if st.session_state.mode == "Manual" and not st.session_state.running:
    s1, s2, s3 = st.columns([1, 6, 1])
    with s1:
        if st.button("STEP ▶", use_container_width=True):
            st.session_state.idx = min(st.session_state.idx + 1, len(df) - 1)
    with s3:
        if st.button("STEP x10 ▶▶", use_container_width=True):
            st.session_state.idx = min(st.session_state.idx + 10, len(df) - 1)
    with s2:
        new_idx = st.slider(
            "Data Position", 50, len(df) - 1,
            st.session_state.idx, key="pos_slider",
        )
        if new_idx != st.session_state.idx:
            st.session_state.idx = new_idx
            st.session_state.risk_history = {}
            st.session_state.risk_scores = []

# ---------------------------------------------------------------------------
# Placeholder — only this container refreshes during streaming
# ---------------------------------------------------------------------------
main_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Dashboard Render
# ---------------------------------------------------------------------------
def render_dashboard(idx: int):
    """Render all dashboard panels for data index `idx`."""
    row = df.iloc[idx]
    prev = df.iloc[max(0, idx - 1)]

    # ---- Compute ----
    risk, details = compute_risk_score(
        df, idx, st.session_state.risk_history, ml_model=ml_model,
    )
    st.session_state.risk_scores.append(risk)
    advisory = generate_advisory(risk, details)

    for evt in generate_events(df, idx, None):
        st.session_state.event_log.insert(0, evt)
    st.session_state.event_log = st.session_state.event_log[:50]

    time_str = (
        row["Time"].strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(row["Time"], "strftime") else str(row["Time"])
    )
    level, color = get_risk_level(risk)
    risk_class = f"risk-{level.lower()}"
    rf_prob = details.get("rf_probability", 0)
    if_score = details.get("if_anomaly_score", 0)
    feat_imp = details.get("feature_importances", {})

    # ================================================================
    with main_placeholder.container():

        # ---- Top Bar ----
        st.markdown(
            T.top_bar(row["DEPTH"], time_str, st.session_state.mode,
                      risk, level, risk_class),
            unsafe_allow_html=True,
        )

        # ---- 3-Column Layout ----
        left_col, center_col, right_col = st.columns([1.5, 4, 2])

        # ---- LEFT: Metrics ----
        with left_col:
            st.markdown(T.section_title("Live Parameters"), unsafe_allow_html=True)
            for key in ("WOB", "RPM", "TRQ", "ROP", "SPP", "FLOW_IN"):
                st.markdown(
                    T.metric_card(
                        label=DISPLAY_LABELS.get(key, key),
                        value=row[key],
                        unit=UNITS.get(key, ""),
                        arrow=trend_arrow(row[key], prev[key]),
                        trend_color=trend_color(row[key], prev[key]),
                    ),
                    unsafe_allow_html=True,
                )

        # ---- CENTER: Charts ----
        with center_col:
            st.markdown(T.section_title("Trend Charts"), unsafe_allow_html=True)
            _render_charts(idx, window_size)

        # ---- RIGHT: Advisory + Model ----
        with right_col:
            st.markdown(T.section_title("Advisory Engine"), unsafe_allow_html=True)
            st.markdown(
                T.advisory_panel(level, advisory), unsafe_allow_html=True,
            )

            st.markdown(
                T.section_title("ML Model Output", margin_top=16),
                unsafe_allow_html=True,
            )
            st.markdown(
                T.model_scores(rf_prob, if_score, risk, advisory["color"]),
                unsafe_allow_html=True,
            )

            if feat_imp:
                st.markdown(
                    T.section_title("Feature Importance", margin_top=16),
                    unsafe_allow_html=True,
                )
                _render_importance_chart(feat_imp, idx)

        # ---- Bottom Row ----
        bl, br = st.columns([3, 2])
        with bl:
            st.markdown(T.section_title("Event Log"), unsafe_allow_html=True)
            st.markdown(
                T.event_log(st.session_state.event_log),
                unsafe_allow_html=True,
            )
        with br:
            st.markdown(T.section_title("Model Information"), unsafe_allow_html=True)
            st.markdown(
                T.model_info(training_metrics), unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Chart Helpers
# ---------------------------------------------------------------------------
def _render_charts(idx: int, win: int):
    """Build and display the 4-row Plotly trend chart."""
    start = max(0, idx - win)
    cd = df.iloc[start : idx + 1]
    rd = st.session_state.risk_scores[
        max(0, len(st.session_state.risk_scores) - win) :
    ]
    x = cd["Time"] if hasattr(cd.iloc[0]["Time"], "strftime") else list(range(len(cd)))

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=("Rate of Penetration", "Surface Torque",
                        "Standpipe Pressure", "Risk Score"),
        row_heights=[0.25] * 4,
    )

    # Helper — add raw + rolling average
    def _add(row_n, col_name, raw_color):
        fig.add_trace(
            go.Scatter(x=x, y=cd[col_name], mode="lines",
                       line=dict(color=raw_color, width=1.5),
                       showlegend=False),
            row=row_n, col=1,
        )
        if len(cd) > 5:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=cd[col_name].rolling(10, min_periods=1).mean(),
                    mode="lines",
                    line=dict(color="#f59e0b", width=2, dash="dot"),
                    showlegend=False,
                ),
                row=row_n, col=1,
            )

    _add(1, "ROP", "#38bdf8")
    _add(2, "TRQ", "#a78bfa")
    _add(3, "SPP", "#34d399")

    # Risk score area
    rx = x.iloc[-len(rd):] if hasattr(x, "iloc") else list(range(len(rd)))
    fig.add_trace(
        go.Scatter(x=rx, y=rd, mode="lines",
                   line=dict(color="#ef4444", width=2), showlegend=False,
                   fill="tozeroy", fillcolor="rgba(239,68,68,0.1)"),
        row=4, col=1,
    )
    fig.add_hline(y=0.7, line_dash="dash", line_color="#ef4444", line_width=1,
                  row=4, col=1, annotation_text="HIGH",
                  annotation_position="right",
                  annotation_font_color="#ef4444", annotation_font_size=10)
    fig.add_hline(y=0.4, line_dash="dash", line_color="#f59e0b", line_width=1,
                  row=4, col=1, annotation_text="MOD",
                  annotation_position="right",
                  annotation_font_color="#f59e0b", annotation_font_size=10)

    fig.update_layout(
        height=520, margin=dict(l=50, r=20, t=30, b=20),
        paper_bgcolor="#0a0e17", plot_bgcolor="#0f1629",
        font=dict(color="#94a3b8", size=11, family="JetBrains Mono, monospace"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#1e293b", showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor="#1e293b", showgrid=True, zeroline=False)
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=11, color="#64748b",
                           family="JetBrains Mono, monospace")

    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": False},
        key=f"trend_{idx}",
    )


def _render_importance_chart(feat_imp: dict, idx: int):
    """Horizontal bar chart for the top-8 feature importances."""
    names = [FEAT_NAME_MAP.get(k, k) for k in reversed(feat_imp)]
    vals = list(reversed(feat_imp.values()))

    fig = go.Figure(
        go.Bar(y=names, x=vals, orientation="h",
               marker=dict(color="#38bdf8", line=dict(width=0)))
    )
    fig.update_layout(
        height=200, margin=dict(l=0, r=10, t=5, b=5),
        paper_bgcolor="#0f1629", plot_bgcolor="#0f1629",
        font=dict(color="#94a3b8", size=10, family="JetBrains Mono, monospace"),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False), bargap=0.3,
    )
    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": False},
        key=f"imp_{idx}",
    )


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------
if st.session_state.running and st.session_state.mode == "Auto":
    for _ in range(500):
        if st.session_state.idx >= len(df) - 1:
            st.session_state.running = False
            break
        render_dashboard(st.session_state.idx)
        st.session_state.idx += 1
        time.sleep(speed)
else:
    render_dashboard(st.session_state.idx)
    if st.session_state.mode == "Auto" and not st.session_state.running:
        st.markdown(
            '<div style="text-align:center;color:#64748b;'
            'padding:10px;font-size:13px;">'
            'Press ▶ START to begin real-time streaming</div>',
            unsafe_allow_html=True,
        )
