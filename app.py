"""
Simplified Drilling Advisory Dashboard
========================================
Answers one question: "Will a wiper trip be needed in the next 4 hours?"

Shows:
1. BIG prediction answer (YES / POSSIBLY / UNLIKELY) with probability
2. One risk-over-time chart
3. Plain English "why" explanation
4. Model details in an expander (for those who want to dig deeper)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os

from engine import (
    load_data, compute_risk_score, generate_advisory,
    generate_events, get_risk_level, UNITS, DISPLAY_LABELS,
)
from model import WiperTripPredictor
import templates as T

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Wiper Trip Predictor — 16A(78)-32",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Load CSS
# ---------------------------------------------------------------------------
CSS_PATH = os.path.join(os.path.dirname(__file__), "style.css")
with open(CSS_PATH) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Data & Train Model
# ---------------------------------------------------------------------------
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
with st.spinner("Training prediction model..."):
    ml_model, training_metrics = get_trained_model(df)

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
# Controls — Compact Top Row
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 1, 1])
with c1:
    speed = st.slider("Speed (s)", 0.1, 2.0, 0.5, 0.1, key="speed")
with c2:
    window_size = st.slider("History (pts)", 50, 300, 100, 10, key="window")
with c3:
    mode = st.radio("Mode", ["Auto", "Manual"], horizontal=True, key="mode_select")
    st.session_state.mode = mode
with c4:
    if st.button("▶ START", use_container_width=True, type="primary"):
        st.session_state.running = True
with c5:
    if st.button("■ STOP", use_container_width=True):
        st.session_state.running = False

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
            "Position", 50, len(df) - 1,
            st.session_state.idx, key="pos_slider",
        )
        if new_idx != st.session_state.idx:
            st.session_state.idx = new_idx
            st.session_state.risk_history = {}
            st.session_state.risk_scores = []

# ---------------------------------------------------------------------------
# Main Placeholder
# ---------------------------------------------------------------------------
main_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Dashboard Render
# ---------------------------------------------------------------------------
def render_dashboard(idx: int):
    row = df.iloc[idx]
    prev = df.iloc[max(0, idx - 1)]

    # Compute
    risk, details = compute_risk_score(
        df, idx, st.session_state.risk_history, ml_model=ml_model,
    )
    st.session_state.risk_scores.append(risk)
    advisory = generate_advisory(risk, details)

    for evt in generate_events(df, idx, None):
        st.session_state.event_log.insert(0, evt)
    st.session_state.event_log = st.session_state.event_log[:30]

    time_str = (
        row["Time"].strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(row["Time"], "strftime") else str(row["Time"])
    )
    level, color = get_risk_level(risk)
    probability_pct = risk * 100

    # ================================================================
    with main_placeholder.container():

        # ---- 1. HERO: The Big Answer ----
        st.markdown(
            T.prediction_hero(risk, level, advisory["recommendation"],
                              probability_pct),
            unsafe_allow_html=True,
        )

        # ---- 2. Context Bar ----
        st.markdown(
            T.context_bar(row["DEPTH"], time_str, st.session_state.mode),
            unsafe_allow_html=True,
        )

        # ---- 3. Two-Column Layout: Chart + Why ----
        chart_col, why_col = st.columns([3, 2])

        with chart_col:
            _render_risk_chart(idx, window_size)

        with why_col:
            st.markdown(
                T.why_panel(advisory),
                unsafe_allow_html=True,
            )

        # ---- 4. Model Details (collapsed by default) ----
        with st.expander("📊 Model & Sensor Details"):
            det_left, det_right = st.columns(2)

            with det_left:
                st.markdown(
                    T.section_title("Current Sensor Readings"),
                    unsafe_allow_html=True,
                )
                sensor_data = {
                    "Parameter": [],
                    "Value": [],
                    "Trend": [],
                }
                for key in ("WOB", "RPM", "TRQ", "ROP", "SPP", "FLOW_IN",
                            "DH_TRQ", "HOOKLOAD"):
                    if key in df.columns:
                        curr = row[key]
                        prev_val = prev[key]
                        if prev_val != 0:
                            pct = (curr - prev_val) / abs(prev_val) * 100
                            trend = f"▲ {pct:.0f}%" if pct > 2 else (
                                f"▼ {pct:.0f}%" if pct < -2 else "—"
                            )
                        else:
                            trend = "—"
                        sensor_data["Parameter"].append(
                            DISPLAY_LABELS.get(key, key)
                        )
                        sensor_data["Value"].append(
                            f"{curr:,.1f} {UNITS.get(key, '')}"
                        )
                        sensor_data["Trend"].append(trend)

                st.dataframe(
                    pd.DataFrame(sensor_data),
                    hide_index=True,
                    use_container_width=True,
                )

            with det_right:
                st.markdown(
                    T.section_title("Model Information"),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    T.model_info_compact(training_metrics),
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Risk Chart — Single, Clean
# ---------------------------------------------------------------------------
def _render_risk_chart(idx: int, win: int):
    """One clean risk-over-time chart."""
    start = max(0, idx - win)
    cd = df.iloc[start:idx + 1]
    rd = st.session_state.risk_scores[
        max(0, len(st.session_state.risk_scores) - win):
    ]
    x = cd["Time"] if hasattr(cd.iloc[0]["Time"], "strftime") else list(range(len(cd)))

    fig = go.Figure()

    rx = x.iloc[-len(rd):] if hasattr(x, "iloc") else list(range(len(rd)))

    # Risk area
    fig.add_trace(
        go.Scatter(
            x=rx, y=rd, mode="lines",
            line=dict(color="#38bdf8", width=2.5),
            fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.08)",
            showlegend=False,
            hovertemplate="Risk: %{y:.2f}<extra></extra>",
        )
    )

    # Threshold lines
    fig.add_hline(
        y=0.7, line_dash="dash", line_color="#ef4444", line_width=1,
        annotation_text="HIGH", annotation_position="right",
        annotation_font_color="#ef4444", annotation_font_size=11,
    )
    fig.add_hline(
        y=0.4, line_dash="dash", line_color="#f59e0b", line_width=1,
        annotation_text="MODERATE", annotation_position="right",
        annotation_font_color="#f59e0b", annotation_font_size=11,
    )

    fig.update_layout(
        title=dict(
            text="Wiper Trip Probability Over Time",
            font=dict(size=14, color="#94a3b8"),
            x=0.5,
        ),
        height=340,
        margin=dict(l=40, r=70, t=40, b=30),
        paper_bgcolor="#0f1629",
        plot_bgcolor="#0f1629",
        font=dict(color="#94a3b8", size=11, family="Inter, sans-serif"),
        hovermode="x unified",
        yaxis=dict(
            range=[0, 1],
            gridcolor="#1e293b",
            title=dict(text="Probability", font=dict(size=12)),
        ),
        xaxis=dict(
            gridcolor="#1e293b",
            showgrid=False,
        ),
    )

    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": False},
        key=f"risk_{idx}",
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
            '<div style="text-align:center;color:#475569;'
            'padding:16px;font-size:13px;">'
            'Press ▶ START to simulate real-time drilling</div>',
            unsafe_allow_html=True,
        )
