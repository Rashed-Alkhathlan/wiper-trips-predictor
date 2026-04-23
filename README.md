# 🛢️ Real-Time Drilling Advisory System — Wiper Trip Predictor

An AI-powered real-time drilling advisory system that uses machine learning to predict wiper trip necessity from live drilling parameters. Built on a two-model ensemble (**Gradient Boosted Trees + Isolation Forest**) trained on **real ground-truth labels** mined from 163 daily drilling reports, the system provides actionable operational recommendations through an industrial-grade Streamlit dashboard.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Data Sources](#data-sources)
- [Machine Learning Model](#machine-learning-model)
  - [Model Architecture](#model-architecture)
  - [Ground-Truth Label Mining](#ground-truth-label-mining)
  - [Feature Engineering](#feature-engineering)
  - [Training Pipeline](#training-pipeline)
  - [Model Performance](#model-performance)
- [Dashboard Components](#dashboard-components)
- [Risk Scoring System](#risk-scoring-system)
- [Advisory Decision Logic](#advisory-decision-logic)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Configuration & Tuning](#configuration--tuning)
- [Technical Details](#technical-details)
- [Dependencies](#dependencies)

---

## Overview

During drilling operations, cuttings and debris can accumulate in the wellbore, leading to stuck pipe, increased torque, pressure buildup, and loss of rate of penetration. **Wiper trips** — pulling the drill string out of the hole to clean the wellbore — are critical but costly operations. The key challenge is knowing *when* a wiper trip is necessary.

This system solves that problem by:

1. **Streaming real drilling data** from well 16A(78)-32 (608,000+ rows at 10-second intervals, 36 sensor channels)
2. **Mining ground-truth labels** from 163 daily drilling report PDFs (125 real events across 54 dates)
3. **Training ML models** on 77 engineered features using a GBT + Isolation Forest ensemble
4. **Computing real-time risk scores** with contextual recommendations and action items

---

## Features

| Feature | Description |
|---|---|
| **ML Ensemble Prediction** | Gradient Boosted Trees + Isolation Forest anomaly detector |
| **Real Ground-Truth Labels** | 125 events mined from 163 PDF daily drilling reports |
| **77 Engineered Features** | Rolling stats, derivatives, cross-ratios, lags, MSE, hookload, pit G/L, gas |
| **36-Channel Sensor Data** | Full dataset with hookload, block position, gas, return flow, pit gain/loss |
| **Real-Time Streaming** | Simulates live drilling with smooth, flicker-free updates |
| **Industrial Dark Theme** | Professional UI mimicking real drilling control systems |
| **Live Parameter Monitoring** | WOB, RPM, Torque, ROP, Pressure, Flow with trend indicators |
| **Time-Series Visualization** | 4 synchronized Plotly charts with rolling averages |
| **Contextual Advisory** | AI-generated recommendations with confidence scores |
| **Feature Importance** | Visual display of what drives the model's predictions |
| **Event Detection** | Automatic logging of threshold crossings and anomalies |
| **Model Transparency** | Full metrics display (AUC-ROC, Precision, Recall, F1) |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│  DATA SOURCES                                           │
│                                                         │
│  ┌───────────────────────┐  ┌────────────────────────┐  │
│  │ Full CSV Dataset      │  │ Daily Drilling Reports │  │
│  │ 608K rows × 36 cols   │  │ 163 PDFs (Oct–Jan)     │  │
│  │ 10-second intervals   │  │ Real operational logs  │  │
│  └───────────┬───────────┘  └──────────┬─────────────┘  │
└──────────────┼──────────────────────────┼───────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  engine.py           │    │  report_parser.py        │
│  load_data()         │    │  parse_all_reports()     │
│  36 cols → cleaned   │    │  build_label_series()    │
│  sentinel handling   │    │  125 events → labels     │
└──────────┬───────────┘    └──────────┬───────────────┘
           │                           │
           ▼                           ▼
┌─────────────────────────────────────────────────────────┐
│  ML Training Pipeline (model.py)                        │
│                                                         │
│  Feature Engineering → 77 features                      │
│  Label Generation → Real labels + pseudo-label fallback │
│                                                         │
│  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │ GradientBoostingCls │  │ Isolation Forest         │  │
│  │ 200 trees, depth 6  │  │ 100 estimators, 15%      │  │
│  │ learning_rate=0.1   │  │ contamination            │  │
│  └─────────────────────┘  └──────────────────────────┘  │
│                                                         │
│  Ensemble: 0.65 × GBT + 0.35 × IF                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Streamlit Dashboard (app.py)                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Top Bar: Well · Rig · Depth · Time · Risk Badge   │  │
│  ├───────┬────────────────────┬──────────────────────┤  │
│  │ Live  │  Trend Charts      │ Advisory Engine +    │  │
│  │ Params│  (4 stacked plots) │ ML Scores + Feature  │  │
│  │       │                    │ Importance           │  │
│  ├───────┴────────────────────┴──────────────────────┤  │
│  │ Event Log              │ Model Information        │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Data Sources

### 1. Full Sensor Dataset
- **File**: `16A(78)-32_time_data_10s_intervals.csv`
- **Size**: 608,680 rows × 36 columns
- **Interval**: 10-second telemetry snapshots
- **Duration**: Full drilling run of well 16A(78)-32 (Oct 2020 – Jan 2021)

#### Key Sensor Channels (36 total)

| Channel | Internal Name | Why It Matters |
|---|---|---|
| Weight on Bit | WOB | Axial force applied to the bit |
| Hookload | HOOKLOAD | Drag/friction — direct hole condition indicator |
| Block Position | BLOCK_POS | Pipe movement patterns (tripping, connections) |
| Return Flow | RETURN_FLOW | Loss/gain detection — wellbore instability |
| Pit G/L Active | PIT_GL | Fluid balance shifts — hole cleaning health |
| Gas Total | GAS | Kick/influx detection |
| Trip Volume | TRIP_VOL | Fluid tracking during trips |
| On Bottom | ON_BOTTOM | Drilling vs. off-bottom state |
| Slips Set | SLIPS | Connection vs. drilling state |
| Drill Mode | DRILL_MODE | Operational mode encoding |

### 2. Daily Drilling Reports
- **Directory**: `16A(78)-32_Daily_Reports/drilling/`
- **Count**: 163 PDF reports
- **Date Range**: October 21, 2020 – January 3, 2021
- **Content**: Timestamped operational logs with drilling parameters, events, and crew notes

#### Mined Event Types

| Event Type | Count | Severity Weight |
|---|---|---|
| Trip Out / POOH | 109 | 0.75 |
| Reaming | 7 | 0.85 |
| Short Trip | 3 | 1.00 |
| Wash | 2 | 0.60 |
| High Torque | 2 | 0.80 |
| Ream Shoe | 1 | 0.70 |
| Drag | 1 | 0.70 |
| **Total** | **125** | — |

---

## Machine Learning Model

### Model Architecture

| Component | Type | Purpose |
|---|---|---|
| **Gradient Boosted Trees** | Supervised Classifier | Learns sequential patterns from real-labeled drilling data. 200 trees, depth 6, learning rate 0.1. |
| **Isolation Forest** | Unsupervised Anomaly Detector | Identifies unusual parameter combinations. 100 estimators, 15% contamination. |
| **Ensemble** | Weighted Average | Final risk = `0.65 × GBT_prob + 0.35 × IF_score`, smoothed with EMA (α=0.35). |

#### Why GBT over Random Forest?
- Better at learning sequential/temporal feature patterns
- Built-in regularization via learning rate and subsampling
- More efficient feature importance via gradient-based gain
- Faster convergence on 60K+ training samples

### Ground-Truth Label Mining

The `report_parser.py` module extracts real operational events from 163 PDF drilling reports using PyMuPDF:

1. **PDF Parsing**: Extract text from each report page
2. **Pattern Matching**: 14 regex patterns for wiper trip, ream, short trip, POOH, tight spot, high torque, stuck pipe, pack-off, overpull, drag
3. **Time Extraction**: Parse `HH:MM HH:MM` time ranges from event lines
4. **Depth Extraction**: Parse `X,XXX to Y,YYY` depth ranges
5. **Label Mapping**: Align events to 10-second time series with ±30min expansion + 2-hour approach window

**Label Strategy** (priority order):
1. **Real labels** from reports when available (125 events, weight ≥ 0.7)
2. **Pseudo-labels** from domain heuristics as supplement for uncovered periods
3. **Blended**: Real labels dominate; pseudo-labels add only high-confidence (>0.5) points

### Feature Engineering

The pipeline transforms **36 raw sensor channels** into **77 predictive features** across 10 categories:

#### Base Features (8+5 = 13)
- 8 core drilling parameters: WOB, ROP, RPM, TRQ, SPP, FLOW_IN, DH_TRQ, DIFF_P
- 5 extended channels: HOOKLOAD, GAS, RETURN_FLOW, PIT_GL, TRIP_GL
- MSE (Mechanical Specific Energy) computed from WOB, TRQ, RPM, ROP

#### Rolling Statistics (24)
- Rolling means (windows 10, 30) for all 8 base parameters
- Rolling std (windows 10, 30) for ROP, TRQ, SPP, DH_TRQ

#### Rate of Change (5)
- First-order differences for ROP, TRQ, SPP, FLOW_IN, DH_TRQ

#### Trend Comparisons (3)
- `TRQ_pct_10v30`, `SPP_pct_10v30`, `ROP_pct_10v30`

#### Cross-Feature Ratios (5)
- `TRQ_ROP_ratio`, `MSE_x_RPM`, `DH_TRQ_diff`, `Flow_pressure_ratio`, `WOB_TRQ_ratio`

#### Lagged Values (6)
- 5-step and 10-step lags for ROP, TRQ, SPP

#### MSE Features (4)
- `MSE_mean_10`, `MSE_mean_30`, `MSE_std_10`, `MSE_roc`

#### Hookload Features (4)
- Mean, std, rate-of-change, drag estimate (delta from rolling baseline)

#### Block Position Features (3)
- Velocity, acceleration, absolute velocity (pipe movement patterns)

#### Fluid Balance Features (7)
- Pit G/L rolling sums (10, 30), max absolute G/L
- Flow ratio (return/input), flow imbalance
- Gas rolling max, gas rate-of-change
- Trip G/L rolling sum

#### Operational State (1)
- On-bottom run duration (cumulative counter)

### Training Pipeline

```
1. Load full 36-column CSV (subsample=10 → 60,868 rows)
2. Engineer features → 77 columns
3. Parse 163 PDF reports → 125 ground-truth events
4. Map events to time series → binary labels (18.3% positive rate)
5. StandardScaler normalization
6. 80/20 stratified train/test split
7. GBT: fit on training set (200 trees, depth 6, lr=0.1)
8. IF: fit on full dataset (unsupervised)
9. Evaluate on held-out test set
10. Cache models via @st.cache_resource
```

### Model Performance

| Metric | Value |
|---|---|
| **Training Samples** | 60,868 |
| **Features** | 77 |
| **Label Source** | Report-Mined (125 events) |
| **Positive Rate** | 18.3% |
| **AUC-ROC** | 0.9989 |
| **Precision** | 0.990 |
| **Recall** | 0.930 |
| **F1-Score** | 0.959 |
| **Accuracy** | 0.986 |

> **Why metrics are strong**: Unlike the previous pseudo-label approach where the model trained on the same features used to generate labels, the real report-mined labels provide independent ground truth. The model learns genuine sensor patterns that precede actual operational events documented by drilling engineers on-site.

---

## Dashboard Components

### Top Status Bar
- **Well**: 16A(78)-32 | **Rig**: Demo Rig-01 | **Depth TVD** | **Time** | **Status**: DRILLING
- **Model**: GBT + IF Ensemble
- **Wiper Trip Risk**: Color-coded badge (green < 0.4, yellow 0.4–0.7, red > 0.7)

### Live Parameters Panel
Six large-format metric cards with trend arrows: WOB, RPM, Torque, ROP, SPP, Flow Rate

### Time-Series Trend Charts
Four stacked Plotly charts: ROP, Torque, Pressure, Risk Score

### Advisory Engine
Contextual recommendations with analysis, interpretation, action items, and confidence scores

### ML Model Output Panel
- **Gradient Boost Probability**: Supervised model confidence
- **Isolation Forest Score**: Anomaly detection score
- **Ensemble (0.65 GBT + 0.35 IF)**: Weighted combination

### Model Information Panel
Training summary with label source, event count, AUC-ROC, Precision, Recall, F1, Accuracy

---

## Risk Scoring System

```
risk_raw = 0.65 × GBT.predict_proba(X)[1]
         + 0.35 × normalized(IF.decision_function(X))

risk_smoothed = 0.35 × risk_raw + 0.65 × previous_risk   (EMA, α=0.35)
```

| Score Range | Level | Color | Action |
|---|---|---|---|
| 0.00 – 0.40 | LOW | 🟢 Green | Continue drilling |
| 0.40 – 0.70 | MODERATE | 🟡 Yellow | Increase flow / monitor |
| 0.70 – 1.00 | HIGH | 🔴 Red | Perform wiper trip |

---

## Project Structure

```
wiper-trips-predictor/
├── app.py                  # Streamlit dashboard UI + streaming loop
├── engine.py               # Data loading, risk scoring, advisory logic
├── model.py                # ML model: GBT + IF ensemble, 77-feature pipeline
├── report_parser.py        # PDF report mining for ground-truth labels
├── templates.py            # HTML template generators for dashboard panels
├── style.css               # Industrial dark-theme CSS with anti-jitter layout
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── 16A(78)-32_time_data_10s_intervals.csv           # Full 36-column dataset
├── 16A(78)-32_time_data_10s_intervals_simplified.csv # Simplified 14-column fallback
└── 16A(78)-32_Daily_Reports/                         # 163 PDF drilling reports
    └── 16A(78)-32_Daily_Reports/
        ├── drilling/       # 163 daily drilling report PDFs
        └── completion/     # Completion phase reports
```

### File Responsibilities

| File | Responsibility |
|---|---|
| **`app.py`** | Streamlit layout, Plotly charts, streaming loop, session state |
| **`engine.py`** | Data loading (36 cols), column mapping, risk scoring, advisory generation |
| **`model.py`** | `WiperTripPredictor` class, 77-feature engineering, GBT/IF training, prediction |
| **`report_parser.py`** | PDF parsing, event extraction, label mapping to time series |
| **`templates.py`** | HTML generators for top bar, metric cards, advisory panel, model info |
| **`style.css`** | Dark theme, anti-jitter tabular-nums, fixed container sizing |

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Steps

```bash
# 1. Navigate to the project directory
cd wiper-trips-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
python -m streamlit run app.py
```

The application will:
1. Load the full 36-column CSV dataset (60K rows after subsampling)
2. Parse 163 PDF reports for ground-truth labels (~2 seconds)
3. Train GBT + IF models on 77 features (~20 seconds on first run, cached afterward)
4. Open the dashboard at `http://localhost:8501`

---

## Usage Guide

### Auto Mode (Real-Time Streaming)
1. Set the **Refresh Interval** slider (default: 0.5s per frame)
2. Ensure **Mode** is set to "Auto"
3. Click **▶ START** to begin streaming
4. Click **■ STOP** to pause

### Manual Mode (Step-by-Step)
1. Switch **Mode** to "Manual"
2. Use **STEP ▶** / **STEP x10 ▶▶** to advance
3. Drag the **Data Position** slider to jump to any point

---

## Configuration & Tuning

### Risk Sensitivity
In `model.py`, adjust:
- **GBT `max_depth`**: Higher = more sensitive (default: 6)
- **GBT `n_estimators`**: More trees = more robust (default: 200)
- **IF `contamination`**: Higher = more anomalies flagged (default: 0.15)
- **Ensemble weights**: Change `0.65 / 0.35` split in `predict()`

### Label Thresholds
In `report_parser.py`:
- **Event weight threshold**: Currently ≥ 0.7 (filter low-confidence events)
- **Time window expansion**: ±30 min around reported events
- **Approach window**: 2 hours before events (early warning training)

### EMA Smoothing
In `engine.py` → `compute_risk_score()`:
- **α = 0.35**: Higher = more reactive; lower = smoother

---

## Technical Details

### Mechanical Specific Energy (MSE)
```
MSE = (480 × Torque) / (D² × ROP) + (4 × WOB) / (π × D²)
```
Where D = bit diameter (8.5 inches). Higher MSE = drilling inefficiency.

### Streaming Architecture
Uses a **placeholder-based loop** instead of full-page reruns for flicker-free updates.

### Caching Strategy
- **`@st.cache_data`**: Dataset loading — cached until file changes
- **`@st.cache_resource`**: ML model training — trained once, persisted across reruns

---

## Dependencies

| Package | Purpose |
|---|---|
| streamlit | Dashboard framework |
| pandas | Data manipulation and time-series processing |
| plotly | Interactive time-series charts |
| numpy | Numerical computations |
| scikit-learn | GradientBoostingClassifier, IsolationForest, StandardScaler |
| pymupdf | PDF report parsing for ground-truth label extraction |

```bash
pip install -r requirements.txt
```
