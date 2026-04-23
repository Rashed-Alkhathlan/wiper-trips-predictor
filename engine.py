"""
Drilling Advisory Engine
========================
Core logic for real-time wiper trip risk assessment.
Computes rolling features, MSE, composite risk score,
and generates contextual advisory recommendations.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column aliases
# ---------------------------------------------------------------------------
COL_MAP = {
    "Time": "Time",
    "Weight on Bit": "WOB",
    "ROP Depth/Hour": "ROP",
    "Top Drive RPM": "RPM",
    "Top Drive Torque (ft-lbs)": "TRQ",
    "Flow In": "FLOW_IN",
    "Pump Pressure": "SPP",
    "SPM Total": "SPM",
    "Pit Volume Active": "PIT_VOL",
    "Bit RPM": "BIT_RPM",
    "Depth Hole TVD": "DEPTH",
    "Differential Pressure": "DIFF_P",
    "Downhole Torque": "DH_TRQ",
    "MUD TEMP": "MUD_TEMP",
    # --- Full CSV extra columns ---
    "Block Position": "BLOCK_POS",
    "Hookload": "HOOKLOAD",
    "Slips Set": "SLIPS",
    "On Bottom": "ON_BOTTOM",
    "Gas Total - units": "GAS",
    "Return Flow": "RETURN_FLOW",
    "Pit G/L Active": "PIT_GL",
    "Trip Volume Active": "TRIP_VOL",
    "Trip G/L": "TRIP_GL",
    "Total Depth": "TOTAL_DEPTH",
    "RigEventCode": "RIG_EVENT",
    "Drill Mode": "DRILL_MODE",
    "MWD Inclination": "MWD_INC",
}

UNITS = {
    "WOB": "klbs",
    "ROP": "m/hr",
    "RPM": "rpm",
    "TRQ": "ft-lbs",
    "FLOW_IN": "gpm",
    "SPP": "psi",
    "SPM": "spm",
    "PIT_VOL": "bbl",
    "BIT_RPM": "rpm",
    "DEPTH": "m",
    "DIFF_P": "psi",
    "DH_TRQ": "ft-lbs",
    "MUD_TEMP": "°F",
    "HOOKLOAD": "klbs",
    "GAS": "units",
    "RETURN_FLOW": "gpm",
    "PIT_GL": "bbl/hr",
    "MWD_INC": "°",
}

# Display labels for the UI
DISPLAY_LABELS = {
    "WOB": "Weight on Bit",
    "ROP": "Rate of Penetration",
    "RPM": "Top Drive RPM",
    "TRQ": "Surface Torque",
    "FLOW_IN": "Flow Rate",
    "SPP": "Standpipe Pressure",
    "DEPTH": "Hole Depth TVD",
    "DH_TRQ": "Downhole Torque",
    "DIFF_P": "Differential Pressure",
    "MUD_TEMP": "Mud Temperature",
    "HOOKLOAD": "Hookload",
    "RETURN_FLOW": "Return Flow",
    "PIT_GL": "Pit Gain/Loss",
    "MWD_INC": "MWD Inclination",
}

# Bit diameter in inches (assumed for MSE calculation)
BIT_DIAMETER_IN = 8.5

# Sentinel value used for missing data in the raw CSV
_SENTINEL = -999.25

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(path: str, subsample: int = 6) -> pd.DataFrame:
    """Load CSV, rename columns, and subsample for performance.

    Supports both simplified (14 cols) and full (36 cols) CSVs.
    Handles -999.25 sentinel values in the full dataset.
    """
    df = pd.read_csv(path)
    df.rename(columns=COL_MAP, inplace=True)
    df = df.iloc[::subsample].reset_index(drop=True)

    # Parse time
    df["Time"] = pd.to_datetime(df["Time"], format="mixed", dayfirst=False)

    # Replace sentinel value with NaN across all numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].replace(_SENTINEL, np.nan)

    # Forward-fill sensor columns (sensors hold last reading)
    sensor_cols = [
        c for c in [
            "WOB", "ROP", "RPM", "TRQ", "FLOW_IN", "SPP", "SPM",
            "PIT_VOL", "BIT_RPM", "DEPTH", "DIFF_P", "DH_TRQ",
            "MUD_TEMP", "HOOKLOAD", "BLOCK_POS", "GAS",
            "RETURN_FLOW", "PIT_GL", "TRIP_VOL", "TRIP_GL",
            "TOTAL_DEPTH", "MWD_INC",
        ] if c in df.columns
    ]
    df[sensor_cols] = df[sensor_cols].ffill().fillna(0)

    # Clean up obviously spurious values (sensor spikes)
    for col in ["FLOW_IN", "SPM", "BIT_RPM"]:
        if col in df.columns:
            q99 = df[col].quantile(0.99)
            if q99 > 0:
                df.loc[df[col] > q99 * 3, col] = np.nan
                df[col] = df[col].ffill()

    # Boolean / categorical columns
    for col in ["SLIPS", "ON_BOTTOM"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

WINDOW = 10  # rolling window size (in rows)


def compute_rolling_features(df: pd.DataFrame, idx: int) -> dict:
    """Compute rolling averages and trend indicators up to index `idx`."""
    start = max(0, idx - WINDOW)
    window = df.iloc[start : idx + 1]
    prev_start = max(0, idx - 2 * WINDOW)
    prev_window = df.iloc[prev_start : start] if start > 0 else window

    feats = {}
    for col in ["ROP", "TRQ", "SPP", "FLOW_IN", "DH_TRQ"]:
        current_mean = window[col].mean()
        prev_mean = prev_window[col].mean() if len(prev_window) > 0 else current_mean
        feats[f"{col}_mean"] = current_mean
        feats[f"{col}_prev_mean"] = prev_mean
        if prev_mean != 0:
            feats[f"{col}_pct_change"] = (current_mean - prev_mean) / abs(prev_mean) * 100
        else:
            feats[f"{col}_pct_change"] = 0.0

    return feats


def compute_mse(row: pd.Series) -> float:
    """Mechanical Specific Energy — an indicator of drilling efficiency.

    MSE = (480 * WOB * RPM * Torque) / (D^2 * ROP)
    Simplified version. Higher MSE = less efficient = more risk.
    """
    wob = max(row["WOB"], 0.1)
    rpm = max(row["RPM"], 0.1)
    trq = max(row["TRQ"], 0.1)
    rop = max(row["ROP"], 0.1)
    d = BIT_DIAMETER_IN

    mse = (480.0 * trq) / (d ** 2 * rop) + (4.0 * wob) / (np.pi * d ** 2)
    return mse


# ---------------------------------------------------------------------------
# Risk Score
# ---------------------------------------------------------------------------

def _normalize(value: float, vmin: float, vmax: float) -> float:
    """Clamp and normalize to [0, 1]."""
    if vmax <= vmin:
        return 0.0
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


def compute_risk_score(df: pd.DataFrame, idx: int, history: dict,
                       ml_model=None) -> tuple[float, dict]:
    """Compute wiper trip risk score using ML model or fallback rules.

    If ml_model is provided, uses the trained ensemble.
    Otherwise falls back to the weighted rule-based formula.

    Returns (risk_score, details_dict).
    """
    # --- ML Model Path ---
    if ml_model is not None and ml_model.is_trained:
        result = ml_model.predict(df, idx)
        risk = result["risk_score"]

        # Smooth with EMA
        alpha = 0.35
        prev_risk = history.get("prev_risk", risk)
        risk = alpha * risk + (1 - alpha) * prev_risk
        risk = round(max(0.0, min(1.0, risk)), 3)
        history["prev_risk"] = risk

        details = result["details"]
        details["rf_probability"] = result["rf_probability"]
        details["if_anomaly_score"] = result["if_anomaly_score"]
        details["feature_importances"] = result.get("feature_importances", {})
        return risk, details

    # --- Fallback: Rule-based Path ---
    row = df.iloc[idx]
    feats = compute_rolling_features(df, idx)

    mse = compute_mse(row)
    mse_norm = _normalize(mse, 10, 120)

    trq_change = feats["TRQ_pct_change"]
    trq_norm = _normalize(trq_change, -5, 25)

    spp_change = feats["SPP_pct_change"]
    spp_norm = _normalize(spp_change, -5, 20)

    rop_change = feats["ROP_pct_change"]
    rop_drop_norm = _normalize(-rop_change, -10, 30)

    flow_mean = feats["FLOW_IN_mean"]
    flow_prev = feats["FLOW_IN_prev_mean"]
    if flow_prev > 0:
        flow_imb = abs(flow_mean - flow_prev) / flow_prev
    else:
        flow_imb = 0.0
    flow_norm = _normalize(flow_imb, 0, 0.15)

    risk = (
        0.30 * mse_norm
        + 0.20 * trq_norm
        + 0.20 * spp_norm
        + 0.20 * rop_drop_norm
        + 0.10 * flow_norm
    )

    alpha = 0.3
    prev_risk = history.get("prev_risk", risk)
    risk = alpha * risk + (1 - alpha) * prev_risk
    risk = round(max(0.0, min(1.0, risk)), 2)

    details = {
        "mse": round(mse, 1),
        "mse_norm": round(mse_norm, 2),
        "trq_change_pct": round(trq_change, 1),
        "trq_norm": round(trq_norm, 2),
        "spp_change_pct": round(spp_change, 1),
        "spp_norm": round(spp_norm, 2),
        "rop_change_pct": round(rop_change, 1),
        "rop_drop_norm": round(rop_drop_norm, 2),
        "flow_imbalance": round(flow_imb * 100, 1),
        "flow_norm": round(flow_norm, 2),
    }

    history["prev_risk"] = risk
    return risk, details


# ---------------------------------------------------------------------------
# Advisory & Decision Logic
# ---------------------------------------------------------------------------

def get_risk_level(risk: float) -> tuple[str, str]:
    """Return (level_label, css_color)."""
    if risk > 0.7:
        return "HIGH", "#ef4444"
    elif risk > 0.4:
        return "MODERATE", "#f59e0b"
    else:
        return "LOW", "#22c55e"


def generate_advisory(risk: float, details: dict) -> dict:
    """Generate contextual advisory based on risk score and parameter details."""
    level, color = get_risk_level(risk)

    # Build reasons list
    reasons = []
    if details["trq_change_pct"] > 5:
        reasons.append(f"Surface torque increased by {details['trq_change_pct']:.0f}%")
    if details["spp_change_pct"] > 5:
        reasons.append(f"Standpipe pressure increased by {details['spp_change_pct']:.0f}%")
    if details["rop_change_pct"] < -5:
        reasons.append(f"ROP decreased by {abs(details['rop_change_pct']):.0f}%")
    if details["flow_imbalance"] > 5:
        reasons.append(f"Flow imbalance detected ({details['flow_imbalance']:.0f}%)")
    if details["mse_norm"] > 0.6:
        reasons.append(f"MSE elevated ({details['mse']:.0f} ksi) — reduced drilling efficiency")

    if not reasons:
        reasons.append("All parameters within normal operating range")

    # Interpretation
    if risk > 0.7:
        interpretation = "Likely cuttings accumulation and inadequate hole cleaning. Wellbore conditions are deteriorating."
        actions = [
            "Increase flow rate by 10-15%",
            "Reduce ROP to improve hole cleaning",
            "Prepare for wiper trip if condition persists",
            "Monitor downhole torque closely",
        ]
        recommendation = "Perform Wiper Trip"
    elif risk > 0.4:
        interpretation = "Early signs of hole cleaning deterioration detected. Increased monitoring recommended."
        actions = [
            "Increase flow rate by 5-10%",
            "Monitor torque and pressure trends",
            "Maintain current ROP or reduce slightly",
            "Prepare contingency for wiper trip",
        ]
        recommendation = "Increase Flow / Monitor"
    else:
        interpretation = "Wellbore conditions are stable. Normal drilling operations can continue."
        actions = [
            "Continue current drilling parameters",
            "Maintain flow rate",
            "Standard monitoring protocol",
        ]
        recommendation = "Continue Drilling"

    confidence = min(0.95, 0.60 + risk * 0.35 + len(reasons) * 0.03)

    return {
        "level": level,
        "color": color,
        "recommendation": recommendation,
        "reasons": reasons,
        "interpretation": interpretation,
        "actions": actions,
        "confidence": round(confidence, 2),
    }


# ---------------------------------------------------------------------------
# Event Log Generator
# ---------------------------------------------------------------------------

def generate_events(df: pd.DataFrame, idx: int, prev_feats: dict | None) -> list[dict]:
    """Detect threshold-crossing events and return log entries."""
    if idx < WINDOW:
        return []

    feats = compute_rolling_features(df, idx)
    row = df.iloc[idx]
    time_str = row["Time"].strftime("%H:%M:%S") if hasattr(row["Time"], "strftime") else str(row["Time"])

    events = []

    # Torque spike
    if feats["TRQ_pct_change"] > 8:
        events.append({
            "time": time_str,
            "message": f"Surface torque increased {feats['TRQ_pct_change']:.0f}%",
            "severity": "warning",
        })
    elif feats["TRQ_pct_change"] > 15:
        events.append({
            "time": time_str,
            "message": f"Surface torque spike: +{feats['TRQ_pct_change']:.0f}%",
            "severity": "critical",
        })

    # Pressure increase
    if feats["SPP_pct_change"] > 5:
        events.append({
            "time": time_str,
            "message": f"Standpipe pressure increased {feats['SPP_pct_change']:.0f}%",
            "severity": "warning",
        })

    # ROP drop
    if feats["ROP_pct_change"] < -15:
        events.append({
            "time": time_str,
            "message": f"ROP decreased {abs(feats['ROP_pct_change']):.0f}%",
            "severity": "warning",
        })

    # Downhole torque increase
    if feats.get("DH_TRQ_pct_change", 0) > 10:
        events.append({
            "time": time_str,
            "message": f"Downhole torque increased {feats['DH_TRQ_pct_change']:.0f}%",
            "severity": "warning",
        })

    return events


# ---------------------------------------------------------------------------
# Trend Arrow Helper
# ---------------------------------------------------------------------------

def trend_arrow(current: float, previous: float, threshold_pct: float = 2.0) -> str:
    """Return trend arrow based on % change."""
    if previous == 0:
        return ""
    pct = (current - previous) / abs(previous) * 100
    if pct > threshold_pct:
        return "▲"
    elif pct < -threshold_pct:
        return "▼"
    return "—"


def trend_color(current: float, previous: float, threshold_pct: float = 2.0) -> str:
    """Return CSS color for trend direction."""
    if previous == 0:
        return "#94a3b8"
    pct = (current - previous) / abs(previous) * 100
    if pct > threshold_pct:
        return "#ef4444"
    elif pct < -threshold_pct:
        return "#22c55e"
    return "#94a3b8"
