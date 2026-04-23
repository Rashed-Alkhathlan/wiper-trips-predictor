"""
Wiper Trip ML Predictor
========================
Two-model ensemble for wiper trip risk prediction:
  1. Gradient Boosted Trees — trained on real labels from daily reports
     (with pseudo-label fallback when no reports available)
  2. Isolation Forest — unsupervised anomaly detection

Final score: 0.65 × GBT_probability + 0.35 × IF_anomaly_score
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Bit diameter for MSE calculation
BIT_DIAMETER_IN = 8.5


# ---------------------------------------------------------------------------
# Feature Engineering Pipeline
# ---------------------------------------------------------------------------

def compute_mse_series(df: pd.DataFrame) -> pd.Series:
    """Compute Mechanical Specific Energy for entire dataframe."""
    wob = df["WOB"].clip(lower=0.1)
    rpm = df["RPM"].clip(lower=0.1)
    trq = df["TRQ"].clip(lower=0.1)
    rop = df["ROP"].clip(lower=0.1)
    d = BIT_DIAMETER_IN
    mse = (480.0 * trq) / (d**2 * rop) + (4.0 * wob) / (np.pi * d**2)
    return mse


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer 50+ features from raw drilling parameters.

    Features include:
    - Raw parameters (base + extended)
    - Rolling means (windows: 10, 30)
    - Rolling standard deviations
    - Rate of change (first derivative)
    - Cross-feature ratios
    - Lagged values
    - Hookload/drag features (if available)
    - Pit gain/loss features (if available)
    - Block position derivative (pipe velocity)
    """
    feat = pd.DataFrame(index=df.index)

    # Base columns (always available)
    base_cols = ["WOB", "ROP", "RPM", "TRQ", "SPP", "FLOW_IN", "DH_TRQ", "DIFF_P"]

    # Extended columns (full CSV only — add if present)
    ext_cols = ["HOOKLOAD", "GAS", "RETURN_FLOW", "PIT_GL", "TRIP_GL", "MWD_INC"]
    available_ext = [c for c in ext_cols if c in df.columns]

    # --- Raw values ---
    for col in base_cols:
        feat[col] = df[col]
    for col in available_ext:
        feat[col] = df[col]

    # --- MSE ---
    feat["MSE"] = compute_mse_series(df)

    # --- Rolling means (10, 30) ---
    for win in [10, 30]:
        for col in base_cols:
            feat[f"{col}_mean_{win}"] = df[col].rolling(win, min_periods=1).mean()

    # --- Rolling std (volatility) ---
    for col in ["ROP", "TRQ", "SPP", "DH_TRQ"]:
        feat[f"{col}_std_10"] = df[col].rolling(10, min_periods=1).std().fillna(0)
        feat[f"{col}_std_30"] = df[col].rolling(30, min_periods=1).std().fillna(0)

    # --- Rate of change (derivative) ---
    for col in ["ROP", "TRQ", "SPP", "FLOW_IN", "DH_TRQ"]:
        feat[f"{col}_roc"] = df[col].diff().fillna(0)

    # --- Percent change over rolling windows ---
    for col in ["ROP", "TRQ", "SPP"]:
        mean_10 = df[col].rolling(10, min_periods=1).mean()
        mean_30 = df[col].rolling(30, min_periods=1).mean()
        feat[f"{col}_pct_10v30"] = ((mean_10 - mean_30) / mean_30.clip(lower=1)).fillna(0) * 100

    # --- Cross-feature ratios ---
    feat["TRQ_ROP_ratio"] = (df["TRQ"] / df["ROP"].clip(lower=0.1))
    feat["MSE_x_RPM"] = feat["MSE"] * df["RPM"]
    feat["DH_TRQ_diff"] = df["DH_TRQ"] - df["TRQ"]
    feat["Flow_pressure_ratio"] = df["FLOW_IN"] / df["SPP"].clip(lower=1)
    feat["WOB_TRQ_ratio"] = df["WOB"] / df["TRQ"].clip(lower=1)

    # --- Lagged values ---
    for col in ["ROP", "TRQ", "SPP"]:
        feat[f"{col}_lag_5"] = df[col].shift(5).fillna(df[col])
        feat[f"{col}_lag_10"] = df[col].shift(10).fillna(df[col])

    # --- MSE rolling features ---
    feat["MSE_mean_10"] = feat["MSE"].rolling(10, min_periods=1).mean()
    feat["MSE_mean_30"] = feat["MSE"].rolling(30, min_periods=1).mean()
    feat["MSE_std_10"] = feat["MSE"].rolling(10, min_periods=1).std().fillna(0)
    feat["MSE_roc"] = feat["MSE"].diff().fillna(0)

    # ===========================================================
    # Extended features from full CSV
    # ===========================================================

    # --- Hookload features (drag/friction indicator) ---
    if "HOOKLOAD" in df.columns:
        feat["HOOKLOAD_mean_10"] = df["HOOKLOAD"].rolling(10, min_periods=1).mean()
        feat["HOOKLOAD_std_10"] = df["HOOKLOAD"].rolling(10, min_periods=1).std().fillna(0)
        feat["HOOKLOAD_roc"] = df["HOOKLOAD"].diff().fillna(0)
        # Drag estimate: delta from rolling mean
        feat["HOOKLOAD_drag"] = (
            df["HOOKLOAD"] - df["HOOKLOAD"].rolling(30, min_periods=1).mean()
        ).fillna(0)

    # --- Block position features (pipe movement) ---
    if "BLOCK_POS" in df.columns:
        feat["BLOCK_VEL"] = df["BLOCK_POS"].diff().fillna(0)
        feat["BLOCK_ACCEL"] = feat["BLOCK_VEL"].diff().fillna(0)
        feat["BLOCK_VEL_abs"] = feat["BLOCK_VEL"].abs()

    # --- Pit gain/loss (hole cleaning indicator) ---
    if "PIT_GL" in df.columns:
        feat["PIT_GL_sum_10"] = df["PIT_GL"].rolling(10, min_periods=1).sum().fillna(0)
        feat["PIT_GL_sum_30"] = df["PIT_GL"].rolling(30, min_periods=1).sum().fillna(0)
        feat["PIT_GL_abs_max_10"] = (
            df["PIT_GL"].abs().rolling(10, min_periods=1).max().fillna(0)
        )

    # --- Return flow ratio (loss/gain detection) ---
    if "RETURN_FLOW" in df.columns and "FLOW_IN" in df.columns:
        feat["FLOW_RATIO"] = (
            df["RETURN_FLOW"] / df["FLOW_IN"].clip(lower=1)
        ).fillna(1.0)
        feat["FLOW_IMBALANCE"] = (df["RETURN_FLOW"] - df["FLOW_IN"]).fillna(0)

    # --- Gas proximity ---
    if "GAS" in df.columns:
        feat["GAS_max_10"] = df["GAS"].rolling(10, min_periods=1).max().fillna(0)
        feat["GAS_roc"] = df["GAS"].diff().fillna(0)

    # --- Trip volume features ---
    if "TRIP_GL" in df.columns:
        feat["TRIP_GL_sum_10"] = df["TRIP_GL"].rolling(10, min_periods=1).sum().fillna(0)

    # --- On-bottom duration ---
    if "ON_BOTTOM" in df.columns:
        feat["ON_BOTTOM_run"] = (
            df["ON_BOTTOM"]
            .groupby((df["ON_BOTTOM"] != df["ON_BOTTOM"].shift()).cumsum())
            .cumcount()
        )

    # --- MWD Inclination features (critical for hole cleaning) ---
    # Higher angles (>30°) dramatically worsen cuttings transport;
    # near-horizontal (>60°) requires aggressive wiper trip management.
    if "MWD_INC" in df.columns:
        feat["MWD_INC"] = df["MWD_INC"]
        feat["INC_HIGH_ANGLE"] = (df["MWD_INC"] > 30).astype(float)
        feat["INC_CRITICAL"] = (df["MWD_INC"] > 60).astype(float)
        # Interaction: torque at high angle is much worse than at vertical
        feat["INC_x_TRQ"] = df["MWD_INC"] * df["TRQ"] / 1000.0
        # Interaction: MSE at high angle = compounded inefficiency
        feat["INC_x_MSE"] = df["MWD_INC"] * feat["MSE"] / 1000.0

    # Replace inf/nan
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)

    return feat


# ---------------------------------------------------------------------------
# Label Generation (Real + Pseudo)
# ---------------------------------------------------------------------------

def generate_labels(df: pd.DataFrame, feat: pd.DataFrame) -> pd.Series:
    """Generate wiper trip risk labels using real report data + pseudo-labels.

    Priority:
    1. Real labels from daily drilling reports (if available)
    2. Pseudo-labels from domain heuristics (fallback / supplement)
    """
    # --- Try real labels from reports ---
    try:
        from report_parser import build_label_series
        real_labels = build_label_series(df)
        n_real = int(real_labels.sum())
    except Exception:
        real_labels = pd.Series(0, index=df.index, dtype=int)
        n_real = 0

    # --- Pseudo-labels from domain rules ---
    mse = feat["MSE"]
    mse_p75 = mse.quantile(0.75)
    mse_p90 = mse.quantile(0.90)

    high_mse = (mse > mse_p75).astype(float)
    very_high_mse = (mse > mse_p90).astype(float)
    trq_increasing = (feat["TRQ_pct_10v30"] > 5).astype(float)
    spp_increasing = (feat["SPP_pct_10v30"] > 3).astype(float)
    rop_decreasing = (feat["ROP_pct_10v30"] < -5).astype(float)
    trq_volatile = (feat["TRQ_std_10"] > feat["TRQ_std_10"].quantile(0.80)).astype(float)
    dh_trq_diff = (feat["DH_TRQ_diff"] > feat["DH_TRQ_diff"].quantile(0.85)).astype(float)

    # Add extended signals if available
    hookload_signal = 0.0
    if "HOOKLOAD_drag" in feat.columns:
        hookload_signal = (
            feat["HOOKLOAD_drag"].abs() > feat["HOOKLOAD_drag"].abs().quantile(0.85)
        ).astype(float)

    pit_gl_signal = 0.0
    if "PIT_GL_abs_max_10" in feat.columns:
        pit_gl_signal = (
            feat["PIT_GL_abs_max_10"] > feat["PIT_GL_abs_max_10"].quantile(0.85)
        ).astype(float)

    pseudo_score = (
        0.20 * high_mse
        + 0.12 * very_high_mse
        + 0.12 * trq_increasing
        + 0.12 * spp_increasing
        + 0.10 * rop_decreasing
        + 0.08 * trq_volatile
        + 0.08 * dh_trq_diff
        + 0.09 * hookload_signal
        + 0.09 * pit_gl_signal
    )
    pseudo_labels = (pseudo_score > 0.30).astype(int)

    # --- Blend: real labels dominate ---
    if n_real > 50:
        # Real labels available — use them as primary, pseudo as supplement
        labels = real_labels.copy()
        # Add pseudo-labeled high-confidence points not covered by reports
        supplement_mask = (pseudo_labels == 1) & (real_labels == 0) & (pseudo_score > 0.5)
        labels.loc[supplement_mask] = 1
    else:
        # No real labels — fall back to pseudo-labels
        labels = pseudo_labels

    return labels


# ---------------------------------------------------------------------------
# Wiper Trip Predictor Class
# ---------------------------------------------------------------------------

class WiperTripPredictor:
    """Ensemble model for wiper trip risk prediction."""

    def __init__(self):
        self.gbt_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        )
        self.if_model = IsolationForest(
            n_estimators=100,
            contamination=0.15,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.training_metrics = {}

    def train(self, df: pd.DataFrame) -> dict:
        """Train both models on the provided drilling data.

        Args:
            df: DataFrame with renamed columns (WOB, ROP, TRQ, etc.)

        Returns:
            Dictionary with training metrics.
        """
        # Engineer features
        feat = engineer_features(df)
        self.feature_names = list(feat.columns)

        # Generate labels (real + pseudo)
        labels = generate_labels(df, feat)

        # Scale features
        X = self.scaler.fit_transform(feat)

        # Train/test split for validation metrics
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Train Gradient Boosted Trees
        self.gbt_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.gbt_model.predict(X_test)
        y_proba = self.gbt_model.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.5

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Train Isolation Forest on all data
        self.if_model.fit(X)

        self.is_trained = True

        # Label source info
        try:
            from report_parser import get_event_summary
            evt_summary = get_event_summary()
            n_report_events = evt_summary.get("n_events", 0)
            label_source = "Report-Mined" if n_report_events > 50 else "Pseudo-Labels"
        except Exception:
            n_report_events = 0
            label_source = "Pseudo-Labels"

        self.training_metrics = {
            "n_samples": len(df),
            "n_features": len(self.feature_names),
            "positive_rate": float(labels.mean()),
            "auc_roc": round(auc, 4),
            "precision": round(report.get("1", {}).get("precision", 0), 3),
            "recall": round(report.get("1", {}).get("recall", 0), 3),
            "f1_score": round(report.get("1", {}).get("f1-score", 0), 3),
            "accuracy": round(report.get("accuracy", 0), 3),
            "label_source": label_source,
            "n_report_events": n_report_events,
            "model_type": "GBT + Isolation Forest",
        }

        return self.training_metrics

    def predict(self, df: pd.DataFrame, idx: int) -> dict:
        """Predict wiper trip risk for the given index.

        Returns dict with:
            - risk_score: float 0-1
            - rf_probability: float 0-1  (GBT probability, key kept for compat)
            - if_anomaly_score: float 0-1
            - feature_importances: dict of top features
            - details: dict with component details
        """
        if not self.is_trained:
            return {
                "risk_score": 0.0,
                "rf_probability": 0.0,
                "if_anomaly_score": 0.0,
                "feature_importances": {},
                "details": {},
            }

        # Engineer features for the window up to idx
        start = max(0, idx - 50)
        window_df = df.iloc[start : idx + 1].copy()
        feat = engineer_features(window_df)

        # Get last row features
        last_feat = feat.iloc[[-1]]
        X = self.scaler.transform(last_feat)

        # GBT prediction
        gbt_proba = self.gbt_model.predict_proba(X)[0][1]

        # Isolation Forest anomaly score
        if_raw = self.if_model.decision_function(X)[0]
        if_score = max(0.0, min(1.0, 0.5 - if_raw * 2))

        # Ensemble: 0.65 GBT + 0.35 IF
        risk_score = 0.65 * gbt_proba + 0.35 * if_score
        risk_score = round(max(0.0, min(1.0, risk_score)), 3)

        # Feature importances (top 8)
        importances = self.gbt_model.feature_importances_
        feat_imp = dict(zip(self.feature_names, importances))
        top_features = dict(
            sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:8]
        )

        # Build details
        details = {
            "rf_probability": round(gbt_proba, 3),
            "if_anomaly_score": round(if_score, 3),
            "mse": round(float(last_feat["MSE"].iloc[0]), 1),
            "trq_change_pct": round(float(last_feat.get("TRQ_pct_10v30", pd.Series([0])).iloc[0]), 1),
            "spp_change_pct": round(float(last_feat.get("SPP_pct_10v30", pd.Series([0])).iloc[0]), 1),
            "rop_change_pct": round(float(last_feat.get("ROP_pct_10v30", pd.Series([0])).iloc[0]), 1),
            "mse_norm": round(gbt_proba, 2),
            "trq_norm": round(float(last_feat.get("TRQ_pct_10v30", pd.Series([0])).iloc[0]) / 25, 2),
            "spp_norm": round(float(last_feat.get("SPP_pct_10v30", pd.Series([0])).iloc[0]) / 20, 2),
            "rop_drop_norm": round(-float(last_feat.get("ROP_pct_10v30", pd.Series([0])).iloc[0]) / 30, 2),
            "flow_imbalance": round(abs(float(last_feat.get("Flow_pressure_ratio", pd.Series([0])).iloc[0])), 1),
            "flow_norm": round(if_score, 2),
        }

        return {
            "risk_score": risk_score,
            "rf_probability": round(gbt_proba, 3),
            "if_anomaly_score": round(if_score, 3),
            "feature_importances": top_features,
            "details": details,
        }

    def get_feature_importance_df(self) -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame."""
        if not self.is_trained:
            return pd.DataFrame()

        imp = self.gbt_model.feature_importances_
        df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": imp,
        }).sort_values("Importance", ascending=False)
        return df
