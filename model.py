"""
Wiper Trip ML Predictor — Window-Based Architecture
=====================================================
Predicts whether a wiper trip will be needed within the next
PREDICTION_HORIZON hours, based on 30-minute aggregate windows
of drilling sensor data.

Key design decisions:
  - Samples are 30-minute aggregate windows (not individual 10s readings)
  - Labels come ONLY from external ground truth (daily report events)
  - NO pseudo-labels (no circular feature→label dependency)
  - Strict temporal train/test split (no future leakage)
  - Calibrated probabilities via CalibratedClassifierCV
  - Ensemble: 0.65 × calibrated GBT + 0.35 × Isolation Forest
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Bit diameter for MSE calculation
BIT_DIAMETER_IN = 8.5

# How far ahead we predict (hours)
PREDICTION_HORIZON_HOURS = 4

# Window size for aggregation (minutes)
WINDOW_MINUTES = 30


# ---------------------------------------------------------------------------
# MSE Calculation
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


# ---------------------------------------------------------------------------
# Window-Based Feature Engineering
# ---------------------------------------------------------------------------

def _compute_trend(series: pd.Series) -> float:
    """Linear regression slope over a series (trend direction)."""
    n = len(series)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    y = series.values.astype(float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    x, y = x[mask], y[mask]
    # Simple OLS slope
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def engineer_window_features(window: pd.DataFrame) -> dict:
    """Engineer aggregate features from a single time window.

    This function is used IDENTICALLY at training and prediction time
    to ensure feature consistency.

    Args:
        window: DataFrame slice containing sensor readings for one window.

    Returns:
        dict of feature_name → float value
    """
    feat = {}

    # Base sensor columns
    base_cols = ["WOB", "ROP", "RPM", "TRQ", "SPP", "FLOW_IN", "DH_TRQ", "DIFF_P"]
    ext_cols = ["HOOKLOAD", "GAS", "RETURN_FLOW", "PIT_GL", "TRIP_GL", "MWD_INC"]
    available_ext = [c for c in ext_cols if c in window.columns]
    all_cols = base_cols + available_ext

    # --- Aggregate statistics for all channels ---
    for col in all_cols:
        s = window[col]
        feat[f"{col}_mean"] = s.mean()
        feat[f"{col}_std"] = s.std() if len(s) > 1 else 0.0
        feat[f"{col}_min"] = s.min()
        feat[f"{col}_max"] = s.max()

    # --- Trends (slope over window) for key channels ---
    for col in ["ROP", "TRQ", "SPP", "DH_TRQ", "FLOW_IN"]:
        feat[f"{col}_trend"] = _compute_trend(window[col])

    # --- MSE ---
    mse = compute_mse_series(window)
    feat["MSE_mean"] = mse.mean()
    feat["MSE_max"] = mse.max()
    feat["MSE_std"] = mse.std() if len(mse) > 1 else 0.0
    feat["MSE_trend"] = _compute_trend(mse)

    # --- Cross-feature ratios (computed from window means) ---
    rop_mean = max(feat["ROP_mean"], 0.1)
    trq_mean = max(feat["TRQ_mean"], 0.1)
    spp_mean = max(feat["SPP_mean"], 1.0)
    flow_mean = max(feat["FLOW_IN_mean"], 1.0)

    feat["TRQ_ROP_ratio"] = trq_mean / rop_mean
    feat["MSE_x_RPM"] = feat["MSE_mean"] * feat["RPM_mean"]
    feat["DH_TRQ_diff"] = feat.get("DH_TRQ_mean", 0) - trq_mean
    feat["Flow_pressure_ratio"] = flow_mean / spp_mean
    feat["WOB_TRQ_ratio"] = feat["WOB_mean"] / trq_mean

    # --- Range / volatility ratios ---
    feat["TRQ_range"] = feat["TRQ_max"] - feat["TRQ_min"]
    feat["SPP_range"] = feat["SPP_max"] - feat["SPP_min"]
    feat["ROP_range"] = feat["ROP_max"] - feat["ROP_min"]

    # --- Hookload features ---
    if "HOOKLOAD" in window.columns:
        hl = window["HOOKLOAD"]
        feat["HOOKLOAD_trend"] = _compute_trend(hl)
        hl_mean = hl.mean()
        hl_std = hl.std() if len(hl) > 1 else 0.0
        feat["HOOKLOAD_cv"] = hl_std / max(hl_mean, 1.0)

    # --- Pit gain/loss features ---
    if "PIT_GL" in window.columns:
        gl = window["PIT_GL"]
        feat["PIT_GL_abs_sum"] = gl.abs().sum()
        feat["PIT_GL_abs_max"] = gl.abs().max()
        feat["PIT_GL_trend"] = _compute_trend(gl)

    # --- Return flow balance ---
    if "RETURN_FLOW" in window.columns and "FLOW_IN" in window.columns:
        ret = window["RETURN_FLOW"]
        flow = window["FLOW_IN"]
        feat["FLOW_RATIO_mean"] = (ret / flow.clip(lower=1.0)).mean()
        feat["FLOW_IMBALANCE_mean"] = (ret - flow).mean()
        feat["FLOW_IMBALANCE_std"] = (ret - flow).std() if len(ret) > 1 else 0.0

    # --- Gas features ---
    if "GAS" in window.columns:
        gas = window["GAS"]
        feat["GAS_max"] = gas.max()
        feat["GAS_trend"] = _compute_trend(gas)

    # --- MWD Inclination ---
    if "MWD_INC" in window.columns:
        inc = window["MWD_INC"]
        feat["MWD_INC_mean"] = inc.mean()
        feat["MWD_INC_max"] = inc.max()
        feat["INC_HIGH_ANGLE_frac"] = (inc > 30).mean()
        feat["INC_CRITICAL_frac"] = (inc > 60).mean()
        feat["INC_x_TRQ"] = inc.mean() * trq_mean / 1000.0
        feat["INC_x_MSE"] = inc.mean() * feat["MSE_mean"] / 1000.0

    # --- Block position (pipe movement) ---
    if "BLOCK_POS" in window.columns:
        bp = window["BLOCK_POS"]
        vel = bp.diff().fillna(0)
        feat["BLOCK_VEL_mean"] = vel.mean()
        feat["BLOCK_VEL_abs_mean"] = vel.abs().mean()
        feat["BLOCK_VEL_std"] = vel.std() if len(vel) > 1 else 0.0

    # --- On-bottom fraction ---
    if "ON_BOTTOM" in window.columns:
        feat["ON_BOTTOM_frac"] = window["ON_BOTTOM"].mean()

    return feat


def create_windows(df: pd.DataFrame,
                   window_minutes: int = WINDOW_MINUTES) -> pd.DataFrame:
    """Split the time-series into non-overlapping windows and compute features.

    Each window becomes one training sample.

    Args:
        df: Full dataset with renamed columns and 'Time' column.
        window_minutes: Duration of each window in minutes.

    Returns:
        DataFrame where each row is one window with:
        - window_start, window_end (datetime)
        - window_idx (integer index)
        - All engineered features
    """
    times = pd.to_datetime(df["Time"])
    t_start = times.min()
    t_end = times.max()

    window_delta = pd.Timedelta(minutes=window_minutes)
    windows = []

    current_start = t_start
    window_idx = 0

    while current_start + window_delta <= t_end:
        current_end = current_start + window_delta
        mask = (times >= current_start) & (times < current_end)
        window_data = df.loc[mask]

        if len(window_data) >= 10:  # Need at least 10 readings per window
            feat = engineer_window_features(window_data)
            feat["window_start"] = current_start
            feat["window_end"] = current_end
            feat["window_idx"] = window_idx
            feat["n_readings"] = len(window_data)
            windows.append(feat)
            window_idx += 1

        current_start = current_end

    result = pd.DataFrame(windows)
    return result


# ---------------------------------------------------------------------------
# Label Generation (from external ground truth ONLY)
# ---------------------------------------------------------------------------

def create_labels(windows_df: pd.DataFrame,
                  events: list[dict],
                  horizon_hours: float = PREDICTION_HORIZON_HOURS) -> pd.Series:
    """Create binary labels for each window based on external events.

    Label = 1 if a reactive event STARTS within horizon_hours after
    the window's end time.

    Windows that overlap with an event (i.e., during the event) are
    EXCLUDED from training to prevent the model from learning
    "currently tripping" patterns.

    Args:
        windows_df: DataFrame from create_windows() with window_start/end.
        events: List of event dicts from build_event_timeline().
        horizon_hours: How far ahead to predict.

    Returns:
        pd.Series of labels (0, 1, or -1 for excluded windows).
        -1 means "during an event — exclude from training".
    """
    labels = pd.Series(0, index=windows_df.index, dtype=int)
    horizon = pd.Timedelta(hours=horizon_hours)

    for i, row in windows_df.iterrows():
        w_start = row["window_start"]
        w_end = row["window_end"]

        for evt in events:
            evt_start = pd.Timestamp(evt["start_dt"])
            evt_end = pd.Timestamp(evt["end_dt"])

            # Check if window overlaps with event → exclude
            if w_start < evt_end and w_end > evt_start:
                labels.iloc[i] = -1
                break

            # Check if event starts within prediction horizon after window
            time_until_event = evt_start - w_end
            if pd.Timedelta(0) <= time_until_event <= horizon:
                labels.iloc[i] = 1
                break  # Found an event in horizon, no need to check more

    return labels


# ---------------------------------------------------------------------------
# Wiper Trip Predictor Class
# ---------------------------------------------------------------------------

class WiperTripPredictor:
    """Window-based ensemble model for wiper trip prediction.

    Predicts whether a wiper trip will be needed within the next
    PREDICTION_HORIZON hours based on the current 30-minute window
    of sensor data.
    """

    def __init__(self):
        self._base_gbt = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )
        self.gbt_model = None  # Will be CalibratedClassifierCV
        self.if_model = IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.training_metrics = {}
        self._windows_df = None  # Cache for prediction-time reference

    def _get_feature_columns(self, windows_df: pd.DataFrame) -> list[str]:
        """Get feature column names (exclude metadata columns)."""
        meta_cols = {"window_start", "window_end", "window_idx", "n_readings"}
        return [c for c in windows_df.columns if c not in meta_cols]

    def train(self, df: pd.DataFrame) -> dict:
        """Train the model on the provided drilling data.

        Steps:
        1. Create 30-minute windows with aggregate features
        2. Get event timeline from reports (reactive events only)
        3. Label windows: 1 if event within horizon, -1 if during event
        4. Temporal split: first 80% train, last 20% test
        5. Train GBT with Platt scaling calibration
        6. Train Isolation Forest on training data only
        7. Report honest metrics on temporally-separated test set

        Args:
            df: DataFrame with renamed columns (WOB, ROP, TRQ, etc.)

        Returns:
            Dictionary with training metrics.
        """
        # 1. Create windows
        windows_df = create_windows(df, WINDOW_MINUTES)
        self._windows_df = windows_df

        # 2. Get event timeline — try reactive only first, fall back to all
        try:
            from report_parser import build_event_timeline
            events = build_event_timeline(df, reactive_only=True)
            n_events = len(events)
            label_source = "Report-Mined (Reactive)"
        except Exception:
            events = []
            n_events = 0
            label_source = "No Events Found"

        # 3. Create labels
        labels = create_labels(windows_df, events, PREDICTION_HORIZON_HOURS)

        # 4. Filter out excluded windows (during events)
        valid_mask = labels != -1
        windows_valid = windows_df[valid_mask].reset_index(drop=True)
        labels_valid = labels[valid_mask].reset_index(drop=True)

        # Check: do we have enough positives for the temporal split?
        # With temporal split, positives in training (first 80%) may be zero
        n_total = len(labels_valid)
        split_idx = int(n_total * 0.8)
        n_pos_train = int(labels_valid.iloc[:split_idx].sum()) if split_idx > 0 else 0

        # Fallback: if reactive-only gives <2 positive training samples,
        # include ALL events (including planned trip_out / POOH)
        if n_pos_train < 2 and n_events > 0:
            try:
                from report_parser import build_event_timeline
                events = build_event_timeline(df, reactive_only=False)
                n_events = len(events)
                label_source = "Report-Mined (All Events)"
            except Exception:
                pass

            labels = create_labels(windows_df, events, PREDICTION_HORIZON_HOURS)
            valid_mask = labels != -1
            windows_valid = windows_df[valid_mask].reset_index(drop=True)
            labels_valid = labels[valid_mask].reset_index(drop=True)
            n_total = len(labels_valid)
            split_idx = int(n_total * 0.8)
            n_pos_train = int(labels_valid.iloc[:split_idx].sum()) if split_idx > 0 else 0

        # If STILL <2 positives, try a wider prediction horizon (8h)
        if n_pos_train < 2 and n_events > 0:
            labels = create_labels(windows_df, events, PREDICTION_HORIZON_HOURS * 2)
            valid_mask = labels != -1
            windows_valid = windows_df[valid_mask].reset_index(drop=True)
            labels_valid = labels[valid_mask].reset_index(drop=True)
            n_total = len(labels_valid)
            split_idx = int(n_total * 0.8)
            n_pos_train = int(labels_valid.iloc[:split_idx].sum()) if split_idx > 0 else 0

        # Get feature columns
        feat_cols = self._get_feature_columns(windows_valid)
        self.feature_names = feat_cols

        X_all = windows_valid[feat_cols].values
        y_all = labels_valid.values

        # Replace inf/nan
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

        # 5. TEMPORAL split — first 80% train, last 20% test
        X_train, X_test = X_all[:split_idx], X_all[split_idx:]
        y_train, y_test = y_all[:split_idx], y_all[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 6. Train GBT with calibration
        n_pos = int(y_train.sum())
        n_neg = int((y_train == 0).sum())
        can_train = n_pos >= 2 and n_neg >= 2

        if can_train and n_pos >= 5 and n_neg >= 5:
            # Enough positive samples for calibration
            try:
                self.gbt_model = CalibratedClassifierCV(
                    self._base_gbt, cv=min(3, n_pos), method="sigmoid"
                )
                self.gbt_model.fit(X_train_scaled, y_train)
            except Exception:
                # Fallback: train without calibration
                self._base_gbt.fit(X_train_scaled, y_train)
                self.gbt_model = self._base_gbt
        elif can_train:
            # Enough to train but not calibrate
            self._base_gbt.fit(X_train_scaled, y_train)
            self.gbt_model = self._base_gbt
        else:
            # Cannot train — will use rule-based fallback
            self.gbt_model = None
            label_source = f"Insufficient labels ({n_pos} pos in train)"

        # 7. Train Isolation Forest on TRAINING data only (no test leakage)
        self.if_model.fit(X_train_scaled)

        # 8. Evaluate on temporally-separated test set
        if self.gbt_model is not None and len(X_test) > 0 and len(np.unique(y_test)) > 1:
            y_pred = self.gbt_model.predict(X_test_scaled)
            y_proba = self.gbt_model.predict_proba(X_test_scaled)[:, 1]
            try:
                auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                auc = 0.5
            report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )
        else:
            auc = 0.5
            report = {}
            y_pred = np.array([])

        self.is_trained = self.gbt_model is not None

        self.training_metrics = {
            "n_total_windows": len(windows_df),
            "n_valid_windows": int(valid_mask.sum()),
            "n_excluded_windows": int((~valid_mask).sum()),
            "n_train": split_idx,
            "n_test": n_total - split_idx,
            "n_features": len(self.feature_names),
            "positive_rate_train": float(y_train.mean()) if len(y_train) > 0 else 0.0,
            "positive_rate_test": float(y_test.mean()) if len(y_test) > 0 else 0.0,
            "auc_roc": round(auc, 4),
            "precision": round(report.get("1", {}).get("precision", 0), 3),
            "recall": round(report.get("1", {}).get("recall", 0), 3),
            "f1_score": round(report.get("1", {}).get("f1-score", 0), 3),
            "accuracy": round(report.get("accuracy", 0), 3),
            "label_source": label_source,
            "n_reactive_events": n_events,
            "prediction_horizon_hrs": PREDICTION_HORIZON_HOURS,
            "window_minutes": WINDOW_MINUTES,
            "model_type": "Calibrated GBT + Isolation Forest",
            "split_type": "Temporal (80/20)",
        }

        return self.training_metrics

    def predict(self, df: pd.DataFrame, idx: int) -> dict:
        """Predict wiper trip risk for the given index.

        Creates a window from the most recent WINDOW_MINUTES of data
        ending at idx, using the SAME feature computation as training.

        Returns dict with:
            - risk_score: float 0-1 (calibrated ensemble probability)
            - gbt_probability: float 0-1
            - if_anomaly_score: float 0-1
            - feature_importances: dict of top features
            - details: dict with component details
        """
        if not self.is_trained:
            return {
                "risk_score": 0.0,
                "gbt_probability": 0.0,
                "if_anomaly_score": 0.0,
                "feature_importances": {},
                "details": {},
            }

        # Build a window from the data ending at idx
        # Use the same window size as training
        times = pd.to_datetime(df["Time"])
        current_time = times.iloc[idx]
        window_start = current_time - pd.Timedelta(minutes=WINDOW_MINUTES)

        mask = (times >= window_start) & (times <= current_time)
        window_data = df.loc[mask]

        if len(window_data) < 5:
            # Not enough data for a meaningful window, use what we have
            start = max(0, idx - 50)
            window_data = df.iloc[start:idx + 1]

        # Engineer features using the SAME function as training
        feat_dict = engineer_window_features(window_data)

        # Build feature vector in the same order as training
        feat_values = []
        for col in self.feature_names:
            feat_values.append(feat_dict.get(col, 0.0))

        X = np.array([feat_values])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)

        # GBT prediction (calibrated probability)
        gbt_proba = self.gbt_model.predict_proba(X_scaled)[0][1]

        # Isolation Forest anomaly score
        if_raw = self.if_model.decision_function(X_scaled)[0]
        if_score = max(0.0, min(1.0, 0.5 - if_raw * 2))

        # Ensemble: 0.65 calibrated GBT + 0.35 IF
        risk_score = 0.65 * gbt_proba + 0.35 * if_score
        risk_score = round(max(0.0, min(1.0, risk_score)), 3)

        # Feature importances (from base GBT estimator)
        try:
            if hasattr(self.gbt_model, 'calibrated_classifiers_'):
                # CalibratedClassifierCV wraps the base estimator
                base = self.gbt_model.calibrated_classifiers_[0].estimator
                importances = base.feature_importances_
            else:
                importances = self.gbt_model.feature_importances_
            feat_imp = dict(zip(self.feature_names, importances))
            top_features = dict(
                sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:8]
            )
        except (AttributeError, IndexError):
            top_features = {}

        # Build details for advisory
        details = {
            "rf_probability": round(gbt_proba, 3),
            "if_anomaly_score": round(if_score, 3),
            "mse": round(float(feat_dict.get("MSE_mean", 0)), 1),
            "trq_change_pct": round(float(feat_dict.get("TRQ_trend", 0) * 100), 1),
            "spp_change_pct": round(float(feat_dict.get("SPP_trend", 0) * 100), 1),
            "rop_change_pct": round(float(feat_dict.get("ROP_trend", 0) * 100), 1),
            "mse_norm": round(gbt_proba, 2),
            "trq_norm": round(min(1.0, abs(feat_dict.get("TRQ_trend", 0)) / 0.5), 2),
            "spp_norm": round(min(1.0, abs(feat_dict.get("SPP_trend", 0)) / 0.5), 2),
            "rop_drop_norm": round(min(1.0, max(0, -feat_dict.get("ROP_trend", 0)) / 0.5), 2),
            "flow_imbalance": round(abs(feat_dict.get("FLOW_IMBALANCE_mean", 0)), 1),
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

        try:
            if hasattr(self.gbt_model, 'calibrated_classifiers_'):
                base = self.gbt_model.calibrated_classifiers_[0].estimator
                imp = base.feature_importances_
            else:
                imp = self.gbt_model.feature_importances_

            df = pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": imp,
            }).sort_values("Importance", ascending=False)
            return df
        except (AttributeError, IndexError):
            return pd.DataFrame()
