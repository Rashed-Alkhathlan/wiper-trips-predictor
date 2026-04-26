import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from model.engine import load_data
from model.model import WiperTripPredictor, engineer_features


@dataclass
class EventStats:
    historical_events: int
    recommended_events: int
    reduction_pct: float
    captured_events: int
    capture_rate_pct: float
    false_recommendations: int
    precision_pct: float


def resolve_data_file(base_dir: str) -> str:
    full_data = os.path.join(base_dir, "16A(78)-32_time_data_10s_intervals.csv")
    simplified_data = os.path.join(base_dir, "16A(78)-32_time_data_10s_intervals_simplified.csv")
    if os.path.exists(full_data):
        return full_data
    if os.path.exists(simplified_data):
        return simplified_data
    raise FileNotFoundError("Dataset not found.")


def get_label_series(df: pd.DataFrame, predictor: WiperTripPredictor) -> tuple[pd.Series, str]:
    candidates = ["label", "LABEL", "target", "TARGET", "y", "trip_label"]
    for col in candidates:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0, upper=1)
            return y.astype(int), f"dataset column '{col}'"
    return predictor.training_labels.astype(int), "training label pipeline"


def compute_smoothed_risk(df: pd.DataFrame, predictor: WiperTripPredictor, alpha: float = 0.35) -> pd.Series:
    feat = engineer_features(df)
    x_all = predictor.scaler.transform(feat)

    gbt_prob = predictor.gbt_model.predict_proba(x_all)[:, 1]
    if_raw = predictor.if_model.decision_function(x_all)
    if_score = np.clip(0.5 - if_raw * 2.0, 0.0, 1.0)
    raw = np.clip(0.65 * gbt_prob + 0.35 * if_score, 0.0, 1.0)

    smoothed = np.zeros_like(raw)
    smoothed[0] = raw[0]
    for i in range(1, len(raw)):
        smoothed[i] = alpha * raw[i] + (1.0 - alpha) * smoothed[i - 1]

    return pd.Series(smoothed, index=df.index)


def detect_historical_trip_events(labels: pd.Series, indices: np.ndarray) -> np.ndarray:
    events = []
    prev_idx = None
    prev_label = 0
    for idx in indices:
        cur_label = int(labels.iloc[idx])
        contiguous = prev_idx is not None and idx == prev_idx + 1
        prev_for_transition = prev_label if contiguous else 0
        if cur_label == 1 and prev_for_transition == 0:
            events.append(idx)
        prev_idx = idx
        prev_label = cur_label
    return np.array(events, dtype=int)


def detect_recommended_trip_events(
    risk: pd.Series,
    indices: np.ndarray,
    threshold: float,
    cooldown_steps: int,
) -> np.ndarray:
    events = []
    last_event_idx = -10**9
    prev_risk = None

    for idx in indices:
        cur_risk = float(risk.iloc[idx])
        crossed = prev_risk is not None and prev_risk < threshold <= cur_risk
        at_start_high = prev_risk is None and cur_risk >= threshold

        if (crossed or at_start_high) and (idx - last_event_idx >= cooldown_steps):
            events.append(idx)
            last_event_idx = idx

        prev_risk = cur_risk

    return np.array(events, dtype=int)


def evaluate(
    labels: pd.Series,
    risk: pd.Series,
    indices: np.ndarray,
    threshold: float,
    cooldown_steps: int,
    lead_steps: int,
) -> EventStats:
    hist_events = detect_historical_trip_events(labels, indices)
    rec_events = detect_recommended_trip_events(risk, indices, threshold, cooldown_steps)

    captured = 0
    matched_recs = set()
    for h in hist_events:
        candidates = [r for r in rec_events if (h - lead_steps) <= r <= h]
        if candidates:
            captured += 1
            matched_recs.add(candidates[-1])

    historical_events = int(len(hist_events))
    recommended_events = int(len(rec_events))
    reduction_pct = (100.0 * (historical_events - recommended_events) / historical_events) if historical_events else 0.0
    capture_rate_pct = (100.0 * captured / historical_events) if historical_events else 0.0

    false_recs = recommended_events - len(matched_recs)
    precision_pct = (100.0 * len(matched_recs) / recommended_events) if recommended_events else 0.0

    return EventStats(
        historical_events=historical_events,
        recommended_events=recommended_events,
        reduction_pct=reduction_pct,
        captured_events=captured,
        capture_rate_pct=capture_rate_pct,
        false_recommendations=int(false_recs),
        precision_pct=precision_pct,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest trip reduction on evaluation split.")
    parser.add_argument("--subsample", type=int, default=10, help="Data subsample step (default: 10).")
    parser.add_argument("--threshold", type=float, default=0.7, help="Trip trigger threshold on risk score.")
    parser.add_argument(
        "--cooldown-hours",
        type=float,
        default=4.0,
        help="Minimum hours between two recommended trips (default: 4).",
    )
    parser.add_argument(
        "--lead-hours",
        type=float,
        default=2.0,
        help="A recommendation counts as captured if within this many hours before event (default: 2).",
    )
    parser.add_argument(
        "--scope",
        choices=["test", "all"],
        default="test",
        help="Evaluation scope: test split only (default) or all rows.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    data_file = resolve_data_file(base_dir)
    df = load_data(data_file, subsample=args.subsample)

    predictor = WiperTripPredictor()
    metrics = predictor.train(df)
    labels, label_source = get_label_series(df, predictor)
    risk = compute_smoothed_risk(df, predictor)

    if args.scope == "test" and len(predictor.test_indices) > 0:
        eval_indices = np.array(sorted(set(int(i) for i in predictor.test_indices if 0 <= int(i) < len(df))), dtype=int)
        scope_name = "test split"
    else:
        eval_indices = np.arange(len(df), dtype=int)
        scope_name = "all rows"

    if "Time" in df.columns and len(df) > 1:
        dt_sec = pd.to_datetime(df["Time"]).diff().dt.total_seconds().median()
        dt_sec = float(dt_sec) if pd.notna(dt_sec) and dt_sec > 0 else 10.0
    else:
        dt_sec = 10.0

    cooldown_steps = max(1, int((args.cooldown_hours * 3600.0) / dt_sec))
    lead_steps = max(1, int((args.lead_hours * 3600.0) / dt_sec))

    stats = evaluate(
        labels=labels,
        risk=risk,
        indices=eval_indices,
        threshold=args.threshold,
        cooldown_steps=cooldown_steps,
        lead_steps=lead_steps,
    )

    print("\n=== Trip Reduction Backtest ===")
    print(f"Data file: {data_file}")
    print(f"Rows loaded: {len(df):,} (subsample={args.subsample})")
    print(f"Scope: {scope_name} ({len(eval_indices):,} rows)")
    print(f"Label source: {label_source}")
    print(f"Model label source metric: {metrics.get('label_source', 'unknown')}")
    print(f"Split strategy: {metrics.get('split_strategy', 'unknown')}")
    print(f"Threshold: {args.threshold:.3f}")
    print(f"Cooldown: {args.cooldown_hours:.2f}h ({cooldown_steps} steps)")
    print(f"Lead window: {args.lead_hours:.2f}h ({lead_steps} steps)")

    print("\n--- Outcomes ---")
    print(f"Historical apparent trip events: {stats.historical_events}")
    print(f"Model recommended trip events: {stats.recommended_events}")
    print(f"Trip reduction estimate: {stats.reduction_pct:.2f}%")
    print(f"Captured historical events within lead window: {stats.captured_events}")
    print(f"Capture rate: {stats.capture_rate_pct:.2f}%")
    print(f"False recommendations (not matched): {stats.false_recommendations}")
    print(f"Recommendation precision: {stats.precision_pct:.2f}%")

    print("\nInterpretation: this is offline backtest evidence, not causal proof.")


if __name__ == "__main__":
    main()
