import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

from engine import load_data
from model import WiperTripPredictor, engineer_features


def resolve_data_file(base_dir: str) -> str:
    """Pick the full dataset if available, else fallback to simplified."""
    full_data = os.path.join(base_dir, "16A(78)-32_time_data_10s_intervals.csv")
    simplified_data = os.path.join(base_dir, "16A(78)-32_time_data_10s_intervals_simplified.csv")

    if os.path.exists(full_data):
        return full_data
    if os.path.exists(simplified_data):
        return simplified_data
    raise FileNotFoundError("Could not find full or simplified dataset CSV.")


def compute_scores(df: pd.DataFrame, predictor: WiperTripPredictor) -> pd.DataFrame:
    """Compute row-wise model scores for all samples in df."""
    feat = engineer_features(df)
    x_all = predictor.scaler.transform(feat)

    gbt_prob = predictor.gbt_model.predict_proba(x_all)[:, 1]
    if_raw = predictor.if_model.decision_function(x_all)
    if_score = np.clip(0.5 - if_raw * 2.0, 0.0, 1.0)
    risk_score = np.clip(0.65 * gbt_prob + 0.35 * if_score, 0.0, 1.0)

    out = pd.DataFrame(
        {
            "gbt_prob": gbt_prob,
            "if_score": if_score,
            "risk_score": risk_score,
        },
        index=df.index,
    )
    return out


def pick_label_series(df: pd.DataFrame, predictor: WiperTripPredictor) -> tuple[pd.Series, str]:
    """Use a native label column when present, else fallback to training labels."""
    candidate_cols = ["label", "LABEL", "target", "TARGET", "y", "trip_label"]
    for col in candidate_cols:
        if col in df.columns:
            labels = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0, upper=1)
            return labels.astype(int), f"dataset column '{col}'"

    return predictor.training_labels.astype(int), "training label pipeline"


def summarize(
    df: pd.DataFrame,
    labels: pd.Series,
    scores: pd.DataFrame,
    threshold: float,
    output_csv: Optional[str],
) -> None:
    """Print mismatch stats and optionally write mismatch samples to CSV."""
    no_trip_pred = scores["risk_score"] < threshold
    apparent_trip = labels == 1

    mismatch_mask = no_trip_pred & apparent_trip
    n_mismatch = int(mismatch_mask.sum())
    n_total = int(len(df))
    n_trips = int(apparent_trip.sum())

    pct_of_all = (100.0 * n_mismatch / n_total) if n_total else 0.0
    pct_of_trips = (100.0 * n_mismatch / n_trips) if n_trips else 0.0

    print("\n=== No-Trip Prediction vs Apparent Trip Label ===")
    print(f"Total samples: {n_total:,}")
    print(f"Apparent trip-labeled samples (label=1): {n_trips:,}")
    print(f"No-trip threshold on risk score: {threshold:.3f}")
    print(f"Mismatch count (model says no trip, label says trip): {n_mismatch:,}")
    print(f"Mismatch rate over all samples: {pct_of_all:.2f}%")
    print(f"Mismatch rate over trip-labeled samples: {pct_of_trips:.2f}%")

    mismatch_df = df.loc[mismatch_mask, ["Time"]].copy() if "Time" in df.columns else df.loc[mismatch_mask].copy()
    mismatch_df["true_label"] = labels.loc[mismatch_mask].values
    mismatch_df["risk_score"] = scores.loc[mismatch_mask, "risk_score"].values
    mismatch_df["gbt_prob"] = scores.loc[mismatch_mask, "gbt_prob"].values
    mismatch_df["if_score"] = scores.loc[mismatch_mask, "if_score"].values

    if len(mismatch_df) > 0:
        print("\nFirst 10 mismatch rows:")
        print(mismatch_df.head(10).to_string(index=False))
    else:
        print("\nNo mismatches found for this threshold.")

    if output_csv:
        mismatch_df.to_csv(output_csv, index=False)
        print(f"\nSaved mismatch samples to: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count samples where model predicts no trip but label indicates a trip."
        )
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=10,
        help="Row subsampling factor for loading data (default: 10, same as app full-data path).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Risk threshold below which sample is treated as no-trip prediction (default: 0.7).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional path to save mismatch samples as CSV.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    data_file = resolve_data_file(base_dir)

    print(f"Loading data file: {data_file}")
    print(f"Using subsample factor: {args.subsample}")

    df = load_data(data_file, subsample=args.subsample)

    predictor = WiperTripPredictor()
    metrics = predictor.train(df)

    print("\n=== Training Context ===")
    print(f"Label source: {metrics.get('label_source', 'unknown')}")
    print(f"Validation split: {metrics.get('split_strategy', 'unknown')}")
    print(f"Samples: {metrics.get('n_samples', len(df)):,}")

    labels, label_source = pick_label_series(df, predictor)
    print(f"Label series used for mismatch count: {label_source}")

    scores = compute_scores(df, predictor)

    out_path = args.output_csv.strip() or None
    summarize(df, labels, scores, args.threshold, out_path)


if __name__ == "__main__":
    main()
