"""
Additional data-access tools for the Agentic Drilling Advisor.
Each tool is a @langchain_core.tools.tool function that operates on a
shared pandas DataFrame (`df`) and configuration dicts (`KEY_PARAMS`, `UNITS`).

Usage (in the notebook):
    from agent_tools import create_data_tools
    new_tools = create_data_tools(df, KEY_PARAMS, UNITS)
"""

import pandas as pd
import numpy as np
from langchain_core.tools import tool


def create_data_tools(df: pd.DataFrame, KEY_PARAMS: list, UNITS: dict):
    """Factory that returns 5 LangChain tools bound to the given DataFrame."""

    # ------------------------------------------------------------------ #
    # Tool 1: Query by time range
    # ------------------------------------------------------------------ #
    @tool
    def query_by_time_range(start_time: str, end_time: str, parameters: str = "") -> str:
        """Query sensor data between two timestamps. Returns summary stats
        and trends for the specified time window.

        Args:
            start_time: Start timestamp, e.g. '2020-11-10 12:00'
            end_time: End timestamp, e.g. '2020-11-10 14:00'
            parameters: Comma-separated sensor names, e.g. 'TRQ,HOOKLOAD,SPP'.
                        Leave empty for all key parameters.
        """
        try:
            t1 = pd.to_datetime(start_time)
            t2 = pd.to_datetime(end_time)
        except Exception:
            return (f"Error: Could not parse timestamps. "
                    f"Use format like '2020-11-10 12:00'. "
                    f"Got start='{start_time}', end='{end_time}'")

        mask = (df['Time'] >= t1) & (df['Time'] <= t2)
        window = df.loc[mask]

        if len(window) == 0:
            return (f"No data found between {t1} and {t2}.\n"
                    f"Available range: {df['Time'].min()} to {df['Time'].max()}")

        params = _parse_params(parameters, KEY_PARAMS, df)
        if isinstance(params, str):
            return params  # error message

        lines = [
            f"TIME RANGE QUERY: {t1} to {t2}",
            f"Data points: {len(window):,}",
            f"Duration: {t2 - t1}",
            f"Depth range: {window['DEPTH'].min():.0f} - {window['DEPTH'].max():.0f} m",
            "",
            "Summary Statistics:",
        ]
        for p in params:
            col = window[p]
            lines.append(
                f"  {p}: mean={col.mean():.2f}, std={col.std():.2f}, "
                f"min={col.min():.2f}, max={col.max():.2f} {UNITS.get(p, '')}"
            )

        lines.append("")
        lines.append("First reading:")
        first = window.iloc[0]
        for p in params:
            lines.append(f"  {p}: {first[p]:.2f}  at {first['Time']}")

        lines.append("")
        lines.append("Last reading:")
        last = window.iloc[-1]
        for p in params:
            lines.append(f"  {p}: {last[p]:.2f}  at {last['Time']}")

        lines.append("")
        lines.append("Trends:")
        for p in params:
            s, e = window[p].iloc[0], window[p].iloc[-1]
            pct = ((e - s) / abs(s) * 100) if s else 0
            arrow = "↑" if pct > 2 else "↓" if pct < -2 else "→"
            lines.append(f"  {p}: {arrow} {pct:+.1f}%")

        return '\n'.join(lines)

    # ------------------------------------------------------------------ #
    # Tool 2: Compute statistics
    # ------------------------------------------------------------------ #
    @tool
    def compute_statistics(parameters: str,
                           aggregation: str = "mean,std,min,max",
                           last_n_points: int = 0) -> str:
        """Compute statistical aggregations on sensor parameters.

        Args:
            parameters: Comma-separated parameter names, e.g. 'TRQ,HOOKLOAD,SPP'
            aggregation: Comma-separated stats: mean, median, std, min, max,
                         percentile_25, percentile_75. Default 'mean,std,min,max'
            last_n_points: If > 0, restrict to the last N data points. 0 = all.
        """
        params = _parse_params(parameters, KEY_PARAMS, df)
        if isinstance(params, str):
            return params

        aggs = [a.strip().lower() for a in aggregation.split(',')]
        data = df.iloc[-last_n_points:] if last_n_points > 0 else df

        header = f"{'Parameter':<12} " + " ".join(f"{a:>14}" for a in aggs)
        sep = "-" * len(header)
        lines = [
            f"STATISTICAL ANALYSIS ({len(data):,} data points)",
            "",
            header,
            sep,
        ]

        agg_map = {
            'mean':   lambda s: s.mean(),
            'median': lambda s: s.median(),
            'std':    lambda s: s.std(),
            'min':    lambda s: s.min(),
            'max':    lambda s: s.max(),
            'percentile_25': lambda s: s.quantile(0.25),
            'percentile_75': lambda s: s.quantile(0.75),
        }

        for p in params:
            vals = []
            for a in aggs:
                fn = agg_map.get(a)
                vals.append(f"{fn(data[p]):>14.2f}" if fn else f"{'N/A':>14}")
            lines.append(f"{p:<12} " + " ".join(vals))

        lines.append("")
        lines.append(
            f"Units: {', '.join(f'{p}={UNITS.get(p, 'N/A')}' for p in params)}"
        )
        return '\n'.join(lines)

    # ------------------------------------------------------------------ #
    # Tool 3: Detect anomalies
    # ------------------------------------------------------------------ #
    @tool
    def detect_anomalies(parameters: str,
                         sigma_threshold: float = 2.5,
                         start_time: str = "",
                         end_time: str = "") -> str:
        """Detect anomalous sensor readings using sigma-based thresholds.

        Args:
            parameters: Comma-separated parameter names, e.g. 'TRQ,DIFF_P'
            sigma_threshold: Std-dev multiplier for anomaly detection. Default 2.5
            start_time: Optional start timestamp to scope the search.
            end_time: Optional end timestamp to scope the search.
        """
        params = _parse_params(parameters, KEY_PARAMS, df)
        if isinstance(params, str):
            return params

        data = df.copy()
        if start_time.strip():
            try:
                data = data[data['Time'] >= pd.to_datetime(start_time)]
            except Exception:
                pass
        if end_time.strip():
            try:
                data = data[data['Time'] <= pd.to_datetime(end_time)]
            except Exception:
                pass

        lines = [
            f"ANOMALY DETECTION (σ threshold: {sigma_threshold})",
            f"Data points analyzed: {len(data):,}",
            "",
        ]

        total = 0
        for p in params:
            mean, std = df[p].mean(), df[p].std()
            upper = mean + sigma_threshold * std
            lower = mean - sigma_threshold * std
            anom = data[(data[p] > upper) | (data[p] < lower)]
            count = len(anom)
            total += count

            lines.append(f"{p}:")
            lines.append(f"  Baseline: mean={mean:.2f}, std={std:.2f} {UNITS.get(p, '')}")
            lines.append(f"  Thresholds: [{lower:.2f}, {upper:.2f}]")
            lines.append(f"  Anomalies: {count} ({count / max(len(data), 1) * 100:.1f}%)")

            if count > 0:
                tmp = anom.copy()
                tmp['_dev'] = ((tmp[p] - mean) / std).abs()
                worst = tmp.nlargest(min(3, count), '_dev')
                lines.append("  Worst:")
                for _, row in worst.iterrows():
                    dev = (row[p] - mean) / std
                    lines.append(f"    {row['Time']}: {row[p]:.2f} ({dev:+.1f}σ)")
            else:
                lines.append("  ✓ No anomalies detected")
            lines.append("")

        sev = "HIGH" if total > 20 else "MEDIUM" if total > 5 else "LOW"
        lines.append(f"Overall: {total} anomalies across {len(params)} params (Severity: {sev})")
        return '\n'.join(lines)

    # ------------------------------------------------------------------ #
    # Tool 4: Correlate parameters
    # ------------------------------------------------------------------ #
    @tool
    def correlate_parameters(parameters: str, last_n_points: int = 0) -> str:
        """Compute Pearson correlation matrix between sensor parameters.

        Args:
            parameters: Comma-separated names (≥2), e.g. 'TRQ,HOOKLOAD,SPP,ROP'
            last_n_points: If > 0, restrict to the last N points. 0 = all.
        """
        params = _parse_params(parameters, KEY_PARAMS, df)
        if isinstance(params, str):
            return params
        if len(params) < 2:
            avail = ', '.join(KEY_PARAMS)
            return f"Error: Need at least 2 valid parameters. Available: {avail}"

        data = df.iloc[-last_n_points:] if last_n_points > 0 else df
        corr = data[params].corr()

        header = f"{'':>12} " + " ".join(f"{p:>10}" for p in params)
        lines = [
            f"CORRELATION MATRIX ({len(data):,} data points)",
            "",
            header,
        ]
        for p1 in params:
            row = f"{p1:>12} "
            row += " ".join(f"{corr.loc[p1, p2]:>10.3f}" for p2 in params)
            lines.append(row)

        lines.append("")
        lines.append("Notable correlations:")
        found = False
        for i, p1 in enumerate(params):
            for j, p2 in enumerate(params):
                if j <= i:
                    continue
                r = corr.loc[p1, p2]
                if abs(r) > 0.7:
                    strength = "STRONG" if abs(r) > 0.85 else "MODERATE"
                    direction = "positive" if r > 0 else "negative"
                    lines.append(f"  {p1} ↔ {p2}: r={r:.3f} ({strength} {direction})")
                    found = True
        if not found:
            lines.append("  No strong correlations (|r| > 0.7) found")

        return '\n'.join(lines)

    # ------------------------------------------------------------------ #
    # Tool 5: Query by depth
    # ------------------------------------------------------------------ #
    @tool
    def query_by_depth(min_depth: float, max_depth: float,
                       parameters: str = "") -> str:
        """Query sensor data within a depth interval (meters TVD).

        Args:
            min_depth: Minimum depth in meters, e.g. 4000
            max_depth: Maximum depth in meters, e.g. 4200
            parameters: Comma-separated sensor names. Leave empty for key params.
        """
        if 'DEPTH' not in df.columns:
            return "Error: DEPTH column not available in dataset."

        mask = (df['DEPTH'] >= min_depth) & (df['DEPTH'] <= max_depth)
        window = df.loc[mask]

        if len(window) == 0:
            return (f"No data for depth {min_depth:.0f}-{max_depth:.0f} m.\n"
                    f"Available: {df['DEPTH'].min():.0f} - {df['DEPTH'].max():.0f} m")

        params = _parse_params(parameters, KEY_PARAMS, df)
        if isinstance(params, str):
            return params

        lines = [
            f"DEPTH RANGE QUERY: {min_depth:.0f} - {max_depth:.0f} m TVD",
            f"Data points: {len(window):,}",
            f"Time span: {window['Time'].min()} to {window['Time'].max()}",
            f"Actual depth: {window['DEPTH'].min():.1f} - {window['DEPTH'].max():.1f} m",
            "",
            "Parameter Statistics:",
        ]
        for p in params:
            col = window[p]
            lines.append(
                f"  {p}: mean={col.mean():.2f}, std={col.std():.2f}, "
                f"min={col.min():.2f}, max={col.max():.2f} {UNITS.get(p, '')}"
            )

        lines.append("")
        lines.append("Trends (entry → exit):")
        for p in params:
            s, e = window[p].iloc[0], window[p].iloc[-1]
            pct = ((e - s) / abs(s) * 100) if s else 0
            arrow = "↑" if pct > 2 else "↓" if pct < -2 else "→"
            lines.append(f"  {p}: {s:.2f} → {e:.2f} ({arrow} {pct:+.1f}%)")

        return '\n'.join(lines)

    return [
        query_by_time_range,
        compute_statistics,
        detect_anomalies,
        correlate_parameters,
        query_by_depth,
    ]


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #
def _parse_params(parameters: str, KEY_PARAMS: list, df: pd.DataFrame):
    """Parse comma-separated parameter string. Returns list or error string."""
    if parameters.strip():
        params = [p.strip().upper() for p in parameters.split(',')]
        params = [p for p in params if p in df.columns]
        if not params:
            return f"Error: No valid parameters found. Available: {', '.join(KEY_PARAMS)}"
    else:
        params = [p for p in KEY_PARAMS if p in df.columns]
    return params
