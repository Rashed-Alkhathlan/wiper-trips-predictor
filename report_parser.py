"""
Daily Drilling Report Parser
==============================
Extracts real wiper trip / ream / short trip events from PDF reports.
Provides an event timeline for window-based ML labeling.

Usage:
    from report_parser import build_event_timeline
    events = build_event_timeline(df)  # list of (start_dt, end_dt, type, weight)
"""

import os
import re
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# Try importing pymupdf; gracefully degrade if unavailable
try:
    import pymupdf  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# ---------------------------------------------------------------------------
# Report directory (relative to this file)
# ---------------------------------------------------------------------------
_REPORT_DIR = os.path.join(
    os.path.dirname(__file__),
    "16A(78)-32_Daily_Reports",
    "16A(78)-32_Daily_Reports",
    "drilling",
)

# ---------------------------------------------------------------------------
# Event patterns — ordered by severity / specificity
# ---------------------------------------------------------------------------
_EVENT_PATTERNS = [
    # (regex_pattern, event_type, severity_weight)
    (r"short\s*trip",                   "short_trip",    1.0),
    (r"wiper\s*trip",                   "wiper_trip",    1.0),
    (r"back\s*ream",                    "back_ream",     0.9),
    (r"ream(?:ing)?\s+(?:from|curve)",  "reaming",       0.85),
    (r"ream\s+float",                   "ream_shoe",     0.7),
    (r"tight\s*spot",                   "tight_spot",    0.9),
    (r"high\s+torque",                  "high_torque",   0.8),
    (r"pull\s+out\s+of\s+hole|POOH",   "trip_out",      0.75),
    (r"trip\s+out\s+of\s+(?:the\s+)?hole", "trip_out",   0.75),
    (r"wash\s+down|wash\s+up",         "wash",           0.6),
    (r"stuck\s+pipe",                   "stuck_pipe",    1.0),
    (r"pack\s*off",                     "pack_off",      0.95),
    (r"over\s*pull",                    "overpull",      0.85),
    (r"drag",                           "drag",          0.7),
]

_COMPILED_PATTERNS = [
    (re.compile(p, re.IGNORECASE), etype, weight)
    for p, etype, weight in _EVENT_PATTERNS
]

# ---------------------------------------------------------------------------
# Reactive event types — events that indicate the *need* for a wiper trip.
# Planned trip-out / POOH operations are excluded because they don't
# represent deteriorating conditions that require prediction.
# ---------------------------------------------------------------------------
REACTIVE_EVENT_TYPES = {
    "wiper_trip", "short_trip", "reaming", "back_ream", "ream_shoe",
    "tight_spot", "high_torque", "stuck_pipe", "pack_off", "overpull", "drag",
}


# ---------------------------------------------------------------------------
# Parse a single report
# ---------------------------------------------------------------------------
def _parse_report(filepath: str) -> list[dict]:
    """Parse a single PDF daily drilling report.

    Returns a list of event dicts with keys:
        date, event_type, weight, depth_md, description
    """
    if not HAS_PYMUPDF:
        return []

    events = []
    try:
        doc = pymupdf.open(filepath)
    except Exception:
        return []

    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    # Extract report date
    m = re.search(r"RPT\s*DATE:\s*(\d{1,2}/\d{1,2}/\d{4})", text)
    if not m:
        return []
    try:
        report_date = datetime.strptime(m.group(1), "%m/%d/%Y").date()
    except ValueError:
        return []

    # Extract current MD/TVD
    depth_md = None
    m_depth = re.search(r"MD/TVD:\s*([\d,]+)", text)
    if m_depth:
        try:
            depth_md = float(m_depth.group(1).replace(",", ""))
        except ValueError:
            pass

    # Scan all lines for events
    for line in text.split("\n"):
        for pattern, etype, weight in _COMPILED_PATTERNS:
            if pattern.search(line):
                # Try to extract time range from the line
                time_match = re.search(
                    r"(\d{1,2}:\d{2})\s+(\d{1,2}:\d{2})", line
                )
                start_hour = None
                end_hour = None
                if time_match:
                    try:
                        start_hour = datetime.strptime(
                            time_match.group(1), "%H:%M"
                        ).time()
                        end_hour = datetime.strptime(
                            time_match.group(2), "%H:%M"
                        ).time()
                    except ValueError:
                        pass

                # Try to extract depth from line
                depth_match = re.search(
                    r"(\d{1,2},?\d{3})\s*['\"]?\s*(?:to|-)\s*(\d{1,2},?\d{3})",
                    line,
                )
                depth_from = None
                depth_to = None
                if depth_match:
                    try:
                        depth_from = float(
                            depth_match.group(1).replace(",", "")
                        )
                        depth_to = float(
                            depth_match.group(2).replace(",", "")
                        )
                    except ValueError:
                        pass

                events.append({
                    "date": report_date,
                    "start_hour": start_hour,
                    "end_hour": end_hour,
                    "event_type": etype,
                    "weight": weight,
                    "depth_md": depth_md,
                    "depth_from": depth_from,
                    "depth_to": depth_to,
                    "description": line.strip()[:200],
                })
                break  # Only one event type per line

    return events


# ---------------------------------------------------------------------------
# Parse all reports
# ---------------------------------------------------------------------------
def parse_all_reports(report_dir: Optional[str] = None) -> list[dict]:
    """Parse all PDF reports in the drilling reports directory.

    Returns a list of event dicts sorted by date.
    """
    rdir = report_dir or _REPORT_DIR
    if not os.path.isdir(rdir):
        warnings.warn(f"Report directory not found: {rdir}")
        return []

    all_events = []
    seen = set()  # de-duplicate across duplicate report files

    for fname in sorted(os.listdir(rdir)):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(rdir, fname)
        for evt in _parse_report(fpath):
            # De-duplicate: same date + event_type + full description
            key = (evt["date"], evt["event_type"], evt["description"])
            if key not in seen:
                seen.add(key)
                all_events.append(evt)

    all_events.sort(key=lambda e: (e["date"], e.get("start_hour") or ""))
    return all_events


# ---------------------------------------------------------------------------
# Build event timeline for window-based labeling
# ---------------------------------------------------------------------------
def build_event_timeline(df: pd.DataFrame,
                         reactive_only: bool = True) -> list[dict]:
    """Build a list of events with precise datetime ranges.

    Only includes events with precise time information (no midday guessing).
    When reactive_only=True, excludes planned trip_out/POOH events.

    Args:
        df: DataFrame with a 'Time' column (datetime) — used to determine
            the data time range for filtering.
        reactive_only: If True, only include REACTIVE_EVENT_TYPES.

    Returns:
        List of dicts with keys:
            start_dt, end_dt, event_type, weight, description
    """
    all_events = parse_all_reports()
    if not all_events:
        return []

    # Determine data time range
    times = pd.to_datetime(df["Time"] if "Time" in df.columns else df.index)
    data_start = times.min()
    data_end = times.max()

    timeline = []
    for evt in all_events:
        # Filter by event type
        if reactive_only and evt["event_type"] not in REACTIVE_EVENT_TYPES:
            continue

        # Skip events without precise time (no midday guessing)
        if not evt["start_hour"] or not evt["end_hour"]:
            continue

        # Build precise datetime range
        start_dt = datetime.combine(evt["date"], evt["start_hour"])
        end_dt = datetime.combine(evt["date"], evt["end_hour"])

        # Handle overnight (e.g. 22:00 → 02:00)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        # Skip events outside data range
        if end_dt < data_start.to_pydatetime() or start_dt > data_end.to_pydatetime():
            continue

        timeline.append({
            "start_dt": start_dt,
            "end_dt": end_dt,
            "event_type": evt["event_type"],
            "weight": evt["weight"],
            "description": evt["description"],
        })

    timeline.sort(key=lambda e: e["start_dt"])
    return timeline


# ---------------------------------------------------------------------------
# Legacy label series (kept for backward compatibility)
# ---------------------------------------------------------------------------
def build_label_series(df: pd.DataFrame) -> pd.Series:
    """Build a binary label series (0/1) for the drilling DataFrame.

    DEPRECATED: Use build_event_timeline() with window-based labeling instead.
    This function is kept for backward compatibility with the dumb-classifier
    notebook.

    Maps report-mined events onto the time-series data by matching
    report dates. For each report date with events, a time window
    around the event is labeled as 1 (high risk).

    If no report PDFs are available, returns all zeros.
    """
    labels = pd.Series(0, index=df.index, dtype=int)
    events = parse_all_reports()

    if not events:
        return labels

    times = pd.to_datetime(df["Time"] if "Time" in df.columns else df.index)

    for evt in events:
        weight = evt["weight"]
        if weight < 0.7:
            continue

        if evt["start_hour"] and evt["end_hour"]:
            start_dt = datetime.combine(evt["date"], evt["start_hour"])
            end_dt = datetime.combine(evt["date"], evt["end_hour"])
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)

            start_dt -= timedelta(minutes=30)
            end_dt += timedelta(minutes=30)

            time_mask = (times >= start_dt) & (times <= end_dt)
            labels.loc[time_mask] = 1
        # No midday fallback — events without time are skipped

        if evt["start_hour"]:
            approach_start = datetime.combine(
                evt["date"], evt["start_hour"]
            ) - timedelta(hours=2)
            approach_end = datetime.combine(evt["date"], evt["start_hour"])
            approach_mask = (times >= approach_start) & (times <= approach_end)
            labels.loc[approach_mask] = 1

    return labels


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def get_event_summary() -> dict:
    """Get a summary of parsed events for display."""
    events = parse_all_reports()
    if not events:
        return {"n_events": 0, "n_dates": 0, "event_types": {}}

    dates = set(e["date"] for e in events)
    type_counts = {}
    for e in events:
        t = e["event_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    # Count reactive events
    reactive_count = sum(
        1 for e in events if e["event_type"] in REACTIVE_EVENT_TYPES
    )
    reactive_with_time = sum(
        1 for e in events
        if e["event_type"] in REACTIVE_EVENT_TYPES and e["start_hour"]
    )

    return {
        "n_events": len(events),
        "n_reactive_events": reactive_count,
        "n_reactive_with_time": reactive_with_time,
        "n_dates": len(dates),
        "event_types": type_counts,
        "date_range": (min(dates).isoformat(), max(dates).isoformat()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    summary = get_event_summary()
    print(f"Parsed {summary['n_events']} events across {summary['n_dates']} dates")
    print(f"  Reactive events: {summary['n_reactive_events']}")
    print(f"  Reactive with time: {summary['n_reactive_with_time']}")
    print(f"Date range: {summary.get('date_range', 'N/A')}")
    print("\nEvent type breakdown:")
    for etype, count in sorted(
        summary["event_types"].items(), key=lambda x: -x[1]
    ):
        marker = " ← reactive" if etype in REACTIVE_EVENT_TYPES else " (excluded)"
        print(f"  {etype:20s} {count:3d}{marker}")
