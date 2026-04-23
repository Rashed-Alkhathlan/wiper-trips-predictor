"""
Daily Drilling Report Parser
==============================
Extracts real wiper trip / ream / short trip events from PDF reports.
Maps them onto 10-second interval timestamps for ML training labels.

Usage:
    from report_parser import build_label_series
    labels = build_label_series(df)  # df with DatetimeIndex or 'Time' col
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
            # De-duplicate: same date + event_type + description
            key = (evt["date"], evt["event_type"], evt["description"][:80])
            if key not in seen:
                seen.add(key)
                all_events.append(evt)

    all_events.sort(key=lambda e: (e["date"], e.get("start_hour") or ""))
    return all_events


# ---------------------------------------------------------------------------
# Build label series aligned to a DataFrame
# ---------------------------------------------------------------------------
def build_label_series(df: pd.DataFrame) -> pd.Series:
    """Build a binary label series (0/1) for the drilling DataFrame.

    Maps report-mined events onto the time-series data by matching
    report dates. For each report date with events, a time window
    around the event is labeled as 1 (high risk).

    If no report PDFs are available, returns all zeros.

    Args:
        df: DataFrame with a 'Time' column (datetime).

    Returns:
        pd.Series of int labels (0 or 1), same index as df.
    """
    labels = pd.Series(0, index=df.index, dtype=int)
    events = parse_all_reports() 

    if not events:
        return labels

    # Ensure Time column is datetime
    times = pd.to_datetime(df["Time"] if "Time" in df.columns else df.index)

    for evt in events:
        evt_date = evt["date"]
        weight = evt["weight"]

        # Only label high-confidence events
        if weight < 0.7:
            continue

        # Find rows matching this report date
        date_mask = times.dt.date == evt_date

        if evt["start_hour"] and evt["end_hour"]:
            # Precise time window from the report
            start_dt = datetime.combine(evt_date, evt["start_hour"])
            end_dt = datetime.combine(evt_date, evt["end_hour"])

            # Handle overnight (e.g. 22:00 → 02:00)
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)

            # Expand window by 30 min on each side (early warning)
            start_dt -= timedelta(minutes=30)
            end_dt += timedelta(minutes=30)

            time_mask = (times >= start_dt) & (times <= end_dt)
            labels.loc[time_mask] = 1
        else:
            # No precise time — label a 4-hour window around midday
            # (avoids labeling entire 24h which inflates positive rate)
            mid = datetime.combine(evt_date, datetime.strptime("12:00", "%H:%M").time())
            fallback_start = mid - timedelta(hours=2)
            fallback_end = mid + timedelta(hours=2)
            time_mask = (times >= fallback_start) & (times <= fallback_end)
            labels.loc[time_mask] = 1

        # Also label a 2-hour "approach window" before the event
        # (the model should learn to predict risk BEFORE the event)
        if evt["start_hour"]:
            approach_start = datetime.combine(
                evt_date, evt["start_hour"]
            ) - timedelta(hours=2)
            approach_end = datetime.combine(evt_date, evt["start_hour"])
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

    return {
        "n_events": len(events),
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
    print(f"Date range: {summary.get('date_range', 'N/A')}")
    print("\nEvent type breakdown:")
    for etype, count in sorted(
        summary["event_types"].items(), key=lambda x: -x[1]
    ):
        print(f"  {etype:20s} {count:3d}")
