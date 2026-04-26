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
    "..",
    "data",
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
    (r"pull\s+out\s+of\s+hole|POOH",   "trip_out",      0.65),
    (r"trip\s+out\s+of\s+(?:the\s+)?hole", "trip_out",   0.65),
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

_REACTIVE_EVENT_TYPES = {
    "tight_spot", "high_torque", "stuck_pipe", "pack_off", "overpull", "drag"
}
_SCHEDULED_EVENT_TYPES = {
    "trip_out", "wiper_trip", "short_trip", "wash", "back_ream"
}
_SCHEDULED_TEXT_CUES = (
    "planned", "schedule", "routine", "program", "circulate and condition"
)
_REACTIVE_TEXT_CUES = (
    "high torque", "tight", "stuck", "pack off", "overpull", "drag", "unable"
)


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

    if "TIME BREAKDOWN" in text:
        text = text.split("TIME BREAKDOWN")[-1]

    current_start = None
    current_end = None

    # Scan all lines for events
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        m_time_single = re.fullmatch(r"(\d{1,2}:\d{2})", line)
        if m_time_single:
            if current_start and not current_end:
                current_end = m_time_single.group(1)
            else:
                current_start = m_time_single.group(1)
                current_end = None
            continue

        m_time_double = re.search(r"(\d{1,2}:\d{2})\s+(\d{1,2}:\d{2})", line)
        if m_time_double:
            current_start = m_time_double.group(1)
            current_end = m_time_double.group(2)

        for pattern, etype, weight in _COMPILED_PATTERNS:
            if pattern.search(line):
                # Ignore lines that are just planning or preparing to do something
                if re.search(r"(?i)\b(?:prepare|plan|planned|forward|future)\b", line):
                    continue
                # Try to extract time range from the line
                start_hour = None
                end_hour = None
                if current_start and current_end:
                    try:
                        start_hour = datetime.strptime(
                            current_start, "%H:%M"
                        ).time()
                        end_hour = datetime.strptime(
                            current_end, "%H:%M"
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


def _safe_ratio(a: float, b: float) -> float:
    """Return a stable ratio for anomaly calculations."""
    if abs(b) < 1e-6:
        return 1.0
    return float(a) / float(b)


def _event_text_profile(evt: dict) -> dict:
    """Classify event as reactive/scheduled/ambiguous using type and text cues."""
    event_type = evt.get("event_type", "")
    desc = (evt.get("description") or "").lower()

    reactive_hit = any(token in desc for token in _REACTIVE_TEXT_CUES)
    scheduled_hit = any(token in desc for token in _SCHEDULED_TEXT_CUES)

    if event_type in _REACTIVE_EVENT_TYPES or reactive_hit:
        profile = "reactive"
    elif event_type in _SCHEDULED_EVENT_TYPES or scheduled_hit:
        profile = "scheduled"
    else:
        profile = "ambiguous"

    has_precise_time = bool(evt.get("start_hour") and evt.get("end_hour"))
    return {
        "profile": profile,
        "has_precise_time": has_precise_time,
        "reactive_text_hit": reactive_hit,
        "scheduled_text_hit": scheduled_hit,
    }


def _compute_precursor_score(df: pd.DataFrame, times: pd.Series, evt: dict) -> float:
    """Estimate pre-event anomaly strength from operational signals."""
    if not evt.get("start_hour"):
        return 0.0

    actual_date = evt["date"] - timedelta(days=1) if evt["start_hour"].hour >= 6 else evt["date"]
    evt_start = datetime.combine(actual_date, evt["start_hour"])
    pre_mask = (times >= evt_start - timedelta(minutes=60)) & (times < evt_start)
    base_mask = (times >= evt_start - timedelta(minutes=180)) & (times < evt_start - timedelta(minutes=60))

    if pre_mask.sum() < 5 or base_mask.sum() < 5:
        return 0.0

    pre = df.loc[pre_mask]
    base = df.loc[base_mask]

    score = 0.0
    trq_ratio = _safe_ratio(pre["TRQ"].mean(), base["TRQ"].mean())
    spp_ratio = _safe_ratio(pre["SPP"].mean(), base["SPP"].mean())
    rop_ratio = _safe_ratio(pre["ROP"].mean(), base["ROP"].mean())

    score += max(0.0, min(1.0, (trq_ratio - 1.0) / 0.35)) * 0.4
    score += max(0.0, min(1.0, (spp_ratio - 1.0) / 0.35)) * 0.3
    score += max(0.0, min(1.0, (1.0 - rop_ratio) / 0.35)) * 0.3

    return float(np.clip(score, 0.0, 1.0))


def _event_confidence(evt: dict, precursor_score: float) -> dict:
    """Compute event confidence and tier for weighted supervision."""
    profile = _event_text_profile(evt)
    base_weight = float(evt.get("weight", 0.7))

    # Convert report pattern severity to a baseline confidence range.
    conf = 0.35 + 0.45 * np.clip((base_weight - 0.6) / 0.4, 0.0, 1.0)
    if profile["has_precise_time"]:
        conf += 0.08
    conf += 0.30 * precursor_score

    if profile["profile"] == "reactive":
        conf += 0.10
    elif profile["profile"] == "scheduled":
        conf -= 0.12

    conf = float(np.clip(conf, 0.15, 1.0))
    if conf >= 0.75:
        tier = "high"
    elif conf >= 0.5:
        tier = "medium"
    else:
        tier = "low"

    return {
        "confidence": conf,
        "tier": tier,
        "profile": profile["profile"],
        "has_precise_time": profile["has_precise_time"],
    }


# ---------------------------------------------------------------------------
# Build label series aligned to a DataFrame
# ---------------------------------------------------------------------------
def build_label_bundle(df: pd.DataFrame) -> dict:
    """Build binary labels plus per-sample confidence for the DataFrame.

    Returns a dict with:
        labels: binary series (0/1)
        confidence: confidence for positive windows (0..1)
        event_profiles: event-level confidence metadata
    """
    labels = pd.Series(0, index=df.index, dtype=int)
    confidence = pd.Series(0.0, index=df.index, dtype=float)
    events = parse_all_reports()
    event_profiles = []

    if not events:
        return {
            "labels": labels,
            "confidence": confidence,
            "event_profiles": event_profiles,
        }

    times = pd.to_datetime(df["Time"] if "Time" in df.columns else df.index)

    for evt in events:
        evt_date = evt["date"]
        weight = evt["weight"]

        if weight < 0.7:
            continue

        precursor = _compute_precursor_score(df, times, evt)
        conf_meta = _event_confidence(evt, precursor)
        evt_conf = conf_meta["confidence"]

        if evt["start_hour"] and evt["end_hour"]:
            actual_date = evt_date - timedelta(days=1) if evt["start_hour"].hour >= 6 else evt_date
            start_dt = datetime.combine(actual_date, evt["start_hour"])
            end_dt = datetime.combine(actual_date, evt["end_hour"])
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)

            start_dt -= timedelta(minutes=30)
            end_dt += timedelta(minutes=30)
            event_mask = (times >= start_dt) & (times <= end_dt)
            labels.loc[event_mask] = 1
            confidence.loc[event_mask] = np.maximum(confidence.loc[event_mask], evt_conf)
        else:
            mid = datetime.combine(evt_date, datetime.strptime("12:00", "%H:%M").time())
            fallback_start = mid - timedelta(hours=2)
            fallback_end = mid + timedelta(hours=2)
            event_mask = (times >= fallback_start) & (times <= fallback_end)
            labels.loc[event_mask] = 1
            confidence.loc[event_mask] = np.maximum(confidence.loc[event_mask], evt_conf * 0.85)

        if evt["start_hour"]:
            approach_start = datetime.combine(evt_date, evt["start_hour"]) - timedelta(hours=2)
            approach_end = datetime.combine(evt_date, evt["start_hour"])
            approach_mask = (times >= approach_start) & (times <= approach_end)
            labels.loc[approach_mask] = 1
            confidence.loc[approach_mask] = np.maximum(confidence.loc[approach_mask], evt_conf * 0.90)

        event_profiles.append({
            "date": evt_date,
            "event_type": evt.get("event_type"),
            "profile": conf_meta["profile"],
            "tier": conf_meta["tier"],
            "confidence": round(evt_conf, 3),
            "precursor_score": round(float(precursor), 3),
            "has_precise_time": conf_meta["has_precise_time"],
        })

    return {
        "labels": labels,
        "confidence": confidence,
        "event_profiles": event_profiles,
    }


def build_label_series(df: pd.DataFrame) -> pd.Series:
    """Build binary labels for training compatibility with existing code."""
    return build_label_bundle(df)["labels"]


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
