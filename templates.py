"""
HTML Templates for the Drilling Advisory Dashboard
====================================================
Keeps all HTML generation out of app.py for readability.
Each function returns an HTML string ready for st.markdown().
"""


# ---------------------------------------------------------------------------
# Top Status Bar
# ---------------------------------------------------------------------------
def top_bar(depth: float, time_str: str, mode: str, risk: float,
            level: str, risk_class: str) -> str:
    return (
        '<div class="top-bar">'
        '<div class="top-bar-item">'
        '  <div class="top-bar-label">Well</div>'
        '  <div class="top-bar-value">16A(78)-32</div>'
        '</div>'
        '<div class="top-bar-item">'
        '  <div class="top-bar-label">Rig</div>'
        '  <div class="top-bar-value">Demo Rig-01</div>'
        '</div>'
        '<div class="top-bar-item">'
        '  <div class="top-bar-label">Depth TVD</div>'
        f'  <div class="top-bar-value">{depth:,.1f}'
        '    <span style="font-size:12px;color:#64748b;">m</span>'
        '  </div>'
        '</div>'
        '<div class="top-bar-item">'
        '  <div class="top-bar-label">Time</div>'
        f'  <div class="top-bar-value" style="font-size:14px;">{time_str}</div>'
        '</div>'
        '<div class="top-bar-item">'
        '  <div class="top-bar-label">Status</div>'
        '  <div class="top-bar-value" style="color:#22c55e;">DRILLING</div>'
        '</div>'
        '<div class="top-bar-item">'
        '  <div class="top-bar-label">Model</div>'
        '  <div class="top-bar-value" style="font-size:14px;color:#38bdf8;">'
        '    GBT + IF Ensemble</div>'
        '</div>'
        '<div class="top-bar-item">'
        '  <div class="top-bar-label">Wiper Trip Risk</div>'
        f'  <div class="risk-badge {risk_class}">{risk:.2f} — {level}</div>'
        '</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Metric Card
# ---------------------------------------------------------------------------
def metric_card(label: str, value: float, unit: str,
                arrow: str, trend_color: str) -> str:
    return (
        '<div class="metric-card">'
        f'  <div class="metric-label">{label}</div>'
        f'  <div>'
        f'    <span class="metric-value">{value:,.1f}</span>'
        f'    <span class="metric-unit">{unit}</span>'
        f'    <span class="metric-trend" style="color:{trend_color};">'
        f'      {arrow}</span>'
        f'  </div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Advisory Panel
# ---------------------------------------------------------------------------
_REC_BG = {
    "HIGH":     "rgba(239,68,68,0.15)",
    "MODERATE": "rgba(245,158,11,0.15)",
    "LOW":      "rgba(34,197,94,0.15)",
}


def advisory_panel(level: str, advisory: dict) -> str:
    rec_bg = _REC_BG.get(level, "#1e293b")
    color = advisory["color"]

    reasons = "".join(
        f'<div class="advisory-reason">{r}</div>' for r in advisory["reasons"]
    )
    actions = "".join(
        f'<div class="advisory-action">{i}. {a}</div>'
        for i, a in enumerate(advisory["actions"], 1)
    )
    conf_pct = advisory["confidence"] * 100

    return (
        '<div class="advisory-panel">'
        '  <div class="advisory-header">Recommendation</div>'
        f'  <div class="advisory-rec" style="background:{rec_bg};'
        f'    color:{color}; border:1px solid {color};">'
        f'    {advisory["recommendation"]}</div>'
        '  <div class="advisory-section-title">Analysis</div>'
        f'  {reasons}'
        f'  <div class="advisory-interpretation">'
        f'    {advisory["interpretation"]}</div>'
        '  <div class="advisory-section-title">Recommended Actions</div>'
        f'  {actions}'
        '  <div class="advisory-section-title">Confidence</div>'
        f'  <div class="confidence-bar-container">'
        f'    <div class="confidence-bar"'
        f'         style="width:{conf_pct}%;background:{color};"></div></div>'
        f'  <div class="confidence-label">{advisory["confidence"]:.2f}</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# ML Model Output Panel
# ---------------------------------------------------------------------------
def _score_color(score: float) -> str:
    if score > 0.7:
        return "#ef4444"
    elif score > 0.4:
        return "#f59e0b"
    return "#22c55e"


def model_scores(rf_prob: float, if_score: float,
                 risk: float, risk_color: str) -> str:
    rf_c = _score_color(rf_prob)
    if_c = _score_color(if_score)
    return (
        '<div class="model-panel">'
        '  <div class="model-score-row">'
        '    <span class="model-score-label">Gradient Boost Prob.</span>'
        f'    <span class="model-score-value" style="color:{rf_c};">'
        f'      {rf_prob:.3f}</span></div>'
        '  <div class="model-score-row">'
        '    <span class="model-score-label">Isolation Forest Score</span>'
        f'    <span class="model-score-value" style="color:{if_c};">'
        f'      {if_score:.3f}</span></div>'
        '  <div class="model-score-row"'
        '       style="border-top:1px solid #334155;padding-top:8px;">'
        '    <span class="model-score-label">Ensemble (0.65 GBT + 0.35 IF)</span>'
        f'    <span class="model-score-value" style="color:{risk_color};">'
        f'      {risk:.3f}</span></div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Model Information Panel
# ---------------------------------------------------------------------------
def model_info(metrics: dict) -> str:
    model_type = metrics.get("model_type", "GBT + Isolation Forest")
    label_src = metrics.get("label_source", "Pseudo-Labels")
    n_events = metrics.get("n_report_events", 0)
    split_strategy = metrics.get("split_strategy", "Chronological 80/20")
    label_display = f"{label_src} ({n_events} events)" if n_events else label_src
    label_color = "#22c55e" if "Report" in label_src else "#f59e0b"

    rows = [
        ("Model Type",       model_type, None),
        ("Label Source",     label_display, label_color),
        ("Validation Split", split_strategy, "#38bdf8"),
        ("Features",         str(metrics.get("n_features", "—")), None),
        ("Training Samples", f'{metrics.get("n_samples", 0):,}', None),
        ("Avg Pos Weight",   f'{metrics.get("avg_positive_weight", 0):.3f}', None),
        ("Avg Neg Weight",   f'{metrics.get("avg_negative_weight", 0):.3f}', None),
        ("PU Positives",     f'{metrics.get("pu_positive_count", 0):,}', None),
        ("Reliable Negatives", f'{metrics.get("pu_reliable_negative_count", 0):,}', None),
        ("Ambiguous Unlabeled", f'{metrics.get("pu_ambiguous_unlabeled_count", 0):,}', None),
        ("Reactive Events",  str(metrics.get("n_reactive_events", 0)), None),
        ("Scheduled Events", str(metrics.get("n_scheduled_events", 0)), None),
        ("False Alerts/Day", f'{metrics.get("false_alerts_per_day", 0):.2f}', None),
        ("Time Saved (h)*",  f'{metrics.get("time_saved_hours_proxy", 0):.1f}', "#22c55e"),
        ("Reduced Cost ($)*", f'{metrics.get("reduced_cost_usd_proxy", 0):,.0f}', "#22c55e"),
        ("Money Increase ($)*", f'{metrics.get("money_increase_usd_proxy", 0):,.0f}', "#22c55e"),
        ("ROI Proxy (x)*",   f'{metrics.get("roi_proxy", 0):.2f}', "#22c55e"),
        ("AUC-ROC",          f'{metrics.get("auc_roc", 0):.4f}', "#22c55e"),
        ("Precision",        f'{metrics.get("precision", 0):.3f}', None),
        ("Recall",           f'{metrics.get("recall", 0):.3f}', None),
        ("F1-Score",         f'{metrics.get("f1_score", 0):.3f}', None),
        ("Accuracy",         f'{metrics.get("accuracy", 0):.3f}', None),
    ]
    html = '<div class="model-panel">'
    for label, value, color in rows:
        style = f' style="color:{color};"' if color else ""
        html += (
            '<div class="model-stat">'
            f'  <span class="model-stat-label">{label}</span>'
            f'  <span class="model-stat-value"{style}>{value}</span>'
            '</div>'
        )
    html += (
        '<div class="model-stat" style="border-top:1px solid #334155;margin-top:6px;">'
        '  <span class="model-stat-label" style="font-size:11px;color:#64748b;">'
        '  * Proxy values use configurable cost/production assumptions'
        '  </span>'
        '  <span></span>'
        '</div>'
    )
    html += '</div>'
    return html


# ---------------------------------------------------------------------------
# Event Log
# ---------------------------------------------------------------------------
def event_log(events: list[dict]) -> str:
    if not events:
        return (
            '<div class="event-log"><div class="event-entry">'
            '<span class="event-info">'
            'Monitoring active — no threshold crossings detected'
            '</span></div></div>'
        )

    html = '<div class="event-log">'
    for evt in events[:20]:
        sev = f"event-{evt['severity']}"
        html += (
            f'<div class="event-entry">'
            f'  <span class="event-time">[{evt["time"]}]</span>'
            f'  <span class="{sev}">{evt["message"]}</span>'
            f'</div>'
        )
    html += '</div>'
    return html


# ---------------------------------------------------------------------------
# Section Title
# ---------------------------------------------------------------------------
def section_title(text: str, margin_top: int = 0) -> str:
    style = f' style="margin-top:{margin_top}px;"' if margin_top else ""
    return f'<div class="section-title"{style}>{text}</div>'
