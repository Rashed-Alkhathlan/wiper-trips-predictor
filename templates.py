"""
HTML Templates — Simplified Drilling Advisory Dashboard
=========================================================
Focused on one clear answer: "Will a wiper trip be needed?"
"""


def prediction_hero(risk: float, level: str, recommendation: str,
                    probability_pct: float) -> str:
    """The big hero section showing the prediction answer."""
    level_lower = level.lower()

    if level_lower == "high":
        answer = "LIKELY YES"
        prob_color = "#ef4444"
    elif level_lower == "moderate":
        answer = "POSSIBLY"
        prob_color = "#f59e0b"
    else:
        answer = "UNLIKELY"
        prob_color = "#22c55e"

    return (
        f'<div class="prediction-hero prediction-hero-{level_lower}">'
        '  <div class="prediction-question">'
        '    Will a wiper trip be needed in the next 4 hours?'
        '  </div>'
        f'  <div class="prediction-answer prediction-answer-{level_lower}">'
        f'    {answer}'
        '  </div>'
        f'  <div class="prediction-probability" style="color:{prob_color};">'
        f'    Probability: {probability_pct:.0f}%'
        '  </div>'
        f'  <div class="prediction-recommendation prediction-rec-{level_lower}">'
        f'    ⟶ {recommendation}'
        '  </div>'
        '</div>'
    )


def context_bar(depth: float, time_str: str, mode: str) -> str:
    """Compact context bar showing well info."""
    return (
        '<div class="context-bar">'
        '  <div class="context-item">'
        '    <div class="context-label">Well</div>'
        '    <div class="context-value">16A(78)-32</div>'
        '  </div>'
        '  <div class="context-item">'
        '    <div class="context-label">Depth TVD</div>'
        f'    <div class="context-value">{depth:,.0f} m</div>'
        '  </div>'
        '  <div class="context-item">'
        '    <div class="context-label">Time</div>'
        f'    <div class="context-value">{time_str}</div>'
        '  </div>'
        '  <div class="context-item">'
        '    <div class="context-label">Mode</div>'
        f'    <div class="context-value">{mode}</div>'
        '  </div>'
        '</div>'
    )


def why_panel(advisory: dict) -> str:
    """Plain English explanation of why the prediction is what it is."""
    level = advisory.get("level", "LOW")

    # Build reasons HTML
    reasons_html = ""
    for reason in advisory["reasons"]:
        if level == "HIGH":
            cls = "why-reason why-reason-critical"
        elif level == "MODERATE":
            cls = "why-reason why-reason-warning"
        else:
            cls = "why-reason why-reason-ok"
        reasons_html += f'<div class="{cls}">{reason}</div>'

    # Build actions HTML
    actions_html = ""
    for i, action in enumerate(advisory["actions"], 1):
        actions_html += f'<div class="why-action">{i}. {action}</div>'

    interpretation = advisory.get("interpretation", "")

    return (
        '<div class="why-panel">'
        '  <div class="why-title">Why?</div>'
        f'  {reasons_html}'
        f'  <div style="font-size:13px;color:#64748b;font-style:italic;'
        f'    margin:12px 0;padding:10px;background:#1a1f35;border-radius:6px;">'
        f'    {interpretation}'
        '  </div>'
        '  <div class="why-title" style="margin-top:16px;">Recommended Actions</div>'
        f'  {actions_html}'
        '</div>'
    )


def model_info_compact(metrics: dict) -> str:
    """Compact model info for the expander."""
    rows = [
        ("Model", metrics.get("model_type", "—")),
        ("Labels From", metrics.get("label_source", "—")),
        ("Prediction", f'{metrics.get("prediction_horizon_hrs", 4)}h ahead'),
        ("Window", f'{metrics.get("window_minutes", 30)} min'),
        ("Split", metrics.get("split_type", "—")),
        ("Features", str(metrics.get("n_features", "—"))),
        ("Train / Test", f'{metrics.get("n_train", 0):,} / {metrics.get("n_test", 0):,}'),
        ("AUC-ROC", f'{metrics.get("auc_roc", 0):.4f}'),
        ("Precision", f'{metrics.get("precision", 0):.3f}'),
        ("Recall", f'{metrics.get("recall", 0):.3f}'),
        ("F1", f'{metrics.get("f1_score", 0):.3f}'),
    ]

    html = '<div class="model-compact">'
    for label, value in rows:
        html += (
            '<div class="model-compact-row">'
            f'  <span class="model-compact-label">{label}</span>'
            f'  <span class="model-compact-value">{value}</span>'
            '</div>'
        )
    html += '</div>'
    return html


# ---------------------------------------------------------------------------
# Section Title (reused)
# ---------------------------------------------------------------------------
def section_title(text: str, margin_top: int = 0) -> str:
    style = f' style="margin-top:{margin_top}px;"' if margin_top else ""
    return (
        f'<div style="font-size:12px;color:#64748b;text-transform:uppercase;'
        f'letter-spacing:1px;font-weight:600;margin-bottom:10px;'
        f'padding-bottom:6px;border-bottom:1px solid #1e293b;"{style}>{text}</div>'
    )
