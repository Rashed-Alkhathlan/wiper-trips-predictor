# Wiper Trip Predictor Upgrade Report

## Overview
This report summarizes the recent upgrade of the Wiper Trip Predictor and explains why the new approach is more credible for real operations. The core goal of this project is to support drilling decisions by estimating when wellbore cleaning risk is rising and a wiper trip should be considered. In practice, this is a difficult prediction problem because operational labels are noisy: continuing drilling does not always mean conditions are safe, and a recorded trip event may be reactive, scheduled, or ambiguous.

To address this, the model pipeline was redesigned from a mostly binary-label workflow into a confidence-weighted learning workflow with time-aware validation. This directly improves scientific rigor, operational realism, and explainability for decision-makers.

## Problem We Needed to Solve
The original logic treated labels as mostly hard truth. That is risky in drilling operations:

- A label of 0 (continued drilling) can include delayed action under rising risk.
- A label of 1 (trip occurred) does not always indicate an urgent reactive condition.
- Random train/test splits can leak future behavior into training when data is time-series.

This can make offline metrics look strong while underestimating deployment risk.

## What We Changed
### 1. Confidence-Weighted Labeling
Instead of trusting one class and distrusting another, we now assign confidence levels to labeled windows.

Key additions:
- Event profiling into reactive, scheduled, or ambiguous categories.
- Text cues from report descriptions to refine event type confidence.
- Precursor anomaly scoring before events using operational signals:
  - Torque trend (TRQ)
  - Standpipe pressure trend (SPP)
  - Rate of penetration behavior (ROP)
- Per-sample weights during supervised training rather than hard include/exclude decisions.

Why this matters:
- The model learns from all data but pays more attention to high-confidence supervision.
- Soft negatives and uncertain positives no longer dominate training in the wrong way.

### 2. Time-Aware Validation
We replaced random splitting with chronological 80/20 splitting by default.

Key additions:
- Primary split strategy: chronological train then future test.
- Controlled fallback to stratified random split only if chronological split lacks class coverage.

Why this matters:
- Prevents information leakage from the future.
- Produces more realistic evaluation for live deployment.

### 3. Operational and Business KPI Layer
We added performance outputs that are useful for leadership and operations, not only ML teams.

Operational KPI:
- False alerts per day

Business proxy KPIs:
- Time saved (hours)
- Reduced cost (USD)
- Money increase proxy from potential production gain (USD)
- ROI proxy

Important note:
- These business values are explicitly marked as proxies and assumption-driven.

## Current Results Snapshot
Using the current local run configuration, the training output produced the following:

- Training samples: 16,718
- Features: 82
- Validation split: Chronological 80/20
- AUC-ROC: 0.9997
- Precision: 0.980
- Recall: 0.985
- F1-score: 0.982
- Accuracy: 0.995
- False alerts/day: 0.78
- Time saved proxy: 336.8 hours
- Reduced cost proxy: $4,041,000
- Money increase proxy: $424,305
- ROI proxy: 24.81x

## Interpretation of Results
The quality metrics are very high and indicate excellent separability under the current label regime. However, two governance points are important for a credible presentation:

1. Label source in this run is reported as Pseudo-Labels with zero report-mined events.
2. This means report-PDF parsing is not currently active in the environment (commonly due to missing PDF dependency/runtime parsing path conditions).

So the right executive message is:
- The pipeline is now architecturally ready for robust confidence-weighted supervision.
- Current numbers are strong, but final field-grade validation should be done after enabling report-mined labels in the runtime environment.

## Why This Is Better Than the Previous Mindset
A previous idea was to trust label 0 and distrust label 1. The new system improves that by treating label quality as a continuum, not a binary rule.

Previous mindset:
- Hard trust by class

Current mindset:
- Confidence-weighted denoising
- Event context awareness (reactive vs scheduled)
- Time-safe validation

This is closer to real drilling behavior, where actions are often delayed, preventive, or constrained by operational economics.

## Business Impact Story for Presentation
You can frame the value in three layers:

1. Technical reliability:
- Better supervision under noisy labels
- Less leakage risk
- Better explainability through confidence tiers

2. Operational reliability:
- Fewer nuisance alarms per day
- Earlier risk recognition in approach windows before events

3. Financial relevance:
- Proxy reduction in non-productive time
- Proxy cost savings and production-related uplift
- High ROI potential under conservative assumptions

## Risks and Controls
Main risks:
- Confidence heuristics may still encode bias.
- Proxy economics may be challenged by finance/operations teams.
- Metrics may change when full report-mined labels are enabled.

Controls:
- Keep heuristics transparent and reviewable.
- Present ROI in conservative/base/aggressive ranges.
- Add walk-forward validation and period-by-period error analysis.

## Recommended Next Steps
1. Enable and verify report-mined labels in environment (PyMuPDF path and report parsing checks).
2. Run side-by-side comparison: pseudo-only vs report-mined confidence-weighted training.
3. Add walk-forward multi-period validation to show consistency.
4. Parameterize economic assumptions in the dashboard for scenario analysis.

## Conclusion
This upgrade moves the Wiper Trip Predictor from a label-fragile classifier toward an operationally credible decision-support system. The combination of confidence-weighted supervision, chronological validation, and business KPI outputs makes the model easier to trust, explain, and justify in a real drilling workflow.

In short: the project is now in a much stronger position for both technical review and management presentation, with clear next steps to convert strong prototype performance into deployment-grade confidence.
