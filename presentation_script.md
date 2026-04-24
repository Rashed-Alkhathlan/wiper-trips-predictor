# Presentation Script: Wiper Trip Predictor

## Slide 1 - Title and Context
Good [morning/afternoon], everyone.

Today I will present our Wiper Trip Predictor project, which is an AI-assisted drilling advisory system designed to estimate when wellbore cleaning risk is increasing, so the team can take action earlier and more confidently.

The business problem is simple but high-impact: wiper trips are necessary for safe drilling, but they are expensive and operationally disruptive. If we trip too early, we lose efficiency. If we trip too late, we increase the risk of stuck pipe, high torque escalation, and non-productive time.

Our goal is to turn raw drilling telemetry into an actionable risk signal that supports better timing decisions.

---

## Slide 2 - Problem We Are Solving
The core challenge is label quality.

In real drilling operations:
- A label of 0, meaning drilling continued, does not always mean conditions were safe.
- A label of 1, meaning a trip occurred, does not always mean a reactive emergency; some trips are scheduled.

So we are dealing with noisy operational labels, not perfect ground truth.

A second challenge is evaluation realism. In time-series data, random train/test splits can leak future information into training and create overly optimistic results.

So our project solves two things together:
1. How to train robustly with noisy labels.
2. How to evaluate realistically for deployment.

---

## Slide 3 - Data and Feature Pipeline
Our input is high-frequency drilling telemetry from Well 16A(78)-32.

From this data we engineer a rich feature set including:
- Core drilling signals like WOB, ROP, RPM, torque, pressure, and flow.
- Rolling trend and volatility features.
- Cross-feature interactions.
- Mechanical Specific Energy, or MSE, related features.
- Optional operational channels like hookload, pit gain/loss, and return flow when available.

This gives the model both instantaneous state and trend context, which is essential for pre-event risk detection.

---

## Slide 4 - Model Architecture
We use a two-model ensemble:

1. GradientBoostingClassifier
- Supervised branch.
- Learns structured relationships between engineered features and risk labels.

2. IsolationForest
- Unsupervised anomaly branch.
- Detects unusual operating patterns even when labels are imperfect.

Final risk score:
- 0.65 times supervised probability
- plus 0.35 times anomaly score

This gives us both pattern learning and anomaly sensitivity.

---

## Slide 5 - Key Upgrade: Confidence-Weighted Supervision
The most important upgrade is moving from hard trust in labels to confidence-weighted learning.

Instead of saying:
- trust all label 0
- distrust all label 1

we now do this:
- classify events as reactive, scheduled, or ambiguous
- score precursor evidence before events using torque, pressure, and ROP behavior
- assign confidence tiers to event windows
- train with per-sample weights so high-confidence samples influence learning more than uncertain samples

This reflects drilling reality much better than binary trust rules.

---

## Slide 6 - Validation Strategy and Leakage Control
We changed evaluation to a chronological 80/20 split by default.

That means the model trains on earlier data and is tested on future data, which is closer to real operations.

We only use stratified random fallback if class coverage is invalid in a strict time split.

This design reduces leakage risk and improves credibility of reported performance.

---

## Slide 7 - Current Results Snapshot
From the latest local run, we observed:
- Samples: 16,718
- Features: 82
- Split: Chronological 80/20
- AUC-ROC: 0.9997
- Precision: 0.980
- Recall: 0.985
- F1: 0.982
- Accuracy: 0.995
- False alerts per day: 0.78

Business proxy outputs:
- Time saved: 336.8 hours
- Reduced cost: 4.04 million USD
- Money increase proxy: 424 thousand USD
- ROI proxy: 24.81x

Important governance note:
In this run, label source was pseudo-labels, with zero report-mined events active. So these are strong prototype indicators, but final field-grade claims require enabling report-mined labels in runtime and re-validating.

---

## Slide 8 - Business Value and Why It Matters
The value story is three-layered:

1. Technical value
- Better learning under noisy labels
- Lower leakage risk
- More transparent model behavior

2. Operational value
- Earlier warning before severe conditions
- Fewer nuisance alarms
- Better support for drilling decisions

3. Financial value
- Lower non-productive time
- Lower intervention-related costs
- Potential production uplift

This is not just an ML demo; it is a decision-support foundation for safer and more efficient drilling.

---

## Slide 9 - Risks and Controls
We are transparent about limitations.

Risks:
- Confidence heuristics can still embed bias.
- Proxy economics depend on assumptions.
- Performance can shift when full report-derived labels are enabled.

Controls:
- Keep heuristics explicit and auditable.
- Present conservative, base, and aggressive economic scenarios.
- Add walk-forward validation across multiple periods.
- Compare pseudo-only versus report-mined training head-to-head.

---

## Slide 10 - Next Steps and Close
Our immediate next steps are:
1. Fully enable and verify report-mined labeling in the runtime environment.
2. Run side-by-side model validation: pseudo-labels versus report-enriched labels.
3. Add walk-forward evaluation and period-by-period error stability checks.
4. Make economic assumptions configurable in the dashboard.

To conclude:
This project has moved from a label-fragile classifier toward an operationally credible advisory system. The key improvement is not only model choice; it is the full pipeline design: confidence-aware labeling, time-safe evaluation, and business-linked outputs.

Thank you. I am happy to answer questions on model design, assumptions, and deployment readiness.

---

## Optional Q&A Short Answers
Q: Why not trust label 0 as safe?
A: Because continued drilling can include delayed action under rising risk; that makes label 0 noisy.

Q: Why combine supervised and anomaly models?
A: Supervised learning captures known patterns, while anomaly detection catches unusual behavior not cleanly represented in labels.

Q: Are ROI numbers guaranteed?
A: No, they are scenario-based proxies. They guide decision discussions and must be calibrated with field economics.

Q: What is needed before deployment?
A: Full report-mined label activation, walk-forward validation, and threshold tuning with drilling domain experts.
