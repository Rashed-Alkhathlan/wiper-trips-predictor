# Wiper Trip Predictor - Implementation Plan

## Objective
Build a robust, operations-focused predictor that handles noisy labels, improves early warning quality, and quantifies business impact (time saved, reduced cost, ROI).

## Why This Plan
Current labels are imperfect:
- Label 1 (trip) can include both reactive and scheduled actions.
- Label 0 (continue drilling) is not always a true safe negative.

Therefore, the model should use confidence-weighted supervision, not hard trust/distrust by class.

## Success Criteria
- Better early warning with fewer false alarms.
- Time-aware validation (no leakage).
- Explainable alert reasons for operations teams.
- Business metrics in dashboard: avoided NPT, reduced costs, ROI proxy.

---

## Phase 1 - Label Quality Audit (Week 1)

### Tasks
- Parse all report events and classify event quality:
  - Reactive trip
  - Scheduled/maintenance trip
  - Ambiguous
- Build a label audit table with:
  - Event type
  - Time precision available (exact window vs fallback)
  - Pre-event anomaly evidence
  - Confidence tier

### Code Targets
- report_parser.py
- model.py

### Deliverables
- CSV/DF summary of event confidence tiers.
- Baseline noise report for labels 0 and 1.

### Done When
- We can quantify how many positives are high/medium/low confidence.
- We can quantify likely delayed-action negatives.

---

## Phase 2 - Confidence-Weighted Labeling (Week 1-2)

### Tasks
- Keep binary target for compatibility, but add per-sample confidence score.
- Generate sample weights from confidence:
  - High-confidence positive: high weight
  - Medium-confidence positive: medium weight
  - Soft negative: low-to-medium weight
  - Strong negative: medium/high weight
- Preserve approach-window labeling before events for anticipation.

### Code Targets
- report_parser.py (event confidence metadata)
- model.py (return labels + weights)

### Deliverables
- New label pipeline returning:
  - y_binary
  - sample_weight
  - label_source/confidence stats

### Done When
- Training uses sample_weight consistently.
- Label confidence distributions are logged in metrics.

---

## Phase 3 - Time-Aware Training and Validation (Week 2)

### Tasks
- Replace random split with chronological split (or grouped by date).
- Ensure train set always occurs before test set in time.
- Add walk-forward evaluation for robustness.

### Code Targets
- model.py

### Deliverables
- Leakage-safe validation pipeline.
- Before/after comparison vs random split.

### Done When
- No future data appears in training for any test interval.
- Performance is reported per time fold.

---

## Phase 4 - Reactive Trip Detection Logic (Week 2-3)

### Tasks
- Add heuristics to separate likely reactive vs scheduled trips:
  - Rising torque/pressure
  - Falling ROP
  - Hookload drag and flow imbalance evidence
  - Report text cues for planned operations
- Increase confidence for reactive events with precursor evidence.
- Decrease confidence for likely scheduled events.

### Code Targets
- report_parser.py
- model.py

### Deliverables
- Event scoring function for trip type confidence.
- Improved positive label precision.

### Done When
- Positive labels are split into confidence tiers with transparent rules.

---

## Phase 5 - Modeling Improvements (Week 3)

### Tasks
- Keep GradientBoosting + IsolationForest ensemble.
- Feed anomaly score into supervised branch as a feature.
- Retune ensemble weights based on early-warning objectives.

### Code Targets
- model.py

### Deliverables
- Updated ensemble calibration.
- Feature-importance update and top-driver report.

### Done When
- Improved lead time and stable false alert rate.

---

## Phase 6 - Ops Metrics and Business Value (Week 3-4)

### Tasks
- Add operational metrics:
  - Lead time before trip/risk event
  - False alerts per day
  - Missed critical events
- Add economic outputs:
  - Estimated NPT hours avoided
  - Reduced intervention/remediation cost
  - ROI estimate (benefit/cost ratio)
  - Time saved and money increase proxy

### Code Targets
- app.py
- engine.py
- README.md

### Deliverables
- Dashboard section with value metrics.
- Clearly documented assumptions for cost model.

### Done When
- Users can see both model quality and business impact in one view.

---

## Phase 7 - Documentation and Governance (Week 4)

### Tasks
- Document label assumptions and known limitations.
- Add model card section:
  - Intended use
  - Failure modes
  - Monitoring plan
- Add reproducible evaluation script/checklist.

### Code Targets
- README.md
- optional docs/*.md

### Deliverables
- Updated documentation and runbook.

### Done When
- Another engineer can reproduce training/evaluation end-to-end.

---

## Risk Register and Mitigations

- Risk: Label confidence heuristics introduce bias.
  - Mitigation: Keep heuristics transparent; validate with manual sample review.
- Risk: Time-aware split reduces headline metrics.
  - Mitigation: Prioritize realistic deployment performance over inflated offline scores.
- Risk: ROI estimates may be challenged.
  - Mitigation: Show scenario ranges (conservative/base/aggressive).

---

## Immediate Next 3 Actions

1. Implement confidence output in report parsing.
2. Add sample_weight training path in model training.
3. Switch to chronological validation and publish baseline comparison.

## Definition of Done

- Confidence-weighted labels integrated.
- Time-aware validation integrated.
- Alert quality and business metrics visible in dashboard.
- README updated with assumptions, limitations, and reproducible workflow.
