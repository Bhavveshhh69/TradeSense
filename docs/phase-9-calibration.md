# Phase 9: Probability Calibration & Confidence Discipline (Locked)

Phase 9 is a post-model layer that makes probabilities honest and confidence cautious. It does not change the model or features.

## Overview
Raw model probabilities are often overconfident. Calibration exists to make reported probabilities better aligned with reality. This is about honesty, not accuracy.

## Probability Contract (Mandatory)
- `probability` = calibrated probability returned to clients.
- `probability_raw` = raw model output before calibration.
- Calibration is required. If artifacts are missing, inference fails fast.

Example:
```json
{
  "probability": 0.62,
  "probability_raw": 0.66,
  "probability_calibrated": 0.62,
  "confidence_level": "moderate",
  "confidence_reason": "Medium volatility regime reduces certainty; confidence capped at moderate."
}
```

Rationale: this prevents the API from presenting raw, overconfident outputs as truth.

## Calibration Architecture (High Level)
- Method: Platt scaling (logistic regression) on a held-out validation set.
- Artifact location: the calibrator is persisted in the model bundle alongside the model and feature names.
- Inference: raw probability is computed first, then calibrated. The calibrated value is what `probability` returns.
- Missing artifact: inference raises an explicit error that explains what is missing and how to fix it.

## Confidence Discipline
- Confidence levels are derived from the calibrated probability.
- Volatility regimes cap confidence so it never increases when volatility rises.
- This means confidence can drop even when the calibrated probability is high.

Design intent: high volatility implies more uncertainty, even if the model is confident.

## Failure Modes & Guarantees
Guarantees:
- Calibrated probabilities are returned and bounded to [0, 1].
- Confidence is monotonic with respect to volatility (never higher in higher volatility).
- Calibration artifacts are required; no silent fallbacks.

Refuses to do:
- Alter model features or retrain the model.
- Pretend raw probabilities are calibrated.
- Increase confidence during volatile regimes.

Common misconception:
- Calibration does not improve accuracy; it improves probability honesty.

## Phase Lock Marker
Phase 9 is locked. Changes require deliberate re-opening of calibration logic.
