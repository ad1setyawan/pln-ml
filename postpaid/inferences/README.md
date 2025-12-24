# Postpaid Model Inference

This directory contains inference scripts for trained postpaid anomaly detection models.

## Prerequisites

Make sure you have trained the models first:

```bash
# Train all models
python postpaid/training/train_anomaly_score.py
python postpaid/training/train_is_anomaly.py
python postpaid/training/train_final_score_rumus1.py
python postpaid/training/train_score_rumus2.py
python postpaid/training/train_final_score_rumus2.py
python postpaid/training/train_anomaly_type.py
python postpaid/training/train_severity_level.py
```

## Available Inference Scripts

### 1. Anomaly Score Prediction

**Script:** `predict_anomaly_score.py`

**Description:** Predict deviation score between actual and baseline electricity consumption (0-100 scale)

**Usage:**
```bash
python postpaid/inferences/predict_anomaly_score.py --pemakaian <value> --baseline <value>
```

**Example:**
```bash
# Consumption lower than baseline (will predict)
python postpaid/inferences/predict_anomaly_score.py --pemakaian 30.0 --baseline 40.0

# Consumption equal or higher than baseline (returns 0, no prediction needed)
python postpaid/inferences/predict_anomaly_score.py --pemakaian 45.0 --baseline 40.0
```

**Business Rule:**
- If `pemakaian >= baseline`: Anomaly score = 0 (no anomaly, consumption is normal or higher)
- If `pemakaian < baseline`: Model predicts anomaly score (consumption is lower than expected)

**Rationale:** Anomaly detection focuses on detecting abnormally low consumption (potential theft/loss), not high consumption.

**Output:**
- Anomaly score (0-100)
- Whether business rule was applied
- Interpretation of the score
- Status classification (NORMAL, SLIGHT DEVIATION, MODERATE ANOMALY, HIGH ANOMALY)

**Features:**
- `pemakaian`: Actual electricity consumption (kWh)
- `baseline`: Expected/baseline electricity consumption (kWh)

## Using Inference Utilities

You can also use the inference utilities programmatically:

```python
from postpaid.inferences.utils import make_prediction

# Make prediction
features = {
    'pemakaian': 41.0,
    'baseline': 40.0
}

result = make_prediction('anomaly_score', features)

print(f"Raw prediction: {result['raw_prediction']}")
print(f"Final prediction: {result['final_prediction']}")
```

## Pipeline Inference

For complete anomaly detection, you need to run models in pipeline order:

1. `anomaly_score` - Calculate deviation from baseline
2. `is_anomaly` - Binary classification (anomaly or not)
3. `final_score_rumus1` - Severity score based on persistence
4. `score_rumus2` - Contextual score compared to gardu average
5. `final_score_rumus2` - Normalized contextual score
6. `anomaly_type` - Root cause classification (baseline, gardu, both, none)
7. `severity_level` - Final severity categorization (low, medium, high, critical)

See [CLAUDE.md](../../CLAUDE.md) for complete pipeline documentation.

## Error Handling

If you see "Model file not found" error:
- Make sure you've trained the model using the training script
- Check that the model file exists in `postpaid/models/v1/`

If you see "Missing required features" error:
- Check the required features for the model in `postpaid/config/settings.py`
- Provide all required features as command-line arguments
