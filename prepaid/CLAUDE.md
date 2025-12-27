# CLAUDE.md - Prepaid

This file provides guidance for working with the Prepaid anomaly detection system.

## Project Overview

PLN ML v2 Prepaid is a machine learning system for detecting anomalies in prepaid electricity consumption. The system uses a sequential multi-model pipeline to identify abnormal usage patterns, classify root causes, and assign severity levels.

## Architecture

### Sequential Pipeline Design

The core architecture is an **8-stage sequential pipeline** where each model's output becomes input for subsequent models. Pipeline execution order is defined in `PIPELINE_ORDER` in `/prepaid/config/settings.py:137-145`.

**Pipeline Flow:**
```
Input Features (including 12-month historical data)
    ↓
1. is_rutin (binary classification)
    ↓
2. anomaly_score (regression)
    ↓
3. is_anomaly (binary classification)
    ↓
4. final_score_rumus1 (regression)
    ↓
5. score_rumus2 (regression)
    ↓
6. final_score_rumus2 (regression)
    ↓
7. anomaly_type (multiclass classification)
    ↓
8. severity_level (ordinal classification)
    ↓
Final Output
```

### Model Configuration

All models are configured in `MODEL_CONFIGS` dictionary in `/prepaid/config/settings.py:43-133`. Each model specification includes:

- **task**: Model type (regression, binary_classification, multiclass_classification, ordinal_classification)
- **features**: Input features (can be raw input features or outputs from previous models)
- **model_family**: Algorithm family (e.g., 'tree' for XGBoost)
- **postprocess**: Optional post-processing steps (e.g., 'clip_0_100')
- **classes**: Class labels for classification tasks
- **threshold**: Decision threshold for binary classification

### Feature Specifications

Input features are defined in `FEATURE_SPEC` at `/prepaid/config/settings.py:6-40`:

**Original Table Columns:**
- **pemakaian**: Actual electricity consumption (numeric, required)
- **baseline**: Expected/normal consumption level (numeric, required)
- **avg_pemakaian_gardu**: Average consumption at the electrical substation (numeric, required)
- **threshold_drop_consume**: Anomaly detection threshold (numeric, required)

**12-Month Historical Features:**
- **avg_12m_pemakaian**: Average consumption over 12 months (numeric, required)
- **std_12m_pemakaian**: Standard deviation of consumption over 12 months (numeric, required)
- **min_12m_pemakaian**: Minimum consumption over 12 months (numeric, required)
- **max_12m_pemakaian**: Maximum consumption over 12 months (numeric, required)
- **freq_tx_12m**: Frequency of transactions in 12 months (numeric, required)
- **avg_gap_days_12m**: Average gap between transactions in days (numeric, required)

## Development Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Data Format
Training data uses **semicolon-separated CSV format** (not comma-separated):
```bash
# View training data
head -20 prepaid/data/train.csv

# Load in pandas with correct separator
python3 -c "import pandas as pd; df = pd.read_csv('prepaid/data/train.csv', sep=';'); print(df.head())"
```

### Configuration Management
- **Training settings**: Modify `TRAINING_SETTINGS` in `/prepaid/config/settings.py:148-152`
- **Inference settings**: Modify `INFERENCE_SETTINGS` at `/prepaid/config/settings.py:154-157`
- **Monitoring**: Configure metrics and drift detection in `MONITORING_SETTINGS` at `/prepaid/config/settings.py:159-169`

## Domain Terminology

- **pemakaian**: Electricity consumption/usage (actual reading)
- **gardu**: Electrical substation or distribution point
- **baseline**: Normal or expected consumption level for a customer
- **rutin**: Regular/patterned behavior (Indonesian)
- **anomaly**: Deviation from baseline consumption pattern
- **rumus**: Formula/rule (Indonesian) - refers to different scoring methodologies

## Key Implementation Notes

1. **Pipeline Dependency Order**: When modifying `MODEL_CONFIGS`, ensure dependencies are respected. A model can only use features from:
   - Raw input features (defined in `FEATURE_SPEC`)
   - Outputs of models that appear earlier in `PIPELINE_ORDER`

2. **Regular Behavior Detection**: The prepaid pipeline starts with `is_rutin` classification, which identifies customers with regular patterned behavior vs irregular behavior. This is a key difference from postpaid.

3. **Historical Features**: Prepaid models rely heavily on 12-month historical features (avg_12m_pemakaian, std_12m_pemakaian, etc.) for regularity detection.

4. **Multi-Task Learning**: Each target variable has its own model. This is intentional - the system uses specialized models rather than a single multi-output model.

5. **Data Separator**: CSV files use semicolon (`;`) as separator, not comma. Always specify `sep=';'` when using pandas.

6. **Model Storage**: Trained models should be saved to `/prepaid/models/` directory.

## Project Structure

```
pln-ml-v2/
├── prepaid/
│   ├── config/
│   │   └── settings.py          # Core configuration - modify pipeline/models here
│   ├── data/
│   │   └── train.csv            # Training data (semicolon-separated, to be added)
│   ├── models/                  # Model artifacts (to be populated)
│   ├── training/                # Training module implementation (to be added)
│   └── inferences/              # Inference module implementation (to be added)
├── postpaid/                    # Postpaid system (already implemented)
├── requirements.txt             # Python dependencies
└── CLAUDE.md                    # Main project documentation
```

## Current Development Status

Prepaid system is in initial setup phase:
- ✅ Complete configuration framework
- ✅ Pipeline architecture design
- ✅ Feature and model specifications
- ❌ Training implementation (not yet implemented)
- ❌ Inference implementation (not yet implemented)
- ❌ Model persistence layer (not yet implemented)
- ❌ Unit tests (not yet implemented)
- ❌ Training data (not yet added)
