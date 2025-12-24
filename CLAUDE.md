# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PLN ML v2 is a machine learning system for detecting anomalies in postpaid electricity consumption for PLN (Perusahaan Listrik Negara - Indonesian State Electricity Company). The system uses a sequential multi-model pipeline to identify abnormal usage patterns, classify root causes, and assign severity levels.

## Architecture

### Sequential Pipeline Design

The core architecture is a **7-stage sequential pipeline** where each model's output becomes input for subsequent models. Pipeline execution order is defined in `PIPELINE_ORDER` in `/postpaid/config/settings.py:93-101`.

**Pipeline Flow:**
```
Input Features
    ↓
1. anomaly_score (regression)
    ↓
2. is_anomaly (binary classification)
    ↓
3. final_score_rumus1 (regression)
    ↓
4. score_rumus2 (regression)
    ↓
5. final_score_rumus2 (regression)
    ↓
6. anomaly_type (multiclass classification)
    ↓
7. severity_level (ordinal classification)
    ↓
Final Output
```

### Model Configuration

All models are configured in `MODEL_CONFIGS` dictionary in `/postpaid/config/settings.py:38-86`. Each model specification includes:

- **task**: Model type (regression, binary_classification, multiclass_classification, ordinal_classification)
- **features**: Input features (can be raw input features or outputs from previous models)
- **model_family**: Algorithm family (e.g., 'tree' for XGBoost)
- **postprocess**: Optional post-processing steps (e.g., 'clip_0_1')
- **classes**: Class labels for classification tasks
- **threshold**: Decision threshold for binary classification

### Feature Specifications

Input features are defined in `FEATURE_SPEC` at `/postpaid/config/settings.py:10-31`:
- **pemakaian**: Actual electricity consumption (numeric, required)
- **baseline**: Expected/normal consumption level (numeric, required)
- **avg_pemakaian_gardu**: Average consumption at the electrical substation (numeric, required)
- **threshold_drop_consume**: Anomaly detection threshold (numeric, required)
- **consecutive_anomaly_count**: Number of consecutive anomalies observed (integer, required)

## Development Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Data Format
Training data uses **semicolon-separated CSV format** (not comma-separated):
```bash
# View training data
head -20 postpaid/data/train.csv

# Load in pandas with correct separator
python3 -c "import pandas as pd; df = pd.read_csv('postpaid/data/train.csv', sep=';'); print(df.head())"
```

### Configuration Management
- **Training settings**: Modify `TRAINING_SETTINGS` in `/postpaid/config/settings.py:107-111`
- **Inference settings**: Modify `INFERENCE_SETTINGS` in `/postpaid/config/settings.py:117-120`
- **Monitoring**: Configure metrics and drift detection in `MONITORING_SETTINGS` at `/postpaid/config/settings.py:126-134`

## Domain Terminology

- **pemakaian**: Electricity consumption/usage (actual reading)
- **gardu**: Electrical substation or distribution point
- **baseline**: Normal or expected consumption level for a customer
- **anomaly**: Deviation from baseline consumption pattern
- **rumus**: Formula/rule (Indonesian) - refers to different scoring methodologies

## Key Implementation Notes

1. **Pipeline Dependency Order**: When modifying `MODEL_CONFIGS`, ensure dependencies are respected. A model can only use features from:
   - Raw input features (defined in `FEATURE_SPEC`)
   - Outputs of models that appear earlier in `PIPELINE_ORDER`

2. **Multi-Task Learning**: Each target variable has its own model. This is intentional - the system uses specialized models rather than a single multi-output model.

3. **Data Separator**: CSV files use semicolon (`;`) as separator, not comma. Always specify `sep=';'` when using pandas.

4. **Model Storage**: Trained models should be saved to `/postpaid/models/` directory (currently empty in initial commit).

5. **Monitoring Integration**: The system includes built-in drift detection and task-specific metrics. See `MONITORING_SETTINGS` for default metrics per task type.

## Project Structure

```
pln-ml-v2/
├── postpaid/
│   ├── config/
│   │   └── settings.py          # Core configuration - modify pipeline/models here
│   ├── data/
│   │   └── train.csv            # Training data (semicolon-separated)
│   ├── models/                  # Model artifacts (empty - to be populated)
│   ├── training/                # Training module implementation
│   └── inferences/              # Inference module implementation
├── requirements.txt             # Python dependencies
└── CLAUDE.md                    # This file
```

## Current Development Status

This is an initial commit with:
- ✅ Complete configuration framework
- ✅ Pipeline architecture design
- ✅ Feature and model specifications
- ❌ Training implementation (not yet implemented)
- ❌ Inference implementation (not yet implemented)
- ❌ Model persistence layer (not yet implemented)
- ❌ Unit tests (not yet implemented)
