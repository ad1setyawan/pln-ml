# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PLN ML v2 is a machine learning system for detecting anomalies in electricity consumption for PLN (Perusahaan Listrik Negara - Indonesian State Electricity Company). The system has **two separate implementations**:

1. **Postpaid** - Anomaly detection for postpaid electricity customers
2. **Prepaid** - Anomaly detection for prepaid electricity customers

Both systems use a sequential multi-model pipeline to identify abnormal usage patterns through different scoring methodologies (rumus).

---

## Postpaid System

### Architecture

**5-Stage Sequential Pipeline** - each model's output becomes input for subsequent models:

```
Input Features (pemakaian, baseline, avg_pemakaian_gardu, threshold_drop_consume, consecutive_anomaly_count)
    ↓
1. anomaly_score (regression) → 0-100 score
    ↓
2. is_anomaly (binary classification) → 0/1
    ↓
3. final_score_rumus1 (ordinal classification) → [0, 20, 40, 60]
    ↓
4. score_rumus2 (regression) → continuous score
    ↓
5. final_score_rumus2 (ordinal classification) → [0, 10, 20, 30, 40]
    ↓
Final Output: anomaly_detected, persistence_score, gardu_deviation_score
```

**Configuration**: `/postpaid/config/settings.py`
- `MODEL_CONFIGS` - Model specifications (lines 56-95)
- `PIPELINE_ORDER` - Execution order (lines 97-103)
- `FEATURE_SPEC` - Feature definitions (lines 6-54)

### Training Scripts

```bash
# Train all models in order
python postpaid/training/train_anomaly_score.py
python postpaid/training/train_is_anomaly.py
python postpaid/training/train_final_score_rumus1.py
python postpaid/training/train_score_rumus2.py
python postpaid/training/train_final_score_rumus2.py
```

**Data Format**: Semicolon-separated CSV (`;`) - located at `/postpaid/data/train.csv`

**Models Directory**: `/postpaid/models/v1/` - all trained models saved here

### Inference Scripts

**Single Model Inference:**
```bash
# Example: predict anomaly score
python postpaid/inferences/predict_anomaly_score.py \
  --pemakaian 41.0 --baseline 40.0

# Example: predict is_anomaly
python postpaid/inferences/predict_is_anomaly.py \
  --anomaly_score 45.5 --threshold_drop_consume 30.0
```

**Full Pipeline Inference (Recommended):**
```bash
python postpaid/inferences/predict_pipeline.py \
  --pemakaian 41.0 \
  --baseline 40.0 \
  --avg_pemakaian_gardu 60.0 \
  --threshold_drop_consume 30.0 \
  --consecutive_anomaly_count 2
```

Output: anomaly_detected + persistence_score + gardu_deviation_score

---

## Prepaid System

### Architecture

**4-Stage Sequential Pipeline** with customer regularity detection:

```
Input Features (avg_12m_pemakaian, std_12m_pemakaian, freq_tx_12m, avg_gap_days_12m,
              consecutive_anomaly_count, pemakaian, avg_pemakaian_gardu)
    ↓
1. is_rutin (binary classification) → 0/1 (regularity check)
    ↓
2. final_score_rumus1 (ordinal classification) → [0, 20, 30, 40, 60]
    ↓
3. final_score_rumus2 (ordinal classification) → [0, 10, 20, 30, 40, 60]
    ↓
4. anomaly_score (regression) → 0-100 (final combined score)
    ↓
Final Output: anomaly_score (0-100)
```

**Key Difference**: Classes for final_score_rumus1/2 depend on `is_rutin` value:
- If `is_rutin=1` (regular): different class ranges
- If `is_rutin=0` (irregular): different class ranges

**Configuration**: `/prepaid/config/settings.py`
- `MODEL_CONFIGS` - Model specifications (lines 49-97)
- `PIPELINE_ORDER` - Execution order (lines 100-105)
- `FEATURE_SPEC` - Feature definitions (lines 7-46)

### Training Scripts

```bash
# Train all models in order
python prepaid/training/train_is_rutin.py
python prepaid/training/train_final_score_rumus1.py
python prepaid/training/train_final_score_rumus2.py
python prepaid/training/train_anomaly_score.py
```

**Data Filtering**: Prepaid training includes automatic data sufficiency filtering:
- Minimum 3 transactions in 12 months
- Average gap between transactions < 120 days
- Must have consumption variation

**Data Format**: Semicolon-separated CSV (`;`) - located at `/prepaid/data/train.csv`

**Models Directory**: `/prepaid/models/v1/` - all trained models saved here

### Inference Scripts

**Single Model Inference:**
```bash
# Example: predict is_rutin
python prepaid/inferences/predict_is_rutin.py \
  --avg_12m_pemakaian 150.5 --std_12m_pemakaian 25.3 \
  --freq_tx_12m 12 --avg_gap_days_12m 30.5

# Example: predict final_score_rumus1
python prepaid/inferences/predict_final_score_rumus1.py \
  --is_rutin 1 --consecutive_anomaly_count 3
```

**Full Pipeline Inference (Recommended):**
```bash
python prepaid/inferences/predict_pipeline.py \
  --avg_12m_pemakaian 150.5 \
  --std_12m_pemakaian 25.3 \
  --freq_tx_12m 12 \
  --avg_gap_days_12m 30.5 \
  --consecutive_anomaly_count 3 \
  --pemakaian 145.0 \
  --avg_pemakaian_gardu 140.0
```

Output: is_rutin + final_score_rumus1 + final_score_rumus2 + anomaly_score

---

## Common Development Tasks

### Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- pandas, numpy
- scikit-learn
- xgboost
- joblib

### Data Format (Both Systems)

**CRITICAL**: All CSV files use **semicolon (`;`) separator**, NOT comma.

```python
# Correct way to load data
import pandas as pd
df = pd.read_csv('path/to/train.csv', sep=';')

# WRONG - will cause parsing errors
df = pd.read_csv('path/to/train.csv')  # ❌
```

### Configuration Management

Both systems follow the same pattern:
- `config/settings.py` - All model configurations, pipeline order, feature specs
- `training/utils.py` - Shared training utilities
- `inferences/utils.py` - Shared inference utilities

### Model Storage

Models are versioned and stored in:
- `/postpaid/models/{MODEL_VERSION}/` (default: `v1`)
- `/prepaid/models/{MODEL_VERSION}/` (default: `v1`)

Each model saves:
- `{model_name}.pkl` - Trained model file
- Optional: metadata files for label mappings

---

## Domain Terminology

- **pemakaian**: Electricity consumption/usage (actual reading in kWh)
- **gardu**: Electrical substation or distribution point
- **baseline**: Normal or expected consumption level for a customer
- **rutin**: Regular/consistent pattern (prepaid system)
- **anomaly**: Deviation from expected consumption pattern
- **rumus**: Formula/rule (Indonesian) - different scoring methodologies
  - **Rumus 1**: Based on persistence/consecutive anomalies
  - **Rumus 2**: Based on deviation from gardu (substation) average

---

## Key Implementation Notes

### Pipeline Dependency Order
Models can ONLY use features from:
1. Raw input features (defined in `FEATURE_SPEC`)
2. Outputs of models that appear **earlier** in `PIPELINE_ORDER`

Validation is automatic in training scripts.

### Multi-Task Learning
Each target variable has its own model (specialized, not multi-output).

### Data Filtering (Prepaid Only)
Prepaid training automatically filters insufficient data:
- `tx_count_lt_3` - Less than 3 transactions
- `freq_too_low` - Average transaction gap > 120 days
- `no_consumption_var` - No variation in consumption

Filtered rows are dropped with statistics displayed.

### Missing Values
Training scripts automatically DROP rows with missing values - no imputation.

### Ordinal Classification
Models like `final_score_rumus1/2` use ordinal classification:
- Classes have meaningful order (0 < 20 < 30 < 40 < 60)
- XGBoost multi-class classifier used
- Label mapping saved for inference

---

## Project Structure

```
pln-ml-v2/
├── postpaid/                          # Postpaid system
│   ├── config/
│   │   └── settings.py                # Configuration
│   ├── data/
│   │   └── train.csv                  # Training data (semicolon-separated)
│   ├── models/v1/                     # Trained models
│   ├── training/                      # Training scripts
│   │   ├── train_*.py                 # Individual model training
│   │   └── utils.py                   # Shared utilities
│   └── inferences/                    # Inference scripts
│       ├── predict_*.py               # Single model inference
│       ├── predict_pipeline.py        # Full pipeline inference
│       └── utils.py                   # Shared utilities
│
├── prepaid/                           # Prepaid system
│   ├── config/
│   │   └── settings.py                # Configuration
│   ├── data/
│   │   └── train.csv                  # Training data (semicolon-separated)
│   ├── models/v1/                     # Trained models
│   ├── training/                      # Training scripts
│   │   ├── train_*.py                 # Individual model training
│   │   └── utils.py                   # Shared utilities (includes filtering)
│   └── inferences/                    # Inference scripts
│       ├── predict_*.py               # Single model inference
│       ├── predict_pipeline.py        # Full pipeline inference
│       └── utils.py                   # Shared utilities
│
├── requirements.txt                    # Python dependencies
└── CLAUDE.md                           # This file
```

---

## Current Development Status

### Postpaid System ✅
- ✅ Complete configuration framework
- ✅ Training implementation (5 models)
- ✅ Inference implementation (single + pipeline)
- ✅ Model persistence layer
- ❌ Unit tests (not yet implemented)

### Prepaid System ✅
- ✅ Complete configuration framework
- ✅ Training implementation (4 models)
- ✅ Inference implementation (single + pipeline)
- ✅ Data filtering for training
- ✅ Model persistence layer
- ❌ Unit tests (not yet implemented)

---

## Quick Start Examples

### Postpaid - Train & Predict
```bash
# Train models
cd postpaid
python training/train_anomaly_score.py
python training/train_is_anomaly.py
# ... train remaining models

# Run full pipeline inference
python inferences/predict_pipeline.py \
  --pemakaian 41.0 --baseline 40.0 \
  --avg_pemakaian_gardu 60.0 \
  --threshold_drop_consume 30.0 \
  --consecutive_anomaly_count 2
```

### Prepaid - Train & Predict
```bash
# Train models
cd prepaid
python training/train_is_rutin.py
python training/train_final_score_rumus1.py
# ... train remaining models

# Run full pipeline inference
python inferences/predict_pipeline.py \
  --avg_12m_pemakaian 150.5 --std_12m_pemakaian 25.3 \
  --freq_tx_12m 12 --avg_gap_days_12m 30.5 \
  --consecutive_anomaly_count 3 \
  --pemakaian 145.0 --avg_pemakaian_gardu 140.0
```
