"""
settings.py
Central configuration for anomaly detection ML pipeline - Prepaid
"""

# Input feature specifications
FEATURE_SPEC = {
    # Original table columns
    'pemakaian': {
        'type': 'numeric',
        'required': True,
        'description': 'Actual electricity consumption'
    },
    'baseline': {
        'type': 'numeric',
        'required': True,
        'description': 'Expected/normal consumption level'
    },
    'avg_pemakaian_gardu': {
        'type': 'numeric',
        'required': True,
        'description': 'Average consumption at the electrical substation'
    },
    'threshold_drop_consume': {
        'type': 'numeric',
        'required': True,
        'description': 'Anomaly detection threshold'
    },

    # 12-month historical features
    'avg_12m_pemakaian': {
        'type': 'numeric',
        'required': True,
        'description': 'Average consumption over 12 months'
    },
    'std_12m_pemakaian': {
        'type': 'numeric',
        'required': True,
        'description': 'Standard deviation of consumption over 12 months'
    },
    'freq_tx_12m': {
        'type': 'numeric',
        'required': True,
        'description': 'Frequency of transactions in 12 months'
    },
    'avg_gap_days_12m': {
        'type': 'numeric',
        'required': True,
        'description': 'Average gap between transactions in days'
    },
}

# Model configurations for each target variable
MODEL_CONFIGS = {
    # Stage 1: Regularity detection
    'is_rutin': {
        'task': 'binary_classification',
        'features': [
            'avg_12m_pemakaian',
            'std_12m_pemakaian',
            'freq_tx_12m',
            'avg_gap_days_12m'
        ],
        'model_family': 'tree',
        'threshold': 0.5,
        'description': 'Binary classification for regular customer behavior pattern',
    },

    # Stage 2: Anomaly scoring
    'anomaly_score': {
        'task': 'regression',
        'features': ['is_rutin', 'pemakaian', 'baseline'],
        'model_family': 'tree',
        'postprocess': ['clip_0_100'],
        'description': 'Deviation score between actual usage and baseline (0-100)',
    },

    # Stage 3: Anomaly detection
    'is_anomaly': {
        'task': 'binary_classification',
        'features': ['anomaly_score', 'threshold_drop_consume'],
        'threshold': 0.5,
        'description': 'Binary anomaly decision based on anomaly_score and threshold',
    },

    # Stage 4a: Final score rumus 1
    'final_score_rumus1': {
        'task': 'regression',
        'features': ['is_rutin', 'anomaly_score'],
        'model_family': 'tree',
        'postprocess': ['clip_0_100'],
        'description': 'Severity score based on regularity and anomaly score',
    },

    # Stage 4b: Score rumus 2
    'score_rumus2': {
        'task': 'regression',
        'features': ['is_rutin', 'avg_pemakaian_gardu', 'pemakaian'],
        'model_family': 'tree',
        'description': 'Contextual score compared to gardu average',
    },

    # Stage 5: Final score rumus 2
    'final_score_rumus2': {
        'task': 'regression',
        'features': ['is_rutin', 'score_rumus2'],
        'model_family': 'tree',
        'postprocess': ['clip_0_100'],
        'description': 'Final severity score based on regularity and gardu comparison',
    },

    # Stage 6: Anomaly type classification
    'anomaly_type': {
        'task': 'multiclass_classification',
        'features': ['final_score_rumus1', 'final_score_rumus2'],
        'classes': ['baseline', 'gardu', 'both', 'none'],
        'description': 'Root cause classification: baseline deviation, gardu deviation, both, or none',
    },

    # Stage 7: Severity level
    'severity_level': {
        'task': 'ordinal_classification',
        'features': ['final_score_rumus1', 'final_score_rumus2'],
        'classes': ['low', 'medium', 'high', 'critical'],
        'description': 'Final severity categorization in ordinal order',
    },
}

# Pipeline execution order (models must execute in this order due to dependencies)
PIPELINE_ORDER = [
    'is_rutin',
    'anomaly_score',
    'is_anomaly',
    'final_score_rumus1',
    'score_rumus2',
    'final_score_rumus2',
    'anomaly_type',
    'severity_level',
]

# Training settings
TRAINING_SETTINGS = {
    'train_test_split': 0.8,
    'random_state': 42,
    'cross_validation_folds': 5,
}

# Inference settings
INFERENCE_SETTINGS = {
    'batch_size': 1000,
    'output_format': 'csv',
}

# Monitoring settings
MONITORING_SETTINGS = {
    'drift_threshold': 0.1,
    'metrics': {
        'regression': ['rmse', 'mae', 'r2'],
        'binary_classification': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        'multiclass_classification': ['accuracy', 'f1_macro', 'f1_weighted'],
        'ordinal_classification': ['accuracy', 'mae', 'quadratic_weighted_kappa'],
    }
}

# Model versioning
MODEL_VERSION = "v1"
DEFAULT_TRAIN_DATA = "train.csv"
