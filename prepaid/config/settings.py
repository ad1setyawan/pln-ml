"""
settings.py
Central configuration for anomaly detection ML pipeline - Prepaid
"""

# Input feature specifications
FEATURE_SPEC = {
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

    # Additional features
    'consecutive_anomaly_count': {
        'type': 'integer',
        'required': True,
        'description': 'Number of consecutive anomalies observed'
    },
    'pemakaian': {
        'type': 'numeric',
        'required': True,
        'description': 'Actual electricity consumption'
    },
    'avg_pemakaian_gardu': {
        'type': 'numeric',
        'required': True,
        'description': 'Average consumption at the electrical substation'
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
        'description': 'Binary classification for regular customer behavior pattern (0=not regular, 1=regular)',
    },

    # Stage 2: Final score rumus 1
    # Classes depend on is_rutin:
    # - if is_rutin=1: [0, 20, 40, 60]
    # - if is_rutin=0: [0, 20, 30, 40]
    'final_score_rumus1': {
        'task': 'ordinal_classification',
        'features': ['is_rutin', 'consecutive_anomaly_count'],  # is_rutin as conditional feature
        'model_family': 'tree',
        'classes': [0, 20, 30, 40, 60],  # All possible classes (union of both conditions)
        'description': 'Ordinal score based on consecutive anomaly count. Classes differ by is_rutin value.',
    },

    # Stage 3: Final score rumus 2
    # Classes depend on is_rutin:
    # - if is_rutin=1: [0, 10, 20, 30, 40]
    # - if is_rutin=0: [0, 20, 40, 60]
    'final_score_rumus2': {
        'task': 'ordinal_classification',
        'features': ['is_rutin', 'pemakaian', 'avg_pemakaian_gardu'],  # is_rutin as conditional feature
        'model_family': 'tree',
        'classes': [0, 10, 20, 30, 40, 60],  # All possible classes (union of both conditions)
        'description': 'Ordinal score based on consumption vs gardu average. Classes differ by is_rutin value.',
    },

    # Stage 4: Anomaly score
    # Calculated as: final_score_rumus1 + final_score_rumus2
    'anomaly_score': {
        'task': 'regression',
        'features': ['final_score_rumus1', 'final_score_rumus2'],
        'model_family': 'tree',
        'postprocess': ['clip_0_100'],
        'description': 'Combined anomaly score (sum of final_score_rumus1 and final_score_rumus2)',
    },
}

# Pipeline execution order (models must execute in this order due to dependencies)
PIPELINE_ORDER = [
    'is_rutin',
    'final_score_rumus1',
    'final_score_rumus2',
    'anomaly_score',
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
