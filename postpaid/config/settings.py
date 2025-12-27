"""
settings.py
Central configuration for anomaly detection ML pipeline
"""
# Only information
FEATURE_SPEC = {
    'pemakaian': {
        'type': 'numeric',
        'required': True,
    },
    'baseline': {
        'type': 'numeric',
        'required': True,
    },
    'avg_pemakaian_gardu': {
        'type': 'numeric',
        'required': True,
    },
    'threshold_drop_consume': {
        'type': 'numeric',
        'required': True,
    },
    'consecutive_anomaly_count': {
        'type': 'integer',
        'required': True,
    },
    # Derived features for anomaly_score (computed from pemakaian & baseline)
    'ratio': {
        'type': 'numeric',
        'required': False,  # Computed from pemakaian / (baseline + 1)
        'derived_from': ['pemakaian', 'baseline']
    },
    'diff': {
        'type': 'numeric',
        'required': False,  # Computed from baseline - pemakaian
        'derived_from': ['pemakaian', 'baseline']
    },
    'is_over_baseline': {
        'type': 'integer',
        'required': False,  # Computed from pemakaian > baseline
        'derived_from': ['pemakaian', 'baseline']
    },
    # Derived features for score_rumus2 (computed from pemakaian & avg_pemakaian_gardu)
    'ratio_gardu': {
        'type': 'numeric',
        'required': False,  # Computed from pemakaian / (avg_pemakaian_gardu + 1)
        'derived_from': ['pemakaian', 'avg_pemakaian_gardu']
    },
    'pct_diff_gardu': {
        'type': 'numeric',
        'required': False,  # Computed from (pemakaian - avg_pemakaian_gardu) / (avg_pemakaian_gardu + 1)
        'derived_from': ['pemakaian', 'avg_pemakaian_gardu']
    },
}

MODEL_CONFIGS = {

    'anomaly_score': {
        'task': 'regression',
        'features': ['pemakaian', 'baseline', 'ratio', 'diff', 'is_over_baseline'],
        'model_family': 'tree',
        'postprocess': ['clip_0_100'],
        'description': 'Deviation score between actual usage and baseline (0-100)',
    },

    'is_anomaly': {
        'task': 'binary_classification',
        'features': ['anomaly_score'],
        'threshold': 0.5,
        'description': 'Binary anomaly decision',
    },

    'final_score_rumus1': {
        'task': 'ordinal_classification',
        'features': ['consecutive_anomaly_count'],
        'classes': [0, 20, 40, 60],
        'model_family': 'tree',
        'description': 'Severity score based on anomaly persistence',
    },

    'score_rumus2': {
        'task': 'regression',
        'features': ['ratio_gardu', 'pct_diff_gardu'],
        'model_family': 'tree',
        'description': 'Contextual score compared to gardu average',
    },

    'final_score_rumus2': {
        'task': 'ordinal_classification',
        'features': ['score_rumus2'],
        'classes': [0, 10, 20, 30, 40],
        'model_family': 'tree',
        'description': 'Severity score based on gardu deviation percentage (grouped by rule)',
    },
}

PIPELINE_ORDER = [
    'anomaly_score',
    'is_anomaly',
    'final_score_rumus1',
    'score_rumus2',
    'final_score_rumus2',
]

TRAINING_SETTINGS = {
    'train_test_split': 0.8,
    'random_state': 42,
    'cross_validation_folds': 5,
}

MODEL_VERSION = "v1"
DEFAULT_TRAIN_DATA = "train.csv"