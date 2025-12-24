"""
settings.py
Central configuration for anomaly detection ML pipeline
"""

# =========================================================
# FEATURE SPECIFICATION
# =========================================================

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
}

# =========================================================
# MODEL / TARGET CONFIGURATION
# 1 target = 1 model
# =========================================================

MODEL_CONFIGS = {

    'anomaly_score': {
        'task': 'regression',
        'features': ['pemakaian', 'baseline'],
        'model_family': 'tree',
        'postprocess': ['clip_0_1'],
        'description': 'Deviation score between actual usage and baseline',
    },

    'is_anomaly': {
        'task': 'binary_classification',
        'features': ['anomaly_score', 'threshold_drop_consume'],
        'threshold': 0.5,
        'description': 'Binary anomaly decision',
    },

    'final_score_rumus1': {
        'task': 'regression',
        'features': ['anomaly_score', 'consecutive_anomaly_count'],
        'description': 'Severity score based on anomaly persistence',
    },

    'score_rumus2': {
        'task': 'regression',
        'features': ['avg_pemakaian_gardu', 'pemakaian'],
        'description': 'Contextual score compared to gardu average',
    },

    'final_score_rumus2': {
        'task': 'regression',
        'features': ['score_rumus2'],
        'description': 'Normalized contextual score',
    },

    'anomaly_type': {
        'task': 'multiclass_classification',
        'features': ['final_score_rumus1', 'final_score_rumus2'],
        'classes': ['baseline', 'gardu', 'both', 'none'],
        'description': 'Root cause classification',
    },

    'severity_level': {
        'task': 'ordinal_classification',
        'features': ['final_score_rumus1', 'final_score_rumus2'],
        'classes': ['low', 'medium', 'high', 'critical'],
        'description': 'Final severity categorization',
    },
}

# =========================================================
# PIPELINE EXECUTION ORDER
# (dependency-safe)
# =========================================================

PIPELINE_ORDER = [
    'anomaly_score',
    'is_anomaly',
    'final_score_rumus1',
    'score_rumus2',
    'final_score_rumus2',
    'anomaly_type',
    'severity_level',
]

# =========================================================
# GLOBAL TRAINING SETTINGS
# =========================================================

TRAINING_SETTINGS = {
    'train_test_split': 0.8,
    'random_state': 42,
    'cross_validation_folds': 5,
}

# =========================================================
# GLOBAL INFERENCE SETTINGS
# =========================================================

INFERENCE_SETTINGS = {
    'fail_on_missing_feature': True,
    'default_output_on_error': None,
}

# =========================================================
# MONITORING & EVALUATION
# =========================================================

MONITORING_SETTINGS = {
    'enable_drift_detection': True,
    'default_metrics': {
        'regression': ['mae', 'rmse'],
        'binary_classification': ['precision', 'recall', 'f1'],
        'multiclass_classification': ['accuracy', 'f1_macro'],
        'ordinal_classification': ['accuracy'],
    },
}
