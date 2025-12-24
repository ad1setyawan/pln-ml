"""
Inference utilities for postpaid models
"""

import os
import sys
from pathlib import Path
import joblib
import pandas as pd
from typing import Dict, Any


def load_trained_model(model_name: str, model_version: str = None, model_dir: str = None):
    """
    Load trained model from disk.

    Args:
        model_name: Name of the model (e.g., 'anomaly_score')
        model_version: Version of the model (optional, defaults to config)
        model_dir: Directory containing trained model (optional)

    Returns:
        Loaded model object
    """
    if model_dir is None:
        from config.settings import MODEL_VERSION
        model_version = model_version or MODEL_VERSION
        current_dir = Path(__file__).parent.parent
        model_dir = os.path.join(current_dir, 'models', model_version)

    model_path = os.path.join(model_dir, f'{model_name}.pkl')

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print(f"Please train the model first using: python postpaid/training/train_{model_name}.py")
        sys.exit(1)

    model = joblib.load(model_path)
    return model


def validate_features(features: Dict[str, Any], required_features: list) -> None:
    """
    Validate that all required features are present in input.

    Args:
        features: Dictionary of feature names and values
        required_features: List of required feature names

    Raises:
        SystemExit: If validation fails
    """
    missing_features = [feat for feat in required_features if feat not in features]

    if missing_features:
        print(f"Error: Missing required features: {missing_features}")
        print(f"Required features: {required_features}")
        print(f"Provided features: {list(features.keys())}")
        sys.exit(1)


def prepare_input_dataframe(features: Dict[str, Any], feature_order: list) -> pd.DataFrame:
    """
    Convert input features to DataFrame format expected by model.

    Args:
        features: Dictionary of feature names and values
        feature_order: List of feature names in correct order for model

    Returns:
        DataFrame with features in correct order
    """
    # Create DataFrame with features in correct order
    X = pd.DataFrame([[features[feat] for feat in feature_order]], columns=feature_order)
    return X


def apply_postprocessing(prediction: Any, postprocess_steps: list) -> Any:
    """
    Apply postprocessing steps to model prediction.

    Args:
        prediction: Raw model prediction (can be single value or array)
        postprocess_steps: List of postprocessing step names

    Returns:
        Postprocessed prediction
    """
    result = prediction

    for step in postprocess_steps:
        if step == 'clip_0_100':
            if hasattr(result, '__iter__') and not isinstance(result, str):
                # Handle array-like predictions
                result = [max(0, min(100, val)) for val in result]
            else:
                # Handle single value
                result = max(0, min(100, result))
        elif step == 'clip_0_1':
            if hasattr(result, '__iter__') and not isinstance(result, str):
                result = [max(0, min(1, val)) for val in result]
            else:
                result = max(0, min(1, result))

    return result


def make_prediction(
    model_name: str,
    features: Dict[str, Any],
    model_version: str = None,
    model_dir: str = None
) -> Dict[str, Any]:
    """
    Generic prediction function for any trained model.

    Args:
        model_name: Name of the model to use
        features: Dictionary of feature names and values
        model_version: Version of the model (optional)
        model_dir: Directory containing trained model (optional)

    Returns:
        Dictionary containing prediction results
    """
    from config.settings import MODEL_CONFIGS

    # Get model configuration
    model_config = MODEL_CONFIGS[model_name]

    # Validate input features
    validate_features(features, model_config['features'])

    # Load model
    model = load_trained_model(model_name, model_version, model_dir)

    # Prepare input
    X = prepare_input_dataframe(features, model_config['features'])

    # Make prediction
    raw_prediction = model.predict(X)

    # Handle single vs multiple predictions
    if hasattr(raw_prediction, '__iter__') and not isinstance(raw_prediction, str):
        raw_prediction = raw_prediction[0] if len(raw_prediction) == 1 else raw_prediction

    # Apply postprocessing
    postprocess_steps = model_config.get('postprocess', [])
    final_prediction = apply_postprocessing(raw_prediction, postprocess_steps)

    # Prepare result
    result = {
        'model_name': model_name,
        'features': features,
        'raw_prediction': raw_prediction,
        'final_prediction': final_prediction,
        'postprocessing': postprocess_steps
    }

    return result
