"""
Inference utilities for prepaid models
"""

import os
import sys
from pathlib import Path
import joblib
import pandas as pd
from typing import Dict, Any, List


def load_trained_model(model_name: str, model_version: str = None, model_dir: str = None):
    """
    Load trained model from disk.

    Args:
        model_name: Name of the model (e.g., 'is_rutin')
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
        print(f"Please train the model first using: python prepaid/training/train_{model_name}.py")
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


def decode_class_label(predicted_class: int, label_mapping: Dict[str, int]) -> Any:
    """
    Decode predicted class integer back to original label.

    Args:
        predicted_class: Predicted class integer
        label_mapping: Dictionary mapping class names to integer labels

    Returns:
        Original class label
    """
    # Reverse the mapping (integer -> label)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    return reverse_mapping.get(predicted_class, predicted_class)
