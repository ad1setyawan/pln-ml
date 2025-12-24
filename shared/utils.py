"""
Shared utilities for the project
"""

import os
import sys
import pandas as pd
import joblib
from typing import List, Dict


def load_training_data(data_path: str) -> pd.DataFrame:
    """
    Load training data from CSV file.

    Args:
        data_path: Path to training CSV file (semicolon-separated)

    Returns:
        DataFrame with training data
    """
    df = pd.read_csv(data_path, sep=';')
    return df


def validate_features_and_target(df: pd.DataFrame, features: list, target: str) -> None:
    """
    Validate that all features and target exist in the dataframe.

    Args:
        df: Training DataFrame
        features: List of feature names to validate
        target: Target column name to validate

    Raises:
        SystemExit: If validation fails
    """
    # Validate features exist in data
    missing_features = [feat for feat in features if feat not in df.columns]
    if missing_features:
        print(f"Error: Missing features in data: {missing_features}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Validate target exists in data
    if target not in df.columns:
        print(f"Error: Target '{target}' not found in data")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)


def validate_pipeline_order(
    features: List[str],
    target: str,
    pipeline_order: List[str],
    feature_spec: Dict[str, dict]
) -> None:
    """
    Validate that features don't use outputs from models that appear later in pipeline.

    A model can only use:
    1. Raw input features (defined in FEATURE_SPEC)
    2. Outputs from models that appear BEFORE it in PIPELINE_ORDER

    Args:
        features: List of feature names to validate
        target: Target model name
        pipeline_order: List of model names in execution order
        feature_spec: Dictionary of raw input features specification

    Raises:
        SystemExit: If validation fails

    Example:
        If PIPELINE_ORDER = ['anomaly_score', 'is_anomaly', 'anomaly_type']
        - anomaly_score (pos 0) can only use raw features
        - is_anomaly (pos 1) can use raw features + anomaly_score
        - anomaly_type (pos 2) can use raw features + anomaly_score + is_anomaly
    """
    target_position = pipeline_order.index(target)

    # Get allowed features (models that appear before target in pipeline)
    allowed_model_outputs = set(pipeline_order[:target_position])

    # Check each feature
    invalid_features = []
    for feature in features:
        # Feature is valid if it's a raw input feature OR
        # if it's an output from a model that appears before target
        if feature not in feature_spec and feature not in allowed_model_outputs:
            invalid_features.append(feature)

    if invalid_features:
        print(f"Error: Target '{target}' cannot use features from models that appear later in pipeline")
        print(f"Invalid features: {invalid_features}")
        print(f"\nAllowed features for '{target}':")
        print(f"  - Raw input features: {list(feature_spec.keys())}")
        print(f"  - Model outputs before position {target_position}: {list(allowed_model_outputs)}")
        print(f"\nFull pipeline order: {pipeline_order}")
        sys.exit(1)


def save_model(model, model_dir: str, model_name: str) -> tuple:
    """
    Save trained model to disk.

    Args:
        model: Trained model object
        model_dir: Directory to save the model
        model_name: Name of the model file (without extension)

    Returns:
        Tuple of (model_path, file_size_kb)
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    joblib.dump(model, model_path)

    # Get file size
    file_size = os.path.getsize(model_path)
    file_size_kb = round(file_size / 1024, 2)

    return model_path, file_size_kb
