"""
Training utilities for postpaid models
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Tuple


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


def encode_classes(
    df: pd.DataFrame,
    target: str,
    classes: List[str]
) -> Tuple[List[str], Dict[str, int], pd.Series]:
    """
    Encode target classes to consecutive integers for multiclass classification.

    Handles cases where training data doesn't contain all classes defined in config.
    Creates a warning if there's a mismatch between config and actual data.

    Args:
        df: Training DataFrame
        target: Target column name
        classes: List of class names defined in config

    Returns:
        Tuple of (active_classes, label_mapping, y_encoded):
        - active_classes: List of classes actually present in data
        - label_mapping: Dictionary mapping class names to integer labels
        - y_encoded: Encoded target as pandas Series with integer labels
    """
    # Get unique classes actually present in training data
    unique_classes_in_data = sorted(df[target].unique())

    # Filter to only use classes that are actually present in data
    # and create consecutive integer labels
    active_classes = [c for c in classes if c in unique_classes_in_data]
    label_mapping = {label: idx for idx, label in enumerate(active_classes)}
    y_encoded = df[target].map(label_mapping).astype(int)

    print(f"Classes defined in config: {classes}")
    print(f"Classes actually in data: {unique_classes_in_data}")
    print(f"Active classes for training: {active_classes}")
    print(f"Label mapping: {label_mapping}")

    # Error if config classes don't match data classes
    if set(classes) != set(unique_classes_in_data):
        print("\n" + "="*60)
        print("ERROR: Class mismatch detected!")
        print("="*60)
        print(f"Defined classes in config: {classes}")
        print(f"Classes found in training data: {unique_classes_in_data}")

        missing_in_data = set(classes) - set(unique_classes_in_data)
        extra_in_data = set(unique_classes_in_data) - set(classes)

        if missing_in_data:
            print(f"Classes in config but NOT in data: {missing_in_data}")
            print("Action: Add training data with these classes, OR update config to remove them")
        if extra_in_data:
            print(f"Classes in data but NOT in config: {extra_in_data}")
            print("Action: Add these classes to config, OR remove them from training data")

        print("\nTraining STOPPED due to class mismatch!")
        print("Reason: Model trained with incomplete classes will FAIL at inference time")
        print("       when encountering data with missing classes.")
        print("="*60)
        sys.exit(1)

    return active_classes, label_mapping, y_encoded


def get_class_predictions(model, X: pd.DataFrame) -> np.ndarray:
    """
    Get class predictions from a classification model.

    Handles cases where model.predict() returns probabilities instead of class labels.

    Args:
        model: Trained classification model
        X: Feature DataFrame

    Returns:
        Array of class labels (not probabilities)
    """
    predictions_proba = model.predict(X)

    # Check if predict returns probabilities (2D array with multiple columns)
    if len(predictions_proba.shape) > 1 and predictions_proba.shape[1] > 1:
        # If predict returns probabilities, take argmax to get class labels
        predictions = predictions_proba.argmax(axis=1)
    else:
        # If predict already returns class labels
        predictions = predictions_proba

    return predictions


def print_training_summary(summary: dict, total_time: float) -> None:
    """
    Print training summary in a formatted way.

    Args:
        summary: Dictionary containing training summary
        total_time: Total execution time in seconds
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"\nModel: {summary['model_name']}")
    print(f"Training time: {summary['training_time_seconds']}s ({summary['training_time_seconds']/60:.2f} min)")
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Model path: {summary['model_path']}")
    print(f"Model size: {summary['model_size_kb']} KB")

    # Print class info for classification models
    if 'classes' in summary:
        print(f"Classes: {summary['classes']}")
        if 'label_mapping' in summary:
            print(f"Label mapping: {summary['label_mapping']}")

    print(f"\nDataset info:")
    print(f"  - Number of samples: {summary['n_samples']}")
    print(f"  - Number of features: {summary['n_features']}")
    print(f"  - Features: {summary['feature_names']}")

    print(f"\nPerformance metrics:")
    for metric_name, value in summary['metrics'].items():
        print(f"  - {metric_name}: {value}")

    print(f"\nFeature importance:")
    for feat, imp in summary['feature_importance'].items():
        print(f"  - {feat}: {imp}")

    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
