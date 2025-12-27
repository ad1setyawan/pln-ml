"""
Training utilities for prepaid models
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


def filter_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter dataset based on data sufficiency checks.

    Args:
        df: Input dataframe with prepaid features

    Returns:
        Tuple of (sufficient_df, insufficient_df)
    """
    results = []

    for _, row in df.iterrows():
        is_sufficient, reason = check_data_sufficiency(
            freq_tx_12m=row.get('freq_tx_12m'),
            avg_gap_days_12m=row.get('avg_gap_days_12m'),
            min_12m_pemakaian=row.get('min_12m_pemakaian'),
            max_12m_pemakaian=row.get('max_12m_pemakaian'),
            avg_12m_pemakaian=row.get('avg_12m_pemakaian'),
            std_12m_pemakaian=row.get('std_12m_pemakaian')
        )

        results.append({
            'is_sufficient': is_sufficient,
            'filter_reason': reason
        })

    # Add filter results to dataframe
    filter_df = pd.DataFrame(results)
    df_with_filter = df.copy()
    df_with_filter['is_sufficient'] = filter_df['is_sufficient']
    df_with_filter['filter_reason'] = filter_df['filter_reason']

    # Split into sufficient and insufficient
    sufficient_df = df_with_filter[df_with_filter['is_sufficient']].copy()
    insufficient_df = df_with_filter[~df_with_filter['is_sufficient']].copy()

    # Drop the filter columns from sufficient data
    sufficient_df = sufficient_df.drop(columns=['is_sufficient', 'filter_reason'])

    return sufficient_df, insufficient_df


def check_data_sufficiency(
    freq_tx_12m: int,
    avg_gap_days_12m: float,
    min_12m_pemakaian: float = None,
    max_12m_pemakaian: float = None,
    avg_12m_pemakaian: float = None,
    std_12m_pemakaian: float = None
) -> Tuple[bool, str]:
    """
    Check if customer has sufficient data for model training/inference.

    Args:
        freq_tx_12m: Number of transactions in 12 months
        avg_gap_days_12m: Average gap between transactions in days
        min_12m_pemakaian: Minimum consumption in 12 months
        max_12m_pemakaian: Maximum consumption in 12 months
        avg_12m_pemakaian: Average consumption in 12 months
        std_12m_pemakaian: Std deviation of consumption in 12 months

    Returns:
        Tuple of (is_sufficient: bool, reason: str)
    """
    # Check if any required features are missing
    required_features = {
        'freq_tx_12m': freq_tx_12m,
        'avg_gap_days_12m': avg_gap_days_12m
    }

    for feat_name, feat_value in required_features.items():
        if pd.isna(feat_value):
            return False, f'missing_features_{feat_name}'

    # Check minimum transaction count
    if freq_tx_12m < 3:
        return False, 'tx_count_lt_3'

    # If avg gap is very high, transaction frequency might be too low
    if avg_gap_days_12m > 120:
        return False, 'freq_too_low'

    # Check if consumption has variation
    if all(v is not None for v in [min_12m_pemakaian, max_12m_pemakaian, avg_12m_pemakaian]):
        if min_12m_pemakaian == max_12m_pemakaian == avg_12m_pemakaian:
            return False, 'no_consumption_var'

        if std_12m_pemakaian is not None and std_12m_pemakaian == 0:
            return False, 'no_consumption_var'

    return True, 'eligible'


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

    Args:
        features: List of feature names to validate
        target: Target model name
        pipeline_order: List of model names in execution order
        feature_spec: Dictionary of raw input features specification

    Raises:
        SystemExit: If validation fails
    """
    target_position = pipeline_order.index(target)

    # Get allowed features (models that appear before target in pipeline)
    allowed_model_outputs = set(pipeline_order[:target_position])

    # Check each feature
    invalid_features = []
    for feature in features:
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

    Args:
        df: Training DataFrame
        target: Target column name
        classes: List of class names defined in config

    Returns:
        Tuple of (active_classes, label_mapping, y_encoded)
    """
    # Get unique classes actually present in training data
    unique_classes_in_data = sorted(df[target].unique())

    # Filter to only use classes that are actually present in data
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

    Args:
        model: Trained classification model
        X: Feature DataFrame

    Returns:
        Array of class labels (not probabilities)
    """
    predictions_proba = model.predict(X)

    # Check if predict returns probabilities (2D array with multiple columns)
    if len(predictions_proba.shape) > 1 and predictions_proba.shape[1] > 1:
        predictions = predictions_proba.argmax(axis=1)
    else:
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

    if 'n_filtered' in summary:
        print(f"  - Filtered out: {summary['n_filtered']} samples ({summary.get('filter_stats', {})})")

    print(f"\nPerformance metrics:")
    for metric_name, value in summary['metrics'].items():
        print(f"  - {metric_name}: {value}")

    print(f"\nFeature importance:")
    for feat, imp in summary['feature_importance'].items():
        print(f"  - {feat}: {imp}")

    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
