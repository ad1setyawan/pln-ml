"""
Training script for final_score_rumus1 model
Usage: python postpaid/training/train_final_score_rumus1.py
"""

import os
import sys
from pathlib import Path
import time

import pandas as pd
import numpy as np
import xgboost as xgb

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_CONFIGS, TRAINING_SETTINGS, DEFAULT_TRAIN_DATA, MODEL_VERSION, PIPELINE_ORDER, FEATURE_SPEC
from training.utils import load_training_data, save_model, validate_features_and_target, validate_pipeline_order, encode_classes, get_class_predictions, print_training_summary

MODEL_NAME = 'final_score_rumus1'

def train_final_score_rumus1_model(df: pd.DataFrame, model_dir: str) -> dict:
    """
    Train the final_score_rumus1 model and save it.

    Args:
        df: Training DataFrame
        model_dir: Directory to save the trained model

    Returns:
        Dictionary containing training summary
    """
    # Get model configuration
    model_config = MODEL_CONFIGS[MODEL_NAME]
    features = model_config['features']
    target = MODEL_NAME
    classes = model_config['classes']

    print(f"Model: {MODEL_NAME}")
    print(f"Features: {features}")
    print(f"Target: {target}")
    print(f"Classes: {classes}")

    # Validate pipeline order
    print("Validating pipeline order...")
    validate_pipeline_order(features, target, PIPELINE_ORDER, FEATURE_SPEC)

    # Validate features and target exist in data
    print("Validating features and target in training data...")
    validate_features_and_target(df, features, target)
    print("Validation successful\n")

    # Extract features
    X = df[features]

    # Encode target classes to consecutive integers
    active_classes, label_mapping, y = encode_classes(df, target, classes)

    # Initialize XGBoost classifier
    random_state = TRAINING_SETTINGS.get('random_state', 42)
    hyperparameters = {
        'random_state': random_state,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'objective': 'multi:softprob',
        'num_class': len(active_classes)
    }
    model = xgb.XGBClassifier(**hyperparameters)

    # Train the model
    print("Starting model training...")
    training_start = time.time()

    model.fit(X, y)

    training_time = time.time() - training_start
    print(f"Training completed in {training_time:.2f}s")

    # Save the model
    print("Saving the trained model...")
    model_path, file_size_kb = save_model(model, model_dir, MODEL_NAME)
    print("Model saved")

    # Feature importance
    importance = model.feature_importances_

    # Model predictions on training data for metrics
    predictions = get_class_predictions(model, X)

    # Calculate metrics
    accuracy = np.mean(y == predictions)

    # Prepare summary dictionary
    summary = {
        'model_name': MODEL_NAME,
        'training_time_seconds': round(training_time, 2),
        'model_path': model_path,
        'model_size_kb': file_size_kb,
        'n_samples': len(df),
        'n_features': len(features),
        'feature_names': features,
        'classes': active_classes,
        'label_mapping': label_mapping,
        'metrics': {
            'accuracy': round(accuracy, 4)
        },
        'feature_importance': dict(zip(features, [round(imp, 4) for imp in importance])),
        'hyperparameters': hyperparameters
    }

    return summary


def main():
    """Main training function."""
    start_time = time.time()

    print("="*60)
    print("START MODEL TRAINING")
    print("="*60)

    # Paths
    current_dir = Path(__file__).parent.parent
    data_path = os.path.join(current_dir, 'data', DEFAULT_TRAIN_DATA)
    model_dir = os.path.join(current_dir, 'models', MODEL_VERSION)

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    # Load data
    print(f"Loading training data from {data_path}...")
    df = load_training_data(data_path)
    print(f"Loaded {len(df)} records\n")

    # Train model
    summary = train_final_score_rumus1_model(df, model_dir)

    # Final summary
    total_time = time.time() - start_time
    print_training_summary(summary, total_time)


if __name__ == "__main__":
    main()
