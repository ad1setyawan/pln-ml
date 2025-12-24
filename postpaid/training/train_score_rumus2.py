"""
Training script for score_rumus2 model
Usage: python postpaid/training/train_score_rumus2.py
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

# Add root directory to path to import shared utilities
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from shared.utils import load_training_data, save_model, validate_features_and_target, validate_pipeline_order, print_training_summary

MODEL_NAME = 'score_rumus2'

def train_score_rumus2_model(df: pd.DataFrame, model_dir: str) -> dict:
    """
    Train the score_rumus2 model and save it.

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

    print(f"Model: {MODEL_NAME}")
    print(f"Features: {features}")
    print(f"Target: {target}")

    # Validate pipeline order
    print("Validating pipeline order...")
    validate_pipeline_order(features, target, PIPELINE_ORDER, FEATURE_SPEC)

    # Validate features and target exist in data
    print("Validating features and target in training data...")
    validate_features_and_target(df, features, target)
    print("Validation successful\n")

    # Extract features and target
    X = df[features]
    y = df[target]

    # Initialize XGBoost regressor
    random_state = TRAINING_SETTINGS.get('random_state', 42)
    hyperparameters = {
        'random_state': random_state,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'objective': 'reg:squarederror'
    }
    model = xgb.XGBRegressor(**hyperparameters)

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
    predictions = model.predict(X)

    # Calculate metrics
    mae = np.mean(np.abs(y - predictions))
    rmse = np.sqrt(np.mean((y - predictions)**2))
    mape = np.mean(np.abs((y - predictions) / (y + 1e-8))) * 100

    # Prepare summary dictionary
    summary = {
        'model_name': MODEL_NAME,
        'training_time_seconds': round(training_time, 2),
        'model_path': model_path,
        'model_size_kb': file_size_kb,
        'n_samples': len(df),
        'n_features': len(features),
        'feature_names': features,
        'metrics': {
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mape': round(mape, 2)
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
    summary = train_score_rumus2_model(df, model_dir)

    # Final summary
    total_time = time.time() - start_time
    print_training_summary(summary, total_time)


if __name__ == "__main__":
    main()
