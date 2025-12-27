"""
Training script for is_rutin model
Usage: python prepaid/training/train_is_rutin.py
"""

import os
import sys
from pathlib import Path
import time

import pandas as pd
import xgboost as xgb

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_CONFIGS, TRAINING_SETTINGS, DEFAULT_TRAIN_DATA, MODEL_VERSION, PIPELINE_ORDER, FEATURE_SPEC
from training.utils import load_training_data, save_model, validate_features_and_target, validate_pipeline_order, print_training_summary, filter_dataset

MODEL_NAME = 'is_rutin'


def train_is_rutin_model(df: pd.DataFrame, model_dir: str) -> dict:
    """
    Train the is_rutin model and save it.

    Args:
        df: Training DataFrame
        model_dir: Directory to save the trained model

    Returns:
        Dictionary containing training summary
    """
    # Get model configuration
    model_config = MODEL_CONFIGS[MODEL_NAME]
    features = model_config['features']
    target = MODEL_NAME  # Target is the same as model name

    print(f"Model: {MODEL_NAME}")
    print(f"Features: {features}")
    print(f"Target: {target}")

    # Data filtering
    print("Filtering data...")
    df_sufficient, df_insufficient = filter_dataset(df)

    n_filtered = len(df_insufficient)
    print(f"Original samples: {len(df)}")
    print(f"Sufficient samples: {len(df_sufficient)}")
    print(f"Filtered out: {n_filtered} ({n_filtered/len(df)*100:.2f}%)")

    if not df_insufficient.empty:
        filter_stats = df_insufficient['filter_reason'].value_counts().to_dict()
        print(f"Filter reasons: {filter_stats}")

    df = df_sufficient

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

    # Check for missing values and drop if any
    print("Checking for missing values...")

    missing_features = X.isnull().sum().sum()
    missing_target = y.isnull().sum()

    if missing_features > 0 or missing_target > 0:
        if missing_features > 0:
            print(f"Warning: Found {missing_features} missing values in features")
        if missing_target > 0:
            print(f"Warning: Found {missing_target} missing values in target")

        print("Dropping rows with missing values...")

        # Combine X and y to drop together
        combined = pd.concat([X, y], axis=1)
        combined_clean = combined.dropna()

        n_dropped = len(combined) - len(combined_clean)
        print(f"Dropped {n_dropped} rows ({n_dropped/len(combined)*100:.2f}%)")

        X = combined_clean[features]
        y = combined_clean[target]

        print(f"Remaining samples: {len(X)}")
    else:
        print("No missing values found")

    # Initialize XGBoost classifier
    random_state = TRAINING_SETTINGS.get('random_state', 42)
    hyperparameters = {
        'random_state': random_state,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    model = xgb.XGBClassifier(**hyperparameters)

    # Train the model
    print("Starting model training...")
    training_start = time.time()

    model.fit(X, y)

    training_time = time.time() - training_start

    # Save the model
    print("Saving the trained model...")
    model_path, file_size_kb = save_model(model, model_dir, MODEL_NAME)
    print("Model saved")

    # Feature importance
    importance = model.feature_importances_

    # Model predictions on training data for metrics
    predictions = model.predict(X)
    predictions_proba = model.predict_proba(X)[:, 1]

    # Calculate metrics
    accuracy = float((y == predictions).mean())
    precision = float((predictions * y).sum() / (predictions.sum() + 1e-8))
    recall = float((predictions * y).sum() / (y.sum() + 1e-8))
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))

    # Calculate ROC AUC
    from sklearn.metrics import roc_auc_score
    roc_auc = float(roc_auc_score(y, predictions_proba))

    # Prepare summary dictionary
    summary = {
        'model_name': MODEL_NAME,
        'training_time_seconds': round(training_time, 2),
        'model_path': model_path,
        'model_size_kb': file_size_kb,
        'n_samples': len(df),
        'n_features': len(features),
        'feature_names': features,
        'n_filtered': n_filtered,
        'metrics': {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'roc_auc': round(roc_auc, 4)
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
    summary = train_is_rutin_model(df, model_dir)

    # Final summary
    total_time = time.time() - start_time
    print_training_summary(summary, total_time)


if __name__ == "__main__":
    main()
