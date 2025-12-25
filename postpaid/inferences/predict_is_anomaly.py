"""
Inference script for is_anomaly model
Usage: python postpaid/inferences/predict_is_anomaly.py --anomaly_score 45.5
"""

import os
import sys
import argparse
import joblib
import pandas as pd
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_CONFIGS, MODEL_VERSION

MODEL_NAME = 'is_anomaly'


def load_model(model_path: str):
    """
    Load trained model from disk.

    Args:
        model_path: Path to model file (.pkl)

    Returns:
        Loaded model object
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print(f"Please train the model first using: python postpaid/training/train_{MODEL_NAME}.py")
        sys.exit(1)

    model = joblib.load(model_path)
    return model


def get_base_features(model_features: list) -> list:
    """
    Get base features (non-derived) from model feature list.

    Args:
        model_features: List of all features used by model

    Returns:
        List of base features that user needs to provide
    """
    from config.settings import FEATURE_SPEC

    base_features = []
    for feat in model_features:
        feat_spec = FEATURE_SPEC.get(feat, {})
        # If feature has 'derived_from', it's a derived feature
        if 'derived_from' not in feat_spec:
            base_features.append(feat)

    return base_features


def validate_input(features: dict, model_config: dict) -> None:
    """
    Validate that all required base features are present.

    Args:
        features: Dictionary of feature names and values (only base features needed)
        model_config: Model configuration dictionary

    Raises:
        SystemExit: If validation fails
    """
    # Get base features dynamically (non-derived features)
    base_features = get_base_features(model_config['features'])
    missing_features = [feat for feat in base_features if feat not in features]

    if missing_features:
        print(f"Error: Missing required features: {missing_features}")
        print(f"Required features: {base_features}")
        sys.exit(1)


def preprocess_input(features: dict) -> pd.DataFrame:
    """
    Convert input features to DataFrame format expected by model.

    Args:
        features: Dictionary of base feature names and values

    Returns:
        DataFrame with features in correct order
    """
    model_config = MODEL_CONFIGS[MODEL_NAME]
    feature_order = model_config['features']

    # Create DataFrame with features in correct order
    X = pd.DataFrame([[features.get(feat, 0) for feat in feature_order]], columns=feature_order)
    return X


def predict(features: dict, model_dir: str = None) -> dict:
    """
    Make prediction using trained is_anomaly model.

    Args:
        features: Dictionary of feature names and values
                 Example: {'anomaly_score': 45.5}
        model_dir: Directory containing trained model (optional)
                   Defaults to postpaid/models/{MODEL_VERSION}/

    Returns:
        Dictionary containing prediction results
    """
    # Get model configuration
    model_config = MODEL_CONFIGS[MODEL_NAME]

    # Validate input features
    validate_input(features, model_config)

    # Determine model path
    if model_dir is None:
        current_dir = Path(__file__).parent.parent
        model_dir = os.path.join(current_dir, 'models', MODEL_VERSION)

    model_path = os.path.join(model_dir, f'{MODEL_NAME}.pkl')

    # Load model
    model = load_model(model_path)

    # Preprocess input
    X = preprocess_input(features)

    # Make prediction (get probability)
    proba = model.predict_proba(X)[0]
    raw_prediction = model.predict(X)[0]

    # Apply threshold from config
    threshold = model_config.get('threshold', 0.5)
    final_prediction = 1 if proba[1] >= threshold else 0

    # Prepare result
    result = {
        'model_name': MODEL_NAME,
        'model_path': model_path,
        'features': features,
        'raw_prediction': int(raw_prediction),
        'probability_anomaly': round(proba[1], 4),
        'probability_normal': round(proba[0], 4),
        'threshold': threshold,
        'final_prediction': int(final_prediction),
        'is_anomaly': bool(final_prediction)
    }

    return result


def main():
    """Main inference function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Predict binary anomaly classification for postpaid electricity consumption'
    )
    parser.add_argument(
        '--anomaly_score',
        type=float,
        required=True,
        help='Anomaly score from anomaly_score model (0-100)'
    )

    args = parser.parse_args()

    # Prepare features
    features = {
        'anomaly_score': args.anomaly_score
    }

    # Make prediction
    print("="*60)
    print("BINARY ANOMALY CLASSIFICATION")
    print("="*60)
    print(f"\nInput features:")
    print(f"  - anomaly_score: {features['anomaly_score']}")

    result = predict(features)

    # Print result
    print(f"\nPrediction Result:")
    print(f"  - Model: {result['model_name']}")
    print(f"  - Probability (Normal): {result['probability_normal']:.2%}")
    print(f"  - Probability (Anomaly): {result['probability_anomaly']:.2%}")
    print(f"  - Threshold: {result['threshold']}")
    print(f"  - Final Prediction: {'ANOMALY' if result['is_anomaly'] else 'NORMAL'}")

    # Interpretation
    print(f"\nInterpretation:")
    if result['is_anomaly']:
        print(f"  - Status: ANOMALY DETECTED")
        print(f"  - Consumption pattern is abnormal")
        print(f"  - Further investigation recommended")
    else:
        print(f"  - Status: NORMAL")
        print(f"  - Consumption pattern is within expected range")
        print(f"  - No immediate action needed")

    print("="*60)


if __name__ == "__main__":
    main()
