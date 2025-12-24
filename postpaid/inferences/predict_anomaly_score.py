"""
Inference script for anomaly_score model
Usage: python postpaid/inferences/predict_anomaly_score.py --pemakaian 41.0 --baseline 40.0
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

MODEL_NAME = 'anomaly_score'


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
        print("Please train the model first using: python postpaid/training/train_anomaly_score.py")
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


def compute_derived_features(features: dict) -> dict:
    """
    Compute derived features from base features.

    Args:
        features: Dictionary of base feature names and values

    Returns:
        Dictionary with all features (base + derived)
    """
    result = features.copy()

    # Compute ratio = pemakaian / (baseline + 1)
    if 'pemakaian' in features and 'baseline' in features:
        result['ratio'] = features['pemakaian'] / (features['baseline'] + 1)

    # Compute diff = baseline - pemakaian
    if 'pemakaian' in features and 'baseline' in features:
        result['diff'] = features['baseline'] - features['pemakaian']

    # Compute is_over_baseline = pemakaian > baseline
    if 'pemakaian' in features and 'baseline' in features:
        result['is_over_baseline'] = 1 if features['pemakaian'] > features['baseline'] else 0

    return result


def preprocess_input(features: dict) -> pd.DataFrame:
    """
    Convert input features to DataFrame format expected by model.
    Performs feature engineering to create all required features.

    Args:
        features: Dictionary of base feature names and values

    Returns:
        DataFrame with features in correct order (including engineered features)
    """
    model_config = MODEL_CONFIGS[MODEL_NAME]
    feature_order = model_config['features']

    # Compute derived features from base features
    all_features = compute_derived_features(features)

    # Create DataFrame with features in correct order
    X = pd.DataFrame([[all_features[feat] for feat in feature_order]], columns=feature_order)
    return X


def postprocess_prediction(prediction: float, model_config: dict) -> float:
    """
    Apply postprocessing steps to model prediction.

    Args:
        prediction: Raw model prediction
        model_config: Model configuration dictionary

    Returns:
        Postprocessed prediction
    """
    postprocess_steps = model_config.get('postprocess', [])

    for step in postprocess_steps:
        if step == 'clip_0_100':
            prediction = max(0, min(100, prediction))
        elif step == 'clip_0_1':
            prediction = max(0, min(1, prediction))

    return prediction


def predict(features: dict, model_dir: str = None) -> dict:
    """
    Make prediction using trained anomaly_score model.

    Args:
        features: Dictionary of feature names and values
                 Example: {'pemakaian': 41.0, 'baseline': 40.0}
        model_dir: Directory containing trained model (optional)
                   Defaults to postpaid/models/{MODEL_VERSION}/

    Returns:
        Dictionary containing prediction results
    """
    # Get model configuration
    model_config = MODEL_CONFIGS[MODEL_NAME]

    # Validate input features
    validate_input(features, model_config)

    # Business rule: If pemakaian >= baseline, no anomaly (score = 0)
    # Anomaly only occurs when consumption is significantly lower than baseline
    pemakaian = features['pemakaian']
    baseline = features['baseline']

    if pemakaian >= baseline:
        # No need to predict, consumption is normal or higher than expected
        result = {
            'model_name': MODEL_NAME,
            'model_path': None,
            'features': features,
            'raw_prediction': 0.0,
            'final_prediction': 0.0,
            'postprocessing': ['business_rule_pemakaian_ge_baseline'],
            'note': 'pemakaian >= baseline, no anomaly detected (score = 0)'
        }
        return result

    # Determine model path
    if model_dir is None:
        current_dir = Path(__file__).parent.parent
        model_dir = os.path.join(current_dir, 'models', MODEL_VERSION)

    model_path = os.path.join(model_dir, f'{MODEL_NAME}.pkl')

    # Load model
    model = load_model(model_path)

    # Preprocess input
    X = preprocess_input(features)

    # Make prediction
    raw_prediction = model.predict(X)[0]

    # Apply postprocessing
    final_prediction = postprocess_prediction(raw_prediction, model_config)

    # Prepare result
    result = {
        'model_name': MODEL_NAME,
        'model_path': model_path,
        'features': features,
        'raw_prediction': round(raw_prediction, 4),
        'final_prediction': round(final_prediction, 2),
        'postprocessing': model_config.get('postprocess', [])
    }

    return result


def main():
    """Main inference function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Predict anomaly score for postpaid electricity consumption'
    )
    parser.add_argument(
        '--pemakaian',
        type=float,
        required=True,
        help='Actual electricity consumption (kWh)'
    )
    parser.add_argument(
        '--baseline',
        type=float,
        required=True,
        help='Baseline/expected electricity consumption (kWh)'
    )

    args = parser.parse_args()

    # Prepare features
    features = {
        'pemakaian': args.pemakaian,
        'baseline': args.baseline
    }

    # Make prediction
    print("="*60)
    print("ANOMALY SCORE PREDICTION")
    print("="*60)
    print(f"\nInput features:")
    print(f"  - pemakaian (actual): {features['pemakaian']} kWh")
    print(f"  - baseline (expected): {features['baseline']} kWh")

    result = predict(features)

    # Print result
    print(f"\nPrediction Result:")
    print(f"  - Model: {result['model_name']}")

    # Check if business rule was applied
    if 'business_rule_pemakaian_ge_baseline' in result.get('postprocessing', []):
        print(f"  - Business Rule Applied: pemakaian >= baseline")
        print(f"  - Note: {result['note']}")
        print(f"  - Final anomaly score: {result['final_prediction']} (0-100 scale)")
        print(f"\nInterpretation:")
        print(f"  - Status: NORMAL")
        print(f"  - Consumption is normal or higher than baseline")
        print(f"  - No anomaly detection needed")
    else:
        print(f"  - Raw prediction: {result['raw_prediction']}")
        if result['postprocessing']:
            print(f"  - Postprocessing applied: {result['postprocessing']}")
        print(f"  - Final anomaly score: {result['final_prediction']} (0-100 scale)")

        # Interpretation
        score = result['final_prediction']
        print(f"\nInterpretation:")
        if score < 20:
            print(f"  - Status: NORMAL")
            print(f"  - Actual consumption is very close to baseline")
        elif score <= 30:
            print(f"  - Status: SLIGHT DEVIATION")
            print(f"  - Actual consumption differs somewhat from baseline")
        else:
            print(f"  - Status: IS ANOMALY")
            print(f"  - Anomaly score > 30: Investigation strongly recommended")
            print(f"  - Severe deviation from baseline detected")

    print("="*60)


if __name__ == "__main__":
    main()
