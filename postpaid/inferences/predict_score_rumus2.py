"""
Inference script for score_rumus2 model
Usage: python postpaid/inferences/predict_score_rumus2.py --pemakaian 50.0 --avg_pemakaian_gardu 60.0
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

MODEL_NAME = 'score_rumus2'


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


def compute_derived_features(features: dict) -> dict:
    """
    Compute derived features from base features.

    Args:
        features: Dictionary of base feature names and values

    Returns:
        Dictionary with all features (base + derived)
    """
    result = features.copy()

    # Compute ratio_gardu = pemakaian / (avg_pemakaian_gardu + 1)
    if 'pemakaian' in features and 'avg_pemakaian_gardu' in features:
        result['ratio_gardu'] = features['pemakaian'] / (features['avg_pemakaian_gardu'] + 1)

    # Compute pct_diff_gardu = (pemakaian - avg_pemakaian_gardu) / (avg_pemakaian_gardu + 1)
    if 'pemakaian' in features and 'avg_pemakaian_gardu' in features:
        result['pct_diff_gardu'] = (features['pemakaian'] - features['avg_pemakaian_gardu']) / (features['avg_pemakaian_gardu'] + 1)

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


def predict(features: dict, model_dir: str = None) -> dict:
    """
    Make prediction using trained score_rumus2 model.

    Args:
        features: Dictionary of feature names and values
                 Example: {'pemakaian': 50.0, 'avg_pemakaian_gardu': 60.0}
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

    # Make prediction
    raw_prediction = model.predict(X)[0]
    final_prediction = raw_prediction

    # Prepare result
    result = {
        'model_name': MODEL_NAME,
        'model_path': model_path,
        'input_features': features,
        'engineered_features': {
            'ratio_gardu': round(X['ratio_gardu'].values[0], 4),
            'pct_diff_gardu': round(X['pct_diff_gardu'].values[0], 4)
        },
        'raw_prediction': round(raw_prediction, 4),
        'final_prediction': round(final_prediction, 2)
    }

    return result


def main():
    """Main inference function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Predict gardu deviation score for postpaid electricity consumption'
    )
    parser.add_argument(
        '--pemakaian',
        type=float,
        required=True,
        help='Actual electricity consumption (kWh)'
    )
    parser.add_argument(
        '--avg_pemakaian_gardu',
        type=float,
        required=True,
        help='Average electricity consumption at gardu/substation level (kWh)'
    )

    args = parser.parse_args()

    # Prepare features
    features = {
        'pemakaian': args.pemakaian,
        'avg_pemakaian_gardu': args.avg_pemakaian_gardu
    }

    # Make prediction
    print("="*60)
    print("GARDU DEVIATION SCORE PREDICTION")
    print("="*60)
    print(f"\nInput features:")
    print(f"  - pemakaian (actual): {features['pemakaian']} kWh")
    print(f"  - avg_pemakaian_gardu (gardu average): {features['avg_pemakaian_gardu']} kWh")

    result = predict(features)

    # Print result
    print(f"\nPrediction Result:")
    print(f"  - Model: {result['model_name']}")
    print(f"\nEngineered Features:")
    print(f"  - ratio_gardu: {result['engineered_features']['ratio_gardu']}")
    print(f"    (Ratio of actual to gardu average)")
    print(f"  - pct_diff_gardu: {result['engineered_features']['pct_diff_gardu']:.2%}")
    print(f"    (Percentage deviation from gardu average)")
    print(f"\nFinal score_rumus2: {result['final_prediction']}")

    # Interpretation
    pct_diff = result['engineered_features']['pct_diff_gardu']
    print(f"\nInterpretation:")
    if pct_diff > 0.1:
        print(f"  - Status: ABOVE AVERAGE")
        print(f"  - Consumption is {pct_diff:.1%} higher than gardu average")
    elif pct_diff < -0.1:
        print(f"  - Status: BELOW AVERAGE")
        print(f"  - Consumption is {abs(pct_diff):.1%} lower than gardu average")
    else:
        print(f"  - Status: NORMAL")
        print(f"  - Consumption is close to gardu average (±10%)")

    print("="*60)


if __name__ == "__main__":
    main()
