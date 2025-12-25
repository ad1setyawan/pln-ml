"""
Inference script for final_score_rumus2 model
Usage: python postpaid/inferences/predict_final_score_rumus2.py --score_rumus2 -0.35
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import json
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_CONFIGS, MODEL_VERSION

MODEL_NAME = 'final_score_rumus2'


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


def load_label_mapping(model_dir: str) -> dict:
    """
    Load label mapping for class interpretation.

    Args:
        model_dir: Directory containing the trained model

    Returns:
        Dictionary mapping encoded classes to original labels
    """
    # Try to load label mapping from file
    mapping_path = os.path.join(model_dir, f'{MODEL_NAME}_label_mapping.json')

    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            return json.load(f)

    # Fallback: return mapping from config
    return {str(i): label for i, label in enumerate(MODEL_CONFIGS[MODEL_NAME]['classes'])}


def predict(features: dict, model_dir: str = None) -> dict:
    """
    Make prediction using trained final_score_rumus2 model.

    Args:
        features: Dictionary of feature names and values
                 Example: {'score_rumus2': -0.35}
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

    # Load label mapping
    label_mapping = load_label_mapping(model_dir)

    # Preprocess input
    X = preprocess_input(features)

    # Make prediction
    prediction_encoded = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    # Get predicted class label
    predicted_class = label_mapping.get(str(prediction_encoded), prediction_encoded)

    # Prepare result
    result = {
        'model_name': MODEL_NAME,
        'model_path': model_path,
        'features': features,
        'prediction_encoded': int(prediction_encoded),
        'predicted_class': predicted_class,
        'probabilities': {label_mapping.get(str(i), i): round(prob, 4)
                         for i, prob in enumerate(proba)},
        'confidence': round(float(max(proba)), 4)
    }

    return result


def main():
    """Main inference function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Predict final score rumus 2 (gardu deviation severity)'
    )
    parser.add_argument(
        '--score_rumus2',
        type=float,
        required=True,
        help='Gardu deviation score from score_rumus2 model'
    )

    args = parser.parse_args()

    # Prepare features
    features = {
        'score_rumus2': args.score_rumus2
    }

    # Make prediction
    print("="*60)
    print("FINAL SCORE RUMUS 2 PREDICTION")
    print("="*60)
    print(f"\nInput features:")
    print(f"  - score_rumus2: {features['score_rumus2']}")

    result = predict(features)

    # Print result
    print(f"\nPrediction Result:")
    print(f"  - Model: {result['model_name']}")
    print(f"\nClass Probabilities:")
    for class_label, prob in result['probabilities'].items():
        print(f"  - Score {class_label}: {prob:.2%}")

    print(f"\nFinal Prediction:")
    print(f"  - Predicted Score: {result['predicted_class']}")
    print(f"  - Confidence: {result['confidence']:.2%}")

    # Interpretation
    score = result['predicted_class']
    print(f"\nInterpretation:")
    if score == 0:
        print(f"  - Level: NORMAL")
        print(f"  - < 20% consumption drop from gardu average")
        print(f"  - No significant deviation")
    elif score == 10:
        print(f"  - Level: LEVEL 1")
        print(f"  - 20-30% consumption drop from gardu average")
        print(f"  - Slight deviation detected")
    elif score == 20:
        print(f"  - Level: LEVEL 2")
        print(f"  - 30-40% consumption drop from gardu average")
        print(f"  - Moderate deviation - monitor closely")
    elif score == 30:
        print(f"  - Level: LEVEL 3")
        print(f"  - 40-50% consumption drop from gardu average")
        print(f"  - Significant deviation - investigation needed")
    elif score == 40:
        print(f"  - Level: LEVEL 4")
        print(f"  - ≥ 50% consumption drop from gardu average")
        print(f"  - Severe deviation - immediate action required")

    print("="*60)


if __name__ == "__main__":
    main()
