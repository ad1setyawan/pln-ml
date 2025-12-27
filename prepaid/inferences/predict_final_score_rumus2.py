"""
Inference script for final_score_rumus2 model
Usage: python prepaid/inferences/predict_final_score_rumus2.py --is_rutin 1 --pemakaian 150.5 --avg_pemakaian_gardu 140.0
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_CONFIGS, MODEL_VERSION
from inferences.utils import load_trained_model, validate_features, prepare_input_dataframe, decode_class_label

MODEL_NAME = 'final_score_rumus2'


def load_label_mapping(model_dir: str = None) -> dict:
    """
    Load label mapping from model metadata.

    Args:
        model_dir: Directory containing trained model

    Returns:
        Label mapping dictionary
    """
    # For now, return default mapping
    # In production, this should load from saved metadata
    return {0: 0, 1: 10, 2: 20, 3: 30, 4: 40, 5: 60}


def predict(features: dict, model_dir: str = None) -> dict:
    """
    Make prediction using trained final_score_rumus2 model.

    Args:
        features: Dictionary of feature names and values
                 Example: {'is_rutin': 1, 'pemakaian': 150.5, 'avg_pemakaian_gardu': 140.0}
        model_dir: Directory containing trained model (optional)

    Returns:
        Dictionary containing prediction results
    """
    # Get model configuration
    model_config = MODEL_CONFIGS[MODEL_NAME]
    feature_order = model_config['features']
    classes = model_config['classes']

    # Validate input features
    validate_features(features, feature_order)

    # Determine model path
    if model_dir is None:
        current_dir = Path(__file__).parent.parent
        model_dir = os.path.join(current_dir, 'models', MODEL_VERSION)

    # Load model
    model = load_trained_model(MODEL_NAME, model_dir=model_dir)

    # Prepare input
    X = prepare_input_dataframe(features, feature_order)

    # Make prediction
    prediction = model.predict(X)[0]

    # Get probability distribution
    prediction_proba = model.predict_proba(X)[0]

    # Load label mapping (default for now)
    label_mapping = load_label_mapping(model_dir)

    # Decode class to score
    predicted_score = decode_class_label(prediction, label_mapping)

    # Determine expected classes based on is_rutin
    is_rutin = features['is_rutin']
    if is_rutin == 1:
        expected_classes = [0, 10, 20, 30, 40]
    else:
        expected_classes = [0, 20, 40, 60]

    # Prepare result
    result = {
        'model_name': MODEL_NAME,
        'features': features,
        'is_rutin': is_rutin,
        'predicted_class': int(prediction),
        'predicted_score': predicted_score,
        'expected_classes': expected_classes,
        'probability_distribution': {str(label_mapping[i]): round(float(pred), 4) for i, pred in enumerate(prediction_proba)}
    }

    return result


def main():
    """Main inference function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Predict final score rumus 2 for prepaid anomaly detection'
    )
    parser.add_argument(
        '--is_rutin',
        type=int,
        required=True,
        choices=[0, 1],
        help='Customer regularity status (0=not regular, 1=regular)'
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
        help='Average consumption at electrical substation (kWh)'
    )

    args = parser.parse_args()

    # Prepare features
    features = {
        'is_rutin': args.is_rutin,
        'pemakaian': args.pemakaian,
        'avg_pemakaian_gardu': args.avg_pemakaian_gardu
    }

    # Make prediction
    print("="*60)
    print("FINAL_SCORE_RUMUS2 PREDICTION")
    print("="*60)
    print(f"\nInput features:")
    print(f"  - is_rutin: {features['is_rutin']} ({'regular' if features['is_rutin'] == 1 else 'irregular'})")
    print(f"  - pemakaian: {features['pemakaian']} kWh")
    print(f"  - avg_pemakaian_gardu: {features['avg_pemakaian_gardu']} kWh")

    # Calculate deviation from gardu average
    deviation = features['pemakaian'] - features['avg_pemakaian_gardu']
    deviation_pct = (deviation / features['avg_pemakaian_gardu']) * 100 if features['avg_pemakaian_gardu'] > 0 else 0

    print(f"  - Deviation from gardu: {deviation:.2f} kWh ({deviation_pct:+.2f}%)")

    result = predict(features)

    # Print result
    print(f"\nPrediction Result:")
    print(f"  - Model: {result['model_name']}")
    print(f"  - Customer type: {'regular' if result['is_rutin'] == 1 else 'irregular'}")
    print(f"  - Expected score classes: {result['expected_classes']}")
    print(f"  - Predicted score: {result['predicted_score']}")
    print(f"  - Probability distribution:")
    for score, prob in result['probability_distribution'].items():
        print(f"      Score {score}: {prob}")

    # Interpretation
    print(f"\nInterpretation:")
    score = result['predicted_score']
    if score == 0:
        print(f"  - No anomaly: Consumption matches gardu average")
    elif result['is_rutin'] == 1:
        # Regular customer scoring
        if score == 10:
            print(f"  - Very low deviation: Slight difference from gardu average")
        elif score == 20:
            print(f"  - Low deviation: Small difference from gardu average")
        elif score == 30:
            print(f"  - Medium deviation: Moderate difference from gardu average")
        elif score == 40:
            print(f"  - High deviation: Significant difference from gardu average")
    else:
        # Irregular customer scoring
        if score == 20:
            print(f"  - Low deviation: Small difference from gardu average")
        elif score == 40:
            print(f"  - Medium deviation: Moderate difference from gardu average")
        elif score == 60:
            print(f"  - High deviation: Significant difference from gardu average")

    print("="*60)


if __name__ == "__main__":
    main()
