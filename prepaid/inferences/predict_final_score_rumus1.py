"""
Inference script for final_score_rumus1 model
Usage: python prepaid/inferences/predict_final_score_rumus1.py --is_rutin 1 --consecutive_anomaly_count 3
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).path.parent.parent))

from config.settings import MODEL_CONFIGS, MODEL_VERSION
from inferences.utils import load_trained_model, validate_features, prepare_input_dataframe, decode_class_label

MODEL_NAME = 'final_score_rumus1'


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
    return {0: 0, 1: 20, 2: 30, 3: 40, 4: 60}


def predict(features: dict, model_dir: str = None) -> dict:
    """
    Make prediction using trained final_score_rumus1 model.

    Args:
        features: Dictionary of feature names and values
                 Example: {'is_rutin': 1, 'consecutive_anomaly_count': 3}
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
        expected_classes = [0, 20, 40, 60]
    else:
        expected_classes = [0, 20, 30, 40]

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
        description='Predict final score rumus 1 for prepaid anomaly detection'
    )
    parser.add_argument(
        '--is_rutin',
        type=int,
        required=True,
        choices=[0, 1],
        help='Customer regularity status (0=not regular, 1=regular)'
    )
    parser.add_argument(
        '--consecutive_anomaly_count',
        type=int,
        required=True,
        help='Number of consecutive anomalies observed'
    )

    args = parser.parse_args()

    # Prepare features
    features = {
        'is_rutin': args.is_rutin,
        'consecutive_anomaly_count': args.consecutive_anomaly_count
    }

    # Make prediction
    print("="*60)
    print("FINAL_SCORE_RUMUS1 PREDICTION")
    print("="*60)
    print(f"\nInput features:")
    print(f"  - is_rutin: {features['is_rutin']} ({'regular' if features['is_rutin'] == 1 else 'irregular'})")
    print(f"  - consecutive_anomaly_count: {features['consecutive_anomaly_count']}")

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
        print(f"  - No severity: No consecutive anomalies detected")
    elif result['is_rutin'] == 1:
        # Regular customer scoring
        if score == 20:
            print(f"  - Low severity: Few consecutive anomalies")
        elif score == 40:
            print(f"  - Medium severity: Moderate consecutive anomalies")
        elif score == 60:
            print(f"  - High severity: Many consecutive anomalies")
    else:
        # Irregular customer scoring
        if score == 20:
            print(f"  - Low severity: Few consecutive anomalies")
        elif score == 30:
            print(f"  - Medium severity: Moderate consecutive anomalies")
        elif score == 40:
            print(f"  - High severity: Many consecutive anomalies")

    print("="*60)


if __name__ == "__main__":
    main()
