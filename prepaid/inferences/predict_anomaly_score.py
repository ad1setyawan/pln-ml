"""
Inference script for anomaly_score model
Usage: python prepaid/inferences/predict_anomaly_score.py --final_score_rumus1 40 --final_score_rumus2 20
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_CONFIGS, MODEL_VERSION
from inferences.utils import load_trained_model, validate_features, prepare_input_dataframe, apply_postprocessing

MODEL_NAME = 'anomaly_score'


def predict(features: dict, model_dir: str = None) -> dict:
    """
    Make prediction using trained anomaly_score model.

    Args:
        features: Dictionary of feature names and values
                 Example: {'final_score_rumus1': 40, 'final_score_rumus2': 20}
        model_dir: Directory containing trained model (optional)

    Returns:
        Dictionary containing prediction results
    """
    # Get model configuration
    model_config = MODEL_CONFIGS[MODEL_NAME]
    feature_order = model_config['features']
    postprocess = model_config.get('postprocess', [])

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
    raw_prediction = model.predict(X)[0]

    # Apply postprocessing
    final_prediction = apply_postprocessing(raw_prediction, postprocess)

    # Prepare result
    result = {
        'model_name': MODEL_NAME,
        'features': features,
        'raw_prediction': round(float(raw_prediction), 4),
        'final_prediction': round(float(final_prediction), 2),
        'postprocessing': postprocess
    }

    return result


def main():
    """Main inference function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Predict anomaly score for prepaid electricity consumption'
    )
    parser.add_argument(
        '--final_score_rumus1',
        type=int,
        required=True,
        help='Final score from rumus 1 (0-60 scale)'
    )
    parser.add_argument(
        '--final_score_rumus2',
        type=int,
        required=True,
        help='Final score from rumus 2 (0-60 scale)'
    )

    args = parser.parse_args()

    # Prepare features
    features = {
        'final_score_rumus1': args.final_score_rumus1,
        'final_score_rumus2': args.final_score_rumus2
    }

    # Calculate sum manually for reference
    manual_sum = features['final_score_rumus1'] + features['final_score_rumus2']

    # Make prediction
    print("="*60)
    print("ANOMALY_SCORE PREDICTION")
    print("="*60)
    print(f"\nInput features:")
    print(f"  - final_score_rumus1: {features['final_score_rumus1']}")
    print(f"  - final_score_rumus2: {features['final_score_rumus2']}")
    print(f"  - Manual sum: {manual_sum}")

    result = predict(features)

    # Print result
    print(f"\nPrediction Result:")
    print(f"  - Model: {result['model_name']}")
    print(f"  - Raw prediction: {result['raw_prediction']}")
    if result['postprocessing']:
        print(f"  - Postprocessing applied: {result['postprocessing']}")
    print(f"  - Final anomaly score: {result['final_prediction']} (0-100 scale)")

    # Interpretation
    score = result['final_prediction']
    print(f"\nInterpretation:")
    if score < 20:
        print(f"  - Status: NORMAL")
        print(f"  - Very low anomaly score - consumption pattern is normal")
    elif score < 40:
        print(f"  - Status: LOW ANOMALY")
        print(f"  - Slight deviation detected - may need monitoring")
    elif score < 60:
        print(f"  - Status: MEDIUM ANOMALY")
        print(f"  - Moderate deviation detected - investigation recommended")
    elif score < 80:
        print(f"  - Status: HIGH ANOMALY")
        print(f"  - Significant deviation detected - investigation strongly recommended")
    else:
        print(f"  - Status: CRITICAL ANOMALY")
        print(f"  - Severe deviation detected - immediate investigation required")

    print("="*60)


if __name__ == "__main__":
    main()
