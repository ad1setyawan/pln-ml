"""
Inference script for is_rutin model
Usage: python prepaid/inferences/predict_is_rutin.py --avg_12m_pemakaian 150.5 --std_12m_pemakaian 25.3 --freq_tx_12m 12 --avg_gap_days_12m 30.5
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_CONFIGS, MODEL_VERSION
from inferences.utils import load_trained_model, validate_features, prepare_input_dataframe

MODEL_NAME = 'is_rutin'


def predict(features: dict, model_dir: str = None) -> dict:
    """
    Make prediction using trained is_rutin model.

    Args:
        features: Dictionary of feature names and values
                 Example: {'avg_12m_pemakaian': 150.5, 'std_12m_pemakaian': 25.3, ...}
        model_dir: Directory containing trained model (optional)
                   Defaults to prepaid/models/{MODEL_VERSION}/

    Returns:
        Dictionary containing prediction results
    """
    # Get model configuration
    model_config = MODEL_CONFIGS[MODEL_NAME]
    feature_order = model_config['features']
    threshold = model_config.get('threshold', 0.5)

    # Validate input features
    validate_features(features, feature_order)

    # Determine model path
    if model_dir is None:
        current_dir = Path(__file__).parent.parent
        model_dir = os.path.join(current_dir, 'models', MODEL_VERSION)

    model_path = os.path.join(model_dir, f'{MODEL_NAME}.pkl')

    # Load model
    model = load_trained_model(MODEL_NAME, model_dir=model_dir)

    # Prepare input
    X = prepare_input_dataframe(features, feature_order)

    # Make prediction (get probability)
    prediction_proba = model.predict_proba(X)[0]
    prediction_class = int(prediction_proba[1] >= threshold)  # Class 1 if prob >= threshold

    # Prepare result
    result = {
        'model_name': MODEL_NAME,
        'model_path': model_path,
        'features': features,
        'prediction_class': prediction_class,
        'prediction_label': 'rutin' if prediction_class == 1 else 'not_rutin',
        'probability_class_0': round(prediction_proba[0], 4),
        'probability_class_1': round(prediction_proba[1], 4),
        'threshold': threshold
    }

    return result


def main():
    """Main inference function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Predict regularity (is_rutin) for prepaid electricity customer'
    )
    parser.add_argument(
        '--avg_12m_pemakaian',
        type=float,
        required=True,
        help='Average consumption over 12 months (kWh)'
    )
    parser.add_argument(
        '--std_12m_pemakaian',
        type=float,
        required=True,
        help='Standard deviation of consumption over 12 months'
    )
    parser.add_argument(
        '--freq_tx_12m',
        type=int,
        required=True,
        help='Frequency of transactions in 12 months'
    )
    parser.add_argument(
        '--avg_gap_days_12m',
        type=float,
        required=True,
        help='Average gap between transactions in days'
    )

    args = parser.parse_args()

    # Prepare features
    features = {
        'avg_12m_pemakaian': args.avg_12m_pemakaian,
        'std_12m_pemakaian': args.std_12m_pemakaian,
        'freq_tx_12m': args.freq_tx_12m,
        'avg_gap_days_12m': args.avg_gap_days_12m
    }

    # Make prediction
    print("="*60)
    print("IS_RUTIN PREDICTION")
    print("="*60)
    print(f"\nInput features:")
    print(f"  - avg_12m_pemakaian: {features['avg_12m_pemakaian']} kWh")
    print(f"  - std_12m_pemakaian: {features['std_12m_pemakaian']}")
    print(f"  - freq_tx_12m: {features['freq_tx_12m']} transactions")
    print(f"  - avg_gap_days_12m: {features['avg_gap_days_12m']} days")

    result = predict(features)

    # Print result
    print(f"\nPrediction Result:")
    print(f"  - Model: {result['model_name']}")
    print(f"  - Prediction: {result['prediction_label']} (class {result['prediction_class']})")
    print(f"  - Probability (not_rutin): {result['probability_class_0']}")
    print(f"  - Probability (rutin): {result['probability_class_1']}")
    print(f"  - Threshold: {result['threshold']}")

    # Interpretation
    print(f"\nInterpretation:")
    if result['prediction_class'] == 1:
        print(f"  - Customer has REGULAR behavior pattern")
        print(f"  - Consistent consumption and transaction pattern")
        print(f"  - Suitable for standard anomaly detection rules")
    else:
        print(f"  - Customer has IRREGULAR behavior pattern")
        print(f"  - Inconsistent consumption or transaction pattern")
        print(f"  - May require special handling in anomaly detection")

    print("="*60)


if __name__ == "__main__":
    main()
