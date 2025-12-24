"""
Inference script for anomaly_score model
Usage: python postpaid/inferences/predict.py --pemakaian 41.0 --baseline 40.0
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np

import joblib
import pandas as pd

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_CONFIGS, INFERENCE_SETTINGS

# Add root directory to path to import shared utilities
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from shared.utils import log


def load_model(model_path: str):
    """
    Load trained model from file.

    Args:
        model_path: Path to saved model file

    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run training first: python postpaid/training/train.py")
        sys.exit(1)

    model = joblib.load(model_path)
    return model


def validate_features(pemakaian: float, baseline: float) -> None:
    """
    Validate input features.

    Args:
        pemakaian: Actual electricity consumption
        baseline: Expected/normal consumption

    Raises:
        ValueError: If validation fails
    """
    if pd.isna(pemakaian) or pd.isna(baseline):
        raise ValueError("Features cannot be NaN")

    if not isinstance(pemakaian, (int, float)) or not isinstance(baseline, (int, float)):
        raise ValueError("Features must be numeric")

    if pemakaian < 0 or baseline < 0:
        raise ValueError("Features cannot be negative")


def apply_postprocessing(prediction: np.ndarray, postprocess: list) -> np.ndarray:
    """
    Apply postprocessing steps to prediction.

    Args:
        prediction: Raw prediction from model
        postprocess: List of postprocessing steps

    Returns:
        Postprocessed prediction
    """
    result = prediction.copy()

    for step in postprocess:
        if step == 'clip_0_1':
            result = np.clip(result, 0, 1)
        elif step == 'clip_0_100':
            result = np.clip(result, 0, 100)
        # Add more postprocessing steps here if needed

    return result


def predict(pemakaian: float, baseline: float, model_dir: str) -> float:
    """
    Make prediction for anomaly_score.

    Args:
        pemakaian: Actual electricity consumption
        baseline: Expected/normal consumption
        model_dir: Directory containing the trained model

    Returns:
        Anomaly score (0-1 range)
    """
    # Get model configuration
    model_config = MODEL_CONFIGS['anomaly_score']
    features = model_config['features']
    postprocess = model_config.get('postprocess', [])

    # Validate features
    try:
        validate_features(pemakaian, baseline)
    except ValueError as e:
        print(f"Validation Error: {e}")
        sys.exit(1)

    # Load model
    model_path = os.path.join(model_dir, 'anomaly_score.pkl')
    model = load_model(model_path)

    # Prepare input data
    input_data = pd.DataFrame([[pemakaian, baseline]], columns=features)

    # Make prediction
    raw_prediction = model.predict(input_data)

    # Apply postprocessing
    final_prediction = apply_postprocessing(raw_prediction, postprocess)

    return float(final_prediction[0])


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Predict anomaly score for electricity consumption'
    )
    parser.add_argument(
        '--pemakaian',
        type=float,
        required=True,
        help='Actual electricity consumption'
    )
    parser.add_argument(
        '--baseline',
        type=float,
        required=True,
        help='Expected/normal consumption level'
    )

    args = parser.parse_args()

    print("="*60)
    print("Anomaly Score Prediction")
    print("="*60)
    print(f"Input:")
    print(f"  pemakaian: {args.pemakaian}")
    print(f"  baseline:  {args.baseline}")

    # Paths
    current_dir = Path(__file__).parent.parent.parent
    model_dir = os.path.join(current_dir, 'postpaid', 'models')

    # Make prediction
    try:
        score = predict(args.pemakaian, args.baseline, model_dir)

        print(f"\nPrediction:")
        print(f"  anomaly_score: {score:.2f}")
        print("="*60)

        # Additional interpretation
        if score < 30:
            print("Interpretation: Low anomaly - Normal consumption pattern")
        elif score < 60:
            print("Interpretation: Medium anomaly - Slight deviation from baseline")
        else:
            print("Interpretation: High anomaly - Significant deviation from baseline")

    except Exception as e:
        print(f"\nError during prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
