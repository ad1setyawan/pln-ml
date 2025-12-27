"""
Complete ML Pipeline Inference for Prepaid Anomaly Detection
Runs all models sequentially according to PIPELINE_ORDER

Usage:
  python prepaid/inferences/predict_pipeline.py \\
    --avg_12m_pemakaian 150.5 \\
    --std_12m_pemakaian 25.3 \\
    --freq_tx_12m 12 \\
    --avg_gap_days_12m 30.5 \\
    --consecutive_anomaly_count 3 \\
    --pemakaian 145.0 \\
    --avg_pemakaian_gardu 140.0
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Tuple

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_CONFIGS, MODEL_VERSION, PIPELINE_ORDER, FEATURE_SPEC


def load_model(model_name: str, model_dir: str):
    """Load trained model from disk."""
    model_path = os.path.join(model_dir, f'{model_name}.pkl')

    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print(f"Skipping {model_name} - Please train the model first")
        return None

    model = joblib.load(model_path)
    return model


def load_label_mapping(model_name: str, model_dir: str) -> dict:
    """Load label mapping for classification models."""
    mapping_path = os.path.join(model_dir, f'{model_name}_label_mapping.json')

    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            return json.load(f)

    # Fallback - use classes from config
    classes = MODEL_CONFIGS[model_name].get('classes', [])
    return {str(i): label for i, label in enumerate(classes)}


def run_model_inference(model_name: str, features: Dict[str, Any], model_dir: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Run inference for a single model.

    Returns:
        Tuple of (prediction, metadata)
    """
    model_config = MODEL_CONFIGS[model_name]
    model = load_model(model_name, model_dir)

    if model is None:
        return None, {'error': 'Model not found', 'model_name': model_name}

    # Get feature order
    feature_order = model_config['features']

    # Prepare input DataFrame
    X = pd.DataFrame([[features.get(feat, 0) for feat in feature_order]], columns=feature_order)

    # Make prediction based on task type
    task_type = model_config['task']

    if task_type == 'regression':
        prediction = model.predict(X)[0]

        # Apply postprocessing if configured
        postprocess_steps = model_config.get('postprocess', [])
        for step in postprocess_steps:
            if step == 'clip_0_100':
                prediction = max(0, min(100, prediction))
            elif step == 'clip_0_1':
                prediction = max(0, min(1, prediction))

        metadata = {
            'task': task_type,
            'features_used': feature_order,
            'prediction': round(float(prediction), 4)
        }

    elif task_type in ['binary_classification', 'multiclass_classification', 'ordinal_classification']:
        # Get probabilities and prediction
        proba = model.predict_proba(X)[0]
        prediction_encoded = model.predict(X)[0]

        # Load label mapping
        label_mapping = load_label_mapping(model_name, model_dir)
        predicted_label = label_mapping.get(str(prediction_encoded), prediction_encoded)

        # Apply threshold for binary classification
        if task_type == 'binary_classification':
            threshold = model_config.get('threshold', 0.5)
            final_prediction = 1 if proba[1] >= threshold else 0
            predicted_label = int(final_prediction)

        # Create probabilities dictionary
        probabilities = {
            label_mapping.get(str(i), i): round(prob, 4)
            for i, prob in enumerate(proba)
        }

        metadata = {
            'task': task_type,
            'features_used': feature_order,
            'prediction_encoded': int(prediction_encoded),
            'predicted_label': predicted_label,
            'probabilities': probabilities,
            'confidence': round(float(max(proba)), 4)
        }

        prediction = predicted_label

    else:
        metadata = {
            'task': task_type,
            'features_used': feature_order,
            'error': f'Unknown task type: {task_type}'
        }
        prediction = None

    return prediction, metadata


def run_pipeline(features: Dict[str, Any], model_dir: str = None) -> Dict[str, Any]:
    """
    Run complete ML pipeline inference.

    Args:
        features: Dictionary of base input features
        model_dir: Directory containing trained models

    Returns:
        Dictionary with all model predictions
    """
    if model_dir is None:
        current_dir = Path(__file__).parent.parent
        model_dir = os.path.join(current_dir, 'models', MODEL_VERSION)

    # Store all predictions
    all_predictions = {}
    all_metadata = {}

    # Features will be updated as we progress through pipeline
    current_features = features.copy()

    print("\n" + "="*60)
    print("RUNNING PIPELINE INFERENCE")
    print("="*60)

    # Run models in pipeline order
    for model_name in PIPELINE_ORDER:
        print(f"\n[{PIPELINE_ORDER.index(model_name) + 1}/{len(PIPELINE_ORDER)}] Running {model_name}...")

        # Run inference
        prediction, metadata = run_model_inference(model_name, current_features, model_dir)

        if prediction is not None:
            # Add prediction to features for next models
            current_features[model_name] = prediction

            # Store results
            all_predictions[model_name] = prediction
            all_metadata[model_name] = metadata

            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {metadata.get('confidence', 'N/A')}")
        else:
            print(f"  SKIPPED: {metadata.get('error', 'Unknown error')}")
            all_metadata[model_name] = metadata

    # Prepare final result
    result = {
        'input_features': features,
        'pipeline_order': PIPELINE_ORDER,
        'predictions': all_predictions,
        'metadata': all_metadata,
        'model_directory': model_dir
    }

    return result


def main():
    """Main inference function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Complete ML pipeline inference for prepaid anomaly detection'
    )

    # Base input features (all raw features needed)
    parser.add_argument('--avg_12m_pemakaian', type=float, required=True,
                        help='Average consumption over 12 months (kWh)')
    parser.add_argument('--std_12m_pemakaian', type=float, required=True,
                        help='Standard deviation of consumption over 12 months')
    parser.add_argument('--freq_tx_12m', type=int, required=True,
                        help='Frequency of transactions in 12 months')
    parser.add_argument('--avg_gap_days_12m', type=float, required=True,
                        help='Average gap between transactions in days')
    parser.add_argument('--consecutive_anomaly_count', type=int, required=True,
                        help='Number of consecutive anomalies observed')
    parser.add_argument('--pemakaian', type=float, required=True,
                        help='Actual electricity consumption (kWh)')
    parser.add_argument('--avg_pemakaian_gardu', type=float, required=True,
                        help='Average consumption at electrical substation (kWh)')

    args = parser.parse_args()

    # Prepare base features
    features = {
        'avg_12m_pemakaian': args.avg_12m_pemakaian,
        'std_12m_pemakaian': args.std_12m_pemakaian,
        'freq_tx_12m': args.freq_tx_12m,
        'avg_gap_days_12m': args.avg_gap_days_12m,
        'consecutive_anomaly_count': args.consecutive_anomaly_count,
        'pemakaian': args.pemakaian,
        'avg_pemakaian_gardu': args.avg_pemakaian_gardu
    }

    print("="*60)
    print("PREPAID ANOMALY DETECTION - PIPELINE INFERENCE")
    print("="*60)
    print("\nInput Features:")
    for feat, val in features.items():
        print(f"  - {feat}: {val}")

    # Run pipeline
    result = run_pipeline(features)

    # Print summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)

    print("\nPredictions:")
    for model_name in PIPELINE_ORDER:
        if model_name in result['predictions']:
            prediction = result['predictions'][model_name]
            metadata = result['metadata'][model_name]
            print(f"\n  {model_name}:")
            print(f"    Prediction: {prediction}")

            if 'probabilities' in metadata:
                print(f"    Probabilities: {metadata['probabilities']}")
                print(f"    Confidence: {metadata['confidence']}")

            if 'task' in metadata:
                print(f"    Task: {metadata['task']}")

    # Final interpretation
    print("\n" + "="*60)
    print("FINAL INTERPRETATION")
    print("="*60)

    if 'is_rutin' in result['predictions']:
        is_rutin = result['predictions']['is_rutin']
        print(f"\nCustomer Type: {'REGULAR' if is_rutin == 1 else 'IRREGULAR'}")

    if 'anomaly_score' in result['predictions']:
        anomaly_score = result['predictions']['anomaly_score']
        print(f"Final Anomaly Score: {anomaly_score} (0-100)")

        if anomaly_score < 20:
            print("Status: NORMAL - No significant anomaly detected")
        elif anomaly_score < 40:
            print("Status: LOW ANOMALY - Minor deviation, monitor closely")
        elif anomaly_score < 60:
            print("Status: MEDIUM ANOMALY - Investigation recommended")
        elif anomaly_score < 80:
            print("Status: HIGH ANOMALY - Investigation strongly recommended")
        else:
            print("Status: CRITICAL ANOMALY - Immediate action required")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
