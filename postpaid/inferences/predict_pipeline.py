"""
Complete ML Pipeline Inference for Anomaly Detection
Runs all models sequentially according to PIPELINE_ORDER

Usage:
  python postpaid/inferences/predict_pipeline.py \\
    --pemakaian 41.0 \\
    --baseline 40.0 \\
    --avg_pemakaian_gardu 60.0 \\
    --threshold_drop_consume 30.0 \\
    --consecutive_anomaly_count 2
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

    # Fallback
    classes = MODEL_CONFIGS[model_name].get('classes', [])
    return {str(i): label for i, label in enumerate(classes)}


def compute_derived_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Compute all derived features from base features."""
    result = features.copy()

    # Derived features for anomaly_score
    if 'pemakaian' in features and 'baseline' in features:
        result['ratio'] = features['pemakaian'] / (features['baseline'] + 1)
        result['diff'] = features['baseline'] - features['pemakaian']
        result['is_over_baseline'] = 1 if features['pemakaian'] > features['baseline'] else 0

    # Derived features for score_rumus2
    if 'pemakaian' in features and 'avg_pemakaian_gardu' in features:
        result['ratio_gardu'] = features['pemakaian'] / (features['avg_pemakaian_gardu'] + 1)
        result['pct_diff_gardu'] = (features['pemakaian'] - features['avg_pemakaian_gardu']) / (features['avg_pemakaian_gardu'] + 1)

    return result


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
        features: Dictionary of base features
                 Example: {
                     'pemakaian': 41.0,
                     'baseline': 40.0,
                     'avg_pemakaian_gardu': 60.0,
                     'threshold_drop_consume': 30.0,
                     'consecutive_anomaly_count': 2
                 }
        model_dir: Directory containing trained models (optional)

    Returns:
        Dictionary containing predictions from all pipeline stages
    """
    if model_dir is None:
        current_dir = Path(__file__).parent.parent
        model_dir = os.path.join(current_dir, 'models', MODEL_VERSION)

    # Compute all derived features
    all_features = compute_derived_features(features)

    # Store results for each pipeline stage
    results = {
        'input_features': features,
        'engineered_features': {k: v for k, v in all_features.items() if k not in features},
        'pipeline_stages': {}
    }

    # Track intermediate predictions
    intermediate_predictions = {}

    # Run each model in pipeline order
    for model_name in PIPELINE_ORDER:
        prediction, metadata = run_model_inference(model_name, all_features, model_dir)

        # Store result
        results['pipeline_stages'][model_name] = metadata

        # Add prediction to features for next models
        if prediction is not None:
            all_features[model_name] = prediction
            intermediate_predictions[model_name] = prediction

            results['pipeline_stages'][model_name]['prediction'] = prediction

    # Add final summary
    results['summary'] = {
        'anomaly_detected': intermediate_predictions.get('is_anomaly', False),
        'persistence_score': intermediate_predictions.get('final_score_rumus1', None),
        'gardu_deviation_score': intermediate_predictions.get('final_score_rumus2', None)
    }

    return results


def print_pipeline_results(results: Dict[str, Any]):
    """Print pipeline results in a formatted way."""
    print("="*70)
    print("COMPLETE ML PIPELINE INFERENCE RESULTS")
    print("="*70)

    # Input features
    print("\n📥 INPUT FEATURES:")
    for feat, value in results['input_features'].items():
        print(f"  - {feat}: {value}")

    # Engineered features
    if results['engineered_features']:
        print("\n🔧 ENGINEERED FEATURES:")
        for feat, value in results['engineered_features'].items():
            if isinstance(value, float):
                if 'ratio' in feat or 'pct' in feat:
                    print(f"  - {feat}: {value:.4f}")
                else:
                    print(f"  - {feat}: {value:.2f}")
            else:
                print(f"  - {feat}: {value}")

    # Pipeline stages
    print("\n🔄 PIPELINE STAGES:")
    for model_name, metadata in results['pipeline_stages'].items():
        if 'error' in metadata:
            print(f"\n  {model_name}:")
            print(f"    ⚠️  {metadata['error']}")
            continue

        task = metadata['task']
        print(f"\n  {model_name} ({task}):")

        if task == 'regression':
            pred = metadata.get('prediction', 'N/A')
            print(f"    Prediction: {pred}")

        elif task in ['binary_classification', 'multiclass_classification', 'ordinal_classification']:
            label = metadata.get('predicted_label', 'N/A')
            conf = metadata.get('confidence', 'N/A')
            print(f"    Prediction: {label}")
            print(f"    Confidence: {conf}")

            if metadata.get('probabilities'):
                print(f"    Probabilities:")
                for class_label, prob in metadata['probabilities'].items():
                    print(f"      - {class_label}: {prob:.2%}")

    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY:")
    print("="*70)

    summary = results['summary']

    # Anomaly detection status
    is_anomaly = summary.get('anomaly_detected', False)
    print(f"\nAnomaly Detected: {'🔴 YES' if is_anomaly else '🟢 NO'}")

    # Scores
    persistence_score = summary.get('persistence_score')
    if persistence_score is not None:
        print(f"Persistence Score (Rumus1): {persistence_score}")

    gardu_score = summary.get('gardu_deviation_score')
    if gardu_score is not None:
        print(f"Gardu Deviation Score (Rumus2): {gardu_score}")

    # Recommendations
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS:")
    print("="*70)

    if not is_anomaly:
        print("\n✅ No anomaly detected - Continue routine monitoring")
    else:
        print("\n⚠️  Anomaly detected - Investigation recommended")
        print("   - Review consumption patterns")
        print("   - Check persistence score and gardu deviation")

        if persistence_score is not None and persistence_score >= 40:
            print("   - High persistence score indicates consistent anomaly")

        if gardu_score is not None and gardu_score >= 30:
            print("   - High gardu deviation detected - investigate area")

    print("\n" + "="*70)


def main():
    """Main inference function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Run complete ML pipeline for anomaly detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python postpaid/inferences/predict_pipeline.py \\
    --pemakaian 41.0 \\
    --baseline 40.0 \\
    --avg_pemakaian_gardu 60.0 \\
    --threshold_drop_consume 30.0 \\
    --consecutive_anomaly_count 2
        """
    )

    # Required base features
    parser.add_argument('--pemakaian', type=float, required=True,
                       help='Actual electricity consumption (kWh)')
    parser.add_argument('--baseline', type=float, required=True,
                       help='Baseline/expected electricity consumption (kWh)')
    parser.add_argument('--avg_pemakaian_gardu', type=float, required=True,
                       help='Average electricity consumption at gardu level (kWh)')
    parser.add_argument('--threshold_drop_consume', type=float, required=True,
                       help='Threshold for consumption drop detection')
    parser.add_argument('--consecutive_anomaly_count', type=int, required=True,
                       help='Number of consecutive anomalies detected')

    args = parser.parse_args()

    # Prepare input features
    features = {
        'pemakaian': args.pemakaian,
        'baseline': args.baseline,
        'avg_pemakaian_gardu': args.avg_pemakaian_gardu,
        'threshold_drop_consume': args.threshold_drop_consume,
        'consecutive_anomaly_count': args.consecutive_anomaly_count
    }

    # Run pipeline
    results = run_pipeline(features)

    # Print results
    print_pipeline_results(results)


if __name__ == "__main__":
    main()
