"""
Shared utilities for the project
"""

import os
import pandas as pd
import joblib


def load_training_data(data_path: str) -> pd.DataFrame:
    """
    Load training data from CSV file.

    Args:
        data_path: Path to training CSV file (semicolon-separated)

    Returns:
        DataFrame with training data
    """
    df = pd.read_csv(data_path, sep=';')
    return df


def save_model(model, model_dir: str, model_name: str) -> tuple:
    """
    Save trained model to disk.

    Args:
        model: Trained model object
        model_dir: Directory to save the model
        model_name: Name of the model file (without extension)

    Returns:
        Tuple of (model_path, file_size_kb)
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    joblib.dump(model, model_path)

    # Get file size
    file_size = os.path.getsize(model_path)
    file_size_kb = round(file_size / 1024, 2)
    
    print("Model saved")

    return model_path, file_size_kb
