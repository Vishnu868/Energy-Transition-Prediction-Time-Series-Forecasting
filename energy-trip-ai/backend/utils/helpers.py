"""
Utility helper functions used across multiple modules.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


def normalize_series(series: np.ndarray) -> tuple:
    """
    Min-max normalize a numpy array.
    
    Returns:
        (normalized_array, min_val, max_val) for later denormalization
    """
    min_val = series.min()
    max_val = series.max()
    
    if max_val - min_val == 0:
        return np.zeros_like(series), min_val, max_val
    
    normalized = (series - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def denormalize(value: float, min_val: float, max_val: float) -> float:
    """Reverse min-max normalization."""
    return value * (max_val - min_val) + min_val


def create_sequences(data: np.ndarray, seq_len: int, pred_len: int = 1) -> tuple:
    """
    Convert time-series array into supervised learning sequences.
    
    Args:
        data: 2D array of shape (timesteps, features)
        seq_len: Number of past timesteps as input (lookback window)
        pred_len: Number of future timesteps to predict
    
    Returns:
        X: shape (samples, seq_len, features)
        y: shape (samples, pred_len)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_len - pred_len + 1):
        # Input window: past seq_len timesteps
        X.append(data[i : i + seq_len])
        # Target: next pred_len power values (first feature assumed to be power)
        y.append(data[i + seq_len : i + seq_len + pred_len, 0])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def create_classification_sequences(features: np.ndarray, labels: np.ndarray, seq_len: int) -> tuple:
    """
    Create sequences paired with classification labels.
    
    Args:
        features: 2D array (timesteps, n_features)
        labels: 1D array of integer labels (0=Normal, 1=Warning, 2=Trip)
        seq_len: Lookback window size
    
    Returns:
        X: (samples, seq_len, n_features)
        y: (samples,) integer labels
    """
    X, y = [], []
    
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        # Label for this window = label at the END of the window
        y.append(labels[i + seq_len])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def compute_stability_score(trip_prob: float, load_factor: float, rate_of_change: float) -> float:
    """
    Compute a 0-100 stability score where 100 = perfectly stable.
    
    Combines:
    - Trip probability (higher = less stable)
    - Load factor relative to capacity (higher = less stable)
    - Rate of change (higher = less stable)
    
    Args:
        trip_prob: Probability of trip [0,1]
        load_factor: Current load / rated capacity [0,1+]
        rate_of_change: Normalized rate of power change [0,1]
    
    Returns:
        Stability score [0, 100]
    """
    # Each component penalizes stability
    trip_penalty = trip_prob * 40          # Max 40 points penalty
    load_penalty = min(load_factor, 1.5) * 30  # Max 45 points penalty
    roc_penalty = min(rate_of_change, 1.0) * 15  # Max 15 points penalty
    
    score = 100 - trip_penalty - load_penalty - roc_penalty
    return max(0.0, min(100.0, score))