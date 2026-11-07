"""
Feature engineering for time-series power data.

Implements all required time-series concepts:
- Rolling mean, std, rate of change
- Lag features
- Peak detection
- Load factor, peak-to-average ratio
- Change point detection (CUSUM)
- Stationarity check (ADF test)
"""

import os, sys
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import pandas as pd
import numpy as np
from typing import Tuple
from statsmodels.tsa.stattools import adfuller
from scipy.signal import find_peaks
from utils.logger import get_logger

logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame, voltage: float = 230.0) -> pd.DataFrame:
    """
    Add all engineered features to the preprocessed DataFrame.
    
    Args:
        df: Preprocessed DataFrame with 'total_power' column
        voltage: System voltage for current calculation
    
    Returns:
        DataFrame with additional feature columns
    """
    logger.info("Starting feature engineering")
    
    power = df['total_power'].copy()
    
    # --- Current calculation: I(t) = P(t) / V ---
    df['current'] = power / voltage
    
    # --- Rolling statistics (window=60 seconds) ---
    for window in [10, 60, 300]:
        df[f'rolling_mean_{window}s'] = power.rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}s'] = power.rolling(window=window, min_periods=1).std().fillna(0)
    
    # --- Rate of change (first derivative) ---
    df['rate_of_change'] = power.diff().fillna(0)
    df['rate_of_change_abs'] = df['rate_of_change'].abs()
    
    # Second derivative (acceleration of power change)
    df['rate_of_change_2nd'] = df['rate_of_change'].diff().fillna(0)
    
    # --- Lag features (past values as features) ---
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'lag_{lag}'] = power.shift(lag).fillna(0)
    
    # --- Load factor = mean power / peak power (in rolling window) ---
    rolling_max = power.rolling(window=60, min_periods=1).max()
    rolling_mean = power.rolling(window=60, min_periods=1).mean()
    df['load_factor'] = (rolling_mean / (rolling_max + 1e-6)).clip(0, 1)
    
    # --- Peak-to-average ratio ---
    overall_mean = power.mean()
    df['peak_to_avg_ratio'] = power / (overall_mean + 1e-6)
    
    # --- Cumulative energy (Watt-seconds) ---
    df['cumulative_energy'] = power.cumsum()
    
    # --- Change point score using CUSUM ---
    df['cusum_score'] = _compute_cusum(power.values)
    
    # --- Stationarity: ADF test on rolling windows ---
    df['adf_pvalue'] = _rolling_adf(power.values, window=120)
    
    # Interpretation: low p-value means stationary (stable), high = non-stationary (unstable)
    df['non_stationary_flag'] = (df['adf_pvalue'] > 0.05).astype(int)
    
    logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
    return df


def _compute_cusum(values: np.ndarray, threshold: float = None) -> np.ndarray:
    """
    CUSUM (Cumulative Sum) change point detection.
    
    Detects sustained shifts in the mean of the time series.
    High CUSUM score indicates a change point (possible trip precursor).
    
    Args:
        values: 1D power array
        threshold: Detection threshold (default = 5 * std)
    
    Returns:
        cusum_scores: 1D array, high values indicate change points
    """
    mean = np.mean(values)
    std = np.std(values) + 1e-6
    
    if threshold is None:
        threshold = 5.0 * std
    
    cusum_pos = np.zeros(len(values))
    cusum_neg = np.zeros(len(values))
    
    for i in range(1, len(values)):
        # Positive CUSUM: detect upward shift
        cusum_pos[i] = max(0, cusum_pos[i-1] + (values[i] - mean) - 0.5 * std)
        # Negative CUSUM: detect downward shift
        cusum_neg[i] = max(0, cusum_neg[i-1] - (values[i] - mean) - 0.5 * std)
    
    # Combined score normalized by threshold
    combined = (cusum_pos + cusum_neg) / (threshold + 1e-6)
    return np.clip(combined, 0, 1)


def _rolling_adf(values: np.ndarray, window: int = 120) -> np.ndarray:
    """
    Compute rolling Augmented Dickey-Fuller test p-values.
    
    A high p-value (>0.05) means non-stationary = power is drifting = unstable.
    
    Args:
        values: 1D power array
        window: Rolling window size for ADF test
    
    Returns:
        pvalues: 1D array of ADF p-values
    """
    n = len(values)
    pvalues = np.ones(n)  # Default p=1 (assume non-stationary until enough data)
    
    for i in range(window, n):
        segment = values[i - window : i]
        try:
            result = adfuller(segment, autolag='AIC')
            pvalues[i] = result[1]  # p-value
        except Exception:
            pvalues[i] = 1.0  # Assume non-stationary on failure
    
    return pvalues


def get_peak_timestamps(df: pd.DataFrame, prominence: float = 100.0) -> pd.DatetimeIndex:
    """
    Detect peaks in total power using scipy's find_peaks.
    
    Args:
        df: DataFrame with 'total_power'
        prominence: Minimum peak prominence in Watts
    
    Returns:
        DatetimeIndex of peak timestamps
    """
    peaks, _ = find_peaks(df['total_power'].values, prominence=prominence)
    return df.index[peaks]