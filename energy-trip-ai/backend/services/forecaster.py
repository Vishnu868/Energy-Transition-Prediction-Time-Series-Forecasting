"""
Forecasting service — fixed to read n_features from checkpoint.
"""

import os, sys
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from models.lstm_forecaster import LSTMForecaster
from utils.helpers import normalize_series
from utils.logger import get_logger

logger    = get_logger(__name__)
MODEL_PATH = Path(BACKEND_DIR) / "saved_models" / "forecaster.pt"


def _load_forecaster_from_checkpoint(fallback_n_features: int, pred_len: int = 10):
    """Load forecaster, reading architecture from checkpoint."""
    if not MODEL_PATH.exists():
        logger.warning("No saved forecaster — using untrained model.")
        return LSTMForecaster(n_features=fallback_n_features, pred_len=pred_len), None, None, None

    checkpoint = torch.load(MODEL_PATH, map_location='cpu')

    feature_cols = checkpoint.get('feature_cols', None)
    pred_len_ckpt = checkpoint.get('pred_len', pred_len)
    norm_params = checkpoint.get('norm_params', None)

    # Infer n_features from weight matrix
    state = checkpoint['model_state_dict']
    n_features = state['lstm.weight_ih_l0'].shape[1]

    model = LSTMForecaster(n_features=n_features, pred_len=pred_len_ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Forecaster loaded: n_features={n_features}, pred_len={pred_len_ckpt}")
    return model, feature_cols, pred_len_ckpt, norm_params


def _align_features(df: pd.DataFrame, trained_cols: List[str]) -> np.ndarray:
    aligned = pd.DataFrame(index=df.index)
    for col in trained_cols:
        if col in df.columns:
            aligned[col] = df[col]
        else:
            logger.warning(f"Forecaster: feature '{col}' missing — filling 0")
            aligned[col] = 0.0
    return aligned.values.astype(np.float32)


def run_forecast(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 60,
    pred_len: int = 10
) -> Tuple[List[float], List[float]]:
    """
    Multi-step power forecast. Architecture and features read from checkpoint.
    """
    model, trained_cols, pred_len_ckpt, norm_params = _load_forecaster_from_checkpoint(
        fallback_n_features=len(feature_cols),
        pred_len=pred_len
    )

    # Use checkpoint feature list if available
    if trained_cols is not None:
        features = _align_features(df, trained_cols)
    else:
        features = df[feature_cols].values.astype(np.float32)

    # Normalize power column
    power_col = features[:, 0].copy()
    if norm_params:
        p_min, p_max = norm_params['min'], norm_params['max']
        norm_power = (power_col - p_min) / (p_max - p_min + 1e-9)
    else:
        norm_power, p_min, p_max = normalize_series(power_col)
    features[:, 0] = norm_power

    features = np.nan_to_num(features, nan=0.0)
    features = np.clip(features, -5.0, 5.0)

    if len(features) < seq_len:
        logger.warning(f"Not enough data ({len(features)} rows, need {seq_len})")
        return list(df['total_power'].tail(seq_len).values), [0.0] * pred_len_ckpt

    input_window = features[-seq_len:]
    input_tensor = torch.tensor(input_window).unsqueeze(0)

    with torch.no_grad():
        pred_norm = model(input_tensor).squeeze(0).numpy()

    pred_watts = [max(0.0, float(v) * (p_max - p_min) + p_min) for v in pred_norm]
    historical = list(df['total_power'].tail(seq_len).round(2).values)

    logger.info(f"Forecast: {[round(p, 1) for p in pred_watts]}")
    return historical, pred_watts