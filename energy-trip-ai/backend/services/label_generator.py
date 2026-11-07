"""
Breaker simulation and label generation.

Two Modes Supported:

1) Physics Mode (strict electrical thresholds)
2) Quantile Mode (adaptive thresholds for ML stability)

Labels:
    0 = Normal
    1 = Warning
    2 = Trip
"""

import pandas as pd
import numpy as np
from typing import Dict
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_labels(
    df: pd.DataFrame,
    rated_current: float,
    voltage: float = 230.0,
    mode: str = "quantile"   # 🔥 default changed to quantile
) -> pd.DataFrame:
    """
    Generate trip labels using either:
        - "physics"  → strict breaker thresholds
        - "quantile" → adaptive percentiles (recommended for ML)

    Args:
        df: DataFrame containing 'total_power'
        rated_current: breaker current rating
        voltage: system voltage
        mode: "physics" or "quantile"

    Returns:
        DataFrame with label columns added
    """

    logger.info(
        f"Generating labels | mode={mode} | rated_current={rated_current}A | voltage={voltage}V"
    )

    if mode == "physics":
        # ---------------------------
        # Strict breaker thresholds
        # ---------------------------
        trip_power = voltage * rated_current
        warning_power = 0.8 * trip_power

        logger.info(
            f"[Physics Mode] Trip threshold={trip_power:.2f}W | "
            f"Warning threshold={warning_power:.2f}W"
        )

    elif mode == "quantile":
        # ---------------------------
        # Adaptive ML-friendly thresholds
        # ---------------------------
        trip_power = df['total_power'].quantile(0.95)
        warning_power = df['total_power'].quantile(0.85)

        logger.info(
            f"[Quantile Mode] Trip threshold={trip_power:.2f}W (95th pct) | "
            f"Warning threshold={warning_power:.2f}W (85th pct)"
        )

    else:
        raise ValueError("mode must be either 'physics' or 'quantile'")

    # ---------------------------
    # Compute current demand
    # ---------------------------
    df['current_demand'] = df['total_power'] / voltage

    # Store thresholds
    df['trip_power_threshold'] = trip_power
    df['warning_power_threshold'] = warning_power

    # ---------------------------
    # Assign labels
    # ---------------------------
    conditions = [
        df['total_power'] > trip_power,        # Trip
        df['total_power'] > warning_power,     # Warning
    ]

    choices = [2, 1]

    df['label'] = np.select(conditions, choices, default=0)

    # ---------------------------
    # Guarantee all 3 classes exist (safety)
    # ---------------------------
    present_classes = df['label'].unique()

    if len(present_classes) < 3:
        logger.warning(
            "Less than 3 classes detected. Forcing balanced split via quantiles."
        )

        # Force 3-class distribution safely
        q_trip = df['total_power'].quantile(0.90)
        q_warn = df['total_power'].quantile(0.75)

        df['label'] = np.select(
            [
                df['total_power'] > q_trip,
                df['total_power'] > q_warn,
            ],
            [2, 1],
            default=0
        )

    # Label names
    label_map = {0: 'Normal', 1: 'Warning', 2: 'Trip'}
    df['label_name'] = df['label'].map(label_map)

    # Log distribution
    counts = df['label'].value_counts().sort_index()
    logger.info(f"Label distribution:\n{counts}")

    return df


def get_label_stats(df: pd.DataFrame) -> Dict:
    """
    Return label statistics dictionary.
    """
    total = len(df)
    counts = df['label'].value_counts().sort_index()

    return {
        "total_samples": total,
        "normal_count": int(counts.get(0, 0)),
        "warning_count": int(counts.get(1, 0)),
        "trip_count": int(counts.get(2, 0)),
        "normal_pct": round(100 * counts.get(0, 0) / total, 2),
        "warning_pct": round(100 * counts.get(1, 0) / total, 2),
        "trip_pct": round(100 * counts.get(2, 0) / total, 2),
    }