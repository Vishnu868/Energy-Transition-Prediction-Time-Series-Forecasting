"""
Time-series preprocessing pipeline for UK-DALE and generic appliance CSV files.

Optimized for large datasets and LSTM training.

Handles:
- Timestamp parsing (Unix or datetime string)
- Deduplication
- Resampling to uniform intervals
- Missing value imputation
- Multi-appliance vs single-column detection
- Dataset size limiting for ML stability
"""

import os, sys
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import pandas as pd
import numpy as np
from typing import Tuple, List
from utils.logger import get_logger          # absolute, no dots

logger = get_logger(__name__)


def load_and_preprocess(
    filepath: str,
    resample_freq: str = "1min",   # 🔥 changed default from 1s → 1min
    voltage: float = 230.0,
    max_rows: int = 50000         # 🔥 limit dataset size for ML stability
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Main preprocessing entry point.

    Args:
        filepath: Path to uploaded CSV file
        resample_freq: Pandas resample frequency ('1min', '5min', etc.)
        voltage: Assumed voltage if not present in data
        max_rows: Maximum rows to keep after resampling

    Returns:
        df: Preprocessed DataFrame with DatetimeIndex
        appliance_cols: List of appliance column names detected
    """

    logger.info(f"Loading CSV from {filepath}")

    # --- Step 1: Read CSV ---
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Cannot read CSV: {e}")

    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # --- Step 2: Parse timestamps ---
    df = _parse_timestamps(df)

    # --- Step 3: Deduplicate ---
    df = df[~df.index.duplicated(keep='first')]
    logger.info(f"After dedup: {len(df)} rows")

    # --- Step 4: Detect appliance columns ---
    appliance_cols = _detect_appliance_columns(df)
    logger.info(f"Detected appliance columns: {appliance_cols}")

    # --- Step 5: Ensure numeric ---
    for col in appliance_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Step 6: Resample ---
    df = df[appliance_cols].resample(resample_freq).mean()
    logger.info(f"After resampling to {resample_freq}: {len(df)} rows")

    # --- Step 7: Limit dataset size (VERY IMPORTANT) ---
    if len(df) > max_rows:
        df = df.iloc[-max_rows:]
        logger.info(f"Dataset limited to last {max_rows} rows")

    # --- Step 8: Interpolation ---
    df = df.interpolate(method='linear', limit=10)
    df = df.bfill().fillna(0.0)   # 🔥 updated (no deprecated method)

    # --- Step 9: Add voltage if missing ---
    if 'voltage' not in [c.lower() for c in df.columns]:
        df['voltage'] = voltage
        logger.info(f"No voltage column found. Using assumed voltage = {voltage}V")

    # --- Step 10: Compute total power ---
    if len(appliance_cols) > 1:
        df['total_power'] = df[appliance_cols].sum(axis=1)
        logger.info("Computed total_power as sum of appliances")
    else:
        df['total_power'] = df[appliance_cols[0]]

    # --- Step 11: Remove negative noise ---
    df[df < 0] = 0.0

    logger.info("Preprocessing complete")
    return df, appliance_cols


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect timestamp column and convert to DatetimeIndex.
    Supports:
    - Unix timestamps
    - ISO datetime strings
    """

    ts_candidates = ['timestamp', 'time', 'datetime', 'unix_time', 'ts', 'date']

    ts_col = None
    for col in df.columns:
        if col.lower() in ts_candidates:
            ts_col = col
            break

    if ts_col is None:
        ts_col = df.columns[0]
        logger.warning(f"No obvious timestamp column. Using first column: '{ts_col}'")

    sample_val = df[ts_col].iloc[0]

    try:
        # Unix timestamp detection
        if isinstance(sample_val, (int, float)) or str(sample_val).isdigit():
            df.index = pd.to_datetime(df[ts_col].astype(float), unit='s')
            logger.info("Parsed Unix timestamps")
        else:
            df.index = pd.to_datetime(df[ts_col])
            logger.info("Parsed datetime string timestamps")
    except Exception as e:
        raise ValueError(f"Cannot parse timestamps from column '{ts_col}': {e}")

    df = df.drop(columns=[ts_col])
    df.index.name = 'datetime'
    return df


def _detect_appliance_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify numeric power columns.
    Excludes voltage and timestamp-like columns.
    """

    exclude_keywords = ['voltage', 'volt', 'v_rms', 'datetime', 'time', 'index']

    power_cols = []
    for col in df.columns:
        if col.lower() in exclude_keywords:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            power_cols.append(col)

    if not power_cols:
        raise ValueError("No numeric power columns detected in CSV.")

    return power_cols