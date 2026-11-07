"""
Training script for LSTM power forecaster.
Uses sys.path insertion — no relative imports.
"""

import argparse
import sys
import os

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from services.preprocessor import load_and_preprocess
from services.feature_engineer import engineer_features
from models.lstm_forecaster import LSTMForecaster
from utils.helpers import normalize_series, create_sequences
from utils.logger import get_logger

logger  = get_logger("train_forecaster")
SAVE_DIR = Path(BACKEND_DIR) / "saved_models"
SAVE_DIR.mkdir(exist_ok=True)

FORECASTER_FEATURE_COLS = [
    'total_power', 'current',
    'rolling_mean_10s', 'rolling_std_10s',
    'rolling_mean_60s', 'rolling_std_60s',
    'rate_of_change', 'lag_1', 'lag_2', 'lag_5',
    'load_factor', 'cusum_score'
]


def get_available_features(df, desired_cols):
    available = [c for c in desired_cols if c in df.columns]
    missing   = [c for c in desired_cols if c not in df.columns]
    if missing:
        logger.warning(f"Skipping missing features: {missing}")
    logger.info(f"Using {len(available)} features: {available}")
    return available


def train_forecaster(
    csv_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    seq_len: int = 60,
    pred_len: int = 10,
    train_split: float = 0.8,
    voltage: float = 230.0
):
    logger.info(f"=== Training Forecaster === CSV: {csv_path}")

    # ── 1. Load & engineer ───────────────────────────────────
    df, appliance_cols = load_and_preprocess(
        csv_path, resample_freq='5s', voltage=voltage
    )
    df = engineer_features(df, voltage=voltage)

    feature_cols = get_available_features(df, FORECASTER_FEATURE_COLS)
    n_features   = len(feature_cols)

    # ── 2. Normalise power column ────────────────────────────
    features = df[feature_cols].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    power_col        = features[:, 0].copy()
    norm_power, p_min, p_max = normalize_series(power_col)
    features[:, 0]   = norm_power
    features         = np.clip(features, -5.0, 5.0)

    # ── 3. Sequences ─────────────────────────────────────────
    X, y = create_sequences(features, seq_len=seq_len, pred_len=pred_len)
    logger.info(f"Sequences — X: {X.shape}  y: {y.shape}")

    if len(X) < 10:
        raise ValueError(
            f"Too few sequences ({len(X)}). "
            "Use a larger CSV or reduce seq_len."
        )

    # ── 4. Split ─────────────────────────────────────────────
    split    = int(len(X) * train_split)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True
    )

    # ── 5. Model ─────────────────────────────────────────────
    model     = LSTMForecaster(
        n_features=n_features, hidden_size=128,
        num_layers=2, pred_len=pred_len
    )
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion  = nn.MSELoss()

    best_val_loss    = float('inf')
    patience_counter = 0
    early_stop       = 10

    # ── 6. Training loop ─────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(X_test))
            val_loss = criterion(val_pred, torch.from_numpy(y_test)).item()
        model.train()

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train: {total_loss/max(n_batches,1):.6f} | "
                f"Val: {val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':         val_loss,
                'feature_cols':     feature_cols,
                'n_features':       n_features,
                'seq_len':          seq_len,
                'pred_len':         pred_len,
                'norm_params':      {'min': float(p_min), 'max': float(p_max)}
            }, SAVE_DIR / "forecaster.pt")
        else:
            patience_counter += 1
            if patience_counter >= early_stop:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # ── 7. Evaluate ──────────────────────────────────────────
    checkpoint = torch.load(SAVE_DIR / "forecaster.pt", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        test_pred = model(torch.from_numpy(X_test)).numpy()

    test_pred_w = test_pred * (p_max - p_min) + p_min
    y_test_w    = y_test    * (p_max - p_min) + p_min

    rmse = float(np.sqrt(np.mean((test_pred_w - y_test_w) ** 2)))
    mae  = float(np.mean(np.abs(test_pred_w - y_test_w)))

    logger.info("=== FORECASTER EVALUATION ===")
    logger.info(f"RMSE : {rmse:.2f} W")
    logger.info(f"MAE  : {mae:.2f} W")
    logger.info(f"Model saved: {SAVE_DIR / 'forecaster.pt'}")
    logger.info(f"n_features : {n_features}")

    return rmse, mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Power Forecaster")
    parser.add_argument('--csv',        type=str,   required=True)
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--seq_len',    type=int,   default=60)
    parser.add_argument('--pred_len',   type=int,   default=10)
    parser.add_argument('--voltage',    type=float, default=230.0)
    args = parser.parse_args()

    train_forecaster(
        csv_path   = args.csv,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        seq_len    = args.seq_len,
        pred_len   = args.pred_len,
        voltage    = args.voltage,
    )