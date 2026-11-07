"""
Training script for LSTM classifier.
Uses sys.path insertion for absolute imports — works when run as:
    python -m training.train_classifier ...
    OR
    python training/train_classifier.py ...
"""

import argparse
import sys
import os

# ── Fix imports: insert backend/ into path BEFORE any local imports ──
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from pathlib import Path

# Now use ABSOLUTE imports (no leading dots)
from services.preprocessor import load_and_preprocess
from services.feature_engineer import engineer_features
from services.label_generator import generate_labels
from models.lstm_classifier import LSTMClassifier
from utils.helpers import create_classification_sequences
from utils.logger import get_logger

logger = get_logger("train_classifier")

SAVE_DIR = Path(BACKEND_DIR) / "saved_models"
SAVE_DIR.mkdir(exist_ok=True)

# ── Single source of truth for feature columns ───────────────
CLASSIFIER_FEATURE_COLS = [
    'total_power', 'current',
    'rolling_mean_10s', 'rolling_std_10s',
    'rolling_mean_60s', 'rolling_std_60s',
    'rate_of_change', 'rate_of_change_abs', 'rate_of_change_2nd',
    'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10',
    'load_factor', 'peak_to_avg_ratio',
    'cusum_score', 'adf_pvalue', 'non_stationary_flag'
]


def get_available_features(df, desired_cols):
    """Return only columns that actually exist in df, warn on missing."""
    available = [c for c in desired_cols if c in df.columns]
    missing   = [c for c in desired_cols if c not in df.columns]
    if missing:
        logger.warning(f"Skipping missing features: {missing}")
    logger.info(f"Using {len(available)} features: {available}")
    return available


def train_classifier(
    csv_path: str,
    breaker_amps: float = 15.0,
    voltage: float = 230.0,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    seq_len: int = 10,
    train_split: float = 0.8
):
    logger.info(f"=== Training Classifier === breaker={breaker_amps}A, voltage={voltage}V")

    # ── 1. Load & preprocess ────────────────────────────────
    df, appliance_cols = load_and_preprocess(
        csv_path, resample_freq='5s', voltage=voltage
    )
    df = engineer_features(df, voltage=voltage)
    df = generate_labels(df, rated_current=breaker_amps, voltage=voltage)

    # ── 2. Resolve available features ───────────────────────
    feature_cols = get_available_features(df, CLASSIFIER_FEATURE_COLS)
    n_features   = len(feature_cols)

    # ── 3. Build arrays ─────────────────────────────────────
    features = df[feature_cols].values.astype(np.float32)
    labels   = df['label'].values.astype(np.int64)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -10.0, 10.0)

    # ── 4. Create sequences ─────────────────────────────────
    X, y = create_classification_sequences(features, labels, seq_len=seq_len)
    logger.info(f"Sequences — X: {X.shape}  y: {y.shape}")
    logger.info(f"Class distribution: {np.bincount(y)}")

    if len(X) < 10:
        raise ValueError(
            f"Too few sequences ({len(X)}) to train. "
            "Use a larger CSV or reduce seq_len / resample_freq."
        )

    # ── 5. Train / test split ────────────────────────────────
    split    = int(len(X) * train_split)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ── 6. Weighted sampler for class imbalance ──────────────
    class_counts     = np.bincount(y_train, minlength=3)
    weights_per_class = 1.0 / (class_counts.astype(float) + 1e-6)
    sample_weights   = torch.tensor(
        [weights_per_class[lbl] for lbl in y_train], dtype=torch.float
    )
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        sampler=sampler
    )

    # ── 7. Model, loss, optimiser ────────────────────────────
    model      = LSTMClassifier(n_features=n_features, hidden_size=64, num_layers=2)
    cw         = torch.tensor(weights_per_class, dtype=torch.float32)
    criterion  = nn.CrossEntropyLoss(weight=cw)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    # ── 8. Training loop ─────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss   = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_test))
            val_pred   = val_logits.argmax(dim=1).numpy()
            val_acc    = (val_pred == y_test).mean()

        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Loss: {total_loss/max(n_batches,1):.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        # Save best checkpoint — store feature metadata so inference never mismatches
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_accuracy':     val_acc,
                'feature_cols':     feature_cols,   # ← exact list used in training
                'n_features':       n_features,     # ← exact count used in training
                'seq_len':          seq_len,
            }, SAVE_DIR / "classifier.pt")

    # ── 9. Final evaluation ──────────────────────────────────
    checkpoint = torch.load(SAVE_DIR / "classifier.pt", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        test_pred = model(torch.from_numpy(X_test)).argmax(dim=1).numpy()

    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.utils.multiclass import unique_labels

    # Dynamically build target_names for only the classes present in test set
    # This prevents the crash when a rare class (e.g. Trip) has no test samples
    ALL_CLASS_NAMES = {0: 'Normal', 1: 'Warning', 2: 'Trip'}
    present_labels  = sorted(unique_labels(y_test, test_pred))
    present_names   = [ALL_CLASS_NAMES[l] for l in present_labels]

    logger.info("=== FINAL EVALUATION ===")
    logger.info(
        f"\n{classification_report(y_test, test_pred, labels=present_labels, target_names=present_names, zero_division=0)}"
    )
    logger.info(f"Confusion Matrix (labels={present_labels}):\n{confusion_matrix(y_test, test_pred, labels=present_labels)}")
    logger.info(f"Best Val Accuracy : {best_val_acc:.4f}")
    logger.info(f"Model saved       : {SAVE_DIR / 'classifier.pt'}")
    logger.info(f"n_features saved  : {n_features}")
    logger.info(f"feature_cols saved: {feature_cols}")

    return best_val_acc