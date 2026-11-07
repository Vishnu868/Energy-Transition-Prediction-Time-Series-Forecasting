"""
Evaluation script — absolute imports, no relative dots.
"""

import argparse
import sys
import os

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import torch
import numpy as np
from pathlib import Path

from services.preprocessor import load_and_preprocess
from services.feature_engineer import engineer_features
from services.label_generator import generate_labels
from models.lstm_classifier import LSTMClassifier
from models.lstm_forecaster import LSTMForecaster
from utils.helpers import (
    create_classification_sequences, create_sequences, normalize_series
)
from utils.logger import get_logger
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

logger   = get_logger("evaluate")
SAVE_DIR = Path(BACKEND_DIR) / "saved_models"


def evaluate_all(csv_path: str, breaker_amps: float = 15.0, voltage: float = 230.0):

    df, _ = load_and_preprocess(csv_path, resample_freq='5s', voltage=voltage)
    df    = engineer_features(df, voltage=voltage)
    df    = generate_labels(df, rated_current=breaker_amps, voltage=voltage)

    # ── CLASSIFIER ───────────────────────────────────────────
    print("\n" + "="*60)
    print("CLASSIFIER EVALUATION")
    print("="*60)

    clf_path = SAVE_DIR / "classifier.pt"
    if clf_path.exists():
        ckpt       = torch.load(clf_path, map_location='cpu')
        feat_cols  = ckpt.get('feature_cols', [])
        n_feat     = ckpt.get('n_features', len(feat_cols))
        seq_len    = ckpt.get('seq_len', 10)

        # Align features
        aligned = {}
        for col in feat_cols:
            aligned[col] = df[col].values if col in df.columns else np.zeros(len(df))
        import pandas as pd
        feat_df  = pd.DataFrame(aligned, index=df.index)
        features = feat_df.values.astype(np.float32)
        labels   = df['label'].values.astype(np.int64)
        features = np.nan_to_num(features, nan=0.0)
        features = np.clip(features, -10.0, 10.0)

        X, y  = create_classification_sequences(features, labels, seq_len=seq_len)
        split = int(len(X) * 0.8)
        X_test, y_test = X[split:], y[split:]

        model = LSTMClassifier(n_features=n_feat)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        with torch.no_grad():
            y_pred = model(torch.from_numpy(X_test)).argmax(dim=1).numpy()

        print(classification_report(
            y_test, y_pred,
            target_names=['Normal', 'Warning', 'Trip'],
            zero_division=0
        ))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        for avg in ['macro', 'weighted']:
            p = precision_score(y_test, y_pred, average=avg, zero_division=0)
            r = recall_score   (y_test, y_pred, average=avg, zero_division=0)
            f = f1_score       (y_test, y_pred, average=avg, zero_division=0)
            print(f"{avg.capitalize():10s} — P: {p:.4f}  R: {r:.4f}  F1: {f:.4f}")
    else:
        print("No classifier checkpoint found. Run train_classifier first.")

    # ── FORECASTER ───────────────────────────────────────────
    print("\n" + "="*60)
    print("FORECASTER EVALUATION")
    print("="*60)

    fcast_path = SAVE_DIR / "forecaster.pt"
    if fcast_path.exists():
        ckpt       = torch.load(fcast_path, map_location='cpu')
        feat_cols  = ckpt.get('feature_cols', [])
        pred_len   = ckpt.get('pred_len', 10)
        seq_len    = ckpt.get('seq_len', 60)
        norm_p     = ckpt.get('norm_params', None)
        state      = ckpt['model_state_dict']
        n_feat     = state['lstm.weight_ih_l0'].shape[1]

        import pandas as pd
        aligned = {}
        for col in feat_cols:
            aligned[col] = df[col].values if col in df.columns else np.zeros(len(df))
        features = pd.DataFrame(aligned, index=df.index).values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)

        power_col = features[:, 0].copy()
        if norm_p:
            p_min, p_max = norm_p['min'], norm_p['max']
            features[:, 0] = (power_col - p_min) / (p_max - p_min + 1e-9)
        else:
            norm_p2, p_min, p_max = normalize_series(power_col)
            features[:, 0] = norm_p2
        features = np.clip(features, -5.0, 5.0)

        X, y  = create_sequences(features, seq_len=seq_len, pred_len=pred_len)
        split = int(len(X) * 0.8)
        X_test, y_test = X[split:], y[split:]

        model = LSTMForecaster(n_features=n_feat, pred_len=pred_len)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        with torch.no_grad():
            y_pred = model(torch.from_numpy(X_test)).numpy()

        y_pred_w = y_pred  * (p_max - p_min) + p_min
        y_test_w = y_test  * (p_max - p_min) + p_min

        rmse = float(np.sqrt(np.mean((y_pred_w - y_test_w) ** 2)))
        mae  = float(np.mean(np.abs(y_pred_w - y_test_w)))
        mape = float(np.mean(np.abs((y_test_w - y_pred_w) / (np.abs(y_test_w) + 1e-6)))) * 100

        print(f"RMSE : {rmse:.2f} W")
        print(f"MAE  : {mae:.2f} W")
        print(f"MAPE : {mape:.2f}%")
    else:
        print("No forecaster checkpoint found. Run train_forecaster first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',     required=True)
    parser.add_argument('--breaker', type=float, default=15.0)
    parser.add_argument('--voltage', type=float, default=230.0)
    args = parser.parse_args()
    evaluate_all(args.csv, args.breaker, args.voltage)