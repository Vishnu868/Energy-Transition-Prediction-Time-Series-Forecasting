"""
Trip classification service.
Loads trained LSTM classifier and computes trip probability + class.
FIXED: n_features is now read from checkpoint to prevent size mismatch.
"""

import os, sys
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import torch
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import List, Dict
from models.lstm_classifier import LSTMClassifier
from utils.logger import get_logger

logger    = get_logger(__name__)
MODEL_PATH = Path(BACKEND_DIR) / "saved_models" / "classifier.pt"
_classifier_model = None
_classifier_feature_cols = None   # feature list stored in checkpoint


def _load_classifier_from_checkpoint():
    """
    Load classifier using EXACTLY the feature count and feature list
    stored inside the checkpoint file. Never infers shape from caller.
    """
    global _classifier_model, _classifier_feature_cols

    if not MODEL_PATH.exists():
        logger.warning("No saved classifier found. Using untrained model with default 16 features.")
        # Return a dummy model — inference will give random but won't crash
        model = LSTMClassifier(n_features=16)
        _classifier_feature_cols = None
        _classifier_model = model
        return model, None

    checkpoint = torch.load(MODEL_PATH, map_location='cpu')

    # Read n_features from checkpoint (saved during training)
    n_features = checkpoint.get('n_features', None)
    feature_cols = checkpoint.get('feature_cols', None)

    if n_features is None and feature_cols is not None:
        n_features = len(feature_cols)

    if n_features is None:
        # Last resort: infer from weight matrix shape
        state = checkpoint['model_state_dict']
        n_features = state['lstm.weight_ih_l0'].shape[1]
        logger.warning(f"n_features not in checkpoint; inferred from weights: {n_features}")

    model = LSTMClassifier(n_features=n_features)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _classifier_model = model
    _classifier_feature_cols = feature_cols
    logger.info(f"Classifier loaded: n_features={n_features}, cols={feature_cols}")
    return model, feature_cols


def _align_features(df: pd.DataFrame, trained_cols: List[str]) -> np.ndarray:
    """
    Align the inference DataFrame columns to exactly match
    what the model was trained on.

    - Missing columns → filled with 0.0
    - Extra columns   → dropped
    - Order           → matches trained_cols exactly
    """
    aligned = pd.DataFrame(index=df.index)
    for col in trained_cols:
        if col in df.columns:
            aligned[col] = df[col]
        else:
            logger.warning(f"Feature '{col}' missing in inference data — filling with 0")
            aligned[col] = 0.0
    return aligned.values.astype(np.float32)


def run_classification(
    df: pd.DataFrame,
    feature_cols: List[str],   # kept for API compatibility; actual cols come from checkpoint
    seq_len: int = 10
) -> Dict:
    """
    Classify current power state as Normal / Warning / Trip.
    Uses feature list embedded in the checkpoint, not the caller's list.
    """
    model, trained_cols = _load_classifier_from_checkpoint()

    # Use trained feature list if available, else fall back to caller's list
    effective_cols = trained_cols if trained_cols is not None else feature_cols

    if trained_cols is not None:
        features = _align_features(df, trained_cols)
    else:
        features = df[feature_cols].values.astype(np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    features = np.clip(features, -10.0, 10.0)

    if len(features) < seq_len:
        logger.warning("Insufficient data for classification")
        return _default_classification()

    input_window = features[-seq_len:]
    input_tensor = torch.tensor(input_window).unsqueeze(0)

    probs = model.predict_proba(input_tensor).squeeze(0).numpy()

    predicted_class = int(np.argmax(probs))
    class_names = {0: "Normal", 1: "Warning", 2: "Trip"}

    result = {
        "predicted_class": predicted_class,
        "class_name": class_names[predicted_class],
        "probabilities": {
            "normal": round(float(probs[0]), 4),
            "warning": round(float(probs[1]), 4),
            "trip":    round(float(probs[2]), 4),
        },
        "trip_probability": round(float(probs[2]), 4),
        "confidence":       round(float(probs[predicted_class]), 4),
    }

    logger.info(f"Classification: {result['class_name']} (trip_prob={result['trip_probability']})")
    return result


def run_shap_explanation(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 10,
    n_background: int = 50
) -> List[Dict]:
    """
    SHAP explainability using features aligned to checkpoint.
    """
    try:
        model, trained_cols = _load_classifier_from_checkpoint()
        effective_cols = trained_cols if trained_cols is not None else feature_cols

        if trained_cols is not None:
            features = _align_features(df, trained_cols)
        else:
            features = df[feature_cols].values.astype(np.float32)

        features = np.nan_to_num(features, nan=0.0)
        features = np.clip(features, -10.0, 10.0)

        n_feat = features.shape[1]

        if len(features) < seq_len + n_background:
            logger.warning("Insufficient data for SHAP analysis")
            return []

        def predict_fn(x_flat):
            n_samples = x_flat.shape[0]
            x_3d = x_flat.reshape(n_samples, seq_len, n_feat)
            tensor = torch.tensor(x_3d, dtype=torch.float32)
            return model.predict_proba(tensor).numpy()

        n_avail = len(features) - seq_len
        indices = np.random.choice(
            n_avail,
            size=min(n_background, n_avail),
            replace=False
        )
        background = np.array(
            [features[i: i + seq_len].flatten() for i in indices],
            dtype=np.float32
        )
        query = features[-seq_len:].flatten().reshape(1, -1)

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(query, nsamples=100)

        trip_shap = shap_values[2][0]
        trip_shap_reshaped = trip_shap.reshape(seq_len, n_feat)
        feature_importance = np.mean(np.abs(trip_shap_reshaped), axis=0)

        shap_results = [
            {"feature": col, "shap_value": round(float(feature_importance[i]), 6)}
            for i, col in enumerate(effective_cols)
        ]
        shap_results.sort(key=lambda x: x['shap_value'], reverse=True)
        logger.info(f"SHAP done. Top: {shap_results[0]['feature']}")
        return shap_results[:10]

    except Exception as e:
        logger.error(f"SHAP failed: {e}")
        return []


def _default_classification() -> Dict:
    return {
        "predicted_class": 0,
        "class_name": "Normal",
        "probabilities": {"normal": 1.0, "warning": 0.0, "trip": 0.0},
        "trip_probability": 0.0,
        "confidence": 1.0,
    }