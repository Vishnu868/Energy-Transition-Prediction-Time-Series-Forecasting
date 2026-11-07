"""
Prediction endpoint — absolute imports only.
"""

import os
import sys
import glob

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.preprocessor    import load_and_preprocess
from services.feature_engineer import engineer_features
from services.label_generator  import generate_labels, get_label_stats
from services.fault_detector   import rank_appliances_by_fault
from services.forecaster       import run_forecast
from services.classifier       import run_classification, run_shap_explanation
from utils.helpers             import compute_stability_score
from utils.logger              import get_logger

router    = APIRouter()
logger    = get_logger(__name__)
UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")


class PredictRequest(BaseModel):
    session_id:          str   = Field(...,    description="Session ID from upload")
    breaker_rating_amps: float = Field(15.0,  ge=1.0,   le=400.0)
    voltage:             float = Field(230.0, ge=100.0,  le=480.0)
    location_type:       str   = Field("home")
    resample_freq:       str   = Field("5s")
    forecast_steps:      int   = Field(10,    ge=1,      le=100)


@router.post("/predict")
async def predict(req: PredictRequest):
    logger.info(f"Predict: session={req.session_id}, breaker={req.breaker_rating_amps}A")

    # ── Find uploaded file ───────────────────────────────────
    matches = glob.glob(os.path.join(UPLOAD_DIR, f"{req.session_id}_*"))
    if not matches:
        raise HTTPException(404, f"No file for session '{req.session_id}'")

    filepath = matches[0]

    try:
        # ── 1. Preprocess ────────────────────────────────────
        df, appliance_cols = load_and_preprocess(
            filepath, resample_freq=req.resample_freq, voltage=req.voltage
        )
        if len(df) < 20:
            raise HTTPException(400, f"Too few rows after preprocessing ({len(df)}). Need ≥ 20.")

        # ── 2. Feature engineering ───────────────────────────
        df = engineer_features(df, voltage=req.voltage)

        # ── 3. Label generation ──────────────────────────────
        df          = generate_labels(df, rated_current=req.breaker_rating_amps, voltage=req.voltage)
        label_stats = get_label_stats(df)

        # ── 4. Feature column list (inference uses checkpoint) ─
        feature_cols = [
            'total_power', 'current',
            'rolling_mean_10s', 'rolling_std_10s',
            'rolling_mean_60s', 'rolling_std_60s',
            'rate_of_change', 'rate_of_change_abs', 'rate_of_change_2nd',
            'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10',
            'load_factor', 'peak_to_avg_ratio',
            'cusum_score', 'adf_pvalue', 'non_stationary_flag'
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]

        # ── 5. Classification ────────────────────────────────
        classification = run_classification(df, feature_cols=feature_cols, seq_len=10)

        # ── 6. Forecasting ───────────────────────────────────
        historical_power, forecast_power = run_forecast(
            df, feature_cols=feature_cols,
            seq_len=60, pred_len=req.forecast_steps
        )

        # ── 7. Fault ranking ─────────────────────────────────
        fault_ranking = rank_appliances_by_fault(df, appliance_cols=appliance_cols)

        # ── 8. SHAP ──────────────────────────────────────────
        shap_results = run_shap_explanation(df, feature_cols=feature_cols, seq_len=10)

        # ── 9. Stability score ───────────────────────────────
        current_load_factor = float(df['load_factor'].tail(30).mean())
        current_roc = float(
            df['rate_of_change_abs'].tail(30).mean() / (df['total_power'].mean() + 1)
        )
        stability_score = compute_stability_score(
            trip_prob=classification['trip_probability'],
            load_factor=current_load_factor,
            rate_of_change=current_roc
        )

        # ── 10. Current state summary ────────────────────────
        latest            = df.tail(1).iloc[0]
        trip_threshold_w  = req.voltage * req.breaker_rating_amps
        current_power     = float(latest['total_power'])
        load_pct          = 100.0 * current_power / trip_threshold_w

        # Chart data (last 200 points)
        chart_tail   = df.tail(200)
        timestamps   = [t.isoformat() for t in chart_tail.index]
        power_values = [round(float(v), 2) for v in chart_tail['total_power']]
        label_values = [int(v) for v in chart_tail['label']]

        # Forecast timestamps
        freq_sec = _freq_to_seconds(req.resample_freq)
        last_ts  = df.index[-1]
        forecast_timestamps = [
            (last_ts + pd.Timedelta(seconds=(i + 1) * freq_sec)).isoformat()
            for i in range(req.forecast_steps)
        ]

        return {
            "success":       True,
            "session_id":    req.session_id,
            "location_type": req.location_type,
            "data_info": {
                "total_rows":         len(df),
                "appliances_detected": appliance_cols,
                "time_range": {
                    "start": df.index[0].isoformat(),
                    "end":   df.index[-1].isoformat()
                }
            },
            "current_state": {
                "total_power_w":        round(current_power, 2),
                "current_a":            round(current_power / req.voltage, 3),
                "breaker_rating_a":     req.breaker_rating_amps,
                "trip_threshold_w":     round(trip_threshold_w, 2),
                "warning_threshold_w":  round(0.8 * trip_threshold_w, 2),
                "load_percentage":      round(load_pct, 1),
                "status":               str(latest['label_name']),
            },
            "classification":  classification,
            "stability_score": round(stability_score, 1),
            "label_stats":     label_stats,
            "chart_data": {
                "timestamps":           timestamps,
                "power_values":         power_values,
                "label_values":         label_values,
                "forecast_timestamps":  forecast_timestamps,
                "forecast_values":      [round(float(v), 2) for v in forecast_power],
                "historical_for_chart": [round(float(v), 2) for v in historical_power[-60:]],
            },
            "fault_ranking":    fault_ranking,
            "shap_explanation": shap_results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(500, f"Prediction error: {str(e)}")


def _freq_to_seconds(freq: str) -> int:
    return {'1s': 1, '5s': 5, '10s': 10, '30s': 30, '1T': 60, '5T': 300, '1H': 3600}.get(freq, 5)