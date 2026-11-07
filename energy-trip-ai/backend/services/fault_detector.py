"""
Appliance fault detection and ranking.

During a predicted trip or warning event:
1. Compute each appliance's contribution to total power
2. Rank by contribution (highest first)
3. Compute risk score per appliance
4. Identify the most probable faulty/overloading appliance
"""

import os, sys
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import pandas as pd
import numpy as np
from typing import List, Dict
from utils.logger import get_logger

logger = get_logger(__name__)


def rank_appliances_by_fault(
    df: pd.DataFrame,
    appliance_cols: List[str],
    window_seconds: int = 60
) -> List[Dict]:
    """
    Analyze appliance contributions during high-load periods.
    
    Args:
        df: Preprocessed DataFrame with appliance columns
        appliance_cols: List of appliance column names
        window_seconds: Number of recent seconds to analyze
    
    Returns:
        List of dicts sorted by risk_score descending:
        [
            {
                "appliance": "kettle",
                "contribution_pct": 45.2,
                "avg_power_w": 2300,
                "risk_score": 0.87,
                "status": "CRITICAL"
            },
            ...
        ]
    """
    # Use the most recent window for analysis
    recent = df.tail(window_seconds)
    
    if len(recent) == 0:
        return []
    
    total_avg = recent['total_power'].mean()
    
    if total_avg == 0:
        logger.warning("Total power is zero, cannot compute appliance contributions")
        return []
    
    results = []
    
    for appliance in appliance_cols:
        if appliance == 'total_power':
            continue
        
        if appliance not in df.columns:
            continue
        
        # Average power in recent window
        avg_power = recent[appliance].mean()
        
        # Contribution as percentage of total
        contribution_pct = 100.0 * (avg_power / (total_avg + 1e-6))
        contribution_pct = max(0.0, min(100.0, contribution_pct))
        
        # Spike detection: ratio of max to mean in window
        max_power = recent[appliance].max()
        spike_ratio = max_power / (avg_power + 1e-6)
        
        # Risk score: combines contribution and spike behavior
        risk_score = min(1.0, (contribution_pct / 100.0) * 0.6 + min(spike_ratio / 10.0, 0.4))
        
        # Status classification
        if risk_score > 0.7:
            status = "CRITICAL"
        elif risk_score > 0.4:
            status = "HIGH"
        elif risk_score > 0.2:
            status = "MODERATE"
        else:
            status = "LOW"
        
        results.append({
            "appliance": appliance,
            "contribution_pct": round(contribution_pct, 2),
            "avg_power_w": round(avg_power, 2),
            "max_power_w": round(max_power, 2),
            "spike_ratio": round(spike_ratio, 2),
            "risk_score": round(risk_score, 4),
            "status": status
        })
    
    # Sort by risk score descending
    results.sort(key=lambda x: x['risk_score'], reverse=True)
    
    logger.info(f"Ranked {len(results)} appliances. Top: {results[0]['appliance'] if results else 'none'}")
    return results