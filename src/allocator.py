"""
allocator.py
------------
Map predicted macro regime to asset allocation weights and build a portfolio
dataframe suitable for back‑testing.
"""

import pandas as pd
from typing import Dict, Literal

Regime = Literal["Expansion", 
                 "Recession", 
                 ]

REGIME_WEIGHTS: Dict[Regime, Dict[str, float]] = {
    "Expansion": {"SPY": 0.70, "TLT": 0.20, "GLD": 0.10},
    "Recession": {"SPY": 0.0, "TLT": 0.50, "GLD": 0.30, "BIL": 0.20},
}

def get_weights(regime: Regime) -> Dict[str, float]:
    """Return dictionary of asset weights for a given regime."""
    return REGIME_WEIGHTS[regime]

def regime_series_to_weights(regimes: pd.Series) -> pd.DataFrame:
    """
    Convert a time‑series of regimes to a weights dataframe (index aligned with
    regimes, columns are tickers, values are weights).
    """
    weight_frames = []
    for date, r in regimes.items():
        w = pd.Series(get_weights(r), name=date)
        weight_frames.append(w)

    return pd.DataFrame(weight_frames).fillna(0.0)