"""Live-inference entry point shaped for a field sensor node.

``predict(readings, model=...)`` takes a chronological list of recent readings
(one per day is expected) and returns a risk assessment for the next horizon.

Each reading is a mapping with:
    date | timestamp   ISO string or anything pandas can parse
    temp_c             float
    humidity_pct       float
    tilt_deg           float   (tilt magnitude at the node)
    distance_mm        float   (distance to the fixed overhead reference;
                                increasing => ground settling)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import RISK_CLASSES
from .config import Config, load_config
from .features import RAW_CHANNELS, backward_window_features, detrend_window
from .utils import get_logger

log = get_logger("minesub.predict")

_MODEL_FILES = {"gbdt": "gbdt.joblib", "torch": "lstm.pt"}


def _to_frame(readings) -> pd.DataFrame:
    df = pd.DataFrame(list(readings)).copy()
    date_col = "date" if "date" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
    if date_col is None:
        raise ValueError("each reading needs a 'date' or 'timestamp' field")
    df["date"] = pd.to_datetime(df[date_col])
    need = set(RAW_CHANNELS)
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"readings missing channels: {sorted(missing)}")
    return df.sort_values("date").reset_index(drop=True)


def predict(readings, model: str = "gbdt", cfg: Config | None = None,
            model_path: str | None = None) -> dict:
    if model not in _MODEL_FILES:
        raise ValueError(f"model must be one of {list(_MODEL_FILES)}")
    cfg = cfg or load_config()
    df = _to_frame(readings)

    W = int(cfg["features"]["backward_window_days"])
    seq_len = int(cfg["model"]["torch"]["seq_len"])
    need = W if model == "gbdt" else seq_len
    if len(df) < min(need, int(cfg["features"]["min_periods"])):
        raise ValueError(f"need >= {need} readings for model '{model}', got {len(df)}")

    path = model_path or (cfg.paths["models"] / _MODEL_FILES[model])

    if model == "gbdt":
        from .models.gbdt_model import GbdtRiskModel

        m = GbdtRiskModel.load(cfg, path)
        win = df.iloc[-W:][["date", *RAW_CHANNELS]]
        feats = backward_window_features(win)
        feats["days_since_start"] = float((df["date"].iloc[-1] - df["date"].iloc[0]).days)
        x = np.array([[feats.get(c, 0.0) for c in m.feature_cols]], dtype=float)
        proba = m.predict_proba(x)[0]
        signals = {k: round(float(feats[k]), 4) for k in
                   ("distance_rate", "distance_accel", "tilt_rate", "tilt_accel")}
    else:
        from .models.torch_model import LSTMRiskModel

        m = LSTMRiskModel.load(cfg, path)
        chunk = df.iloc[-seq_len:][RAW_CHANNELS].to_numpy(dtype=np.float32)
        if len(chunk) < seq_len:
            chunk = np.vstack([np.repeat(chunk[:1], seq_len - len(chunk), axis=0), chunk])
        proba = m.predict_proba(detrend_window(chunk)[None, ...])[0]
        signals = {}

    idx = int(np.argmax(proba))
    return {
        "risk_class": RISK_CLASSES[idx],
        "probabilities": {c: round(float(p), 4) for c, p in zip(RISK_CLASSES, proba)},
        "horizon_days": int(cfg["features"]["forward_horizon_days"]),
        "model": model,
        "signals": signals,
        "as_of": df["date"].iloc[-1].isoformat(),
    }
