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

import json

import numpy as np
import pandas as pd

from . import RISK_CLASSES
from .config import Config, load_config
from .features import RAW_CHANNELS, backward_window_features, detrend_window
from .utils import get_logger

log = get_logger("minesub.predict")

_MODEL_FILES = {"gbdt": "gbdt.joblib", "torch": "lstm.pt"}


def _regression_alert(rate: float, cfg: Config, model: str) -> dict:
    """Map a predicted forward *excess* settlement rate (mm/day) to an alert.

    Primary signal = the recall-tuned operating-point threshold learned at
    training time (reports/metrics_<model>.json); secondary = the fixed
    geotechnical bands from config.labels.absolute.
    """
    out = {"predicted_excess_rate_mm_day": round(float(rate), 5)}
    op_thr = None
    mfile = cfg.paths["reports"] / f"metrics_{model}.json"
    if mfile.exists():
        try:
            cop = json.loads(mfile.read_text()).get("chosen_operating_point") or {}
            op_thr = cop.get("threshold")
        except Exception:  # noqa: BLE001
            op_thr = None
    if op_thr is not None:
        out["alert"] = "elevated" if rate >= op_thr else "normal"
        out["operating_threshold_mm_day"] = round(float(op_thr), 5)
    a = cfg["labels"].get("absolute", {}).get("settlement_rate_mm_day", {})
    if a:
        w, c = float(a.get("watch", np.inf)), float(a.get("critical", np.inf))
        out["geotech_band"] = "critical" if rate >= c else "watch" if rate >= w else "safe"
    return out


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
    kalman = cfg["features"].get("kalman")

    if model == "gbdt":
        from .models.gbdt_model import GbdtRiskModel

        m = GbdtRiskModel.load(cfg, path)
        win = df.iloc[-W:][["date", *RAW_CHANNELS]]
        feats = backward_window_features(win, kalman=kalman)
        feats["days_since_start"] = float((df["date"].iloc[-1] - df["date"].iloc[0]).days)
        x = np.array([[feats.get(c, 0.0) for c in m.feature_cols]], dtype=float)
        task = getattr(m, "task", "classification")
        raw = m.predict(x)[0] if task == "regression" else m.predict_proba(x)[0]
        signals = {k: round(float(feats[k]), 4) for k in
                   ("distance_rate", "distance_accel", "tilt_rate", "tilt_accel")
                   if k in feats}
        if kalman and kalman.get("enabled"):
            signals.update({k: round(float(feats[k]), 5) for k in
                            ("dist_kf_rate", "dist_kf_accel") if k in feats})
    else:
        from .models.torch_model import LSTMRiskModel

        m = LSTMRiskModel.load(cfg, path)
        task = getattr(m, "task", "classification")
        chunk = df.iloc[-seq_len:][RAW_CHANNELS].to_numpy(dtype=np.float32)
        if len(chunk) < seq_len:
            chunk = np.vstack([np.repeat(chunk[:1], seq_len - len(chunk), axis=0), chunk])
        win_in = detrend_window(chunk)[None, ...]
        raw = m.predict(win_in)[0] if task == "regression" else m.predict_proba(win_in)[0]
        signals = {}

    common = {
        "horizon_days": int(cfg["features"]["forward_horizon_days"]),
        "model": model,
        "task": task,
        "signals": signals,
        "as_of": df["date"].iloc[-1].isoformat(),
    }
    if task == "regression":
        return {**_regression_alert(float(raw), cfg, model), **common}

    proba = raw
    idx = int(np.argmax(proba))
    return {
        "risk_class": RISK_CLASSES[idx],
        "probabilities": {c: round(float(p), 4) for c, p in zip(RISK_CLASSES, proba)},
        **common,
    }
