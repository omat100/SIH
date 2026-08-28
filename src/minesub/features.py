"""Windowed feature construction.

Each sample sits at a time ``t`` for one station:
  * features  -> summarise the BACKWARD window  [t - W + 1, t]   (what a sensor
                 node has observed so far)
  * fwd_*     -> summarise the FORWARD horizon  [t + 1, t + H]    (used only to
                 derive the early-warning label; never fed to the model)

This forward/backward split makes the task a genuine forecast and keeps the
label out of the feature set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .utils import get_logger

log = get_logger("minesub.features")

RAW_CHANNELS = ["temp_c", "humidity_pct", "tilt_deg", "distance_mm"]
# channels detrended per window (subtract first value) before the LSTM sees them
_DETREND_IDX = [RAW_CHANNELS.index("tilt_deg"), RAW_CHANNELS.index("distance_mm")]


def detrend_window(chunk: np.ndarray) -> np.ndarray:
    """chunk: [seq_len, len(RAW_CHANNELS)] -> copy with distance/tilt made
    relative to the window's first sample."""
    out = np.asarray(chunk, dtype=np.float32).copy()
    out[:, _DETREND_IDX] -= out[0, _DETREND_IDX]
    return out


def _slope_per_day(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _accel_per_day2(y: np.ndarray) -> float:
    n = len(y)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    return float(2.0 * np.polyfit(x, y, 2)[0])


def _backward_features(win: pd.DataFrame) -> dict[str, float]:
    d = win["distance_mm"].to_numpy()
    ti = win["tilt_deg"].to_numpy()
    hu = win["humidity_pct"].to_numpy()
    te = win["temp_c"].to_numpy()
    doy = int(win["date"].iloc[-1].dayofyear)
    return {
        "distance_last": d[-1],
        "distance_delta": d[-1] - d[0],
        "distance_rate": _slope_per_day(d),
        "distance_accel": _accel_per_day2(d),
        "distance_std": float(np.std(d)),
        "distance_range": float(d.max() - d.min()),
        "tilt_last": ti[-1],
        "tilt_delta": ti[-1] - ti[0],
        "tilt_rate": _slope_per_day(ti),
        "tilt_accel": _accel_per_day2(ti),
        "tilt_std": float(np.std(ti)),
        "tilt_abs_last": abs(ti[-1]),
        "humidity_mean": float(np.mean(hu)),
        "humidity_delta": hu[-1] - hu[0],
        "humidity_std": float(np.std(hu)),
        "temp_mean": float(np.mean(te)),
        "temp_range": float(te.max() - te.min()),
        "temp_std": float(np.std(te)),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }


def _robust_rate(y: np.ndarray) -> float:
    """Endpoint-mean slope (mm/day): resistant to daily measurement noise."""
    n = len(y)
    if n < 4:
        return _slope_per_day(y)
    k = max(2, n // 4)
    return float((y[-k:].mean() - y[:k].mean()) / (n - k))


def _forward_targets(fwd: pd.DataFrame) -> dict[str, float]:
    # smooth the forward window before measuring rate/accel: the LABEL should
    # capture sustained deformation, not a 2-3 day noise blip. (Backward features
    # stay raw — that's what a field node actually observes.)
    d = fwd["distance_mm"].rolling(7, center=True, min_periods=1).mean().to_numpy()
    ti = fwd["tilt_deg"].rolling(7, center=True, min_periods=1).mean().to_numpy()
    h = len(d) // 2
    return {
        "fwd_settlement_rate": _robust_rate(d),                       # mm / day
        # acceleration = (2nd-half rate) - (1st-half rate)
        "fwd_settlement_accel": _robust_rate(d[h:]) - _robust_rate(d[:h]),
        "fwd_tilt_rate": _robust_rate(ti),                            # deg / day
    }


# Public alias – reused by the live-inference path in predict.py
backward_window_features = _backward_features


def build_feature_table(cfg: Config, ts: pd.DataFrame | None = None,
                        save: bool = True) -> pd.DataFrame:
    fcfg = cfg["features"]
    W = int(fcfg["backward_window_days"])
    H = int(fcfg["forward_horizon_days"])
    step = int(fcfg["step_days"])
    min_periods = int(fcfg["min_periods"])

    if ts is None:
        ts = pd.read_parquet(cfg.paths["processed"] / "timeseries.parquet")
    ts = ts.sort_values(["station_id", "date"])

    rows: list[dict] = []
    for sid, g in ts.groupby("station_id", sort=False):
        g = g.reset_index(drop=True)
        g = g.set_index("date")
        first_day = g.index.min()
        # iterate window end-points
        end_positions = range(W - 1, len(g) - H, step)
        for pos in end_positions:
            t = g.index[pos]
            back = g.iloc[pos - W + 1: pos + 1]
            fwd = g.iloc[pos + 1: pos + 1 + H]
            if len(back) < min_periods or len(fwd) < max(3, H // 2):
                continue
            feat = _backward_features(back.reset_index())
            tgt = _forward_targets(fwd.reset_index())
            rows.append({
                "station_id": sid,
                "t": t,
                "days_since_start": float((t - first_day).days),
                "lat": float(g["lat"].iloc[pos]),
                "lon": float(g["lon"].iloc[pos]),
                "data_source_disp": g["data_source_disp"].iloc[pos],
                **feat,
                **tgt,
            })

    samples = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    log.info("feature table: %d samples, %d stations, %s -> %s",
             len(samples), samples["station_id"].nunique(),
             samples["t"].min().date(), samples["t"].max().date())
    if save:
        out = cfg.paths["processed"] / "samples.parquet"
        samples.to_parquet(out, index=False)
        log.info("wrote %s", out)
    return samples


def feature_columns(samples: pd.DataFrame) -> list[str]:
    drop = {"station_id", "t", "data_source_disp", "data_source_wx",
            "risk_class", "hazard_score", "y", "lat", "lon"}
    return [c for c in samples.columns
            if c not in drop and not c.startswith("fwd_")]


def build_sequences(cfg: Config, samples: pd.DataFrame,
                    ts: pd.DataFrame) -> np.ndarray:
    """Raw [seq_len, 4] window per sample, aligned to ``samples`` row order.
    Used by the PyTorch model."""
    seq_len = int(cfg["model"]["torch"]["seq_len"])
    ts = ts.sort_values(["station_id", "date"])
    by_station = {sid: g.set_index("date") for sid, g in ts.groupby("station_id")}
    # window-local detrend: distance/tilt carry a huge per-station DC offset and
    # secular trend that would swamp standardisation — subtract the window's
    # first value so the LSTM sees the *change* a node reports, like the field.
    arr = np.zeros((len(samples), seq_len, len(RAW_CHANNELS)), dtype=np.float32)
    for i, row in enumerate(samples.itertuples(index=False)):
        g = by_station[row.station_id]
        end = g.index.get_indexer([row.t])[0]
        start = max(0, end - seq_len + 1)
        chunk = g.iloc[start: end + 1][RAW_CHANNELS].to_numpy(dtype=np.float32)
        if len(chunk) < seq_len:  # left-pad short heads with the first row
            pad = np.repeat(chunk[:1], seq_len - len(chunk), axis=0)
            chunk = np.vstack([pad, chunk])
        arr[i] = detrend_window(chunk)
    return arr
