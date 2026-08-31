"""Assemble the per-station daily timeseries used for features & labels.

Output schema (one row per station per day):
    station_id, date, lat, lon,
    temp_c, humidity_pct,           # weather (real or seasonal fallback)
    tilt_deg,                       # derived: arctan(differential settlement / baseline)
    distance_mm,                    # cumulative ground settlement == the field
                                    # "distance" channel (fixed overhead reference,
                                    # increasing distance => ground dropping)
    data_source_disp, data_source_wx
"""
### not my work
# _resample_daily() - Returns daily settlement data for each station by interpolating sparse displacement readings.
# _derive_tilt() - Returns the daily dataset with a calculated tilt_deg value based on settlement differences between neighbouring stations.
# build_timeseries() - Returns the complete per-station daily timeseries by combining displacement, derived tilt, and weather data; optionally saves it as a Parquet file.
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger, haversine_m
from .sources import load_ground_displacement, load_weather

log = get_logger("minesub.data.build")


def _resample_daily(disp: pd.DataFrame) -> pd.DataFrame:
    """CA DWR surveys are sparse (monthly/quarterly); interpolate to daily."""
    out = []
    for sid, g in disp.groupby("station_id", sort=False):
        g = g.sort_values("date").drop_duplicates("date")
        s = g.set_index("date")["settlement_mm"].resample("D").mean()
        s = s.interpolate(method="time", limit_direction="both")
        d = pd.DataFrame({"station_id": sid, "date": s.index, "distance_mm": s.to_numpy()})
        d["lat"] = float(g["lat"].iloc[0])
        d["lon"] = float(g["lon"].iloc[0])
        d["data_source_disp"] = g["data_source"].iloc[0]
        out.append(d)
    return pd.concat(out, ignore_index=True)


def _derive_tilt(daily: pd.DataFrame, max_neighbor_m: float, fallback_baseline_m: float,
                 sensor_noise_deg: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """tilt = arctan(differential settlement between a station and its nearest
    neighbour / horizontal baseline). This is exactly how a geotechnical
    tiltmeter is interpreted; here the baseline is the inter-station distance.
    ``sensor_noise_deg`` adds independent per-reading noise so tilt is not a
    perfectly deterministic function of the settlement channel."""
    rng = np.random.default_rng(seed)
    coords = (daily.groupby("station_id")[["lat", "lon"]].first().reset_index())
    ids = coords["station_id"].to_numpy()
    lat = coords["lat"].to_numpy()
    lon = coords["lon"].to_numpy()

    # pairwise nearest neighbour
    neigh: dict[str, tuple[str, float]] = {}
    for i, sid in enumerate(ids):
        d = haversine_m(lat[i], lon[i], lat, lon)
        d[i] = np.inf
        j = int(np.argmin(d))
        neigh[sid] = (ids[j], float(d[j]))

    wide = daily.pivot_table(index="date", columns="station_id", values="distance_mm")
    global_mean = wide.mean(axis=1)

    tilt_cols = {}
    for sid in wide.columns:
        nb, dist = neigh[sid]
        if np.isfinite(dist) and dist <= max_neighbor_m and nb in wide.columns:
            baseline_mm = max(dist, 1.0) * 1000.0
            diff = wide[sid] - wide[nb]
        else:  # isolated station: gradient vs the regional mean over a nominal baseline
            baseline_mm = fallback_baseline_m * 1000.0
            diff = wide[sid] - global_mean
        t = np.degrees(np.arctan(diff / baseline_mm))
        if sensor_noise_deg > 0:
            t = t + rng.normal(0.0, sensor_noise_deg, size=len(t))
        tilt_cols[sid] = t

    tilt = pd.DataFrame(tilt_cols, index=wide.index).reset_index().melt(
        id_vars="date", var_name="station_id", value_name="tilt_deg")
    return daily.merge(tilt, on=["date", "station_id"], how="left")


def build_timeseries(cfg: Config, save: bool = True) -> pd.DataFrame:
    cfg.ensure_dirs()
    disp = load_ground_displacement(cfg)
    daily = _resample_daily(disp)

    tcfg = cfg["tilt"]
    daily = _derive_tilt(daily, float(tcfg["max_neighbor_distance_m"]),
                         float(tcfg["fallback_baseline_m"]),
                         sensor_noise_deg=float(tcfg.get("sensor_noise_deg", 0.0)),
                         seed=int(cfg["seed"]))

    stations = daily.groupby("station_id")[["lat", "lon"]].first().reset_index()
    wx = load_weather(cfg, stations)
    daily = daily.merge(
        wx.rename(columns={"data_source": "data_source_wx"}),
        on=["station_id", "date"], how="left",
    )
    # fill any weather gaps at the edges of the join
    for col in ("temp_c", "humidity_pct"):
        daily[col] = daily.groupby("station_id")[col].transform(
            lambda s: s.interpolate(limit_direction="both"))
    daily["rain_mm"] = daily["rain_mm"].fillna(0.0)
    daily["data_source_wx"] = daily["data_source_wx"].fillna("synthetic_fallback")

    daily = daily.dropna(subset=["distance_mm", "tilt_deg", "temp_c", "humidity_pct"])
    daily = daily.sort_values(["station_id", "date"]).reset_index(drop=True)

    log.info("timeseries: %d rows, %d stations, %s -> %s",
             len(daily), daily["station_id"].nunique(),
             daily["date"].min().date(), daily["date"].max().date())
    log.info("displacement source(s): %s | weather source(s): %s",
             sorted(daily["data_source_disp"].unique()),
             sorted(daily["data_source_wx"].unique()))

    if save:
        out = cfg.paths["processed"] / "timeseries.parquet"
        daily.to_parquet(out, index=False)
        log.info("wrote %s", out)
    return daily
