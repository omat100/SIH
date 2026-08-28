"""Acquire the two real inputs, each with a physical synthetic fallback.

1. ``load_ground_displacement`` -> CA DWR ground-surface displacement, normalised
   to columns: station_id, date, lat, lon, settlement_mm  (positive = subsidence).
2. ``load_weather`` -> daily temperature / relative humidity per station.

Both prefer real data. If the network is unreachable or the remote schema is
unrecognised, and ``data.allow_synthetic_fallback`` is true, a labelled
surrogate with the identical schema is produced so the pipeline still runs
end to end. Every row carries a ``data_source`` column so this is never hidden.
"""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from ..config import Config
from ..utils import get_logger

log = get_logger("minesub.data.sources")

_GD_COLUMNS = ["station_id", "date", "lat", "lon", "settlement_mm", "data_source"]
_WX_COLUMNS = ["station_id", "date", "temp_c", "humidity_pct", "data_source"]

# San Joaquin Valley bounding box (the CA DWR subsidence monitoring footprint).
_BBOX = dict(lat_min=35.0, lat_max=37.4, lon_min=-120.7, lon_max=-119.1)


# ---------------------------------------------------------------------------
# CA DWR ground-surface displacement
# ---------------------------------------------------------------------------
def load_ground_displacement(cfg: Config) -> pd.DataFrame:
    dcfg = cfg["data"]
    start = pd.Timestamp(dcfg["min_date"])
    end = pd.Timestamp(dcfg["max_date"])
    try:
        df = _fetch_ca_dwr(dcfg["ca_dwr"]["ckan_base"], dcfg["ca_dwr"]["dataset_id"])
        df = _normalise_displacement(df)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        if df["station_id"].nunique() < 3 or len(df) < 500:
            raise ValueError(f"real feed too small after filtering: {df.shape}")
        df["data_source"] = "ca_dwr"
        log.info("CA DWR feed: %d rows, %d stations", len(df), df["station_id"].nunique())
        return df[_GD_COLUMNS].sort_values(["station_id", "date"]).reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001 - any failure -> fallback
        if not dcfg.get("allow_synthetic_fallback", True):
            raise
        log.warning("CA DWR unavailable (%s); using synthetic_fallback surrogate", exc)
        return _synthetic_displacement(
            n_stations=int(dcfg.get("synthetic_n_stations", 45)),
            start=start,
            end=end,
            seed=int(cfg["seed"]),
            noise_mm=float(dcfg.get("synthetic_noise_mm", 2.5)),
        )


def _read_csv_url(url: str) -> pd.DataFrame:
    raw = requests.get(url, timeout=120)
    raw.raise_for_status()
    return pd.read_csv(io.BytesIO(raw.content), low_memory=False)


def _fetch_ca_dwr(ckan_base: str, dataset_id: str) -> pd.DataFrame:
    """CA DWR ships this dataset as separate CSVs: a per-station daily
    displacement timeseries + a station metadata table. Locate both by name and
    join them into one station/date/lat/lon/displacement frame."""
    resp = requests.get(f"{ckan_base.rstrip('/')}/package_show",
                        params={"id": dataset_id}, timeout=30)
    resp.raise_for_status()
    res = [r for r in resp.json()["result"]["resources"]
           if str(r.get("format", "")).lower() == "csv" and r.get("url")]
    if not res:
        raise ValueError("no CSV resources on CKAN dataset")

    def _find(*needles):
        for r in res:
            hay = f"{r.get('name', '')} {r.get('url', '')}".lower()
            if any(n in hay for n in needles):
                return r
        return None

    ts_res = _find("daily", "timeseries", "time series", "displacement-daily")
    st_res = _find("station")
    if ts_res is None:  # last resort: biggest CSV is probably the timeseries
        ts_res = max(res, key=lambda r: r.get("size") or 0)

    ts = _read_csv_url(ts_res["url"])
    log.info("CA DWR timeseries resource '%s': %s", ts_res.get("name"), ts.shape)

    st = _read_csv_url(st_res["url"]) if st_res is not None else None
    s_id = _pick(list(ts.columns), "station") or _pick(list(ts.columns), "site")
    if st is not None and s_id is not None:
        st_id = _pick(list(st.columns), "station") or _pick(list(st.columns), "site")
        lat = _pick(list(st.columns), "lat")
        lon = _pick(list(st.columns), "lon") or _pick(list(st.columns), "long")
        if st_id and lat and lon:
            st = st[[st_id, lat, lon]].rename(columns={st_id: s_id})
            ts = ts.merge(st, on=s_id, how="left")
            log.info("joined station coords (%d stations)", st[s_id].nunique())
    return ts


def _pick(cols: list[str], *needles: str) -> str | None:
    low = {c.lower().strip(): c for c in cols}
    for key, orig in low.items():
        if all(n in key for n in needles):
            return orig
    return None


def _normalise_displacement(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    c_station = _pick(cols, "station") or _pick(cols, "site") or _pick(cols, "name")
    c_lat = _pick(cols, "lat")
    c_lon = _pick(cols, "lon") or _pick(cols, "long")
    c_date = _pick(cols, "date") or _pick(cols, "time")
    c_disp = (
        _pick(cols, "vertical", "dsp")
        or _pick(cols, "vertical")
        or _pick(cols, "displacement")
        or _pick(cols, "subsidence")
        or _pick(cols, "elevation", "change")
        or _pick(cols, "settlement")
    )
    c_qc = _pick(cols, "dsp", "qc") or _pick(cols, "qc") or _pick(cols, "quality")
    missing = [n for n, c in dict(station=c_station, lat=c_lat, lon=c_lon,
                                  date=c_date, displacement=c_disp).items() if c is None]
    if missing:
        raise ValueError(f"unrecognised CA DWR schema, missing {missing}; saw {cols}")

    out = pd.DataFrame({
        "station_id": df[c_station].astype(str).str.strip(),
        "date": pd.to_datetime(df[c_date], errors="coerce"),
        "lat": pd.to_numeric(df[c_lat], errors="coerce"),
        "lon": pd.to_numeric(df[c_lon], errors="coerce"),
        "disp_raw": pd.to_numeric(df[c_disp], errors="coerce"),
        "qc": pd.to_numeric(df[c_qc], errors="coerce") if c_qc else 1,
    }).dropna(subset=["date", "lat", "lon", "disp_raw"])

    # Keep only good-quality readings (CA DWR QC code 1 = "good data").
    if c_qc:
        before = len(out)
        out = out[out["qc"].isin([1, 2])]
        log.info("QC filter kept %d / %d rows", len(out), before)

    # Heuristic unit -> millimetres. CA DWR extensometer VERTICAL_DSP is in feet
    # (values ~0.01-20); older/other dumps have used cm / m / mm.
    span = out["disp_raw"].abs().quantile(0.99)
    name = (c_disp or "").lower()
    if "mm" in name or span > 300:
        scale = 1.0
    elif "cm" in name or span > 30:
        scale = 10.0
    elif "ft" in name or "feet" in name or 0.02 < span <= 30:
        scale = 304.8
    else:  # metres
        scale = 1000.0
    log.info("displacement unit scale -> mm: x%.4g (99p span=%.3g raw)", scale, span)
    out["settlement_mm"] = out["disp_raw"] * scale

    # Orient so that subsidence (ground going down) is positive. CA DWR reports
    # elevation change, i.e. negative when subsiding -> flip if that's the case.
    if out["settlement_mm"].mean() < 0:
        out["settlement_mm"] = -out["settlement_mm"]

    return out.drop(columns="disp_raw")


# ---------------------------------------------------------------------------
# Weather (Meteostat, with a seasonal fallback)
# ---------------------------------------------------------------------------
def load_weather(cfg: Config, stations: pd.DataFrame) -> pd.DataFrame:
    """``stations`` needs columns station_id, lat, lon. Returns daily weather."""
    dcfg = cfg["data"]
    start = pd.Timestamp(dcfg["min_date"]).to_pydatetime()
    end = pd.Timestamp(dcfg["max_date"]).to_pydatetime()
    provider = dcfg.get("weather", {}).get("provider", "meteostat")

    frames: list[pd.DataFrame] = []
    real_ok = 0
    if provider == "meteostat":
        try:
            from meteostat import Daily, Stations  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            log.warning("meteostat import failed (%s); seasonal fallback for all", exc)
        else:
            for row in stations.itertuples(index=False):
                wx = _meteostat_one(row.station_id, row.lat, row.lon, start, end)
                if wx is not None and len(wx) > 30:
                    frames.append(wx)
                    real_ok += 1
                else:
                    frames.append(_synthetic_weather_one(row.station_id, row.lat,
                                                         start, end, seed=int(cfg["seed"])))
    if not frames:  # provider unknown or everything failed
        for row in stations.itertuples(index=False):
            frames.append(_synthetic_weather_one(row.station_id, row.lat, start, end,
                                                 seed=int(cfg["seed"])))

    out = pd.concat(frames, ignore_index=True)
    log.info("weather: %d rows, %d/%d stations from real feed",
             len(out), real_ok, len(stations))
    return out[_WX_COLUMNS].sort_values(["station_id", "date"]).reset_index(drop=True)


def _meteostat_one(station_id, lat, lon, start, end) -> pd.DataFrame | None:
    try:
        from meteostat import Daily, Stations

        near = Stations().nearby(float(lat), float(lon)).fetch(1)
        if near is None or near.empty:
            return None
        ms_id = near.index[0]
        daily = Daily(ms_id, start, end).fetch()
        if daily is None or daily.empty or "tavg" not in daily:
            return None
        daily = daily.reset_index().rename(columns={"time": "date"})
        temp = pd.to_numeric(daily["tavg"], errors="coerce")
        if temp.notna().mean() < 0.5:
            return None
        temp = temp.interpolate().bfill().ffill()
        if "rhum" in daily and pd.to_numeric(daily["rhum"], errors="coerce").notna().mean() > 0.5:
            hum = pd.to_numeric(daily["rhum"], errors="coerce").interpolate().bfill().ffill()
        else:  # derive a plausible RH from temperature + precipitation
            prcp = pd.to_numeric(daily.get("prcp", 0.0), errors="coerce").fillna(0.0)
            hum = (72.0 - 1.4 * (temp - temp.mean()) + 6.0 * np.tanh(prcp / 5.0)).clip(8, 100)
        return pd.DataFrame({
            "station_id": station_id,
            "date": pd.to_datetime(daily["date"]),
            "temp_c": temp.to_numpy(),
            "humidity_pct": np.asarray(hum, dtype=float),
            "data_source": "meteostat",
        })
    except Exception as exc:  # noqa: BLE001
        log.debug("meteostat failed for %s: %s", station_id, exc)
        return None


# ---------------------------------------------------------------------------
# Synthetic surrogates (physical, clearly labelled)
# ---------------------------------------------------------------------------
def _synthetic_weather_one(station_id, lat, start, end, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash((station_id, seed))) % (2**32))
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.to_numpy()
    mean_annual = 18.0 - 0.9 * (float(lat) - 35.0)
    seasonal = -np.cos(2 * np.pi * (doy - 15) / 365.25)
    temp = mean_annual + 11.0 * seasonal + rng.normal(0, 2.4, len(dates))
    humidity = (58.0 - 1.3 * (temp - mean_annual) + 12.0 * (-seasonal)
                + rng.normal(0, 6.0, len(dates))).clip(8, 100)
    return pd.DataFrame({
        "station_id": station_id, "date": dates,
        "temp_c": temp, "humidity_pct": humidity,
        "data_source": "synthetic_fallback",
    })


def _synthetic_displacement(n_stations: int, start, end, seed: int,
                            noise_mm: float = 2.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    t_years = (dates - dates[0]).days.to_numpy() / 365.25
    doy = dates.dayofyear.to_numpy()

    lats = rng.uniform(_BBOX["lat_min"], _BBOX["lat_max"], n_stations)
    lons = rng.uniform(_BBOX["lon_min"], _BBOX["lon_max"], n_stations)

    # Smooth secular subsidence field: a couple of Gaussian "bowls".
    centres = [(36.05, -119.9), (36.6, -120.1)]
    base_rate = np.full(n_stations, 5.0)  # mm/yr background
    for clat, clon in centres:
        d2 = (lats - clat) ** 2 + ((lons - clon) * 0.8) ** 2
        base_rate += rng.uniform(35, 60) * np.exp(-d2 / (2 * 0.25**2))
    base_rate += rng.normal(0, 3, n_stations)
    base_rate = base_rate.clip(1, None)

    common_mode = np.cumsum(rng.normal(0, 0.15, len(dates)))  # regional AR-ish drift
    dt = float(t_years[1] - t_years[0])
    n = len(dates)
    # seasonal "wetness" (0..1): acceleration episodes are more likely to start a
    # few weeks after wet spells -> a genuine but noisy leading indicator that
    # the models can partly learn from the humidity / seasonal features.
    wetness = 0.5 + 0.5 * (-np.cos(2 * np.pi * (doy - 40) / 365.25))
    onset_bias = wetness / wetness.sum()

    frames = []
    for i in range(n_stations):
        rate = base_rate[i]
        secular = rate * t_years
        seas_amp = rng.uniform(3, 18)
        seasonal = seas_amp * (-np.cos(2 * np.pi * (doy - 40) / 365.25)) \
            + 0.4 * seas_amp * np.sin(4 * np.pi * doy / 365.25)

        # Time-varying extra rate: abrupt-onset acceleration episodes (drought /
        # over-pumping / roof caving analogue) AND partial recoveries (wet year /
        # recharge). Onsets are sharp and randomly timed so a trailing 30-day
        # window often gives little warning of the next 14 days -> genuine
        # forecasting difficulty rather than a deterministic ramp.
        idx = np.arange(n)
        extra_rate = np.zeros(n)                       # mm/yr, added on top
        # wet-spell-linked episodes: a short accelerating-creep precursor (a real
        # early-warning signal) followed by a sharper main onset ~2-3 weeks after
        # a wet period. The precursor is partly visible in a trailing window ->
        # legitimately learnable; noise + the random onsets below keep it hard.
        for _ in range(rng.integers(1, 5)):
            k0 = int(rng.choice(n, p=onset_bias))
            k0 = min(n - 1, k0 + int(rng.uniform(10, 24)))  # lag after the wet spell
            add = rng.uniform(45, 150)
            hold = rng.uniform(0.25, 1.0)
            pre = int(rng.uniform(14, 32))                   # precursor length (days)
            precursor = np.clip((idx - (k0 - pre)) / pre, 0, 1) ** 1.6 * (idx < k0)
            main = np.clip((idx - k0) / rng.uniform(4, 12), 0, 1)
            decay = np.clip(1 - (idx - k0) * dt / hold, 0, 1) ** 0.5
            extra_rate += add * (0.45 * precursor + main * (0.35 + 0.65 * decay))
        # a few purely random abrupt onsets -> irreducible forecast error
        for _ in range(rng.integers(0, 3)):
            k0 = rng.integers(int(0.1 * n), n - 1)
            ramp = np.clip((idx - k0) / rng.uniform(1, 4), 0, 1)
            extra_rate += rng.uniform(30, 90) * ramp * np.clip(
                1 - (idx - k0) * dt / rng.uniform(0.15, 0.5), 0, 1)
        for _ in range(rng.integers(0, 2)):            # recovery episodes
            k0 = rng.integers(int(0.2 * n), n - 1)
            extra_rate -= rng.uniform(10, 40) * np.clip((idx - k0) / 30, 0, 1)

        red = np.zeros(n)                              # AR(1) red measurement noise
        for k in range(1, n):
            red[k] = 0.85 * red[k - 1] + rng.normal(0, 1.0)
        white = rng.normal(0, float(noise_mm), n)
        steps = np.zeros(n)                            # rare abrupt settlement jumps
        for _ in range(rng.integers(0, 3)):
            k0 = rng.integers(0, n)
            steps[k0:] += rng.uniform(4, 18)

        settlement = (secular + seasonal
                      + np.cumsum(extra_rate) * dt
                      + steps + 1.6 * red + white + 0.8 * common_mode)
        settlement -= settlement[0]

        frames.append(pd.DataFrame({
            "station_id": f"SYN{i:03d}",
            "date": dates,
            "lat": lats[i],
            "lon": lons[i],
            "settlement_mm": settlement,
            "data_source": "synthetic_fallback",
        }))
    out = pd.concat(frames, ignore_index=True)
    log.info("synthetic surrogate: %d rows, %d stations", len(out), n_stations)
    return out[_GD_COLUMNS]
