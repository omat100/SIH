"""End-to-end data assembly: raw sources -> labelled windowed samples."""
from __future__ import annotations

import pandas as pd

from .config import Config
from .data.build import build_timeseries
from .features import build_feature_table
from .labels import add_risk_labels
from .utils import get_logger, set_seed

log = get_logger("minesub.pipeline")


def run_pipeline(cfg: Config) -> pd.DataFrame:
    set_seed(int(cfg["seed"]))
    cfg.ensure_dirs()
    log.info("[1/3] building per-station daily timeseries (CA DWR + tilt + weather)")
    ts = build_timeseries(cfg, save=True)
    log.info("[2/3] building windowed feature table")
    samples = build_feature_table(cfg, ts=ts, save=True)
    log.info("[3/3] assigning 3-class early-warning labels")
    labelled = add_risk_labels(cfg, samples, save=True)
    log.info("pipeline complete: %d labelled samples -> %s",
             len(labelled), cfg.paths["processed"] / "samples_labeled.parquet")
    return labelled
