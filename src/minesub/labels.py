"""Turn the forward-horizon deformation into a 3-class early-warning label.

Two strategies (``labels.method`` in config):

* ``quantile``  – build a composite hazard score from the forward settlement
  rate, its acceleration and the forward tilt rate, then cut it at two
  quantiles. Always yields a trainable 3-class problem, whatever the input
  data looks like. This is the default for training on the CA DWR hybrid.

* ``absolute``  – fixed geotechnical thresholds on the raw forward rates. This
  is what a field deployment should switch to, with site-calibrated numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import RISK_CLASSES
from .config import Config
from .utils import get_logger

log = get_logger("minesub.labels")

_CLASS_TO_INT = {c: i for i, c in enumerate(RISK_CLASSES)}


def _robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    iqr = x.quantile(0.75) - x.quantile(0.25)
    scale = iqr / 1.349 if iqr > 1e-9 else (x.std() or 1.0)
    return (x - med) / scale


def add_risk_labels(cfg: Config, samples: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    lcfg = cfg["labels"]
    method = lcfg.get("method", "quantile")
    s = samples.copy()

    hazard = (
        _robust_z(s["fwd_settlement_rate"])
        + _robust_z(s["fwd_settlement_accel"]).clip(lower=0.0)
        + _robust_z(s["fwd_tilt_rate"].abs())
    )
    s["hazard_score"] = hazard

    if method == "absolute":
        a = lcfg["absolute"]
        sr, tr = s["fwd_settlement_rate"], s["fwd_tilt_rate"].abs()
        acc = s["fwd_settlement_accel"]
        crit = (
            (sr >= a["settlement_rate_mm_day"]["critical"])
            | (tr >= a["tilt_rate_deg_day"]["critical"])
            | (acc >= a["settlement_accel_mm_day2_critical"])
        )
        watch = (
            (sr >= a["settlement_rate_mm_day"]["watch"])
            | (tr >= a["tilt_rate_deg_day"]["watch"])
        )
        cls = np.where(crit, "critical", np.where(watch, "watch", "safe"))
    else:  # quantile
        qw = s["hazard_score"].quantile(lcfg.get("quantile_watch", 0.60))
        qc = s["hazard_score"].quantile(lcfg.get("quantile_critical", 0.90))
        cls = np.where(s["hazard_score"] >= qc, "critical",
                       np.where(s["hazard_score"] >= qw, "watch", "safe"))

    s["risk_class"] = pd.Categorical(cls, categories=list(RISK_CLASSES), ordered=True)
    s["y"] = s["risk_class"].map(_CLASS_TO_INT).astype(int)

    dist = s["risk_class"].value_counts().reindex(RISK_CLASSES)
    log.info("label method=%s  distribution: %s", method,
             {k: int(v) for k, v in dist.items()})
    if (dist.fillna(0) == 0).any():
        log.warning("a risk class is empty under method=%s — consider 'quantile' "
                    "or re-tuning thresholds", method)

    if save:
        out = cfg.paths["processed"] / "samples_labeled.parquet"
        s.to_parquet(out, index=False)
        log.info("wrote %s", out)
    return s
