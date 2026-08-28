"""Leakage-free temporal split: earlier windows train, later windows test."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .utils import get_logger

log = get_logger("minesub.datasplit")


def temporal_split(cfg: Config, samples: pd.DataFrame):
    scfg = cfg["split"]
    val_f = float(scfg["val_fraction"])
    test_f = float(scfg["test_fraction"])
    order = samples["t"].rank(method="first")
    q = order / order.max()
    train_mask = q <= (1 - val_f - test_f)
    val_mask = (q > (1 - val_f - test_f)) & (q <= (1 - test_f))
    test_mask = q > (1 - test_f)

    cut_val = samples.loc[val_mask, "t"].min()
    cut_test = samples.loc[test_mask, "t"].min()
    log.info("split: train=%d (<%s)  val=%d  test=%d (>=%s)",
             int(train_mask.sum()), cut_val.date(),
             int(val_mask.sum()), int(test_mask.sum()), cut_test.date())
    return train_mask.to_numpy(), val_mask.to_numpy(), test_mask.to_numpy()
