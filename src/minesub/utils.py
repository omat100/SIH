"""Small shared helpers: logging, seeding, geodesy."""
from __future__ import annotations

import logging
import os
import random

# Cap OpenMP threads before numpy / lightgbm / torch pull in their runtimes —
# avoids an intermittent libomp race in LightGBM's Dataset build on macOS/arm.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

_LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False        # avoid double emit via parent handlers
    return logger


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:  # torch optional at import time
        pass


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. Accepts scalars or numpy arrays."""
    r = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def pick_torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
