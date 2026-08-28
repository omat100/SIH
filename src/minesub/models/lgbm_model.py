"""Primary model: LightGBM gradient-boosted trees on the windowed features."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.utils.class_weight import compute_class_weight

# LightGBM's multithreaded Dataset construction can race intermittently on
# macOS/arm libomp; a fixed, modest thread count makes it deterministic.
_N_JOBS = max(1, min(4, (os.cpu_count() or 2)))


def _as_matrix(X) -> np.ndarray:
    """LightGBM segfaults on some non-C-contiguous float64 blocks (pandas 3
    ``.to_numpy()`` returns F-order). Force a clean C-contiguous copy."""
    return np.ascontiguousarray(np.asarray(X, dtype=np.float64))

from .. import RISK_CLASSES
from ..config import Config
from ..utils import get_logger

log = get_logger("minesub.models.lgbm")


class LgbmRiskModel:
    def __init__(self, cfg: Config, feature_cols: list[str]):
        self.cfg = cfg
        self.feature_cols = feature_cols
        self.model: LGBMClassifier | None = None

    def fit(self, Xtr, ytr, Xva, yva):
        p = self.cfg["model"]["lgbm"]
        Xtr, Xva = _as_matrix(Xtr), _as_matrix(Xva)
        classes = np.arange(len(RISK_CLASSES))
        w = compute_class_weight("balanced", classes=classes, y=ytr)
        cw = {int(c): float(wi) for c, wi in zip(classes, w)}
        self.model = LGBMClassifier(
            objective="multiclass",
            num_class=len(RISK_CLASSES),
            n_estimators=int(p["n_estimators"]),
            learning_rate=float(p["learning_rate"]),
            num_leaves=int(p["num_leaves"]),
            max_depth=int(p["max_depth"]),
            subsample=float(p["subsample"]),
            subsample_freq=1,
            colsample_bytree=float(p["colsample_bytree"]),
            min_child_samples=int(p["min_child_samples"]),
            class_weight=cw,
            random_state=int(self.cfg["seed"]),
            force_col_wise=True,
            n_jobs=_N_JOBS,
        )
        self.model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(int(p["early_stopping_rounds"]), verbose=False),
                       log_evaluation(0)],
        )
        log.info("trained; best_iteration=%s", self.model.best_iteration_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(_as_matrix(X))

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.feature_cols
                         ).sort_values(ascending=False)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols,
                     "classes": list(RISK_CLASSES)}, path)
        log.info("saved -> %s", path)

    @classmethod
    def load(cls, cfg: Config, path: str | Path) -> "LgbmRiskModel":
        blob = joblib.load(path)
        obj = cls(cfg, blob["feature_cols"])
        obj.model = blob["model"]
        return obj
