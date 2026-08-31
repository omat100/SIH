"""Primary model: histogram gradient-boosted trees on the windowed features.

Uses scikit-learn's ``HistGradientBoostingClassifier`` (same histogram-GBDT
family as LightGBM/XGBoost, but pure-Python/NumPy with no external OpenMP
library, so it can't hit the native crashes some LightGBM macOS builds show).
Set ``model.gbdt.backend: lightgbm`` in config to use LightGBM instead if it is
installed and stable on your platform.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight

from .. import RISK_CLASSES
from ..config import Config
from ..utils import get_logger

log = get_logger("minesub.models.gbdt")


def _matrix(X) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(X, dtype=np.float64))


def _sample_weights(y: np.ndarray) -> np.ndarray:
    classes = np.arange(len(RISK_CLASSES))
    w = compute_class_weight("balanced", classes=classes, y=y)
    return w[y]


class GbdtRiskModel:
    def __init__(self, cfg: Config, feature_cols: list[str]):
        self.cfg = cfg
        self.feature_cols = feature_cols
        self.model = None
        self._imp: pd.Series | None = None

    def fit(self, Xtr, ytr, Xva, yva):
        p = self.cfg["model"]["gbdt"]
        Xtr, Xva = _matrix(Xtr), _matrix(Xva)
        ytr = np.asarray(ytr)

        self.model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=float(p["learning_rate"]),
            max_iter=int(p["max_iter"]),
            max_leaf_nodes=int(p["max_leaf_nodes"]),
            max_depth=(None if int(p["max_depth"]) < 0 else int(p["max_depth"])),
            min_samples_leaf=int(p["min_samples_leaf"]),
            l2_regularization=float(p["l2_regularization"]),
            early_stopping=True,
            validation_fraction=float(p["validation_fraction"]),
            n_iter_no_change=int(p["n_iter_no_change"]),
            random_state=int(self.cfg["seed"]),
        )
        self.model.fit(Xtr, ytr, sample_weight=_sample_weights(ytr))
        log.info("trained; n_iter=%d (of max %d)", self.model.n_iter_, int(p["max_iter"]))

        # permutation importance on the external validation split
        try:
            r = permutation_importance(
                self.model, Xva, np.asarray(yva), n_repeats=5,
                random_state=int(self.cfg["seed"]), scoring="f1_macro",
            )
            self._imp = pd.Series(r.importances_mean, index=self.feature_cols
                                  ).sort_values(ascending=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("permutation importance failed: %s", exc)
            self._imp = pd.Series(dtype=float)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(_matrix(X))

    def feature_importance(self) -> pd.Series:
        return self._imp if self._imp is not None else pd.Series(dtype=float)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols,
                     "importance": self._imp, "classes": list(RISK_CLASSES)}, path)
        log.info("saved -> %s", path)

    @classmethod
    def load(cls, cfg: Config, path: str | Path) -> "GbdtRiskModel":
        blob = joblib.load(path)
        obj = cls(cfg, blob["feature_cols"])
        obj.model = blob["model"]
        obj._imp = blob.get("importance")
        return obj
