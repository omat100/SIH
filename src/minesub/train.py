"""Train a risk classifier and evaluate it on the held-out (latest) window."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .datasplit import temporal_split
from .evaluate import evaluate_predictions, evaluate_regression
from .features import build_sequences, feature_columns
from .utils import get_logger, set_seed

log = get_logger("minesub.train")

_MODEL_FILES = {"gbdt": "gbdt.joblib", "torch": "lstm.pt"}


def _load_inputs(cfg: Config):
    proc = cfg.paths["processed"]
    samples = pd.read_parquet(proc / "samples_labeled.parquet")
    ts = pd.read_parquet(proc / "timeseries.parquet")
    return samples, ts


def train(cfg: Config, model_name: str) -> dict:
    if model_name not in _MODEL_FILES:
        raise ValueError(f"model must be one of {list(_MODEL_FILES)}")
    set_seed(int(cfg["seed"]))
    cfg.ensure_dirs()
    samples, ts = _load_inputs(cfg)

    synthetic = bool((samples["data_source_disp"] == "synthetic_fallback").all())
    if synthetic:
        log.warning("=" * 70)
        log.warning("Training on the SYNTHETIC surrogate (CA DWR feed not used).")
        log.warning("Metrics below exercise the pipeline; they are NOT a benchmark.")
        log.warning("Connect the real CA DWR feed for a meaningful evaluation.")
        log.warning("=" * 70)

    task = cfg["labels"].get("task", "classification")
    tr, va, te = temporal_split(cfg, samples)
    y = samples["y_reg" if task == "regression" else "y"].to_numpy(
        dtype=float if task == "regression" else int)
    reports = cfg.paths["reports"]
    models_dir = cfg.paths["models"]
    log.info("task=%s  model=%s", task, model_name)

    if model_name == "gbdt":
        from .models.gbdt_model import GbdtRiskModel

        feats = feature_columns(samples)
        X = samples[feats].to_numpy(dtype=float)
        model = GbdtRiskModel(cfg, feats, task=task)
        model.fit(X[tr], y[tr], X[va], y[va])
        pred = model.predict(X[te]) if task == "regression" else model.predict_proba(X[te])
        model.save(models_dir / _MODEL_FILES["gbdt"])

        imp = model.feature_importance()
        if not imp.empty:
            imp.to_csv(reports / "feature_importance_gbdt.csv", header=["perm_importance"])
            log.info("top features:\n%s", imp.head(12).to_string())
        extra = {"top_features": imp.head(12).round(4).to_dict()}
    else:
        from .models.torch_model import LSTMRiskModel

        seq = build_sequences(cfg, samples, ts)
        model = LSTMRiskModel(cfg, task=task)
        model.fit(seq[tr], y[tr], seq[va], y[va])
        pred = model.predict(seq[te]) if task == "regression" else model.predict_proba(seq[te])
        model.save(models_dir / _MODEL_FILES["torch"])
        extra = {"device": model.device, "seq_len": int(cfg["model"]["torch"]["seq_len"])}

    extra["synthetic_data"] = synthetic
    if task == "regression":
        metrics = evaluate_regression(y[te], pred, reports, prefix=model_name,
                                      cfg_eval=cfg.get("evaluate", {}), extra=extra)
    else:
        metrics = evaluate_predictions(y[te], pred, reports, prefix=model_name, extra=extra)
    return metrics
