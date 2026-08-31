"""Shared evaluation: metrics + confusion-matrix plot, written to reports/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score)

from . import RISK_CLASSES
from .utils import get_logger

log = get_logger("minesub.evaluate")


def evaluate_predictions(y_true, proba, reports_dir: Path, prefix: str,
                         extra: dict | None = None) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = proba.argmax(1)
    n_cls = len(RISK_CLASSES)

    rep = classification_report(y_true, y_pred, labels=list(range(n_cls)),
                                target_names=list(RISK_CLASSES),
                                output_dict=True, zero_division=0)
    onehot = np.eye(n_cls)[y_true]
    pr_auc = {}
    for i, name in enumerate(RISK_CLASSES):
        pr_auc[name] = (float(average_precision_score(onehot[:, i], proba[:, i]))
                        if onehot[:, i].any() else None)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    crit = RISK_CLASSES.index("critical")

    metrics = {
        "model": prefix,
        "n_samples": int(len(y_true)),
        "accuracy": float(rep["accuracy"]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(rep["macro avg"]["precision"]),
        "macro_recall": float(rep["macro avg"]["recall"]),
        "recall_critical": float(rep["critical"]["recall"]),
        "precision_critical": float(rep["critical"]["precision"]),
        "pr_auc_ovr": pr_auc,
        "per_class": {k: rep[k] for k in RISK_CLASSES},
        "confusion_matrix": cm.tolist(),
        "confusion_rows_true_cols_pred": list(RISK_CLASSES),
    }
    if extra:
        metrics.update(extra)

    out_json = reports_dir / f"metrics_{prefix}.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    _plot_confusion(cm, reports_dir / f"confusion_{prefix}.png", prefix)
    log.info("[%s] acc=%.3f  macro-F1=%.3f  recall(critical)=%.3f  PR-AUC(critical)=%s",
             prefix, metrics["accuracy"], metrics["macro_f1"],
             metrics["recall_critical"],
             f"{pr_auc['critical']:.3f}" if pr_auc["critical"] is not None else "n/a")
    log.info("wrote %s", out_json)
    return metrics


def _plot_confusion(cm, path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        log.warning("matplotlib unavailable, skipping plot: %s", exc)
        return
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(RISK_CLASSES)), RISK_CLASSES)
    ax.set_yticks(range(len(RISK_CLASSES)), RISK_CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title} — row-normalised")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info("wrote %s", path)
