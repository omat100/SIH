"""Shared evaluation: metrics + confusion-matrix plot, written to reports/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_curve, r2_score)

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


# ---------------------------------------------------------------------------
# Regression task: predict the forward excess settlement rate (mm/day), then
# score it as a hazard-ranking problem with a recall-tuned operating point.
# ---------------------------------------------------------------------------
def evaluate_regression(y_true, y_pred, reports_dir: Path, prefix: str,
                        cfg_eval: dict | None = None, extra: dict | None = None) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ce = cfg_eval or {}
    top_frac = float(ce.get("hazard_top_frac", 0.10))
    recall_targets = list(ce.get("recall_targets", [0.5, 0.6, 0.7, 0.8]))
    topk_fracs = list(ce.get("topk_fracs", [0.05, 0.10, 0.20]))
    op_recall = float(ce.get("operating_recall", 0.60))

    resid = y_pred - y_true
    ss = np.argsort(y_pred)  # ascending
    pear = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else None
    # Spearman = Pearson on ranks
    rt, rp = np.argsort(np.argsort(y_true)), np.argsort(np.argsort(y_pred))
    spear = float(np.corrcoef(rt, rp)[0, 1]) if len(y_true) > 2 else None

    # hazard-ranking framing: positives = top `top_frac` of the TRUE excess rate
    thr_true = np.quantile(y_true, 1.0 - top_frac)
    pos = y_true >= thr_true
    n_pos = int(pos.sum())

    prec, rec, pr_thr = precision_recall_curve(pos.astype(int), y_pred)
    pr_auc = float(average_precision_score(pos.astype(int), y_pred)) if n_pos else None

    # operating points: lowest score threshold that still reaches each target recall
    ops = {}
    order = np.argsort(-y_pred)  # predictions high -> low
    pos_sorted = pos[order]
    tp = np.cumsum(pos_sorted)
    fp = np.cumsum(~pos_sorted)
    recall_curve = tp / max(n_pos, 1)
    precision_curve = tp / np.maximum(tp + fp, 1)
    for tgt in recall_targets:
        idx = np.searchsorted(recall_curve, tgt)
        if idx < len(y_pred):
            ops[f"recall>={tgt}"] = {
                "threshold": float(y_pred[order][idx]),
                "precision": float(precision_curve[idx]),
                "recall": float(recall_curve[idx]),
                "flagged_frac": float((idx + 1) / len(y_pred)),
            }
        else:
            ops[f"recall>={tgt}"] = None

    topk = {}
    for fr in topk_fracs:
        k = max(1, int(round(fr * len(y_pred))))
        sel = order[:k]
        topk[f"top_{int(fr*100)}pct"] = {
            "precision": float(pos[sel].mean()),
            "recall": float(pos[sel].sum() / max(n_pos, 1)),
            "n": k,
        }

    chosen = ops.get(f"recall>={op_recall}") or next(
        (v for v in ops.values() if v), None)

    metrics = {
        "model": prefix,
        "task": "regression",
        "n_samples": int(len(y_true)),
        "target": "fwd_settlement_excess_mm_day",
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": pear,
        "spearman_rho": spear,
        "bias_mean_resid": float(resid.mean()),
        "hazard_top_frac": top_frac,
        "hazard_pos_threshold_true": float(thr_true),
        "hazard_n_positive": n_pos,
        "hazard_pr_auc": pr_auc,
        "operating_points": ops,
        "precision_at_topk": topk,
        "chosen_operating_point": chosen,
    }
    if extra:
        metrics.update(extra)

    out_json = reports_dir / f"metrics_{prefix}.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    _plot_regression(y_true, y_pred, pos, recall_curve, precision_curve,
                     reports_dir / f"regression_{prefix}.png", prefix)
    cop = chosen or {}
    log.info("[%s] MAE=%.4f  R2=%.3f  spearman=%.3f  PR-AUC(hazard)=%s  | op: recall=%.2f precision=%.2f flag=%.1f%%",
             prefix, metrics["mae"], metrics["r2"],
             spear if spear is not None else float("nan"),
             f"{pr_auc:.3f}" if pr_auc is not None else "n/a",
             cop.get("recall", float("nan")), cop.get("precision", float("nan")),
             100 * cop.get("flagged_frac", float("nan")))
    log.info("wrote %s", out_json)
    return metrics


def _plot_regression(y_true, y_pred, pos, recall_curve, precision_curve,
                     path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        log.warning("matplotlib unavailable, skipping plot: %s", exc)
        return
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax[0].scatter(y_true, y_pred, s=6, alpha=.3)
    lo, hi = np.percentile(np.concatenate([y_true, y_pred]), [1, 99])
    ax[0].plot([lo, hi], [lo, hi], "r--", lw=1)
    ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
    ax[0].set_xlabel("true excess rate (mm/day)")
    ax[0].set_ylabel("predicted")
    ax[0].set_title(f"{title} — predicted vs true")
    ax[1].plot(recall_curve, precision_curve, lw=1.5)
    ax[1].set_xlabel("recall (top-hazard class)")
    ax[1].set_ylabel("precision")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("hazard-ranking PR curve")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info("wrote %s", path)
