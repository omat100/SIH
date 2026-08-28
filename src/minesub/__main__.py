"""CLI: ``python -m minesub <command>`` (also installed as ``minesub``)."""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .utils import get_logger

log = get_logger("minesub")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="path to config.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minesub", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("data", "features", "label", "pipeline"):
        _add_common(sub.add_parser(name, help=f"run the '{name}' stage"))

    p_tr = sub.add_parser("train", help="train a model on the labelled samples")
    _add_common(p_tr)
    p_tr.add_argument("--model", choices=["lgbm", "torch", "both"], default="lgbm")

    p_ev = sub.add_parser("evaluate", help="re-score a saved model on the test split")
    _add_common(p_ev)
    p_ev.add_argument("--model", choices=["lgbm", "torch"], default="lgbm")

    p_pr = sub.add_parser("predict", help="run live inference on a JSON list of readings")
    _add_common(p_pr)
    p_pr.add_argument("--model", choices=["lgbm", "torch"], default="lgbm")
    p_pr.add_argument("--input", required=True, help="JSON file: list of reading objects")

    p_demo = sub.add_parser("demo", help="pipeline + train both models + sample prediction")
    _add_common(p_demo)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "data":
        from .data.build import build_timeseries
        build_timeseries(cfg)
    elif args.cmd == "features":
        from .features import build_feature_table
        build_feature_table(cfg)
    elif args.cmd == "label":
        import pandas as pd
        from .labels import add_risk_labels
        add_risk_labels(cfg, pd.read_parquet(cfg.paths["processed"] / "samples.parquet"))
    elif args.cmd == "pipeline":
        from .pipeline import run_pipeline
        run_pipeline(cfg)
    elif args.cmd == "train":
        from .train import train
        names = ["lgbm", "torch"] if args.model == "both" else [args.model]
        summary = {n: train(cfg, n) for n in names}
        print(json.dumps({k: _slim(v) for k, v in summary.items()}, indent=2))
    elif args.cmd == "evaluate":
        _evaluate_saved(cfg, args.model)
    elif args.cmd == "predict":
        from .predict import predict
        with open(args.input, "r", encoding="utf-8") as fh:
            readings = json.load(fh)
        print(json.dumps(predict(readings, model=args.model, cfg=cfg), indent=2))
    elif args.cmd == "demo":
        _demo(cfg)
    return 0


def _slim(m: dict) -> dict:
    keys = ("accuracy", "macro_f1", "recall_critical", "precision_critical", "pr_auc_ovr")
    return {k: m[k] for k in keys if k in m}


def _evaluate_saved(cfg, model_name: str) -> None:
    import pandas as pd

    from .datasplit import temporal_split
    from .evaluate import evaluate_predictions
    from .features import build_sequences, feature_columns

    proc = cfg.paths["processed"]
    samples = pd.read_parquet(proc / "samples_labeled.parquet")
    _, _, te = temporal_split(cfg, samples)
    y = samples["y"].to_numpy()

    if model_name == "lgbm":
        from .models.lgbm_model import LgbmRiskModel
        m = LgbmRiskModel.load(cfg, cfg.paths["models"] / "lgbm.joblib")
        proba = m.predict_proba(samples[feature_columns(samples)].to_numpy(dtype=float)[te])
    else:
        from .models.torch_model import LSTMRiskModel
        ts = pd.read_parquet(proc / "timeseries.parquet")
        m = LSTMRiskModel.load(cfg, cfg.paths["models"] / "lstm.pt")
        proba = m.predict_proba(build_sequences(cfg, samples, ts)[te])
    evaluate_predictions(y[te], proba, cfg.paths["reports"], prefix=model_name)


def _demo(cfg) -> None:
    import numpy as np
    import pandas as pd

    from .pipeline import run_pipeline
    from .predict import predict
    from .train import train

    run_pipeline(cfg)
    results = {n: _slim(train(cfg, n)) for n in ("lgbm", "torch")}
    print(json.dumps(results, indent=2))

    # a fabricated "accelerating" recent window to show the inference shape
    n = max(int(cfg["features"]["backward_window_days"]),
            int(cfg["model"]["torch"]["seq_len"])) + 5
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n)
    readings = [
        {
            "date": d.isoformat(),
            "temp_c": float(18 + 6 * np.sin(2 * np.pi * i / 365)),
            "humidity_pct": float(55 + 5 * np.cos(2 * np.pi * i / 365)),
            "tilt_deg": float(0.002 * i + 2e-5 * i * i),
            "distance_mm": float(0.4 * i + 0.01 * i * i),   # accelerating settlement
        }
        for d, i in zip(dates, t)
    ]
    for model in ("lgbm", "torch"):
        print(f"\n--- sample predict ({model}) ---")
        print(json.dumps(predict(readings, model=model, cfg=cfg), indent=2))


if __name__ == "__main__":
    sys.exit(main())
