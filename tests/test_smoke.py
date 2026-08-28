"""Fast offline end-to-end check: synthetic fallback -> pipeline -> train -> predict."""
from __future__ import annotations

import copy

import pytest

from minesub.config import Config, load_config


@pytest.fixture()
def cfg(tmp_path) -> Config:
    base = load_config()
    raw = copy.deepcopy(base.raw)
    # force the synthetic surrogate (bogus dataset id) and shrink everything
    raw["data"]["ca_dwr"]["dataset_id"] = "does-not-exist-force-fallback"
    raw["data"]["min_date"] = "2020-01-01"
    raw["data"]["max_date"] = "2021-09-30"
    raw["data"]["synthetic_n_stations"] = 8
    raw["data"]["weather"]["provider"] = "none"          # seasonal fallback, no network
    raw["features"]["step_days"] = 10
    raw["model"]["torch"]["max_epochs"] = 2
    raw["model"]["torch"]["seq_len"] = 20
    raw["model"]["gbdt"]["max_iter"] = 60
    for k in raw["paths"]:
        raw["paths"][k] = str(tmp_path / raw["paths"][k])
    c = Config(raw=raw, path=base.path)
    c.ensure_dirs()
    return c


def test_pipeline_train_predict(cfg):
    from minesub.pipeline import run_pipeline
    from minesub.predict import predict
    from minesub.train import train

    labelled = run_pipeline(cfg)
    assert len(labelled) > 50
    assert set(labelled["risk_class"].cat.categories) == {"safe", "watch", "critical"}
    assert (cfg.paths["processed"] / "timeseries.parquet").exists()

    m = train(cfg, "gbdt")
    assert 0.0 <= m["accuracy"] <= 1.0
    assert "recall_critical" in m
    assert (cfg.paths["models"] / "gbdt.joblib").exists()

    m_t = train(cfg, "torch")
    assert 0.0 <= m_t["macro_f1"] <= 1.0
    assert (cfg.paths["models"] / "lstm.pt").exists()

    import pandas as pd
    ts = pd.read_parquet(cfg.paths["processed"] / "timeseries.parquet")
    one = ts[ts["station_id"] == ts["station_id"].iloc[0]].tail(40)
    readings = [
        {"date": r.date.isoformat(), "temp_c": r.temp_c, "humidity_pct": r.humidity_pct,
         "tilt_deg": r.tilt_deg, "distance_mm": r.distance_mm}
        for r in one.itertuples(index=False)
    ]
    out = predict(readings, model="gbdt", cfg=cfg)
    assert out["risk_class"] in {"safe", "watch", "critical"}
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-5
