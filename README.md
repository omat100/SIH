# Mine-Subsidence Early-Warning (sensor-based)

Ground-based **sensor** monitoring for mine subsidence — no satellite / InSAR.
Field nodes stream four channels: **temperature, humidity, tilt angle, distance**
(distance to a fixed overhead reference; increasing distance ⇒ ground settling).

Because no public dataset carries those four channels with a mine-subsidence
risk label, this repo trains on a **CA DWR hybrid**:

| Channel | Source in training data | Real? |
|---|---|---|
| `distance_mm` | CA DWR *Ground Surface Displacement / Land Subsidence Monitoring* (cumulative settlement) | ✅ measured |
| `tilt_deg` | derived: `arctan(differential settlement between neighbouring stations / inter-station baseline)` — standard tiltmeter interpretation | ✅ derivation of measured data |
| `temp_c`, `humidity_pct` | Meteostat daily weather, nearest station, joined on date | ✅ measured |
| `risk_class` | rule on the **forward-horizon** deformation (see below) | rule-based |

If the CA DWR feed or Meteostat is unreachable, a clearly-labelled physical
surrogate with the **same schema** is generated so the pipeline always runs
end to end (`data_source = "synthetic_fallback"` on every such row).

> Field deployment: keep the feature pipeline, swap the label rule for
> site-calibrated geotechnical thresholds (`labels.method: absolute` in
> `config/config.yaml`), and feed live readings through `minesub predict`.

## Task framing

For a sample at time `t` (one station):

* **features** summarise the backward window `[t − 30d, t]` — rates, accelerations,
  rolling spread of each channel, seasonal encodings. This is what a node has
  observed so far.
* **label** summarises the forward horizon `[t + 1, t + 14d]` — forward
  settlement rate, its acceleration, forward tilt rate → composite hazard score →
  `safe` / `watch` / `critical`. Never fed to the model.

So it is a genuine 14-day-ahead forecast, and the label cannot leak into the
features. Train/val/test is a **temporal** split (earliest windows train, latest
test).

## Models

| Role | Model | Input |
|---|---|---|
| Primary (`gbdt`) | **Histogram GBDT** (scikit-learn `HistGradientBoostingClassifier`), 3-class, class-balanced, early stopping | windowed feature vector |
| Secondary (`torch`) | **PyTorch** 2-layer LSTM + MLP head | raw `[seq_len, 4]` daily sequence |

The primary backend is scikit-learn's histogram GBDT (same algorithm family as
LightGBM/XGBoost, no native OpenMP dependency). LightGBM is available as an
optional drop-in — `pip install -e ".[lightgbm]"` and set
`model.gbdt.backend: lightgbm` — but some macOS LightGBM builds crash in native
code, so it is not the default.

Metrics (written to `reports/`): accuracy, macro-F1, **recall on `critical`**,
one-vs-rest PR-AUC, confusion-matrix PNG, and permutation feature importances.

## Setup (Python 3.11)

```bash
python3.11 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e .
```

## Run

```bash
# assemble dataset + train both models + a sample prediction
.venv/bin/python -m minesub demo

# or step by step
.venv/bin/python -m minesub pipeline            # -> data/processed/*.parquet
.venv/bin/python -m minesub train --model both  # -> artifacts/models/, reports/
.venv/bin/python -m minesub evaluate --model gbdt
.venv/bin/python -m minesub predict --model gbdt --input my_readings.json
```

`scripts/run_pipeline.sh` chains pipeline + train.

### `predict` input format

```json
[
  {"date": "2024-01-01", "temp_c": 17.9, "humidity_pct": 58.1, "tilt_deg": 0.004, "distance_mm": 12.7},
  {"date": "2024-01-02", "temp_c": 18.3, "humidity_pct": 57.4, "tilt_deg": 0.005, "distance_mm": 13.1}
]
```
≥ 30 daily readings for `gbdt`, ≥ 30 for `torch` (`seq_len`). Returns the risk
class, class probabilities, the 14-day horizon, and the raw rate/acceleration
signals that drove it.

## Layout

```
config/config.yaml         all knobs: dates, windows, thresholds, model params
src/minesub/
  data/sources.py          CA DWR + Meteostat download, synthetic fallbacks
  data/build.py            daily resample, tilt derivation, weather join
  features.py              backward-window features + forward-horizon targets
  labels.py                hazard score -> safe/watch/critical
  datasplit.py             temporal split
  models/gbdt_model.py     primary (histogram GBDT)
  models/torch_model.py    secondary (PyTorch LSTM)
  train.py / evaluate.py / predict.py / pipeline.py
tests/test_smoke.py        offline end-to-end check
```

## What's still missing / next

* **Real CA DWR schema pinning** — the CKAN dump's column names/units have
  changed across releases; `data/sources.py` normalises heuristically. Once you
  download it, pin the exact resource id + column map in config.
* **Rain gauge** — humidity is only a weak proxy for saturation; add `rainfall_mm`
  if the field kit allows it (it's the real hydrological driver).
* **Vibration / microseismic** — the strongest subsidence precursor; not in the
  4-channel kit. `seismic-bumps` (UCI) could pretrain a separate head if you add
  a geophone later.
* **Per-node baselines** — field thresholds should be learned per installation
  during a stable commissioning period.
