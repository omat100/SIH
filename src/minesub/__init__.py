"""Sensor-based mine-subsidence early-warning.

Pipeline: CA DWR ground-surface displacement (real) -> derived tilt (real
geometric relationship) -> weather join (real) -> threshold risk labels ->
LightGBM (primary) and PyTorch LSTM (secondary) 3-class classifiers.
"""

__version__ = "0.1.0"

RISK_CLASSES = ("safe", "watch", "critical")
