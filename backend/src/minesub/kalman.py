"""Constant-acceleration Kalman filter + RTS smoother for the settlement channel.

The raw ``distance_mm`` / ``tilt_deg`` series carry AR(1) "red" measurement noise
(see ``data/sources.py``), so a plain ``polyfit`` slope over a 30-day window is a
noisy rate estimate. A constant-acceleration state-space model instead tracks

    state x = [level, rate, accel]           (mm, mm/day, mm/day^2)

and returns a smoothed level/rate/accel at every step. It runs online (forward
filter) so it is deployable on a field node; ``smooth=True`` adds a backward RTS
pass that is still causal *within the observed window* (it never sees the
forward horizon), which is what feature extraction wants.
"""
from __future__ import annotations

import numpy as np

__all__ = ["kalman_rate"]


def kalman_rate(y, dt: float = 1.0, process_var: float = 1e-4,
                meas_var: float = 4.0, smooth: bool = True) -> dict[str, np.ndarray]:
    """Return dict with ``level``, ``rate``, ``accel`` arrays aligned to ``y``.

    ``process_var`` is the variance of the random-walk driving the acceleration
    state (bigger -> the rate tracks changes faster, follows noise more).
    ``meas_var`` is the measurement-noise variance of ``y``.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = {k: np.full(n, np.nan) for k in ("level", "rate", "accel")}
    if n == 0:
        return out
    good = np.isfinite(y)
    if good.sum() < 2:
        out["level"][:] = np.nanmean(y) if good.any() else 0.0
        out["rate"][:] = 0.0
        out["accel"][:] = 0.0
        return out

    F = np.array([[1.0, dt, 0.5 * dt * dt],
                  [0.0, 1.0, dt],
                  [0.0, 0.0, 1.0]])
    H = np.array([[1.0, 0.0, 0.0]])
    # continuous white-noise-acceleration Q, discretised for step dt
    q = float(process_var)
    Q = q * np.array([
        [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
        [dt**4 / 8.0,  dt**3 / 3.0, dt**2 / 2.0],
        [dt**3 / 6.0,  dt**2 / 2.0, dt],
    ])
    R = np.array([[float(meas_var)]])

    x = np.array([np.nanmean(y[:3]) if np.isfinite(y[:3]).any() else y[good][0], 0.0, 0.0])
    P = np.diag([meas_var, 1.0, 0.1])

    xs_f = np.zeros((n, 3)); Ps_f = np.zeros((n, 3, 3))
    xs_p = np.zeros((n, 3)); Ps_p = np.zeros((n, 3, 3))

    for k in range(n):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        xs_p[k] = x; Ps_p[k] = P
        # update (skip when the measurement is missing)
        if good[k]:
            innov = y[k] - (H @ x)[0]
            S = (H @ P @ H.T + R)[0, 0]
            K = (P @ H.T / S).ravel()
            x = x + K * innov
            P = (np.eye(3) - np.outer(K, H)) @ P
        xs_f[k] = x; Ps_f[k] = P

    if smooth:
        xs = xs_f.copy()
        Ps = Ps_f.copy()
        for k in range(n - 2, -1, -1):
            A = Ps_f[k] @ F.T @ np.linalg.pinv(Ps_p[k + 1])
            xs[k] = xs_f[k] + A @ (xs[k + 1] - xs_p[k + 1])
            Ps[k] = Ps_f[k] + A @ (Ps[k + 1] - Ps_p[k + 1]) @ A.T
        est = xs
    else:
        est = xs_f

    out["level"] = est[:, 0]
    out["rate"] = est[:, 1]
    out["accel"] = est[:, 2]
    return out
