"""Linear fit of MSD vs time for Onsager L_ij."""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress

BIG = 1e6


def fit_in_best_fit_interval(
    f: np.ndarray,
    times: np.ndarray,
    start: int,
    end: int,
) -> tuple[float, dict]:
    slope, intercept, r_value, p_value, std_err = linregress(
        times[start:end], f[start:end]
    )
    min_mae = BIG
    fit_dict: dict = {"fit_err": std_err, "slope_err": min_mae, "interval": [start, end]}
    int_len = max(2, (end - start) // 10)
    scale = 1
    slope_tol = 0.5
    best_idx = 0

    while scale < 10 and int_len > 1:
        intervals = np.linspace(start, end, int_len, dtype=int)
        for i in range(len(intervals) - 1):
            a, b = intervals[i], intervals[i + 1]
            with np.errstate(invalid="ignore"):
                sl, _, _, _, _ = linregress(
                    np.log(times[a:b]), np.log(f[a:b])
                )
            mae = np.abs(sl - 1)
            if mae < min_mae and mae < slope_tol:
                min_mae = mae
                best_idx = i
                break
        scale *= 2
        int_len = max(2, (end - start) // (20 * scale))

    if int_len > 1:
        intervals = np.linspace(start, end, int_len, dtype=int)
        best_idx = min(best_idx, len(intervals) - 2)
        a, b = intervals[best_idx], intervals[best_idx + 1]
        slope, _, _, _, std_err = linregress(times[a:b], f[a:b])
        fit_dict = {"fit_err": std_err, "slope_err": min_mae, "interval": [int(a), int(b)]}
    return slope, fit_dict


def fit_with_blockavg(
    f: np.ndarray,
    times: np.ndarray,
    start: int,
    end: int,
) -> tuple[float, dict]:
    skip = (end - start) // 20
    f = f[start + skip : end - skip]
    t = times[start + skip : end - skip]
    L = len(f)
    scale = max(1, L // 20)
    block_lengths = np.arange(1, L // 2, max(1, L // scale), dtype=int)
    min_mae = BIG
    fit_dict: dict = {"fit_err": min_mae, "slope_err": min_mae, "block_length": 1}
    best_slope = 0.0

    for bl in block_lengths:
        if bl < 1:
            continue
        n_blocks = L // bl
        if n_blocks < 2:
            continue
        ba = np.array([f[i * bl : (i + 1) * bl].mean() for i in range(n_blocks)])
        ta = np.array([t[i * bl : (i + 1) * bl].mean() for i in range(n_blocks)])
        with np.errstate(invalid="ignore"):
            sl, _, _, _, std_err = linregress(np.log(ta), np.log(ba))
        mae = np.abs(sl - 1)
        if mae < min_mae:
            sl, _, _, _, std_err = linregress(ta, ba)
            min_mae = np.abs(sl - 1)
            fit_dict = {"fit_err": std_err, "slope_err": min_mae, "block_length": int(bl)}
            best_slope = sl
    if min_mae == BIG:
        best_slope, _, _, _, std_err = linregress(t, f)
        fit_dict["fit_err"] = std_err
    return best_slope, fit_dict


def fit_data(
    f: np.ndarray,
    times: np.ndarray,
    start: int,
    end: int,
    smoothing: str = "best_fit",
) -> tuple[float, dict]:
    if smoothing == "best_fit":
        return fit_in_best_fit_interval(f, times, start, end)
    if smoothing == "blockavg":
        return fit_with_blockavg(f, times, start, end)
    slope, _, _, _, std_err = linregress(times[start:end], f[start:end])
    with np.errstate(invalid="ignore"):
        sl_log, _, _, _, _ = linregress(np.log(times[start:end]), np.log(f[start:end]))
    return slope, {"fit_err": std_err, "slope_err": float(np.abs(sl_log - 1))}
