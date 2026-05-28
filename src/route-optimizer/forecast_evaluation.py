from __future__ import annotations

"""Forecast evaluation utilities for bin health predictions.

Provides:
- MAE, RMSE, MAPE
- Overflow precision / recall / f1 (point forecasts)
- Calibration summary (binned calibration error)
- Rolling backtest harness that runs predictions in a historical, causal way

Also documents recommended train/test split, rolling validation and backtesting.

Usage: run this file as a script for a small smoke test.
"""

from typing import List, Tuple, Dict
import math
import numpy as np


def mae(y_true: List[float], y_pred: List[float]) -> float:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_t - y_p))) if y_t.size else float('nan')


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2))) if y_t.size else float('nan')


def mape(y_true: List[float], y_pred: List[float]) -> float:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    # avoid division by zero: ignore points where true == 0
    mask = np.abs(y_t) > 1e-9
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])))


def overflow_metrics(
    current: List[float],
    y_true: List[float],
    y_pred: List[float],
    capacity: List[float],
    threshold: float = 1.0,
) -> Dict[str, float]:
    """Compute precision/recall/f1 for overflow prediction.

    At each sample t we have `current[t]` (current load at day t), and predict
    next-day collected `y_pred[t]`. We declare predicted overflow if
    (current + y_pred) >= threshold * capacity. The true overflow uses y_true.
    """
    cur = np.asarray(current, dtype=float)
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    cap = np.asarray(capacity, dtype=float)

    pred_over = (cur + yp) >= (threshold * cap)
    true_over = (cur + yt) >= (threshold * cap)

    tp = float(np.logical_and(pred_over, true_over).sum())
    fp = float(np.logical_and(pred_over, np.logical_not(true_over)).sum())
    fn = float(np.logical_and(np.logical_not(pred_over), true_over).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def calibration_error(y_true: List[float], y_pred: List[float], n_bins: int = 10) -> float:
    """Simple calibration: group predictions into `n_bins` buckets and compare
    average predicted vs average actual per bucket. Returns RMSE between the
    mean predicted and mean observed across bins (lower is better).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if y_p.size == 0:
        return float('nan')
    # compute quantile bins on predictions
    percentiles = np.linspace(0.0, 100.0, n_bins + 1)
    bins = np.percentile(y_p, percentiles)
    # to avoid zero-width bins, add tiny jitter
    bins = np.unique(bins)
    if bins.size <= 1:
        return float('nan')

    # digitize
    inds = np.digitize(y_p, bins, right=False)
    mean_preds = []
    mean_trues = []
    for b in range(1, bins.size + 1):
        mask = inds == b
        if not mask.any():
            continue
        mean_preds.append(float(np.mean(y_p[mask])))
        mean_trues.append(float(np.mean(y_t[mask])))
    if not mean_preds:
        return float('nan')
    return float(np.sqrt(np.mean((np.asarray(mean_preds) - np.asarray(mean_trues)) ** 2)))


def rolling_backtest_for_series(
    series: List[float],
    capacity: float,
    forecast_fn,
    window: int = 7,
) -> Dict[str, float]:
    """Run a simple rolling backtest on a single-bin series.

    For t from (window-1) .. len(series)-2:
      - use series[:t+1] (most recent `window` inside `forecast_fn`) to predict day t+1
      - collect predictions and actuals

    Returns dictionary of MAE/RMSE/MAPE and overflow metrics + calibration error.
    """
    n = len(series)
    if n < 2:
        return {}
    # If the series is short (e.g. length == window) allow a smaller effective
    # window so at least one backtest prediction is produced. Use at most n-1
    # days for history when necessary.
    effective_window = min(window, max(1, n - 1))

    preds = []
    trues = []
    currents = []
    caps = []
    for t in range(effective_window - 1, n - 1):
        # input slice up to and including t using effective_window
        hist = series[max(0, t - effective_window + 1) : t + 1]
        cur = float(hist[-1]) if hist else 0.0
        yhat = float(forecast_fn(hist))
        ytrue = float(series[t + 1])
        preds.append(yhat)
        trues.append(ytrue)
        currents.append(cur)
        caps.append(capacity)

    def _safe(x):
        try:
            if x is None:
                return None
            if isinstance(x, float) and math.isnan(x):
                return None
        except Exception:
            pass
        return x

    metrics = {
        "mae": _safe(mae(trues, preds)),
        "rmse": _safe(rmse(trues, preds)),
        "mape": _safe(mape(trues, preds)),
        "overflow": overflow_metrics(currents, trues, preds, caps),
        "calibration_error": _safe(calibration_error(trues, preds)),
    }
    return metrics


def rolling_backtest_all_bins(
    daily_series: Dict[int, List[float]],
    capacity_map: Dict[int, float],
    forecast_fn,
    window: int = 7,
) -> Dict[str, object]:
    """Run rolling backtest across all bins and aggregate results.

    Returns aggregated metrics and per-bin summaries.
    """
    all_mae = []
    all_rmse = []
    all_mape = []
    all_cal = []
    # aggregate overflow counts
    agg_tp = agg_fp = agg_fn = 0.0

    per_bin = {}
    for bid, series in daily_series.items():
        cap = float(capacity_map.get(bid, 0.0))
        res = rolling_backtest_for_series(series, cap, forecast_fn, window)
        if not res:
            continue
        per_bin[bid] = res
        all_mae.append(res.get("mae", float('nan')))
        all_rmse.append(res.get("rmse", float('nan')))
        all_mape.append(res.get("mape", float('nan')))
        all_cal.append(res.get("calibration_error", float('nan')))
        ov = res.get("overflow", {})
        agg_tp += ov.get("tp", 0.0)
        agg_fp += ov.get("fp", 0.0)
        agg_fn += ov.get("fn", 0.0)

    precision = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else 0.0
    recall = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else 0.0

    def _safe_mean(lst):
        try:
            if not lst:
                return None
            v = float(np.nanmean(lst))
            return None if math.isnan(v) else v
        except Exception:
            return None

    summary = {
        "mae": _safe_mean(all_mae),
        "rmse": _safe_mean(all_rmse),
        "mape": _safe_mean(all_mape),
        "calibration_error": _safe_mean(all_cal),
        "overflow_precision": None if math.isnan(precision) else precision,
        "overflow_recall": None if math.isnan(recall) else recall,
        "overflow_f1": None if (precision + recall) <= 0 else (2 * precision * recall / (precision + recall)),
        "per_bin": per_bin,
    }
    return summary


if __name__ == "__main__":
    # small smoke test using simple synthetic series
    from bin_health import predict_next_day_weight

    # synthetic: low steady then ramp up
    series = [0.5, 0.6, 0.7, 0.8, 1.0, 1.3, 1.8, 2.5, 3.0, 2.8, 2.6]
    capacity = 6.0
    res = rolling_backtest_for_series(series, capacity, predict_next_day_weight, window=7)
    print("Rolling backtest (single series):")
    print(res)
