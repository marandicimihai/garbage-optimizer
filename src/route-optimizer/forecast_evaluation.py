from __future__ import annotations

"""Forecast evaluation utilities for bin health predictions.

Provides:
- MAE
- Overflow precision / recall / f1 (point forecasts)
- Calibration summary (binned calibration error)
- Rolling backtest harness that runs predictions in a historical, causal way

Also documents recommended train/test split, rolling validation and backtesting.

Usage: run this file as a script for a small smoke test.
"""

from typing import List, Tuple, Dict
import math
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix


def mae(y_true: List[float], y_pred: List[float]) -> float:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if y_t.size == 0:
        return float('nan')
    return float(mean_absolute_error(y_t, y_p))





# MAPE removed: project uses MAE only


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

    # use confusion_matrix to extract tp/fp/fn when possible
    try:
        y_true_bin = (true_over.astype(int)).tolist()
        y_pred_bin = (pred_over.astype(int)).tolist()
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])
    except Exception:
        tp = float(np.logical_and(pred_over, true_over).sum())
        fp = float(np.logical_and(pred_over, np.logical_not(true_over)).sum())
        fn = float(np.logical_and(np.logical_not(pred_over), true_over).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {"precision": precision, "recall": recall, "tp": float(tp), "fp": float(fp), "fn": float(fn)}


# calibration_error removed per user request


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

    Returns dictionary of MAE/MAPE and overflow metrics + calibration error.
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
        "overflow": overflow_metrics(currents, trues, preds, caps),
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
    # removed: tracking only MAE and overflow precision/recall
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
        # previously recorded per-bin RMSE; no longer collected
        # mape and calibration removed
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
        # rmse removed: only MAE is returned
        "overflow_precision": None if math.isnan(precision) else precision,
        "overflow_recall": None if math.isnan(recall) else recall,
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
