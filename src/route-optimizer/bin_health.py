from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np


def _day_index_map(day_labels: List[str]) -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(sorted(day_labels))}


def _recent_window(series: List[float], window: int = 7) -> List[float]:
    if window <= 0:
        return list(series)
    return list(series[-window:])


def _weighted_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    weights = np.linspace(0.6, 1.2, len(values))
    return float(np.average(np.asarray(values, dtype=float), weights=weights))


def _estimate_capacity(series: List[float], city_baseline: float) -> float:
    recent = [value for value in series if value > 1e-9]
    if recent:
        typical = float(np.median(recent))
        peak = float(np.percentile(recent, 90))
        volatility = float(np.std(recent))
    else:
        typical = float(np.mean(series)) if series else 0.0
        peak = typical
        volatility = float(np.std(series)) if series else 0.0

    capacity = 6.0 + typical * 4.5 + peak * 1.8 + volatility * 0.8
    if city_baseline > 0.0:
        capacity = max(capacity, city_baseline * 2.8)
    return float(max(6.0, capacity))


def _forecast_next_day(series: List[float]) -> float:
    recent = _recent_window(series)
    if not recent:
        return 0.0

    smoothed = _weighted_mean(recent)
    if len(recent) < 3 or not any(value > 1e-9 for value in recent):
        return float(max(0.0, smoothed))

    x = np.arange(len(recent), dtype=float)
    y = np.asarray(recent, dtype=float)
    try:
        slope, _intercept = np.polyfit(x, y, 1)
    except Exception:
        slope = 0.0

    forecast = smoothed + float(slope) * 0.75
    return float(max(0.0, forecast))


def build_daily_series(
    bins: List[dict], waste_events: List[dict], day_labels: List[str]
) -> Dict[int, List[float]]:
    """Return per-bin daily weight series keyed by binId."""
    day_index = _day_index_map(day_labels)
    series: Dict[int, List[float]] = {}
    for b in bins:
        series[int(b["binId"])] = [0.0] * len(day_labels)

    for ev in waste_events:
        label = str(ev.get("date"))
        if label not in day_index:
            continue
        idx = day_index[label]
        bid = int(ev.get("binId"))
        wt = float(ev.get("weight", 0.0))
        series.setdefault(bid, [0.0] * len(day_labels))[idx] += wt
    return series


def predict_next_day_weight(series: List[float]) -> float:
    return _forecast_next_day(series)


def compute_bin_health(
    bins: List[dict], waste_events: List[dict], day_labels: List[str]
) -> Dict[int, Dict[str, object]]:
    """Compute per-bin health predictions and scores.

    Outputs (per binId):
    - urgency: 0-100
    - hours_until_overflow: float|None
    - anomaly_score: float (z-score)
    - expected_collected_kg: float
    - days_since_last_collection: int
    - fill_ratio: float
    - daily_series: list[float]
    """
    daily_series = build_daily_series(bins, waste_events, day_labels)
    city_daily_totals = [sum(series[index] for series in daily_series.values()) for index in range(len(day_labels))]
    if city_daily_totals and bins:
        city_baseline = float(np.mean(_recent_window(city_daily_totals))) / float(max(1, len(bins)))
    else:
        city_baseline = 0.0

    # map last day label to date for days_since computation
    last_day = None
    if day_labels:
        try:
            last_day = datetime.fromisoformat(day_labels[-1]).date()
        except Exception:
            last_day = None

    results: Dict[int, Dict[str, object]] = {}
    for b in bins:
        bid = int(b["binId"])
        series = daily_series.get(bid, [])
        recent = _recent_window(series)
        current_load = float(recent[-1]) if recent else 0.0
        expected_next = predict_next_day_weight(recent)
        capacity_kg = _estimate_capacity(recent, city_baseline)

        fill_ratio = min(1.0, current_load / capacity_kg) if capacity_kg > 0 else 0.0
        projected_fill_ratio = min(1.0, (current_load + expected_next) / capacity_kg) if capacity_kg > 0 else 0.0

        # Compare the latest day against the bin's own recent behavior.
        mean = _weighted_mean(recent)
        std = float(np.std(recent)) if recent else 0.0
        anomaly = 0.0
        if std > 1e-9:
            anomaly = (current_load - mean) / std
        anomaly_score = float(max(0.0, min(5.0, abs(anomaly))))

        # hours until overflow: forecast the time it would take to consume the remaining headroom.
        daily_rate = expected_next if expected_next > 0 else max(mean, city_baseline, 1e-3)
        rate_per_hour = daily_rate / 24.0
        remaining = max(0.0, capacity_kg - current_load)
        hours_until_overflow = float(remaining / rate_per_hour) if rate_per_hour > 0 else None

        # days since last collection (last day where series value > 0)
        days_since = 999
        try:
            if series:
                last_nonzero = next((i for i in range(len(series) - 1, -1, -1) if series[i] > 1e-9), None)
                if last_nonzero is None:
                    days_since = len(series)
                else:
                    days_since = len(series) - 1 - last_nonzero
        except Exception:
            days_since = 999

        # simple missed-service risk: increases with days_since and forecast pressure.
        missed_risk = float(min(1.0, days_since / 7.0))
        pressure = float(max(fill_ratio, projected_fill_ratio))
        momentum = 0.0
        if mean > 1e-9:
            momentum = float(max(0.0, min(2.5, expected_next / mean)))

        # urgency 0-100: pressure plus per-bin anomaly and service gap.
        urgency = int(max(0, min(100, round(pressure * 64.0 + anomaly_score * 8.0 + missed_risk * 18.0 + momentum * 4.0))))

        results[bid] = {
            "binId": bid,
            "urgency": urgency,
            "hours_until_overflow": round(hours_until_overflow, 1) if hours_until_overflow is not None else None,
            "anomaly_score": round(anomaly_score, 3),
            "expected_collected_kg": round(expected_next, 3),
            "current_load_kg": round(current_load, 3),
            "capacity_kg": round(capacity_kg, 3),
            "fill_ratio": round(fill_ratio, 3),
            "projected_fill_ratio": round(projected_fill_ratio, 3),
            "days_since_last_collection": int(days_since),
            "daily_series": [round(value, 3) for value in series],
        }

    return results


if __name__ == "__main__":
    # quick CLI for manual runs (not used by server)
    print("bin_health module")
