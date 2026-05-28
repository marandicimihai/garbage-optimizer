from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np

from global_city_model import CHISINAU_PROFILE, predict_expected_kg


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


def _haversine_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    radius_km = 6371.0
    lat_a_rad = np.radians(lat_a)
    lat_b_rad = np.radians(lat_b)
    delta_lat = np.radians(lat_b - lat_a)
    delta_lon = np.radians(lon_b - lon_a)
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat_a_rad) * np.cos(lat_b_rad) * np.sin(delta_lon / 2.0) ** 2
    return float(2.0 * radius_km * np.arcsin(np.sqrt(a)))


def _relevant_poi_weight(poi_type: str) -> float:
    type_name = str(poi_type).lower()
    if type_name in {"cafe", "restaurant", "fast_food"}:
        return 1.35
    if type_name in {"supermarket", "mall", "convenience"}:
        return 1.15
    if type_name in {"parking", "fuel", "library"}:
        return 0.70
    return 0.45


def build_city_context(
    bins: List[dict],
    waste_events: List[dict],
    day_labels: List[str],
    pois: List[dict] | None = None,
    street_lines: List[list[list[float]]] | None = None,
) -> Dict[str, float]:
    bin_coordinates = [
        (float(entry.get("lat", 0.0)), float(entry.get("lon", 0.0)))
        for entry in bins
        if entry.get("lat") is not None and entry.get("lon") is not None
    ]
    poi_coordinates = [
        (float(entry.get("lat", 0.0)), float(entry.get("lon", 0.0)))
        for entry in (pois or [])
        if entry.get("lat") is not None and entry.get("lon") is not None
    ]
    street_coordinates: list[tuple[float, float]] = []
    if street_lines:
        for line in street_lines:
            for point in line:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    street_coordinates.append((float(point[1]), float(point[0])))

    coordinates = bin_coordinates + poi_coordinates + street_coordinates
    if coordinates:
        latitudes = [lat for lat, _ in coordinates]
        longitudes = [lon for _, lon in coordinates]
        min_lat = min(latitudes)
        max_lat = max(latitudes)
        min_lon = min(longitudes)
        max_lon = max(longitudes)
        center_source = bin_coordinates or coordinates
        center_lat = float(sum(lat for lat, _ in center_source) / max(1, len(center_source)))
        center_lon = float(sum(lon for _, lon in center_source) / max(1, len(center_source)))
        width_km = abs(max_lon - min_lon) * 111.32 * max(0.2, float(np.cos(np.radians(center_lat))))
        height_km = abs(max_lat - min_lat) * 110.57
        area_km2 = float(max(1.0, width_km * height_km))
    else:
        center_lat = 47.0245
        center_lon = 28.8323
        area_km2 = 25.0

    weighted_poi_count = sum(
        _relevant_poi_weight(str(entry.get("type", entry.get("amenity", "other"))))
        for entry in (pois or [])
    )
    street_segments = 0
    if street_lines:
        for line in street_lines:
            if isinstance(line, list) and len(line) > 1:
                street_segments += len(line) - 1

    poi_density_index = float(min(2.5, weighted_poi_count / max(1.0, len(bins) * 5.0)))
    street_density_index = float(min(2.5, street_segments / max(1.0, len(bins) * 18.0)))

    daily_series = build_daily_series(bins, waste_events, day_labels) if day_labels else {}
    city_daily_totals = [sum(series[index] for series in daily_series.values()) for index in range(len(day_labels))] if daily_series else []
    observed_baseline = 0.0
    if city_daily_totals and bins:
        observed_baseline = float(np.mean(_recent_window(city_daily_totals))) / float(max(1, len(bins)))

    model_baseline = float(
        0.9
        + 0.32 * float(CHISINAU_PROFILE["population_millions"])
        + 0.00010 * float(CHISINAU_PROFILE["density_km2"])
        + 0.045 * float(CHISINAU_PROFILE["avg_temp_c"])
        + 0.060 * float(CHISINAU_PROFILE["gdp_per_capita_k"])
        + 0.45 * poi_density_index
        + 0.32 * street_density_index
        - 1.10 * float(CHISINAU_PROFILE["recycling_rate"])
    )
    city_baseline_kg = float(max(1.0, model_baseline, observed_baseline))

    return {
        "population_millions": float(CHISINAU_PROFILE["population_millions"]),
        "density_km2": float(CHISINAU_PROFILE["density_km2"]),
        "avg_temp_c": float(CHISINAU_PROFILE["avg_temp_c"]),
        "recycling_rate": float(CHISINAU_PROFILE["recycling_rate"]),
        "gdp_per_capita_k": float(CHISINAU_PROFILE["gdp_per_capita_k"]),
        "poi_density_index": poi_density_index,
        "street_density_index": street_density_index,
        "city_baseline_kg": city_baseline_kg,
        "city_center_lat": center_lat,
        "city_center_lon": center_lon,
        "area_km2": area_km2,
    }


def _bin_activity_score(bin_lat: float, bin_lon: float, pois: List[dict] | None) -> float:
    if not pois:
        return 0.5

    score = 0.0
    for entry in pois:
        try:
            poi_lat = float(entry.get("lat", entry.get("y", 0.0)))
            poi_lon = float(entry.get("lon", entry.get("x", 0.0)))
        except (TypeError, ValueError):
            continue
        distance_km = _haversine_km(bin_lat, bin_lon, poi_lat, poi_lon)
        if distance_km > 1.5:
            continue
        proximity = max(0.0, 1.5 - distance_km) / 1.5
        score += _relevant_poi_weight(str(entry.get("type", entry.get("amenity", "other")))) * proximity

    return float(min(2.5, 0.35 + score / 3.5))


def _nearest_poi_distance_km(bin_lat: float, bin_lon: float, pois: List[dict] | None) -> float:
    if not pois:
        return 2.0

    best = None
    for entry in pois:
        try:
            poi_lat = float(entry.get("lat", entry.get("y", 0.0)))
            poi_lon = float(entry.get("lon", entry.get("x", 0.0)))
        except (TypeError, ValueError):
            continue
        distance_km = _haversine_km(bin_lat, bin_lon, poi_lat, poi_lon)
        if best is None or distance_km < best:
            best = distance_km

    return float(best if best is not None else 2.0)


def _forecast_next_day(
    series: List[float],
    *,
    city_context: Dict[str, float] | None = None,
    bin_context: Dict[str, float] | None = None,
) -> float:
    recent = _recent_window(series)
    if not recent:
        return 0.0

    current_load = float(recent[-1])
    if len(recent) == 1:
        days_since = 0.0 if current_load > 1e-9 else 1.0
    else:
        last_nonzero = next((index for index in range(len(recent) - 1, -1, -1) if recent[index] > 1e-9), None)
        days_since = float(len(recent) - 1 - last_nonzero) if last_nonzero is not None else float(len(recent))

    context = city_context or {}
    context_bin = bin_context or {}
    feature_values = {
        "population_millions": float(context.get("population_millions", CHISINAU_PROFILE["population_millions"])),
        "density_km2": float(context.get("density_km2", CHISINAU_PROFILE["density_km2"])),
        "avg_temp_c": float(context.get("avg_temp_c", CHISINAU_PROFILE["avg_temp_c"])),
        "recycling_rate": float(context.get("recycling_rate", CHISINAU_PROFILE["recycling_rate"])),
        "gdp_per_capita_k": float(context.get("gdp_per_capita_k", CHISINAU_PROFILE["gdp_per_capita_k"])),
        "poi_density_index": float(context.get("poi_density_index", 0.8)),
        "street_density_index": float(context.get("street_density_index", 0.7)),
        "city_baseline_kg": float(context.get("city_baseline_kg", 1.5)),
        "current_load_kg": current_load,
        "days_since_collection": float(context_bin.get("days_since_collection", days_since)),
        "local_activity_score": float(context_bin.get("local_activity_score", 0.5)),
        "center_distance_km": float(context_bin.get("center_distance_km", 0.0)),
    }
    return predict_expected_kg(feature_values)


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


def predict_next_day_weight(
    series: List[float],
    *,
    city_context: Dict[str, float] | None = None,
    bin_context: Dict[str, float] | None = None,
) -> float:
    return _forecast_next_day(series, city_context=city_context, bin_context=bin_context)


def compute_bin_health(
    bins: List[dict], waste_events: List[dict], day_labels: List[str], pois: List[dict] | None = None, street_lines: List[list[list[float]]] | None = None
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
    city_context = build_city_context(bins, waste_events, day_labels, pois=pois, street_lines=street_lines)
    city_baseline = float(city_context.get("city_baseline_kg", 0.0))

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
        last_nonzero = next((index for index in range(len(recent) - 1, -1, -1) if recent[index] > 1e-9), None) if recent else None
        days_since = int(len(recent) - 1 - last_nonzero) if last_nonzero is not None else int(len(recent)) if recent else 999
        bin_lat = float(b.get("lat", city_context.get("city_center_lat", 47.0245)))
        bin_lon = float(b.get("lon", city_context.get("city_center_lon", 28.8323)))
        bin_activity = _bin_activity_score(bin_lat, bin_lon, pois)
        nearest_poi_distance = _nearest_poi_distance_km(bin_lat, bin_lon, pois)
        center_distance = _haversine_km(bin_lat, bin_lon, float(city_context.get("city_center_lat", bin_lat)), float(city_context.get("city_center_lon", bin_lon)))
        local_activity_score = float(min(2.5, bin_activity + max(0.0, 1.2 - nearest_poi_distance) * 0.6 + max(0.0, 1.5 - center_distance) * 0.15))
        expected_next = _forecast_next_day(
            recent,
            city_context=city_context,
            bin_context={
                "days_since_collection": float(days_since),
                "local_activity_score": local_activity_score,
                "center_distance_km": float(center_distance),
            },
        )
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
            "local_activity_score": round(local_activity_score, 3),
            "nearest_poi_distance_km": round(nearest_poi_distance, 3),
            "daily_series": [round(value, 3) for value in series],
        }

    return results


if __name__ == "__main__":
    # quick CLI for manual runs (not used by server)
    print("bin_health module")
