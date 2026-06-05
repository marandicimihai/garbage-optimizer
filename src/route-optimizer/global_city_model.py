from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class LinearRegressionModel:
    feature_names: tuple[str, ...]
    intercept: float
    coefficients: np.ndarray

    def predict(self, feature_rows: Iterable[dict[str, float]]) -> np.ndarray:
        rows = []
        for row in feature_rows:
            rows.append([float(row.get(name, 0.0)) for name in self.feature_names])
        if not rows:
            return np.asarray([], dtype=float)
        matrix = np.asarray(rows, dtype=float)
        return self.intercept + matrix @ self.coefficients


FEATURE_NAMES = (
    "population_millions",
    "density_km2",
    "avg_temp_c",
    "recycling_rate",
    "gdp_per_capita_k",
    "poi_density_index",
    "street_density_index",
    "city_baseline_kg",
    "current_load_kg",
    "days_since_collection",
    "local_activity_score",
    "center_distance_km",
)


CHISINAU_PROFILE = {
    "city": "Chisinau",
    "country": "Moldova",
    "population_millions": 0.63,
    "density_km2": 4300.0,
    "avg_temp_c": 10.0,
    "recycling_rate": 0.12,
    "gdp_per_capita_k": 7.4,
}


WORLD_CITY_PROFILES = [
    {"city": "New York", "region": "North America", "population_millions": 8.3, "density_km2": 11000.0, "avg_temp_c": 12.5, "recycling_rate": 0.30, "gdp_per_capita_k": 80.0, "poi_density_index": 1.15, "street_density_index": 1.05},
    {"city": "Los Angeles", "region": "North America", "population_millions": 4.0, "density_km2": 3200.0, "avg_temp_c": 18.5, "recycling_rate": 0.25, "gdp_per_capita_k": 72.0, "poi_density_index": 0.90, "street_density_index": 0.82},
    {"city": "Chicago", "region": "North America", "population_millions": 2.7, "density_km2": 4600.0, "avg_temp_c": 10.0, "recycling_rate": 0.34, "gdp_per_capita_k": 62.0, "poi_density_index": 0.83, "street_density_index": 0.79},
    {"city": "Mexico City", "region": "North America", "population_millions": 9.2, "density_km2": 6100.0, "avg_temp_c": 17.0, "recycling_rate": 0.12, "gdp_per_capita_k": 20.0, "poi_density_index": 0.94, "street_density_index": 0.78},
    {"city": "Sao Paulo", "region": "South America", "population_millions": 12.4, "density_km2": 7400.0, "avg_temp_c": 20.0, "recycling_rate": 0.18, "gdp_per_capita_k": 17.0, "poi_density_index": 0.98, "street_density_index": 0.84},
    {"city": "Buenos Aires", "region": "South America", "population_millions": 3.1, "density_km2": 14500.0, "avg_temp_c": 17.5, "recycling_rate": 0.20, "gdp_per_capita_k": 21.0, "poi_density_index": 1.06, "street_density_index": 0.92},
    {"city": "Rio de Janeiro", "region": "South America", "population_millions": 6.8, "density_km2": 5300.0, "avg_temp_c": 23.5, "recycling_rate": 0.14, "gdp_per_capita_k": 19.0, "poi_density_index": 0.91, "street_density_index": 0.74},
    {"city": "London", "region": "Europe", "population_millions": 9.0, "density_km2": 5700.0, "avg_temp_c": 11.5, "recycling_rate": 0.44, "gdp_per_capita_k": 89.0, "poi_density_index": 1.22, "street_density_index": 1.10},
    {"city": "Paris", "region": "Europe", "population_millions": 2.1, "density_km2": 20500.0, "avg_temp_c": 12.0, "recycling_rate": 0.47, "gdp_per_capita_k": 86.0, "poi_density_index": 1.35, "street_density_index": 1.18},
    {"city": "Berlin", "region": "Europe", "population_millions": 3.8, "density_km2": 4000.0, "avg_temp_c": 10.5, "recycling_rate": 0.47, "gdp_per_capita_k": 65.0, "poi_density_index": 0.96, "street_density_index": 0.88},
    {"city": "Madrid", "region": "Europe", "population_millions": 3.3, "density_km2": 5400.0, "avg_temp_c": 15.0, "recycling_rate": 0.42, "gdp_per_capita_k": 60.0, "poi_density_index": 0.92, "street_density_index": 0.83},
    {"city": "Moscow", "region": "Europe", "population_millions": 12.5, "density_km2": 4900.0, "avg_temp_c": 5.5, "recycling_rate": 0.28, "gdp_per_capita_k": 55.0, "poi_density_index": 0.88, "street_density_index": 0.80},
    {"city": "Istanbul", "region": "Europe", "population_millions": 15.5, "density_km2": 2900.0, "avg_temp_c": 14.5, "recycling_rate": 0.10, "gdp_per_capita_k": 30.0, "poi_density_index": 0.84, "street_density_index": 0.77},
    {"city": "Cairo", "region": "Africa", "population_millions": 10.2, "density_km2": 19000.0, "avg_temp_c": 22.0, "recycling_rate": 0.05, "gdp_per_capita_k": 18.0, "poi_density_index": 0.78, "street_density_index": 0.69},
    {"city": "Lagos", "region": "Africa", "population_millions": 15.9, "density_km2": 6500.0, "avg_temp_c": 27.5, "recycling_rate": 0.03, "gdp_per_capita_k": 12.0, "poi_density_index": 0.72, "street_density_index": 0.63},
    {"city": "Nairobi", "region": "Africa", "population_millions": 4.7, "density_km2": 6500.0, "avg_temp_c": 19.0, "recycling_rate": 0.07, "gdp_per_capita_k": 14.0, "poi_density_index": 0.74, "street_density_index": 0.66},
    {"city": "Johannesburg", "region": "Africa", "population_millions": 5.9, "density_km2": 3000.0, "avg_temp_c": 16.5, "recycling_rate": 0.12, "gdp_per_capita_k": 15.0, "poi_density_index": 0.76, "street_density_index": 0.68},
    {"city": "Mumbai", "region": "Asia", "population_millions": 20.7, "density_km2": 29000.0, "avg_temp_c": 27.0, "recycling_rate": 0.04, "gdp_per_capita_k": 9.0, "poi_density_index": 1.02, "street_density_index": 0.86},
    {"city": "Delhi", "region": "Asia", "population_millions": 32.9, "density_km2": 11300.0, "avg_temp_c": 25.0, "recycling_rate": 0.08, "gdp_per_capita_k": 10.0, "poi_density_index": 0.95, "street_density_index": 0.79},
    {"city": "Bangkok", "region": "Asia", "population_millions": 10.7, "density_km2": 5300.0, "avg_temp_c": 29.0, "recycling_rate": 0.18, "gdp_per_capita_k": 19.0, "poi_density_index": 0.99, "street_density_index": 0.81},
    {"city": "Singapore", "region": "Asia", "population_millions": 5.9, "density_km2": 7900.0, "avg_temp_c": 27.0, "recycling_rate": 0.52, "gdp_per_capita_k": 95.0, "poi_density_index": 1.42, "street_density_index": 1.20},
    {"city": "Tokyo", "region": "Asia", "population_millions": 14.0, "density_km2": 6400.0, "avg_temp_c": 16.0, "recycling_rate": 0.56, "gdp_per_capita_k": 92.0, "poi_density_index": 1.38, "street_density_index": 1.14},
    {"city": "Seoul", "region": "Asia", "population_millions": 9.7, "density_km2": 16000.0, "avg_temp_c": 13.0, "recycling_rate": 0.60, "gdp_per_capita_k": 88.0, "poi_density_index": 1.30, "street_density_index": 1.11},
    {"city": "Sydney", "region": "Oceania", "population_millions": 5.3, "density_km2": 4000.0, "avg_temp_c": 18.0, "recycling_rate": 0.41, "gdp_per_capita_k": 75.0, "poi_density_index": 0.92, "street_density_index": 0.85},
    {"city": "Auckland", "region": "Oceania", "population_millions": 1.7, "density_km2": 3200.0, "avg_temp_c": 15.0, "recycling_rate": 0.38, "gdp_per_capita_k": 68.0, "poi_density_index": 0.84, "street_density_index": 0.73},
    {"city": "Chisinau", "region": "Europe", "population_millions": 0.63, "density_km2": 4300.0, "avg_temp_c": 10.0, "recycling_rate": 0.12, "gdp_per_capita_k": 7.4, "poi_density_index": 0.73, "street_density_index": 0.62},
]


def _base_city_baseline(profile: dict[str, float | str]) -> float:
    return float(
        0.9
        + 0.32 * float(profile["population_millions"])
        + 0.00010 * float(profile["density_km2"])
        + 0.045 * float(profile["avg_temp_c"])
        + 0.060 * float(profile["gdp_per_capita_k"])
        + 0.45 * float(profile["poi_density_index"])
        + 0.32 * float(profile["street_density_index"])
        - 1.10 * float(profile["recycling_rate"])
    )


def _build_training_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    samples = (
        (1.18, 0.7, 1.1, 0.4),
        (0.98, 3.0, 0.8, 1.7),
        (0.74, 5.0, 0.55, 3.2),
    )
    for index, profile in enumerate(WORLD_CITY_PROFILES):
        city_baseline = _base_city_baseline(profile)
        for sample_index, (load_factor, days_since, activity_factor, distance_km) in enumerate(samples):
            current_load = float(city_baseline * load_factor * 0.42)
            local_activity = float(activity_factor + 0.04 * (index % 3) - 0.03 * sample_index)
            target = float(
                city_baseline
                + 0.18 * current_load
                + 0.22 * days_since
                + 0.68 * local_activity
                - 0.14 * distance_km
                + 0.03 * float(profile["poi_density_index"])
            )
            rows.append(
                {
                    "population_millions": float(profile["population_millions"]),
                    "density_km2": float(profile["density_km2"]),
                    "avg_temp_c": float(profile["avg_temp_c"]),
                    "recycling_rate": float(profile["recycling_rate"]),
                    "gdp_per_capita_k": float(profile["gdp_per_capita_k"]),
                    "poi_density_index": float(profile["poi_density_index"]),
                    "street_density_index": float(profile["street_density_index"]),
                    "city_baseline_kg": city_baseline,
                    "current_load_kg": current_load,
                    "days_since_collection": float(days_since),
                    "local_activity_score": local_activity,
                    "center_distance_km": float(distance_km),
                    "expected_collected_kg": target,
                }
            )
    return rows


def _fit_linear_regression(rows: list[dict[str, float]]) -> LinearRegressionModel:
    x_rows = []
    y_rows = []
    for row in rows:
        x_rows.append([float(row[name]) for name in FEATURE_NAMES])
        y_rows.append(float(row["expected_collected_kg"]))

    X = np.asarray(x_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    lr = LinearRegression()
    lr.fit(X, y)
    intercept = float(lr.intercept_)
    weights = np.asarray(lr.coef_, dtype=float)
    return LinearRegressionModel(feature_names=FEATURE_NAMES, intercept=intercept, coefficients=weights)


@lru_cache(maxsize=1)
def get_global_model() -> LinearRegressionModel:
    return _fit_linear_regression(_build_training_rows())


def predict_expected_kg(feature_values: dict[str, float]) -> float:
    model = get_global_model()
    prediction = model.predict([feature_values])[0]
    return float(max(0.0, prediction))


def training_row_count() -> int:
    return len(_build_training_rows())


def training_city_count() -> int:
    return len(WORLD_CITY_PROFILES)
