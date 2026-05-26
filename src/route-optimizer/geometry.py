from __future__ import annotations

import math


DEFAULT_COORD_PRECISION = 6
DEFAULT_MIN_POINT_SPACING_METERS = 1.0
DEFAULT_COLLINEAR_TOLERANCE_METERS = 1.5


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
	lat1, lon1 = a
	lat2, lon2 = b
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlambda = math.radians(lon2 - lon1)
	h = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
	return 2.0 * 6_371_000.0 * math.asin(math.sqrt(h))


def round_point(point: tuple[float, float], precision: int = DEFAULT_COORD_PRECISION) -> tuple[float, float]:
	return round(point[0], precision), round(point[1], precision)


def normalize_street_line(
	line: list[tuple[float, float]],
	*,
	precision: int = DEFAULT_COORD_PRECISION,
	min_point_spacing_m: float = DEFAULT_MIN_POINT_SPACING_METERS,
	collinear_tolerance_m: float = DEFAULT_COLLINEAR_TOLERANCE_METERS,
) -> list[tuple[float, float]]:
	if len(line) < 2:
		return []

	rounded = [round_point(point, precision) for point in line]
	cleaned: list[tuple[float, float]] = [rounded[0]]

	for point in rounded[1:]:
		if haversine_m(cleaned[-1], point) <= min_point_spacing_m:
			continue
		cleaned.append(point)

	if len(cleaned) < 2:
		return []

	simplified: list[tuple[float, float]] = [cleaned[0]]
	for point in cleaned[1:]:
		simplified.append(point)
		while len(simplified) >= 3:
			start = simplified[-3]
			middle = simplified[-2]
			end = simplified[-1]
			if (
				haversine_m(start, middle) + haversine_m(middle, end) - haversine_m(start, end)
				> collinear_tolerance_m
			):
				break
			simplified.pop(-2)

	return simplified if len(simplified) >= 2 else []


def normalize_street_lines(
	lines: list[list[tuple[float, float]]],
	*,
	precision: int = DEFAULT_COORD_PRECISION,
	min_point_spacing_m: float = DEFAULT_MIN_POINT_SPACING_METERS,
	collinear_tolerance_m: float = DEFAULT_COLLINEAR_TOLERANCE_METERS,
) -> list[list[tuple[float, float]]]:
	normalized: list[list[tuple[float, float]]] = []
	for line in lines:
		normalized_line = normalize_street_line(
			line,
			precision=precision,
			min_point_spacing_m=min_point_spacing_m,
			collinear_tolerance_m=collinear_tolerance_m,
		)
		if len(normalized_line) >= 2:
			normalized.append(normalized_line)
	return normalized