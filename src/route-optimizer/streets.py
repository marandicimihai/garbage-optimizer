from __future__ import annotations
from pathlib import Path

import csv

import numpy as np
import osmnx as ox


DEFAULT_PLACE = "Chisinau, Moldova"
DEFAULT_DISTANCE_METERS = 1000
DEFAULT_OUTPUT = "street_lines.csv"
DEFAULT_GRID_SIZE = 100


def distance_bounds(place: str, distance_meters: int) -> tuple[float, float, float, float]:
    center_lat, center_lon = ox.geocode(place)
    lat_degrees_per_meter = 1 / 111_000.0
    lon_degrees_per_meter = 1 / (111_000.0 * np.cos(np.radians(center_lat)))
    lat_span = distance_meters * 2 * lat_degrees_per_meter
    lon_span = distance_meters * 2 * lon_degrees_per_meter

    miny = center_lat - lat_span / 2
    maxy = center_lat + lat_span / 2
    minx = center_lon - lon_span / 2
    maxx = center_lon + lon_span / 2
    return minx, miny, maxx, maxy


def fetch_street_lines(place: str, distance_meters: int) -> list[list[tuple[float, float]]]:
    """Fetch street geometries for the given distance-based area."""
    try:
        center = ox.geocode(place)
        graph = ox.graph_from_point(center, dist=distance_meters, network_type="drive")
    except Exception as point_exc:
        print(f"OSM point fetch failed: {point_exc}")
        try:
            graph = ox.graph_from_place(place, network_type="drive")
        except Exception as place_exc:
            print(f"OSM place fallback failed: {place_exc}")
            return []

    lines = []
    for u, v, data in graph.edges(data=True):
        if "geometry" in data:
            geom = data["geometry"]
            if hasattr(geom, "coords"):
                coords = list(geom.coords)
                line = [(lat, lon) for lon, lat in coords]
                if len(line) >= 2:
                    lines.append(line)
            elif hasattr(geom, "geoms"):
                for segment in geom.geoms:
                    coords = list(segment.coords)
                    line = [(lat, lon) for lon, lat in coords]
                    if len(line) >= 2:
                        lines.append(line)
    return lines


def generated_dir() -> Path:
    return Path(__file__).resolve().parent / "generated"


def output_path() -> Path:
    return generated_dir() / DEFAULT_OUTPUT


def save_street_lines_csv(
    lines: list[list[tuple[float, float]]],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["line_id", "point_order", "lat", "lon"])

        for line_id, line in enumerate(lines):
            for point_order, (lat, lon) in enumerate(line):
                writer.writerow([line_id, point_order, lat, lon])


def main() -> None:
    place = DEFAULT_PLACE
    distance_meters = DEFAULT_DISTANCE_METERS
    output = output_path()

    lines = fetch_street_lines(place, distance_meters)

    if not lines:
        print("No street lines fetched.")
        return

    save_street_lines_csv(lines, output)
    print(f"Saved {len(lines)} street lines to {output}")


if __name__ == "__main__":
    main()