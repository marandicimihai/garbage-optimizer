from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from streets import distance_bounds, fetch_street_lines, DEFAULT_GRID_SIZE
from typing import Tuple, List


@dataclass
class Bin:
    bin_id: int
    x: int
    y: int
    current_load: float = 0.0
    capacity: float = 100.0

    def get_load_ratio(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return min(1.0, self.current_load / self.capacity)


def haversine_meters(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    aa = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(aa), math.sqrt(1 - aa))


def sample_segment_points(seg: list[tuple[float, float]], step_m: float) -> Iterable[tuple[float, float]]:
    # seg is sequence of (lat, lon) points
    for i in range(len(seg) - 1):
        a = seg[i]
        b = seg[i + 1]
        length = haversine_meters(a, b)
        if length == 0:
            continue
        steps = max(1, int(math.floor(length / step_m)))
        for s in range(steps + 1):
            t = s / steps
            lat = a[0] + t * (b[0] - a[0])
            lon = a[1] + t * (b[1] - a[1])
            yield (lat, lon)


def sample_street_points(street_lines: list[list[tuple[float, float]]], step_m: float) -> list[tuple[float, float]]:
    pts = []
    for seg in street_lines:
        for p in sample_segment_points(seg, step_m):
            pts.append(p)
    # deduplicate
    seen = set()
    uniq = []
    for lat, lon in pts:
        key = (round(lat, 6), round(lon, 6))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((lat, lon))
    return uniq


def latlon_to_grid(lat: float, lon: float, bounds: tuple[float, float, float, float], grid_size: int) -> tuple[int, int] | None:
    minx, miny, maxx, maxy = bounds
    cell_w = (maxx - minx) / grid_size
    cell_h = (maxy - miny) / grid_size
    if not (minx <= lon <= maxx and miny <= lat <= maxy):
        return None
    x = int((lon - minx) / cell_w)
    y = int((lat - miny) / cell_h)
    x = max(0, min(grid_size - 1, x))
    y = max(0, min(grid_size - 1, y))
    return x, y


def build_streets_grid(points: list[tuple[float, float]], bounds: tuple[float, float, float, float], grid_size: int) -> np.ndarray:
    grid = np.zeros((grid_size, grid_size), dtype=int)
    for lat, lon in points:
        idx = latlon_to_grid(lat, lon, bounds, grid_size)
        if idx is None:
            continue
        x, y = idx
        grid[x, y] = 1
    return grid


def place_bins_on_streets(place: str, distance_meters: int, grid_size: int, bin_spacing_meters: float) -> Tuple[List[Bin], np.ndarray, tuple]:
    bounds = distance_bounds(place, distance_meters)
    street_lines = fetch_street_lines(place, distance_meters)
    if not street_lines:
        return [], np.zeros((grid_size, grid_size), dtype=int), bounds

    sampled = sample_street_points(street_lines, bin_spacing_meters)
    streets_grid = build_streets_grid(sampled, bounds, grid_size)

    seen_cells = set()
    bins: List[Bin] = []
    next_id = 1
    for lat, lon in sampled:
        idx = latlon_to_grid(lat, lon, bounds, grid_size)
        if idx is None:
            continue
        if idx in seen_cells:
            continue
        seen_cells.add(idx)
        x, y = idx
        b = Bin(bin_id=next_id, x=x, y=y, current_load=0.0, capacity=100.0)
        bins.append(b)
        next_id += 1

    return bins, streets_grid, bounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Place bins on streets at a fixed spacing and visualize them")
    parser.add_argument("place", nargs="?", default=None)
    parser.add_argument("--place", dest="place_opt", default=None)
    parser.add_argument("--distance-meters", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--bin-spacing-meters", type=float, default=40.0)
    parser.add_argument("--output", type=str, default="generated/system_bins_on_streets.html")
    args = parser.parse_args()
    place = args.place_opt or args.place
    if place is None:
        place = "Chisinau, Moldova"

    bounds = distance_bounds(place, args.distance_meters)
    bins, streets_grid, bounds = place_bins_on_streets(place, args.distance_meters, args.grid_size, args.bin_spacing_meters)
    print(f"Placing {len(bins)} bins on the street network (grid size {args.grid_size})")

    # write outputs: grid and simple bins.csv
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    grid_out = str(Path(args.output).with_suffix("").with_name(Path(args.output).stem + "_grid.npy"))
    np.save(grid_out, streets_grid)
    bins_out = str(Path(args.output).with_suffix("").with_name(Path(args.output).stem + "_bins.csv"))
    with open(bins_out, "w", encoding="utf-8") as fh:
        fh.write("bin_id,x,y,current_load,capacity\n")
        for b in bins:
            fh.write(f"{b.bin_id},{b.x},{b.y},{b.current_load},{b.capacity}\n")
    print(f"Saved streets grid to {grid_out} and bins to {bins_out}")


if __name__ == "__main__":
    main()
