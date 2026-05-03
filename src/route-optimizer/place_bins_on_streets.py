from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, List

import numpy as np

from streets import distance_bounds, fetch_street_lines


DEFAULT_GRID_SIZE = 100

DEFAULT_PLACE = "Chisinau, Moldova"
DEFAULT_DISTANCE_METERS = 1000


@dataclass
class Bin:
    bin_id: int
    x: int
    y: int
    lat: float
    lon: float
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
        if idx is not None:
            x, y = idx
            grid[y, x] = 1
    return grid


def place_bins_on_streets(
    place: str,
    distance_meters: int,
    grid_size: int,
    bin_spacing_meters: float,
) -> Tuple[List[Bin], np.ndarray, tuple[float, float, float, float]]:
    bounds = distance_bounds(place, distance_meters)
    street_lines = fetch_street_lines(place, distance_meters)

    if not street_lines:
        return [], np.zeros((grid_size, grid_size), dtype=int), bounds

    sample_step = bin_spacing_meters / 4
    all_points = sample_street_points(street_lines, sample_step)
    
    selected_bins = []
    min_dist_sq = bin_spacing_meters ** 2
    
    for lat, lon in all_points:
        idx = latlon_to_grid(lat, lon, bounds, grid_size)
        if idx is None:
            continue
        
        too_close = False
        for existing in selected_bins:
            dlat = lat - existing.lat
            dlon = lon - existing.lon
            dist_sq = (dlat * 111000) ** 2 + (dlon * 111000 * np.cos(np.radians(lat))) ** 2
            if dist_sq < min_dist_sq:
                too_close = True
                break
        
        if not too_close:
            x, y = idx
            b = Bin(bin_id=len(selected_bins), x=x, y=y, lat=lat, lon=lon, current_load=0.0, capacity=100.0)
            selected_bins.append(b)

    points = [(b.lat, b.lon) for b in selected_bins]
    streets_grid = build_streets_grid(points, bounds, grid_size)

    return selected_bins, streets_grid, bounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Place bins on streets at a fixed spacing and visualize them")
    parser.add_argument("place", nargs="?", default=None)
    parser.add_argument("--place", dest="place_opt", default=None)
    parser.add_argument("--distance-meters", type=int, default=DEFAULT_DISTANCE_METERS)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--bin-spacing-meters", type=float, default=40.0)
    parser.add_argument("--output", type=str, default="generated/system_bins_on_streets")
    args = parser.parse_args()

    place = args.place_opt or args.place or DEFAULT_PLACE
    bounds = distance_bounds(place, args.distance_meters)
    bins, streets_grid, bounds = place_bins_on_streets(place, args.distance_meters, args.grid_size, args.bin_spacing_meters)
    print(f"Placing {len(bins)} bins on the street network (grid size {args.grid_size})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    grid_out = str(Path(args.output).with_suffix("").with_name(Path(args.output).stem + "_grid.npy"))
    np.save(grid_out, streets_grid)
    bins_out = str(Path(args.output).with_suffix("").with_name(Path(args.output).stem + "_bins.csv"))
    with open(bins_out, "w", encoding="utf-8") as fh:
        fh.write("bin_id,x,y,lat,lon,current_load,capacity\n")
        for b in bins:
            fh.write(f"{b.bin_id},{b.x},{b.y},{b.lat},{b.lon},{b.current_load},{b.capacity}\n")
    print(f"Saved streets grid to {grid_out} and bins to {bins_out}")


if __name__ == "__main__":
    main()