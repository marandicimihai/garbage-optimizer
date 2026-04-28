from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import requests
import simpy

from get_poi import build_grid, build_leaflet_map, clean_pois, fetch_pois


WASTE_TYPES = ["plastic", "paper", "glass", "metal"]


@dataclass(slots=True)
class WasteItem:
	x: int
	y: int
	type: str
	time: float


def time_factor(t: float, period: float = 10.0) -> float:
	"""Smooth daily-like wave in [0.3, 1.3] to avoid zero/negative rates."""
	return 0.8 + 0.5 * math.sin(t / period)


def waste_generator(
	env: simpy.Environment,
	x: int,
	y: int,
	intensity: float,
	waste_items: list[WasteItem],
	base_interval: float,
	use_time_variation: bool,
):
	while True:
		factor = time_factor(env.now) if use_time_variation else 1.0
		interval = max(0.25, base_interval / ((intensity + 1.0) * factor))
		yield env.timeout(interval)
		waste_items.append(WasteItem(x=x, y=y, type=random.choice(WASTE_TYPES), time=env.now))


def run_simulation(
	grid: np.ndarray,
	run_until: float = 50.0,
	base_interval: float = 8.0,
	use_time_variation: bool = True,
	seed: int | None = 42,
):
	if seed is not None:
		random.seed(seed)

	env = simpy.Environment()
	waste_items: list[WasteItem] = []

	for x in range(grid.shape[0]):
		for y in range(grid.shape[1]):
			intensity = float(grid[x, y])
			if intensity <= 0:
				continue
			env.process(
				waste_generator(
					env,
					x=x,
					y=y,
					intensity=intensity,
					waste_items=waste_items,
					base_interval=base_interval,
					use_time_variation=use_time_variation,
				)
			)

	env.run(until=run_until)
	return waste_items


def waste_items_to_map(waste_items: list[WasteItem], shape: tuple[int, int]) -> np.ndarray:
	waste_map = np.zeros(shape, dtype=float)
	for item in waste_items:
		waste_map[item.x, item.y] += 1
	return waste_map


def save_heatmap(waste_map: np.ndarray, output_path: str, show_plot: bool = False):
	# ensure parent directory exists
	output_path = str(output_path)
	from pathlib import Path

	Path(output_path).parent.mkdir(parents=True, exist_ok=True)

	plt.figure(figsize=(7, 6))
	plt.imshow(waste_map.T, origin="lower", cmap="hot")
	plt.title("Waste Generated (SimPy)")
	plt.colorbar(label="Generated items")
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	if show_plot:
		plt.show()
	plt.close()


def main():
	parser = argparse.ArgumentParser(description="Simulate waste generation over a POI-derived grid using SimPy.")
	parser.add_argument("place", nargs="?", default="Chisinau, Moldova")
	parser.add_argument("--grid-size", type=int, default=50)
	parser.add_argument("--distance-meters", type=int, default=1000)
	parser.add_argument("--requests-timeout", type=int, default=60)
	parser.add_argument("--run-until", type=float, default=50.0)
	parser.add_argument("--base-interval", type=float, default=8.0)
	parser.add_argument("--no-time-variation", action="store_true")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output", default="generated/waste_map.png")
	parser.add_argument("--map-output", default="generated/waste_map.html")
	parser.add_argument("--overlay-output", default="generated/waste_overlay.png")
	parser.add_argument("--overlay-opacity", type=float, default=0.45)
	parser.add_argument("--show", action="store_true")
	args = parser.parse_args()

	import osmnx as ox

	ox.settings.requests_timeout = args.requests_timeout

	print(f"Fetching POIs for {args.place}...")
	try:
		raw = fetch_pois(args.place, distance_meters=args.distance_meters)
	except requests.RequestException as exc:
		raise SystemExit(
			"Failed to fetch POIs from OpenStreetMap. "
			"Try again later, increase --requests-timeout, or reduce --distance-meters."
		) from exc

	pois = clean_pois(raw)
	print(f"Filtered POIs: {len(pois)}")

	grid, bounds = build_grid(pois, grid_size=args.grid_size)
	print(f"Grid shape: {grid.shape}, active cells: {int(np.count_nonzero(grid))}")

	waste_items = run_simulation(
		grid,
		run_until=args.run_until,
		base_interval=args.base_interval,
		use_time_variation=not args.no_time_variation,
		seed=args.seed,
	)

	waste_map = waste_items_to_map(waste_items, grid.shape)
	print(f"Generated waste items: {len(waste_items)}")
	print(f"Max cell load: {int(waste_map.max())}")

	save_heatmap(waste_map, output_path=args.output, show_plot=args.show)
	print(f"Saved heatmap to {args.output}")

	map_output_path, overlay_output_path = build_leaflet_map(
		args.place,
		waste_map,
		bounds,
		pois,
		overlay_opacity=args.overlay_opacity,
		map_output_path=args.map_output,
		overlay_output_path=args.overlay_output,
	)
	print(f"Saved interactive map to {map_output_path}")
	print(f"Saved map overlay to {overlay_output_path}")


if __name__ == "__main__":
	main()
