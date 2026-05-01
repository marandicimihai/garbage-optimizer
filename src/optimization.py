from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
import simpy

import osmnx as ox
import requests
from get_poi import fetch_pois, clean_pois, build_grid
from visualization import save_png_plot, create_system_map
from shapely.geometry import LineString


NUM_BINS = 50
BIN_CAPACITY_RANGE = (30.0, 750.0)
WASTE_PER_PERSON = 0.5
OVERFLOW_THRESHOLD = 0.95
SIM_TIME = 500

USERS_PER_BIN = 10
TRUCK_CAPACITY = 1000

GRID_SIZE = 50
STREET_WIDTH = 2

WASTE_TYPES = ["plastic", "paper", "glass", "metal", "organic"]
WASTE_WEIGHT_RANGE = (0.1, 1.0)

DEFAULT_PLACE = "Chisinau, Moldova"
POI_DISTANCE_METERS = 1000
POI_REQUESTS_TIMEOUT = 60


def fetch_poi_grid(
	place: str = DEFAULT_PLACE,
	grid_size: int = GRID_SIZE,
	distance_meters: int = POI_DISTANCE_METERS,
	requests_timeout: int = POI_REQUESTS_TIMEOUT,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
	"""Fetch real POI data from OpenStreetMap and build intensity grid."""
	
	try:
		print(f"Fetching POIs for {place}...")
		ox.settings.requests_timeout = requests_timeout
		raw = fetch_pois(place, distance_meters=distance_meters)
		pois = clean_pois(raw)
		
		if pois.empty:
			print("No POIs found. Using synthetic grid.")
			return None
		
		print(f"Fetched {len(pois)} POIs")
		grid, bounds = build_grid(pois, grid_size=grid_size)
		
		grid_max = grid.max()
		if grid_max > 0:
			normalized = grid / grid_max
		else:
			normalized = grid
		
		print(f"Built intensity grid {normalized.shape} from {len(pois)} POIs")
		return normalized.T, bounds
		
	except requests.RequestException as e:
		print(f"Failed to fetch POIs: {e}")
		print("Using synthetic grid fallback.")
		return None
	except Exception as e:
		print(f"Error processing POIs: {e}")
		print("Using synthetic grid fallback.")
		return None


def get_intensity_grid(
	poi_place: str | None = None,
	grid_size: int = GRID_SIZE,
	poi_distance: int = POI_DISTANCE_METERS,
) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
	"""Get intensity grid from POI data or fall back to synthetic."""
	result = fetch_poi_grid(
		place=poi_place or DEFAULT_PLACE,
		grid_size=grid_size,
		distance_meters=poi_distance,
	)
	if result is not None:
		grid, bounds = result
		return grid, bounds
	
	print(f"Generating synthetic intensity grid ({grid_size}x{grid_size})")
	grid = np.random.uniform(0.3, 0.8, (grid_size, grid_size))
	
	for _ in range(3):
		hx, hy = np.random.randint(5, grid_size - 5, 2)
		size = np.random.randint(3, 8)
		grid[hx:hx+size, hy:hy+size] = np.minimum(1.0, grid[hx:hx+size, hy:hy+size] + 0.4)
	
	return grid, None


@dataclass(slots=True)
class WasteItem:
	"""Waste item with location, type, weight, and timestamp."""
	x: int
	y: int
	type: str
	weight: float
	time: float


@dataclass
class Bin:
	"""Active bin entity in SimPy environment."""
	env: simpy.Environment
	bin_id: int
	x: int
	y: int
	capacity: float
	
	current_load: float = 0.0
	overflow: bool = False
	items: list[WasteItem] = field(default_factory=list)
	overflow_events: int = 0
	total_waste_received: float = 0.0
	
	def __post_init__(self):
		"""Start the bin process."""
		self.process = self.env.process(self.run())
	
	def add_waste(self, item: WasteItem) -> None:
		"""Receive waste item and update load."""
		self.items.append(item)
		self.current_load += item.weight
		self.total_waste_received += item.weight
		
		if self.current_load >= self.capacity * OVERFLOW_THRESHOLD:
			if not self.overflow:
				self.overflow = True
				self.overflow_events += 1
	
	def run(self) -> Generator[Any, Any, Any]:
		"""Bin process loop."""
		while True:
			yield self.env.timeout(1)
	
	def get_load_ratio(self) -> float:
		"""Return current load as percentage of capacity."""
		return self.current_load / self.capacity if self.capacity > 0 else 0.0
	
	def __repr__(self) -> str:
		return f"Bin({self.bin_id}, pos=({self.x},{self.y}), load={self.current_load:.1f}/{self.capacity:.1f}kg, overflow={self.overflow})"


def generate_streets(grid_size: int, street_width: int = 2) -> np.ndarray:
	"""Generate a grid marking street cells."""
	streets = np.zeros((grid_size, grid_size), dtype=bool)
	for i in range(0, grid_size, street_width * 3):
		streets[i:i+street_width, :] = True
	for j in range(0, grid_size, street_width * 3):
		streets[:, j:j+street_width] = True
	return streets


def coords_to_cell(lon: float, lat: float, bounds: tuple[float, float, float, float], grid_size: int) -> tuple[int, int]:
	minx, miny, maxx, maxy = bounds
	if maxx == minx or maxy == miny:
		return 0, 0
	ix = int((lon - minx) / (maxx - minx) * grid_size)
	iy = int((lat - miny) / (maxy - miny) * grid_size)
	ix = max(0, min(grid_size - 1, ix))
	iy = max(0, min(grid_size - 1, iy))
	return ix, iy


def cell_center_coords(ix: int, iy: int, bounds: tuple[float, float, float, float], grid_size: int) -> tuple[float, float]:
	minx, miny, maxx, maxy = bounds
	cell_w = (maxx - minx) / grid_size
	cell_h = (maxy - miny) / grid_size
	lon = minx + (ix + 0.5) * cell_w
	lat = miny + (iy + 0.5) * cell_h
	return lon, lat


def haversine_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
	"""Return distance in meters between two lon/lat points."""
	R = 6371000.0
	lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
	c = 2 * math.asin(math.sqrt(a))
	return R * c


def build_street_mask_from_osm(bounds: tuple[float, float, float, float], grid_size: int) -> np.ndarray:
	"""Rasterize OSM street geometries into a boolean grid mask of street cells."""
	minx, miny, maxx, maxy = bounds
	# north, south, east, west
	north = maxy
	south = miny
	east = maxx
	west = minx

	# fetch walking network inside bbox
	G = None
	# try several call styles to handle osmnx API differences
	try:
		G = ox.graph_from_bbox(north, south, east, west, network_type="walk")
	except TypeError as te1:
		print(f"OSM bbox positional call failed: {te1}")
		try:
			# try passing bbox as a single tuple (some versions expect this)
			G = ox.graph_from_bbox((north, south, east, west), network_type="walk")
		except TypeError as te2:
			print(f"OSM bbox tuple call failed: {te2}")
			try:
				# try keyword args (some versions have different signature)
				G = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type="walk")
			except Exception as e_kw:
				print(f"OSM bbox keyword call failed: {e_kw}")
				G = None
	except Exception as e:
		print(f"OSM bbox fetch error: {e}")

	if G is None:
		# try fallback by place name
		try:
			G = ox.graph_from_place(DEFAULT_PLACE, network_type="walk")
			print("OSM fallback graph_from_place succeeded")
		except Exception as e2:
			print(f"OSM fallback fetch failed: {e2}")
			return np.zeros((grid_size, grid_size), dtype=bool)

	streets = np.zeros((grid_size, grid_size), dtype=bool)

	# iterate edges and mark cells touched by edge geometries
	for u, v, data in G.edges(data=True):
		geom = data.get("geometry")
		if geom is None:
			# use node coordinates
			try:
				xu, yu = G.nodes[u]["x"], G.nodes[u]["y"]
				xv, yv = G.nodes[v]["x"], G.nodes[v]["y"]
				pts = [(xu, yu), (xv, yv)]
			except Exception:
				continue
		else:
			pts = list(geom.coords)

		# sample along the edge points
		for lon, lat in pts:
			ix, iy = coords_to_cell(lon, lat, bounds, grid_size)
			streets[ix, iy] = True

	return streets


def get_street_cells(streets: np.ndarray) -> list[tuple[int, int]]:
	"""Get all coordinates of street cells."""
	return list(zip(*np.where(streets)))


def place_bins_on_streets(
	env: simpy.Environment,
	num_bins: int,
	streets: np.ndarray,
	locations: list[tuple[int, int]] | None = None,
) -> list[Bin]:
	"""Place bins on street cells.

	If `locations` is provided, place bins at those coordinates (useful for optimized placement).
	Otherwise pick random street cells as before.
	"""
	street_cells = get_street_cells(streets)

	if locations is None:
		if len(street_cells) < num_bins:
			raise ValueError(
				f"Not enough street cells ({len(street_cells)}) for {num_bins} bins. "
				"Try reducing NUM_BINS or increasing street width."
			)
		selected_cells = random.sample(street_cells, num_bins)
	else:
		# validate provided locations
		for (x, y) in locations:
			if (x, y) not in street_cells:
				raise ValueError(f"Location {(x,y)} is not a street cell")
		if len(locations) != num_bins:
			raise ValueError("Length of locations must equal num_bins")
		selected_cells = locations

	bins = []
	for bin_id, (x, y) in enumerate(selected_cells):
		capacity = random.uniform(*BIN_CAPACITY_RANGE)
		bin_obj = Bin(env=env, bin_id=bin_id, x=x, y=y, capacity=capacity)
		bins.append(bin_obj)

	return bins


def compute_p_median_locations(
	grid: np.ndarray, streets: np.ndarray, p: int, bounds: tuple[float, float, float, float]
) -> list[tuple[int, int]]:
	"""Greedy p-median heuristic using geographic distances (meters).
	Demand points are grid cell centers weighted by grid values; candidates are street cells.
	"""
	grid_size = grid.shape[0]
	# demand points: list of (lon, lat), weight
	demand = []
	for x in range(grid_size):
		for y in range(grid_size):
			w = float(grid[x, y])
			if w <= 0:
				continue
			lon, lat = cell_center_coords(x, y, bounds, grid_size)
			demand.append(((lon, lat), w))

	candidates_cells = get_street_cells(streets)
	if len(candidates_cells) < p:
		raise ValueError("Not enough candidate locations for p-median")

	candidates = [cell_center_coords(ix, iy, bounds, grid_size) for (ix, iy) in candidates_cells]

	# precompute distances (meters) between demand points and candidates
	dist_cache = {}
	for d_idx, (dcoord, _) in enumerate(demand):
		for c_idx, ccoord in enumerate(candidates):
			dist_cache[(d_idx, c_idx)] = haversine_meters(dcoord[0], dcoord[1], ccoord[0], ccoord[1])

	best_dist = {d_idx: float('inf') for d_idx in range(len(demand))}
	selected_indices: list[int] = []

	for _ in range(p):
		best_candidate_idx = None
		best_obj = float('inf')

		for c_idx in range(len(candidates)):
			if c_idx in selected_indices:
				continue
			total = 0.0
			for d_idx, (_, w) in enumerate(demand):
				d = min(best_dist[d_idx], dist_cache[(d_idx, c_idx)])
				total += w * d
			if total < best_obj:
				best_obj = total
				best_candidate_idx = c_idx

		if best_candidate_idx is None:
			break

		selected_indices.append(best_candidate_idx)
		for d_idx in range(len(demand)):
			best_dist[d_idx] = min(best_dist[d_idx], dist_cache[(d_idx, best_candidate_idx)])

	# convert selected candidate indices back to cell coordinates
	selected_cells = [candidates_cells[i] for i in selected_indices]
	return selected_cells


def compute_mclp_locations(
	grid: np.ndarray, streets: np.ndarray, p: int, radius_m: float, bounds: tuple[float, float, float, float]
) -> list[tuple[int, int]]:
	"""Greedy MCLP using geographic distance (meters). Select up to p sites maximizing covered demand within radius_m."""
	grid_size = grid.shape[0]
	demand = []
	for x in range(grid_size):
		for y in range(grid_size):
			w = float(grid[x, y])
			if w <= 0:
				continue
			lon, lat = cell_center_coords(x, y, bounds, grid_size)
			demand.append(((lon, lat), w, (x, y)))

	candidates_cells = get_street_cells(streets)
	if len(candidates_cells) < p:
		raise ValueError("Not enough candidate locations for MCLP")

	selected: list[tuple[int, int]] = []
	uncovered = {(dcoord[0], dcoord[1]): w for (dcoord, w, _) in demand}

	for _ in range(p):
		best_candidate = None
		best_gain = 0.0

		for ccell in candidates_cells:
			if ccell in selected:
				continue
			c_lon, c_lat = cell_center_coords(ccell[0], ccell[1], bounds, grid_size)
			gain = 0.0
			for (d_lon, d_lat), w in uncovered.items():
				if haversine_meters(d_lon, d_lat, c_lon, c_lat) <= radius_m:
					gain += w

			if gain > best_gain:
				best_gain = gain
				best_candidate = ccell

		if best_candidate is None or best_gain <= 0.0:
			break

		# remove covered demand
		c_lon, c_lat = cell_center_coords(best_candidate[0], best_candidate[1], bounds, grid_size)
		to_remove = []
		for (d_lon, d_lat) in list(uncovered.keys()):
			if haversine_meters(d_lon, d_lat, c_lon, c_lat) <= radius_m:
				to_remove.append((d_lon, d_lat))
		for key in to_remove:
			uncovered.pop(key, None)

		selected.append(best_candidate)

	# pad if needed
	if len(selected) < p:
		remaining = [c for c in candidates_cells if c not in selected]
		random.shuffle(remaining)
		selected.extend(remaining[: max(0, p - len(selected))])

	return selected


def generate_weight() -> float:
	"""Generate random waste item weight."""
	return random.uniform(*WASTE_WEIGHT_RANGE)


def waste_generator(
	env: simpy.Environment,
	x: int,
	y: int,
	intensity: float,
	bins: list[Bin],
	waste_items: list[WasteItem],
	base_interval: float = 8.0,
	use_time_variation: bool = True,
) -> Generator[Any, Any, Any]:
	"""Generate waste at a location and route to nearest bin."""
	while True:
		factor = time_factor(env.now) if use_time_variation else 1.0
		interval = max(0.25, base_interval / ((intensity + 1.0) * factor))
		yield env.timeout(interval)
		
		weight = generate_weight()
		item = WasteItem(
			x=x,
			y=y,
			type=random.choice(WASTE_TYPES),
			weight=weight,
			time=env.now,
		)
		waste_items.append(item)
		
		nearest_bin = find_nearest_bin(item, bins)
		nearest_bin.add_waste(item)


def distance_squared(x1: int, y1: int, x2: int, y2: int) -> float:
	"""Euclidean distance squared."""
	return (x1 - x2) ** 2 + (y1 - y2) ** 2


def find_nearest_bin(item: WasteItem, bins: list[Bin]) -> Bin:
	"""Find bin closest to waste item."""
	return min(
		bins,
		key=lambda b: distance_squared(item.x, item.y, b.x, b.y),
	)


def time_factor(t: float, period: float = 10.0) -> float:
	"""Smooth daily-like wave."""
	return 0.8 + 0.5 * math.sin(t / period)


def run_simulation(
	grid: np.ndarray,
	streets: np.ndarray,
	run_until: float = SIM_TIME,
	base_interval: float = 8.0,
	use_time_variation: bool = True,
	seed: int | None = 42,
	num_bins: int = NUM_BINS,
	locations: list[tuple[int, int]] | None = None,
) -> tuple[list[Bin], list[WasteItem]]:
	"""Run the full waste management simulation."""
	if seed is not None:
		random.seed(seed)
	
	env = simpy.Environment()
	
	bins = place_bins_on_streets(env, num_bins, streets, locations=locations)
	print(f"Placed {len(bins)} bins on {np.sum(streets)} street cells")
	
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
					bins=bins,
					waste_items=waste_items,
					base_interval=base_interval,
					use_time_variation=use_time_variation,
				)
			)
	
	env.run(until=run_until)
	return bins, waste_items


def compute_metrics(bins: list[Bin], waste_items: list[WasteItem]) -> dict:
	"""Compute system-level metrics after simulation."""
	overflow_bins = sum(1 for b in bins if b.overflow)
	total_overflow_events = sum(b.overflow_events for b in bins)
	total_waste = sum(item.weight for item in waste_items)
	avg_bin_load = np.mean([b.get_load_ratio() for b in bins])
	max_bin_load = max((b.get_load_ratio() for b in bins), default=0.0)
	
	return {
		"total_bins": len(bins),
		"total_waste_generated_kg": total_waste,
		"overflow_bins": overflow_bins,
		"overflow_percentage": 100.0 * overflow_bins / len(bins) if bins else 0.0,
		"total_overflow_events": total_overflow_events,
		"avg_bin_load_ratio": avg_bin_load,
		"max_bin_load_ratio": max_bin_load,
		"total_waste_items": len(waste_items),
	}


def print_metrics(metrics: dict) -> None:
	"""Print simulation metrics."""
	print("\n" + "=" * 60)
	print("SIMULATION METRICS")
	print("=" * 60)
	print(f"Total bins:              {metrics['total_bins']}")
	print(f"Total waste generated:   {metrics['total_waste_generated_kg']:.1f} kg")
	print(f"Total waste items:       {metrics['total_waste_items']}")
	print(f"Overflow bins:           {metrics['overflow_bins']} ({metrics['overflow_percentage']:.1f}%)")
	print(f"Total overflow events:   {metrics['total_overflow_events']}")
	print(f"Avg bin load ratio:      {metrics['avg_bin_load_ratio']:.2%}")
	print(f"Max bin load ratio:      {metrics['max_bin_load_ratio']:.2%}")
	print("=" * 60 + "\n")


def main():
	"""Run the waste management simulation."""
	parser = argparse.ArgumentParser(description="SimPy waste management system with POI integration")
	parser.add_argument("--grid-size", type=int, default=GRID_SIZE, help="Grid size")
	parser.add_argument("--num-bins", type=int, default=NUM_BINS, help="Number of bins")
	parser.add_argument("--run-until", type=float, default=SIM_TIME, help="Simulation end time")
	parser.add_argument("--output-png", type=str, default="generated/system_design.png", help="Output PNG path")
	parser.add_argument("--output-html", type=str, default="generated/system_design.html", help="Output HTML path")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--place", type=str, default=DEFAULT_PLACE, help="Place name for POI fetch")
	parser.add_argument("--poi-distance", type=int, default=POI_DISTANCE_METERS, help="POI fetch distance in meters")
	parser.add_argument("--optimize", type=str, choices=["none", "p-median", "mclp"], default="none", help="Optimization method for bin placement")
	parser.add_argument("--service-radius", type=float, default=100.0, help="Service radius (in meters) for MCLP")
	args = parser.parse_args()
	
	grid, map_bounds = get_intensity_grid(
		poi_place=args.place,
		grid_size=args.grid_size,
		poi_distance=args.poi_distance,
	)

	# build street mask from OSM when bounds are available; fall back to synthetic streets
	if map_bounds is not None:
		print("Building street mask from OSM within map bounds...")
		streets = build_street_mask_from_osm(map_bounds, args.grid_size)
		if not streets.any():
			print("OSM street fetch failed or returned empty; falling back to synthetic streets")
			streets = generate_streets(args.grid_size, STREET_WIDTH)
	else:
		streets = generate_streets(args.grid_size, STREET_WIDTH)
	print(f"Generated streets ({np.sum(streets)} cells)")
	
	# compute optimized locations if requested
	locations = None
	if args.optimize != "none":
		print(f"Computing optimized bin locations: {args.optimize}")
		if args.optimize == "p-median":
			locations = compute_p_median_locations(grid, streets, args.num_bins, map_bounds)
		elif args.optimize == "mclp":
			locations = compute_mclp_locations(grid, streets, args.num_bins, args.service_radius, map_bounds)

	bins, waste_items = run_simulation(
		grid,
		streets,
		run_until=args.run_until,
		seed=args.seed,
		num_bins=args.num_bins,
		locations=locations,
	)
	
	metrics = compute_metrics(bins, waste_items)
	print_metrics(metrics)
	
	save_png_plot(bins, streets, args.grid_size, output_path=args.output_png)
	create_system_map(bins, streets, bounds=map_bounds, output_path=args.output_html)


if __name__ == "__main__":
	main()
