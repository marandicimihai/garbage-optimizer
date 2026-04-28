from __future__ import annotations

"""Minimal POI -> grid -> Leaflet overlay utilities.

This module provides small, import-friendly functions to fetch POIs from
OpenStreetMap, build a weighted grid, and save a Leaflet HTML map with the
grid as a semi-transparent overlay plus POI markers.

Keep usage minimal: import the functions you need (e.g. `build_grid`).
"""

import argparse
from pathlib import Path
from typing import Iterable

import folium
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import requests


DEFAULT_TAGS = {
	"amenity": True,
	"shop": True,
}

DEFAULT_RELEVANT_TYPES = [
	"cafe",
	"restaurant",
	"fast_food",
	"supermarket",
	"mall",
	"convenience",
]

DEFAULT_WEIGHTS = {
	"cafe": 2,
	"restaurant": 3,
	"fast_food": 3,
	"supermarket": 4,
	"mall": 5,
	"convenience": 2,
}

TYPE_COLORS = {
	"cafe": "#ff7f0e",
	"restaurant": "#d62728",
	"fast_food": "#9467bd",
	"supermarket": "#2ca02c",
	"mall": "#1f77b4",
	"convenience": "#8c564b",
}

__all__ = [
    "fetch_pois",
    "clean_pois",
    "build_grid",
    "build_leaflet_map",
    "save_grid_overlay",
    "add_poi_markers",
]


def fetch_pois(place: str, tags: dict[str, object] | None = None, distance_meters: int = 3000):
	"""Fetch POIs near the center of a place from OpenStreetMap."""
	center = ox.geocode(place)
	return ox.features_from_point(center, tags or DEFAULT_TAGS, dist=distance_meters)


def infer_poi_type(row) -> str:
	for column in ("amenity", "shop"):
		value = row.get(column)
		if value is not None and not pd.isna(value):
			return str(value)
	return "other"


def normalize_geometries(gdf):
	"""Convert polygons and lines to interior points so we can grid them."""
	normalized = gdf.copy()
	normalized["geometry"] = normalized.geometry.representative_point()
	return normalized


def clean_pois(
	gdf,
	relevant_types: Iterable[str] | None = None,
):
	columns = [column for column in ("geometry", "amenity", "shop") if column in gdf.columns]
	useful = gdf[columns].copy()
	useful["type"] = useful.apply(infer_poi_type, axis=1)
	useful = useful[useful["type"] != "other"]

	relevant = set(relevant_types or DEFAULT_RELEVANT_TYPES)
	useful = useful[useful["type"].isin(relevant)]
	useful = normalize_geometries(useful)
	useful["x"] = useful.geometry.x
	useful["y"] = useful.geometry.y
	return useful


def build_grid(pois, grid_size: int = 50, weights: dict[str, float] | None = None):
	if pois.empty:
		raise ValueError("No POIs matched the selected filters.")

	minx, miny, maxx, maxy = pois.total_bounds
	weight_map = weights or DEFAULT_WEIGHTS

	# Build grid from explicit bin edges to avoid half-cell offset on map overlays.
	x_edges = np.linspace(minx, maxx, grid_size + 1)
	y_edges = np.linspace(miny, maxy, grid_size + 1)
	weights_array = np.array([weight_map.get(poi_type, 1) for poi_type in pois["type"]], dtype=float)
	grid, _, _ = np.histogram2d(
		pois["y"].to_numpy(),
		pois["x"].to_numpy(),
		bins=[y_edges, x_edges],
		weights=weights_array,
	)

	return grid, (minx, miny, maxx, maxy)


def save_grid_overlay(grid, output_path: str = "poi_overlay.png"):
	# ensure parent directory exists
	output_path = str(output_path)
	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	flipped = np.flipud(grid)
	plt.imsave(output_path, flipped, cmap="hot")
	return output_path


def poi_color(poi_type: str) -> str:
	return TYPE_COLORS.get(poi_type, "#111111")


def add_poi_markers(map_view, pois):
	marker_layer = folium.FeatureGroup(name="POIs", show=True)
	for _, row in pois.iterrows():
		folium.CircleMarker(
			location=[row["y"], row["x"]],
			radius=4,
			color=poi_color(row["type"]),
			fill=True,
			fill_color=poi_color(row["type"]),
			fill_opacity=0.9,
			weight=1,
			popup=f'{row["type"]}',
		).add_to(marker_layer)
	marker_layer.add_to(map_view)
	return marker_layer


def build_leaflet_map(
	place: str,
	grid,
	bounds,
	pois,
	overlay_opacity: float = 0.45,
	map_output_path: str = "generated/poi_map.html",
	overlay_output_path: str = "generated/poi_overlay.png",
):
	minx, miny, maxx, maxy = bounds
	center_lat = (miny + maxy) / 2
	center_lon = (minx + maxx) / 2

	map_view = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")
	overlay_path = save_grid_overlay(grid, overlay_output_path)
	folium.raster_layers.ImageOverlay(
		name="POI grid",
		image=str(Path(overlay_path).resolve()),
		bounds=[[miny, minx], [maxy, maxx]],
		opacity=overlay_opacity,
		interactive=False,
		cross_origin=False,
		zindex=1,
	).add_to(map_view)
	add_poi_markers(map_view, pois)
	folium.LayerControl(collapsed=False).add_to(map_view)
	map_view.fit_bounds([[miny, minx], [maxy, maxx]])
	# ensure parent directory for the map exists
	Path(map_output_path).parent.mkdir(parents=True, exist_ok=True)
	map_view.save(map_output_path)
	return map_output_path, overlay_path


def main():
	parser = argparse.ArgumentParser(description="Build a POI density grid from OpenStreetMap data.")
	parser.add_argument("place", nargs="?", default="Chisinau, Moldova")
	parser.add_argument("--grid-size", type=int, default=100)
	parser.add_argument("--distance-meters", type=int, default=1000)
	parser.add_argument("--requests-timeout", type=int, default=60)
	parser.add_argument("--overlay-opacity", type=float, default=0.45)
	parser.add_argument("--map-output", default="generated/poi_map.html")
	parser.add_argument("--overlay-output", default="generated/poi_overlay.png")
	args = parser.parse_args()

	ox.settings.requests_timeout = args.requests_timeout

	print(f"Fetching POIs for {args.place} with timeout {args.requests_timeout}s...")
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
	map_output_path, overlay_output_path = build_leaflet_map(
		args.place,
		grid,
		bounds,
		pois,
		overlay_opacity=args.overlay_opacity,
		map_output_path=args.map_output,
		overlay_output_path=args.overlay_output,
	)
	print(f"Saved Leaflet map to {map_output_path}")
	print(f"Saved POI overlay to {overlay_output_path}")


if __name__ == "__main__":
	main()
