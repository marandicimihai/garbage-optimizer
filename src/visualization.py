from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import folium
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

if TYPE_CHECKING:
	from optimization import Bin

COLORMAP = "coolwarm"
STREET_COLOR = "#333333"
BIN_EMPTY_COLOR = "#90EE90"
BIN_MODERATE_COLOR = "#FFD700"
BIN_OVERFLOW_COLOR = "#FF6347"

TYPE_COLORS = {
	"cafe": "#ff7f0e",
	"restaurant": "#d62728",
	"fast_food": "#9467bd",
	"supermarket": "#2ca02c",
	"mall": "#1f77b4",
	"convenience": "#8c564b",
}


def save_png_plot(
	bins: list[Bin],
	streets: np.ndarray,
	grid_size: int,
	output_path: str | None = None,
) -> None:
	"""Save bin state visualization as PNG."""
	bin_map = np.zeros((grid_size, grid_size), dtype=float)
	load_ratio_map = np.zeros((grid_size, grid_size), dtype=float)
	
	for b in bins:
		bin_map[b.x, b.y] = b.current_load
		load_ratio_map[b.x, b.y] = b.get_load_ratio()
	
	fig, axes = plt.subplots(1, 3, figsize=(15, 4))
	
	axes[0].imshow(streets.astype(float), cmap="Greys", origin="lower")
	axes[0].set_title("Street Network")
	axes[0].set_xlabel("X")
	axes[0].set_ylabel("Y")
	
	im1 = axes[1].imshow(bin_map.T, origin="lower", cmap="YlOrRd")
	axes[1].set_title("Bin Load (kg)")
	axes[1].set_xlabel("X")
	axes[1].set_ylabel("Y")
	plt.colorbar(im1, ax=axes[1])
	
	im2 = axes[2].imshow(load_ratio_map.T, origin="lower", cmap="coolwarm", vmin=0, vmax=1)
	axes[2].set_title("Bin Load Ratio (%)")
	axes[2].set_xlabel("X")
	axes[2].set_ylabel("Y")
	plt.colorbar(im2, ax=axes[2], label="Load %")
	
	plt.tight_layout()
	if output_path:
		Path(output_path).parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(output_path, dpi=120, bbox_inches="tight")
		print(f"Saved PNG to {output_path}")
	else:
		plt.show()


def create_system_map(
	bins: list[Bin],
	streets: np.ndarray,
	bounds: tuple[float, float, float, float] | None = None,
	output_path: str = "generated/system_map.html",
) -> str:
	grid_size = streets.shape[0]
	
	if bounds is None:
		chisinau_lat = 47.16
		chisinau_lon = 28.66
		area_size_km = 5.0
		lat_degrees_per_km = 1 / 111.0
		lon_degrees_per_km = 1 / (111.0 * np.cos(np.radians(chisinau_lat)))
		
		lat_span = area_size_km * lat_degrees_per_km
		lon_span = area_size_km * lon_degrees_per_km
		
		miny = chisinau_lat - lat_span / 2
		maxy = chisinau_lat + lat_span / 2
		minx = chisinau_lon - lon_span / 2
		maxx = chisinau_lon + lon_span / 2
		bounds = (minx, miny, maxx, maxy)
	
	minx, miny, maxx, maxy = bounds
	center_lat = (miny + maxy) / 2
	center_lon = (minx + maxx) / 2
	
	m = folium.Map(
		location=[center_lat, center_lon],
		zoom_start=14,
		tiles="CartoDB positron",
	)
	
	street_group = folium.FeatureGroup(name="Streets", show=True)
	bin_group = folium.FeatureGroup(name="Bins", show=True)
	heatmap_group = folium.FeatureGroup(name="Load Heatmap", show=True)
	
	cell_width = (maxx - minx) / grid_size
	cell_height = (maxy - miny) / grid_size
	
	norm = Normalize(vmin=0, vmax=1)
	cmap_fn = plt.cm.get_cmap(COLORMAP)
	
	for x in range(grid_size):
		for y in range(grid_size):
			if streets[x, y]:
				x1 = minx + x * cell_width
				y1 = miny + y * cell_height
				x2 = x1 + cell_width
				y2 = y1 + cell_height
				
				folium.Rectangle(
					bounds=[[y1, x1], [y2, x2]],
					color=STREET_COLOR,
					fill=True,
					fillColor=STREET_COLOR,
					fillOpacity=0.4,
					weight=1,
				).add_to(street_group)
	
	for b in bins:
		x1 = minx + b.x * cell_width
		y1 = miny + b.y * cell_height
		x2 = x1 + cell_width
		y2 = y1 + cell_height
		
		load_ratio = b.get_load_ratio()
		if load_ratio < 0.7:
			color = BIN_EMPTY_COLOR
		elif load_ratio < 0.95:
			color = BIN_MODERATE_COLOR
		else:
			color = BIN_OVERFLOW_COLOR
		
		folium.Rectangle(
			bounds=[[y1, x1], [y2, x2]],
			color=color,
			fill=True,
			fillColor=color,
			fillOpacity=0.7,
			weight=2,
			popup=f"Bin {b.bin_id}<br>Load: {load_ratio:.1%}<br>{b.current_load:.1f}/{b.capacity:.1f}kg",
		).add_to(bin_group)
	
	for b in bins:
		x1 = minx + b.x * cell_width
		y1 = miny + b.y * cell_height
		x2 = x1 + cell_width
		y2 = y1 + cell_height
		
		load_ratio = b.get_load_ratio()
		rgba = cmap_fn(norm(load_ratio))
		hex_color = "#{:02x}{:02x}{:02x}".format(
			int(rgba[0] * 255),
			int(rgba[1] * 255),
			int(rgba[2] * 255),
		)
		
		folium.Rectangle(
			bounds=[[y1, x1], [y2, x2]],
			color=hex_color,
			fill=True,
			fillColor=hex_color,
			fillOpacity=0.5,
			weight=0.5,
			popup=f"Load: {load_ratio:.1%}",
		).add_to(heatmap_group)
	
	street_group.add_to(m)
	bin_group.add_to(m)
	heatmap_group.add_to(m)
	folium.LayerControl(collapsed=False).add_to(m)
	m.fit_bounds([[miny, minx], [maxy, maxx]])
	
	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	m.save(output_path)
	print(f"Saved interactive map to {output_path}")
	
	return output_path


def create_poi_map(
	grid: np.ndarray,
	bounds: tuple[float, float, float, float],
	pois,
	place: str,
	overlay_opacity: float = 0.45,
	map_output_path: str = "generated/poi_map.html",
	overlay_output_path: str = "generated/poi_overlay.png",
) -> tuple[str, str]:
	minx, miny, maxx, maxy = bounds
	center_lat = (miny + maxy) / 2
	center_lon = (minx + maxx) / 2
	
	m = folium.Map(
		location=[center_lat, center_lon],
		zoom_start=14,
		tiles="CartoDB positron",
	)
	
	save_grid_overlay(grid, overlay_output_path)
	
	folium.raster_layers.ImageOverlay(
		name="POI grid",
		image=str(Path(overlay_output_path).resolve()),
		bounds=[[miny, minx], [maxy, maxx]],
		opacity=overlay_opacity,
		interactive=False,
		cross_origin=False,
		zindex=1,
	).add_to(m)
	
	add_poi_markers(m, pois)
	folium.LayerControl(collapsed=False).add_to(m)
	m.fit_bounds([[miny, minx], [maxy, maxx]])
	
	Path(map_output_path).parent.mkdir(parents=True, exist_ok=True)
	m.save(map_output_path)
	print(f"Saved POI map to {map_output_path}")
	
	return map_output_path, overlay_output_path


def save_grid_overlay(grid: np.ndarray, output_path: str) -> str:
	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	flipped = np.flipud(grid)
	plt.imsave(output_path, flipped, cmap="hot")
	return output_path


def add_poi_markers(m, pois) -> None:
	marker_layer = folium.FeatureGroup(name="POIs", show=True)
	for _, row in pois.iterrows():
		poi_color_val = TYPE_COLORS.get(row["type"], "#111111")
		folium.CircleMarker(
			location=[row["y"], row["x"]],
			radius=4,
			color=poi_color_val,
			fill=True,
			fill_color=poi_color_val,
			fill_opacity=0.9,
			weight=1,
			popup=row["type"],
		).add_to(marker_layer)
	marker_layer.add_to(m)
