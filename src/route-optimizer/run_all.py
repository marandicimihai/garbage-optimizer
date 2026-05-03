from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import folium

from get_poi import fetch_pois, clean_pois, build_grid, save_grid_overlay
from streets import fetch_street_lines, distance_bounds
from place_bins_on_streets import place_bins_on_streets, build_streets_grid
from visualization import create_system_map, add_poi_markers


def build_combined_map(place: str, distance_meters: int, grid_size: int, bin_spacing_meters: float, output: str) -> str:
    bounds = distance_bounds(place, distance_meters)

    raw = fetch_pois(place, distance_meters=distance_meters)
    pois = clean_pois(raw)
    poi_grid, bounds = build_grid(pois, grid_size=grid_size)
    overlay_path = save_grid_overlay(poi_grid, str(Path(output).with_name(Path(output).stem + "_poi_overlay.png")))

    street_lines = fetch_street_lines(place, distance_meters)
    bins, streets_grid, _ = place_bins_on_streets(place, distance_meters, grid_size, bin_spacing_meters)

    # Build base map
    minx, miny, maxx, maxy = bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

    # POI overlay
    folium.raster_layers.ImageOverlay(
        name="POI grid",
        image=str(Path(overlay_path).resolve()),
        bounds=[[miny, minx], [maxy, maxx]],
        opacity=0.45,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    # streets
    streets_group = folium.FeatureGroup(name="Streets", show=True)
    for seg in street_lines:
        folium.PolyLine(seg, color="#222222", weight=2, opacity=0.85).add_to(streets_group)
    streets_group.add_to(m)

    # POI markers
    add_poi_markers(m, pois)

    # Bins: reuse create_system_map-like rectangles but add to this map
    grid_size = streets_grid.shape[0]
    minx, miny, maxx, maxy = bounds
    cell_width = (maxx - minx) / grid_size
    cell_height = (maxy - miny) / grid_size

    bin_group = folium.FeatureGroup(name="Bins", show=True)
    for b in bins:
        # bins are points on streets (lat, lon)
        folium.CircleMarker(
            location=[b.lat, b.lon],
            radius=5,
            color="#2ca02c",
            fill=True,
            fillColor="#2ca02c",
            fillOpacity=0.9,
            weight=1,
            popup=f"Bin {b.bin_id}<br>Load: {b.get_load_ratio():.1%}<br>{b.current_load:.1f}/{b.capacity:.1f}kg",
        ).add_to(bin_group)
    bin_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    m.save(output)
    print(f"Saved combined map to {output}")
    return output


def main():
    # No-CLI defaults
    place = "Chisinau, Moldova"
    distance_meters = 1000
    grid_size = 100
    bin_spacing_meters = 70.0
    output = "generated/combined_map.html"

    build_combined_map(place, distance_meters, grid_size, bin_spacing_meters, output)


if __name__ == "__main__":
    main()
