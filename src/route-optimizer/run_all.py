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
        x1 = minx + b.x * cell_width
        y1 = miny + b.y * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        folium.Rectangle(
            bounds=[[y1, x1], [y2, x2]],
            color="#90EE90",
            fill=True,
            fillColor="#90EE90",
            fillOpacity=0.7,
            weight=2,
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
    parser = argparse.ArgumentParser(description="Run POI, streets and place-bins, then generate a unified HTML map")
    parser.add_argument("place", nargs="?", default="Chisinau, Moldova")
    parser.add_argument("--distance-meters", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--bin-spacing-meters", type=float, default=40.0)
    parser.add_argument("--output", type=str, default="generated/combined_map.html")
    args = parser.parse_args()

    build_combined_map(args.place, args.distance_meters, args.grid_size, args.bin_spacing_meters, args.output)


if __name__ == "__main__":
    main()
