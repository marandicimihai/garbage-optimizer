from __future__ import annotations

import argparse
from pathlib import Path

import folium
import numpy as np
import osmnx as ox

from get_poi import DEFAULT_WEIGHTS, clean_pois, fetch_pois
from visualization import add_poi_markers, save_grid_overlay


DEFAULT_PLACE = "Chisinau, Moldova"
DEFAULT_DISTANCE_METERS = 1000
DEFAULT_OUTPUT = "generated/poi_map.html"
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


def fetch_pois_for_area(place: str, distance_meters: int, bounds: tuple[float, float, float, float]):
    raw = fetch_pois(place, distance_meters=distance_meters)
    pois = clean_pois(raw)
    minx, miny, maxx, maxy = bounds
    mask = (pois["x"] >= minx) & (pois["x"] <= maxx) & (pois["y"] >= miny) & (pois["y"] <= maxy)
    return pois.loc[mask].copy()


def build_grid_for_bounds(pois, bounds: tuple[float, float, float, float], grid_size: int):
    if pois.empty:
        raise ValueError("No POIs matched the selected filters.")

    minx, miny, maxx, maxy = bounds
    x_edges = np.linspace(minx, maxx, grid_size + 1)
    y_edges = np.linspace(miny, maxy, grid_size + 1)
    weights_array = np.array([DEFAULT_WEIGHTS.get(poi_type, 1) for poi_type in pois["type"]], dtype=float)
    grid, _, _ = np.histogram2d(
        pois["y"].to_numpy(),
        pois["x"].to_numpy(),
        bins=[y_edges, x_edges],
        weights=weights_array,
    )
    return grid, bounds


def fetch_street_lines(place: str, distance_meters: int):
    """Fetch street geometries for the same distance-based area used for POIs."""
    graph = None
    try:
        graph = ox.graph_from_point(ox.geocode(place), dist=distance_meters, network_type="drive")
    except Exception as point_exc:
        print(f"OSM point fetch failed: {point_exc}")
        try:
            graph = ox.graph_from_place(place, network_type="drive")
        except Exception as place_exc:
            print(f"OSM place fallback failed: {place_exc}")
            return []

    lines = []
    for u, v, data in graph.edges(data=True):
        geom = data.get("geometry")
        if geom is None:
            try:
                xu, yu = graph.nodes[u]["x"], graph.nodes[u]["y"]
                xv, yv = graph.nodes[v]["x"], graph.nodes[v]["y"]
                lines.append([(yu, xu), (yv, xv)])
            except Exception:
                continue
        else:
            lines.append([(lat, lon) for lon, lat in geom.coords])

    return lines


def create_combined_map(pois, grid, bounds, street_lines, output_path: str) -> str:
    minx, miny, maxx, maxy = bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

    overlay_output_path = str(Path(output_path).with_name(Path(output_path).stem + "_overlay.png"))
    save_grid_overlay(grid, overlay_output_path)

    folium.raster_layers.ImageOverlay(
        name="POI grid",
        image=str(Path(overlay_output_path).resolve()),
        bounds=[[miny, minx], [maxy, maxx]],
        opacity=0.45,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    streets_group = folium.FeatureGroup(name="Streets", show=True)
    for seg in street_lines:
        folium.PolyLine(seg, color="#222222", weight=2, opacity=0.85).add_to(streets_group)
    streets_group.add_to(m)

    add_poi_markers(m, pois)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(output_path)
    print(f"Saved combined POI + streets map to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Render streets and POIs together for the same distance-based area")
    parser.add_argument("place", nargs="?", default=None)
    parser.add_argument("--place", dest="place_opt", default=None)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--distance-meters", type=int, default=DEFAULT_DISTANCE_METERS)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    place = args.place_opt or args.place or DEFAULT_PLACE

    print(f"Fetching POIs for {place}...")
    bounds = distance_bounds(place, args.distance_meters)
    pois = fetch_pois_for_area(place, args.distance_meters, bounds)
    print(f"Filtered POIs: {len(pois)}")

    grid, bounds = build_grid_for_bounds(pois, bounds, grid_size=args.grid_size)
    street_lines = fetch_street_lines(place, args.distance_meters)
    if not street_lines:
        print("No street lines found via OSM")

    create_combined_map(pois, grid, bounds, street_lines, args.output)


if __name__ == "__main__":
    main()
