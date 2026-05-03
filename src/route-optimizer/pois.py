from pathlib import Path
from typing import Iterable

from geopandas import GeoDataFrame
import numpy as np
import osmnx as ox
import pandas as pd
import requests

DEFAULT_RELEVANT_TYPES = [
	"cafe",
	"restaurant",
	"fast_food",
	"supermarket",
	"mall",
	"convenience",
]

def infer_poi_type(row) -> str:
	for column in ("amenity", "shop"):
		value = row.get(column)
		if value is not None and not pd.isna(value):
			return str(value)
	return "other"

def normalize_geometries(gdf):
	normalized = gdf.copy()
	normalized["geometry"] = normalized.geometry.representative_point()
	return normalized

def clean_pois(
	gdf,
	relevant_types: Iterable[str] | None = None,
) -> GeoDataFrame:
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

def main():
	place = "Chisinau, Moldova"
	distance_meters = 1000
	requests_timeout = 60
	
	pois_output = "generated/pois.csv"

	ox.settings.requests_timeout = requests_timeout

	print(f"Fetching POIs for {place} with timeout {requests_timeout}s...")
	try:
		raw = ox.features_from_point(ox.geocode(place), {
			"amenity": True,
			"shop": True,
		}, dist=distance_meters)
	except requests.RequestException as exc:
		raise SystemExit(
			"Failed to fetch POIs from OpenStreetMap. "
			"Try again later, increase requests_timeout, or reduce distance_meters."
		) from exc

	Path(pois_output).parent.mkdir(parents=True, exist_ok=True)
	pois = clean_pois(raw)
	pois = pois.drop(columns=["amenity", "shop"])
	pois.to_csv(pois_output, index=False)


if __name__ == "__main__":
	main()
