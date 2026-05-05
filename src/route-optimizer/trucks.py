from __future__ import annotations

import csv
import heapq
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


DEFAULT_OUTPUT = "truck_routes.csv"
EMISSION_FACTOR_KG_PER_KM = 2.0


def project_root() -> Path:
	return Path(__file__).resolve().parents[2]


def generated_dir() -> Path:
	return project_root() / "src" / "route-optimizer" / "generated"


def bins_path() -> Path:
	return generated_dir() / "bins.csv"


def waste_events_path() -> Path:
	return generated_dir() / "waste_events.csv"


def streets_path() -> Path:
	return generated_dir() / "street_lines.csv"


def output_path() -> Path:
	return generated_dir() / DEFAULT_OUTPUT


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
	lat1, lon1 = a
	lat2, lon2 = b
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlambda = math.radians(lon2 - lon1)
	h = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
	return 2.0 * 6_371_000.0 * math.asin(math.sqrt(h))


def read_bins(path: Path) -> list[dict[str, float | int]]:
	bins: list[dict[str, float | int]] = []
	with path.open("r", encoding="utf-8", newline="") as file_handle:
		reader = csv.DictReader(file_handle)
		for row in reader:
			try:
				bin_id = int(row["binId"])
				lon = float(row["x"])
				lat = float(row["y"])
			except (KeyError, TypeError, ValueError):
				continue
			bins.append({"binId": bin_id, "lat": lat, "lon": lon})
	return bins


def read_waste_events(path: Path) -> list[dict[str, object]]:
	events: list[dict[str, object]] = []
	with path.open("r", encoding="utf-8", newline="") as file_handle:
		reader = csv.DictReader(file_handle)
		for row in reader:
			try:
				timestamp = str(row["timestamp"])
				day = timestamp[:10]
				bin_id = int(row["binId"])
				weight = float(row["weight"])
			except (KeyError, TypeError, ValueError):
				continue
			events.append({"date": day, "binId": bin_id, "weight": weight})
	return events


def read_street_lines(path: Path) -> list[list[tuple[float, float]]]:
	lines_by_id: dict[int, list[tuple[int, float, float]]] = {}
	with path.open("r", encoding="utf-8", newline="") as file_handle:
		reader = csv.DictReader(file_handle)
		for row in reader:
			try:
				line_id = int(row["line_id"])
				point_order = int(row["point_order"])
				lat = float(row["lat"])
				lon = float(row["lon"])
			except (KeyError, TypeError, ValueError):
				continue
			lines_by_id.setdefault(line_id, []).append((point_order, lat, lon))

	polylines: list[list[tuple[float, float]]] = []
	for line_id in sorted(lines_by_id):
		ordered = sorted(lines_by_id[line_id], key=lambda item: item[0])
		coords = [(lat, lon) for _, lat, lon in ordered]
		if len(coords) >= 2:
			polylines.append(coords)
	return polylines


def build_graph(lines: list[list[tuple[float, float]]]) -> tuple[dict[tuple[float, float], list[tuple[tuple[float, float], float]]], list[tuple[float, float]]]:
	graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]] = {}
	for line in lines:
		for start, end in zip(line, line[1:]):
			length = haversine_m(start, end)
			if length <= 0.0:
				continue
			graph.setdefault(start, []).append((end, length))
			graph.setdefault(end, []).append((start, length))
	for node in list(graph):
		graph.setdefault(node, [])
	return graph, list(graph)


def dijkstra_paths(
	graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
	start: tuple[float, float],
	target: tuple[float, float],
) -> list[tuple[float, float]]:
	if start == target:
		return [start]

	dist: dict[tuple[float, float], float] = {start: 0.0}
	prev: dict[tuple[float, float], tuple[float, float]] = {}
	heap: list[tuple[float, tuple[float, float]]] = [(0.0, start)]

	while heap:
		current_dist, node = heapq.heappop(heap)
		if node == target:
			break
		if current_dist > dist.get(node, float("inf")):
			continue
		for neighbor, weight in graph.get(node, []):
			candidate = current_dist + weight
			if candidate + 1e-9 < dist.get(neighbor, float("inf")):
				dist[neighbor] = candidate
				prev[neighbor] = node
				heapq.heappush(heap, (candidate, neighbor))

	if target not in dist:
		return [start, target]

	path = [target]
	current = target
	while current != start:
		current = prev[current]
		path.append(current)
	path.reverse()
	return path


def nearest_node(
	point: tuple[float, float],
	nodes: list[tuple[float, float]],
) -> tuple[float, float]:
	return min(nodes, key=lambda node: haversine_m(point, node))


def compute_depot(bins: list[dict[str, float | int]]) -> tuple[float, float]:
	lat_values = [float(bin_item["lat"]) for bin_item in bins]
	lon_values = [float(bin_item["lon"]) for bin_item in bins]
	if not lat_values or not lon_values:
		return 47.0105, 28.8638
	return sum(lat_values) / len(lat_values), sum(lon_values) / len(lon_values)


def build_waste_matrix(
	bins: list[dict[str, float | int]],
	waste_events: list[dict[str, object]],
	day_labels: list[str],
) -> tuple[dict[int, list[float]], np.ndarray, np.ndarray]:
	day_index = {label: idx for idx, label in enumerate(day_labels)}
	daily_loads: dict[int, list[float]] = {int(b["binId"]): [0.0] * len(day_labels) for b in bins}
	
	for event in waste_events:
		label = str(event["date"])
		idx = day_index.get(label)
		if idx is None:
			continue
		bin_id = int(event["binId"])
		weight = float(event["weight"])
		daily_loads.setdefault(bin_id, [0.0] * len(day_labels))[idx] += weight
	
	bin_ids = sorted(daily_loads.keys())
	X = np.array([daily_loads[bid] for bid in bin_ids])
	y = np.sum(X, axis=1)
	return daily_loads, X, y


def train_fullness_model(
	X: np.ndarray,
	y: np.ndarray,
) -> tuple[LinearRegression, StandardScaler]:
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	model = LinearRegression()
	model.fit(X_scaled, y)
	return model, scaler


def predict_bin_fullness_for_day(
	bins: list[dict[str, float | int]],
	waste_events: list[dict[str, object]],
	target_day: str,
	model: LinearRegression,
	scaler: StandardScaler,
	daily_loads: dict[int, list[float]],
	day_labels: list[str],
) -> dict[int, float]:
	day_index = {label: idx for idx, label in enumerate(day_labels)}
	target_idx = day_index.get(target_day, 0)
	
	bin_ids = sorted({int(b["binId"]) for b in bins})
	predictions = {}
	
	for bin_id in bin_ids:
		historical = daily_loads.get(bin_id, [0.0] * len(day_labels))
		X_bin = np.array(historical).reshape(1, -1)
		X_scaled = scaler.transform(X_bin)
		predicted_weight = max(0.0, model.predict(X_scaled)[0])
		
		current_load = historical[target_idx] if target_idx < len(historical) else 0.0
		capacity_factor = min(1.0, (current_load + predicted_weight) / 100.0)
		predictions[bin_id] = capacity_factor
	
	return predictions


def compute_bin_priority_scores(
	bins: list[dict[str, float | int]],
	daily_loads: dict[int, list[float]],
	day_index_val: int,
	fullness_predictions: dict[int, float],
) -> dict[int, float]:
	series_length = len(next(iter(daily_loads.values()), []))
	empty_series = [0.0] * series_length
	scores = {}
	for bin_item in bins:
		bin_id = int(bin_item["binId"])
		current = daily_loads.get(bin_id, empty_series)[day_index_val]
		predicted = fullness_predictions.get(bin_id, 0.5)
		combined_fullness = min(1.0, (current / 100.0) + predicted * 0.5)
		scores[bin_id] = combined_fullness
	
	return scores


def route_order_ml(
	bins: list[dict[str, float | int]],
	depot: tuple[float, float],
	bin_nodes: dict[int, tuple[float, float]],
	graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
	daily_loads: dict[int, list[float]],
	day_index_val: int,
	fullness_predictions: dict[int, float],
	min_fullness_threshold: float = 0.2,
) -> list[dict[str, float | int]]:
	priority_scores = compute_bin_priority_scores(bins, daily_loads, day_index_val, fullness_predictions)
	
	bins_by_priority = sorted(
		bins,
		key=lambda b: -priority_scores.get(int(b["binId"]), 0.0),
	)
	
	high_priority = [
		b for b in bins_by_priority
		if priority_scores.get(int(b["binId"]), 0.0) >= min_fullness_threshold
	]
	
	if not high_priority:
		high_priority = bins_by_priority[:max(1, len(bins_by_priority) // 3)]
	
	nodes_list = list(graph)
	depot_node = nearest_node(depot, nodes_list)
	
	unvisited = {int(b["binId"]): b for b in high_priority}
	ordered = []
	current_node = depot_node
	
	while unvisited:
		nearest_bin_id = min(
			unvisited.keys(),
			key=lambda bid: street_distance(current_node, bin_nodes[bid], bin_nodes, graph),
		)
		bin_item = unvisited.pop(nearest_bin_id)
		ordered.append(bin_item)
		current_node = bin_nodes[nearest_bin_id]
	
	return ordered


def street_distance(
	start: tuple[float, float],
	end: tuple[float, float],
	bin_nodes: dict[int, tuple[float, float]],
	graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
) -> float:
	if start == end:
		return 0.0
	path = dijkstra_paths(graph, start, end)
	total = 0.0
	for p1, p2 in zip(path, path[1:]):
		total += haversine_m(p1, p2)
	return total


def route_order(
	bins: list[dict[str, float | int]],
	depot: tuple[float, float],
	bin_nodes: dict[int, tuple[float, float]],
	graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
) -> list[dict[str, float | int]]:
	if not bins:
		return []

	nodes_list = list(graph)
	depot_node = nearest_node(depot, nodes_list)

	unvisited = {int(bin_item["binId"]): bin_item for bin_item in bins}
	ordered = []
	current_node = depot_node

	while unvisited:
		nearest_bin_id = min(
			unvisited.keys(),
			key=lambda bid: street_distance(current_node, bin_nodes[bid], bin_nodes, graph),
		)
		bin_item = unvisited.pop(nearest_bin_id)
		ordered.append(bin_item)
		current_node = bin_nodes[nearest_bin_id]

	return ordered


def route_distance_m(
	depot: tuple[float, float],
	ordered_bins: list[dict[str, float | int]],
	bin_nodes: dict[int, tuple[float, float]],
	graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
) -> float:
	if not ordered_bins:
		return 0.0
	total = 0.0
	current = nearest_node(depot, list(graph))
	for bin_item in ordered_bins:
		target = bin_nodes[int(bin_item["binId"])]
		path = dijkstra_paths(graph, current, target)
		for start, end in zip(path, path[1:]):
			total += haversine_m(start, end)
		current = target
	path = dijkstra_paths(graph, current, nearest_node(depot, list(graph)))
	for start, end in zip(path, path[1:]):
		total += haversine_m(start, end)
	return total


def build_route_path(
	depot: tuple[float, float],
	ordered_bins: list[dict[str, float | int]],
	bin_nodes: dict[int, tuple[float, float]],
	graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
) -> list[list[float]]:
	if not ordered_bins or not graph:
		return [[round(depot[0], 6), round(depot[1], 6)]]

	path_coords: list[list[float]] = []
	current = nearest_node(depot, list(graph))
	path_coords.append([round(current[0], 6), round(current[1], 6)])

	for bin_item in ordered_bins:
		target = bin_nodes[int(bin_item["binId"])]
		segment = dijkstra_paths(graph, current, target)
		for node in segment[1:]:
			path_coords.append([round(node[0], 6), round(node[1], 6)])
		current = target

	segment = dijkstra_paths(graph, current, nearest_node(depot, list(graph)))
	for node in segment[1:]:
		path_coords.append([round(node[0], 6), round(node[1], 6)])

	return path_coords


def build_routes_for_mode(
	bins: list[dict[str, float | int]],
	waste_events: list[dict[str, object]],
	route_type: str,
	depot: tuple[float, float],
	graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
	nodes: list[tuple[float, float]],
	bin_nodes: dict[int, tuple[float, float]],
	daily_loads: dict[int, list[float]],
	day_labels: list[str],
	day_index: dict[str, int],
	model: LinearRegression,
	scaler: StandardScaler,
) -> list[dict[str, object]]:
	routes: list[dict[str, object]] = []
	shared_order: list[dict[str, float | int]] | None = None
	shared_route_path: list[list[float]] | None = None
	shared_distance_m = 0.0
	shared_co2_kg = 0.0
	if route_type == "normal":
		shared_order = route_order(bins, depot, bin_nodes, graph)
		shared_route_path = build_route_path(depot, shared_order, bin_nodes, graph)
		shared_distance_m = round(route_distance_m(depot, shared_order, bin_nodes, graph), 3)
		shared_co2_kg = round((shared_distance_m / 1000.0) * EMISSION_FACTOR_KG_PER_KM, 3)

	for label in day_labels:
		index = day_index[label]
		if route_type == "normal":
			ordered_bins = shared_order or []
		else:
			fullness_predictions = predict_bin_fullness_for_day(
				bins, waste_events, label, model, scaler, daily_loads, day_labels
			)
			ordered_bins = route_order_ml(
				bins,
				depot,
				bin_nodes,
				graph,
				daily_loads,
				index,
				fullness_predictions,
				min_fullness_threshold=0.15,
			)

		if route_type == "normal":
			route_path = shared_route_path or [[round(depot[0], 6), round(depot[1], 6)]]
			distance_m = shared_distance_m
			co2_kg = shared_co2_kg
		else:
			route_path = build_route_path(depot, ordered_bins, bin_nodes, graph)
			distance_m = round(route_distance_m(depot, ordered_bins, bin_nodes, graph), 3)
			co2_kg = round((distance_m / 1000.0) * EMISSION_FACTOR_KG_PER_KM, 3)

		# Ensure any bins that lie on the computed route path and have
		# non-zero collection for this day are included in the ordered list.
		# This covers bins that are located on intermediate nodes the truck
		# passes by but were excluded by the priority selection.
		if route_path:
			path_node_set = {(round(n[0], 6), round(n[1], 6)) for n in route_path}
			existing_ids = {int(b["binId"]) for b in ordered_bins}
			extra_bins: list[dict[str, float | int]] = []
			for bin_item in bins:
				bid = int(bin_item["binId"])
				if bid in existing_ids:
					continue
				node = bin_nodes.get(bid)
				if not node:
					continue
				if (round(node[0], 6), round(node[1], 6)) in path_node_set:
					collected = daily_loads.get(bid, [0.0] * len(day_labels))[index]
					if collected and float(collected) > 1e-9:
						extra_bins.append(bin_item)

			if extra_bins:
				# merge ordered_bins and extra_bins, sorting by their position on the path
				def path_pos_for(bin_item: dict[str, float | int]) -> int:
					node = bin_nodes.get(int(bin_item["binId"]))
					if not node:
						return 10**9
					rounded = (round(node[0], 6), round(node[1], 6))
					for i, n in enumerate(route_path):
						if (round(n[0], 6), round(n[1], 6)) == rounded:
							return i
					return 10**9

				combined = list(ordered_bins) + list(extra_bins)
				combined.sort(key=path_pos_for)
				ordered_bins = combined

		day_collection = 0.0
		for bin_item in ordered_bins:
			day_collection += daily_loads.get(int(bin_item["binId"]), [0.0] * len(day_labels))[index]

		for stop_order, bin_item in enumerate(ordered_bins):
			routes.append(
				{
					"route_type": route_type,
					"date": label,
					"day_index": index,
					"stop_order": stop_order,
					"binId": int(bin_item["binId"]),
					"lat": round(float(bin_item["lat"]), 6),
					"lon": round(float(bin_item["lon"]), 6),
					"depot_lat": round(depot[0], 6),
					"depot_lon": round(depot[1], 6),
					"collected_kg": round(daily_loads.get(int(bin_item["binId"]), [0.0] * len(day_labels))[index], 3),
					"day_collection_kg": round(day_collection, 3),
					"day_distance_m": distance_m,
					"day_co2_kg": co2_kg,
					"route_path": route_path,
				}
			)

	return routes


def build_routes(
	bins: list[dict[str, float | int]],
	waste_events: list[dict[str, object]],
) -> list[dict[str, object]]:
	day_labels = sorted({str(event["date"]) for event in waste_events})
	if not day_labels:
		return []

	day_index = {label: index for index, label in enumerate(day_labels)}
	daily_loads, X, y = build_waste_matrix(bins, waste_events, day_labels)
	model, scaler = train_fullness_model(X, y)

	depot = compute_depot(bins)
	street_lines = read_street_lines(streets_path())
	graph, nodes = build_graph(street_lines)
	bin_nodes = {
		int(bin_item["binId"]): nearest_node((float(bin_item["lat"]), float(bin_item["lon"])), nodes)
		for bin_item in bins
	}

	routes: list[dict[str, object]] = []
	routes.extend(
		build_routes_for_mode(
			bins,
			waste_events,
			"normal",
			depot,
			graph,
			nodes,
			bin_nodes,
			daily_loads,
			day_labels,
			day_index,
			model,
			scaler,
		)
	)
	routes.extend(
		build_routes_for_mode(
			bins,
			waste_events,
			"ml",
			depot,
			graph,
			nodes,
			bin_nodes,
			daily_loads,
			day_labels,
			day_index,
			model,
			scaler,
		)
	)
	return routes


def write_routes(path: Path, routes: list[dict[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as file_handle:
		writer = csv.writer(file_handle)
		writer.writerow([
			"route_type",
			"date",
			"day_index",
			"stop_order",
			"binId",
			"lat",
			"lon",
			"depot_lat",
			"depot_lon",
			"collected_kg",
			"day_collection_kg",
			"day_distance_m",
			"day_co2_kg",
			"route_path",
		])
		for route in routes:
			writer.writerow([
				route["route_type"],
				route["date"],
				route["day_index"],
				route["stop_order"],
				route["binId"],
				route["lat"],
				route["lon"],
				route["depot_lat"],
				route["depot_lon"],
				route["collected_kg"],
				route["day_collection_kg"],
				route["day_distance_m"],
				route["day_co2_kg"],
				json.dumps(route["route_path"], ensure_ascii=True),
			])


def main() -> None:
	bins_file = bins_path()
	waste_events_file = waste_events_path()
	output_file = output_path()

	if not bins_file.exists():
		raise SystemExit(f"Missing bin input: {bins_file}")
	if not waste_events_file.exists():
		raise SystemExit(f"Missing waste event input: {waste_events_file}")
	if not streets_path().exists():
		raise SystemExit(f"Missing street input: {streets_path()}")

	bins = read_bins(bins_file)
	waste_events = read_waste_events(waste_events_file)
	if not bins:
		raise SystemExit(f"No bins loaded from {bins_file}")
	if not waste_events:
		raise SystemExit(f"No waste events loaded from {waste_events_file}")

	routes = build_routes(bins, waste_events)
	if not routes:
		raise SystemExit("No truck routes could be built")

	write_routes(output_file, routes)
	day_count = len({route["date"] for route in routes})
	mode_count = len({route["route_type"] for route in routes})
	print(f"Saved {len(routes)} truck sweep rows across {day_count} days and {mode_count} truck modes to {output_file}")


if __name__ == "__main__":
	main()
