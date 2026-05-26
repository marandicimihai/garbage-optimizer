from __future__ import annotations

import csv
import heapq
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from geometry import normalize_street_lines


DEFAULT_OUTPUT = "truck_routes.csv"
EMISSION_FACTOR_KG_PER_KM = 2.0
DEFAULT_SWEEP_GRID_METERS = 200.0


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
	return normalize_street_lines(polylines)


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


def insert_bins_into_lines(lines: list[list[tuple[float, float]]], bins: list[dict[str, float | int]]) -> tuple[list[list[tuple[float, float]]], dict[int, tuple[float, float]]]:
	"""Insert projected bin points into polylines by splitting segments.

	Returns (new_lines, mapping binId -> projected latlon).
	"""
	bin_to_best: dict[int, tuple[int, int, float, tuple[float, float]]] = {}
	# iterate lines and segments to find best projection for each bin
	for line_idx, line in enumerate(lines):
		for seg_idx, (a, b) in enumerate(zip(line, line[1:])):
			a_latlon = a
			b_latlon = b
			for bitem in bins:
				bid = int(bitem["binId"])
				point = (float(bitem["lat"]), float(bitem["lon"]))
				proj, dist = project_point_on_segment(point, a_latlon, b_latlon)
				# store best across all lines/segments
				prev = bin_to_best.get(bid)
				if prev is None or dist < prev[2]:
					bin_to_best[bid] = (line_idx, seg_idx, dist, proj)

	# prepare new lines with inserted projection points
	new_lines: list[list[tuple[float, float]]] = []
	# mapping bin id -> proj coordinate
	bin_proj_map: dict[int, tuple[float, float]] = {}

	for line_idx, line in enumerate(lines):
		# collect inserts for this line: per segment index a list of (t, proj)
		inserts_by_seg: dict[int, list[tuple[float, tuple[float, float]]]] = {}
		for bid, (li, seg_idx, dist, proj) in list(bin_to_best.items()):
			if li != line_idx:
				continue
			# compute t along segment for ordering
			a = line[seg_idx]
			b = line[seg_idx + 1]
			# compute t in local meters
			mean_lat = math.radians((a[0] + b[0] + proj[0]) / 3.0)
			cos_lat = math.cos(mean_lat)
			deg_to_m = 111000.0
			ax = a[1] * deg_to_m * cos_lat
			ay = a[0] * deg_to_m
			bx = b[1] * deg_to_m * cos_lat
			by = b[0] * deg_to_m
			px = proj[1] * deg_to_m * cos_lat
			py = proj[0] * deg_to_m
			vx = bx - ax
			vy = by - ay
			denom = vx * vx + vy * vy
			if denom == 0.0:
				t = 0.0
			else:
				t = ((px - ax) * vx + (py - ay) * vy) / denom
				t = max(0.0, min(1.0, t))
			inserts_by_seg.setdefault(seg_idx, []).append((t, proj, bid))

		# build new line
		new_pts: list[tuple[float, float]] = []
		for idx, pt in enumerate(line):
			new_pts.append(pt)
			if idx < len(line) - 1:
				seg_inserts = inserts_by_seg.get(idx, [])
				if seg_inserts:
					seg_inserts.sort(key=lambda x: x[0])
					for t, proj, bid in seg_inserts:
						# avoid duplicates very close to existing points
						last = new_pts[-1]
						if abs(last[0] - proj[0]) < 1e-7 and abs(last[1] - proj[1]) < 1e-7:
							continue
						new_pts.append(proj)
						bin_proj_map[bid] = proj
		# ensure line has at least two points
		if len(new_pts) >= 2:
			new_lines.append(new_pts)

	return new_lines, bin_proj_map


def project_point_on_segment(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> tuple[tuple[float, float], float]:
	"""Project `point` (lat,lon) to segment a->b. Returns (proj_latlon, distance_m).

	Uses a local equirectangular projection for small distances.
	"""
	lat, lon = point
	lat1, lon1 = a
	lat2, lon2 = b
	mean_lat = math.radians((lat1 + lat2 + lat) / 3.0)
	cos_lat = math.cos(mean_lat)
	deg_to_m = 111000.0
	ax = (lon1 - lon) * deg_to_m * cos_lat
	ay = (lat1 - lat) * deg_to_m
	bx = (lon2 - lon) * deg_to_m * cos_lat
	by = (lat2 - lat) * deg_to_m
	px = 0.0
	py = 0.0
	vx = bx - ax
	vy = by - ay
	wvx = -ax
	wvy = -ay
	denom = vx * vx + vy * vy
	if denom == 0.0:
		t = 0.0
	else:
		t = (wvx * vx + wvy * vy) / denom
		t = max(0.0, min(1.0, t))
	proj_x = ax + t * vx
	proj_y = ay + t * vy
	# convert back to lat/lon
	proj_lon = lon + proj_x / (deg_to_m * cos_lat)
	proj_lat = lat + proj_y / deg_to_m
	distance = haversine_m((lat, lon), (proj_lat, proj_lon))
	return (proj_lat, proj_lon), distance


def point_to_segment_distance_m(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
	proj, dist = project_point_on_segment(point, a, b)
	return dist


def add_bins_to_graph(
	graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
	bins: list[dict[str, float | int]],
) -> dict[tuple[float, float], list[tuple[tuple[float, float], float]]]:
	"""Split nearest graph edges and insert nodes at bin projections.

	Mutates and returns the graph with new nodes added for each bin.
	"""
	# build list of edges
	edges = []
	for u, neighbors in graph.items():
		for v, w in neighbors:
			# only include each undirected edge once (u < v by tuple compare)
			if u < v:
				edges.append((u, v))

	for b in bins:
		point = (float(b["lat"]), float(b["lon"]))
		best = None
		best_dist = float("inf")
		best_proj = None
		best_edge = None
		for u, v in edges:
			proj, dist = project_point_on_segment(point, u, v)
			if dist < best_dist:
				best_dist = dist
				best = proj
				best_edge = (u, v)
		if best is None or best_edge is None:
			continue
		proj = best
		u, v = best_edge
		# if projection equals an existing node (within 1e-6 deg), reuse it
		def close(a, b):
			return abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6

		reuse_node = None
		for node in (u, v):
			if close(node, proj):
				reuse_node = node
				break
		if reuse_node is not None:
			b_node = reuse_node
		else:
			# create new node at proj and split edge u-v
			# remove v from u's neighbors and u from v's neighbors
			# note: use list copies to avoid mutating during iteration
			def remove_neighbor(a, bnode):
				lst = graph.get(a, [])
				graph[a] = [t for t in lst if not (abs(t[0][0] - bnode[0]) < 1e-9 and abs(t[0][1] - bnode[1]) < 1e-9)]

			remove_neighbor(u, v)
			remove_neighbor(v, u)
			dist_u_proj = haversine_m(u, proj)
			dist_proj_v = haversine_m(proj, v)
			# attach new proj node
			graph.setdefault(u, []).append((proj, dist_u_proj))
			graph.setdefault(v, []).append((proj, dist_proj_v))
			graph.setdefault(proj, []).append((u, dist_u_proj))
			graph.setdefault(proj, []).append((v, dist_proj_v))
			b_node = proj
			edges.append((u, proj))
			edges.append((proj, v))
		# annotate bin with node coordinate so callers can use it
		b["_graph_node"] = b_node

	return graph


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
	daily_peak = 0.0
	for bin_item in bins:
		bin_id = int(bin_item["binId"])
		daily_peak = max(daily_peak, daily_loads.get(bin_id, empty_series)[day_index_val])
	if daily_peak <= 0.0:
		daily_peak = 1.0
	scores = {}
	for bin_item in bins:
		bin_id = int(bin_item["binId"])
		current = daily_loads.get(bin_id, empty_series)[day_index_val]
		predicted = fullness_predictions.get(bin_id, 0.5)
		current_score = min(1.0, current / daily_peak)
		combined_fullness = min(1.0, current_score * 0.7 + predicted * 0.6)
		scores[bin_id] = combined_fullness
	
	return scores


def select_normal_route_bins(
	bins: list[dict[str, float | int]],
	daily_loads: dict[int, list[float]],
	day_index_val: int,
) -> list[dict[str, float | int]]:
	series_length = len(next(iter(daily_loads.values()), []))
	empty_series = [0.0] * series_length
	loads = sorted(
		[
			(
				daily_loads.get(int(bin_item["binId"]), empty_series)[day_index_val],
				bin_item,
			)
			for bin_item in bins
		],
		key=lambda item: (item[0], int(item[1]["binId"])),
		reverse=True,
	)
	if not loads:
		return []

	peak = loads[0][0]
	if peak <= 0.0:
		return [bin_item for _, bin_item in loads[: max(1, len(bins) // 10)]]

	cutoff = max(peak * 0.35, 0.05)
	selected = [bin_item for load, bin_item in loads if load >= cutoff]
	if not selected:
		selected = [bin_item for _, bin_item in loads[: max(1, len(bins) // 10)]]
	return selected


def sweep_order(
	bins: list[dict[str, float | int]],
	grid_meters: float = DEFAULT_SWEEP_GRID_METERS,
) -> list[dict[str, float | int]]:
	"""Return a boustrophedon ordering aligned with the city's principal axis.

	We project bin coordinates into a local meter-space, compute the principal
	axis using SVD, rotate points so the principal axis becomes the X axis,
	then perform a lawnmower sweep across Y bands. This produces long straight
	passes that follow the dominant city orientation instead of strictly
	north-south.
	"""
	if not bins:
		return []

	lat_arr = np.array([float(b["lat"]) for b in bins])
	lon_arr = np.array([float(b["lon"]) for b in bins])

	# convert degrees to meters locally using equirectangular approx
	mean_lat = float(lat_arr.mean())
	mean_lon = float(lon_arr.mean())
	deg_to_m = 111000.0
	cos_lat = math.cos(math.radians(mean_lat))
	xs = (lon_arr - mean_lon) * deg_to_m * cos_lat
	ys = (lat_arr - mean_lat) * deg_to_m

	points = np.column_stack((xs, ys))
	if points.shape[0] < 2 or np.allclose(points.std(axis=0), 0.0):
		# fallback to simple lon-based sweep
		return sorted(bins, key=lambda b: (float(b["lat"]), float(b["lon"])))

	# SVD to find principal directions
	_, _, vt = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
	rotation = vt.T  # columns are principal directions
	rotated = (points - points.mean(axis=0)).dot(rotation)

	min_y = rotated[:, 1].min()

	rows: dict[int, list[tuple[dict[str, float | int], float]]] = {}
	for idx, b in enumerate(bins):
		y = float(rotated[idx, 1])
		row = int((y - min_y) // grid_meters)
		rows.setdefault(row, []).append((b, float(rotated[idx, 0])))

	ordered: list[dict[str, float | int]] = []
	for i, row in enumerate(sorted(rows.keys())):
		row_bins = rows[row]
		reverse = (i % 2) == 1
		row_bins_sorted = sorted(row_bins, key=lambda t: t[1], reverse=reverse)
		for b, x in row_bins_sorted:
			# annotate bins with sweep metadata so callers can group by row
			b["_sweep_row"] = row
			b["_sweep_x"] = x
			ordered.append(b)

	return ordered


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
	bins_by_priority = sorted(bins, key=lambda b: (-priority_scores.get(int(b["binId"]), 0.0), int(b["binId"])))
	if not bins_by_priority:
		return []

	priority_mass = sum(priority_scores.get(int(bin_item["binId"]), 0.0) for bin_item in bins_by_priority)
	target_mass = max(min_fullness_threshold, priority_mass * 0.4)
	high_priority: list[dict[str, float | int]] = []
	covered_mass = 0.0
	for bin_item in bins_by_priority:
		score = priority_scores.get(int(bin_item["binId"]), 0.0)
		if score <= 0.0:
			continue
		high_priority.append(bin_item)
		covered_mass += score
		if covered_mass >= target_mass and len(high_priority) >= max(1, len(bins_by_priority) // 8):
			break

	if not high_priority:
		high_priority = bins_by_priority[:max(1, len(bins_by_priority) // 6)]
	
	nodes_list = list(graph)
	depot_node = nearest_node(depot, nodes_list)
	
	unvisited = {int(b["binId"]): b for b in high_priority}
	ordered = []
	current_node = depot_node
	
	while unvisited:
		nearest_bin_id = min(
			unvisited.keys(),
			key=lambda bid: street_distance(current_node, bin_nodes[bid], bin_nodes, graph)
			/ max(0.15, priority_scores.get(bid, 0.0)),
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


def route_path_distance_m(route_path: list[list[float]]) -> float:
	if len(route_path) < 2:
		return 0.0
	total = 0.0
	for start, end in zip(route_path, route_path[1:]):
		total += haversine_m((start[0], start[1]), (end[0], end[1]))
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
	# keep route strictly on-graph: do not append the raw depot coordinate

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

	for label in day_labels:
		index = day_index[label]
		if route_type == "normal":
			# produce a single continuous on-graph sweep path visiting rows
			sweep_bins = sweep_order(bins, grid_meters=DEFAULT_SWEEP_GRID_METERS)
			# group bins by sweep row in order
			rows: dict[int, list[dict[str, float | int]]] = {}
			for b in sweep_bins:
				rows.setdefault(int(b["_sweep_row"]), []).append(b)
			depot_node = nearest_node(depot, nodes)
			current_node = depot_node
			route_path = [[round(current_node[0], 6), round(current_node[1], 6)]]
			for row in sorted(rows.keys()):
				row_bins = rows[row]
				# ensure direction matches sweep ordering
				first_node = bin_nodes[int(row_bins[0]["binId"])]
				seg = dijkstra_paths(graph, current_node, first_node)
				for node in seg[1:]:
					route_path.append([round(node[0], 6), round(node[1], 6)])
				# traverse across the row from first to last bin
				if len(row_bins) > 1:
					last_node = bin_nodes[int(row_bins[-1]["binId"])]
					seg2 = dijkstra_paths(graph, first_node, last_node)
					for node in seg2[1:]:
						route_path.append([round(node[0], 6), round(node[1], 6)])
					current_node = last_node
				else:
					current_node = first_node
			# return to depot at end of sweep
			seg = dijkstra_paths(graph, current_node, nearest_node(depot, nodes))
			for node in seg[1:]:
				route_path.append([round(node[0], 6), round(node[1], 6)])
			# keep all bins in sweep order so the truck covers the whole city in one pass
			ordered_bins = sweep_bins
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
			route_path = build_route_path(depot, ordered_bins, bin_nodes, graph)
			distance_m = round(route_path_distance_m(route_path), 3)
			co2_kg = round((distance_m / 1000.0) * EMISSION_FACTOR_KG_PER_KM, 3)
		else:
			route_path = build_route_path(depot, ordered_bins, bin_nodes, graph)
			distance_m = round(route_distance_m(depot, ordered_bins, bin_nodes, graph), 3)
			co2_kg = round((distance_m / 1000.0) * EMISSION_FACTOR_KG_PER_KM, 3)

		# Ensure any bins that lie on the computed route path and have
		# non-zero collection for this day are included in the ordered list.
		# This covers bins that are located on intermediate nodes the truck
		# passes by but were excluded by the priority selection.
		if route_path and route_type != "normal":
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
	# try to insert projected bin points into street polylines so they become graph nodes
	updated_lines, bin_proj_map = insert_bins_into_lines(street_lines, bins)
	if updated_lines:
		street_lines = updated_lines
	graph, nodes = build_graph(street_lines)
	# map bins to their projected node when available, otherwise nearest node
	bin_nodes = {}
	for bin_item in bins:
		bid = int(bin_item["binId"])
		if bid in bin_proj_map:
			bin_nodes[bid] = tuple(bin_proj_map[bid])
		else:
			bin_nodes[bid] = nearest_node((float(bin_item["lat"]), float(bin_item["lon"])), nodes)

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
