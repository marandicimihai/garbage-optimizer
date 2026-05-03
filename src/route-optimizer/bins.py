from __future__ import annotations

import csv
import heapq
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SPACING_METERS = 60.0


@dataclass(frozen=True)
class Edge:
	edge_id: int
	start: tuple[float, float]
	end: tuple[float, float]
	length_m: float


@dataclass(frozen=True)
class CandidateBin:
	edge_id: int
	pos_m: float
	lat: float
	lon: float


def project_root() -> Path:
	return Path(__file__).resolve().parents[2]


def street_lines_path() -> Path:
	return project_root() / "generated" / "street_lines.csv"


def bins_path() -> Path:
	return project_root() / "generated" / "bins.csv"


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
	lat1, lon1 = a
	lat2, lon2 = b
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlambda = math.radians(lon2 - lon1)
	h = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
	return 2.0 * 6_371_000.0 * math.asin(math.sqrt(h))


def interpolate(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
	lat = a[0] + (b[0] - a[0]) * t
	lon = a[1] + (b[1] - a[1]) * t
	return lat, lon


def read_street_lines(path: Path) -> list[list[tuple[float, float]]]:
	grouped: dict[int, list[tuple[int, float, float]]] = defaultdict(list)

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
			grouped[line_id].append((point_order, lat, lon))

	lines: list[list[tuple[float, float]]] = []
	for line_id in sorted(grouped):
		ordered = sorted(grouped[line_id], key=lambda item: item[0])
		coords = [(lat, lon) for _, lat, lon in ordered]
		if len(coords) >= 2:
			lines.append(coords)
	return lines


def build_graph(lines: list[list[tuple[float, float]]]) -> tuple[dict[tuple[float, float], list[int]], dict[int, Edge], dict[tuple[float, float], int], list[tuple[float, float]]]:
	adjacency: dict[tuple[float, float], list[int]] = defaultdict(list)
	edges: dict[int, Edge] = {}
	indegree: dict[tuple[float, float], int] = defaultdict(int)
	nodes: set[tuple[float, float]] = set()
	seen_undirected_segments: set[tuple[tuple[float, float], tuple[float, float]]] = set()

	edge_id = 0
	for line in lines:
		for i in range(len(line) - 1):
			start = line[i]
			end = line[i + 1]
			segment_key = (start, end) if start <= end else (end, start)
			if segment_key in seen_undirected_segments:
				continue
			seen_undirected_segments.add(segment_key)
			length_m = haversine_m(start, end)
			if length_m <= 0.0:
				continue
			edge = Edge(edge_id=edge_id, start=start, end=end, length_m=length_m)
			edges[edge_id] = edge
			adjacency[start].append(edge_id)
			indegree[end] += 1
			nodes.add(start)
			nodes.add(end)
			edge_id += 1

	for node in nodes:
		adjacency.setdefault(node, [])
		indegree.setdefault(node, 0)

	ordered_nodes = sorted(nodes)
	return adjacency, edges, indegree, ordered_nodes


def place_bins_on_edge(
	edge: Edge,
	carry_m: float,
	spacing_m: float,
) -> tuple[list[CandidateBin], float]:
	bins: list[CandidateBin] = []
	remaining = edge.length_m
	traveled = 0.0
	need = spacing_m - carry_m if carry_m > 0.0 else spacing_m

	while remaining + 1e-9 >= need:
		traveled += need
		t = traveled / edge.length_m
		lat, lon = interpolate(edge.start, edge.end, t)
		bins.append(CandidateBin(edge_id=edge.edge_id, pos_m=traveled, lat=lat, lon=lon))
		remaining -= need
		need = spacing_m

	carry_out = carry_m + edge.length_m
	if carry_out >= spacing_m:
		carry_out = carry_out % spacing_m
	return bins, carry_out


def dfs_place(
	node: tuple[float, float],
	carry_in: float,
	spacing_m: float,
	adjacency: dict[tuple[float, float], list[int]],
	edges: dict[int, Edge],
	visited_edges: set[int],
	output_bins: list[CandidateBin],
) -> None:
	propagated_branch_used = False

	for edge_id in adjacency[node]:
		if edge_id in visited_edges:
			continue
		visited_edges.add(edge_id)

		edge = edges[edge_id]
		edge_carry_in = carry_in if not propagated_branch_used else 0.0
		bins, carry_out = place_bins_on_edge(edge, edge_carry_in, spacing_m)
		output_bins.extend(bins)
		propagated_branch_used = True

		dfs_place(
			edge.end,
			carry_out,
			spacing_m,
			adjacency,
			edges,
			visited_edges,
			output_bins,
		)


def build_undirected_node_graph(edges: dict[int, Edge]) -> dict[tuple[float, float], list[tuple[tuple[float, float], float]]]:
	node_graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]] = defaultdict(list)
	for edge in edges.values():
		node_graph[edge.start].append((edge.end, edge.length_m))
		node_graph[edge.end].append((edge.start, edge.length_m))
	return node_graph


def dijkstra_with_cutoff(
	node_graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
	start: tuple[float, float],
	cutoff: float,
) -> dict[tuple[float, float], float]:
	dist: dict[tuple[float, float], float] = {start: 0.0}
	heap: list[tuple[float, tuple[float, float]]] = [(0.0, start)]

	while heap:
		current_dist, node = heapq.heappop(heap)
		if current_dist > cutoff:
			continue
		if current_dist > dist.get(node, float("inf")):
			continue
		for neighbor, weight in node_graph.get(node, []):
			candidate = current_dist + weight
			if candidate > cutoff:
				continue
			if candidate + 1e-9 < dist.get(neighbor, float("inf")):
				dist[neighbor] = candidate
				heapq.heappush(heap, (candidate, neighbor))

	return dist


def road_distance_between_candidates(
	a: CandidateBin,
	b: CandidateBin,
	edges: dict[int, Edge],
	node_graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
	cutoff: float,
) -> float:
	edge_a = edges[a.edge_id]
	edge_b = edges[b.edge_id]

	if a.edge_id == b.edge_id:
		same_edge_distance = abs(a.pos_m - b.pos_m)
		if same_edge_distance <= cutoff:
			return same_edge_distance

	endpoints_a = [
		(edge_a.start, a.pos_m),
		(edge_a.end, edge_a.length_m - a.pos_m),
	]
	endpoints_b = [
		(edge_b.start, b.pos_m),
		(edge_b.end, edge_b.length_m - b.pos_m),
	]

	best = float("inf")
	for node_a, tail_a in endpoints_a:
		remaining_cutoff = cutoff - tail_a
		if remaining_cutoff <= 0.0:
			continue
		dist_map = dijkstra_with_cutoff(node_graph, node_a, remaining_cutoff)
		for node_b, tail_b in endpoints_b:
			node_dist = dist_map.get(node_b)
			if node_dist is None:
				continue
			total = tail_a + node_dist + tail_b
			if total < best:
				best = total

	return best


def filter_bins_by_road_distance(
	candidates: list[CandidateBin],
	spacing_m: float,
	edges: dict[int, Edge],
	node_graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
) -> list[CandidateBin]:
	accepted: list[CandidateBin] = []
	for candidate in candidates:
		is_valid = True
		for existing in accepted:
			distance_m = road_distance_between_candidates(candidate, existing, edges, node_graph, spacing_m)
			if distance_m + 1e-9 < spacing_m:
				is_valid = False
				break
		if is_valid:
			accepted.append(candidate)
	return accepted


def place_bins(lines: list[list[tuple[float, float]]], spacing_m: float) -> list[tuple[int, float, float]]:
	adjacency, edges, indegree, nodes = build_graph(lines)
	visited_edges: set[int] = set()
	candidates: list[CandidateBin] = []

	roots = [node for node in nodes if indegree[node] == 0]
	for root in roots:
		dfs_place(root, 0.0, spacing_m, adjacency, edges, visited_edges, candidates)

	for node in nodes:
		dfs_place(node, 0.0, spacing_m, adjacency, edges, visited_edges, candidates)

	node_graph = build_undirected_node_graph(edges)
	filtered = filter_bins_by_road_distance(candidates, spacing_m, edges, node_graph)
	bins: list[tuple[int, float, float]] = []
	for index, candidate in enumerate(filtered, start=1):
		bins.append((index, candidate.lon, candidate.lat))

	return bins


def write_bins(path: Path, bins: list[tuple[int, float, float]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as file_handle:
		writer = csv.writer(file_handle)
		writer.writerow(["binId", "x", "y"])
		writer.writerows(bins)


def main() -> None:
	sys.setrecursionlimit(100_000)

	lines = read_street_lines(street_lines_path())
	bins = place_bins(lines, DEFAULT_SPACING_METERS)
	write_bins(bins_path(), bins)


if __name__ == "__main__":
	main()
