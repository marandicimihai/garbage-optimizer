from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_OUTPUT = "generated/waste_events.csv"
DEFAULT_SEED = 42
DAY_COUNT = 7
DAY_START = datetime(2026, 5, 3, tzinfo=timezone.utc)

WASTE_TYPES = ("plastic", "paper", "metal", "glass")

POI_WASTE_MULTIPLIERS = {
	"mall": 12.0,
	"supermarket": 6.0,
	"restaurant": 4.5,
	"fast_food": 4.0,
	"convenience": 2.5,
	"cafe": 1.4,
	"other": 1.0,
}

POI_WASTE_MIX = {
	"mall": (0.42, 0.26, 0.12, 0.20),
	"supermarket": (0.40, 0.30, 0.10, 0.20),
	"restaurant": (0.30, 0.42, 0.08, 0.20),
	"fast_food": (0.34, 0.38, 0.08, 0.20),
	"convenience": (0.36, 0.28, 0.10, 0.26),
	"cafe": (0.28, 0.40, 0.08, 0.24),
	"other": (0.34, 0.30, 0.10, 0.26),
}


@dataclass(frozen=True)
class Poi:
	type: str
	lat: float
	lon: float


@dataclass(frozen=True)
class Bin:
	bin_id: int
	lat: float
	lon: float


@dataclass(frozen=True)
class Event:
	timestamp: datetime
	bin_id: int
	waste_type: str
	weight: float


def project_root() -> Path:
	return Path(__file__).resolve().parents[2]


def pois_path() -> Path:
	return project_root() / "generated" / "pois.csv"


def bins_path() -> Path:
	return project_root() / "generated" / "bins.csv"


def output_path() -> Path:
	return project_root() / DEFAULT_OUTPUT


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
	lat1, lon1 = a
	lat2, lon2 = b
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlambda = math.radians(lon2 - lon1)
	h = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
	return 2.0 * 6_371_000.0 * math.asin(math.sqrt(h))


def load_pois(path: Path) -> list[Poi]:
	pois: list[Poi] = []
	with path.open("r", encoding="utf-8", newline="") as file_handle:
		reader = csv.DictReader(file_handle)
		for row in reader:
			try:
				poi_type = str(row.get("type", "other"))
				lat = float(row["y"])
				lon = float(row["x"])
			except (KeyError, TypeError, ValueError):
				continue
			pois.append(Poi(type=poi_type, lat=lat, lon=lon))
	return pois


def load_bins(path: Path) -> list[Bin]:
	bins: list[Bin] = []
	with path.open("r", encoding="utf-8", newline="") as file_handle:
		reader = csv.DictReader(file_handle)
		for row in reader:
			try:
				bin_id = int(row["binId"])
				lon = float(row["x"])
				lat = float(row["y"])
			except (KeyError, TypeError, ValueError):
				continue
			bins.append(Bin(bin_id=bin_id, lat=lat, lon=lon))
	return bins


def normalize(values: list[float]) -> list[float]:
	cleaned = [max(0.0, value) for value in values]
	total = sum(cleaned)
	if total <= 0.0:
		return [1.0 / len(cleaned) for _ in cleaned]
	return [value / total for value in cleaned]


def type_mixture(poi_type: str, rng: random.Random) -> list[float]:
	base = POI_WASTE_MIX.get(poi_type, POI_WASTE_MIX["other"])
	noisy = [weight * rng.uniform(0.8, 1.2) for weight in base]
	return normalize(noisy)


def daily_amount(poi_type: str, rng: random.Random) -> float:
	multiplier = POI_WASTE_MULTIPLIERS.get(poi_type, POI_WASTE_MULTIPLIERS["other"])
	base_amount = rng.gammavariate(3.2, 0.5)
	return round(base_amount * multiplier, 3)


def nearby_bins(poi: Poi, bins: list[Bin], limit: int = 6) -> list[tuple[Bin, float]]:
	ordered = sorted(
		((bin_item, haversine_m((poi.lat, poi.lon), (bin_item.lat, bin_item.lon))) for bin_item in bins),
		key=lambda item: item[1],
	)
	return ordered[: max(1, min(limit, len(ordered)))]


def bell_curve_shares(poi: Poi, bins: list[Bin], rng: random.Random) -> list[tuple[Bin, float]]:
	nearby = nearby_bins(poi, bins)
	if len(nearby) == 1:
		return [(nearby[0][0], 1.0)]

	distances = [distance for _, distance in nearby]
	center = sum(distances) / len(distances)
	sigma = max(90.0, center / 1.7)
	weights = []
	for _, distance in nearby:
		bell = math.exp(-(distance ** 2) / (2.0 * sigma ** 2))
		weights.append(bell * rng.uniform(0.88, 1.12))

	shares = normalize(weights)
	return [(bin_item, share) for (bin_item, _), share in zip(nearby, shares)]


def event_count_for_budget(budget: float, rng: random.Random) -> int:
	count = int(round(max(1.0, budget * rng.uniform(2.0, 3.5))))
	return max(1, count)


def random_timestamp(day_start: datetime, rng: random.Random) -> datetime:
	seconds = rng.randrange(24 * 60 * 60)
	return day_start + timedelta(seconds=seconds)


def split_budget(total: float, parts: int, rng: random.Random) -> list[float]:
	raw = [rng.gammavariate(1.5, 1.0) for _ in range(parts)]
	shares = normalize(raw)
	return [round(total * share, 3) for share in shares]


def choose_bin(bin_shares: list[tuple[Bin, float]], rng: random.Random) -> Bin:
	roll = rng.random()
	cumulative = 0.0
	for candidate_bin, share in bin_shares:
		cumulative += share
		if roll <= cumulative:
			return candidate_bin
	return bin_shares[-1][0]


def generate_events(pois: list[Poi], bins: list[Bin], seed: int = DEFAULT_SEED) -> list[Event]:
	rng = random.Random(seed)
	events: list[Event] = []

	for day_offset in range(DAY_COUNT):
		day_start = DAY_START + timedelta(days=day_offset)
		for poi in pois:
			total_daily = daily_amount(poi.type, rng) * rng.uniform(0.88, 1.16)
			type_ratios = type_mixture(poi.type, rng)
			bin_shares = bell_curve_shares(poi, bins, rng)

			for waste_type, ratio in zip(WASTE_TYPES, type_ratios):
				budget = round(total_daily * ratio, 3)
				if budget <= 0.0:
					continue
				count = event_count_for_budget(budget, rng)
				weights = split_budget(budget, count, rng)
				for event_weight in weights:
					events.append(
						Event(
							timestamp=random_timestamp(day_start, rng),
							bin_id=choose_bin(bin_shares, rng).bin_id,
							waste_type=waste_type,
							weight=event_weight,
						)
					)

	return sorted(events, key=lambda event: (event.timestamp, event.bin_id, event.waste_type))


def write_events(path: Path, events: list[Event]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8", newline="") as file_handle:
		writer = csv.writer(file_handle)
		writer.writerow(["timestamp", "binId", "type", "weight"])
		for event in events:
			writer.writerow([
				event.timestamp.isoformat(),
				event.bin_id,
				event.waste_type,
				event.weight,
			])


def main() -> None:
	pois_file = pois_path()
	bins_file = bins_path()
	output_file = output_path()

	if not pois_file.exists():
		raise SystemExit(f"Missing POI input: {pois_file}")
	if not bins_file.exists():
		raise SystemExit(f"Missing bin input: {bins_file}")

	pois = load_pois(pois_file)
	bins = load_bins(bins_file)
	if not pois:
		raise SystemExit(f"No POIs loaded from {pois_file}")
	if not bins:
		raise SystemExit(f"No bins loaded from {bins_file}")

	events = generate_events(pois, bins)
	write_events(output_file, events)
	print(f"Saved {len(events)} waste events to {output_file}")


if __name__ == "__main__":
	main()
