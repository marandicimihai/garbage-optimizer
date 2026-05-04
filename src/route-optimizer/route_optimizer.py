"""ML-based garbage collection route optimizer.

Compares two collection strategies every day:

  1. Regular route  – visits ALL bins in a fixed sequential order (sorted by
     bin ID), regardless of fill level.  This mimics a rigid schedule that
     many municipalities follow today.

  2. ML-Optimised route – a two-stage algorithm:
       Stage 1 (ML filter): a rolling-average fill-rate model predicts each
         bin's fill percentage for the day and marks only those bins that meet
         or exceed the FILL_THRESHOLD as needing collection.
       Stage 2 (TSP): a greedy nearest-neighbour heuristic orders the
         selected bins to minimise total travel distance, starting and ending
         at a central depot.

Fuel savings are estimated with a standard truck consumption rate.

Outputs (written to <repo_root>/generated/):
  route_comparison.csv   per-day tabular metrics
  routes.json            route polylines + stats for the HTML visualisation
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ─────────────────────── tuneable constants ───────────────────────────────────

BIN_CAPACITY_KG: float = 50.0
"""Assumed maximum capacity of one bin (kg)."""

FILL_THRESHOLD: float = 0.40
"""Collect a bin only when its predicted fill level is ≥ this fraction."""

FUEL_L_PER_KM: float = 0.30
"""Average garbage-truck fuel consumption (litres / km)."""


# ─────────────────────── data classes ────────────────────────────────────────


@dataclass(frozen=True)
class Bin:
    bin_id: int
    lat: float
    lon: float


@dataclass
class DayStats:
    date: str
    regular_bins: int
    regular_distance_km: float
    regular_fuel_l: float
    optimised_bins: int
    optimised_distance_km: float
    optimised_fuel_l: float
    distance_saved_km: float
    fuel_saved_l: float
    savings_pct: float
    regular_route_coords: list[list[float]] = field(default_factory=list)
    optimised_route_coords: list[list[float]] = field(default_factory=list)


# ─────────────────────── path helpers ────────────────────────────────────────


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def generated_dir() -> Path:
    return project_root() / "generated"


# ─────────────────────── geometry ────────────────────────────────────────────


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points."""
    lat1, lon1 = a
    lat2, lon2 = b
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2.0 * 6_371.0 * math.asin(math.sqrt(h))


# ─────────────────────── I/O helpers ─────────────────────────────────────────


def load_bins(path: Path) -> list[Bin]:
    bins: list[Bin] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                bins.append(
                    Bin(
                        bin_id=int(row["binId"]),
                        lon=float(row["x"]),
                        lat=float(row["y"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    return bins


def load_daily_weights(
    events_path: Path,
    bin_ids: set[int],
) -> dict[str, dict[int, float]]:
    """Return {date_str: {bin_id: total_weight_kg_that_day}}."""
    daily: dict[str, dict[int, float]] = {}
    with events_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ts = datetime.fromisoformat(str(row["timestamp"]))
                day = ts.date().isoformat()
                bin_id = int(row["binId"])
                weight = float(row["weight"])
            except (KeyError, ValueError):
                continue
            if bin_id not in bin_ids:
                continue
            daily.setdefault(day, {}).setdefault(bin_id, 0.0)
            daily[day][bin_id] += weight
    return daily


# ─────────────────────── ML fill predictor ───────────────────────────────────


def build_fill_model(
    daily_weights: dict[str, dict[int, float]],
    bin_ids: set[int],
    capacity: float = BIN_CAPACITY_KG,
) -> dict[str, dict[int, float]]:
    """
    Stage-1 (ML filter): rolling-average fill-rate model.

    For each bin and each day, we estimate the cumulative fill level by
    tracking the running total weight and modelling collection events.
    A bin that is collected on day D resets to zero for day D+1.

    The "ML" component is the per-bin rolling average daily addition rate,
    which is used to predict the next day's fill even before the waste-event
    data for that day arrives.  Here we use 3-day exponential moving average
    as a lightweight learnable parameter.

    Returns {date_str: {bin_id: fill_fraction_0_to_1}}.
    """
    sorted_days = sorted(daily_weights)
    # cumulative fill for each bin (reset when collected)
    current_fill: dict[int, float] = {b: 0.0 for b in bin_ids}
    # EMA of daily addition per bin – the "learned" parameter
    ema_rate: dict[int, float] = {b: 0.0 for b in bin_ids}
    alpha = 0.4  # EMA smoothing factor

    result: dict[str, dict[int, float]] = {}

    for day in sorted_days:
        day_additions = daily_weights.get(day, {})

        # Determine which bins should be collected today (threshold crossing)
        to_collect: set[int] = set()
        for bid in bin_ids:
            predicted = current_fill[bid] + ema_rate[bid]
            if predicted / capacity >= FILL_THRESHOLD:
                to_collect.add(bid)

        # Update actual fill levels with today's real waste data
        for bid in bin_ids:
            added = day_additions.get(bid, 0.0)
            # Update the EMA of daily addition
            ema_rate[bid] = alpha * added + (1 - alpha) * ema_rate[bid]
            if bid in to_collect:
                # Bin was collected; reset fill then add today's waste
                current_fill[bid] = added
            else:
                current_fill[bid] += added

        result[day] = {bid: current_fill[bid] for bid in bin_ids}

    return result


# ─────────────────────── routing algorithms ──────────────────────────────────


def route_distance_km(
    depot: tuple[float, float],
    ordered_bins: list[Bin],
) -> float:
    """Total round-trip distance: depot → bins → depot."""
    if not ordered_bins:
        return 0.0
    total = haversine_km(depot, (ordered_bins[0].lat, ordered_bins[0].lon))
    for i in range(len(ordered_bins) - 1):
        a = (ordered_bins[i].lat, ordered_bins[i].lon)
        b = (ordered_bins[i + 1].lat, ordered_bins[i + 1].lon)
        total += haversine_km(a, b)
    total += haversine_km((ordered_bins[-1].lat, ordered_bins[-1].lon), depot)
    return total


def regular_route(bins: list[Bin]) -> list[Bin]:
    """Visit every bin in a fixed sequential order (sorted by bin_id)."""
    return sorted(bins, key=lambda b: b.bin_id)


def nearest_neighbour_tsp(
    depot: tuple[float, float],
    bins_to_visit: list[Bin],
) -> list[Bin]:
    """
    Stage-2 (TSP): greedy nearest-neighbour tour starting from depot.

    At each step the algorithm picks the unvisited bin closest to the current
    position, which is a classic O(n²) heuristic that typically produces
    routes within 20–25 % of the global optimum.
    """
    if not bins_to_visit:
        return []
    unvisited = list(bins_to_visit)
    route: list[Bin] = []
    current: tuple[float, float] = depot
    while unvisited:
        nearest = min(
            unvisited,
            key=lambda b: haversine_km(current, (b.lat, b.lon)),
        )
        route.append(nearest)
        unvisited.remove(nearest)
        current = (nearest.lat, nearest.lon)
    return route


def optimised_route(
    bins: list[Bin],
    fill_fractions: dict[int, float],
    depot: tuple[float, float],
    threshold: float = FILL_THRESHOLD,
    capacity: float = BIN_CAPACITY_KG,
) -> list[Bin]:
    """Select bins above the fill threshold and order them with NN-TSP."""
    needed = [
        b for b in bins
        if fill_fractions.get(b.bin_id, 0.0) / capacity >= threshold
    ]
    return nearest_neighbour_tsp(depot, needed)


# ─────────────────────── per-day statistics ──────────────────────────────────


def compute_day_stats(
    day: str,
    bins: list[Bin],
    fill_fractions: dict[int, float],
    depot: tuple[float, float],
) -> DayStats:
    reg = regular_route(bins)
    opt = optimised_route(bins, fill_fractions, depot)

    reg_dist = route_distance_km(depot, reg)
    opt_dist = route_distance_km(depot, opt)
    reg_fuel = reg_dist * FUEL_L_PER_KM
    opt_fuel = opt_dist * FUEL_L_PER_KM
    saved_dist = reg_dist - opt_dist
    saved_fuel = reg_fuel - opt_fuel
    pct = 100.0 * saved_dist / reg_dist if reg_dist > 0 else 0.0

    def to_coords(route_bins: list[Bin]) -> list[list[float]]:
        coords = [[depot[0], depot[1]]]
        coords.extend([[b.lat, b.lon] for b in route_bins])
        coords.append([depot[0], depot[1]])
        return coords

    return DayStats(
        date=day,
        regular_bins=len(reg),
        regular_distance_km=round(reg_dist, 3),
        regular_fuel_l=round(reg_fuel, 3),
        optimised_bins=len(opt),
        optimised_distance_km=round(opt_dist, 3),
        optimised_fuel_l=round(opt_fuel, 3),
        distance_saved_km=round(saved_dist, 3),
        fuel_saved_l=round(saved_fuel, 3),
        savings_pct=round(pct, 2),
        regular_route_coords=to_coords(reg),
        optimised_route_coords=to_coords(opt),
    )


# ─────────────────────── output writers ──────────────────────────────────────


def write_comparison_csv(path: Path, stats: list[DayStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "date",
                "regular_bins",
                "regular_distance_km",
                "regular_fuel_l",
                "optimised_bins",
                "optimised_distance_km",
                "optimised_fuel_l",
                "distance_saved_km",
                "fuel_saved_l",
                "savings_pct",
            ]
        )
        for s in stats:
            writer.writerow(
                [
                    s.date,
                    s.regular_bins,
                    s.regular_distance_km,
                    s.regular_fuel_l,
                    s.optimised_bins,
                    s.optimised_distance_km,
                    s.optimised_fuel_l,
                    s.distance_saved_km,
                    s.fuel_saved_l,
                    s.savings_pct,
                ]
            )


def write_routes_json(path: Path, stats: list[DayStats]) -> None:
    records = []
    for s in stats:
        records.append(
            {
                "date": s.date,
                "regular": {
                    "coords": s.regular_route_coords,
                    "distance_km": s.regular_distance_km,
                    "fuel_l": s.regular_fuel_l,
                    "bins": s.regular_bins,
                },
                "optimised": {
                    "coords": s.optimised_route_coords,
                    "distance_km": s.optimised_distance_km,
                    "fuel_l": s.optimised_fuel_l,
                    "bins": s.optimised_bins,
                },
                "saved_km": s.distance_saved_km,
                "saved_l": s.fuel_saved_l,
                "savings_pct": s.savings_pct,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=True), encoding="utf-8")


# ─────────────────────── entry point ─────────────────────────────────────────


def main() -> None:
    gen = generated_dir()
    bins_csv = gen / "bins.csv"
    events_csv = gen / "waste_events.csv"
    comparison_csv = gen / "route_comparison.csv"
    routes_json_path = gen / "routes.json"

    if not bins_csv.exists():
        raise SystemExit(f"Missing required file: {bins_csv}")
    if not events_csv.exists():
        raise SystemExit(f"Missing required file: {events_csv}")

    bins = load_bins(bins_csv)
    if not bins:
        raise SystemExit("No bins loaded — check bins.csv")

    # Depot = centroid of all bins (represents the municipal facility)
    depot: tuple[float, float] = (
        sum(b.lat for b in bins) / len(bins),
        sum(b.lon for b in bins) / len(bins),
    )

    bin_ids = {b.bin_id for b in bins}
    daily_weights = load_daily_weights(events_csv, bin_ids)
    if not daily_weights:
        raise SystemExit("No waste events found — check waste_events.csv")

    fill_model = build_fill_model(daily_weights, bin_ids)
    sorted_days = sorted(fill_model)

    stats: list[DayStats] = []
    for day in sorted_days:
        s = compute_day_stats(day, bins, fill_model[day], depot)
        stats.append(s)
        print(
            f"  {day}: regular {s.regular_distance_km:7.2f} km | "
            f"optimised {s.optimised_distance_km:7.2f} km | "
            f"saved {s.distance_saved_km:6.2f} km ({s.savings_pct:.1f}%)"
        )

    write_comparison_csv(comparison_csv, stats)
    write_routes_json(routes_json_path, stats)

    total_saved_km = sum(s.distance_saved_km for s in stats)
    total_saved_l = sum(s.fuel_saved_l for s in stats)
    avg_pct = sum(s.savings_pct for s in stats) / len(stats) if stats else 0.0

    print(f"\nTotal distance saved : {total_saved_km:.2f} km over {len(stats)} days")
    print(f"Total fuel saved     : {total_saved_l:.2f} L")
    print(f"Average saving       : {avg_pct:.1f} %")
    print(f"Route comparison CSV → {comparison_csv}")
    print(f"Routes JSON          → {routes_json_path}")


if __name__ == "__main__":
    main()
