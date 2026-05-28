from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from io import StringIO

from flask import Flask, Response, jsonify, send_from_directory

from bin_health import compute_bin_health


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def generated_dir() -> Path:
    return project_root() / "src" / "route-optimizer" / "generated"


def read_pois(path: Path) -> list[dict[str, object]]:
    pois: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                lon = float(row.get("x", row.get("lon", 0)))
                lat = float(row.get("y", row.get("lat", 0)))
            except (TypeError, ValueError):
                continue
            pois.append({"lat": lat, "lon": lon, "type": row.get("type", "other")})
    return pois


def read_bins(path: Path) -> list[dict[str, float | int]]:
    items: list[dict[str, float | int]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                bid = int(row.get("binId", row.get("id", 0)))
                lon = float(row.get("x", row.get("lon", 0)))
                lat = float(row.get("y", row.get("lat", 0)))
            except (TypeError, ValueError):
                continue
            items.append({"binId": bid, "lat": lat, "lon": lon})
    return items


def read_street_lines(path: Path) -> list[list[list[float]]]:
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
            return data
        except Exception:
            return []


def read_waste_events(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                timestamp = datetime.fromisoformat(row["timestamp"])
                bid = int(row["binId"])
                wtype = row.get("type", "mixed")
                weight = float(row.get("weight", 0))
            except Exception:
                continue
            events.append({"timestamp": timestamp, "date": timestamp.date().isoformat(), "binId": bid, "type": wtype, "weight": weight})
    return events


def load_visualization_data() -> dict[str, object]:
    gen = generated_dir()
    required = [gen / "pois.csv", gen / "bins.csv", gen / "street_lines.csv", gen / "waste_events.csv"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        missing_lines = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError("Missing required CSV files. Run generators first:\n" + missing_lines)

    pois = read_pois(gen / "pois.csv")
    bins = read_bins(gen / "bins.csv")
    street_lines = read_street_lines(gen / "street_lines.csv")
    waste_events = read_waste_events(gen / "waste_events.csv")

    day_labels = sorted({str(e["date"]) for e in waste_events})
    bin_health = compute_bin_health(bins, waste_events, day_labels)

    return {
        "meta": {"pois": len(pois), "bins": len(bins), "street_lines": len(street_lines), "waste_events": len(waste_events), "days": len(day_labels)},
        "day_labels": day_labels,
        "bin_health": bin_health,
        "pois": pois,
        "bins": bins,
        "street_lines": street_lines,
    }


app = Flask(__name__)


@app.get("/")
def index() -> Response:
    static_dir = project_root() / "src" / "route-optimizer" / "static"
    return send_from_directory(str(static_dir), "bin_health.html")


@app.get("/api/status")
def api_status() -> Response:
    try:
        payload = load_visualization_data()
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    return jsonify({"ok": True, **payload})


@app.get("/api/bin_health.csv")
def api_bin_health_csv() -> Response:
    try:
        payload = load_visualization_data()
    except FileNotFoundError as exc:
        return Response(str(exc), status=500)

    bin_health = payload.get("bin_health", {})
    sio = StringIO()
    writer = csv.writer(sio)
    writer.writerow([
        "binId",
        "urgency",
        "fill_ratio",
        "capacity_kg",
        "hours_until_overflow",
        "expected_collected_kg",
        "current_load_kg",
        "days_since_last_collection",
    ])
    for bid in sorted(bin_health.keys()):
        h = bin_health[bid]
        writer.writerow([
            bid,
            h.get("urgency"),
            h.get("fill_ratio"),
            h.get("capacity_kg"),
            h.get("hours_until_overflow"),
            h.get("expected_collected_kg"),
            h.get("current_load_kg"),
            h.get("days_since_last_collection"),
        ])

    return Response(sio.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=bin_health.csv"})


def main() -> None:
    print("Route optimizer visualization server")
    print("Ensure generated CSVs exist under src/route-optimizer/generated/")
    print("Open: http://127.0.0.1:5002")
    app.run(host="127.0.0.1", port=5002, debug=False)


if __name__ == "__main__":
    main()
