from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from io import StringIO

from flask import Flask, Response, jsonify, send_from_directory

from bin_health import build_city_context, compute_bin_health, predict_next_day_weight
from global_city_model import training_city_count, training_row_count
import forecast_evaluation as forecast_evaluation


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
    city_context = build_city_context(bins, waste_events, day_labels, pois=pois, street_lines=street_lines)
    bin_health = compute_bin_health(bins, waste_events, day_labels, pois=pois, street_lines=street_lines)
    model_info = {
        "model_name": "Global linear regression",
        "target_city": "Chisinau, Moldova",
        "training_rows": training_row_count(),
        "training_cities": training_city_count(),
    }

    return {
        "meta": {"pois": len(pois), "bins": len(bins), "street_lines": len(street_lines), "waste_events": len(waste_events), "days": len(day_labels)},
        "day_labels": day_labels,
        "city_context": city_context,
        "model_info": model_info,
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


@app.get("/api/eval")
def api_eval() -> Response:
    """Run rolling backtest evaluation over available bins.

    Returns aggregated metrics (MAE/RMSE/MAPE), overflow precision/recall,
    calibration_error, and per-bin summaries where available.
    """
    try:
        payload = load_visualization_data()
    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    bin_health = payload.get("bin_health", {})
    city_context = payload.get("city_context", {})
    # build maps for daily_series and capacity
    daily_series_map: dict[int, list[float]] = {}
    capacity_map: dict[int, float] = {}
    for bid_str, entry in bin_health.items():
        try:
            bid = int(bid_str)
        except Exception:
            bid = int(entry.get("binId", 0))
        daily_series_map[bid] = [float(v) for v in entry.get("daily_series", [])]
        capacity_map[bid] = float(entry.get("capacity_kg", 0.0))

    # run rolling backtest across bins using the same forecast function
    summary = forecast_evaluation.rolling_backtest_all_bins(
        daily_series_map,
        capacity_map,
        lambda hist: predict_next_day_weight(hist, city_context=city_context),
        window=7,
    )

    # normalize types to JSON-friendly primitives and replace NaN with null
    def _normalize(obj):
        if obj is None:
            return None
        if isinstance(obj, (int, float)):
            try:
                if isinstance(obj, float) and (obj != obj):
                    return None
            except Exception:
                pass
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): _normalize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_normalize(v) for v in obj]
        return obj

    safe_summary = _normalize(summary)
    # also include an inline per-bin calculation (explicit calls) for debugging
    inline_per_bin = {}
    for bid, series in daily_series_map.items():
        res = forecast_evaluation.rolling_backtest_for_series(series, capacity_map.get(bid, 0.0), lambda hist: predict_next_day_weight(hist, city_context=city_context), window=7)
        inline_per_bin[str(bid)] = _normalize(res)

    return jsonify({"ok": True, "evaluation": safe_summary, "evaluation_inline": {"per_bin": inline_per_bin}})


def main() -> None:
    print("Route optimizer visualization server")
    print("Ensure generated CSVs exist under src/route-optimizer/generated/")
    print("Open: http://127.0.0.1:5002")
    app.run(host="127.0.0.1", port=5002, debug=False)


if __name__ == "__main__":
    main()
