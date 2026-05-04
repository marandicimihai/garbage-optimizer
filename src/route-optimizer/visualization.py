from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Resolve repository root from this script location."""
    return Path(__file__).resolve().parents[2]


def generated_dir() -> Path:
    return project_root() / "generated"


def read_pois(path: Path) -> list[dict[str, object]]:
    pois: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            try:
                lon = float(row["x"])
                lat = float(row["y"])
            except (KeyError, TypeError, ValueError):
                continue
            poi_type = str(row.get("type", "other"))
            pois.append({"lat": lat, "lon": lon, "type": poi_type})
    return pois


def read_street_lines(path: Path) -> list[list[list[float]]]:
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

    polylines: list[list[list[float]]] = []
    for line_id in sorted(lines_by_id):
        ordered = sorted(lines_by_id[line_id], key=lambda item: item[0])
        coords = [[lat, lon] for _, lat, lon in ordered]
        if len(coords) >= 2:
            polylines.append(coords)
    return polylines


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
                timestamp = datetime.fromisoformat(str(row["timestamp"]))
                bin_id = int(row["binId"])
                waste_type = str(row["type"])
                weight = float(row["weight"])
            except (KeyError, TypeError, ValueError):
                continue
            events.append(
                {
                    "timestamp": timestamp,
                    "date": timestamp.date().isoformat(),
                    "binId": bin_id,
                    "type": waste_type,
                    "weight": weight,
                }
            )
    return events


def read_routes_data(path: Path) -> list[dict[str, Any]]:
    """Load routes.json produced by route_optimizer.py.  Returns [] if absent."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def compute_center(
    pois: list[dict[str, object]],
    bins: list[dict[str, float | int]],
    street_lines: list[list[list[float]]],
) -> tuple[float, float]:
    lat_values: list[float] = []
    lon_values: list[float] = []

    for poi in pois:
        lat_values.append(float(poi["lat"]))
        lon_values.append(float(poi["lon"]))

    for bin_item in bins:
        lat_values.append(float(bin_item["lat"]))
        lon_values.append(float(bin_item["lon"]))

    for line in street_lines:
        for lat, lon in line:
            lat_values.append(float(lat))
            lon_values.append(float(lon))

    if not lat_values or not lon_values:
        return 47.0105, 28.8638

    return sum(lat_values) / len(lat_values), sum(lon_values) / len(lon_values)


def compute_bin_load_series(
    waste_events: list[dict[str, object]],
    bins: list[dict[str, float | int]],
) -> tuple[list[str], list[float], list[dict[str, object]]]:
    day_labels = sorted({str(event["date"]) for event in waste_events})
    day_index = {label: index for index, label in enumerate(day_labels)}

    daily_loads: dict[int, list[float]] = {int(bin_item["binId"]): [0.0] * len(day_labels) for bin_item in bins}
    daily_totals = [0.0] * len(day_labels)

    for event in waste_events:
        label = str(event["date"])
        if label not in day_index:
            continue
        index = day_index[label]
        bin_id = int(event["binId"])
        weight = float(event["weight"])
        daily_loads.setdefault(bin_id, [0.0] * len(day_labels))[index] += weight
        daily_totals[index] += weight

    cumulative_totals: list[float] = []
    running_total = 0.0
    for daily_total in daily_totals:
        running_total += daily_total
        cumulative_totals.append(round(running_total, 3))

    series: list[dict[str, object]] = []
    for bin_item in bins:
        bin_id = int(bin_item["binId"])
        cumulative = 0.0
        values: list[float] = []
        for day_load in daily_loads.get(bin_id, [0.0] * len(day_labels)):
            cumulative += day_load
            values.append(round(cumulative, 3))
        peak = round(max(values) if values else 0.0, 3)
        series.append(
            {
                "binId": bin_id,
                "lat": float(bin_item["lat"]),
                "lon": float(bin_item["lon"]),
                "values": values,
                "peak": peak,
                "total": round(values[-1] if values else 0.0, 3),
            }
        )

    series.sort(key=lambda item: (-float(item["peak"]), int(item["binId"])))
    return day_labels, cumulative_totals, series


def build_html(
    pois: list[dict[str, object]],
    bins: list[dict[str, float | int]],
    street_lines: list[list[list[float]]],
    day_labels: list[str],
    day_totals: list[float],
    load_series: list[dict[str, object]],
    center_lat: float,
    center_lon: float,
    routes_data: list[dict[str, Any]] | None = None,
) -> str:
    pois_json = json.dumps(pois, ensure_ascii=True)
    bins_json = json.dumps(bins, ensure_ascii=True)
    streets_json = json.dumps(street_lines, ensure_ascii=True)
    day_labels_json = json.dumps(day_labels, ensure_ascii=True)
    day_totals_json = json.dumps(day_totals, ensure_ascii=True)
    load_series_json = json.dumps(load_series, ensure_ascii=True)
    routes_json_str = json.dumps(routes_data or [], ensure_ascii=True)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Route Optimizer — Bin Data &amp; Route Comparison</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --ink: #15202b;
      --panel: #ffffffde;
      --panel-border: #d0d0d0;
      --bg: #edf2f7;
      --green: #22c55e;
      --red: #ef4444;
      --muted: #64748b;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
      color: var(--ink);
      font-family: "Avenir Next", "Helvetica Neue", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, #ffffff 0, #f4f7fb 36%, #e8edf4 100%);
    }}

    .top {{
      position: sticky;
      top: 0;
      z-index: 1000;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
      padding: 12px 16px 10px;
      background: #ffffffbf;
      border-bottom: 1px solid #d8deea;
      backdrop-filter: blur(4px);
    }}

    .brand {{ min-width: 260px; }}

    .title {{
      margin: 0;
      font-size: clamp(1.05rem, 2vw, 1.4rem);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .counts {{
      margin-top: 6px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      font-size: 0.86rem;
      color: #334155;
    }}

    .controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}

    .toggle-group {{
      display: inline-flex;
      overflow: hidden;
      border: 1px solid #cad3e0;
      border-radius: 999px;
      background: #fff;
      box-shadow: 0 6px 14px #0000000f;
    }}

    .toggle-group button {{
      border: 0;
      background: transparent;
      color: var(--ink);
      cursor: pointer;
      padding: 9px 14px;
      font: inherit;
      transition: background 120ms ease, color 120ms ease;
    }}

    .toggle-group button.active {{
      background: #0f172a;
      color: #fff;
    }}

    .day-control {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid #cad3e0;
      background: #fff;
      box-shadow: 0 6px 14px #0000000f;
    }}

    .day-control label {{
      font-size: 0.86rem;
      font-weight: 600;
      white-space: nowrap;
    }}

    .day-control input[type="range"] {{
      width: min(280px, 46vw);
    }}

    .day-label {{
      min-width: 100px;
      text-align: right;
      font-size: 0.86rem;
      color: #334155;
      white-space: nowrap;
    }}

    .map-wrap {{
      position: relative;
      margin: 10px;
      border: 1px solid #d4deec;
      border-radius: 14px;
      overflow: hidden;
      min-height: min(78vh, 860px);
      box-shadow: 0 14px 30px #00000014;
      animation: rise 0.45s ease-out;
    }}

    .view {{ display: none; width: 100%; height: 100%; }}
    .view.active {{ display: block; }}

    #map {{
      width: 100%;
      height: 100%;
      min-height: min(78vh, 860px);
    }}

    .leaflet-tile {{
      filter: grayscale(100%) contrast(1.05) brightness(1.02);
    }}

    .legend {{
      position: absolute;
      right: 12px;
      top: 12px;
      z-index: 900;
      max-width: min(280px, 72vw);
      padding: 10px 12px;
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      background: var(--panel);
      box-shadow: 0 8px 18px #0000001f;
    }}

    .legend h2 {{
      margin: 0 0 8px;
      font-size: 0.88rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}

    .legend p {{
      margin: 8px 0 0;
      font-size: 0.82rem;
      line-height: 1.4;
      color: #334155;
    }}

    .scale {{
      height: 12px;
      border-radius: 999px;
      border: 1px solid #d1d5db;
      background: linear-gradient(90deg, var(--green), #a3e635 40%, #f59e0b 68%, var(--red));
    }}

    .scale-labels {{
      margin-top: 6px;
      display: flex;
      justify-content: space-between;
      font-size: 0.72rem;
      color: #475569;
    }}

    .load-panel {{
      display: grid;
      grid-template-rows: auto auto 1fr;
      height: 100%;
      min-height: min(78vh, 860px);
    }}

    .load-summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 14px;
    }}

    .summary-card {{
      padding: 12px 14px;
      border: 1px solid var(--panel-border);
      border-radius: 14px;
      background: var(--panel);
      box-shadow: 0 10px 24px #00000010;
    }}

    .summary-card .label {{
      display: block;
      margin-bottom: 6px;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}

    .summary-card strong {{ font-size: 1.15rem; }}

    .view-caption {{
      padding: 0 14px 12px;
      color: #475569;
      font-size: 0.9rem;
    }}

    .table-wrap {{
      overflow: auto;
      margin: 0 14px 14px;
      border: 1px solid #d4deec;
      border-radius: 14px;
      background: #ffffffd8;
      box-shadow: 0 14px 30px #00000010;
    }}

    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 0.84rem;
    }}

    thead th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: #f8fafc;
      border-bottom: 1px solid #d8e1ef;
      padding: 10px 8px;
      text-align: center;
      white-space: nowrap;
    }}

    tbody th {{
      position: sticky;
      left: 0;
      z-index: 1;
      background: #fff;
      padding: 8px 10px;
      border-right: 1px solid #edf2f7;
      text-align: left;
      white-space: nowrap;
      font-weight: 600;
    }}

    tbody td {{
      min-width: 78px;
      padding: 8px 6px;
      border-bottom: 1px solid #eef2f7;
      border-right: 1px solid #eef2f7;
      text-align: center;
      transition: transform 120ms ease, box-shadow 120ms ease;
    }}

    tbody td.current-day,
    thead th.current-day {{
      outline: 2px solid #0f172a;
      outline-offset: -2px;
      box-shadow: inset 0 0 0 999px rgba(255, 255, 255, 0.08);
    }}

    tbody tr:hover td,
    tbody tr:hover th {{
      background: #f8fafc;
    }}

    @keyframes rise {{
      from {{ transform: translateY(8px); opacity: 0; }}
      to {{ transform: translateY(0); opacity: 1; }}
    }}

    @media (max-width: 760px) {{
      .top {{ position: static; }}
      .controls {{ width: 100%; justify-content: flex-start; }}
      .load-summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .map-wrap {{ margin: 6px; border-radius: 10px; }}
      .legend {{ right: 8px; top: 8px; padding: 8px 10px; }}
    }}

    /* ── route comparison view ── */
    .route-panel {{
      display: grid;
      grid-template-rows: auto auto 1fr;
      height: 100%;
      min-height: min(78vh, 860px);
    }}

    .route-summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 14px 14px 0;
    }}

    @media (max-width: 760px) {{
      .route-summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}

    .route-body {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding: 12px 14px 14px;
      overflow: auto;
    }}

    @media (max-width: 900px) {{
      .route-body {{ grid-template-columns: 1fr; }}
    }}

    .route-map-wrap {{
      position: relative;
      border: 1px solid #d4deec;
      border-radius: 14px;
      overflow: hidden;
      min-height: 320px;
      box-shadow: 0 14px 30px #00000014;
    }}

    #route-map {{
      width: 100%;
      height: 100%;
      min-height: 320px;
    }}

    .chart-card {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 14px;
      border: 1px solid var(--panel-border);
      border-radius: 14px;
      background: var(--panel);
      box-shadow: 0 10px 24px #00000010;
    }}

    .chart-card h3 {{
      margin: 0;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}

    .chart-card canvas {{ flex: 1; }}

    .route-legend {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      font-size: 0.82rem;
      padding: 6px 0 0;
    }}

    .route-legend span {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .route-legend i {{
      display: inline-block;
      width: 28px;
      height: 4px;
      border-radius: 2px;
    }}

    .algo-badge {{
      display: inline-block;
      margin-left: 8px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      background: #0f172a;
      color: #ffffff;
      vertical-align: middle;
    }}

    .savings-badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 700;
      background: #dcfce7;
      color: #166534;
    }}

    .route-toggle-group {{
      display: inline-flex;
      overflow: hidden;
      border: 1px solid #cad3e0;
      border-radius: 999px;
      background: #fff;
      box-shadow: 0 4px 10px #0000000d;
      margin: 0 14px 10px;
    }}

    .route-toggle-group button {{
      border: 0;
      background: transparent;
      color: var(--ink);
      cursor: pointer;
      padding: 7px 14px;
      font: inherit;
      font-size: 0.84rem;
      transition: background 120ms ease, color 120ms ease;
    }}

    .route-toggle-group button.active {{
      background: #0f172a;
      color: #fff;
    }}

    .no-routes-msg {{
      padding: 40px;
      text-align: center;
      color: var(--muted);
      font-size: 0.9rem;
    }}
  </style>
</head>
<body>
  <header class="top">
    <div class="brand">
      <h1 class="title">Route Optimizer — Bin Data &amp; Route Comparison</h1>
      <div class="counts" id="counts"></div>
    </div>
    <div class="controls">
      <div class="toggle-group" role="tablist" aria-label="View switcher">
        <button id="map-tab" class="active" type="button" role="tab" aria-selected="true">Map view</button>
        <button id="load-tab" type="button" role="tab" aria-selected="false">Load view</button>
        <button id="route-tab" type="button" role="tab" aria-selected="false">Route comparison</button>
      </div>
      <div class="day-control" aria-label="Day selector">
        <label for="day-slider">Day</label>
        <input id="day-slider" type="range" min="0" max="6" step="1" value="0" />
        <span class="day-label" id="day-label"></span>
      </div>
    </div>
  </header>

  <section class="map-wrap">
    <div class="view active" id="map-view">
      <div id="map" aria-label="Map with street lines, points of interest, bins, and bin fill colors"></div>
      <aside class="legend">
        <h2>Bin Fill</h2>
        <div class="scale" aria-hidden="true"></div>
        <div class="scale-labels"><span>empty</span><span>full</span></div>
        <p id="map-summary">Trash is never collected, so every bin gets heavier from day to day.</p>
      </aside>
    </div>

    <div class="view" id="load-view">
      <div class="load-panel">
        <div class="load-summary">
          <div class="summary-card"><span class="label">Selected day</span><strong id="summary-day">-</strong></div>
          <div class="summary-card"><span class="label">Total trash</span><strong id="summary-total">0.00 kg</strong></div>
          <div class="summary-card"><span class="label">Peak bin load</span><strong id="summary-peak">0.00 kg</strong></div>
          <div class="summary-card"><span class="label">Most loaded bin</span><strong id="summary-bin">#-</strong></div>
        </div>
        <div class="view-caption">Each cell shows the cumulative waste held in a bin on that day. Colors move from green to red as the bin fills.</div>
        <div class="table-wrap">
          <table aria-label="Bin load by day">
            <thead>
              <tr id="load-head-row"><th>Bin</th></tr>
            </thead>
            <tbody id="load-body"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ── Route Comparison view ─────────────────────────────────────────── -->
    <div class="view" id="route-view">
      <div id="no-routes-msg" class="no-routes-msg" style="display:none">
        Route data not found. Run <code>route_optimizer.py</code> first, then re-generate the visualisation.
      </div>
      <div id="route-content" class="route-panel">
        <div class="route-summary">
          <div class="summary-card">
            <span class="label">Selected day</span>
            <strong id="route-summary-day">-</strong>
          </div>
          <div class="summary-card">
            <span class="label">Distance saved</span>
            <strong id="route-summary-dist">-</strong>
          </div>
          <div class="summary-card">
            <span class="label">Fuel saved</span>
            <strong id="route-summary-fuel">-</strong>
          </div>
          <div class="summary-card">
            <span class="label">Saving</span>
            <strong id="route-summary-pct">-</strong>
          </div>
        </div>

        <div class="route-toggle-group" role="group" aria-label="Route type">
          <button id="show-both-btn" class="active" type="button">Both routes</button>
          <button id="show-regular-btn" type="button">Regular only</button>
          <button id="show-opt-btn" type="button">ML-optimised only</button>
        </div>

        <div class="route-body">
          <div class="route-map-wrap">
            <div id="route-map" aria-label="Route comparison map"></div>
          </div>
          <div style="display:flex;flex-direction:column;gap:12px;">
            <div class="chart-card">
              <h3>Distance per day (km)</h3>
              <canvas id="dist-chart" height="160"></canvas>
              <div class="route-legend">
                <span><i style="background:#6b7280"></i>Regular route</span>
                <span><i style="background:#22c55e"></i>ML-optimised <span class="algo-badge">NN-TSP</span></span>
              </div>
            </div>
            <div class="chart-card">
              <h3>Fuel per day (litres)</h3>
              <canvas id="fuel-chart" height="160"></canvas>
              <div class="route-legend">
                <span><i style="background:#6b7280"></i>Regular route</span>
                <span><i style="background:#22c55e"></i>ML-optimised</span>
              </div>
            </div>
            <div class="chart-card">
              <h3>Distance &amp; fuel savings summary</h3>
              <div id="savings-table" style="font-size:0.85rem;line-height:1.9;"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <script>
    const pois = {pois_json};
    const bins = {bins_json};
    const streets = {streets_json};
    const dayLabels = {day_labels_json};
    const dayTotals = {day_totals_json};
    const loadSeries = {load_series_json};
    const routesData = {routes_json_str};

    const center = [{center_lat}, {center_lon}];
    const map = L.map("map", {{ zoomControl: true }}).setView(center, 14);

    L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }}).addTo(map);

    const markerColor = "#111111";
    const green = [34, 197, 94];
    const red = [239, 68, 68];

    function clamp(value, min, max) {{
      return Math.min(max, Math.max(min, value));
    }}

    function colorForRatio(ratio) {{
      const t = clamp(ratio, 0, 1);
      const r = Math.round(green[0] + (red[0] - green[0]) * t);
      const g = Math.round(green[1] + (red[1] - green[1]) * t);
      const b = Math.round(green[2] + (red[2] - green[2]) * t);
      return `rgb(${{r}}, ${{g}}, ${{b}})`;
    }}

    function formatWeight(value) {{
      return `${{value.toFixed(2)}} kg`;
    }}

    function dayLabel(index) {{
      const raw = dayLabels[index] || `Day ${{index + 1}}`;
      const parsed = new Date(`${{raw}}T00:00:00Z`);
      if (Number.isNaN(parsed.getTime())) {{
        return raw;
      }}
      return parsed.toLocaleDateString("en-GB", {{ timeZone: "UTC", month: "short", day: "numeric" }});
    }}

    const streetLayer = L.layerGroup().addTo(map);
    streets.forEach(line => {{
      L.polyline(line, {{ color: "#222222", weight: 2.3, opacity: 0.82 }}).addTo(streetLayer);
    }});

    const poiLayer = L.layerGroup().addTo(map);
    pois.forEach(p => {{
      const marker = L.circleMarker([p.lat, p.lon], {{
        radius: 5,
        fillColor: markerColor,
        color: "#ffffff",
        weight: 0.9,
        fillOpacity: 0.86,
      }});
      marker.bindTooltip(`${{String(p.type || "other")}}<br/>${{p.lat.toFixed(6)}}, ${{p.lon.toFixed(6)}}`);
      marker.addTo(poiLayer);
    }});

    const binLayer = L.layerGroup().addTo(map);
    const binMarkers = new Map();
    bins.forEach(b => {{
      const marker = L.circleMarker([b.lat, b.lon], {{
        radius: 4,
        fillColor: "#22c55e",
        color: "#14532d",
        weight: 0.8,
        fillOpacity: 0.95,
      }});
      marker.bindTooltip(`Bin #${{b.binId}}<br/>Loading...`);
      marker.addTo(binLayer);
      binMarkers.set(b.binId, marker);
    }});

    const countsEl = document.getElementById("counts");
    countsEl.innerHTML = `
      <span><strong>Streets:</strong> ${{streets.length}}</span>
      <span><strong>POIs:</strong> ${{pois.length}}</span>
      <span><strong>Bins:</strong> ${{bins.length}}</span>
      <span><strong>Days:</strong> ${{dayLabels.length}}</span>
    `;

    const viewState = {{ mode: "map", dayIndex: 0 }};
    const mapTab = document.getElementById("map-tab");
    const loadTab = document.getElementById("load-tab");
    const routeTab = document.getElementById("route-tab");
    const mapView = document.getElementById("map-view");
    const loadView = document.getElementById("load-view");
    const routeView = document.getElementById("route-view");
    const daySlider = document.getElementById("day-slider");
    const dayLabelEl = document.getElementById("day-label");
    const mapSummary = document.getElementById("map-summary");
    const summaryDay = document.getElementById("summary-day");
    const summaryTotal = document.getElementById("summary-total");
    const summaryPeak = document.getElementById("summary-peak");
    const summaryBin = document.getElementById("summary-bin");
    const loadHeadRow = document.getElementById("load-head-row");
    const loadBody = document.getElementById("load-body");

    const loadLookup = new Map(loadSeries.map(item => [item.binId, item]));

    function renderLoadTable() {{
      loadHeadRow.innerHTML = "<th>Bin</th>";
      dayLabels.forEach((_, index) => {{
        const th = document.createElement("th");
        th.textContent = dayLabel(index);
        th.dataset.dayIndex = String(index);
        loadHeadRow.appendChild(th);
      }});

      loadBody.innerHTML = "";
      loadSeries.forEach(series => {{
        const row = document.createElement("tr");
        const header = document.createElement("th");
        header.textContent = `#${{series.binId}}`;
        row.appendChild(header);

        series.values.forEach((value, index) => {{
          const cell = document.createElement("td");
          const ratio = series.peak > 0 ? value / series.peak : 0;
          cell.textContent = value.toFixed(2);
          cell.title = `Bin #${{series.binId}} - ${{dayLabel(index)}}: ${{formatWeight(value)}}`;
          cell.dataset.dayIndex = String(index);
          cell.style.background = colorForRatio(ratio);
          cell.style.color = ratio > 0.6 ? "#ffffff" : "#0f172a";
          row.appendChild(cell);
        }});

        loadBody.appendChild(row);
      }});
    }}

    function setActiveDayColumn(dayIndex) {{
      loadHeadRow.querySelectorAll("th[data-day-index]").forEach(th => {{
        th.classList.toggle("current-day", Number(th.dataset.dayIndex) === dayIndex);
      }});
      loadBody.querySelectorAll("td[data-day-index]").forEach(td => {{
        td.classList.toggle("current-day", Number(td.dataset.dayIndex) === dayIndex);
      }});
    }}

    function updateMapColors(dayIndex) {{
      let selectedDayPeak = 0;
      let busiestBin = null;

      loadSeries.forEach(series => {{
        const value = series.values[dayIndex] || 0;
        if (value > selectedDayPeak) {{
          selectedDayPeak = value;
          busiestBin = series.binId;
        }}

        const marker = binMarkers.get(series.binId);
        if (!marker) {{
          return;
        }}

        const ratio = series.peak > 0 ? value / series.peak : 0;
        marker.setStyle({{
          fillColor: colorForRatio(ratio),
          color: ratio > 0.85 ? "#7f1d1d" : "#14532d",
          fillOpacity: 0.95,
          radius: 4 + Math.min(5, value / 6),
        }});
        marker.setTooltipContent(`Bin #${{series.binId}}<br/>${{dayLabel(dayIndex)}}: ${{formatWeight(value)}}<br/>Peak: ${{formatWeight(series.peak)}}`);
      }});

      const totalLoad = dayTotals[dayIndex] || 0;
      mapSummary.textContent = `${{dayLabel(dayIndex)}} | total waste stored: ${{formatWeight(totalLoad)}}`;
      summaryDay.textContent = dayLabel(dayIndex);
      summaryTotal.textContent = formatWeight(totalLoad);
      summaryPeak.textContent = formatWeight(selectedDayPeak);
      summaryBin.textContent = busiestBin === null ? "#-" : `#${{busiestBin}}`;
      setActiveDayColumn(dayIndex);
    }}

    function setMode(mode) {{
      viewState.mode = mode;
      mapTab.classList.toggle("active", mode === "map");
      loadTab.classList.toggle("active", mode === "load");
      routeTab.classList.toggle("active", mode === "route");
      mapTab.setAttribute("aria-selected", String(mode === "map"));
      loadTab.setAttribute("aria-selected", String(mode === "load"));
      routeTab.setAttribute("aria-selected", String(mode === "route"));
      mapView.classList.toggle("active", mode === "map");
      loadView.classList.toggle("active", mode === "load");
      routeView.classList.toggle("active", mode === "route");
      requestAnimationFrame(() => {{
        map.invalidateSize();
        if (mode === "route" && routeMap) routeMap.invalidateSize();
      }});
    }}

    mapTab.addEventListener("click", () => setMode("map"));
    loadTab.addEventListener("click", () => setMode("load"));
    routeTab.addEventListener("click", () => setMode("route"));
    daySlider.max = String(Math.max(0, dayLabels.length - 1));
    daySlider.value = "0";
    daySlider.addEventListener("input", event => {{
      viewState.dayIndex = Number(event.target.value || 0);
      dayLabelEl.textContent = dayLabel(viewState.dayIndex);
      updateMapColors(viewState.dayIndex);
    }});

    renderLoadTable();
    dayLabelEl.textContent = dayLabel(0);
    updateMapColors(0);

    const allCoords = [];
    streets.forEach(line => line.forEach(point => allCoords.push(point)));
    pois.forEach(p => allCoords.push([p.lat, p.lon]));
    bins.forEach(b => allCoords.push([b.lat, b.lon]));
    if (allCoords.length > 1) {{
      map.fitBounds(allCoords, {{ padding: [20, 20] }});
    }}

    // ── Route Comparison ──────────────────────────────────────────────────
    let routeMap = null;
    let regularPolyline = null;
    let optimisedPolyline = null;
    let depotMarker = null;
    let distChart = null;
    let fuelChart = null;
    let routeViewState = {{ dayIndex: 0, show: "both" }};

    function fmtKm(v) {{ return `${{Number(v).toFixed(2)}} km`; }}
    function fmtL(v)  {{ return `${{Number(v).toFixed(2)}} L`; }}
    function fmtPct(v) {{ return `${{Number(v).toFixed(1)}} %`; }}

    function routeDayLabel(day) {{
      const parsed = new Date(`${{day}}T00:00:00Z`);
      if (Number.isNaN(parsed.getTime())) return day;
      return parsed.toLocaleDateString("en-GB", {{ timeZone: "UTC", month: "short", day: "numeric" }});
    }}

    function initRouteView() {{
      if (!routesData || routesData.length === 0) {{
        document.getElementById("no-routes-msg").style.display = "block";
        document.getElementById("route-content").style.display = "none";
        return;
      }}

      // Initialise route map
      routeMap = L.map("route-map", {{ zoomControl: true }}).setView([{center_lat}, {center_lon}], 14);
      L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors"
      }}).addTo(routeMap);

      // Draw street backdrop (lighter, thinner)
      streets.forEach(line => {{
        L.polyline(line, {{ color: "#c0c8d8", weight: 1.5, opacity: 0.6 }}).addTo(routeMap);
      }});

      regularPolyline = L.polyline([], {{ color: "#6b7280", weight: 3.5, opacity: 0.85, dashArray: "8 5" }}).addTo(routeMap);
      optimisedPolyline = L.polyline([], {{ color: "#22c55e", weight: 3.5, opacity: 0.9 }}).addTo(routeMap);
      depotMarker = L.circleMarker([{center_lat}, {center_lon}], {{
        radius: 8, fillColor: "#f59e0b", color: "#92400e", weight: 2, fillOpacity: 1
      }}).bindTooltip("Depot (centroid)").addTo(routeMap);

      // Charts
      const distCtx = document.getElementById("dist-chart").getContext("2d");
      const fuelCtx = document.getElementById("fuel-chart").getContext("2d");
      const routeDates = routesData.map(r => routeDayLabel(r.date));

      distChart = new Chart(distCtx, {{
        type: "bar",
        data: {{
          labels: routeDates,
          datasets: [
            {{ label: "Regular (km)", data: routesData.map(r => r.regular.distance_km), backgroundColor: "#6b7280" }},
            {{ label: "ML-Optimised (km)", data: routesData.map(r => r.optimised.distance_km), backgroundColor: "#22c55e" }},
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ position: "bottom", labels: {{ boxWidth: 14, font: {{ size: 11 }} }} }} }},
          scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: "km" }} }} }}
        }}
      }});

      fuelChart = new Chart(fuelCtx, {{
        type: "bar",
        data: {{
          labels: routeDates,
          datasets: [
            {{ label: "Regular (L)", data: routesData.map(r => r.regular.fuel_l), backgroundColor: "#6b7280" }},
            {{ label: "ML-Optimised (L)", data: routesData.map(r => r.optimised.fuel_l), backgroundColor: "#22c55e" }},
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ position: "bottom", labels: {{ boxWidth: 14, font: {{ size: 11 }} }} }} }},
          scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: "litres" }} }} }}
        }}
      }});

      // Summary table
      const totalRegDist = routesData.reduce((s, r) => s + r.regular.distance_km, 0);
      const totalOptDist = routesData.reduce((s, r) => s + r.optimised.distance_km, 0);
      const totalRegFuel = routesData.reduce((s, r) => s + r.regular.fuel_l, 0);
      const totalOptFuel = routesData.reduce((s, r) => s + r.optimised.fuel_l, 0);
      const totalSavedDist = routesData.reduce((s, r) => s + r.saved_km, 0);
      const totalSavedFuel = routesData.reduce((s, r) => s + r.saved_l, 0);
      const avgPct = routesData.reduce((s, r) => s + r.savings_pct, 0) / routesData.length;

      document.getElementById("savings-table").innerHTML = `
        <table style="width:100%;border-collapse:collapse">
          <thead><tr style="border-bottom:1px solid #e2e8f0">
            <th style="text-align:left;padding:4px 6px;font-size:0.78rem;color:#64748b">Metric</th>
            <th style="text-align:right;padding:4px 6px;font-size:0.78rem;color:#6b7280">Regular</th>
            <th style="text-align:right;padding:4px 6px;font-size:0.78rem;color:#22c55e">ML-Opt.</th>
            <th style="text-align:right;padding:4px 6px;font-size:0.78rem;color:#0ea5e9">Saved</th>
          </tr></thead>
          <tbody>
            <tr><td style="padding:4px 6px">Total distance</td>
                <td style="text-align:right;padding:4px 6px">${{fmtKm(totalRegDist)}}</td>
                <td style="text-align:right;padding:4px 6px">${{fmtKm(totalOptDist)}}</td>
                <td style="text-align:right;padding:4px 6px"><span class="savings-badge">${{fmtKm(totalSavedDist)}}</span></td></tr>
            <tr><td style="padding:4px 6px">Total fuel</td>
                <td style="text-align:right;padding:4px 6px">${{fmtL(totalRegFuel)}}</td>
                <td style="text-align:right;padding:4px 6px">${{fmtL(totalOptFuel)}}</td>
                <td style="text-align:right;padding:4px 6px"><span class="savings-badge">${{fmtL(totalSavedFuel)}}</span></td></tr>
            <tr><td style="padding:4px 6px">Avg. daily saving</td>
                <td colspan="2" style="text-align:right;padding:4px 6px">—</td>
                <td style="text-align:right;padding:4px 6px"><span class="savings-badge">${{fmtPct(avgPct)}}</span></td></tr>
          </tbody>
        </table>`;

      // Wire up route-toggle buttons
      document.getElementById("show-both-btn").addEventListener("click", () => setRouteShow("both"));
      document.getElementById("show-regular-btn").addEventListener("click", () => setRouteShow("regular"));
      document.getElementById("show-opt-btn").addEventListener("click", () => setRouteShow("optimised"));

      updateRouteDay(0);
    }}

    function setRouteShow(which) {{
      routeViewState.show = which;
      document.getElementById("show-both-btn").classList.toggle("active", which === "both");
      document.getElementById("show-regular-btn").classList.toggle("active", which === "regular");
      document.getElementById("show-opt-btn").classList.toggle("active", which === "optimised");
      updateRouteDay(routeViewState.dayIndex);
    }}

    function updateRouteDay(dayIdx) {{
      if (!routesData || routesData.length === 0 || !routeMap) return;
      routeViewState.dayIndex = dayIdx;
      const entry = routesData[dayIdx];
      if (!entry) return;

      const show = routeViewState.show;
      regularPolyline.setLatLngs(show !== "optimised" ? entry.regular.coords : []);
      optimisedPolyline.setLatLngs(show !== "regular" ? entry.optimised.coords : []);

      const savedPct = entry.savings_pct;
      document.getElementById("route-summary-day").textContent = routeDayLabel(entry.date);
      document.getElementById("route-summary-dist").innerHTML =
        `${{fmtKm(entry.saved_km)}} <span class="savings-badge">${{fmtPct(savedPct)}}</span>`;
      document.getElementById("route-summary-fuel").textContent = fmtL(entry.saved_l);
      document.getElementById("route-summary-pct").innerHTML =
        `<span class="savings-badge">${{entry.optimised.bins}} / ${{entry.regular.bins}} bins visited</span>`;

      // Fit map to visible route
      const visible = [];
      if (show !== "optimised") entry.regular.coords.forEach(c => visible.push(c));
      if (show !== "regular") entry.optimised.coords.forEach(c => visible.push(c));
      if (visible.length > 1) routeMap.fitBounds(visible, {{ padding: [24, 24] }});
    }}

    // Wire day slider to both views
    daySlider.addEventListener("input", event => {{
      const idx = Number(event.target.value || 0);
      viewState.dayIndex = idx;
      dayLabelEl.textContent = dayLabel(idx);
      updateMapColors(idx);
      if (routesData && routesData.length > 0) {{
        // find the closest routes day index
        const routeIdx = Math.min(idx, routesData.length - 1);
        updateRouteDay(routeIdx);
      }}
    }});

    // Lazy-init route view on first activation
    let routeViewInitialised = false;
    routeTab.addEventListener("click", () => {{
      if (!routeViewInitialised) {{
        routeViewInitialised = true;
        initRouteView();
      }}
      requestAnimationFrame(() => {{ if (routeMap) routeMap.invalidateSize(); }});
    }});

  </script>
</body>
</html>
"""


def main() -> None:
    gen_dir = generated_dir()
    pois_csv = gen_dir / "pois.csv"
    bins_csv = gen_dir / "bins.csv"
    streets_csv = gen_dir / "street_lines.csv"
    waste_events_csv = gen_dir / "waste_events.csv"
    routes_json = gen_dir / "routes.json"
    output_html = gen_dir / "visualization.html"

    missing = [str(path) for path in (pois_csv, bins_csv, streets_csv, waste_events_csv) if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing required CSV files. Run the generators first:\n"
            + "\n".join(f"- {path}" for path in missing)
        )

    pois = read_pois(pois_csv)
    bins = read_bins(bins_csv)
    street_lines = read_street_lines(streets_csv)
    waste_events = read_waste_events(waste_events_csv)
    routes_data = read_routes_data(routes_json)
    day_labels, day_totals, load_series = compute_bin_load_series(waste_events, bins)
    center_lat, center_lon = compute_center(pois, bins, street_lines)

    html = build_html(
        pois,
        bins,
        street_lines,
        day_labels,
        day_totals,
        load_series,
        center_lat,
        center_lon,
        routes_data,
    )
    output_html.write_text(html, encoding="utf-8")

    print(f"HTML visualization written to: {output_html}")
    print(f"Loaded {len(pois)} POIs, {len(bins)} bins, {len(street_lines)} street lines, and {len(waste_events)} waste events")
    if routes_data:
        print(f"Route comparison data: {len(routes_data)} days embedded")
    else:
        print("Route comparison data not found — run route_optimizer.py first")


if __name__ == "__main__":
    main()