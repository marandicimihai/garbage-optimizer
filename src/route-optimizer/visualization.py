from __future__ import annotations

import csv
import json
from pathlib import Path


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


def build_html(
    pois: list[dict[str, object]],
    bins: list[dict[str, float | int]],
    street_lines: list[list[list[float]]],
    center_lat: float,
    center_lon: float,
) -> str:
    pois_json = json.dumps(pois, ensure_ascii=True)
    bins_json = json.dumps(bins, ensure_ascii=True)
    streets_json = json.dumps(street_lines, ensure_ascii=True)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Route Optimizer Data View</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
  <style>
    :root {{
      --bg-a: #f5f5f5;
      --bg-b: #f5f5f5;
      --ink: #15202b;
      --street: #2f2f2f;
      --panel: #ffffffdd;
      --panel-border: #d0d0d0;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "Helvetica Neue", "Segoe UI", sans-serif;
      background: #f5f5f5;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }}

    .top {{
      padding: 12px 16px;
      border-bottom: 1px solid #d8deea;
      background: #ffffffbf;
      backdrop-filter: blur(3px);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .title {{
      margin: 0;
      font-size: clamp(1.05rem, 2vw, 1.35rem);
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}

    .counts {{
      font-size: 0.9rem;
      opacity: 0.9;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
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
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      padding: 10px 12px;
      box-shadow: 0 8px 18px #0000001f;
      max-width: min(260px, 70vw);
    }}

    .legend h2 {{
      margin: 0 0 8px;
      font-size: 0.88rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}

    .legend ul {{
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 5px;
      font-size: 0.82rem;
    }}

    .legend li {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }}

    .pill {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      margin-right: 6px;
      border: 1px solid #22222255;
    }}

    @keyframes rise {{
      from {{ transform: translateY(8px); opacity: 0; }}
      to {{ transform: translateY(0); opacity: 1; }}
    }}

    @media (max-width: 680px) {{
      .legend {{
        right: 8px;
        top: 8px;
        padding: 8px 10px;
      }}

      .map-wrap {{
        margin: 6px;
        border-radius: 10px;
      }}
    }}
  </style>
</head>
<body>
  <header class="top">
    <h1 class="title">Street Lines + POIs + Bins</h1>
    <div class="counts" id="counts"></div>
  </header>

  <section class="map-wrap">
    <div id="map" aria-label="Map with street lines, points of interest, and bins"></div>
    <aside class="legend">
      <h2>POI Types</h2>
      <ul id="legend-list"></ul>
    </aside>
  </section>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <script>
    const pois = {pois_json};
    const bins = {bins_json};
    const streets = {streets_json};

    const center = [{center_lat}, {center_lon}];
    const map = L.map("map", {{ zoomControl: true }}).setView(center, 14);

    L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }}).addTo(map);

    const markerColor = "#111111";

    const streetLayer = L.layerGroup().addTo(map);
    streets.forEach(line => {{
      L.polyline(line, {{ color: "#222222", weight: 2.3, opacity: 0.82 }}).addTo(streetLayer);
    }});

    const poiLayer = L.layerGroup().addTo(map);
    const typeCounts = {{}};
    pois.forEach(p => {{
      const t = String(p.type || "other");
      typeCounts[t] = (typeCounts[t] || 0) + 1;

      const marker = L.circleMarker([p.lat, p.lon], {{
        radius: 5,
        fillColor: markerColor,
        color: "#ffffff",
        weight: 0.9,
        fillOpacity: 0.86,
      }});
      marker.bindTooltip(`${{t}}<br/>${{p.lat.toFixed(6)}}, ${{p.lon.toFixed(6)}}`);
      marker.addTo(poiLayer);
    }});

    const binLayer = L.layerGroup().addTo(map);
    bins.forEach(b => {{
      const marker = L.circleMarker([b.lat, b.lon], {{
        radius: 4,
        fillColor: "#f97316",
        color: "#111111",
        weight: 0.8,
        fillOpacity: 0.9,
      }});
      marker.bindTooltip(`Bin #${{b.binId}}<br/>${{b.lat.toFixed(6)}}, ${{b.lon.toFixed(6)}}`);
      marker.addTo(binLayer);
    }});

    const countsEl = document.getElementById("counts");
    countsEl.innerHTML = `
      <span><strong>Streets:</strong> ${{streets.length}}</span>
      <span><strong>POIs:</strong> ${{pois.length}}</span>
      <span><strong>Bins:</strong> ${{bins.length}}</span>
      <span><strong>Types:</strong> ${{Object.keys(typeCounts).length}}</span>
    `;

    const legendList = document.getElementById("legend-list");
    Object.keys(typeCounts)
      .sort((a, b) => typeCounts[b] - typeCounts[a])
      .forEach(type => {{
        const item = document.createElement("li");
        item.innerHTML = `<span><span class="pill" style="background:${{markerColor}}"></span>${{type}}</span><strong>${{typeCounts[type]}}</strong>`;
        legendList.appendChild(item);
      }});

    const allCoords = [];
    streets.forEach(line => line.forEach(point => allCoords.push(point)));
    pois.forEach(p => allCoords.push([p.lat, p.lon]));
    bins.forEach(b => allCoords.push([b.lat, b.lon]));
    if (allCoords.length > 1) {{
      map.fitBounds(allCoords, {{ padding: [20, 20] }});
    }}
  </script>
</body>
</html>
"""


def main() -> None:
    gen_dir = generated_dir()
    pois_csv = gen_dir / "pois.csv"
    bins_csv = gen_dir / "bins.csv"
    streets_csv = gen_dir / "street_lines.csv"
    output_html = gen_dir / "visualization.html"

    missing = [str(path) for path in (pois_csv, bins_csv, streets_csv) if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing required CSV files. Run the generators first:\n"
            + "\n".join(f"- {path}" for path in missing)
        )

    pois = read_pois(pois_csv)
    bins = read_bins(bins_csv)
    street_lines = read_street_lines(streets_csv)
    center_lat, center_lon = compute_center(pois, bins, street_lines)

    html = build_html(pois, bins, street_lines, center_lat, center_lon)
    output_html.write_text(html, encoding="utf-8")

    print(f"HTML visualization written to: {output_html}")
    print(f"Loaded {len(pois)} POIs, {len(bins)} bins, and {len(street_lines)} street lines")


if __name__ == "__main__":
    main()