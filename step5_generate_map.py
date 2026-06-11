#!/usr/bin/env python3
"""
Step 5 — Generate Interactive HTML Map
========================================
Creates a single-file Leaflet.js interactive map with:
  • Satellite + OSM basemap toggle
  • Every plant as a colour-coded circle marker
  • NDVI health colour scale
  • Click popup with full plant details
  • Layer controls to filter by health category
  • Stats panel (total, mean NDVI, category breakdown)
  • Mini histogram of NDVI distribution
  • Search / highlight by plant_id
"""

import json
import math
from pathlib import Path
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR  = Path(__file__).parent / "outputs"
OUT_HTML = OUT_DIR / "plant_ndvi_map.html"

# ── NDVI → colour mapping ──────────────────────────────────────────────────────
NDVI_COLOURS = {
    "Very Dense Vegetation":   "#1a7a2e",
    "Dense Vegetation":        "#52b44b",
    "Moderate Vegetation":     "#a8d96c",
    "Sparse Vegetation":       "#f5c842",
    "Bare / Stressed":         "#e07b2a",
    "Water / Non-vegetation":  "#3a6fbf",
    "No Data":                 "#888888",
}

CATEGORY_ORDER = [
    "Very Dense Vegetation",
    "Dense Vegetation",
    "Moderate Vegetation",
    "Sparse Vegetation",
    "Bare / Stressed",
    "Water / Non-vegetation",
    "No Data",
]


def ndvi_to_hex_gradient(val: float | None) -> str:
    """Return a hex colour interpolated across a red→yellow→green gradient."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "#888888"
    val = max(0.0, min(1.0, val))  # clamp
    if val < 0.4:
        # red → yellow
        t = val / 0.4
        r = 220
        g = int(60 + t * 180)
        b = 30
    else:
        # yellow → green
        t = (val - 0.4) / 0.6
        r = int(220 - t * 170)
        g = int(200 + t * 30)
        b = int(30 + t * 10)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_map(plants: list[dict], out_path: Path = OUT_HTML) -> Path:
    """
    Build the Leaflet.js HTML map from the enriched plant list.
    """
    print("=" * 60)
    print("STEP 5 — Generate Interactive Map")
    print("=" * 60)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Compute centre & stats ─────────────────────────────────────────────────
    lats = [p["lat"] for p in plants if p.get("lat")]
    lons = [p["lon"] for p in plants if p.get("lon")]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    ndvi_vals = [p["ndvi"] for p in plants if p.get("ndvi") is not None]
    mean_ndvi  = round(sum(ndvi_vals) / len(ndvi_vals), 4) if ndvi_vals else 0
    cat_counts = Counter(p.get("ndvi_category", "No Data") for p in plants)
    total      = len(plants)

    # ── Build GeoJSON feature collection for Leaflet ───────────────────────────
    print(f"[INFO] Building GeoJSON for {total:,} plants…")
    features = []
    for p in plants:
        colour  = ndvi_to_hex_gradient(p.get("ndvi"))
        cat     = p.get("ndvi_category", "No Data")
        ndvi_v  = p.get("ndvi")
        ndvi_s  = f"{ndvi_v:.4f}" if ndvi_v is not None else "N/A"

        feat = {
            "type": "Feature",
            "geometry": {
                "type":        "Point",
                "coordinates": [p["lon"], p["lat"]],
            },
            "properties": {
                "plant_id":     p.get("plant_id", ""),
                "instance_id":  p.get("instance_id", ""),
                "sector":       p.get("sector_label", ""),
                "row":          p.get("row_index"),
                "col":          p.get("col_index"),
                "ndvi":         ndvi_v,
                "ndvi_s":       ndvi_s,
                "ndvi_mean":    p.get("ndvi_mean"),
                "ndvi_max":     p.get("ndvi_max"),
                "ndvi_cat":     cat,
                "tif_ndvi":     p.get("tif_ndvi"),
                "area_px":      p.get("area_px"),
                "is_noise":     p.get("is_noise", False),
                "colour":       colour,
            }
        }
        features.append(feat)

    geojson_str = json.dumps({"type": "FeatureCollection", "features": features},
                             ensure_ascii=False, separators=(",", ":"))

    # ── Stats for the sidebar ──────────────────────────────────────────────────
    legend_html = ""
    for cat in CATEGORY_ORDER:
        cnt  = cat_counts.get(cat, 0)
        pct  = cnt / total * 100 if total else 0
        col  = NDVI_COLOURS.get(cat, "#888")
        legend_html += f"""
            <div class="legend-item" data-cat="{cat}">
              <span class="legend-dot" style="background:{col}"></span>
              <span class="legend-label">{cat}</span>
              <span class="legend-count">{cnt:,} <small>({pct:.1f}%)</small></span>
            </div>"""

    # Histogram bins (10 bins from -0.1 to 1.0)
    hist_bins  = 11
    hist_edges = [-0.1 + i * 0.1 for i in range(hist_bins + 1)]
    hist_cnts  = [0] * hist_bins
    for v in ndvi_vals:
        for i in range(hist_bins):
            if hist_edges[i] <= v < hist_edges[i + 1]:
                hist_cnts[i] += 1
                break
    hist_max   = max(hist_cnts) if hist_cnts else 1
    hist_json  = json.dumps({"edges": hist_edges, "counts": hist_cnts, "max": hist_max})

    # ── Write HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Plant NDVI Health Map — Pineapple Plantation</title>
<meta name="description" content="Interactive NDVI plant health map showing {total:,} detected pineapple plants with NDVI index values."/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:       #0f1117;
    --surface:  #1a1d27;
    --surface2: #23263a;
    --border:   #2e3248;
    --accent:   #4ade80;
    --accent2:  #22c55e;
    --text:     #e2e8f0;
    --muted:    #8892a4;
    --shadow:   0 4px 24px rgba(0,0,0,.5);
    --radius:   12px;
  }}

  body {{
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}

  /* ── Top bar ─────────────────────────────────────────────── */
  #topbar {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 10px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    z-index: 9000;
    flex-shrink: 0;
  }}
  #topbar h1 {{
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--accent);
    white-space: nowrap;
  }}
  #topbar .subtitle {{
    font-size: .75rem;
    color: var(--muted);
  }}
  #search-wrap {{
    margin-left: auto;
    display: flex;
    gap: 8px;
    align-items: center;
  }}
  #search-input {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 12px;
    color: var(--text);
    font-family: inherit;
    font-size: .82rem;
    width: 220px;
    outline: none;
    transition: border-color .2s;
  }}
  #search-input:focus {{ border-color: var(--accent); }}
  #search-btn {{
    background: var(--accent);
    border: none;
    border-radius: 8px;
    padding: 6px 14px;
    color: #0f1117;
    font-weight: 600;
    font-size: .82rem;
    cursor: pointer;
    transition: background .2s;
  }}
  #search-btn:hover {{ background: var(--accent2); }}

  /* ── Main layout ─────────────────────────────────────────── */
  #main {{
    display: flex;
    flex: 1;
    overflow: hidden;
  }}

  /* ── Sidebar ─────────────────────────────────────────────── */
  #sidebar {{
    width: 320px;
    min-width: 280px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0;
    z-index: 1000;
    flex-shrink: 0;
  }}

  .panel {{
    padding: 16px;
    border-bottom: 1px solid var(--border);
  }}
  .panel h3 {{
    font-size: .72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--muted);
    margin-bottom: 12px;
  }}

  /* KPI cards */
  .kpi-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }}
  .kpi {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px 12px;
  }}
  .kpi .val {{
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
  }}
  .kpi .label {{
    font-size: .68rem;
    color: var(--muted);
    margin-top: 3px;
  }}

  /* Legend */
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 4px;
    border-radius: 6px;
    cursor: pointer;
    transition: background .15s;
    user-select: none;
  }}
  .legend-item:hover {{ background: var(--surface2); }}
  .legend-item.inactive {{ opacity: .35; }}
  .legend-dot {{
    width: 12px; height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
    border: 1.5px solid rgba(255,255,255,.2);
  }}
  .legend-label {{ font-size: .78rem; flex: 1; }}
  .legend-count  {{ font-size: .72rem; color: var(--muted); white-space: nowrap; }}

  /* Histogram */
  #histogram {{ margin-top: 6px; }}
  .hist-bar-wrap {{
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 48px;
    margin-bottom: 4px;
  }}
  .hist-bar {{
    flex: 1;
    border-radius: 2px 2px 0 0;
    min-height: 2px;
    cursor: pointer;
    transition: opacity .2s;
  }}
  .hist-bar:hover {{ opacity: .7; }}
  .hist-labels {{
    display: flex;
    justify-content: space-between;
    font-size: .6rem;
    color: var(--muted);
  }}

  /* Filter toggles */
  .filter-row {{
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }}
  .filter-btn {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: .72rem;
    color: var(--text);
    cursor: pointer;
    transition: all .15s;
    font-family: inherit;
  }}
  .filter-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
  .filter-btn.active {{
    background: var(--accent);
    border-color: var(--accent);
    color: #0f1117;
    font-weight: 600;
  }}

  /* ── Map ─────────────────────────────────────────────────── */
  #map {{
    flex: 1;
    background: #0d1117;
  }}

  /* Leaflet overrides */
  .leaflet-popup-content-wrapper {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0 !important;
  }}
  .leaflet-popup-content {{
    margin: 0 !important;
    width: 260px !important;
  }}
  .leaflet-popup-tip {{ background: var(--surface) !important; }}
  .leaflet-control-layers {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
  }}
  .leaflet-control-layers label {{ color: var(--text) !important; }}
  .leaflet-control-zoom a {{
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
  }}

  /* Popup content */
  .popup-header {{
    background: var(--surface2);
    padding: 10px 14px;
    border-radius: var(--radius) var(--radius) 0 0;
    border-bottom: 1px solid var(--border);
  }}
  .popup-header .pid {{
    font-size: .78rem;
    font-weight: 700;
    color: var(--accent);
    font-family: monospace;
  }}
  .popup-header .cat-badge {{
    display: inline-block;
    margin-top: 4px;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: .65rem;
    font-weight: 600;
    color: #0f1117;
  }}
  .popup-body {{
    padding: 10px 14px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 12px;
  }}
  .popup-row {{ grid-column: 1 / -1; display: flex; justify-content: space-between; align-items: center; }}
  .popup-row.half {{ grid-column: auto; display: flex; flex-direction: column; }}
  .popup-row .key {{ font-size: .66rem; color: var(--muted); }}
  .popup-row .val {{ font-size: .8rem; font-weight: 600; }}
  .popup-ndvi {{
    grid-column: 1 / -1;
    background: var(--surface2);
    border-radius: 8px;
    padding: 8px 10px;
    margin-top: 2px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .popup-ndvi .big-val {{
    font-size: 1.5rem;
    font-weight: 800;
    line-height: 1;
  }}
  .popup-ndvi .ndvi-details {{ font-size: .68rem; color: var(--muted); line-height: 1.6; }}

  /* Highlight ring */
  .highlighted-plant {{
    animation: pulse 1.5s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ stroke-opacity: 1; stroke-width: 3; }}
    50%       {{ stroke-opacity: .4; stroke-width: 6; }}
  }}

  /* Scrollbar */
  #sidebar::-webkit-scrollbar {{ width: 5px; }}
  #sidebar::-webkit-scrollbar-track {{ background: transparent; }}
  #sidebar::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
</style>
</head>
<body>

<!-- ── Top Bar ──────────────────────────────────────────────────────────────── -->
<div id="topbar">
  <div>
    <h1>🌿 Plant NDVI Health Map</h1>
    <div class="subtitle">Pineapple Plantation · {total:,} plants detected · Mean NDVI: {mean_ndvi}</div>
  </div>
  <div id="search-wrap">
    <input id="search-input" type="text" placeholder="Search plant ID…" autocomplete="off"/>
    <button id="search-btn">Find</button>
  </div>
</div>

<!-- ── Main ─────────────────────────────────────────────────────────────────── -->
<div id="main">

  <!-- Sidebar -->
  <div id="sidebar">

    <!-- KPI cards -->
    <div class="panel">
      <h3>Overview</h3>
      <div class="kpi-grid">
        <div class="kpi">
          <div class="val">{total:,}</div>
          <div class="label">Total Plants</div>
        </div>
        <div class="kpi">
          <div class="val">{mean_ndvi}</div>
          <div class="label">Mean NDVI</div>
        </div>
        <div class="kpi">
          <div class="val">{cat_counts.get("Dense Vegetation", 0) + cat_counts.get("Very Dense Vegetation", 0):,}</div>
          <div class="label">Healthy Plants</div>
        </div>
        <div class="kpi">
          <div class="val">{cat_counts.get("Bare / Stressed", 0) + cat_counts.get("Sparse Vegetation", 0):,}</div>
          <div class="label">Stressed Plants</div>
        </div>
      </div>
    </div>

    <!-- NDVI histogram -->
    <div class="panel">
      <h3>NDVI Distribution</h3>
      <div id="histogram"></div>
    </div>

    <!-- Legend / category filter -->
    <div class="panel">
      <h3>Health Categories <small style="font-weight:400;text-transform:none;">(click to toggle)</small></h3>
      <div id="legend">{legend_html}</div>
    </div>

    <!-- View toggles -->
    <div class="panel">
      <h3>Display Options</h3>
      <div class="filter-row">
        <button class="filter-btn active" id="btn-all">All Plants</button>
        <button class="filter-btn" id="btn-healthy">Healthy Only</button>
        <button class="filter-btn" id="btn-stressed">Stressed Only</button>
        <button class="filter-btn" id="btn-no-noise">Hide Noise</button>
      </div>
    </div>

    <!-- Info panel (populated on click) -->
    <div class="panel" id="info-panel" style="display:none;">
      <h3>Selected Plant</h3>
      <div id="info-content"></div>
    </div>

  </div><!-- /sidebar -->

  <!-- Map -->
  <div id="map"></div>

</div><!-- /main -->

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
// ── Data ─────────────────────────────────────────────────────────────────────
const GEOJSON = {geojson_str};
const HIST    = {hist_json};
const NDVI_COLOURS = {json.dumps(NDVI_COLOURS)};

// ── Map init ──────────────────────────────────────────────────────────────────
const map = L.map('map', {{
  center: [{center_lat:.6f}, {center_lon:.6f}],
  zoom: 18,
  maxZoom: 24,
  preferCanvas: true,
}});

const satellite = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{ attribution: 'Esri World Imagery', maxZoom: 23 }}
);
const osm = L.tileLayer(
  'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{ attribution: '© OpenStreetMap', maxZoom: 19 }}
);
satellite.addTo(map);
L.control.layers({{'Satellite': satellite, 'OpenStreetMap': osm}}).addTo(map);

// ── Build histogram ───────────────────────────────────────────────────────────
(function() {{
  const wrap  = document.getElementById('histogram');
  const bars  = document.createElement('div');
  bars.className = 'hist-bar-wrap';
  HIST.counts.forEach((cnt, i) => {{
    const bar = document.createElement('div');
    bar.className  = 'hist-bar';
    const pct = cnt / HIST.max;
    bar.style.height = Math.max(2, pct * 48) + 'px';
    // gradient colour based on bin centre
    const centre = (HIST.edges[i] + HIST.edges[i+1]) / 2;
    bar.style.background = ndviColour(Math.max(0, centre));
    bar.title = `NDVI ${{HIST.edges[i].toFixed(1)}}–${{HIST.edges[i+1].toFixed(1)}}: ${{cnt.toLocaleString()}} plants`;
    bars.appendChild(bar);
  }});
  const labels = document.createElement('div');
  labels.className = 'hist-labels';
  labels.innerHTML = '<span>-0.1</span><span>0.2</span><span>0.5</span><span>0.8</span><span>1.0</span>';
  wrap.appendChild(bars);
  wrap.appendChild(labels);
}})();

// ── Colour helper (JS mirror of Python fn) ────────────────────────────────────
function ndviColour(val) {{
  if (val === null || val === undefined) return '#888888';
  val = Math.max(0, Math.min(1, val));
  let r, g, b;
  if (val < 0.4) {{
    const t = val / 0.4;
    r = 220; g = Math.round(60 + t * 180); b = 30;
  }} else {{
    const t = (val - 0.4) / 0.6;
    r = Math.round(220 - t * 170);
    g = Math.round(200 + t * 30);
    b = Math.round(30 + t * 10);
  }}
  return `#${{r.toString(16).padStart(2,'0')}}${{g.toString(16).padStart(2,'0')}}${{b.toString(16).padStart(2,'0')}}`;
}}

// ── Build circle markers ──────────────────────────────────────────────────────
const markers    = [];          // parallel to features
const markerById = {{}};        // plant_id → marker
let   layerGroup = null;

function buildMarkers(features) {{
  if (layerGroup) map.removeLayer(layerGroup);
  layerGroup = L.layerGroup();
  markers.length = 0;
  Object.keys(markerById).forEach(k => delete markerById[k]);

  features.forEach((feat, idx) => {{
    const p    = feat.properties;
    const latlng = [feat.geometry.coordinates[1], feat.geometry.coordinates[0]];
    const col  = ndviColour(p.ndvi);
    const r    = p.is_noise ? 2 : 3;

    const m = L.circleMarker(latlng, {{
      radius:      r,
      fillColor:   col,
      color:       'rgba(0,0,0,0.4)',
      weight:      0.5,
      opacity:     1,
      fillOpacity: p.is_noise ? 0.4 : 0.85,
    }});

    m.on('click', () => showPopup(feat, m));
    m._plantId = p.plant_id;
    m._props   = p;
    layerGroup.addLayer(m);
    markers.push(m);
    markerById[p.plant_id] = m;
  }});

  layerGroup.addTo(map);
  console.log(`Rendered ${{features.length.toLocaleString()}} markers`);
}}

buildMarkers(GEOJSON.features);

// ── Popup ─────────────────────────────────────────────────────────────────────
const popup = L.popup({{ maxWidth: 280, className: 'custom-popup' }});

function showPopup(feat, marker) {{
  const p     = feat.properties;
  const col   = ndviColour(p.ndvi);
  const ndviS = p.ndvi !== null ? p.ndvi.toFixed(4) : 'N/A';
  const meanS = p.ndvi_mean !== null ? p.ndvi_mean.toFixed(4) : 'N/A';
  const maxS  = p.ndvi_max  !== null ? p.ndvi_max.toFixed(4) : 'N/A';
  const tifS  = p.tif_ndvi  !== null ? p.tif_ndvi.toFixed(4) : 'N/A';

  const html = `
    <div class="popup-header">
      <div class="pid">${{p.plant_id}}</div>
      <span class="cat-badge" style="background:${{col}}">${{p.ndvi_cat}}</span>
    </div>
    <div class="popup-body">
      <div class="popup-ndvi">
        <div class="big-val" style="color:${{col}}">${{ndviS}}</div>
        <div class="ndvi-details">
          NDVI (centre)<br/>
          Mean: <strong>${{meanS}}</strong><br/>
          Max: <strong>${{maxS}}</strong><br/>
          TIF: <strong>${{tifS}}</strong>
        </div>
      </div>
      <div class="popup-row half">
        <span class="key">Sector</span>
        <span class="val">${{p.sector || '—'}}</span>
      </div>
      <div class="popup-row half">
        <span class="key">Row / Col</span>
        <span class="val">${{p.row ?? '—'}} / ${{p.col ?? '—'}}</span>
      </div>
      <div class="popup-row half">
        <span class="key">Instance ID</span>
        <span class="val">#${{p.instance_id}}</span>
      </div>
      <div class="popup-row half">
        <span class="key">Area (px²)</span>
        <span class="val">${{p.area_px?.toLocaleString() ?? '—'}}</span>
      </div>
      <div class="popup-row">
        <span class="key">GPS</span>
        <span class="val" style="font-size:.7rem;font-family:monospace">
          ${{feat.geometry.coordinates[1].toFixed(6)}}, ${{feat.geometry.coordinates[0].toFixed(6)}}
        </span>
      </div>
    </div>`;

  popup.setLatLng([feat.geometry.coordinates[1], feat.geometry.coordinates[0]])
       .setContent(html)
       .openOn(map);

  // Also update sidebar info panel
  document.getElementById('info-panel').style.display = '';
  document.getElementById('info-content').innerHTML = `
    <p style="font-size:.75rem;color:var(--muted)">Plant: <strong style="color:var(--accent)">${{p.plant_id}}</strong></p>
    <p style="font-size:.75rem;margin-top:4px">NDVI: <strong style="color:${{col}};font-size:1rem">${{ndviS}}</strong></p>
    <p style="font-size:.72rem;color:var(--muted);margin-top:4px">${{p.ndvi_cat}}</p>`;
}}

// ── Legend / category toggle ──────────────────────────────────────────────────
const activeCats = new Set(Object.keys(NDVI_COLOURS));
const legendItems = document.querySelectorAll('.legend-item');

legendItems.forEach(item => {{
  item.addEventListener('click', () => {{
    const cat = item.dataset.cat;
    if (activeCats.has(cat)) {{
      activeCats.delete(cat);
      item.classList.add('inactive');
    }} else {{
      activeCats.add(cat);
      item.classList.remove('inactive');
    }}
    applyFilter();
  }});
}});

// ── Quick filter buttons ──────────────────────────────────────────────────────
let noiseVisible  = true;
let filterMode    = 'all';

function applyFilter() {{
  const features = GEOJSON.features.filter(f => {{
    const p = f.properties;
    if (!noiseVisible && p.is_noise) return false;
    if (!activeCats.has(p.ndvi_cat)) return false;
    if (filterMode === 'healthy') {{
      return p.ndvi_cat === 'Dense Vegetation' || p.ndvi_cat === 'Very Dense Vegetation';
    }}
    if (filterMode === 'stressed') {{
      return p.ndvi_cat === 'Bare / Stressed' || p.ndvi_cat === 'Sparse Vegetation';
    }}
    return true;
  }});
  buildMarkers(features);
}}

document.getElementById('btn-all').addEventListener('click', () => {{
  filterMode = 'all'; noiseVisible = true;
  setBtn('btn-all'); applyFilter();
}});
document.getElementById('btn-healthy').addEventListener('click', () => {{
  filterMode = 'healthy'; setBtn('btn-healthy'); applyFilter();
}});
document.getElementById('btn-stressed').addEventListener('click', () => {{
  filterMode = 'stressed'; setBtn('btn-stressed'); applyFilter();
}});
document.getElementById('btn-no-noise').addEventListener('click', () => {{
  noiseVisible = !noiseVisible;
  document.getElementById('btn-no-noise').classList.toggle('active', !noiseVisible);
  applyFilter();
}});

function setBtn(id) {{
  ['btn-all','btn-healthy','btn-stressed'].forEach(b => {{
    document.getElementById(b).classList.toggle('active', b === id);
  }});
}}

// ── Search ────────────────────────────────────────────────────────────────────
function doSearch() {{
  const q = document.getElementById('search-input').value.trim().toUpperCase();
  if (!q) return;
  const m = markerById[q] || Object.values(markerById).find(m => m._plantId.toUpperCase().includes(q));
  if (m) {{
    map.setView(m.getLatLng(), 21);
    m.setStyle({{ color: '#ffffff', weight: 3 }});
    setTimeout(() => m.setStyle({{ color: 'rgba(0,0,0,0.4)', weight: 0.5 }}), 2500);
    // find feature
    const feat = GEOJSON.features.find(f => f.properties.plant_id === m._plantId);
    if (feat) showPopup(feat, m);
  }} else {{
    alert('Plant ID not found: ' + q);
  }}
}}

document.getElementById('search-btn').addEventListener('click', doSearch);
document.getElementById('search-input').addEventListener('keydown', e => {{
  if (e.key === 'Enter') doSearch();
}});
</script>
</body>
</html>
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = out_path.stat().st_size / 1e6
    print(f"[INFO] Map saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"[INFO] Open in browser:  open \"{out_path}\"")
    print("[✓] Step 5 complete\n")

    return out_path


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from step1_load_plants import load_plants
    from step2_match_ndvi  import match_plants_to_ndvi
    from step3_raster_sample import validate_and_enrich_from_tif

    plants   = load_plants(include_noise=True)
    enriched = match_plants_to_ndvi(plants, window=1)
    validated, _ = validate_and_enrich_from_tif(enriched, window_size=3)
    build_map(validated)
