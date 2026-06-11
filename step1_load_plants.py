#!/usr/bin/env python3
"""
Step 1 — Load & Prepare Plant Data
====================================
Reads plants_with_ids.json (primary) and plants_points.geojson (cross-validation).
Converts UTM EPSG:32647 geo_x/geo_y → WGS84 lat/lon.
Returns a list of enriched plant dicts ready for NDVI matching.
"""

import json
from pathlib import Path
from pyproj import Transformer

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent          # Drone_imagery_file/
PREV_TEAM   = BASE_DIR / "outputs 2" / "03_ids"

PLANTS_JSON    = PREV_TEAM / "plants_with_ids.json"
PLANTS_GEOJSON = PREV_TEAM / "plants_points.geojson"

# ── CRS transformer ───────────────────────────────────────────────────────────
_transformer = Transformer.from_crs("EPSG:32647", "EPSG:4326", always_xy=True)

def utm_to_wgs84(geo_x: float, geo_y: float):
    """Convert UTM Zone 47N (EPSG:32647) → WGS84 (lat, lon)."""
    lon, lat = _transformer.transform(geo_x, geo_y)
    return round(lat, 8), round(lon, 8)


def load_plants(include_noise: bool = True) -> list[dict]:
    """
    Load all plant records from plants_with_ids.json.

    Parameters
    ----------
    include_noise : bool
        If True, noise-labelled plants are included with a `is_noise=True` flag.
        If False, they are excluded from the output list.

    Returns
    -------
    list[dict]  – plant records with added `lat`, `lon`, `is_noise` fields.
    """
    print("=" * 60)
    print("STEP 1 — Load Plant Data")
    print("=" * 60)

    print(f"[INFO] Reading: {PLANTS_JSON}")
    with open(PLANTS_JSON, encoding="utf-8") as f:
        raw_plants = json.load(f)
    print(f"[INFO] Loaded {len(raw_plants):,} plant records")

    plants = []
    noise_count = 0

    for p in raw_plants:
        is_noise = str(p.get("sector_label", "")).upper() == "NOISE"
        if is_noise:
            noise_count += 1
        if is_noise and not include_noise:
            continue

        lat, lon = utm_to_wgs84(p["geo_x"], p["geo_y"])

        plants.append({
            # identity
            "plant_id":     p["plant_id"],
            "instance_id":  p["instance_id"],
            # spatial — original UTM
            "geo_x":        p["geo_x"],
            "geo_y":        p["geo_y"],
            # spatial — WGS84
            "lat":          lat,
            "lon":          lon,
            # pixel coords in orthomosaic
            "pixel_x":      p["pixel_x"],
            "pixel_y":      p["pixel_y"],
            # bounding box
            "bbox_x":       p.get("bbox_x"),
            "bbox_y":       p.get("bbox_y"),
            "bbox_width":   p.get("bbox_width"),
            "bbox_height":  p.get("bbox_height"),
            # size
            "area_px":      p.get("area_px"),
            # organisation
            "sector_id":    p.get("sector_id"),
            "sector_label": p.get("sector_label"),
            "row_index":    p.get("row_index"),
            "col_index":    p.get("col_index"),
            # flag
            "is_noise":     is_noise,
        })

    valid = sum(1 for p in plants if not p["is_noise"])
    print(f"[INFO] Valid plants  : {valid:,}")
    print(f"[INFO] Noise plants  : {noise_count:,} ({'included' if include_noise else 'excluded'})")
    print(f"[INFO] Total returned: {len(plants):,}")

    # Coordinate sanity check
    lats = [p["lat"] for p in plants]
    lons = [p["lon"] for p in plants]
    print(f"\n[INFO] Lat range: {min(lats):.6f} – {max(lats):.6f}")
    print(f"[INFO] Lon range: {min(lons):.6f} – {max(lons):.6f}")
    print("[✓] Step 1 complete\n")

    return plants


if __name__ == "__main__":
    plants = load_plants(include_noise=True)
    print(f"\nSample plant:\n{json.dumps(plants[0], indent=2)}")
