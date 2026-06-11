#!/usr/bin/env python3
"""
Step 2 — Match Plants to NDVI Values (Fixed)
==============================================
Root cause of v1 bug:
  - plant pixel_x/pixel_y are from the FULL orthomosaic (14929×22012 px)
  - NDVI CSV pixel_row/col are from the NDRE TIF (3693×1918 px, 0.05m/px)
  - The two pixel spaces are DIFFERENT — direct indexing fails

Fix:
  Use plant UTM geo_x/geo_y (EPSG:32647) to compute the correct pixel
  position in the NDRE TIF coordinate system using the TIF's geotransform.

Strategy (pixel-coordinate lookup — fast & exact):
  1. Build a 2-D numpy array: ndvi_grid[row][col] = ndvi_value
     (indexed by NDRE TIF pixel row/col from CSV)
  2. For each plant, compute its NDRE TIF pixel:
       col = (geo_x - tif_left)  / pixel_size
       row = (tif_top  - geo_y)  / pixel_size
  3. Sample ndvi_grid with a small window for robustness
"""

import csv
import math
import time
import json
from pathlib import Path

import numpy as np
import rasterio

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
NDVI_CSV = BASE_DIR / "extracted_ndvi_data.csv"
NDRE_TIF = BASE_DIR / "Task-of-2026-06-05T095436120Z-orthophoto-NDRE (1).tif"


# ── NDVI category thresholds (pineapple-specific) ─────────────────────────────
def ndvi_to_category(val: float) -> str:
    """
    Pineapple NDVI health classification (early/mid-season).
    Thresholds calibrated for UAV-based pineapple plantation monitoring.
    """
    if val is None or math.isnan(val):
        return "No Data"
    if val < 0.2:
        return "Bare Soil"          # replanting required
    if val < 0.6:
        return "Unhealthy"          # immediate scouting needed
    if val < 0.8:
        return "Healthy"            # maintain protocols
    return "Saturated"              # use NDRE for accurate reading


# ── Read TIF geotransform ──────────────────────────────────────────────────────
def get_tif_geotransform(tif_path: Path) -> dict:
    """Extract the geotransform parameters from the NDRE TIF."""
    with rasterio.open(tif_path) as src:
        return {
            "left":      src.bounds.left,
            "top":       src.bounds.top,
            "right":     src.bounds.right,
            "bottom":    src.bounds.bottom,
            "pixel_w":   src.transform.a,   # +ve x resolution (metres/pixel)
            "pixel_h":   abs(src.transform.e),  # +ve y resolution
            "nrows":     src.height,
            "ncols":     src.width,
        }


# ── Build pixel lookup from NDVI CSV ──────────────────────────────────────────
def build_ndvi_grid_from_csv(csv_path: Path, nrows: int, ncols: int):
    """
    Read the NDVI CSV and build a 2-D numpy array indexed by (pixel_row, pixel_col).
    pixel_row/col in the CSV correspond to the NDRE TIF pixel coordinates.

    Returns:
        ndvi_grid  – float32 array shape (nrows, ncols), NaN = no data
        lat_grid   – float64 array for verification
        lon_grid   – float64 array for verification
    """
    print(f"[INFO] Loading NDVI CSV: {csv_path}")
    print(f"[INFO] Allocating {nrows}×{ncols} grid…")

    ndvi_grid = np.full((nrows, ncols), np.nan, dtype=np.float32)
    lat_grid  = np.full((nrows, ncols), np.nan, dtype=np.float64)
    lon_grid  = np.full((nrows, ncols), np.nan, dtype=np.float64)

    t0 = time.time()
    filled = 0
    masked_nodata = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            r = int(row["pixel_row"])
            c = int(row["pixel_col"])
            if 0 <= r < nrows and 0 <= c < ncols:
                dn   = float(row["ndvi_raw_dn"])
                ndvi = float(row["ndvi"])
                # DN=0 is nodata in the NDVI raster (maps to ndvi=-1.0)
                # Skip these — they are background/nodata pixels, not true -1 NDVI
                if dn == 0 or ndvi <= -1.0:
                    masked_nodata += 1
                    continue
                ndvi_grid[r, c] = ndvi
                lat_grid[r, c]  = float(row["latitude"])
                lon_grid[r, c]  = float(row["longitude"])
                filled += 1
            if (i + 1) % 1_000_000 == 0:
                print(f"  …read {i+1:,} rows")

    elapsed = time.time() - t0
    valid   = int(np.sum(~np.isnan(ndvi_grid)))
    print(f"[INFO] Grid built in {elapsed:.1f}s  — {valid:,} valid pixels ({masked_nodata:,} nodata DN=0 masked)")
    print(f"[INFO] NDVI range: {np.nanmin(ndvi_grid):.4f} – {np.nanmax(ndvi_grid):.4f}")
    return ndvi_grid, lat_grid, lon_grid


# ── geo_x/geo_y → NDRE TIF pixel ─────────────────────────────────────────────
def utm_to_tif_pixel(geo_x: float, geo_y: float, gt: dict) -> tuple[int, int]:
    """
    Convert plant centroid UTM coordinates (EPSG:32647) to NDRE TIF pixel.

    Returns (row, col) — integer pixel indices in the NDRE TIF grid.
    """
    col = (geo_x - gt["left"]) / gt["pixel_w"]
    row = (gt["top"] - geo_y)  / gt["pixel_h"]
    return int(round(row)), int(round(col))


# ── Sample NDVI for one plant ─────────────────────────────────────────────────
def sample_ndvi(
    ndvi_grid: np.ndarray,
    lat_grid:  np.ndarray,
    lon_grid:  np.ndarray,
    tif_row:   int,
    tif_col:   int,
    window:    int = 1,
) -> dict:
    """
    Sample NDVI at NDRE TIF pixel (tif_row, tif_col).
    window=1 → 3×3 patch, window=2 → 5×5 patch.
    """
    nrows, ncols = ndvi_grid.shape

    r_lo = max(0,     tif_row - window)
    r_hi = min(nrows, tif_row + window + 1)
    c_lo = max(0,     tif_col - window)
    c_hi = min(ncols, tif_col + window + 1)

    patch = ndvi_grid[r_lo:r_hi, c_lo:c_hi]
    valid_vals = patch[~np.isnan(patch)]

    # Centre pixel value
    center_val = None
    if 0 <= tif_row < nrows and 0 <= tif_col < ncols:
        v = float(ndvi_grid[tif_row, tif_col])
        if not math.isnan(v):
            center_val = round(v, 6)

    if len(valid_vals) == 0:
        return {
            "ndvi":                None,
            "ndvi_mean":           None,
            "ndvi_max":            None,
            "ndvi_category":       "No Data",
            "ndvi_pixels_sampled": 0,
            "tif_row":             tif_row,
            "tif_col":             tif_col,
            "ndvi_matched_lat":    None,
            "ndvi_matched_lon":    None,
        }

    ndvi_mean = round(float(np.mean(valid_vals)), 6)
    ndvi_max  = round(float(np.max(valid_vals)),  6)
    ndvi_val  = center_val if center_val is not None else ndvi_mean

    # Matched lat/lon from grid
    ml = float(lat_grid[tif_row, tif_col]) if (0 <= tif_row < nrows and 0 <= tif_col < ncols) else None
    mln = float(lon_grid[tif_row, tif_col]) if (0 <= tif_row < nrows and 0 <= tif_col < ncols) else None

    return {
        "ndvi":                ndvi_val,
        "ndvi_mean":           ndvi_mean,
        "ndvi_max":            ndvi_max,
        "ndvi_category":       ndvi_to_category(ndvi_val),
        "ndvi_pixels_sampled": int(len(valid_vals)),
        "tif_row":             tif_row,
        "tif_col":             tif_col,
        "ndvi_matched_lat":    round(ml, 6)  if ml  is not None and not math.isnan(ml)  else None,
        "ndvi_matched_lon":    round(mln, 6) if mln is not None and not math.isnan(mln) else None,
    }


# ── Main matching function ─────────────────────────────────────────────────────
def match_plants_to_ndvi(
    plants:   list[dict],
    window:   int  = 1,
    csv_path: Path = NDVI_CSV,
    tif_path: Path = NDRE_TIF,
) -> list[dict]:
    """
    Enrich each plant dict with NDVI values using UTM→TIF pixel mapping.

    Parameters
    ----------
    plants   : list of plant dicts from step1_load_plants
    window   : sampling window half-size (1 → 3×3 patch)
    csv_path : path to extracted_ndvi_data.csv
    tif_path : path to NDRE TIF (used to read geotransform)
    """
    print("=" * 60)
    print("STEP 2 — Match Plants → NDVI  (geo_x/geo_y → TIF pixel)")
    print("=" * 60)

    # 1. Read TIF geotransform so we know how to map UTM → pixel
    print(f"[INFO] Reading TIF geotransform: {tif_path.name}")
    gt = get_tif_geotransform(tif_path)
    print(f"[INFO] TIF bounds: left={gt['left']:.2f}, top={gt['top']:.2f}, "
          f"right={gt['right']:.2f}, bottom={gt['bottom']:.2f}")
    print(f"[INFO] TIF pixel size: {gt['pixel_w']:.4f}m × {gt['pixel_h']:.4f}m")
    print(f"[INFO] TIF grid: {gt['nrows']} rows × {gt['ncols']} cols")

    # 2. Build the NDVI pixel grid from CSV (indexed by NDRE TIF pixel coords)
    ndvi_grid, lat_grid, lon_grid = build_ndvi_grid_from_csv(
        csv_path, gt["nrows"], gt["ncols"]
    )

    # 3. Coverage check
    in_bounds = [
        p for p in plants
        if gt["left"] <= p["geo_x"] <= gt["right"]
        and gt["bottom"] <= p["geo_y"] <= gt["top"]
    ]
    out_of_bounds = len(plants) - len(in_bounds)
    print(f"\n[INFO] Plants within NDRE TIF bounds: {len(in_bounds):,}/{len(plants):,}")
    if out_of_bounds:
        print(f"[WARN] {out_of_bounds:,} plants outside NDRE TIF extent (no NDVI data for them)")

    # 4. Match each plant
    print(f"\n[INFO] Matching {len(plants):,} plants  (window={window}, patch={(2*window+1)}×{(2*window+1)})…")
    t0 = time.time()

    enriched      = []
    no_data_count = 0

    for i, p in enumerate(plants):
        # Convert UTM → NDRE TIF pixel
        tif_row, tif_col = utm_to_tif_pixel(p["geo_x"], p["geo_y"], gt)

        result = sample_ndvi(
            ndvi_grid, lat_grid, lon_grid,
            tif_row, tif_col,
            window=window,
        )

        ep = dict(p)
        ep.update(result)

        if result["ndvi"] is None:
            no_data_count += 1

        enriched.append(ep)

        if (i + 1) % 2000 == 0:
            print(f"  …matched {i+1:,}/{len(plants):,}")

    elapsed = time.time() - t0
    matched = len(enriched) - no_data_count
    print(f"\n[RESULT] Matched : {matched:,}/{len(enriched):,} plants  ({elapsed:.1f}s)")
    print(f"[RESULT] No data : {no_data_count:,} plants")

    # Category summary
    from collections import Counter
    cats = Counter(p["ndvi_category"] for p in enriched)
    print("\n[RESULT] NDVI Category breakdown:")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        pct = cnt / len(enriched) * 100
        print(f"   {cat:<40} {cnt:>6,}  ({pct:.1f}%)")

    ndvi_vals = [p["ndvi"] for p in enriched if p["ndvi"] is not None]
    if ndvi_vals:
        mean_v = sum(ndvi_vals) / len(ndvi_vals)
        sv = sorted(ndvi_vals)
        mid = len(sv) // 2
        med = sv[mid] if len(sv) % 2 else (sv[mid-1] + sv[mid]) / 2
        print(f"\n[RESULT] NDVI mean:   {mean_v:.4f}")
        print(f"[RESULT] NDVI median: {med:.4f}")
        print(f"[RESULT] NDVI min:    {min(ndvi_vals):.4f}")
        print(f"[RESULT] NDVI max:    {max(ndvi_vals):.4f}")

    print("[✓] Step 2 complete\n")
    return enriched


if __name__ == "__main__":
    from step1_load_plants import load_plants
    plants   = load_plants(include_noise=True)
    enriched = match_plants_to_ndvi(plants, window=1)
    # Show a valid match
    for p in enriched:
        if p["ndvi"] is not None:
            print(json.dumps(p, indent=2))
            break
