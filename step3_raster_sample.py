#!/usr/bin/env python3
"""
Step 3 — Direct Raster Sampling (Validation & Enrichment)
===========================================================
Samples the NDRE TIF directly using rasterio for each plant location.
This validates that the CSV-derived NDVI values are consistent and
also provides a secondary NDVI source from the raw raster.

The TIF is in EPSG:32647 so we can use plant's geo_x/geo_y directly.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
NDRE_TIF = BASE_DIR / "Task-of-2026-06-05T095436120Z-orthophoto-NDRE (1).tif"

# ── NDVI category (same thresholds as step2) ──────────────────────────────────
def ndvi_to_category(val: float) -> str:
    if val is None or math.isnan(val):
        return "No Data"
    if val < 0.0:
        return "Water / Non-vegetation"
    if val < 0.2:
        return "Bare / Stressed"
    if val < 0.4:
        return "Sparse Vegetation"
    if val < 0.6:
        return "Moderate Vegetation"
    if val < 0.8:
        return "Dense Vegetation"
    return "Very Dense Vegetation"


def sample_tif_at_utm(
    src: rasterio.DatasetReader,
    geo_x: float,
    geo_y: float,
    window_size: int = 3,
    nodata: float = -9999.0,
) -> dict:
    """
    Sample a rasterio dataset at a UTM coordinate.
    Returns NDVI value from the raster band.

    Parameters
    ----------
    src         : open rasterio dataset
    geo_x/geo_y : plant centroid in the dataset's CRS (EPSG:32647)
    window_size : square window to average (1 = single pixel, 3 = 3×3)
    """
    # Check if point is within raster bounds
    if not (src.bounds.left <= geo_x <= src.bounds.right and
            src.bounds.bottom <= geo_y <= src.bounds.top):
        return {"tif_ndvi": None, "tif_ndvi_category": "Out of Bounds"}

    # Convert UTM → pixel
    row, col = src.index(geo_x, geo_y)
    half = window_size // 2

    r_lo = max(0, row - half)
    r_hi = min(src.height, row + half + 1)
    c_lo = max(0, col - half)
    c_hi = min(src.width,  col + half + 1)

    if r_lo >= r_hi or c_lo >= c_hi:
        return {"tif_ndvi": None, "tif_ndvi_category": "No Data"}

    win  = Window(c_lo, r_lo, c_hi - c_lo, r_hi - r_lo)
    data = src.read(1, window=win).astype(np.float32)

    # Mask nodata — handle None nodata gracefully
    if nodata is not None and not math.isnan(nodata):
        data[data == nodata] = np.nan
    # Also mask extreme values
    data[data < -1.5] = np.nan
    data[data >  1.5] = np.nan

    valid = data[~np.isnan(data)]

    if len(valid) == 0:
        return {"tif_ndvi": None, "tif_ndvi_category": "No Data"}

    val = round(float(np.mean(valid)), 6)
    return {
        "tif_ndvi":          val,
        "tif_ndvi_category": ndvi_to_category(val),
    }


def validate_and_enrich_from_tif(
    plants: list[dict],
    window_size: int = 3,
    tif_path: Path = NDRE_TIF,
) -> tuple[list[dict], dict]:
    """
    For each plant, sample the NDRE TIF directly and append `tif_ndvi` field.
    Also computes agreement statistics between CSV-derived and TIF-derived NDVI.

    Returns
    -------
    enriched_plants : list[dict]  — plants with `tif_ndvi` added
    stats           : dict        — validation statistics
    """
    print("=" * 60)
    print("STEP 3 — Direct TIF Sampling (Validation)")
    print("=" * 60)

    if not tif_path.exists():
        print(f"[WARN] TIF not found: {tif_path}  — skipping TIF validation")
        return plants, {}

    print(f"[INFO] Opening raster: {tif_path.name}")
    t0 = time.time()

    enriched = []
    diffs    = []

    with rasterio.open(tif_path) as src:
        print(f"[INFO] Raster CRS: {src.crs}")
        print(f"[INFO] Raster shape: {src.height} × {src.width}")
        print(f"[INFO] Raster nodata: {src.nodata}")
        print(f"[INFO] Sampling {len(plants):,} plant locations…")

        for i, p in enumerate(plants):
            result = sample_tif_at_utm(
                src,
                p["geo_x"],
                p["geo_y"],
                window_size=window_size,
                nodata=float(src.nodata) if src.nodata is not None else -9999.0,
            )
            ep = dict(p)
            ep.update(result)

            # Track agreement between CSV-NDVI and TIF-NDVI
            if ep.get("ndvi") is not None and result.get("tif_ndvi") is not None:
                diffs.append(abs(ep["ndvi"] - result["tif_ndvi"]))

            enriched.append(ep)

            if (i + 1) % 2000 == 0:
                print(f"  …sampled {i+1:,}/{len(plants):,}")

    elapsed = time.time() - t0
    print(f"\n[INFO] TIF sampling done in {elapsed:.1f}s")

    stats = {}
    if diffs:
        mean_diff = sum(diffs) / len(diffs)
        max_diff  = max(diffs)
        agreed    = sum(1 for d in diffs if d < 0.05)
        stats = {
            "samples_compared": len(diffs),
            "mean_abs_diff":    round(mean_diff, 6),
            "max_abs_diff":     round(max_diff, 6),
            "pct_within_0.05":  round(agreed / len(diffs) * 100, 2),
        }
        print(f"[VALIDATION] Compared {len(diffs):,} plants")
        print(f"[VALIDATION] Mean |NDVI_csv - NDVI_tif|: {mean_diff:.4f}")
        print(f"[VALIDATION] Max  |diff|:                {max_diff:.4f}")
        print(f"[VALIDATION] Within 0.05 tolerance:      {agreed:,} / {len(diffs):,}  ({stats['pct_within_0.05']}%)")
    else:
        print("[WARN] No comparable NDVI pairs found — cannot validate")

    print("[✓] Step 3 complete\n")
    return enriched, stats


if __name__ == "__main__":
    from step1_load_plants import load_plants
    from step2_match_ndvi  import match_plants_to_ndvi

    plants   = load_plants(include_noise=True)
    enriched = match_plants_to_ndvi(plants, window=1)
    validated, stats = validate_and_enrich_from_tif(enriched[:100], window_size=3)

    print("\nSample plant (first non-noise):")
    for p in validated:
        if not p["is_noise"]:
            print(json.dumps(p, indent=2))
            break
    print("\nValidation stats:", json.dumps(stats, indent=2))
