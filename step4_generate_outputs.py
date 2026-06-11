#!/usr/bin/env python3
"""
Step 4 — Generate Output Files
================================
Writes:
  • plants_ndvi_enriched.json  — complete enriched plant data
  • plants_ndvi_summary.csv    — tabular summary for spreadsheets
  • pipeline_stats.json        — run statistics and metadata
"""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
OUT_DIR    = Path(__file__).parent / "outputs"

OUT_JSON   = OUT_DIR / "plants_ndvi_enriched.json"
OUT_CSV    = OUT_DIR / "plants_ndvi_summary.csv"
OUT_STATS  = OUT_DIR / "pipeline_stats.json"

# CSV columns — ordered for readability
CSV_COLUMNS = [
    "plant_id",
    "instance_id",
    "sector_label",
    "row_index",
    "col_index",
    "lat",
    "lon",
    "geo_x",
    "geo_y",
    "pixel_x",
    "pixel_y",
    "area_px",
    "ndvi",
    "ndvi_mean",
    "ndvi_max",
    "ndvi_category",
    "ndvi_pixels_sampled",
    "tif_ndvi",
    "tif_ndvi_category",
    "ndvi_matched_lat",
    "ndvi_matched_lon",
    "is_noise",
]


def save_outputs(
    plants: list[dict],
    validation_stats: dict | None = None,
    run_meta: dict | None = None,
) -> dict:
    """
    Write all output files.

    Parameters
    ----------
    plants           : enriched plant list from step2 / step3
    validation_stats : optional stats dict from step3
    run_meta         : any extra metadata to embed (e.g., pipeline config)

    Returns
    -------
    paths dict  —  {'json': ..., 'csv': ..., 'stats': ...}
    """
    print("=" * 60)
    print("STEP 4 — Save Outputs")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 4a. Enriched JSON ──────────────────────────────────────────────────────
    print(f"[INFO] Writing JSON ({len(plants):,} plants)…")
    t0 = time.time()
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(plants, f, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON saved: {OUT_JSON}  ({OUT_JSON.stat().st_size / 1e6:.1f} MB, {time.time()-t0:.1f}s)")

    # ── 4b. Summary CSV ────────────────────────────────────────────────────────
    print(f"\n[INFO] Writing CSV…")
    t0 = time.time()

    # Determine available columns (some optional fields may not exist)
    available = set(plants[0].keys()) if plants else set()
    cols = [c for c in CSV_COLUMNS if c in available]
    extra = sorted(available - set(CSV_COLUMNS))
    cols += extra  # append any unexpected extra fields

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(plants)
    print(f"[INFO] CSV saved:  {OUT_CSV}  ({OUT_CSV.stat().st_size / 1e6:.1f} MB, {time.time()-t0:.1f}s)")

    # ── 4c. Pipeline stats JSON ────────────────────────────────────────────────
    total      = len(plants)
    valid      = sum(1 for p in plants if not p.get("is_noise"))
    noise      = sum(1 for p in plants if p.get("is_noise"))
    matched    = sum(1 for p in plants if p.get("ndvi") is not None)
    no_data    = total - matched

    ndvi_vals = [p["ndvi"] for p in plants if p.get("ndvi") is not None]
    mean_ndvi = round(sum(ndvi_vals) / len(ndvi_vals), 6) if ndvi_vals else None
    min_ndvi  = round(min(ndvi_vals), 6) if ndvi_vals else None
    max_ndvi  = round(max(ndvi_vals), 6) if ndvi_vals else None

    sorted_v  = sorted(ndvi_vals)
    mid       = len(sorted_v) // 2
    median    = round(
        sorted_v[mid] if len(sorted_v) % 2 else (sorted_v[mid-1] + sorted_v[mid]) / 2,
        6,
    ) if sorted_v else None

    from collections import Counter
    cat_counts = dict(Counter(p.get("ndvi_category", "No Data") for p in plants))

    stats = {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "total_plants":      total,
        "valid_plants":      valid,
        "noise_plants":      noise,
        "matched_plants":    matched,
        "no_data_plants":    no_data,
        "ndvi_stats": {
            "mean":   mean_ndvi,
            "median": median,
            "min":    min_ndvi,
            "max":    max_ndvi,
        },
        "ndvi_by_category":  cat_counts,
        "validation":        validation_stats or {},
        "run_meta":          run_meta or {},
        "output_files": {
            "json": str(OUT_JSON),
            "csv":  str(OUT_CSV),
        },
    }

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Stats saved: {OUT_STATS}")

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE OUTPUT SUMMARY")
    print("=" * 60)
    print(f"  Total plants:    {total:>8,}")
    print(f"  Valid plants:    {valid:>8,}")
    print(f"  Noise plants:    {noise:>8,}")
    print(f"  NDVI matched:    {matched:>8,}")
    print(f"  No NDVI data:    {no_data:>8,}")
    if ndvi_vals:
        print(f"\n  Mean NDVI:       {mean_ndvi:>8.4f}")
        print(f"  Median NDVI:     {median:>8.4f}")
        print(f"  Min NDVI:        {min_ndvi:>8.4f}")
        print(f"  Max NDVI:        {max_ndvi:>8.4f}")
    print("\n  NDVI Categories:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        print(f"    {cat:<35} {cnt:>6,}  ({pct:.1f}%)")

    print("[✓] Step 4 complete\n")

    return {
        "json":  str(OUT_JSON),
        "csv":   str(OUT_CSV),
        "stats": str(OUT_STATS),
    }


if __name__ == "__main__":
    from step1_load_plants import load_plants
    from step2_match_ndvi  import match_plants_to_ndvi
    from step3_raster_sample import validate_and_enrich_from_tif

    plants   = load_plants(include_noise=True)
    enriched = match_plants_to_ndvi(plants, window=1)
    validated, val_stats = validate_and_enrich_from_tif(enriched, window_size=3)
    save_outputs(validated, validation_stats=val_stats)
