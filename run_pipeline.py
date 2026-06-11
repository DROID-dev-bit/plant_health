#!/usr/bin/env python3
"""
run_pipeline.py — Main Entry Point
====================================
Runs all 5 pipeline steps end-to-end.

Usage:
    cd /Users/nithints/Drone_imagery_file
    .venv/bin/python plant_ndvi_pipeline/run_pipeline.py

Outputs (in plant_ndvi_pipeline/outputs/):
    plants_ndvi_enriched.json   — full enriched plant data
    plants_ndvi_summary.csv     — tabular summary
    pipeline_stats.json         — run statistics
    plant_ndvi_map.html         — interactive map (open in browser)
"""

import sys
import time
import json
from pathlib import Path

# Make sure siblings are importable
sys.path.insert(0, str(Path(__file__).parent))

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    # Include NOISE-labelled plants (edge detections) with is_noise=True flag
    "include_noise": True,

    # NDVI sampling window half-size (1 = 3×3 patch around centroid)
    "ndvi_window": 1,

    # TIF validation window size (3 = 3×3 around centroid)
    "tif_window": 3,
}

def main():
    t_start = time.time()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Plant NDVI Pipeline — Full Run                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nConfig: {json.dumps(CONFIG, indent=2)}\n")

    # ── Step 1: Load plants ───────────────────────────────────────────────────
    from step1_load_plants import load_plants
    plants = load_plants(include_noise=CONFIG["include_noise"])

    # ── Step 2: Match NDVI ────────────────────────────────────────────────────
    from step2_match_ndvi import match_plants_to_ndvi
    enriched = match_plants_to_ndvi(plants, window=CONFIG["ndvi_window"])

    # ── Step 3: TIF validation ────────────────────────────────────────────────
    from step3_raster_sample import validate_and_enrich_from_tif
    validated, val_stats = validate_and_enrich_from_tif(
        enriched, window_size=CONFIG["tif_window"]
    )

    # ── Step 4: Save outputs ──────────────────────────────────────────────────
    from step4_generate_outputs import save_outputs
    out_paths = save_outputs(
        validated,
        validation_stats=val_stats,
        run_meta=CONFIG,
    )

    # ── Step 5: Generate map ──────────────────────────────────────────────────
    from step5_generate_map import build_map
    map_path = build_map(validated)

    # ── Step 6: ML Training (optional - comment out if models exist) ─────────
    print("\n[INFO] Running ML training pipeline...")
    from step6_ml_training import main as train_ml
    try:
        ml_summary = train_ml()
    except Exception as e:
        print(f"[WARN] ML training failed: {e}")
        print("[WARN] Skipping ML predictions...")
        ml_summary = None

    # ── Step 7: ML Predictions ────────────────────────────────────────────────
    if ml_summary:
        print("\n[INFO] Generating ML predictions and insights...")
        from step7_ml_predictions import main as predict_ml
        try:
            predict_ml()
        except Exception as e:
            print(f"[WARN] ML predictions failed: {e}")

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PIPELINE COMPLETE (with ML)                             ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Total time: {mins}m {secs}s")
    print("║")
    print(f"║  📄 JSON:  {out_paths['json']}")
    print(f"║  📊 CSV:   {out_paths['csv']}")
    print(f"║  📈 Stats: {out_paths['stats']}")
    print(f"║  🗺  Map:   {map_path}")
    print("║")
    print("║  🤖 ML Outputs (outputs/ml_models/):")
    print("║     • Trained models for growth, yield, nitrogen")
    print("║     • plants_with_predictions.json")
    print("║     • ml_insights_report.json")
    print("║     • high_priority_plants.csv")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  To view the map:                                        ║")
    print(f'║    open "{map_path}"')
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
