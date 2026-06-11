#!/usr/bin/env python3
"""
run_ml_pipeline.py — ML Training & Predictions Only
====================================================
Runs ML training and prediction steps (6-7) independently.
Use this if you've already run steps 1-5 and have the enriched data.

Usage:
    cd /Users/nithints/Drone_imagery_file
    .venv/bin/python plant_ndvi_pipeline/run_ml_pipeline.py

    # Or just training:
    .venv/bin/python plant_ndvi_pipeline/step6_ml_training.py

    # Or just predictions (requires trained models):
    .venv/bin/python plant_ndvi_pipeline/step7_ml_predictions.py

Outputs (in plant_ndvi_pipeline/outputs/ml_models/):
    Growth Stage Models (Random Forest, XGBoost, LightGBM)
    Yield Prediction Models (Random Forest, XGBoost, LightGBM)
    Nitrogen Deficiency Models (Random Forest, XGBoost, LightGBM)
    
    plants_with_predictions.json    — All plants with ML predictions
    ml_insights_report.json         — Actionable insights & recommendations
    sector_analysis.csv             — Performance by sector
    high_priority_plants.csv        — Plants requiring attention
"""

import sys
import time
from pathlib import Path

# Make sure siblings are importable
sys.path.insert(0, str(Path(__file__).parent))


def main():
    t_start = time.time()
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "ML PIPELINE — Full Run (Steps 6-7)" + " " * 12 + "║")
    print("╚" + "═" * 58 + "╝\n")
    
    # Check if enriched data exists
    from pathlib import Path
    enriched_json = Path(__file__).parent / "outputs" / "plants_ndvi_enriched.json"
    
    if not enriched_json.exists():
        print("╔" + "═" * 58 + "╗")
        print("║  ERROR: Enriched data not found!                        ║")
        print("╠" + "═" + "╣")
        print("║  Please run the main pipeline first:                    ║")
        print("║    .venv/bin/python plant_ndvi_pipeline/run_pipeline.py ║")
        print("║                                                          ║")
        print("║  Or ensure plants_ndvi_enriched.json exists in:         ║")
        print(f"║    {str(enriched_json.parent):<55}║")
        print("╚" + "═" * 58 + "╝\n")
        sys.exit(1)
    
    print(f"[✓] Found enriched data: {enriched_json.name}")
    
    # ── Step 6: ML Training ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: ML Model Training")
    print("=" * 60)
    
    from step6_ml_training import main as train_ml
    
    try:
        ml_summary = train_ml()
        print("\n[✓] Training complete!")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n[INFO] You may need to install ML libraries:")
        print("  cd /Users/nithints/Drone_imagery_file")
        print("  .venv/bin/pip install pandas scikit-learn xgboost lightgbm")
        sys.exit(1)
    
    # ── Step 7: ML Predictions ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 7: Generate Predictions & Insights")
    print("=" * 60)
    
    from step7_ml_predictions import main as predict_ml
    
    try:
        predict_ml()
        print("\n[✓] Predictions complete!")
    except Exception as e:
        print(f"\n[ERROR] Predictions failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ── Step 8: Export for Web Dashboard ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 8: Export ML Data for Web Dashboard")
    print("=" * 60)
    
    from step8_export_ml_for_web import export_for_web
    
    try:
        export_for_web()
        print("\n[✓] Web export complete!")
    except Exception as e:
        print(f"\n[ERROR] Web export failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "ML PIPELINE COMPLETE!" + " " * 23 + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Total time: {mins}m {secs}s")
    print("║")
    print("║  🤖 Trained Models:")
    print("║     • Growth Stage Prediction (RF, XGBoost, LightGBM)")
    print("║     • Yield Prediction (RF, XGBoost, LightGBM)")
    print("║     • Nitrogen Deficiency Detection (RF, XGBoost, LightGBM)")
    print("║")
    print("║  📊 Generated Reports:")
    print("║     • plants_with_predictions.json")
    print("║     • ml_insights_report.json")
    print("║     • sector_analysis.csv")
    print("║     • high_priority_plants.csv")
    print("║     • ml_data_for_web.js ⭐ (for dashboard)")
    print("║")
    print("║  Next Steps:")
    print("║     1. Review ml_insights_report.json for recommendations")
    print("║     2. Check high_priority_plants.csv for urgent actions")
    print("║     3. Use ml_data_for_web.js to integrate with web dashboard")
    print("║     4. Read dashboard_integration_guide.md for examples")
    print("╚" + "═" * 58 + "╝\n")


if __name__ == "__main__":
    main()
