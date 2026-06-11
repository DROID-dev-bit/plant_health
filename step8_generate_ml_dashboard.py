#!/usr/bin/env python3
"""
Step 8 — Generate ML-Enhanced Interactive Dashboard
====================================================
Creates a visual web dashboard integrating ML predictions for farm management.

Features:
- Interactive map with ML prediction layers
- Growth stage color coding
- Yield heatmap
- Nitrogen deficiency highlighting
- Sector performance charts
- Actionable recommendations panel
- Priority plant filtering
"""

import json
import time
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent / "outputs"

PREDICTIONS_JSON = OUT_DIR / "plants_with_predictions.json"
INSIGHTS_JSON = OUT_DIR / "ml_insights_report.json"
DASHBOARD_HTML = BASE_DIR / "ml_farm_dashboard.html"


def load_data():
    """Load predictions and insights."""
    print("[INFO] Loading ML predictions...")
    
    if not PREDICTIONS_JSON.exists():
        print(f"\n[ERROR] Predictions file not found!")
        print(f"Expected: {PREDICTIONS_JSON}")
        print("\nPlease run the ML pipeline first:")
        print("  .venv/bin/python plant_ndvi_pipeline/run_ml_pipeline.py\n")
        return None, None
    
    with open(PREDICTIONS_JSON, 'r') as f:
        plants = json.load(f)
    
    insights = {}
    if INSIGHTS_JSON.exists():
        with open(INSIGHTS_JSON, 'r') as f:
            insights = json.load(f)
    
    print(f"[✓] Loaded {len(plants):,} plants with ML predictions")
    
    return plants, insights


def generate_dashboard_html(plants, insights):
    """Generate the complete dashboard HTML."""
    
    print(f"\n[INFO] Generating dashboard HTML...")
    
    # Prepare data for JavaScript
    plants_json = json.dumps(plants)
    insights_json = json.dumps(insights)
    
    # Extract key metrics
    total_plants = len(plants)
    yield_analysis = insights.get('yield_analysis', {})
    nitrogen_analysis = insights.get('nitrogen_analysis', {})
    recommendations = insights.get('recommendations', [])
    
    total_yield = yield_analysis.get('total_predicted_yield_kg', 0)
    avg_yield = yield_analysis.get('average_yield_per_plant_kg', 0)
    deficient_count = nitrogen_analysis.get('deficient_plants', 0)
    deficient_pct = nitrogen_analysis.get('deficient_percentage', 0)
    
    # Growth stage distribution
    growth_dist = insights.get('growth_stage_distribution', {})
    
    # Create the HTML
    html_parts = []
    
    # Part 1: Head and styles
    html_parts.append(generate_html_head())
    
    # Part 2: Body structure
    html_parts.append(generate_html_body(
        total_plants, total_yield, avg_yield,
        deficient_count, deficient_pct, growth_dist, recommendations
    ))
    
    # Part 3: JavaScript with data
    html_parts.append(generate_javascript(plants_json, insights_json))
    
    # Part 4: Closing tags
    html_parts.append("</body></html>")
    
    return "".join(html_parts)


def generate_html_head():
    """Generate HTML head with styles."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ML Farm Dashboard | Dr</title>
</invoke>