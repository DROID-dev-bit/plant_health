# Plant NDVI + NDRE — ML Pipeline

> Drone imagery → Plant detection → NDVI/NDRE enrichment → ML predictions → Interactive dashboard

A complete end-to-end pipeline that takes raw drone multispectral imagery, matches every detected plant to its NDVI and NDRE values, trains machine learning models to predict growth stage, yield, and nitrogen deficiency, and produces an interactive web dashboard with actionable insights.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Pipeline Architecture](#pipeline-architecture)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [ML Models](#ml-models)
- [NDVI vs NDRE](#ndvi-vs-ndre)
- [Dashboard](#dashboard)
- [Replacing Synthetic Data](#replacing-synthetic-data)
- [Retraining Workflow](#retraining-workflow)
- [Troubleshooting](#troubleshooting)

---

## Overview

This pipeline processes drone multispectral imagery of an oil palm plantation and produces per-plant agronomic insights at scale.

**What it does:**

1. Loads ~14,500 YOLOv8-detected plant bounding boxes with UTM coordinates
2. Matches each plant to NDVI values from a CSV grid (via NDRE TIF pixel mapping)
3. Extracts NDRE values directly from the NDRE GeoTIFF (Band 4 = NIR, Band 3 = Red Edge)
4. Trains 9 ML models (Random Forest, XGBoost, LightGBM × 3 tasks)
5. Predicts growth stage, yield, and nitrogen deficiency for all plants
6. Generates a 5-tab interactive HTML dashboard

**Latest run results (2026-06-09):**

| Metric | Value |
|---|---|
| Total plants processed | 14,498 |
| Plants with NDVI | 13,997 |
| Plants with NDRE | 14,371 |
| NDVI mean | 0.518 |
| NDRE mean | 0.395 |
| Total predicted yield | 480,374 kg |
| Nitrogen-deficient plants | 3,850 (27.5%) |
| Critical attention needed | 3,514 plants |
| Worst sector (S02) avg yield | 12.97 kg |
| Best sector (S03) avg yield | 49.07 kg |

---

## Project Structure

```
plant_ndvi_pipeline/
│
├── README.md                          ← This file
│
├── Core Pipeline Scripts
│   ├── run_pipeline.py                ← Full pipeline orchestrator (Steps 1–7)
│   ├── run_ml_pipeline.py             ← ML-only entry point (Steps 6–7)
│   ├── step1_load_plants.py           ← Load YOLOv8 detections, convert UTM → WGS84
│   ├── step2_match_ndvi.py            ← Match plants to NDVI grid via TIF pixel mapping
│   ├── step3_raster_sample.py         ← Direct NDRE TIF sampling + validation
│   ├── step4_generate_outputs.py      ← Write enriched JSON, CSV, stats
│   ├── step5_generate_map.py          ← Generate Leaflet.js interactive map
│   ├── step6_ml_training.py           ← Train 9 ML models (RF, XGBoost, LightGBM)
│   ├── step7_ml_predictions.py        ← Predict growth stage, yield, nitrogen deficiency
│   └── step8_export_ml_for_web.py     ← Export JS data file for web dashboard
│
├── Configuration
│   ├── requirements.txt               ← Python dependencies
│   └── install_ml_deps.sh             ← Convenience install script
│
└── outputs/
    ├── plants_ndvi_enriched.json      ← 9.9 MB — all plants with NDVI + NDRE fields
    ├── plants_ndvi_summary.csv        ← 2.2 MB — tabular version
    ├── pipeline_stats.json            ← Run statistics and metadata
    ├── plant_ndvi_map.html            ← 5.1 MB — Leaflet.js interactive map
    ├── plants_with_predictions.json   ← 15.2 MB — all plants with ML predictions
    ├── ml_insights_report.json        ← Actionable insights and recommendations
    ├── high_priority_plants.csv       ← 3,613 plants flagged for immediate action
    ├── sector_analysis.csv            ← Per-sector performance metrics (14 sectors)
    ├── plants_predictions_summary.csv ← Simplified prediction table
    ├── ml_dashboard.html              ← Final interactive 5-tab dashboard
    └── ml_models/
        ├── growth_stage_random_forest.pkl
        ├── growth_stage_xgboost.pkl
        ├── growth_stage_lightgbm.pkl
        ├── yield_random_forest.pkl
        ├── yield_xgboost.pkl
        ├── yield_lightgbm.pkl
        ├── nitrogen_random_forest.pkl
        ├── nitrogen_xgboost.pkl
        ├── nitrogen_lightgbm.pkl
        └── training_summary.json
```

---

## Data Sources

All source files live in `/Users/nithints/Drone_imagery_file/`:

| File | Description |
|---|---|
| `outputs 2/03_ids/plants_with_ids.json` | YOLOv8 plant detections with UTM coordinates (EPSG:32647) |
| `NDVI_Data_Complete_Final.csv` | NDVI values indexed by NDRE TIF pixel row/col |
| `Task-of-2026-06-05T095436120Z-orthophoto-NDRE.tif` | NDRE GeoTIFF (4-band uint8, EPSG:32647, 3693×1918 px, 5 cm/px) |
| `Task-of-2026-06-05T095436120Z-orthophoto-NDVI (1).tif` | NDVI reference TIF |

**NDRE TIF band layout:**

| Band | Wavelength | Used for |
|---|---|---|
| 1 | Blue | Visualisation only |
| 2 | Green | Visualisation only |
| 3 | Red Edge | NDRE denominator |
| 4 | NIR | NDRE numerator |

**NDRE formula:** `(Band4 − Band3) / (Band4 + Band3)`

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT                                                          │
│  plants_with_ids.json  ·  NDVI CSV  ·  NDRE TIF                │
└───────────────────┬─────────────────────────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │  Step 1             │  Load plants, convert UTM → WGS84
         │  load_plants        │  ~14,498 plants
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Step 2             │  UTM geo_x/y → NDRE TIF pixel → CSV NDVI grid
         │  match_ndvi         │  3×3 window sampling
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Step 3             │  Direct rasterio sampling of NDRE TIF
         │  raster_sample      │  Adds ndre, ndre_mean, ndre_max, ndre_category
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Step 4             │  Write enriched JSON + CSV + stats
         │  generate_outputs   │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Step 5             │  Leaflet.js interactive map (color by NDVI)
         │  generate_map       │  plant_ndvi_map.html
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Step 6             │  Feature engineering (12 features)
         │  ml_training        │  Train 9 models: RF + XGBoost + LightGBM
         │                     │  × growth stage + yield + nitrogen
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Step 7             │  Predict all 13,997 plants
         │  ml_predictions     │  Generate sector analysis + recommendations
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Step 8             │  Export JS data file for dashboard
         │  export_ml_for_web  │  ml_data_for_web.js
         └──────────┬──────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────┐
│  OUTPUTS                                                        │
│  ml_dashboard.html  ·  predictions JSON  ·  CSVs  ·  .pkl      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Setup

**Requirements:** Python 3.8+, virtual environment

```bash
cd /Users/nithints/Drone_imagery_file

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
.venv/bin/pip install -r plant_ndvi_pipeline/requirements.txt
```

**Dependencies:**

| Package | Purpose |
|---|---|
| `rasterio` | GeoTIFF I/O, NDRE raster sampling |
| `pyproj` | CRS transforms (EPSG:32647 ↔ EPSG:4326) |
| `numpy` | Array operations, grid sampling |
| `pandas` | DataFrames, aggregations |
| `scikit-learn` | Random Forest, train/test split, metrics |
| `xgboost` | XGBoost classifier and regressor |
| `lightgbm` | LightGBM classifier and regressor |

---

## Running the Pipeline

All commands run from `/Users/nithints/Drone_imagery_file`.

### Full pipeline (Steps 1–7)

```bash
.venv/bin/python plant_ndvi_pipeline/run_pipeline.py
```

Runs all 7 steps end-to-end: load → NDVI match → NDRE sample → outputs → map → train → predict.

### ML pipeline only (Steps 6–7)

Use this if you already have `plants_ndvi_enriched.json` from a previous run:

```bash
.venv/bin/python plant_ndvi_pipeline/run_ml_pipeline.py
```

### Individual steps

```bash
# Step 6: Train models only
.venv/bin/python plant_ndvi_pipeline/step6_ml_training.py

# Step 7: Generate predictions only (requires trained models)
.venv/bin/python plant_ndvi_pipeline/step7_ml_predictions.py
```

### Configuration

Edit the `CONFIG` dict at the top of `run_pipeline.py`:

```python
CONFIG = {
    "include_noise": True,   # Include NOISE-labelled plants
    "ndvi_window":   1,      # NDVI sampling half-window (1 → 3×3 patch)
    "tif_window":    3,      # NDRE TIF sampling window size
}
```

---

## Outputs

### `plants_ndvi_enriched.json` — 9.9 MB

Every plant record with both NDVI and NDRE fields added:

```json
{
  "plant_id":        "PLT-S00-R001-C001-8F00DB",
  "sector_label":    "S00",
  "lat":             2.93694028,
  "lon":             101.74417551,
  "geo_x":           805112.4,
  "geo_y":           324918.7,
  "ndvi":            0.113725,
  "ndvi_mean":       0.234,
  "ndvi_max":        0.456,
  "ndvi_category":   "Sparse Vegetation",
  "ndre":            0.298,
  "ndre_mean":       0.312,
  "ndre_max":        0.401,
  "ndre_category":   "Moderate Vegetation",
  "tif_ndvi":        0.120,
  "area_px":         1250,
  "row_index":       1,
  "col_index":       1
}
```

**NDRE categories:**

| Category | NDRE Range | Interpretation |
|---|---|---|
| Water / Non-vegetation | < 0 | No crop canopy |
| Bare / Stressed | 0 – 0.1 | Severe stress |
| Low Vegetation | 0.1 – 0.25 | Weak canopy |
| Moderate Vegetation | 0.25 – 0.4 | Developing canopy |
| Good Vegetation | 0.4 – 0.55 | Healthy canopy |
| Very Good Vegetation | > 0.55 | Dense healthy canopy |

---

### `plants_with_predictions.json` — 15.2 MB

All plants enriched with ML predictions:

```json
{
  "plant_id":                        "PLT-S00-R001-C001-8F00DB",
  "ndvi":                            0.113725,
  "ndre":                            0.298,
  "predicted_growth_stage":          0,
  "predicted_growth_stage_name":     "Seedling",
  "growth_stage_confidence":         0.87,
  "predicted_yield_kg":              7.25,
  "predicted_nitrogen_deficiency":   1,
  "predicted_nitrogen_status":       "Deficient",
  "nitrogen_deficiency_confidence":  0.92
}
```

---

### `ml_insights_report.json` — Summary

```json
{
  "total_plants_analyzed": 13997,
  "growth_stage_distribution": {
    "Mature":     7969,
    "Seedling":   3260,
    "Fruiting":   1446,
    "Flowering":   763,
    "Vegetative":  559
  },
  "yield_analysis": {
    "total_predicted_yield_kg":   480373.9,
    "average_yield_per_plant_kg": 34.32,
    "low_yield_plants":           3602,
    "high_yield_plants":            57
  },
  "nitrogen_analysis": {
    "deficient_plants":       3850,
    "deficient_percentage":  27.51,
    "healthy_plants":        10147
  },
  "critical_attention_needed": 3514,
  "recommendations": [
    { "priority": "HIGH",   "action": "Apply nitrogen fertilizer", "affected_plants": 3850 },
    { "priority": "MEDIUM", "action": "Investigate low-yield areas", "affected_plants": 3602 }
  ]
}
```

---

### `high_priority_plants.csv`

3,613 plants flagged by: `(nitrogen_deficient == True) AND (ndvi < 0.3)`.

Columns: `plant_id, sector_label, row_index, col_index, lat, lon, ndvi, ndre, predicted_yield_kg, predicted_nitrogen_status, predicted_growth_stage_name`

---

### `sector_analysis.csv`

14 sectors (S00–S13) with aggregated metrics:

| Sector | Plants | Avg NDVI | Avg NDRE | Avg Yield | N-Deficient | Def % |
|---|---|---|---|---|---|---|
| S00 | 3,249 | 0.668 | 0.421 | 38.5 kg | 615 | 18.9% |
| S01 | 2,259 | 0.672 | 0.438 | 39.3 kg | 411 | 18.2% |
| **S02** | **1,658** | **-0.185** | **-0.195** | **13.0 kg** | **1,217** | **73.4%** |
| S03 | 101 | 0.861 | 0.512 | 49.1 kg | 8 | 7.9% |
| S04 | 2,466 | 0.555 | 0.362 | 35.4 kg | 577 | 23.4% |
| S05 | 2,062 | 0.282 | 0.218 | 27.5 kg | 897 | 43.5% |
| S06–S13 | 2,302 | 0.79+ | 0.47+ | 42–49 kg | < 60 | < 7% |

**S02 is an anomaly** — both NDVI (−0.185) and NDRE (−0.195) are negative, 73% nitrogen deficiency, and average yield is only 12.97 kg vs 34–49 kg in healthy sectors. This points to a systemic issue: irrigation failure, waterlogging, disease, or a different land-cover class within the mapped boundary.

---

## ML Models

### Features (12 total)

| Feature | Type | Source |
|---|---|---|
| `ndvi` | NDVI | Step 2 (CSV grid) |
| `ndvi_mean` | NDVI | Step 2 (window mean) |
| `ndvi_max` | NDVI | Step 2 (window max) |
| `tif_ndvi` | NDVI | Step 3 (rasterio) |
| `ndvi_tif_diff` | NDVI | ndvi − tif_ndvi |
| `ndvi_variability` | NDVI | ndvi_max − ndvi_mean |
| `row_index` | Spatial | Plant row position |
| `col_index` | Spatial | Plant column position |
| `sector_encoded` | Spatial | Label-encoded sector |
| `spatial_cluster` | Spatial | KMeans cluster id |
| `area_px` | Plant | Bounding-box pixel area |
| `ndvi_category_encoded` | Plant | Label-encoded NDVI category |

### Performance (current run)

| Task | Algorithm | Score |
|---|---|---|
| Growth Stage (5-class) | Random Forest | 87.9% accuracy |
| Growth Stage (5-class) | XGBoost | 87.7% accuracy |
| Growth Stage (5-class) | LightGBM | 87.8% accuracy |
| Yield prediction | Random Forest | R² = 0.953, MAE = 3.22 kg |
| Yield prediction | XGBoost | R² = 0.952, MAE = 3.28 kg |
| Yield prediction | LightGBM | R² = 0.954, MAE = 3.19 kg |
| Nitrogen deficiency | Random Forest | 94.4% accuracy |
| Nitrogen deficiency | XGBoost | 94.6% accuracy |
| Nitrogen deficiency | LightGBM | 94.6% accuracy |

Training time: ~3 seconds for 13,997 plants on modern CPU.  
Train/test split: 80%/20%.

### Top feature importances (growth stage, Random Forest)

```
ndvi                   0.313
ndvi_category_encoded  0.174
ndvi_variability       0.159
ndvi_tif_diff          0.155
ndvi_mean              0.088
area_px                0.041
```

---

## NDVI vs NDRE

Both indices are extracted and stored per plant. They measure different things:

| | NDVI | NDRE |
|---|---|---|
| Formula | (NIR − Red) / (NIR + Red) | (NIR − RedEdge) / (NIR + RedEdge) |
| Sensitive to | Canopy density, green biomass | Chlorophyll concentration, early N-stress |
| Saturates at | High biomass (>0.8) | Less saturation in dense canopy |
| Best for | Vegetation presence mapping | Early stress detection, nutrient status |
| Current mean | 0.518 | 0.395 |

NDRE values tend to be lower than NDVI for the same plant — this is expected because the Red Edge band is more reflective than Red in healthy vegetation, compressing the index slightly.

**When indices disagree strongly** (e.g. S02: NDVI = −0.185, NDRE = −0.195), it reinforces the stress signal and rules out instrument noise as the cause.

---

## Dashboard

The final dashboard is `outputs/ml_dashboard.html` — a single self-contained HTML file.

```bash
open plant_ndvi_pipeline/outputs/ml_dashboard.html
```

**5 tabs:**

| Tab | Contents |
|---|---|
| Overview | 6 KPI cards, growth stage/nitrogen/yield charts, recommendations |
| NDRE Analysis | Category breakdown, histogram, per-sector bars, stacked category mix |
| NDVI vs NDRE | Side-by-side comparison charts, sector bar overlay, gap interpretation table |
| Sectors | Yield + deficiency charts, full 14-sector table with NDVI and NDRE bars |
| ML Models | Accuracy table, feature importance chart, training summary, synthetic-data warning |

---

## Replacing Synthetic Data

The current ML targets (growth stage labels, yield values, nitrogen status) are **synthetically generated** from NDVI thresholds plus random noise. They are suitable for demonstrating the pipeline but must be replaced before making real agronomic decisions.

### What to replace

In `step6_ml_training.py`, locate the function `generate_synthetic_targets()` and replace its return value with data loaded from your ground-truth sources.

### Option A — Field survey labels (growth stage)

```python
import pandas as pd

survey = pd.read_csv('ground_truth/field_survey.csv')  # columns: plant_id, growth_stage
df = df.merge(survey[['plant_id','growth_stage']], on='plant_id', how='left')
# drop plants without labels
df = df.dropna(subset=['growth_stage'])
```

### Option B — Harvest records (yield)

```python
harvest = pd.read_csv('ground_truth/harvest_2025.csv')  # columns: plant_id, yield_kg
df = df.merge(harvest[['plant_id','yield_kg']], on='plant_id', how='left')
```

### Option C — Lab analysis (nitrogen)

```python
lab = pd.read_csv('ground_truth/leaf_nitrogen.csv')  # columns: plant_id, nitrogen_ppm
df = df.merge(lab[['plant_id','nitrogen_ppm']], on='plant_id', how='left')
df['nitrogen_deficient'] = (df['nitrogen_ppm'] < threshold).astype(int)
```

### Option D — Expert annotations

```python
annotations = pd.read_csv('ground_truth/expert_labels.csv')
df = df.merge(annotations, on='plant_id', how='left')
```

---

## Retraining Workflow

Retrain when:
- A new growing season begins
- More ground-truth data is collected
- Model predictions diverge from field observations
- A different crop variety or management regime is introduced

```bash
# 1. Update ground-truth data in step6_ml_training.py
# 2. Retrain
.venv/bin/python plant_ndvi_pipeline/step6_ml_training.py

# 3. Generate fresh predictions
.venv/bin/python plant_ndvi_pipeline/step7_ml_predictions.py

# 4. Rebuild dashboard
open plant_ndvi_pipeline/outputs/ml_dashboard.html
```

---

## Troubleshooting

**`No module named 'xgboost'` or `'lightgbm'`**
```bash
.venv/bin/pip install xgboost lightgbm
```

**`Enriched data not found`**  
Run steps 1–5 first:
```bash
.venv/bin/python plant_ndvi_pipeline/run_pipeline.py
```

**`TIF not found` warning in Step 3**  
Check the TIF path in `step3_raster_sample.py` line 23. The pipeline expects the NDRE TIF at:
```
/Users/nithints/Drone_imagery_file/Task-of-2026-06-05T095436120Z-orthophoto-NDRE (1).tif
```
To use the non-`(1)` version, update `NDRE_TIF` in that file.

**Memory error during training**  
Use LightGBM only (most memory-efficient), or sample down:
```python
df = df.sample(n=10000, random_state=42)
```

**Training is slow**  
Reduce estimator count in `step6_ml_training.py`:
```python
n_estimators = 50   # default is 100
```

**Out-of-bounds plants have no NDRE**  
Plants outside the TIF bounding box receive `ndre_category = "Out of Bounds"` (127 plants in current run). This is expected for plants at the plantation edge.

---

## Notes

- All coordinates use **EPSG:32647** (UTM Zone 47N) internally; outputs are converted to **EPSG:4326** (WGS84 lat/lon).
- The NDRE TIF pixel resolution is **5 cm/pixel** (transform: 0.05 m/px).
- NDVI is sourced from the CSV grid and validated against the NDVI TIF; NDRE is sourced directly from the NDRE TIF.
- Model targets are currently synthetic — see [Replacing Synthetic Data](#replacing-synthetic-data).
- Both the NDVI interactive map (`plant_ndvi_map.html`) and the ML dashboard (`ml_dashboard.html`) are standalone single-file HTML pages with no server required.
