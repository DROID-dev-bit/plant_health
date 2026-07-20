# Plant Health Pipeline

End-to-end plant-health analytics for the GGS pineapple plantation. For every
detected plant it measures **three agronomic axes** — biomass, nitrogen, and
structure (height) — from drone imagery plus a 3D point cloud, removes soil
noise, fills gaps with machine learning, classifies health, and renders an
interactive map.

```
plant detections + per-plant canopy masks
NDVI / NDRE / GNDVI / OSAVI index maps
DSM + LAZ point cloud
true-color orthomosaic
                │
                ▼
   output/plant_health_indices.csv     (one row per plant, ~40 columns)
   output/plant_health_map.html        (interactive map)
```

---

## Quick start

```bash
cd final_pipeline
../.venv/bin/python run.py                 # build the CSV
../.venv/bin/python make_visualization.py  # build the HTML map
```

`run.py` runs every stage end-to-end. First run builds a cached CHM
(`cache_chm_laz.tif`) from the point cloud; later runs reuse it.

Requirements: `pip install -r requirements.txt`
(numpy, pandas, rasterio, pyproj, pillow, scikit-learn, scipy, xgboost, laspy, lazrs).

---

## Inputs (all paths in `config.py`)

| Input | File | Role |
|-------|------|------|
| Plant table | `../outputs 2/03_ids/plants_with_ids.csv` | IDs, geo coords, bboxes, mask names |
| Canopy masks | `../outputs 2/04_masks/*.png` | one binary mask per plant (sized to its bbox) |
| NDVI map | `../maps/...-NDVI.tif` | **biomass** (leaf area / canopy density) |
| NDRE map | `../maps/...-NDRE.tif` | **nitrogen** / chlorophyll |
| GNDVI map | `../maps/...-GNDVI.tif` | 2nd chlorophyll index (averaged with NDRE) |
| OSAVI map | `../maps/...-OSAVI.tif` | soil-adjusted index — **soil gate** |
| DSM | `../maps/...-dsm.tif` | surface model (fallback height source) |
| LAZ point cloud | `../maps/...-georeferenced_model.laz` | ground-classified → **canopy height** |
| Orthomosaic | `../ggs-orthophoto (2).tif` | basemap + reference grid for the masks |

Key facts:
- The plant detections were generated **from `ggs-orthophoto (2).tif`**, so plant
  coordinates register to it exactly → pixel-perfect overlay (verified).
- NDVI/NDRE/GNDVI/OSAVI maps are **colorized RGBA** renderings, not raw floats;
  values are recovered by matching each pixel to the nearest colormap bin
  (the GGS RdYlGn palette in `colormaps.py`).
- The DTM is **not used**: ODM's DTM mirrors the continuous canopy, so
  `DSM − DTM ≈ 0`. Height comes from the LAZ instead (see below).

---

## Processing stages (in `pipeline.py`)

### 1. Load plants & coordinates
`plant_loader.py` reads the CSV (geo_x/geo_y → x/y, UTM zone 47N), adds
lat/lon, flags `in_orthomosaic`, and computes `canopy_area_m2 = area_px × GSD²`.

### 2. Canopy index sampling + OSAVI soil gate (`canopy_sampler.py`)
For each plant, its mask pixels are projected from the ggs-ortho grid into each
index raster via a composed affine transform. Then the **OSAVI soil gate**:
every canopy pixel's OSAVI value is checked and pixels with `OSAVI ≤ 0.3` are
treated as **bare soil and discarded**. NDVI / NDRE / GNDVI are averaged over
the **vegetation pixels only**.
- Removes ~2.1M soil pixels; `canopy_soil_frac` records how much of each mask
  was soil (quality flag).
- Without gating, soil dragged the means down; gating yields realistic
  vegetation values.

### 3. Canopy Height Model (`chm_sampler.py` + `laz_chm.py`)
The **structure axis**. Two selectable sources (`config.CHM_SOURCE`):
- **`laz` (default, most accurate):** rasterize the point cloud — canopy = max
  z per cell, ground = min z of **ground-classified (class 2)** points, then
  `CHM = canopy − local_ground`. Cached as `cache_chm_laz.tif`.
- **`dsm` (fallback):** `DSM − local_min(DSM, ~2 m)`.

Validated head-to-head on ~2,000 plants: the two correlate **0.90** (mean diff
0.12 m); the LAZ method is more accurate (uses real ground points), giving
realistic pineapple heights (median ≈ 0.34 m). Outputs `plant_height_m` and
`canopy_volume_m3 = height × canopy_area_m2`.

### 4. XGBoost gap-filling (`xgb_imputer.py`)
Plants the index maps don't cover get NDVI/NDRE **predicted** from canopy
morphology + position + nearest-neighbour indices (hold-out R² ≈ 0.8).
Predicted rows are flagged `index_source = 'predicted'`.

### 5. Health model (`health_classifier.py`)
Three independent axes, each a field-relative **percentile score (0–100)**:

| Axis | Source | Meaning |
|------|--------|---------|
| **Biomass** | NDVI | leaf area / canopy density |
| **Nitrogen** | NDRE + GNDVI (averaged) | chlorophyll / N uptake |
| **Structure** | canopy height (CHM) | physical plant size / vigour |

Each axis also gets an **absolute Low / Medium / High level + colour** from
standard precision-ag thresholds:

| Level | Colour | Biomass (NDVI) | Nitrogen (NDRE) |
|-------|--------|----------------|-----------------|
| High | 🟢 `#2E7D32` | ≥ 0.6 | ≥ 0.6 |
| Medium | 🟡 `#FBC02D` | 0.3–0.6 | 0.2–0.6 |
| Low | 🔴 `#D32F2F` | < 0.3 | < 0.2 |

Pipeline: KD-Tree spatial smoothing → 3 axis scores + levels →
**GaussianMixture** over (NDVI, NDRE, height) for data-driven archetypes
(`health_cluster`) → **IsolationForest** anomaly flag → overall
`health_score = 0.40·biomass + 0.30·nitrogen + 0.30·structure`
(falls back to 55/45 NDVI/NDRE when height is missing).

Produces a 4-tier `health_status` (Healthy / Moderate / Stressed / Critical),
an interpretable `health_diagnosis` — including **"Stunted (low canopy
height)"**, which spectral indices alone cannot detect — and `limiting_factor`
(Biomass / Nitrogen / Structure / Balanced).

`NOISE` detections are left blank throughout.

---

## Modules

| File | Role |
|------|------|
| `config.py` | All paths, source toggles & thresholds |
| `plant_loader.py` | Read the plant table (CSV or GeoJSON) |
| `colormaps.py` | NDVI/NDRE/GNDVI/OSAVI palettes + reverse RGB→value |
| `canopy_sampler.py` | Canopy mean of NDVI/NDRE/GNDVI + OSAVI soil gate |
| `laz_chm.py` | Build (and cache) the CHM from the LAZ point cloud |
| `chm_sampler.py` | Per-plant canopy height (LAZ or DSM source) |
| `raw_index_sampler.py` | Optional — canopy mean from the continuous index CSV |
| `raster_sampler.py` | Window / Otsu-shadow sampling (fallback, no masks) |
| `index_assigner.py` | Per-point sampling wrapper (fallback path) |
| `xgb_imputer.py` | XGBoost NDVI/NDRE gap-filling |
| `health_classifier.py` | 3-axis scores + levels + GMM + anomaly + diagnosis |
| `pipeline.py` | Orchestrates all stages, writes the CSV |
| `make_visualization.py` | Build the interactive HTML map |
| `extract_tif_metadata.py` | Dump GeoTIFF metadata to JSON |
| `run.py` | Entry point |

---

## Output CSV — `output/plant_health_indices.csv`

One row per plant (rows with index values sorted first). Key columns:

| Column | Meaning |
|--------|---------|
| `health_score`, `health_status`, `health_color` | overall 0–100 score + 4-tier class |
| `health_diagnosis`, `limiting_factor` | actionable interpretation |
| `biomass_score`, `biomass_level` | NDVI axis: 0–100 rank + Low/Med/High |
| `nitrogen_score`, `nitrogen_level` | NDRE+GNDVI axis: 0–100 rank + Low/Med/High |
| `structure_score` | height axis: 0–100 field-relative rank |
| `plant_height_m`, `canopy_volume_m3` | canopy height (m) & volume (from CHM) |
| `ndvi`, `ndre`, `gndvi` | soil-free canopy-mean index values |
| `osavi`, `canopy_soil_frac` | soil-gate diagnostics |
| `health_cluster`, `cluster_confidence`, `is_anomaly` | unsupervised ML outputs |
| `index_source` | `measured` or `predicted` (XGBoost) |
| `canopy_area_m2` | physical canopy size (`area_px × GSD²`) |

(plus `*_category`, `*_match_dist`, `*_canopy_px` per index and the original ML
columns: `instance_id`, `sector_*`, `bbox_*`, `geo_*`, `mask_file`,
`in_orthomosaic`, …)

---

## Visualization — `output/plant_health_map.html`

Interactive Leaflet map over the orthomosaic. Uses `L.CRS.Simple` in UTM space
for **pixel-perfect alignment** and a canvas renderer for smooth panning with
~14.5k points.

**Color by** layers:
- **Biomass (NDVI) level** — Low/Med/High discrete colours
- **Nitrogen (NDRE) level** — Low/Med/High discrete colours
- **Plant height (m)** — continuous, metres legend
- **NDVI / NDRE (raw)** — continuous value ramps

Filter by sector; per-level / mean stats in the panel; click any plant for its
biomass & nitrogen level + value, height in metres, diagnosis and limiting
factor. Align X/Y nudge sliders exist for fine registration (default 0).

---

## Configuration cheatsheet (`config.py`)

| Setting | Default | Effect |
|---------|---------|--------|
| `OSAVI_SOIL_THRESHOLD` | `0.3` | drop canopy pixels below this OSAVI as soil |
| `USE_GNDVI` | `True` | average GNDVI into the nitrogen axis |
| `CHM_SOURCE` | `"laz"` | `"laz"` (best) or `"dsm"` (fallback) |
| `CHM_GROUND_WINDOW_M` | `2.0` | window for local-ground estimate |
| `CHM_MAX_M` | `2.5` | clip canopy height to a sane crop range |
| `USE_XGB_IMPUTE` | `True` | fill index gaps with XGBoost |
| `USE_RAW_INDEX_CSV` | `False` | use continuous-value CSV instead of maps |

---

## Notable findings on this dataset

- **Nitrogen-limited field:** biomass is mostly High (dense canopy) but nitrogen
  is overwhelmingly Low/Medium — NDRE is the limiting factor for most plants.
- **Soil contamination was large:** a high fraction of raw mask pixels were
  soil; OSAVI gating was essential for trustworthy means.
- **Height adds non-redundant signal:** thousands of plants look healthy
  spectrally but are physically **stunted** — only the CHM reveals them.
