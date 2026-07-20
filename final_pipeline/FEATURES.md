# Features — Plant Health Pipeline

Full technical detail lives in [README.md](README.md); this is the feature list.

## Analysis
- **Per-plant vegetation indices**: NDVI, NDRE, GNDVI, OSAVI, TDVI sampled from
  drone orthomosaic index maps for every detected plant.
- **OSAVI soil gate**: discards bare-soil pixels (OSAVI ≤ 0.3) from each
  canopy mask before averaging, so soil doesn't drag index values down.
- **Canopy Height Model (CHM)**: per-plant height from the LAZ point cloud
  (ground-classified points) or DSM fallback → `plant_height_m`,
  `canopy_volume_m3`, `canopy_area_m2`.
- **XGBoost gap-filling**: predicts NDVI/NDRE for plants the index maps don't
  cover, from canopy morphology, position, and nearest neighbours.
- **3-axis health scoring** (0–100, field-relative percentile):
  - Biomass (from NDVI)
  - Nitrogen (from NDRE + GNDVI)
  - Structure (from canopy height)
- **Absolute Low/Medium/High level classification** per axis using
  precision-ag thresholds, with colour codes.
- **Overall health score & 4-tier status**: Healthy / Moderate / Stressed /
  Critical, weighted 40% biomass + 30% nitrogen + 30% structure.
- **Actionable diagnosis**: e.g. "Stunted (low canopy height)", plus a
  `limiting_factor` (Biomass / Nitrogen / Structure / Balanced).
- **Unsupervised ML add-ons**: GaussianMixture health archetypes
  (`health_cluster`), IsolationForest anomaly detection (`is_anomaly`).
- **Secondary ML health model**: `ml_health_pred` / `ml_health_confidence` /
  `ml_agrees` cross-check against the rule-based score.
- **Spatial smoothing**: KD-tree neighbour smoothing of index values before
  scoring.
- **Sector/geo tagging**: lat/lon, UTM x/y, pixel coords, bbox, sector label
  and ID per plant.
- **Quality flags**: `canopy_soil_frac`, `canopy_fallback`, `index_source`
  (measured vs. predicted), match distances per index.

## Outputs
- **`output/plant_health_indices.csv`** — one row per plant, ~60 columns
  (all of the above).
- **`output/plant_health_map.html`** — interactive Leaflet map:
  - Pixel-perfect overlay on the orthomosaic (UTM `L.CRS.Simple`)
  - Colour-by: biomass level, nitrogen level, plant height (continuous),
    raw NDVI/NDRE (continuous)
  - Sector filtering, per-level/mean stats panel
  - Click-to-inspect per plant (levels, values, height, diagnosis,
    limiting factor)
  - Manual X/Y alignment nudge sliders
- **`output/orthomosaic_web.png`** — web-ready basemap render.
- **`output/plant_ndvi_ndre_for_frontend.csv`** — slim NDVI/NDRE export for
  downstream frontend use.
- **`output/plant_health_indices_osavi020.csv`** — alternate export at a
  lower (0.20) OSAVI soil threshold for comparison.

## Configuration
All tunable via `config.py`: OSAVI soil threshold, GNDVI on/off, CHM source
(LAZ vs DSM) and clipping, XGBoost imputation on/off, raw-index-CSV mode.

---

# Water Flow Physics Engine (`../physics_engine`)

Interactive rainfall / surface-water-flow simulation over the farm's real
drone-derived terrain (DTM + DSM), runnable standalone in a browser.

## Analysis / Simulation
- **Terrain prep** (`prepare_terrain.py`): reads DTM/DSM GeoTIFFs, cleans and
  downsamples to a 1 m grid, computes canopy height, and derives D8 flow
  direction/accumulation (priority-flood depression filling) → serialized to
  `terrain_data.js` for the browser (no server needed).
- **Rain particle physics**: each drop accelerates down the steepest slope of
  the water surface (`a = -g·∇(z + d) - friction·v`), integrated per frame.
- **Ponding**: stalled drops deposit into a shallow-water height field; ponds
  relax toward flat water surfaces and slowly infiltrate.
- **Drainage channels**: D8 flow accumulation rendered as the terrain's
  natural streamlines.

## Controls / UI (`water_flow_simulation.html`)
- Rain on/off, intensity, and flow-speed sliders
- Click/drag anywhere on the map to pour water at that point
- Layer toggles: channels, ponding, particles, canopy, contours
- Hover any cell for elevation, water depth, and upstream catchment area
