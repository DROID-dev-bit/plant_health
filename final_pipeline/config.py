"""
Central configuration for the plant-health index pipeline.

Every other module imports its paths and constants from here, so the whole
pipeline can be re-pointed at new data by editing a single file.
"""

from pathlib import Path

# ── Project layout ───────────────────────────────────────────────────────────
# config.py lives in <project>/final_pipeline/ , so the project root is one up.
ROOT_DIR     = Path(__file__).resolve().parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR   = PIPELINE_DIR / "output"

# ── Inputs (from the ML team) ────────────────────────────────────────────────
# Plant table produced by the ML team. The CSV carries per-plant geo coords,
# bounding boxes and canopy mask filenames; it was generated from
# ggs-orthophoto (2).tif, so its coordinates register exactly to that ortho.
PLANTS_FILE = ROOT_DIR / "outputs 2" / "03_ids" / "plants_with_ids.csv"

# Per-plant binary canopy masks (one PNG per plant, sized to its bbox).
MASKS_DIR = ROOT_DIR / "outputs 2" / "04_masks"

# Coordinates are UTM zone 47N (the table has no CRS field).
PLANTS_CRS = "EPSG:32647"

# Colorized index orthomosaics (4-band RGBA GeoTIFFs) from the maps/ folder.
# These cover ~99% of the plants. The index is identified by the filename
# suffix (NDVI / NDRE). An OSAVI map also exists but is intentionally unused.
NDVI_TIF = ROOT_DIR / "maps" / "Task-of-2026-03-22T070712342Z-orthophoto-NDVI.tif"
NDRE_TIF = ROOT_DIR / "maps" / "Task-of-2026-03-22T070712342Z-orthophoto-NDRE.tif"

# GNDVI (Green NDVI) — a second chlorophyll/nitrogen-sensitive index that
# saturates less than NDVI; averaged with NDRE for a more robust N reading.
GNDVI_TIF = ROOT_DIR / "maps" / "Task-of-2026-03-22T063838646Z-orthophoto-GNDVI.tif"
USE_GNDVI = True

# TDVI (Transformed Difference Vegetation Index) — a biomass/canopy-density
# index that, unlike NDVI, does NOT saturate over dense canopy, so it separates
# "healthy" from "very healthy" where NDVI flattens. Unlike the other maps this
# is the RAW single-band float32 raster (real continuous values, proper -9999
# nodata) rather than a colorized RGBA render, so it is the highest-fidelity
# vegetation signal in the stack. Averaged with NDVI to strengthen the biomass
# axis (mirroring how GNDVI strengthens the nitrogen axis).
TDVI_TIF = ROOT_DIR / "maps" / "Task-of-2026-06-05T095436120Z-orthophoto-TDVI.tif"
USE_TDVI = True

# Canopy Height Model: DSM minus a local-ground estimate (rolling minimum of
# the DSM), giving plant height above the surrounding furrow/soil. Adds a
# physical STRUCTURE axis (height / canopy volume) to the health model.
# (DSM-minus-DTM is unusable here: ODM's DTM mirrors the canopy on continuous
#  farmland, so DSM-DTM ~ 0.)
DSM_TIF = ROOT_DIR / "maps" / "Task-of-2026-06-28T103132228Z-dsm.tif"
# LAZ point cloud (ground-classified) -> most accurate CHM. Cached as a tif.
LAZ_FILE = ROOT_DIR / "maps" / "Task-of-2026-06-28T103132228Z-georeferenced_model.laz"
CHM_CACHE = PIPELINE_DIR / "cache_chm_laz.tif"
USE_CHM = True
CHM_SOURCE = "laz"          # "laz" (best) or "dsm" (fallback)
CHM_GROUND_WINDOW_M = 2.0   # window for the local-ground estimate
CHM_MAX_M = 2.5             # clip canopy height to a sane crop range

# OSAVI (soil-adjusted) map. Used to mask bare-soil / background pixels inside
# each canopy mask so NDVI/NDRE are averaged over vegetation only.
OSAVI_TIF = ROOT_DIR / "maps" / "Task-of-2026-03-22T063838646Z-orthophoto-OSAVI.tif"
USE_OSAVI_SOIL_MASK = True
# Canopy pixels with OSAVI below this (on the -0.2..1.0 colormap scale) are
# treated as soil and excluded from the NDVI/NDRE canopy mean.
OSAVI_SOIL_THRESHOLD = 0.2

# True-color orthomosaic, used as the visualization basemap, for shadow
# masking, and as the reference extent for the plant-coverage check. This is
# the exact ortho the plant detections were made on (pixel-perfect overlay).
ORTHO_TIF = ROOT_DIR / "ggs-orthophoto (2).tif"

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_CSV = OUTPUT_DIR / "plant_health_indices.csv"

# ── Sampling behaviour ───────────────────────────────────────────────────────
# Radius (in pixels) of the square window averaged around each plant centre.
# 0 = read the single centre pixel only. A small window is more robust to the
# anti-aliasing / compression noise present in the colorized rasters.
SAMPLE_RADIUS_PX = 1

# Pixels equal to this RGB are treated as background / no-data and skipped.
BACKGROUND_RGB = (255, 255, 255)

# Real (continuous) index values exported from the multispectral source.
# Disabled in favour of the wider-coverage maps/ rasters (the CSV only covered
# ~76% of the field). Set True to switch back to continuous values.
RAW_INDEX_CSV = ROOT_DIR / "ndvi_ndre_Unknown_2026-06-10.csv"
USE_RAW_INDEX_CSV = False
RAW_MAX_MATCH_M = 0.5     # max distance from a canopy pixel to a raw sample

# Canopy-mask sampling: average each index over the plant's actual canopy
# pixels (from outputs 2/04_masks). Used (against the colorized rasters) when
# the raw index CSV is unavailable.
USE_CANOPY_MASKS = True

# GSD (m/pixel) of the ggs orthomosaic the masks were drawn on — used to turn
# canopy pixel area into physical m^2.
ORTHO_GSD_M = 0.013458

# Otsu shadow masking: sample index values only from sunlit leaf pixels
# (uses the true-color orthomosaic to detect shadow). Fallback when canopy
# masks are not available. Set False for the simpler centre-window mean.
USE_SHADOW_MASKING = True
SHADOW_WINDOW_RADIUS_PX = 5     # half-size of the ortho window per plant

# XGBoost gap-filling: predict NDVI/NDRE for non-NOISE plants the index maps
# don't cover, from canopy morphology + spatial context. Predicted rows are
# flagged index_source='predicted'.
USE_XGB_IMPUTE = True

# Detections labelled NOISE are weeds / false positives, not crops; their
# index + health values are forced blank.
NOISE_LABEL = "NOISE"

# Trained ML model applied to every plant. Uses the user-supplied weights in
# final_pipeline/ if present, else falls back to a locally trained copy in
# weights/ (produced by train_model.py or the Colab notebook).
USE_ML_MODEL = True
_user_model = PIPELINE_DIR / "plant_health_model.pkl"
ML_MODEL_PATH = _user_model if _user_model.exists() else (
    ROOT_DIR / "weights" / "plant_health_model.pkl")

# Decimal precision for lat/lon written to the CSV.
COORD_DECIMALS = 8
