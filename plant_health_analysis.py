#!/usr/bin/env python3
"""
Plant Health ML Analysis System
================================
Analyzes plant health using NDVI, NDRE, and DSM GeoTIFF inputs.
Trains XGBoost regression + classification models, generates pixel-level
predictions, zone-based management plans, and full visualisation reports.

Usage:
    python plant_health_analysis.py

Output directory: configured in FILE PATHS section below.

How the NDVI and DSM files are decoded:
  - NDVI: 4-band uint8 colourised render; Band 1 DN → NDVI = (DN / 127.5) − 1
  - NDRE: 1-band float32, already in −1…1
  - DSM : 4-band uint8 colourised render; luminance = 0.299R+0.587G+0.114B
           used as relative elevation proxy
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, label, uniform_filter
from scipy.ndimage import map_coordinates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report
import xgboost as xgb

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# FILE PATHS  ← change these to match your environment
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path("/Users/nithints/Drone_imagery_file")

NDVI_PATH = BASE_DIR / "Task-of-2026-06-05T095436120Z-orthophoto-NDVI (1).tif"
NDRE_PATH = BASE_DIR / "Task-of-2026-06-05T095436120Z-orthophoto-NDRE (1).tif"
DSM_PATH  = BASE_DIR / "Task-of-2026-03-23T095842854Z-dsm.tif"

OUT_DIR   = BASE_DIR / "plant_ndvi_pipeline" / "outputs" / "ml_health_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model parameters ──────────────────────────────────────────────────────────
PIXEL_SIZE_M    = 0.05          # metres per pixel (from raster resolution)
SMOOTH_SIGMA    = 3             # Gaussian σ for NDVI variability feature
SLOPE_RISK_DEG  = 15            # slope threshold for terrain risk
DRAIN_RISK_THR  = -0.2          # drainage score below this = waterlogged risk
RANDOM_STATE    = 42
TEST_SIZE       = 0.20

# ── Cost table (INR per hectare) ──────────────────────────────────────────────
COST_PER_HA = {
    "nitrogen":    5_000,
    "drainage":   15_000,
    "pest":        8_000,
    "routine":     3_000,
}

FEATURE_NAMES = [
    "ndvi", "ndre", "nsi", "chlorophyll_idx", "stress_idx",
    "elevation", "slope", "aspect", "tri", "drainage_score",
    "ndvi_variability",
]

HEALTH_LABELS = ["Poor", "Fair", "Good", "Excellent"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_geotiff(path: Path, band: int = 1) -> tuple[np.ndarray, dict]:
    """
    Read a single band from a GeoTIFF. Returns (array, meta).
    meta includes: crs, transform, width, height, nodata.
    """
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
        meta = {
            "crs":       src.crs,
            "transform": src.transform,
            "width":     src.width,
            "height":    src.height,
            "nodata":    src.nodata,
            "bounds":    src.bounds,
            "count":     src.count,
        }
    return data, meta


def load_all_inputs() -> dict:
    """
    Load and decode NDVI, NDRE, and DSM. Reprojects DSM to NDVI grid.
    Returns dict with arrays + shared metadata.
    """
    log.info("Loading NDVI …")
    ndvi_dn, ndvi_meta = load_geotiff(NDVI_PATH, band=1)

    # NDVI: DN (0-255) → float (-1 to 1)
    ndvi = (ndvi_dn / 127.5) - 1.0

    # Alpha mask (band 4): pixels with alpha == 0 are nodata
    with rasterio.open(NDVI_PATH) as src:
        if src.count >= 4:
            alpha = src.read(4)
            ndvi_mask = alpha == 0          # True = invalid pixel
        else:
            ndvi_mask = ndvi_dn == 0

    log.info("Loading NDRE …")
    ndre_raw, ndre_meta = load_geotiff(NDRE_PATH, band=1)

    # Align NDRE to NDVI grid (they share same extent; minor float tolerance)
    if (ndre_raw.shape != ndvi.shape or
            not _transforms_close(ndvi_meta["transform"], ndre_meta["transform"])):
        log.info("Reprojecting NDRE to NDVI grid …")
        ndre_raw = _reproject_to_grid(ndre_raw, ndre_meta, ndvi_meta)

    log.info("Loading DSM …")
    # DSM is a 4-band colourised uint8 render; use luminance as elevation proxy
    with rasterio.open(DSM_PATH) as src:
        R = src.read(1).astype(np.float32)
        G = src.read(2).astype(np.float32)
        B = src.read(3).astype(np.float32)
        if src.count >= 4:
            dsm_alpha = src.read(4)
            dsm_mask  = dsm_alpha == 0
        else:
            dsm_mask = (R == 0) & (G == 0) & (B == 0)
        dsm_raw = 0.299 * R + 0.587 * G + 0.114 * B   # luminance (0-255)
        dsm_meta = {
            "crs": src.crs, "transform": src.transform,
            "width": src.width, "height": src.height,
        }

    dsm_raw[dsm_mask] = np.nan

    log.info("Reprojecting DSM to NDVI grid …")
    dsm = _reproject_to_grid(dsm_raw, dsm_meta, ndvi_meta)

    # Normalise DSM luminance 0-255 → 0-1 (relative elevation)
    dsm_valid = ~np.isnan(dsm)
    if dsm_valid.any():
        dmin, dmax = dsm[dsm_valid].min(), dsm[dsm_valid].max()
        dsm = (dsm - dmin) / max(dmax - dmin, 1e-6)

    # Final combined mask
    combined_mask = ndvi_mask | np.isnan(ndre_raw) | np.isnan(dsm)

    # Apply mask (set to NaN)
    ndvi[combined_mask]    = np.nan
    ndre_raw[combined_mask] = np.nan
    dsm[combined_mask]     = np.nan

    valid_px = int((~combined_mask).sum())
    log.info(f"Valid pixels: {valid_px:,} / {ndvi.size:,}  "
             f"({valid_px/ndvi.size*100:.1f}%)")

    return {
        "ndvi":     ndvi,
        "ndre":     ndre_raw,
        "dsm":      dsm,
        "mask":     combined_mask,
        "meta":     ndvi_meta,
        "pixel_m":  float(abs(ndvi_meta["transform"].a)),
    }


def _transforms_close(t1, t2, tol=1e-4) -> bool:
    return all(abs(getattr(t1, a) - getattr(t2, a)) < tol
               for a in ("a", "b", "c", "d", "e", "f"))


def _reproject_to_grid(src_data: np.ndarray, src_meta: dict,
                       dst_meta: dict) -> np.ndarray:
    """Reproject src_data from src_meta CRS/transform to dst_meta grid."""
    dst = np.full((dst_meta["height"], dst_meta["width"]),
                  np.nan, dtype=np.float32)
    reproject(
        source=src_data,
        destination=dst,
        src_transform=src_meta["transform"],
        src_crs=src_meta["crs"],
        dst_transform=dst_meta["transform"],
        dst_crs=dst_meta["crs"],
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def create_features(data: dict) -> dict:
    """
    Compute 11 derived features from NDVI, NDRE, and DSM.
    Returns dict mapping feature name → 2-D array (NaN where invalid).
    """
    log.info("Engineering features …")
    ndvi = data["ndvi"].copy()
    ndre = data["ndre"].copy()
    dsm  = data["dsm"].copy()
    mask = data["mask"]

    # ── Vegetation indices ────────────────────────────────────────────────
    nsi  = ndre - ndvi                                      # Nitrogen Stress Index
    # Chlorophyll Index: guard against NDVI ≈ 0
    with np.errstate(divide="ignore", invalid="ignore"):
        chlo = np.where(np.abs(ndvi) > 0.01, (ndre / ndvi) - 1.0, np.nan)
    stress = 1.0 - ndvi                                     # Stress Index

    # ── Terrain features ─────────────────────────────────────────────────
    # Fill NaN in DSM with local mean before gradient
    dsm_filled = _fill_nan(dsm)

    pixel_m = data["pixel_m"] if data["pixel_m"] > 0 else 1.0
    gy, gx  = np.gradient(dsm_filled, pixel_m, pixel_m)

    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))          # radians
    slope_deg = np.degrees(slope_rad)
    aspect    = np.degrees(np.arctan2(-gx, gy)) % 360      # 0–360°

    # TRI: standard deviation of 3×3 neighbourhood
    dsm_sq  = uniform_filter(dsm_filled**2, size=3)
    dsm_avg = uniform_filter(dsm_filled,    size=3)
    tri     = np.sqrt(np.maximum(dsm_sq - dsm_avg**2, 0))

    # Drainage score = (slope / max_slope) − (elevation / max_elevation)
    max_slope = float(slope_deg[~mask].max()) if (~mask).any() else 1.0
    max_elev  = float(dsm[~mask].max())       if (~mask).any() else 1.0
    drainage  = (slope_deg / max_slope) - (dsm_filled / max(max_elev, 1e-6))

    # ── Spatial context ───────────────────────────────────────────────────
    ndvi_smooth    = gaussian_filter(_fill_nan(ndvi), sigma=SMOOTH_SIGMA)
    ndvi_variability = np.abs(ndvi - ndvi_smooth)

    # Re-apply mask
    feats = {
        "ndvi":              ndvi,
        "ndre":              ndre,
        "nsi":               nsi,
        "chlorophyll_idx":   chlo.astype(np.float32),
        "stress_idx":        stress,
        "elevation":         dsm,
        "slope":             slope_deg.astype(np.float32),
        "aspect":            aspect.astype(np.float32),
        "tri":               tri.astype(np.float32),
        "drainage_score":    drainage.astype(np.float32),
        "ndvi_variability":  ndvi_variability.astype(np.float32),
    }
    for name, arr in feats.items():
        arr[mask] = np.nan

    return feats


def _fill_nan(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = arr.copy()
    out[np.isnan(out)] = fill
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPARE TRAINING DATA
# ══════════════════════════════════════════════════════════════════════════════

def health_score_from_ndvi(ndvi_arr: np.ndarray) -> np.ndarray:
    """Convert NDVI (−1…1) to health score (0…100)."""
    return np.clip((ndvi_arr + 1.0) / 2.0 * 100.0, 0, 100)


def health_category(score_arr: np.ndarray) -> np.ndarray:
    """
    0 = Poor (0-25), 1 = Fair (25-50), 2 = Good (50-75), 3 = Excellent (75-100).
    Aligned with pineapple thresholds:
      NDVI < 0.2 → score < 60  → Poor/Fair
      NDVI 0.2–0.6 → Fair
      NDVI 0.6–0.8 → Good
      NDVI ≥ 0.8   → Excellent
    """
    cat = np.zeros_like(score_arr, dtype=np.int8)
    cat[score_arr >= 25] = 1
    cat[score_arr >= 60] = 2
    cat[score_arr >= 80] = 3   # NDVI ≥ 0.6 → Good, ≥ 0.8 → Excellent
    return cat


def prepare_training_data(features: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten features into a 2-D sample matrix (N × 11).
    Returns X, y_reg, y_cls, valid_flat_idx.
    """
    log.info("Preparing training data …")

    # Stack features in a consistent order
    stacked = np.stack([features[n] for n in FEATURE_NAMES], axis=-1)  # H×W×11
    H, W, _ = stacked.shape
    flat     = stacked.reshape(-1, len(FEATURE_NAMES))

    # Valid rows: no NaN in any feature
    valid_rows = ~np.isnan(flat).any(axis=1)
    valid_idx  = np.where(valid_rows)[0]
    X          = flat[valid_idx]

    ndvi_flat   = features["ndvi"].ravel()
    y_reg       = health_score_from_ndvi(ndvi_flat[valid_idx])
    y_cls       = health_category(y_reg)

    log.info(f"Training samples: {len(X):,}  |  "
             f"Poor={int((y_cls==0).sum()):,}  Fair={int((y_cls==1).sum()):,}  "
             f"Good={int((y_cls==2).sum()):,}  Excellent={int((y_cls==3).sum()):,}")

    return X, y_reg, y_cls, valid_idx


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_xgboost_models(X: np.ndarray, y_reg: np.ndarray,
                         y_cls: np.ndarray) -> dict:
    """
    Train XGBoost regression + classification models. Returns models + metrics.
    Saves PKL files to OUT_DIR.
    """
    log.info("Splitting data (80/20) …")
    X_tr, X_te, yr_tr, yr_te = train_test_split(
        X, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    _, _,    yc_tr, yc_te   = train_test_split(
        X, y_cls, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Scale features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # ── Regression ───────────────────────────────────────────────────────
    log.info("Training XGBoost Regression (health score 0-100) …")
    reg = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    reg.fit(X_tr_s, yr_tr,
            eval_set=[(X_te_s, yr_te)],
            verbose=False)

    yr_pred   = reg.predict(X_te_s)
    reg_r2    = r2_score(yr_te, yr_pred)
    reg_mae   = mean_absolute_error(yr_te, yr_pred)
    log.info(f"  Regression  →  R²={reg_r2:.4f}  MAE={reg_mae:.2f}")

    # ── Classification ───────────────────────────────────────────────────
    log.info("Training XGBoost Classification (Poor/Fair/Good/Excellent) …")
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X_tr_s, yc_tr,
            eval_set=[(X_te_s, yc_te)],
            verbose=False)

    yc_pred  = clf.predict(X_te_s)
    cls_acc  = accuracy_score(yc_te, yc_pred)
    cls_rep  = classification_report(yc_te, yc_pred,
                                     target_names=HEALTH_LABELS, zero_division=0)
    log.info(f"  Classification  →  Accuracy={cls_acc:.4f}")
    log.info(f"\n{cls_rep}")

    # Feature importance
    fi_reg = pd.Series(reg.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    fi_cls = pd.Series(clf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)

    # Save models
    joblib.dump(reg,    OUT_DIR / "xgboost_regression_model.pkl")
    joblib.dump(clf,    OUT_DIR / "xgboost_classification_model.pkl")
    joblib.dump(scaler, OUT_DIR / "feature_scaler.pkl")

    pd.DataFrame({"regression": reg.feature_importances_,
                  "classification": clf.feature_importances_},
                 index=FEATURE_NAMES).to_csv(OUT_DIR / "feature_importance.csv")

    log.info("Models saved to " + str(OUT_DIR))

    return {
        "reg": reg, "clf": clf, "scaler": scaler,
        "reg_r2": reg_r2, "reg_mae": reg_mae,
        "cls_acc": cls_acc, "cls_report": cls_rep,
        "fi_reg": fi_reg, "fi_cls": fi_cls,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. GENERATE PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def generate_predictions(features: dict, models: dict) -> dict:
    """
    Run trained models over every valid pixel in the map.
    Returns health_score, health_cat, priority, confidence arrays.
    """
    log.info("Generating predictions for entire map …")

    stacked = np.stack([features[n] for n in FEATURE_NAMES], axis=-1)
    H, W, _ = stacked.shape
    flat    = stacked.reshape(-1, len(FEATURE_NAMES))

    valid_rows = ~np.isnan(flat).any(axis=1)
    valid_idx  = np.where(valid_rows)[0]
    X_valid    = flat[valid_idx]

    scaler = models["scaler"]
    reg    = models["reg"]
    clf    = models["clf"]

    X_sc      = scaler.transform(X_valid)
    scores    = reg.predict(X_sc).clip(0, 100)
    cats      = clf.predict(X_sc)
    proba     = clf.predict_proba(X_sc)
    confidence = proba.max(axis=1) * 100  # 0-100

    # Build full-resolution output arrays (NaN/−1 for masked pixels)
    health_score = np.full(H * W, np.nan, dtype=np.float32)
    health_cat   = np.full(H * W, -1,    dtype=np.int8)
    priority     = np.full(H * W, -1,    dtype=np.int8)
    conf_arr     = np.full(H * W, np.nan, dtype=np.float32)

    health_score[valid_idx] = scores
    health_cat[valid_idx]   = cats
    conf_arr[valid_idx]     = confidence

    # Priority: 1=Critical, 2=High, 3=Medium, 4=Low
    p = np.full_like(scores, 4, dtype=np.int8)
    p[scores < 30] = 1
    p[(scores >= 30) & (scores < 50)] = 2
    p[(scores >= 50) & (scores < 70)] = 3
    priority[valid_idx] = p

    log.info(f"  Critical  : {int((p==1).sum()):,} px")
    log.info(f"  High      : {int((p==2).sum()):,} px")
    log.info(f"  Medium    : {int((p==3).sum()):,} px")
    log.info(f"  Low       : {int((p==4).sum()):,} px")

    return {
        "health_score": health_score.reshape(H, W),
        "health_cat":   health_cat.reshape(H, W),
        "priority":     priority.reshape(H, W),
        "confidence":   conf_arr.reshape(H, W),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _rec(score, nsi, stress, drainage) -> tuple[str, str]:
    """Return (recommendation text, action_key) for one pixel."""
    if score < 30:
        if nsi < -0.1:
            return ("Severe nitrogen deficiency. Apply nitrogen fertilizer immediately. "
                    "Irrigation required.", "nitrogen")
        if stress > 0.7:
            return ("Extreme stress. Check for pests/diseases. "
                    "Reduce irrigation temporarily.", "pest")
        if drainage < 0:
            return ("Waterlogged. Improve drainage. Reduce irrigation.", "drainage")
        return ("Critical health issue. Multi-intervention needed.", "pest")
    if score < 50:
        if nsi < 0:
            return "Nitrogen deficiency. Apply nitrogen fertilizer.", "nitrogen"
        if stress > 0.5:
            return ("Plant stress detected. Monitor for pests/diseases.", "pest")
        if drainage < DRAIN_RISK_THR:
            return ("Poor drainage. Reduce irrigation frequency.", "drainage")
        return ("Monitor closely. May need intervention soon.", "routine")
    if score < 70:
        if nsi < 0.05:
            return ("Slight nitrogen deficiency. Plan fertilizer application.", "nitrogen")
        return ("Fair health. Routine monitoring recommended.", "routine")
    return ("Excellent health. Continue current management.", "routine")


def create_recommendations(features: dict, preds: dict) -> np.ndarray:
    """
    Build a 2-D array of recommendation strings for valid pixels only.
    Also returns an action_key array for cost calculations.
    """
    log.info("Creating pixel-level recommendations …")
    H, W = features["ndvi"].shape
    score    = preds["health_score"]
    nsi      = features["nsi"]
    stress   = features["stress_idx"]
    drainage = features["drainage_score"]

    # For zone analysis we only need action_key as uint8 array
    action_map = {"nitrogen": 0, "drainage": 1, "pest": 2, "routine": 3}
    rec_key    = np.full((H, W), 3, dtype=np.uint8)

    valid = ~np.isnan(score)
    for yi in range(H):
        for xi in range(W):
            if not valid[yi, xi]:
                continue
            _, key = _rec(
                float(score[yi, xi]),
                float(nsi[yi, xi])      if not np.isnan(nsi[yi, xi])      else 0.0,
                float(stress[yi, xi])   if not np.isnan(stress[yi, xi])   else 0.5,
                float(drainage[yi, xi]) if not np.isnan(drainage[yi, xi]) else 0.0,
            )
            rec_key[yi, xi] = action_map[key]

    return rec_key   # 0=nitrogen, 1=drainage, 2=pest, 3=routine


def fast_recommendations(features: dict, preds: dict) -> np.ndarray:
    """
    Vectorised recommendation key map (fast path using numpy).
    """
    log.info("Creating vectorised recommendation map …")
    score    = preds["health_score"]
    nsi      = _fill_nan(features["nsi"])
    stress   = _fill_nan(features["stress_idx"], 0.5)
    drainage = _fill_nan(features["drainage_score"])

    # Default: routine (3)
    key = np.full_like(score, 3, dtype=np.uint8)
    valid = ~np.isnan(score)

    # Critical tier (< 30)
    c = valid & (score < 30)
    key[c & (nsi < -0.1)]                          = 0  # nitrogen
    key[c & (nsi >= -0.1) & (stress > 0.7)]        = 2  # pest
    key[c & (nsi >= -0.1) & (stress <= 0.7) & (drainage < 0)] = 1  # drainage

    # High tier (30-50)
    h = valid & (score >= 30) & (score < 50)
    key[h & (nsi < 0)]                             = 0
    key[h & (nsi >= 0) & (stress > 0.5)]           = 2
    key[h & (nsi >= 0) & (stress <= 0.5) & (drainage < DRAIN_RISK_THR)] = 1

    # Medium tier (50-70)
    m = valid & (score >= 50) & (score < 70)
    key[m & (nsi < 0.05)]                          = 0

    return key


# ══════════════════════════════════════════════════════════════════════════════
# 7. ZONE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_zones(features: dict, preds: dict, rec_key: np.ndarray,
                  pixel_m: float) -> pd.DataFrame:
    """
    Identify connected regions at each priority level.
    Returns DataFrame with zone statistics and management plan.
    """
    log.info("Analysing zones …")
    priority   = preds["priority"].copy().astype(np.int8)
    score      = preds["health_score"]
    ndvi       = features["ndvi"]
    ndre       = features["ndre"]
    nsi        = features["nsi"]
    slope      = features["slope"]

    action_names = ["nitrogen", "drainage", "pest", "routine"]
    cost_keys    = ["nitrogen", "drainage", "pest",  "routine"]
    improve_pct  = {"nitrogen": 20, "drainage": 25, "pest": 18, "routine": 5}

    pixel_ha = (pixel_m ** 2) / 10_000   # hectares per pixel

    rows = []
    for pri_level in [1, 2, 3, 4]:
        mask_pri   = priority == pri_level
        labelled, n_zones = label(mask_pri)
        log.info(f"  Priority {pri_level}: {n_zones} zones")

        for zone_id in range(1, min(n_zones + 1, 501)):   # cap at 500 zones/level
            zm = labelled == zone_id
            n_px = int(zm.sum())
            if n_px < 10:   # skip tiny zones (< 10 px)
                continue

            area_ha = n_px * pixel_ha

            def _zmean(arr):
                v = arr[zm & ~np.isnan(arr)]
                return float(v.mean()) if len(v) else np.nan

            avg_score  = _zmean(score)
            avg_ndvi   = _zmean(ndvi)
            avg_ndre   = _zmean(ndre)
            avg_nsi    = _zmean(nsi)
            avg_slope  = _zmean(slope)

            # Primary action: most common recommendation in zone
            keys_in_zone = rec_key[zm]
            primary_key  = int(np.bincount(keys_in_zone.ravel()).argmax())
            action       = action_names[primary_key]
            cost_key     = cost_keys[primary_key]

            cost_inr     = area_ha * COST_PER_HA[cost_key]
            exp_improve  = improve_pct[action]

            priority_label = {1:"Critical",2:"High",3:"Medium",4:"Low"}[pri_level]

            rows.append({
                "zone_id":            f"P{pri_level}-Z{zone_id:04d}",
                "priority_level":     pri_level,
                "priority_label":     priority_label,
                "area_ha":            round(area_ha, 4),
                "pixels":             n_px,
                "avg_health_score":   round(avg_score, 2),
                "avg_ndvi":           round(avg_ndvi, 4),
                "avg_ndre":           round(avg_ndre, 4),
                "avg_nsi":            round(avg_nsi, 4),
                "avg_slope_deg":      round(avg_slope, 2),
                "primary_action":     action,
                "estimated_cost_inr": round(cost_inr, 0),
                "expected_improvement_pct": exp_improve,
            })

    df = pd.DataFrame(rows)
    if len(df):
        df.sort_values(["priority_level", "estimated_cost_inr"],
                       ascending=[True, False], inplace=True)
    df.to_csv(OUT_DIR / "zone_management_plan.csv", index=False)
    log.info(f"Zone plan saved: {len(df)} zones → zone_management_plan.csv")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def create_report_visualization(features: dict, preds: dict,
                                models: dict) -> None:
    """6-panel visualisation saved to plant_health_report.png."""
    log.info("Creating visualisation …")

    score    = preds["health_score"]
    cat      = preds["health_cat"].astype(float)
    priority = preds["priority"].astype(float)
    ndvi     = features["ndvi"]
    nsi      = features["nsi"]
    slope    = features["slope"]

    # Mask nodata
    cat[cat < 0]      = np.nan
    priority[priority < 0] = np.nan

    fig = plt.figure(figsize=(20, 13), facecolor="#0f1117")
    fig.suptitle(
        f"Plant Health Analysis Report  •  Generated {datetime.now():%Y-%m-%d %H:%M}",
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)

    panels = [
        (gs[0, 0], score,    "RdYlGn",  "Health Score (0-100)",  None),
        (gs[0, 1], cat,      "RdYlGn",  "Health Category",       (0, 3)),
        (gs[0, 2], priority, "RdYlGn_r","Priority Level (1=Crit)",""),
        (gs[1, 0], ndvi,     "RdYlGn",  "NDVI",                  (-1, 1)),
        (gs[1, 1], nsi,      "RdYlGn",  "NSI (NDRE − NDVI)",     (-0.5, 0.5)),
        (gs[1, 2], slope,    "viridis",  "Slope (degrees)",       None),
    ]

    for spec, data, cmap, title, clim in panels:
        ax = fig.add_subplot(spec)
        ax.set_facecolor("#1a1d2e")
        im = ax.imshow(data, cmap=cmap, interpolation="nearest",
                       aspect="auto",
                       vmin=clim[0] if clim else None,
                       vmax=clim[1] if clim else None)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     ).ax.yaxis.set_tick_params(color="white", labelcolor="white")
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.axis("off")

    plt.savefig(OUT_DIR / "plant_health_report.png",
                dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    log.info("Report PNG saved.")


# ══════════════════════════════════════════════════════════════════════════════
# 9. TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(features: dict, preds: dict, models: dict,
                    zones: pd.DataFrame) -> None:
    """Generate plant_health_report.txt."""
    log.info("Writing text report …")

    score    = preds["health_score"]
    priority = preds["priority"]
    ndvi     = features["ndvi"]
    nsi      = features["nsi"]

    valid    = ~np.isnan(score)
    n_valid  = int(valid.sum())
    pixel_m  = float(abs(features.get("meta", {}).get("pixel_m", PIXEL_SIZE_M)))
    px_ha    = (PIXEL_SIZE_M ** 2) / 10_000
    total_ha = n_valid * px_ha

    s = score[valid]
    n = ndvi[valid]
    ns = nsi[valid & ~np.isnan(nsi)]

    lines = [
        "=" * 72,
        "  PLANT HEALTH ML ANALYSIS REPORT",
        f"  Generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"  Field area: {total_ha:.2f} ha  ({n_valid:,} valid pixels)",
        "=" * 72,
        "",
        "── OVERALL HEALTH STATISTICS ─────────────────────────────────────",
        f"  Average health score : {s.mean():.1f} / 100",
        f"  Median health score  : {float(np.median(s)):.1f} / 100",
        f"  Std deviation        : {s.std():.1f}",
        f"  Min / Max            : {s.min():.1f} / {s.max():.1f}",
        "",
        "── HEALTH DISTRIBUTION ───────────────────────────────────────────",
    ]

    boundaries = [(0,30,"Poor/Critical"),(30,50,"Fair/High"),(50,70,"Good/Medium"),(70,101,"Excellent/Low")]
    for lo, hi, label_ in boundaries:
        cnt = int(((s >= lo) & (s < hi)).sum())
        lines.append(f"  {label_:<22}: {cnt:>8,} px  ({cnt/n_valid*100:5.1f}%)")

    p = priority[valid]
    p_counts = {lv: int((p == lv).sum()) for lv in [1,2,3,4]}
    labels_p = {1:"Critical",2:"High",3:"Medium",4:"Low"}

    lines += [
        "",
        "── PRIORITY DISTRIBUTION ─────────────────────────────────────────",
    ]
    for lv in [1,2,3,4]:
        cnt = p_counts[lv]
        area = cnt * px_ha
        lines.append(f"  Priority {lv} ({labels_p[lv]:<8}): {cnt:>8,} px  "
                     f"({cnt/n_valid*100:5.1f}%)  ≈ {area:.3f} ha")

    lines += [
        "",
        "── KEY INSIGHTS ──────────────────────────────────────────────────",
    ]
    if len(ns):
        n_def_pct = float((ns < 0).mean()) * 100
        lines.append(f"  Nitrogen (NSI<0)   : {n_def_pct:.1f}% of field shows potential deficiency")
    ndvi_low = float((n < 0.2).mean()) * 100
    lines.append(f"  Bare Soil (NDVI<0.2): {ndvi_low:.1f}% of field may be bare / severely stressed")

    slope_arr = features["slope"][valid]
    high_slope = float((slope_arr > SLOPE_RISK_DEG).mean()) * 100
    lines.append(f"  High slope (>{SLOPE_RISK_DEG}°)   : {high_slope:.1f}% of field has terrain runoff risk")

    drain_arr = features["drainage_score"][valid]
    drain_val = drain_arr[~np.isnan(drain_arr)]
    if len(drain_val):
        poor_drain = float((drain_val < DRAIN_RISK_THR).mean()) * 100
        lines.append(f"  Poor drainage      : {poor_drain:.1f}% of field has waterlogging risk")

    lines += [
        "",
        "── MODEL PERFORMANCE ─────────────────────────────────────────────",
        f"  Regression  R²  : {models['reg_r2']:.4f}",
        f"  Regression  MAE : {models['reg_mae']:.2f} score points",
        f"  Classification accuracy: {models['cls_acc']:.4f}",
        "",
        "── ZONE MANAGEMENT SUMMARY ───────────────────────────────────────",
    ]

    if len(zones):
        total_cost    = zones["estimated_cost_inr"].sum()
        crit_zones    = int((zones["priority_level"] == 1).sum())
        avg_imp       = zones["expected_improvement_pct"].mean()
        lines += [
            f"  Total zones identified  : {len(zones)}",
            f"  Critical zones (P1)     : {crit_zones}",
            f"  Total estimated cost    : ₹{total_cost:,.0f}",
            f"  Avg expected improvement: {avg_imp:.0f}%",
            "",
            "  Top 5 zones by cost:",
        ]
        top5 = zones.nlargest(5, "estimated_cost_inr")
        for _, row in top5.iterrows():
            lines.append(
                f"    {row['zone_id']}  |  {row['area_ha']:.3f} ha  |  "
                f"Action: {row['primary_action']:<10}  |  "
                f"Cost: ₹{row['estimated_cost_inr']:,.0f}  |  "
                f"Improve: {row['expected_improvement_pct']}%"
            )

    lines += [
        "",
        "── RECOMMENDATIONS ───────────────────────────────────────────────",
        "  1. Nitrogen Management:",
        "     • Apply slow-release NPK fertiliser to flagged zones.",
        "     • Target zones where NSI < 0 or NDRE < 0.6.",
        "",
        "  2. Water Management:",
        "     • Inspect drainage channels in high-slope and poor-drainage zones.",
        "     • Consider drip irrigation for Bare Soil zones.",
        "",
        "  3. Pest / Disease Control:",
        "     • Deploy ground scouting teams to Critical-priority zones.",
        "     • Follow up with repeat UAV survey in 2 weeks.",
        "",
        "  4. Terrain Adaptation:",
        "     • On slopes >15°, use contour planting / terracing to reduce runoff.",
        "",
        "  5. Monitoring Schedule:",
        "     • Critical zones : weekly UAV scan",
        "     • High zones     : bi-weekly scan",
        "     • Medium/Low     : monthly scan",
        "",
        "=" * 72,
        "  END OF REPORT",
        "=" * 72,
    ]

    (OUT_DIR / "plant_health_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log.info("Text report saved.")


# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE GEOTIFF OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_results(preds: dict, meta: dict) -> None:
    """
    Write 4 georeferenced GeoTIFF outputs:
      plant_health_score.tif, health_category.tif,
      priority_level.tif, prediction_confidence.tif
    """
    log.info("Saving GeoTIFF outputs …")
    base_kwargs = {
        "driver": "GTiff",
        "crs":    meta["crs"],
        "transform": meta["transform"],
        "width":  meta["width"],
        "height": meta["height"],
        "count":  1,
        "compress": "lzw",
    }

    specs = [
        ("plant_health_score.tif",      preds["health_score"], np.float32, np.nan),
        ("health_category.tif",         preds["health_cat"].astype(np.float32), np.float32, -1),
        ("priority_level.tif",          preds["priority"].astype(np.float32),   np.float32, -1),
        ("prediction_confidence.tif",   preds["confidence"],  np.float32, np.nan),
    ]

    for fname, arr, dtype, nodata in specs:
        out_arr = arr.astype(dtype)
        kw = {**base_kwargs, "dtype": dtype, "nodata": nodata}
        with rasterio.open(OUT_DIR / fname, "w", **kw) as dst:
            dst.write(out_arr, 1)
        log.info(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def validate_inputs() -> None:
    for p in [NDVI_PATH, NDRE_PATH, DSM_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
    # Check same CRS
    crs_list = []
    for p in [NDVI_PATH, NDRE_PATH, DSM_PATH]:
        with rasterio.open(p) as src:
            crs_list.append(src.crs.to_epsg())
    if len(set(crs_list)) > 1:
        log.warning(f"CRS mismatch across inputs: {crs_list}. Will reproject.")
    log.info(f"Input validation passed. CRS: {crs_list}")


def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("Plant Health ML Analysis System")
    log.info(f"Output directory: {OUT_DIR}")
    log.info("=" * 60)

    # Step 1: Validate
    validate_inputs()

    # Step 2: Load data
    data = load_all_inputs()

    # Step 3: Feature engineering
    features = create_features(data)
    features["meta"] = data["meta"]

    # Step 4: Prepare training data
    X, y_reg, y_cls, valid_idx = prepare_training_data(features)

    # Step 5: Train models
    # Skip training if saved models exist
    reg_pkl = OUT_DIR / "xgboost_regression_model.pkl"
    clf_pkl = OUT_DIR / "xgboost_classification_model.pkl"
    scl_pkl = OUT_DIR / "feature_scaler.pkl"

    if reg_pkl.exists() and clf_pkl.exists() and scl_pkl.exists():
        log.info("Loading saved models (delete PKL files to retrain) …")
        models = {
            "reg": joblib.load(reg_pkl),
            "clf": joblib.load(clf_pkl),
            "scaler": joblib.load(scl_pkl),
            "reg_r2": float("nan"), "reg_mae": float("nan"),
            "cls_acc": float("nan"), "cls_report": "N/A (loaded from file)",
            "fi_reg": pd.read_csv(OUT_DIR / "feature_importance.csv", index_col=0)["regression"],
            "fi_cls": pd.read_csv(OUT_DIR / "feature_importance.csv", index_col=0)["classification"],
        }
    else:
        models = train_xgboost_models(X, y_reg, y_cls)

    # Step 6: Generate predictions
    preds = generate_predictions(features, models)

    # Step 7: Recommendations (vectorised)
    rec_key = fast_recommendations(features, preds)

    # Step 8: Zone analysis
    zones = analyze_zones(features, preds, rec_key, data["pixel_m"])

    # Step 9: Visualisation
    create_report_visualization(features, preds, models)

    # Step 10: Text report
    generate_report(features, preds, models, zones)

    # Step 11: Save GeoTIFFs
    save_results(preds, data["meta"])

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"DONE in {elapsed:.1f}s.  All outputs in: {OUT_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
