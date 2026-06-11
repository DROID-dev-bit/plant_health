#!/usr/bin/env python3
"""
Step 6 — Machine Learning Training Pipeline
============================================
Trains Random Forest, XGBoost, and LightGBM models to predict:
  • Growth Stage (classification)
  • Yield Prediction (regression)
  • Nitrogen Deficiency (classification)

Features used:
  - NDVI (from CSV matching)
  - TIF NDVI (direct raster sampling)
  - Spatial features (row, col, sector)
  - Plant area
  - NDVI statistics (mean, max)

This creates a foundation for predictive agriculture insights.
"""

import json
import pickle
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent / "outputs"
ML_DIR = OUT_DIR / "ml_models"
ML_DIR.mkdir(parents=True, exist_ok=True)

ENRICHED_JSON = OUT_DIR / "plants_ndvi_enriched.json"


# ══════════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare_data(json_path: Path) -> pd.DataFrame:
    """Load enriched plant data and convert to DataFrame with ML features."""
    print("=" * 60)
    print("Loading and preparing data for ML...")
    print("=" * 60)
    
    with open(json_path, 'r') as f:
        plants = json.load(f)
    
    df = pd.DataFrame(plants)
    
    # Filter out noise detections
    df = df[df['is_noise'] == False].copy()
    
    print(f"[INFO] Loaded {len(df):,} valid plants (noise excluded)")
    
    # Remove plants with no NDVI data
    df = df[df['ndvi'].notna()].copy()
    print(f"[INFO] {len(df):,} plants with NDVI data")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features for ML models."""
    print("\n[INFO] Engineering features...")
    
    # Spatial features
    df['spatial_cluster'] = df['row_index'] // 10 * 1000 + df['col_index'] // 10
    
    # NDVI-based features
    df['ndvi_tif_diff'] = df['ndvi'] - df['tif_ndvi'].fillna(df['ndvi'])
    df['ndvi_variability'] = df['ndvi_max'] - df['ndvi']
    
    # Encode categorical sector
    le_sector = LabelEncoder()
    df['sector_encoded'] = le_sector.fit_transform(df['sector_label'].fillna('UNKNOWN'))
    
    # NDVI category encoding — pineapple-specific thresholds
    ndvi_cat_map = {
        'Bare Soil':  0,   # < 0.2  — replanting required
        'Unhealthy':  1,   # 0.2–0.6 — scouting needed
        'Healthy':    2,   # 0.6–0.8 — maintain protocols
        'Saturated':  3,   # ≥ 0.8  — use NDRE
        'No Data':   -1,
    }
    df['ndvi_category_encoded'] = df['ndvi_category'].map(ndvi_cat_map).fillna(-1)
    
    print(f"[✓] Feature engineering complete. Total features: {len(df.columns)}")
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic Target Generation (for demonstration)
# ══════════════════════════════════════════════════════════════════════════════
# NOTE: In production, these would be replaced with actual ground-truth labels

def generate_synthetic_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic targets for demonstration purposes.
    
    In a real scenario, you would:
    1. Collect ground truth data through field surveys
    2. Use historical yield records
    3. Get lab analysis for nitrogen levels
    4. Use expert annotations for growth stages
    """
    print("\n" + "=" * 60)
    print("Generating Synthetic Targets (Demo)")
    print("=" * 60)
    print("[WARN] Using synthetic targets for demonstration.")
    print("[WARN] Replace with actual ground-truth data for production use.")
    
    np.random.seed(42)
    n = len(df)
    
    # 1. Growth Stage (based on NDVI with noise)
    # Stages: Seedling, Vegetative, Flowering, Fruiting, Mature
    ndvi = df['ndvi'].values
    growth_stage = np.zeros(n, dtype=int)
    
    for i, v in enumerate(ndvi):
        if v < 0.2:
            growth_stage[i] = 0  # Seedling   — bare soil / just emerged
        elif v < 0.4:
            growth_stage[i] = 1  # Vegetative — unhealthy / low canopy
        elif v < 0.6:
            growth_stage[i] = 2  # Flowering  — lower unhealthy range
        elif v < 0.8:
            growth_stage[i] = 3  # Fruiting   — healthy NDVI band
        else:
            growth_stage[i] = 4  # Mature     — saturated NDVI / peak season
    
    # Add some noise to make it realistic
    noise_mask = np.random.rand(n) < 0.15
    growth_stage[noise_mask] = np.random.randint(0, 5, size=noise_mask.sum())
    
    df['growth_stage'] = growth_stage
    stage_names = ['Seedling', 'Vegetative', 'Flowering', 'Fruiting', 'Mature']
    df['growth_stage_name'] = [stage_names[s] for s in growth_stage]
    
    # 2. Yield Prediction (kg per plant, based on NDVI + area + noise)
    # Higher NDVI and larger area = better yield
    base_yield = (ndvi * 50) + (df['area_px'].values / 1000 * 10)
    noise_yield = np.random.normal(0, 5, n)
    df['yield_kg'] = np.clip(base_yield + noise_yield, 0, 100)
    
    # 3. Nitrogen Deficiency (binary: 0 = healthy, 1 = deficient)
    # Low NDVI + specific spatial patterns indicate deficiency
    n_deficiency = np.zeros(n, dtype=int)
    
    for i, v in enumerate(ndvi):
        if v < 0.2:
            n_deficiency[i] = 1  # Bare soil — definite deficiency / no crop
        elif v < 0.6 and np.random.rand() < 0.55:
            n_deficiency[i] = 1  # Unhealthy NDVI band — high deficiency probability
        elif np.random.rand() < 0.04:
            n_deficiency[i] = 1  # Healthy plants — low background deficiency rate
    
    df['nitrogen_deficiency'] = n_deficiency
    df['nitrogen_status'] = df['nitrogen_deficiency'].map({0: 'Healthy', 1: 'Deficient'})
    
    print(f"\n[INFO] Growth Stage Distribution:")
    for stage, count in df['growth_stage_name'].value_counts().items():
        print(f"  {stage:<15} : {count:>6,} ({count/len(df)*100:.1f}%)")
    
    print(f"\n[INFO] Nitrogen Status Distribution:")
    for status, count in df['nitrogen_status'].value_counts().items():
        print(f"  {status:<15} : {count:>6,} ({count/len(df)*100:.1f}%)")
    
    print(f"\n[INFO] Yield Statistics:")
    print(f"  Mean:   {df['yield_kg'].mean():.2f} kg")
    print(f"  Median: {df['yield_kg'].median():.2f} kg")
    print(f"  Min:    {df['yield_kg'].min():.2f} kg")
    print(f"  Max:    {df['yield_kg'].max():.2f} kg")
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Model Training Functions
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_columns() -> list:
    """Define which columns to use as features."""
    return [
        'ndvi',
        'ndvi_mean',
        'ndvi_max',
        'tif_ndvi',
        'ndvi_tif_diff',
        'ndvi_variability',
        'area_px',
        'row_index',
        'col_index',
        'sector_encoded',
        'ndvi_category_encoded',
        'spatial_cluster',
    ]


def train_growth_stage_models(df: pd.DataFrame) -> Dict[str, Any]:
    """Train classifiers for growth stage prediction."""
    print("\n" + "=" * 60)
    print("Training Growth Stage Prediction Models")
    print("=" * 60)
    
    feature_cols = get_feature_columns()
    X = df[feature_cols].fillna(0)
    y = df['growth_stage']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Training set: {len(X_train):,} samples")
    print(f"[INFO] Test set:     {len(X_test):,} samples")
    
    models = {}
    results = {}
    
    # Random Forest
    print("\n[1/3] Training Random Forest...")
    t0 = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    models['random_forest'] = rf_model
    results['random_forest'] = {
        'accuracy': round(rf_acc, 4),
        'training_time': round(time.time() - t0, 2),
        'predictions': rf_pred.tolist()
    }
    print(f"  Accuracy: {rf_acc:.4f}  ({time.time()-t0:.1f}s)")
    
    # Try XGBoost if available
    try:
        import xgboost as xgb
        print("\n[2/3] Training XGBoost...")
        t0 = time.time()
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        models['xgboost'] = xgb_model
        results['xgboost'] = {
            'accuracy': round(xgb_acc, 4),
            'training_time': round(time.time() - t0, 2),
            'predictions': xgb_pred.tolist()
        }
        print(f"  Accuracy: {xgb_acc:.4f}  ({time.time()-t0:.1f}s)")
    except ImportError:
        print("\n[2/3] XGBoost not available (install: pip install xgboost)")
        results['xgboost'] = None
    
    # Try LightGBM if available
    try:
        import lightgbm as lgb
        print("\n[3/3] Training LightGBM...")
        t0 = time.time()
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        models['lightgbm'] = lgb_model
        results['lightgbm'] = {
            'accuracy': round(lgb_acc, 4),
            'training_time': round(time.time() - t0, 2),
            'predictions': lgb_pred.tolist()
        }
        print(f"  Accuracy: {lgb_acc:.4f}  ({time.time()-t0:.1f}s)")
    except ImportError:
        print("\n[3/3] LightGBM not available (install: pip install lightgbm)")
        results['lightgbm'] = None
    
    # Save models
    for name, model in models.items():
        model_path = ML_DIR / f"growth_stage_{name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"[SAVED] {model_path}")
    
    # Feature importance (from Random Forest)
    feature_importance = dict(zip(feature_cols, rf_model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: -x[1])[:5]
    print(f"\n[INFO] Top 5 Features for Growth Stage:")
    for feat, imp in top_features:
        print(f"  {feat:<25} : {imp:.4f}")
    
    return {
        'models': models,
        'results': results,
        'feature_importance': {k: round(v, 6) for k, v in feature_importance.items()},
        'test_accuracy': {k: v['accuracy'] for k, v in results.items() if v},
        'y_test': y_test.tolist(),
    }


def train_yield_models(df: pd.DataFrame) -> Dict[str, Any]:
    """Train regressors for yield prediction."""
    print("\n" + "=" * 60)
    print("Training Yield Prediction Models")
    print("=" * 60)
    
    feature_cols = get_feature_columns()
    X = df[feature_cols].fillna(0)
    y = df['yield_kg']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"[INFO] Training set: {len(X_train):,} samples")
    print(f"[INFO] Test set:     {len(X_test):,} samples")
    
    models = {}
    results = {}
    
    # Random Forest
    print("\n[1/3] Training Random Forest Regressor...")
    t0 = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    models['random_forest'] = rf_model
    results['random_forest'] = {
        'r2_score': round(rf_r2, 4),
        'mae': round(rf_mae, 4),
        'rmse': round(rf_rmse, 4),
        'training_time': round(time.time() - t0, 2),
    }
    print(f"  R² Score: {rf_r2:.4f}  |  MAE: {rf_mae:.2f} kg  |  RMSE: {rf_rmse:.2f} kg  ({time.time()-t0:.1f}s)")
    
    # Try XGBoost if available
    try:
        import xgboost as xgb
        print("\n[2/3] Training XGBoost Regressor...")
        t0 = time.time()
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_pred)
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        models['xgboost'] = xgb_model
        results['xgboost'] = {
            'r2_score': round(xgb_r2, 4),
            'mae': round(xgb_mae, 4),
            'rmse': round(xgb_rmse, 4),
            'training_time': round(time.time() - t0, 2),
        }
        print(f"  R² Score: {xgb_r2:.4f}  |  MAE: {xgb_mae:.2f} kg  |  RMSE: {xgb_rmse:.2f} kg  ({time.time()-t0:.1f}s)")
    except ImportError:
        print("\n[2/3] XGBoost not available")
        results['xgboost'] = None
    
    # Try LightGBM if available
    try:
        import lightgbm as lgb
        print("\n[3/3] Training LightGBM Regressor...")
        t0 = time.time()
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_r2 = r2_score(y_test, lgb_pred)
        lgb_mae = mean_absolute_error(y_test, lgb_pred)
        lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
        models['lightgbm'] = lgb_model
        results['lightgbm'] = {
            'r2_score': round(lgb_r2, 4),
            'mae': round(lgb_mae, 4),
            'rmse': round(lgb_rmse, 4),
            'training_time': round(time.time() - t0, 2),
        }
        print(f"  R² Score: {lgb_r2:.4f}  |  MAE: {lgb_mae:.2f} kg  |  RMSE: {lgb_rmse:.2f} kg  ({time.time()-t0:.1f}s)")
    except ImportError:
        print("\n[3/3] LightGBM not available")
        results['lightgbm'] = None
    
    # Save models
    for name, model in models.items():
        model_path = ML_DIR / f"yield_{name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"[SAVED] {model_path}")
    
    return {
        'models': models,
        'results': results,
        'y_test': y_test.tolist(),
    }


def train_nitrogen_models(df: pd.DataFrame) -> Dict[str, Any]:
    """Train classifiers for nitrogen deficiency detection."""
    print("\n" + "=" * 60)
    print("Training Nitrogen Deficiency Detection Models")
    print("=" * 60)
    
    feature_cols = get_feature_columns()
    X = df[feature_cols].fillna(0)
    y = df['nitrogen_deficiency']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Training set: {len(X_train):,} samples")
    print(f"[INFO] Test set:     {len(X_test):,} samples")
    
    models = {}
    results = {}
    
    # Random Forest
    print("\n[1/3] Training Random Forest...")
    t0 = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle imbalanced classes
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    models['random_forest'] = rf_model
    results['random_forest'] = {
        'accuracy': round(rf_acc, 4),
        'training_time': round(time.time() - t0, 2),
    }
    print(f"  Accuracy: {rf_acc:.4f}  ({time.time()-t0:.1f}s)")
    
    # Try XGBoost if available
    try:
        import xgboost as xgb
        print("\n[2/3] Training XGBoost...")
        t0 = time.time()
        
        # Calculate scale for imbalanced classes
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        models['xgboost'] = xgb_model
        results['xgboost'] = {
            'accuracy': round(xgb_acc, 4),
            'training_time': round(time.time() - t0, 2),
        }
        print(f"  Accuracy: {xgb_acc:.4f}  ({time.time()-t0:.1f}s)")
    except ImportError:
        print("\n[2/3] XGBoost not available")
        results['xgboost'] = None
    
    # Try LightGBM if available
    try:
        import lightgbm as lgb
        print("\n[3/3] Training LightGBM...")
        t0 = time.time()
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        models['lightgbm'] = lgb_model
        results['lightgbm'] = {
            'accuracy': round(lgb_acc, 4),
            'training_time': round(time.time() - t0, 2),
        }
        print(f"  Accuracy: {lgb_acc:.4f}  ({time.time()-t0:.1f}s)")
    except ImportError:
        print("\n[3/3] LightGBM not available")
        results['lightgbm'] = None
    
    # Save models
    for name, model in models.items():
        model_path = ML_DIR / f"nitrogen_{name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"[SAVED] {model_path}")
    
    return {
        'models': models,
        'results': results,
        'y_test': y_test.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main Training Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the complete ML training pipeline."""
    t_start = time.time()
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "ML TRAINING PIPELINE — STEP 6" + " " * 19 + "║")
    print("╚" + "═" * 58 + "╝\n")
    
    # Load and prepare data
    df = load_and_prepare_data(ENRICHED_JSON)
    df = engineer_features(df)
    df = generate_synthetic_targets(df)
    
    # Train all models
    growth_results = train_growth_stage_models(df)
    yield_results = train_yield_models(df)
    nitrogen_results = train_nitrogen_models(df)
    
    # Compile training summary
    summary = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'total_plants_used': len(df),
        'feature_columns': get_feature_columns(),
        'growth_stage': {
            'models_trained': list(growth_results['results'].keys()),
            'best_accuracy': max([v['accuracy'] for v in growth_results['results'].values() if v]),
            'feature_importance': growth_results['feature_importance'],
        },
        'yield_prediction': {
            'models_trained': list(yield_results['results'].keys()),
            'results': {k: v for k, v in yield_results['results'].items() if v},
        },
        'nitrogen_deficiency': {
            'models_trained': list(nitrogen_results['results'].keys()),
            'best_accuracy': max([v['accuracy'] for v in nitrogen_results['results'].values() if v]),
        },
    }
    
    # Save training summary
    summary_path = ML_DIR / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "TRAINING COMPLETE!" + " " * 22 + "║")
    print("╠" + "═" * 58 + "╣")
    
    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)
    
    print(f"║  Total time: {mins}m {secs}s")
    print("║")
    print(f"║  Models saved in: {ML_DIR}")
    print(f"║  Training summary: {summary_path}")
    print("║")
    print("║  Next Steps:")
    print("║    • Run step7_ml_predictions.py to generate predictions")
    print("║    • Replace synthetic targets with real ground-truth data")
    print("║    • Tune hyperparameters for better performance")
    print("╚" + "═" * 58 + "╝\n")
    
    return summary


if __name__ == "__main__":
    main()
