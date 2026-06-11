#!/usr/bin/env python3
"""
Step 7 — ML Predictions & Insights Generation
==============================================
Uses trained models to generate predictions for all plants and creates
actionable insights beyond just generating maps.

Outputs:
  • plants_with_predictions.json  — All plants with ML predictions
  • ml_insights_report.json       — Actionable insights and recommendations
  • sector_analysis.csv           — Performance metrics by sector
  • high_priority_plants.csv      — Plants requiring immediate attention
"""

import json
import pickle
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent / "outputs"
ML_DIR = OUT_DIR / "ml_models"

ENRICHED_JSON = OUT_DIR / "plants_ndvi_enriched.json"


# ══════════════════════════════════════════════════════════════════════════════
# Load Models
# ══════════════════════════════════════════════════════════════════════════════

def load_models() -> Dict[str, Dict[str, Any]]:
    """Load all trained ML models."""
    print("=" * 60)
    print("Loading Trained Models...")
    print("=" * 60)
    
    models = {
        'growth_stage': {},
        'yield': {},
        'nitrogen': {},
    }
    
    for task in models.keys():
        for algo in ['random_forest', 'xgboost', 'lightgbm']:
            model_path = ML_DIR / f"{task}_{algo}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    models[task][algo] = pickle.load(f)
                print(f"[✓] Loaded {task} - {algo}")
            else:
                print(f"[SKIP] {task} - {algo} not found")
    
    return models


# ══════════════════════════════════════════════════════════════════════════════
# Feature Engineering (same as training)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for prediction (must match training)."""
    from sklearn.preprocessing import LabelEncoder
    
    # Spatial features
    df['spatial_cluster'] = df['row_index'] // 10 * 1000 + df['col_index'] // 10
    
    # NDVI-based features
    df['ndvi_tif_diff'] = df['ndvi'] - df['tif_ndvi'].fillna(df['ndvi'])
    df['ndvi_variability'] = df['ndvi_max'] - df['ndvi']
    
    # Encode categorical sector
    le_sector = LabelEncoder()
    df['sector_encoded'] = le_sector.fit_transform(df['sector_label'].fillna('UNKNOWN'))
    
    # NDVI category encoding
    ndvi_cat_map = {
        'Water / Non-vegetation': 0,
        'Bare / Stressed': 1,
        'Sparse Vegetation': 2,
        'Moderate Vegetation': 3,
        'Dense Vegetation': 4,
        'Very Dense Vegetation': 5,
        'No Data': -1
    }
    df['ndvi_category_encoded'] = df['ndvi_category'].map(ndvi_cat_map).fillna(-1)
    
    return df


def get_feature_columns() -> list:
    """Feature columns (must match training)."""
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


# ══════════════════════════════════════════════════════════════════════════════
# Make Predictions
# ══════════════════════════════════════════════════════════════════════════════

def make_predictions(df: pd.DataFrame, models: Dict) -> pd.DataFrame:
    """Generate predictions using all available models."""
    print("\n" + "=" * 60)
    print("Generating Predictions...")
    print("=" * 60)
    
    feature_cols = get_feature_columns()
    X = df[feature_cols].fillna(0)
    
    stage_names = ['Seedling', 'Vegetative', 'Flowering', 'Fruiting', 'Mature']
    nitrogen_names = ['Healthy', 'Deficient']
    
    # Use the best model (or ensemble average for regression)
    
    # 1. Growth Stage (use Random Forest)
    if 'random_forest' in models['growth_stage']:
        print("[INFO] Predicting growth stages...")
        model = models['growth_stage']['random_forest']
        growth_pred = model.predict(X)
        df['predicted_growth_stage'] = growth_pred
        df['predicted_growth_stage_name'] = [stage_names[int(s)] for s in growth_pred]
        
        # Get prediction probabilities
        proba = model.predict_proba(X)
        df['growth_stage_confidence'] = proba.max(axis=1).round(4)
    
    # 2. Yield Prediction (ensemble average if multiple models available)
    print("[INFO] Predicting yields...")
    yield_preds = []
    for algo, model in models['yield'].items():
        pred = model.predict(X)
        yield_preds.append(pred)
    
    if yield_preds:
        # Average predictions from all models
        df['predicted_yield_kg'] = np.mean(yield_preds, axis=0).round(2)
        
        # If multiple models, also show std as uncertainty
        if len(yield_preds) > 1:
            df['yield_prediction_std'] = np.std(yield_preds, axis=0).round(2)
    
    # 3. Nitrogen Deficiency (use Random Forest)
    if 'random_forest' in models['nitrogen']:
        print("[INFO] Predicting nitrogen deficiency...")
        model = models['nitrogen']['random_forest']
        nitrogen_pred = model.predict(X)
        df['predicted_nitrogen_deficiency'] = nitrogen_pred
        df['predicted_nitrogen_status'] = [nitrogen_names[int(n)] for n in nitrogen_pred]
        
        # Get prediction probabilities
        proba = model.predict_proba(X)
        df['nitrogen_deficiency_confidence'] = proba.max(axis=1).round(4)
    
    print(f"[✓] Predictions complete for {len(df):,} plants")
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Generate Insights
# ══════════════════════════════════════════════════════════════════════════════

def generate_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate actionable insights from predictions."""
    print("\n" + "=" * 60)
    print("Generating Actionable Insights...")
    print("=" * 60)
    
    insights = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'total_plants_analyzed': len(df),
    }
    
    # 1. Growth Stage Distribution
    if 'predicted_growth_stage_name' in df.columns:
        stage_dist = df['predicted_growth_stage_name'].value_counts().to_dict()
        insights['growth_stage_distribution'] = stage_dist
        
        print("\n[INSIGHT] Growth Stage Distribution:")
        for stage, count in stage_dist.items():
            pct = count / len(df) * 100
            print(f"  {stage:<15} : {count:>6,} ({pct:.1f}%)")
    
    # 2. Yield Analysis
    if 'predicted_yield_kg' in df.columns:
        total_yield = df['predicted_yield_kg'].sum()
        avg_yield = df['predicted_yield_kg'].mean()
        
        # Categorize yield performance
        low_yield = df[df['predicted_yield_kg'] < 20]
        high_yield = df[df['predicted_yield_kg'] > 60]
        
        insights['yield_analysis'] = {
            'total_predicted_yield_kg': round(total_yield, 2),
            'average_yield_per_plant_kg': round(avg_yield, 2),
            'median_yield_kg': round(df['predicted_yield_kg'].median(), 2),
            'low_yield_plants': len(low_yield),
            'high_yield_plants': len(high_yield),
            'yield_range': {
                'min': round(df['predicted_yield_kg'].min(), 2),
                'max': round(df['predicted_yield_kg'].max(), 2),
            }
        }
        
        print("\n[INSIGHT] Yield Predictions:")
        print(f"  Total predicted yield:  {total_yield:,.0f} kg")
        print(f"  Average per plant:      {avg_yield:.2f} kg")
        print(f"  Low-yield plants (<20): {len(low_yield):,} ({len(low_yield)/len(df)*100:.1f}%)")
        print(f"  High-yield plants (>60): {len(high_yield):,} ({len(high_yield)/len(df)*100:.1f}%)")
    
    # 3. Nitrogen Deficiency Alert
    if 'predicted_nitrogen_deficiency' in df.columns:
        deficient = df[df['predicted_nitrogen_deficiency'] == 1]
        deficient_pct = len(deficient) / len(df) * 100
        
        insights['nitrogen_analysis'] = {
            'deficient_plants': len(deficient),
            'deficient_percentage': round(deficient_pct, 2),
            'healthy_plants': len(df) - len(deficient),
        }
        
        print("\n[INSIGHT] Nitrogen Status:")
        print(f"  Deficient plants: {len(deficient):,} ({deficient_pct:.1f}%)")
        print(f"  Healthy plants:   {len(df) - len(deficient):,}")
        
        # Critical: plants with both low NDVI and nitrogen deficiency
        critical = deficient[deficient['ndvi'] < 0.3]
        if len(critical) > 0:
            insights['critical_attention_needed'] = len(critical)
            print(f"\n  ⚠️  CRITICAL: {len(critical):,} plants need immediate attention")
            print(f"      (Low NDVI + Nitrogen deficient)")
    
    # 4. Sector Performance Analysis
    if 'sector_label' in df.columns and 'predicted_yield_kg' in df.columns:
        sector_stats = df.groupby('sector_label').agg({
            'predicted_yield_kg': ['mean', 'sum', 'count'],
            'predicted_nitrogen_deficiency': 'sum',
            'ndvi': 'mean'
        }).round(2)
        
        insights['sector_performance'] = {}
        print("\n[INSIGHT] Performance by Sector:")
        print(f"  {'Sector':<15} {'Avg Yield':<12} {'Total Yield':<15} {'Deficient':<12} {'Avg NDVI'}")
        print("  " + "-" * 70)
        
        for sector in sector_stats.index:
            avg_yield = sector_stats.loc[sector, ('predicted_yield_kg', 'mean')]
            total_yield = sector_stats.loc[sector, ('predicted_yield_kg', 'sum')]
            n_def = int(sector_stats.loc[sector, ('predicted_nitrogen_deficiency', 'sum')])
            avg_ndvi = sector_stats.loc[sector, ('ndvi', 'mean')]
            plant_count = int(sector_stats.loc[sector, ('predicted_yield_kg', 'count')])
            
            insights['sector_performance'][str(sector)] = {
                'plant_count': plant_count,
                'avg_yield_kg': float(avg_yield),
                'total_yield_kg': float(total_yield),
                'deficient_plants': n_def,
                'avg_ndvi': float(avg_ndvi)
            }
            
            print(f"  {str(sector):<15} {avg_yield:<12.2f} {total_yield:<15.0f} {n_def:<12} {avg_ndvi:.3f}")
    
    # 5. Recommendations
    recommendations = []
    
    if 'predicted_nitrogen_deficiency' in df.columns:
        deficient_pct = insights.get('nitrogen_analysis', {}).get('deficient_percentage', 0)
        if deficient_pct > 20:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Apply nitrogen fertilizer',
                'reason': f'{deficient_pct:.1f}% of plants show nitrogen deficiency',
                'affected_plants': int(insights['nitrogen_analysis']['deficient_plants'])
            })
    
    if 'predicted_yield_kg' in df.columns:
        low_yield_pct = len(df[df['predicted_yield_kg'] < 20]) / len(df) * 100
        if low_yield_pct > 15:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Investigate low-yield areas',
                'reason': f'{low_yield_pct:.1f}% of plants have predicted yields below 20kg',
                'affected_plants': int(len(df[df['predicted_yield_kg'] < 20]))
            })
    
    if 'predicted_growth_stage_name' in df.columns:
        # Check for growth stage synchronization
        stage_counts = df['predicted_growth_stage_name'].value_counts()
        if len(stage_counts) > 3 and stage_counts.iloc[0] / len(df) < 0.5:
            recommendations.append({
                'priority': 'LOW',
                'action': 'Review planting schedule',
                'reason': 'Plants are at multiple growth stages, indicating inconsistent planting',
                'note': 'Consider synchronized planting for better harvest management'
            })
    
    insights['recommendations'] = recommendations
    
    if recommendations:
        print("\n[RECOMMENDATIONS]")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n  {i}. [{rec['priority']}] {rec['action']}")
            print(f"     Reason: {rec['reason']}")
            if 'affected_plants' in rec:
                print(f"     Affected: {rec['affected_plants']:,} plants")
    
    return insights


# ══════════════════════════════════════════════════════════════════════════════
# Save Outputs
# ══════════════════════════════════════════════════════════════════════════════

def save_prediction_outputs(df: pd.DataFrame, insights: Dict) -> Dict[str, str]:
    """Save all prediction outputs."""
    print("\n" + "=" * 60)
    print("Saving Outputs...")
    print("=" * 60)
    
    output_paths = {}
    
    # 1. Full predictions JSON
    predictions_json = OUT_DIR / "plants_with_predictions.json"
    with open(predictions_json, 'w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=2)
    output_paths['predictions_json'] = str(predictions_json)
    print(f"[✓] {predictions_json}")
    
    # 2. Insights report
    insights_json = OUT_DIR / "ml_insights_report.json"
    with open(insights_json, 'w') as f:
        json.dump(insights, f, indent=2)
    output_paths['insights_json'] = str(insights_json)
    print(f"[✓] {insights_json}")
    
    # 3. Sector analysis CSV
    if 'sector_label' in df.columns:
        sector_csv = OUT_DIR / "sector_analysis.csv"
        sector_analysis = df.groupby('sector_label').agg({
            'ndvi': ['mean', 'std'],
            'predicted_yield_kg': ['mean', 'sum'] if 'predicted_yield_kg' in df.columns else ['count'],
            'predicted_nitrogen_deficiency': 'sum' if 'predicted_nitrogen_deficiency' in df.columns else 'count',
        }).round(3)
        sector_analysis.to_csv(sector_csv)
        output_paths['sector_csv'] = str(sector_csv)
        print(f"[✓] {sector_csv}")
    
    # 4. High-priority plants CSV (need attention)
    high_priority = df[
        ((df.get('predicted_nitrogen_deficiency', 0) == 1) & (df['ndvi'] < 0.35)) |
        (df.get('predicted_yield_kg', 100) < 15)
    ].copy()
    
    if len(high_priority) > 0:
        priority_cols = ['plant_id', 'sector_label', 'row_index', 'col_index', 
                        'lat', 'lon', 'ndvi', 'predicted_yield_kg', 
                        'predicted_nitrogen_status', 'predicted_growth_stage_name']
        priority_cols = [c for c in priority_cols if c in high_priority.columns]
        
        priority_csv = OUT_DIR / "high_priority_plants.csv"
        high_priority[priority_cols].to_csv(priority_csv, index=False)
        output_paths['priority_csv'] = str(priority_csv)
        print(f"[✓] {priority_csv}  ({len(high_priority):,} plants)")
    
    # 5. Summary predictions CSV
    pred_cols = ['plant_id', 'sector_label', 'row_index', 'col_index',
                 'lat', 'lon', 'ndvi', 'predicted_growth_stage_name',
                 'predicted_yield_kg', 'predicted_nitrogen_status']
    pred_cols = [c for c in pred_cols if c in df.columns]
    
    predictions_csv = OUT_DIR / "plants_predictions_summary.csv"
    df[pred_cols].to_csv(predictions_csv, index=False)
    output_paths['predictions_csv'] = str(predictions_csv)
    print(f"[✓] {predictions_csv}")
    
    return output_paths


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the ML prediction and insights pipeline."""
    t_start = time.time()
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 8 + "ML PREDICTIONS & INSIGHTS — STEP 7" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝\n")
    
    # Load data
    with open(ENRICHED_JSON, 'r') as f:
        plants = json.load(f)
    
    df = pd.DataFrame(plants)
    df = df[df['is_noise'] == False].copy()
    df = df[df['ndvi'].notna()].copy()
    
    print(f"[INFO] Loaded {len(df):,} valid plants with NDVI data")
    
    # Prepare features
    df = prepare_features(df)
    
    # Load models
    models = load_models()
    
    # Make predictions
    df = make_predictions(df, models)
    
    # Generate insights
    insights = generate_insights(df)
    
    # Save outputs
    output_paths = save_prediction_outputs(df, insights)
    
    # Summary
    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)
    
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "PREDICTIONS COMPLETE!" + " " * 25 + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Total time: {mins}m {secs}s")
    print("║")
    print("║  Output Files:")
    for key, path in output_paths.items():
        filename = Path(path).name
        print(f"║    • {filename}")
    print("║")
    print("║  Key Insights:")
    if 'yield_analysis' in insights:
        total = insights['yield_analysis']['total_predicted_yield_kg']
        print(f"║    • Total predicted yield: {total:,.0f} kg")
    if 'nitrogen_analysis' in insights:
        deficient = insights['nitrogen_analysis']['deficient_plants']
        print(f"║    • Plants needing fertilizer: {deficient:,}")
    if 'recommendations' in insights and insights['recommendations']:
        print(f"║    • Action items: {len(insights['recommendations'])}")
    print("╚" + "═" * 58 + "╝\n")


if __name__ == "__main__":
    main()
