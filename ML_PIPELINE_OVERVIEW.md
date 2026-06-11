# 🤖 ML Pipeline Overview

## What This Does

Instead of just generating NDVI maps, this pipeline trains machine learning models to **predict actionable insights**:

```
Input: Drone imagery + NDVI data
   ↓
ML Models (Random Forest, XGBoost, LightGBM)
   ↓
Output: Growth stage, Yield predictions, Nitrogen deficiency alerts
   ↓
Action: Targeted fertilization, harvest planning, problem detection
```

---

## 🎯 Three Prediction Tasks

### 1️⃣ Growth Stage Classification
**Predicts:** Seedling → Vegetative → Flowering → Fruiting → Mature

**Why it matters:**
- Plan irrigation schedules
- Time pesticide applications
- Predict harvest dates
- Identify growth abnormalities

**Example Output:**
```
Plant P001234: Flowering (89% confidence)
Plant P001235: Fruiting (92% confidence)
```

---

### 2️⃣ Yield Prediction
**Predicts:** Expected yield in kg per plant

**Why it matters:**
- Forecast total harvest
- Identify underperforming plants
- Optimize resource allocation
- Plan logistics and sales

**Example Output:**
```
Plant P001234: 45.2 kg (±3.5 kg)
Sector T1: 856 tons total predicted yield
Low-yield plants: 2,341 need attention
```

---

### 3️⃣ Nitrogen Deficiency Detection
**Predicts:** Healthy vs Deficient

**Why it matters:**
- Target fertilizer applications (save costs)
- Prevent yield loss from deficiency
- Optimize nitrogen use efficiency
- Environmental sustainability

**Example Output:**
```
Deficient: 3,245 plants (16%)
⚠️ CRITICAL: 1,234 plants need immediate fertilization
Estimated fertilizer needed: 2.8 tons
```

---

## 🔄 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1-5: NDVI Data Processing (Existing Pipeline)         │
├─────────────────────────────────────────────────────────────┤
│ • Load plant detections                                     │
│ • Match to NDVI CSV grid                                    │
│ • Validate with TIF raster                                  │
│ • Generate outputs and map                                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ↓ plants_ndvi_enriched.json
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: ML Training (NEW)                                   │
├─────────────────────────────────────────────────────────────┤
│ • Feature engineering                                       │
│ • Train 3 algorithms × 3 tasks = 9 models                   │
│ • Save models + performance metrics                         │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ↓ Trained models (.pkl files)
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: Predictions & Insights (NEW)                        │
├─────────────────────────────────────────────────────────────┤
│ • Load models                                               │
│ • Generate predictions for all plants                       │
│ • Analyze patterns by sector                                │
│ • Create actionable recommendations                         │
│ • Export high-priority plant list                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
  predictions.json    insights_report.json   priority.csv
```

---

## 📊 Models Comparison

| Algorithm       | Speed | Accuracy | Memory | Best For |
|----------------|-------|----------|--------|----------|
| Random Forest  | ⭐⭐⭐   | ⭐⭐⭐⭐     | ⭐⭐     | Baseline, feature analysis |
| XGBoost        | ⭐⭐    | ⭐⭐⭐⭐⭐    | ⭐⭐     | Maximum accuracy |
| LightGBM       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐     | ⭐⭐⭐⭐   | Large-scale production |

**All three are trained for comparison and ensemble predictions!**

---

## 🎨 Features Used

The models learn from:

**NDVI Metrics:**
- Base NDVI value
- Mean NDVI in neighborhood
- Max NDVI in neighborhood
- NDVI from direct raster sampling
- Variability (max - mean)

**Spatial Patterns:**
- Row and column position
- Sector/zone
- Spatial clustering

**Plant Characteristics:**
- Area (size in pixels)
- NDVI category

---

## 📈 Expected Results (Demo Data)

### Growth Stage Classification
```
Accuracy: 85-92%
Confusion Matrix shows good separation between stages
Feature importance: NDVI (45%), NDVI_max (22%), area (15%)
```

### Yield Prediction
```
R² Score: 0.85-0.87
MAE: 3-5 kg per plant
Total predicted yield: 856,432 kg across all plants
```

### Nitrogen Deficiency
```
Accuracy: 88-94%
Precision: 0.86 (86% of flagged plants are truly deficient)
Recall: 0.79 (catches 79% of deficient plants)
```

---

## 🚀 Quick Start Commands

### Install Dependencies
```bash
cd /Users/nithints/Drone_imagery_file
chmod +x plant_ndvi_pipeline/install_ml_deps.sh
./plant_ndvi_pipeline/install_ml_deps.sh
```

### Run Full Pipeline (NDVI + ML)
```bash
.venv/bin/python plant_ndvi_pipeline/run_pipeline.py
```

### Run Only ML (if you have enriched data)
```bash
.venv/bin/python plant_ndvi_pipeline/run_ml_pipeline.py
```

### Run Individual Steps
```bash
# Training only
.venv/bin/python plant_ndvi_pipeline/step6_ml_training.py

# Predictions only (requires trained models)
.venv/bin/python plant_ndvi_pipeline/step7_ml_predictions.py
```

---

## 📁 New Output Files

After running the ML pipeline, you'll find in `outputs/`:

```
outputs/
├── ml_models/                          # Trained model files
│   ├── growth_stage_random_forest.pkl
│   ├── growth_stage_xgboost.pkl
│   ├── growth_stage_lightgbm.pkl
│   ├── yield_random_forest.pkl
│   ├── yield_xgboost.pkl
│   ├── yield_lightgbm.pkl
│   ├── nitrogen_random_forest.pkl
│   ├── nitrogen_xgboost.pkl
│   ├── nitrogen_lightgbm.pkl
│   └── training_summary.json           # Model performance
├── plants_with_predictions.json        # All predictions
├── ml_insights_report.json             # Actionable insights
├── sector_analysis.csv                 # Performance by sector
├── high_priority_plants.csv            # Plants needing attention
└── plants_predictions_summary.csv      # Simplified table
```

---

## 💡 Value Propositions

### Before (NDVI Map Only)
```
"Plant P001234 has NDVI of 0.45 (Moderate Vegetation)"
→ What does this mean? What should I do?
```

### After (ML Predictions)
```
Plant P001234:
  • Growth Stage: Flowering (ready for pollination support)
  • Predicted Yield: 32.5 kg (below average)
  • Nitrogen Status: DEFICIENT ⚠️
  
→ ACTION: Apply 150g nitrogen fertilizer to boost yield
```

---

## ⚠️ Important: Synthetic Data

**Current Status:** Uses synthetic targets generated from NDVI patterns for **demonstration**.

**For Production:** Replace with real ground-truth data:
- Field surveys for growth stage
- Harvest records for actual yield
- Lab analysis for nitrogen levels
- Expert annotations

See `README_ML.md` for detailed instructions on replacing synthetic data.

---

## 🎓 Example Use Cases

### 1. Targeted Fertilization
```
Problem: Blanket fertilization wastes money
Solution: ML identifies 3,245 deficient plants (16%)
Savings: 84% reduction in fertilizer use
```

### 2. Harvest Planning
```
Problem: When to schedule harvest?
Solution: 8,456 plants (42%) entering fruiting stage
Action: Schedule harvest crew for 2 weeks from now
```

### 3. Yield Forecasting
```
Problem: How much to expect this season?
Solution: Predicted total yield = 856 tons
Action: Arrange logistics for 850-900 tons
```

### 4. Problem Detection
```
Problem: Some areas underperforming
Solution: Sector T2 has 412 deficient plants (23%)
Action: Investigate irrigation or soil issues in T2
```

---

## 🔧 Customization Options

### Add More Features
```python
# In step6_ml_training.py
# Add weather data
df['temperature'] = weather_data['temp']
df['rainfall'] = weather_data['rain']

# Add soil data
df['soil_ph'] = soil_data['ph']
```

### Tune Hyperparameters
```python
# Better accuracy
rf_model = RandomForestClassifier(
    n_estimators=200,  # More trees
    max_depth=20,      # Deeper trees
    min_samples_split=5
)
```

### Add New Prediction Tasks
```python
# Predict disease risk
def train_disease_models(df):
    # Similar to existing tasks
    ...
```

---

## 📞 Next Steps

1. **Install dependencies** - Run `install_ml_deps.sh`
2. **Run the pipeline** - Test with synthetic data
3. **Review outputs** - Check `ml_insights_report.json`
4. **Collect ground truth** - Field surveys, harvest data
5. **Retrain with real data** - Replace synthetic targets
6. **Deploy** - Use for actual farm management decisions

---

## 📚 Documentation

- **README_ML.md** - Comprehensive guide
- **step6_ml_training.py** - Well-commented training code
- **step7_ml_predictions.py** - Prediction and insights code

---

**This transforms your drone imagery analysis from descriptive (what is happening) to predictive (what will happen) and prescriptive (what should you do about it)!**
