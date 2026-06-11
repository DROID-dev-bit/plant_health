#!/usr/bin/env python3
"""
Step 8 — Export ML Data for Web Dashboard
==========================================
Exports ML predictions in a web-friendly format that can be easily
integrated into the existing dashboard or any custom web visualization.

Outputs:
- ml_data_for_web.js (JavaScript data file)
- dashboard_instructions.md (Integration guide)
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

ML_DATA_JS = BASE_DIR / "ml_data_for_web.js"
INSTRUCTIONS_MD = BASE_DIR / "dashboard_integration_guide.md"


def export_for_web():
    """Export ML data in web-friendly format."""
    
    print("=" * 60)
    print("Exporting ML Data for Web Dashboard")
    print("=" * 60)
    
    # Load predictions
    if not PREDICTIONS_JSON.exists():
        print(f"\n[ERROR] Predictions not found: {PREDICTIONS_JSON}")
        print("[INFO] Run ML pipeline first:")
        print("  .venv/bin/python plant_ndvi_pipeline/run_ml_pipeline.py")
        return
    
    print(f"\n[INFO] Loading predictions...")
    with open(PREDICTIONS_JSON, 'r') as f:
        plants = json.load(f)
    
    print(f"[INFO] Loading insights...")
    insights = {}
    if INSIGHTS_JSON.exists():
        with open(INSIGHTS_JSON, 'r') as f:
            insights = json.load(f)
    
    # Simplify plant data for web (only essential fields)
    print(f"\n[INFO] Preparing {len(plants):,} plants for web...")
    web_plants = []
    for p in plants:
        web_plant = {
            'id': p.get('plant_id'),
            'lat': p.get('lat'),
            'lon': p.get('lon'),
            'ndvi': p.get('ndvi'),
            'sector': p.get('sector_label'),
            'row': p.get('row_index'),
            'col': p.get('col_index'),
            # ML Predictions
            'growth_stage': p.get('predicted_growth_stage_name'),
            'growth_confidence': p.get('growth_stage_confidence'),
            'yield_kg': p.get('predicted_yield_kg'),
            'nitrogen': p.get('predicted_nitrogen_status'),
            'nitrogen_confidence': p.get('nitrogen_deficiency_confidence'),
            'is_deficient': p.get('predicted_nitrogen_deficiency') == 1,
            'is_low_yield': p.get('predicted_yield_kg', 100) < 20 if p.get('predicted_yield_kg') else False,
        }
        web_plants.append(web_plant)
    
    # Generate JavaScript file
    print(f"[INFO] Generating JavaScript data file...")
    
    js_content = f'''// ML Predictions Data for Farm Dashboard
// Generated: {datetime.now().isoformat()}
// Total Plants: {len(web_plants):,}

const ML_PREDICTIONS = {json.dumps(web_plants, indent=2)};

const ML_INSIGHTS = {json.dumps(insights, indent=2)};

// Quick access functions
const ML_DATA = {{
  getAllPlants: () => ML_PREDICTIONS,
  
  getPlantById: (id) => ML_PREDICTIONS.find(p => p.id === id),
  
  getByGrowthStage: (stage) => ML_PREDICTIONS.filter(p => p.growth_stage === stage),
  
  getDeficientPlants: () => ML_PREDICTIONS.filter(p => p.is_deficient),
  
  getLowYieldPlants: () => ML_PREDICTIONS.filter(p => p.is_low_yield),
  
  getHighPriorityPlants: () => ML_PREDICTIONS.filter(p => p.is_deficient && p.yield_kg < 20),
  
  getBySector: (sector) => ML_PREDICTIONS.filter(p => p.sector === sector),
  
  getInsights: () => ML_INSIGHTS,
  
  getRecommendations: () => ML_INSIGHTS.recommendations || [],
  
  // Color helpers
  getGrowthStageColor: (stage) => {{
    const colors = {{
      'Seedling': '#8b5cf6',
      'Vegetative': '#3b82f6',
      'Flowering': '#f59e0b',
      'Fruiting': '#10b981',
      'Mature': '#22c55e'
    }};
    return colors[stage] || '#6b7280';
  }},
  
  getYieldColor: (yield_kg) => {{
    if (yield_kg >= 60) return '#22c55e';
    if (yield_kg >= 40) return '#10b981';
    if (yield_kg >= 20) return '#f59e0b';
    return '#ef4444';
  }},
  
  getNitrogenColor: (status) => {{
    return status === 'Deficient' ? '#ef4444' : '#22c55e';
  }}
}};

// Export for use in dashboard
if (typeof module !== 'undefined' && module.exports) {{
  module.exports = ML_DATA;
}}
'''
    
    with open(ML_DATA_JS, 'w') as f:
        f.write(js_content)
    
    print(f"[✓] Saved: {ML_DATA_JS}")
    print(f"[INFO] File size: {ML_DATA_JS.stat().st_size / 1e6:.2f} MB")
    
    # Generate integration instructions
    print(f"\n[INFO] Generating integration guide...")
    generate_instructions(insights)
    
    print(f"\n[✓] Complete!")
    print(f"\n📁 Generated Files:")
    print(f"  • {ML_DATA_JS.name} - ML data for web")
    print(f"  • {INSTRUCTIONS_MD.name} - Integration guide")
    print(f"\n📖 Next: Read {INSTRUCTIONS_MD.name} to integrate with your dashboard")


def generate_instructions(insights):
    """Generate integration instructions."""
    
    yield_analysis = insights.get('yield_analysis', {})
    nitrogen_analysis = insights.get('nitrogen_analysis', {})
    recommendations = insights.get('recommendations', [])
    
    instructions = f'''# Dashboard Integration Guide

## 🎯 Quick Start

Your ML predictions have been exported to **`ml_data_for_web.js`** - a JavaScript file you can directly include in any web page.

---

## 📦 What's Included

### Data Structure

```javascript
ML_PREDICTIONS = [
  {{
    id: "P001234",
    lat: 2.9365,
    lon: 101.7445,
    ndvi: 0.645,
    sector: "T1",
    // ML Predictions
    growth_stage: "Flowering",
    growth_confidence: 0.89,
    yield_kg: 45.2,
    nitrogen: "Healthy",
    nitrogen_confidence: 0.92,
    is_deficient: false,
    is_low_yield: false
  }},
  ...
]
```

---

## 🚀 Integration Methods

### Method 1: Simple Include (Recommended)

```html
<!DOCTYPE html>
<html>
<head>
  <title>Farm Dashboard</title>
</head>
<body>
  <div id="map"></div>
  
  <!-- Include ML data -->
  <script src="ml_data_for_web.js"></script>
  
  <script>
    // Access the data
    console.log("Total plants:", ML_PREDICTIONS.length);
    
    // Get deficient plants
    const deficient = ML_DATA.getDeficientPlants();
    console.log("Deficient plants:", deficient.length);
    
    // Get recommendations
    const recommendations = ML_DATA.getRecommendations();
    console.log("Actions needed:", recommendations);
  </script>
</body>
</html>
```

### Method 2: Dynamic Loading

```javascript
// Load ML data dynamically
fetch('ml_data_for_web.js')
  .then(response => response.text())
  .then(script => {{
    eval(script);  // Execute the JavaScript
    // Now ML_DATA is available
    displayPredictions();
  }});
```

---

## 🗺️ Map Visualization Examples

### Example 1: Color by Growth Stage

```javascript
ML_PREDICTIONS.forEach(plant => {{
  const color = ML_DATA.getGrowthStageColor(plant.growth_stage);
  
  L.circleMarker([plant.lat, plant.lon], {{
    radius: 6,
    fillColor: color,
    fillOpacity: 0.8,
    color: '#fff',
    weight: 1
  }})
  .bindPopup(`
    <b>${{plant.id}}</b><br>
    Growth Stage: ${{plant.growth_stage}}<br>
    Yield: ${{plant.yield_kg}} kg<br>
    Nitrogen: ${{plant.nitrogen}}
  `)
  .addTo(map);
}});
```

### Example 2: Yield Heatmap

```javascript
ML_PREDICTIONS.forEach(plant => {{
  const color = ML_DATA.getYieldColor(plant.yield_kg);
  const radius = Math.max(4, plant.yield_kg / 5);  // Size by yield
  
  L.circleMarker([plant.lat, plant.lon], {{
    radius: radius,
    fillColor: color,
    fillOpacity: 0.7
  }})
  .bindTooltip(`Yield: ${{plant.yield_kg}} kg`)
  .addTo(map);
}});
```

### Example 3: Highlight Priority Plants

```javascript
const highPriority = ML_DATA.getHighPriorityPlants();

highPriority.forEach(plant => {{
  L.circleMarker([plant.lat, plant.lon], {{
    radius: 8,
    fillColor: '#ef4444',
    fillOpacity: 0.9,
    color: '#fff',
    weight: 2,
    className: 'pulse-marker'  // Add CSS animation
  }})
  .bindPopup(`
    <b>⚠️ HIGH PRIORITY</b><br>
    ${{plant.id}}<br>
    Nitrogen: Deficient<br>
    Yield: ${{plant.yield_kg}} kg<br>
    <button onclick="addToActionList('${{plant.id}}')">Add to Action List</button>
  `)
  .addTo(map);
}});
```

---

## 📊 Stats Panel Examples

### Display Key Metrics

```javascript
const insights = ML_DATA.getInsights();

// Total Yield
document.getElementById('total-yield').textContent = 
  insights.yield_analysis.total_predicted_yield_kg.toLocaleString() + ' kg';

// Deficient Plants
document.getElementById('deficient-count').textContent = 
  insights.nitrogen_analysis.deficient_plants.toLocaleString();

// Recommendations
const recsHTML = insights.recommendations.map(rec => `
  <div class="recommendation ${{rec.priority.toLowerCase()}}">
    <span class="priority">${{rec.priority}}</span>
    <strong>${{rec.action}}</strong>
    <p>${{rec.reason}}</p>
  </div>
`).join('');
document.getElementById('recommendations').innerHTML = recsHTML;
```

### Growth Stage Distribution Chart

```javascript
const growthDist = insights.growth_stage_distribution;

new Chart(document.getElementById('growth-chart'), {{
  type: 'doughnut',
  data: {{
    labels: Object.keys(growthDist),
    datasets: [{{
      data: Object.values(growthDist),
      backgroundColor: ['#8b5cf6', '#3b82f6', '#f59e0b', '#10b981', '#22c55e']
    }}]
  }}
}});
```

---

## 🎨 Color Schemes

### Growth Stage Colors
- **Seedling**: `#8b5cf6` (Purple)
- **Vegetative**: `#3b82f6` (Blue)
- **Flowering**: `#f59e0b` (Orange)
- **Fruiting**: `#10b981` (Green)
- **Mature**: `#22c55e` (Bright Green)

### Yield Colors
- **High** (≥60kg): `#22c55e`
- **Good** (40-60kg): `#10b981`
- **Medium** (20-40kg): `#f59e0b`
- **Low** (<20kg): `#ef4444`

### Nitrogen Status
- **Healthy**: `#22c55e`
- **Deficient**: `#ef4444`

---

## 🔍 Filtering Examples

```javascript
// Filter by sector
const sectorT1 = ML_DATA.getBySector('T1');

// Filter by growth stage
const flowering = ML_DATA.getByGrowthStage('Flowering');

// Filter deficient plants
const needsFertilizer = ML_DATA.getDeficientPlants();

// Filter low-yield plants
const underperforming = ML_DATA.getLowYieldPlants();

// Combined filters
const criticalPlants = ML_PREDICTIONS.filter(p => 
  p.is_deficient && p.yield_kg < 20 && p.sector === 'T2'
);
```

---

## 💡 Key Insights from Current Data

{generate_insights_summary(insights)}

---

## 📱 Mobile-Responsive Tips

```css
/* Make markers touch-friendly on mobile */
@media (max-width: 768px) {{
  .leaflet-marker-icon {{
    width: 40px !important;
    height: 40px !important;
  }}
}}
```

---

## 🔄 Updating Data

When you retrain models or get new data:

```bash
# 1. Run ML pipeline
.venv/bin/python plant_ndvi_pipeline/run_ml_pipeline.py

# 2. Export new web data
.venv/bin/python plant_ndvi_pipeline/step8_export_ml_for_web.py

# 3. Refresh browser (ml_data_for_web.js is updated)
```

---

## 📚 Full API Reference

```javascript
// Get all plants
ML_DATA.getAllPlants()

// Get single plant
ML_DATA.getPlantById("P001234")

// Filter by growth stage
ML_DATA.getByGrowthStage("Flowering")

// Get deficient plants
ML_DATA.getDeficientPlants()

// Get low-yield plants
ML_DATA.getLowYieldPlants()

// Get high-priority plants (deficient + low yield)
ML_DATA.getHighPriorityPlants()

// Filter by sector
ML_DATA.getBySector("T1")

// Get insights
ML_DATA.getInsights()

// Get recommendations
ML_DATA.getRecommendations()

// Color helpers
ML_DATA.getGrowthStageColor("Flowering")  // Returns color hex
ML_DATA.getYieldColor(45.2)               // Returns color hex
ML_DATA.getNitrogenColor("Deficient")     // Returns color hex
```

---

## 🎯 Example: Complete Dashboard Integration

See `plantation_dashboard.html` for the base dashboard structure.

To add ML predictions:

1. Include `ml_data_for_web.js` before your map script
2. Add layer toggle buttons for different visualizations
3. Update marker colors based on ML predictions
4. Add ML stats to the sidebar
5. Show recommendations panel

---

## 🆘 Troubleshooting

### "ML_DATA is not defined"
- Make sure `ml_data_for_web.js` is loaded before your script
- Check browser console for errors
- Verify file path is correct

### "Cannot read property of undefined"
- Check if predictions exist for that plant
- Use optional chaining: `plant?.yield_kg`

### Large file size
- The JS file includes all plant data
- Consider loading on-demand for very large farms
- Or filter to visible plants only

---

**Happy Dashboard Building! 🎨**

For more examples, see the ML_PIPELINE_OVERVIEW.md documentation.
'''
    
    with open(INSTRUCTIONS_MD, 'w') as f:
        f.write(instructions)
    
    print(f"[✓] Saved: {INSTRUCTIONS_MD}")


def generate_insights_summary(insights):
    """Generate markdown summary of insights."""
    
    lines = []
    
    yield_analysis = insights.get('yield_analysis', {})
    if yield_analysis:
        total = yield_analysis.get('total_predicted_yield_kg', 0)
        avg = yield_analysis.get('average_yield_per_plant_kg', 0)
        lines.append(f"### Yield Forecast\n- **Total predicted yield**: {total:,.0f} kg")
        lines.append(f"- **Average per plant**: {avg:.1f} kg\n")
    
    nitrogen_analysis = insights.get('nitrogen_analysis', {})
    if nitrogen_analysis:
        deficient = nitrogen_analysis.get('deficient_plants', 0)
        pct = nitrogen_analysis.get('deficient_percentage', 0)
        lines.append(f"### Nitrogen Status\n- **Deficient plants**: {deficient:,} ({pct:.1f}%)")
        lines.append(f"- **Action needed**: Apply fertilizer to deficient areas\n")
    
    recommendations = insights.get('recommendations', [])
    if recommendations:
        lines.append(f"### Priority Actions\n")
        for i, rec in enumerate(recommendations[:3], 1):
            priority = rec.get('priority', 'UNKNOWN')
            action = rec.get('action', '')
            reason = rec.get('reason', '')
            lines.append(f"{i}. **[{priority}]** {action}")
            lines.append(f"   - {reason}")
    
    return "\n".join(lines) if lines else "*(Run ML pipeline to generate insights)*"


if __name__ == "__main__":
    export_for_web()
