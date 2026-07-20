# Water Flow Physics Engine

Interactive rainfall / surface-water-flow simulation over the farm's real
drone-derived terrain (DTM + DSM from `maps/`, survey 2026-06-28).

## Files
| File | Purpose |
|---|---|
| `prepare_terrain.py` | Reads the DTM/DSM GeoTIFFs, cleans + downsamples to a 1 m grid, computes D8 flow direction/accumulation and canopy height, writes `terrain_data.js`. |
| `terrain_data.js` | Pre-computed terrain package the web app loads (no server needed). |
| `water_flow_simulation.html` | The simulation — open directly in any browser. |

## Run
```bash
# regenerate terrain (only needed if the DTM/DSM change)
../.venv/bin/python prepare_terrain.py

# then just open the simulation
open water_flow_simulation.html
```

## Physics
- **Rain particles**: each drop accelerates down the steepest slope of the
  water surface (`a = -g·∇(z + d) - friction·v`), integrated per frame.
- **Ponding**: stalled drops deposit volume into a shallow-water height
  field; ponds relax toward flat water surfaces and slowly infiltrate.
- **Drainage channels**: D8 flow accumulation (priority-flood depression
  filled) shown as blue lines — the terrain's natural streamlines.

## Controls
- Rain on/off, intensity, flow speed sliders
- **Click / drag on the map to pour water anywhere**
- Layer toggles: channels, ponding, particles, canopy, contours
- Hover for per-cell elevation, water depth, upstream catchment area
