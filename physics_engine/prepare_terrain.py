"""
Terrain preprocessor for the water-flow physics engine.

Reads the drone-derived DTM (ground elevation) and DSM (surface incl. canopy)
from maps/, cleans and downsamples them to a simulation grid, computes
hydrology layers (slope, D8 flow direction, flow accumulation), and exports
everything as `terrain_data.js` — a plain JS file the simulation HTML loads
directly (works from file://, no server needed).

Run:
    ../.venv/bin/python prepare_terrain.py
"""

import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
DTM_TIF = ROOT / "maps" / "Task-of-2026-06-28T103132228Z-dtm.tif"
DSM_TIF = ROOT / "maps" / "Task-of-2026-06-28T103132228Z-dsm.tif"
OUT_JS = Path(__file__).resolve().parent / "terrain_data.js"

GRID = 240          # simulation grid size (max dimension)
NODATA = -9999.0

# D8 neighbour offsets (dr, dc)
D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def load_band(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float64)
        arr[arr == ds.nodata] = np.nan
        return arr, ds.res[0], ds.bounds, str(ds.crs)


def clean(arr):
    """Clip elevation outliers (e.g. DTM min of 11 m vs field ~46-80 m)."""
    v = arr[np.isfinite(arr)]
    lo, hi = np.percentile(v, [0.5, 99.5])
    return np.clip(arr, lo, hi)


def downsample(arr, grid):
    """Block-mean downsample to <= grid cells on the longest side."""
    H, W = arr.shape
    f = max(1, int(np.ceil(max(H, W) / grid)))
    Hc, Wc = H // f * f, W // f * f
    a = arr[:Hc, :Wc].reshape(Hc // f, f, Wc // f, f)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(np.nanmean(a, axis=3), axis=1)
    return out, f


def fill_nodata(arr, iters=200):
    """Iteratively fill NaN holes with the mean of valid neighbours."""
    a = arr.copy()
    for _ in range(iters):
        nan = np.isnan(a)
        if not nan.any():
            break
        p = np.pad(a, 1, constant_values=np.nan)
        stack = np.stack([p[1 + dr:1 + dr + a.shape[0], 1 + dc:1 + dc + a.shape[1]]
                          for dr, dc in D8])
        with np.errstate(invalid="ignore"):
            nbr = np.nanmean(stack, axis=0)
        a[nan] = nbr[nan]
    a[np.isnan(a)] = np.nanmin(a)
    return a


def fill_depressions(z, cell, epsilon=1e-4):
    """Priority-flood depression filling so water always has an outlet."""
    import heapq
    H, W = z.shape
    filled = z.copy()
    seen = np.zeros((H, W), bool)
    heap = []
    for r in range(H):
        for c in range(W):
            if r in (0, H - 1) or c in (0, W - 1):
                heapq.heappush(heap, (filled[r, c], r, c))
                seen[r, c] = True
    while heap:
        e, r, c = heapq.heappop(heap)
        for dr, dc in D8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc]:
                seen[nr, nc] = True
                filled[nr, nc] = max(filled[nr, nc], e + epsilon)
                heapq.heappush(heap, (filled[nr, nc], nr, nc))
    return filled


def flow_accumulation(z, cell):
    """D8 flow direction + accumulation (number of upstream cells)."""
    H, W = z.shape
    dist = np.array([np.hypot(dr, dc) for dr, dc in D8]) * cell
    fdir = np.full((H, W), -1, np.int8)
    for r in range(H):
        for c in range(W):
            best, bi = 0.0, -1
            for i, (dr, dc) in enumerate(D8):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    s = (z[r, c] - z[nr, nc]) / dist[i]
                    if s > best:
                        best, bi = s, i
            fdir[r, c] = bi
    order = np.argsort(z, axis=None)[::-1]          # high -> low
    acc = np.ones((H, W))
    for idx in order:
        r, c = divmod(idx, W)
        i = fdir[r, c]
        if i >= 0:
            dr, dc = D8[i]
            acc[r + dr, c + dc] += acc[r, c]
    return fdir, acc


def main():
    print(f"[terrain] reading DTM {DTM_TIF.name}")
    dtm, res, bounds, crs = load_band(DTM_TIF)
    dsm, _, _, _ = load_band(DSM_TIF)

    dtm = clean(dtm)
    dsm = clean(dsm)

    dtm_g, f = downsample(dtm, GRID)
    dsm_g, _ = downsample(dsm, GRID)
    cell = res * f
    print(f"[terrain] grid {dtm_g.shape[1]}x{dtm_g.shape[0]}  cell={cell:.2f} m")

    valid = np.isfinite(dtm_g)                       # true survey footprint
    dtm_g = fill_nodata(dtm_g)
    dsm_g = fill_nodata(dsm_g)
    canopy = np.clip(dsm_g - dtm_g, 0, 3)            # vegetation height, m

    print("[terrain] filling depressions (priority-flood)…")
    zfill = fill_depressions(dtm_g, cell)

    print("[terrain] computing D8 flow accumulation…")
    fdir, acc = flow_accumulation(zfill, cell)
    print(f"[terrain] max upstream area: {acc.max() * cell * cell:,.0f} m^2")

    H, W = dtm_g.shape
    rnd = lambda a, n: np.round(a, n).tolist()
    data = {
        "width": W, "height": H, "cell_m": round(cell, 3),
        "crs": crs, "bounds": list(bounds),
        "elev_min": round(float(np.nanmin(dtm_g)), 2),
        "elev_max": round(float(np.nanmax(dtm_g)), 2),
        "elevation": rnd(dtm_g, 2),          # ground (water flows on this)
        "elevation_filled": rnd(zfill, 3),   # depression-filled (routing)
        "canopy": rnd(canopy, 2),            # DSM - DTM vegetation height
        "flow_acc": rnd(np.log1p(acc), 3),   # log upstream cell count
        "flow_dir": fdir.tolist(),           # D8 index, -1 = pit/edge
        "valid": valid.astype(np.uint8).tolist(),
    }
    OUT_JS.write_text("const TERRAIN = " + json.dumps(data) + ";\n")
    print(f"[terrain] wrote {OUT_JS}  ({OUT_JS.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
