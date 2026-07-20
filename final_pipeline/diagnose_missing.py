"""Diagnose WHY some plants get no direct NDVI/NDRE measurement."""
import numpy as np, rasterio
from PIL import Image
from pathlib import Path
import config
from canopy_sampler import _composed_affine, _map_pixels
from colormaps import OSAVI_CMAP
import plant_loader

plants = plant_loader.load_plants(config.PLANTS_FILE)
crop = plants[plants["sector_label"] != config.NOISE_LABEL].reset_index(drop=True)

with rasterio.open(str(config.ORTHO_TIF)) as o:
    ggs = o.transform
An, rgbn, Hn, Wn = _composed_affine(config.NDVI_TIF, ggs)   # NDVI raster
Ao, rgbo, Ho, Wo = _composed_affine(config.OSAVI_TIF, ggs)  # OSAVI raster
bg = np.array((255, 255, 255), np.int16)
md = Path(config.MASKS_DIR)
thr = config.OSAVI_SOIL_THRESHOLD

bx = crop["bbox_x"].to_numpy(); by = crop["bbox_y"].to_numpy(); mf = crop["mask_file"].to_numpy()
cause = []
for i in range(len(crop)):
    mp = md / str(mf[i])
    if not mp.exists():
        cause.append("mask_missing"); continue
    m = np.asarray(Image.open(mp))
    if m.ndim == 3: m = m[..., 0]
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        cause.append("mask_empty"); continue
    ocol, orow = bx[i] + xs, by[i] + ys
    # NDVI raster coverage
    irn, icn, inbn = _map_pixels(An, ocol, orow, Hn, Wn)
    ndvi_cols = rgbn[:, irn[inbn], icn[inbn]].T if inbn.any() else np.empty((0, 3))
    ndvi_veg = (~np.all(ndvi_cols.astype(np.int16) == bg, axis=1)).sum() if len(ndvi_cols) else 0
    if ndvi_veg == 0:
        cause.append("outside_NDVI_coverage"); continue
    # OSAVI gate
    iro, ico, inbo = _map_pixels(Ao, ocol, orow, Ho, Wo)
    keep_frac_veg = 0
    if inbo.any():
        oc = rgbo[:, iro[inbo], ico[inbo]].T
        nb = ~np.all(oc.astype(np.int16) == bg, axis=1)
        if nb.any():
            ov, _, _ = OSAVI_CMAP.classify(oc[nb])
            keep_frac_veg = int((ov >= thr).sum())
    # emulate gate: out-of-osavi pixels are kept (ungated)
    n_oob_osavi = int((~inbo).sum())
    total_keep = keep_frac_veg + n_oob_osavi
    if total_keep == 0:
        cause.append("all_soil_by_OSAVI"); continue
    cause.append("measured")

import pandas as pd
s = pd.Series(cause)
print(f"\nNon-NOISE crop plants: {len(crop):,}  (threshold={thr})")
print("=== ROOT CAUSE OF MISSING DIRECT MEASUREMENT ===")
for k, v in s.value_counts().items():
    print(f"  {k:<24}{v:>7,}  ({v/len(crop)*100:4.1f}%)")
# where do the outside-coverage plants sit? sector breakdown
miss = crop[s.values != "measured"].copy()
miss["cause"] = s.values[s.values != "measured"]
print("\n=== MISSING plants by sector ===")
print(miss.groupby("sector_label").size().sort_values(ascending=False).head(15).to_string())
print("\n=== MISSING plants by cause x sector (top) ===")
print(pd.crosstab(miss["sector_label"], miss["cause"]).to_string())
