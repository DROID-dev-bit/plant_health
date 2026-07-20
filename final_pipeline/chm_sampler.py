"""
Canopy Height Model (CHM) sampling — the structural axis.

Two CHM sources (config.CHM_SOURCE):
  - "laz"  : built from the ground-classified point cloud (most accurate).
  - "dsm"  : DSM minus a local-ground estimate (rolling minimum). Fallback;
             DSM-minus-DTM is unusable here because ODM's DTM mirrors the
             canopy, so the difference is ~0.

Per plant we report the median canopy height (m); the pipeline turns that into
canopy volume (height x canopy area).
"""

import numpy as np
import rasterio
from PIL import Image


def _chm_from_dsm(dsm_tif, ground_window_m, max_m):
    from scipy.ndimage import minimum_filter
    with rasterio.open(str(dsm_tif)) as s:
        dsm = s.read(1).astype("float32")
        nd = s.nodata
        tf = s.transform
    nan_mask = (dsm == nd) if nd is not None else ~np.isfinite(dsm)
    gsd = abs(tf.a)
    finite = dsm[~nan_mask]
    fill = float(finite.max()) if finite.size else 0.0
    filled = np.where(nan_mask, fill, dsm)
    ground = minimum_filter(filled, size=max(3, int(round(ground_window_m / gsd))))
    chm = dsm - ground
    chm[nan_mask] = np.nan
    return np.clip(chm, 0.0, max_m), tf


def assign_canopy_height(plants_df, ortho_tif, masks_dir, *, source="laz",
                         dsm_tif=None, laz_tif=None, cache_path=None,
                         ground_window_m=2.0, max_m=2.5, noise_label="NOISE"):
    """Return per-plant ``plant_height_m`` and ``canopy_height_px`` columns."""
    from pathlib import Path
    masks_dir = Path(masks_dir)

    if source == "laz" and laz_tif is not None:
        from laz_chm import build_chm_from_laz
        chm, chm_tf = build_chm_from_laz(
            laz_tif, ground_window_m=ground_window_m, max_m=max_m,
            cache_path=cache_path)
        src_name = "LAZ point cloud"
    else:
        chm, chm_tf = _chm_from_dsm(dsm_tif, ground_window_m, max_m)
        src_name = "DSM local-ground"
    Hc, Wc = chm.shape

    with rasterio.open(str(ortho_tif)) as osrc:
        ggs = osrc.transform
    A = (~chm_tf) * ggs   # ggs-ortho (col,row) -> CHM (col,row)

    n = len(plants_df)
    height = np.full(n, np.nan)
    npx = np.zeros(n, dtype=int)
    bx = plants_df["bbox_x"].to_numpy()
    by = plants_df["bbox_y"].to_numpy()
    mf = plants_df["mask_file"].to_numpy()
    lab = plants_df["sector_label"].to_numpy()

    matched = 0
    for i in range(n):
        if lab[i] == noise_label:
            continue
        mpath = masks_dir / str(mf[i])
        if not mpath.exists():
            continue
        mask = np.asarray(Image.open(mpath))
        if mask.ndim == 3:
            mask = mask[..., 0]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        ocol = bx[i] + xs
        orow = by[i] + ys
        ic = np.round(A.a * ocol + A.b * orow + A.c).astype(int)
        ir = np.round(A.d * ocol + A.e * orow + A.f).astype(int)
        inb = (ir >= 0) & (ir < Hc) & (ic >= 0) & (ic < Wc)
        if not inb.any():
            continue
        vals = chm[ir[inb], ic[inb]]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        height[i] = round(float(np.median(vals)), 3)
        npx[i] = len(vals)
        matched += 1

    print(f"[chm_sampler] {matched:,}/{n:,} plants got a canopy height "
          f"({matched / n * 100:.1f}%) via {src_name}")
    return {"plant_height_m": height, "canopy_height_px": npx}
