"""
Canopy-mask index sampling.

For every plant the ML team stored a binary canopy mask (outputs 2/04_masks),
sized to the plant's bounding box and positioned in the ggs-orthomosaic pixel
grid. This module averages each vegetation index over the plant's *actual*
canopy pixels — the most accurate per-plant value available.

Pipeline per plant:
  mask pixels (local) -> ggs ortho pixels -> compose affine -> index-raster
  pixels -> reverse-colormap to index values -> mean over canopy.

All index rasters are sampled in a single pass per plant so each mask PNG is
read only once.
"""

import numpy as np
import rasterio
from PIL import Image


def _composed_affine(index_tif, ggs_transform):
    """Affine mapping ggs-ortho (col,row) -> index-raster (col,row)."""
    with rasterio.open(str(index_tif)) as isrc:
        itf = isrc.transform
        rgb = isrc.read()[:3]
        H, W = rgb.shape[1], rgb.shape[2]
    A = (~itf) * ggs_transform  # applies ggs first, then index inverse
    return A, rgb, H, W


def _map_pixels(A, ocol, orow, H, W):
    """Map ggs-ortho (col,row) arrays to in-bounds index-raster (ir,ic)."""
    ic = np.round(A.a * ocol + A.b * orow + A.c).astype(int)
    ir = np.round(A.d * ocol + A.e * orow + A.f).astype(int)
    inb = (ir >= 0) & (ir < H) & (ic >= 0) & (ic < W)
    return ir, ic, inb


def _composed_affine_raw(index_tif, ggs_transform):
    """Like _composed_affine but for a single-band (float) index raster.

    Returns the composed ggs->index affine, the float band, its nodata value
    and its (H, W). Used for raw continuous rasters (e.g. TDVI) that carry real
    index values instead of a colorized RGBA render.
    """
    with rasterio.open(str(index_tif)) as isrc:
        itf = isrc.transform
        band = isrc.read(1).astype(np.float32)
        nodata = isrc.nodata
        H, W = band.shape
    A = (~itf) * ggs_transform
    return A, band, nodata, H, W


def assign_canopy_raw_indices(plants_df, ortho_tif, masks_dir, raw_specs,
                              noise_label="NOISE"):
    """
    Average each RAW single-band float index over every plant's canopy mask.

    Unlike ``assign_canopy_indices`` (which reverse-maps colorized RGBA rasters
    through a colormap), this reads the real continuous index values directly
    from band 1 and skips nodata pixels — no quantization, no colormap error.

    Parameters
    ----------
    raw_specs : list of (tif_path, prefix). Each raster is sampled over the same
                canopy masks as the colorized indices.

    Returns
    -------
    dict of new columns: for each prefix -> value and ``{prefix}_canopy_px``.
    """
    from pathlib import Path
    masks_dir = Path(masks_dir)

    with rasterio.open(str(ortho_tif)) as osrc:
        ggs = osrc.transform

    rasters = []
    for tif, prefix in raw_specs:
        A, band, nodata, H, W = _composed_affine_raw(tif, ggs)
        rasters.append((A, band, nodata, H, W, prefix))

    n = len(plants_df)
    out = {}
    for _, _, _, _, _, prefix in rasters:
        out[prefix] = np.full(n, np.nan)
        out[f"{prefix}_canopy_px"] = np.zeros(n, dtype=int)

    req = ["bbox_x", "bbox_y", "mask_file", "sector_label"]
    missing = [c for c in req if c not in plants_df.columns]
    if missing:
        raise KeyError(f"raw canopy sampling needs columns {missing}")

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

        any_hit = False
        for A, band, nodata, H, W, prefix in rasters:
            ir, ic, inb = _map_pixels(A, ocol, orow, H, W)
            if not inb.any():
                continue
            vals = band[ir[inb], ic[inb]]
            good = np.isfinite(vals)
            if nodata is not None:
                good &= vals != nodata
            kept = vals[good]
            if len(kept) == 0:
                continue
            out[prefix][i] = round(float(kept.mean()), 4)
            out[f"{prefix}_canopy_px"][i] = len(kept)
            any_hit = True
        if any_hit:
            matched += 1

    print(f"[canopy_sampler] raw index {[p for *_, p in rasters]}: "
          f"{matched:,}/{n:,} plants sampled ({matched / n * 100:.1f}%)")
    return out


def assign_canopy_indices(plants_df, ortho_tif, masks_dir, index_specs,
                          background_rgb=(255, 255, 255), noise_label="NOISE",
                          osavi_spec=None, fallback_min_osavi=0.05):
    """
    Average each index over every plant's canopy mask.

    Parameters
    ----------
    plants_df   : DataFrame with bbox_x, bbox_y, mask_file, sector_label.
    ortho_tif   : the orthomosaic the masks were drawn on (for its transform).
    masks_dir   : directory containing the per-plant mask PNGs.
    index_specs : list of (tif_path, colormap, prefix).
    osavi_spec  : optional (tif_path, colormap, threshold). When given, canopy
                  pixels whose OSAVI value < threshold are treated as bare soil
                  and excluded from every index mean (soil-noise removal).

    Returns
    -------
    dict of new columns: for each prefix -> value / category / match_dist /
    canopy_px, plus (with osavi_spec) ``osavi`` and ``canopy_soil_frac``.
    """
    from pathlib import Path
    masks_dir = Path(masks_dir)

    with rasterio.open(str(ortho_tif)) as osrc:
        ggs = osrc.transform

    rasters = []
    for tif, cmap, prefix in index_specs:
        A, rgb, H, W = _composed_affine(tif, ggs)
        rasters.append((A, rgb, H, W, cmap, prefix))

    osavi = None
    if osavi_spec is not None:
        otif, ocmap, othr = osavi_spec
        A, rgb, H, W = _composed_affine(otif, ggs)
        osavi = (A, rgb, H, W, ocmap, othr)

    n = len(plants_df)
    out = {}
    for _, _, _, _, _, prefix in rasters:
        out[prefix] = np.full(n, np.nan)
        out[f"{prefix}_category"] = np.array([None] * n, dtype=object)
        out[f"{prefix}_match_dist"] = np.full(n, np.nan)
        out[f"{prefix}_canopy_px"] = np.zeros(n, dtype=int)
    if osavi is not None:
        out["osavi"] = np.full(n, np.nan)
        out["canopy_soil_frac"] = np.full(n, np.nan)
        out["canopy_fallback"] = np.zeros(n, dtype=bool)

    bg = np.array(background_rgb, dtype=np.int16)
    req = ["bbox_x", "bbox_y", "mask_file", "sector_label"]
    missing = [c for c in req if c not in plants_df.columns]
    if missing:
        raise KeyError(f"canopy sampling needs columns {missing}")

    bx = plants_df["bbox_x"].to_numpy()
    by = plants_df["bbox_y"].to_numpy()
    mf = plants_df["mask_file"].to_numpy()
    lab = plants_df["sector_label"].to_numpy()

    matched = 0
    soil_dropped = 0
    fallback_used = 0
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
        ocol = bx[i] + xs            # ggs ortho pixel coords of canopy
        orow = by[i] + ys

        # --- OSAVI soil gate: keep only vegetation canopy pixels ---
        if osavi is not None:
            A, rgb, H, W, ocmap, othr = osavi
            ir, ic, inb = _map_pixels(A, ocol, orow, H, W)
            keep = np.ones(len(ocol), dtype=bool)   # out-of-OSAVI = ungated
            if inb.any():
                cols_o = rgb[:, ir[inb], ic[inb]].T
                not_bg = ~np.all(cols_o.astype(np.int16) == bg, axis=1)
                ovals, _, _ = ocmap.classify(cols_o)
                veg = not_bg & (ovals >= othr)
                # Fallback: if the gate removed EVERY pixel but the canopy still
                # has some greenish (non-soil) pixels, keep its greenest ones so
                # a struggling plant is measured from itself, not ML-imputed.
                if not veg.any() and not_bg.any() and ovals[not_bg].max() >= fallback_min_osavi:
                    top = ovals[not_bg].max()
                    veg = not_bg & (ovals >= top - 1e-6)
                    out["canopy_fallback"][i] = True
                    fallback_used += 1
                keep_inb = keep[inb]
                keep_inb[:] = veg
                keep[inb] = keep_inb
                if veg.any():
                    out["osavi"][i] = round(float(ovals[veg].mean()), 3)
            out["canopy_soil_frac"][i] = round(1.0 - keep.mean(), 3)
            if not keep.any():
                continue
            soil_dropped += int((~keep).sum())
            ocol, orow = ocol[keep], orow[keep]

        any_hit = False
        for A, rgb, H, W, cmap, prefix in rasters:
            ir, ic, inb = _map_pixels(A, ocol, orow, H, W)
            if not inb.any():
                continue
            cols_rgb = rgb[:, ir[inb], ic[inb]].T
            not_bg = ~np.all(cols_rgb.astype(np.int16) == bg, axis=1)
            kept = cols_rgb[not_bg]
            if len(kept) == 0:
                continue
            vals, _, dists = cmap.classify(kept)
            mean_val = float(vals.mean())
            out[prefix][i] = round(mean_val, 3)
            out[f"{prefix}_category"][i] = cmap.category_for_value(mean_val)
            out[f"{prefix}_match_dist"][i] = round(float(dists.mean()), 2)
            out[f"{prefix}_canopy_px"][i] = len(kept)
            any_hit = True
        if any_hit:
            matched += 1

    msg = (f"[canopy_sampler] {matched:,}/{n:,} plants sampled "
           f"({matched / n * 100:.1f}%)")
    if osavi is not None:
        msg += (f"; {soil_dropped:,} soil pixels removed via OSAVI; "
                f"{fallback_used:,} sparse plants recovered via greenest-pixel fallback")
    print(msg)
    return out
