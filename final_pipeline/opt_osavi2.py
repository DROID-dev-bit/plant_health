"""
Find the OBJECTIVE soil/leaf OSAVI split inside canopy masks (Otsu).

Maximising mean NDVI is circular (OSAVI~NDVI), so instead we pool the OSAVI
value of every canopy pixel and locate the natural valley between the soil mode
and the leaf mode. That valley is the principled soil-gate threshold.
"""
import numpy as np
import rasterio
from PIL import Image
from pathlib import Path

import config
from canopy_sampler import _composed_affine, _map_pixels
from colormaps import OSAVI_CMAP


def run():
    import plant_loader
    plants = plant_loader.load_plants(config.PLANTS_FILE)
    crop = plants[plants["sector_label"] != config.NOISE_LABEL]

    with rasterio.open(str(config.ORTHO_TIF)) as osrc:
        ggs = osrc.transform
    A, rgb, H, W = _composed_affine(config.OSAVI_TIF, ggs)
    bg = np.array((255, 255, 255), dtype=np.int16)
    masks_dir = Path(config.MASKS_DIR)

    # pooled histogram of canopy-pixel OSAVI (discrete bin midpoints)
    all_vals = []
    bx = crop["bbox_x"].to_numpy(); by = crop["bbox_y"].to_numpy()
    mf = crop["mask_file"].to_numpy()
    for i in range(len(crop)):
        mp = masks_dir / str(mf[i])
        if not mp.exists():
            continue
        m = np.asarray(Image.open(mp))
        if m.ndim == 3:
            m = m[..., 0]
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        ir, ic, inb = _map_pixels(A, bx[i] + xs, by[i] + ys, H, W)
        if not inb.any():
            continue
        cols = rgb[:, ir[inb], ic[inb]].T
        not_bg = ~np.all(cols.astype(np.int16) == bg, axis=1)
        if not not_bg.any():
            continue
        vals, _, _ = OSAVI_CMAP.classify(cols[not_bg])
        all_vals.append(vals)
    v = np.concatenate(all_vals)
    print(f"[opt2] pooled {len(v):,} canopy OSAVI pixels")

    mids = np.unique(v)
    hist = np.array([(v == m).sum() for m in mids], dtype=float)
    pct = hist / hist.sum() * 100
    print("\nOSAVI value distribution across canopy pixels:")
    for mm, h, p in zip(mids, hist, pct):
        bar = "#" * int(p / 2)
        print(f"  {mm:>6.2f} : {int(h):>9,}  {p:5.1f}%  {bar}")

    # Otsu threshold over the discrete histogram
    total = hist.sum()
    cumw = np.cumsum(hist)
    cummean = np.cumsum(hist * mids)
    gmean = cummean[-1] / total
    best_t, best_var = mids[0], -1
    for k in range(len(mids) - 1):
        w0 = cumw[k] / total
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = cummean[k] / cumw[k]
        m1 = (cummean[-1] - cummean[k]) / (total - cumw[k])
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t = var, mids[k + 1]
    print(f"\n[opt2] Otsu soil/leaf split = OSAVI >= {best_t:.2f}")
    print(f"       (soil mode below, vegetation mode above this value)")
    return best_t


if __name__ == "__main__":
    run()
