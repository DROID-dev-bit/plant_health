"""
Canopy Height Model from the LAZ point cloud (ground-classified).

The point cloud carries an ASPRS ground classification (class 2). That gives a
true bare-earth reference under the canopy — more accurate than approximating
ground from the DSM with a rolling minimum. We rasterize:

  canopy_z = max z of all points per cell
  ground_z = min z of ground (class 2) points, gap-filled, local 2 m minimum
  CHM      = canopy_z - ground_z

The result is cached as a GeoTIFF so the 20M-point pass runs only once.
"""

import os
import numpy as np
import rasterio
from rasterio.crs import CRS
from affine import Affine


def build_chm_from_laz(laz_path, res=0.10, ground_window_m=2.0, max_m=2.5,
                       epsg=32647, cache_path=None, force=False):
    """Return (chm_array, transform). Builds + caches from the LAZ if needed."""
    if cache_path and os.path.exists(cache_path) and not force \
            and os.path.getmtime(cache_path) >= os.path.getmtime(str(laz_path)):
        with rasterio.open(cache_path) as s:
            arr = s.read(1)
            if s.nodata is not None:               # convert nodata -> NaN so
                arr = np.where(arr == s.nodata,    # finite -9999 sentinels do
                               np.nan, arr)        # not leak into heights
            return arr, s.transform

    import laspy
    from scipy.ndimage import minimum_filter

    las = laspy.read(str(laz_path))
    x = np.asarray(las.x); y = np.asarray(las.y)
    z = np.asarray(las.z, dtype=np.float32); cls = np.asarray(las.classification)

    x0, y1 = x.min(), y.max()
    W = int(np.ceil((x.max() - x0) / res)) + 1
    H = int(np.ceil((y1 - y.min()) / res)) + 1
    col = ((x - x0) / res).astype(np.int64)
    row = ((y1 - y) / res).astype(np.int64)
    flat = row * W + col

    canopy = np.full(H * W, -9999.0, dtype=np.float32)
    np.maximum.at(canopy, flat, z)
    gm = cls == 2
    ground = np.full(H * W, 9999.0, dtype=np.float32)
    np.minimum.at(ground, flat[gm], z[gm])

    canopy = canopy.reshape(H, W); ground = ground.reshape(H, W)
    canopy[canopy == -9999] = np.nan
    ground[ground == 9999] = np.nan

    gfill = np.where(np.isnan(ground),
                     np.nanmax(ground[np.isfinite(ground)]), ground)
    ground_local = minimum_filter(gfill, size=max(3, int(round(ground_window_m / res))))
    chm = np.clip(canopy - ground_local, 0.0, max_m).astype(np.float32)
    chm[np.isnan(canopy)] = -9999.0

    tf = Affine(res, 0, x0, 0, -res, y1)
    if cache_path:
        with rasterio.open(cache_path, "w", driver="GTiff", height=H, width=W,
                           count=1, dtype="float32", crs=CRS.from_epsg(epsg),
                           transform=tf, nodata=-9999.0, compress="deflate") as dst:
            dst.write(chm, 1)
    chm[chm == -9999.0] = np.nan
    return chm, tf
