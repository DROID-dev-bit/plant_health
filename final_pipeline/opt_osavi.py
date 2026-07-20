"""
Optimise the OSAVI soil-gate threshold for NDVI/NDRE canopy extraction.

Runs the REAL canopy sampler across a range of OSAVI thresholds and measures the
purity vs coverage trade-off, then recommends the knee threshold.
"""
import numpy as np
import pandas as pd

import config
from plant_loader import load_plants
from canopy_sampler import assign_canopy_indices
from colormaps import NDVI_CMAP, NDRE_CMAP, OSAVI_CMAP

THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]


def run():
    plants = load_plants(config.PLANTS_FILE)
    crop = (plants["sector_label"] != config.NOISE_LABEL)
    n_crop = int(crop.sum())
    specs = [(config.NDVI_TIF, NDVI_CMAP, "ndvi"),
             (config.NDRE_TIF, NDRE_CMAP, "ndre")]

    rows = []
    for thr in THRESHOLDS:
        osavi_spec = (config.OSAVI_TIF, OSAVI_CMAP, thr)
        out = assign_canopy_indices(plants.copy(), config.ORTHO_TIF,
                                    config.MASKS_DIR, specs,
                                    noise_label=config.NOISE_LABEL,
                                    osavi_spec=osavi_spec)
        ndvi = pd.Series(out["ndvi"])[crop.values]
        ndre = pd.Series(out["ndre"])[crop.values]
        soil = pd.Series(out["canopy_soil_frac"])[crop.values]
        dist = pd.Series(out["ndvi_match_dist"])[crop.values]
        px = pd.Series(out["ndvi_canopy_px"])[crop.values]
        measured = int(ndvi.notna().sum())
        rows.append({
            "thr": thr,
            "measured": measured,
            "coverage_%": round(measured / n_crop * 100, 1),
            "dropped_all_soil": n_crop - measured,
            "mean_ndvi": round(float(ndvi.mean()), 3),
            "mean_ndre": round(float(ndre.mean()), 3),
            "mean_soil_removed_%": round(float(soil.mean()) * 100, 1),
            "mean_veg_px": round(float(px[px > 0].mean()), 0),
            "mean_match_dist": round(float(dist.mean()), 2),
        })
    df = pd.DataFrame(rows)

    # Knee: threshold where mean_ndvi has captured >=95% of its total rise
    # while coverage is still >= 85% of the field.
    lo, hi = df["mean_ndvi"].iloc[0], df["mean_ndvi"].max()
    rise = (df["mean_ndvi"] - lo) / max(hi - lo, 1e-9)
    ok = df[(rise >= 0.95) & (df["coverage_%"] >= 85)]
    knee = ok.iloc[0] if len(ok) else df.iloc[df["mean_ndvi"].idxmax()]

    print("\n" + "=" * 88)
    print(f"{'thr':>5}{'cov%':>7}{'dropped':>9}{'NDVI':>7}{'NDRE':>7}"
          f"{'soil%':>7}{'vegPx':>7}{'matchDist':>10}")
    print("-" * 88)
    for _, r in df.iterrows():
        star = "  <== knee" if r["thr"] == knee["thr"] else ""
        cur = "  (current)" if abs(r["thr"] - config.OSAVI_SOIL_THRESHOLD) < 1e-6 else ""
        print(f"{r['thr']:>5.2f}{r['coverage_%']:>7}{int(r['dropped_all_soil']):>9}"
              f"{r['mean_ndvi']:>7}{r['mean_ndre']:>7}{r['mean_soil_removed_%']:>7}"
              f"{int(r['mean_veg_px']):>7}{r['mean_match_dist']:>10}{star}{cur}")
    print("=" * 88)
    print(f"RECOMMENDED OSAVI_SOIL_THRESHOLD = {knee['thr']:.2f}  "
          f"(coverage {knee['coverage_%']}%, NDVI {knee['mean_ndvi']}, "
          f"NDRE {knee['mean_ndre']})")
    df.to_csv(config.PIPELINE_DIR / "opt_osavi_results.csv", index=False)
    return df, knee


if __name__ == "__main__":
    run()
