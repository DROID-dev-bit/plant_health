"""
Objective OSAVI-threshold validation via cross-index agreement.

Maximising NDVI is circular. Instead we check whether soil-gating improves the
correlation between NDVI and the *other* canopy indices (NDRE, GNDVI): soil adds
noise shared across all three, so removing it should tighten their agreement —
until the gate goes too far and starts deleting real leaf variation (corr falls
or coverage collapses). The peak, coverage-weighted, is the true optimum.
"""
import numpy as np
import pandas as pd

import config
from plant_loader import load_plants
from canopy_sampler import assign_canopy_indices
from colormaps import NDVI_CMAP, NDRE_CMAP, GNDVI_CMAP, OSAVI_CMAP

THRESHOLDS = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def run():
    plants = load_plants(config.PLANTS_FILE)
    crop = (plants["sector_label"] != config.NOISE_LABEL).values
    n_crop = int(crop.sum())
    specs = [(config.NDVI_TIF, NDVI_CMAP, "ndvi"),
             (config.NDRE_TIF, NDRE_CMAP, "ndre"),
             (config.GNDVI_TIF, GNDVI_CMAP, "gndvi")]

    rows = []
    for thr in THRESHOLDS:
        out = assign_canopy_indices(plants.copy(), config.ORTHO_TIF,
                                    config.MASKS_DIR, specs,
                                    noise_label=config.NOISE_LABEL,
                                    osavi_spec=(config.OSAVI_TIF, OSAVI_CMAP, thr))
        d = pd.DataFrame({k: out[k] for k in ["ndvi", "ndre", "gndvi"]})[crop]
        m = d.dropna()
        cov = len(m) / n_crop
        r_ng = m["ndvi"].corr(m["gndvi"])   # independent-ish validation
        r_nr = m["ndvi"].corr(m["ndre"])
        rows.append({"thr": thr, "coverage": round(cov * 100, 1),
                     "r_ndvi_gndvi": round(r_ng, 3),
                     "r_ndvi_ndre": round(r_nr, 3),
                     "mean_ndvi": round(float(m["ndvi"].mean()), 3),
                     "mean_ndre": round(float(m["ndre"].mean()), 3)})
    df = pd.DataFrame(rows)
    # quality = mean cross-index correlation, weighted by coverage
    df["quality"] = (0.5 * df["r_ndvi_gndvi"] + 0.5 * df["r_ndvi_ndre"]) \
        * (df["coverage"] / 100)
    best = df.loc[df["quality"].idxmax()]

    print("\n" + "=" * 78)
    print(f"{'thr':>5}{'cov%':>7}{'r(NDVI,GNDVI)':>15}{'r(NDVI,NDRE)':>14}"
          f"{'meanNDVI':>10}{'quality':>9}")
    print("-" * 78)
    for _, r in df.iterrows():
        mark = "  <== BEST" if r["thr"] == best["thr"] else ""
        cur = " (current)" if abs(r["thr"] - config.OSAVI_SOIL_THRESHOLD) < 1e-9 else ""
        print(f"{r['thr']:>5.2f}{r['coverage']:>7}{r['r_ndvi_gndvi']:>15}"
              f"{r['r_ndvi_ndre']:>14}{r['mean_ndvi']:>10}{r['quality']:>9.3f}{mark}{cur}")
    print("=" * 78)
    print(f"BEST OSAVI_SOIL_THRESHOLD = {best['thr']:.2f}  "
          f"(coverage {best['coverage']}%, r(NDVI,GNDVI)={best['r_ndvi_gndvi']}, "
          f"r(NDVI,NDRE)={best['r_ndvi_ndre']})")
    df.to_csv(config.PIPELINE_DIR / "opt_osavi3_results.csv", index=False)
    return df, best


if __name__ == "__main__":
    run()
