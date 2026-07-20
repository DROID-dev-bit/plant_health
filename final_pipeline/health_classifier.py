"""
Agronomic plant-health model from NDVI + NDRE.

Two indices, two biological meanings:
  - NDVI  -> plant BIOMASS  (leaf area / green canopy density)
  - NDRE  -> plant NITROGEN (leaf chlorophyll, i.e. N uptake)

A single health number hides *why* a plant struggles, so this model keeps the
two axes and reports both, then diagnoses the limiting factor.

Pipeline:
  1. Spatial smoothing (KD-Tree, 8 neighbours) of each index.
  2. Percentile scores: biomass_score & nitrogen_score (0-100), the plant's
     rank within the field — directly actionable ("bottom 10% for nitrogen").
  3. Unsupervised GaussianMixture over (NDVI, NDRE) -> data-driven plant
     archetypes (health_cluster) with a membership confidence.
  4. IsolationForest -> spectral anomaly flag.
  5. health_score (0-100) + 4-tier health_status (for the map) and an
     interpretable health_diagnosis + limiting_factor.

NOISE / out-of-coverage plants get blank values.
"""

import numpy as np
import pandas as pd

STATUS_COLORS = {
    "Healthy":         "#2E7D32",
    "Moderate":        "#FBC02D",
    "Stressed":        "#FF5722",
    "Critical":        "#D32F2F",
    "Out of Boundary": "#1976D2",
    "NOISE":           "#757575",
}

# Tercile cut points (percentile scores) for low / medium / high on each axis.
LOW_CUT, HIGH_CUT = 33.0, 66.0
# How much one axis must trail the other to be called the limiting factor.
LIMIT_MARGIN = 12.0

# Absolute index thresholds -> Low / Medium / High level per axis
# (standard precision-ag ranges). Used for the level + colour outputs.
NDVI_HI, NDVI_MID = 0.6, 0.3      # biomass  (NDVI)
NDRE_HI, NDRE_MID = 0.6, 0.2      # nitrogen (NDRE)
LEVEL_COLORS = {"High": "#2E7D32", "Medium": "#FBC02D", "Low": "#D32F2F"}


def _abs_level(value, hi_cut, mid_cut):
    if value is None or np.isnan(value):
        return None
    if value >= hi_cut:
        return "High"
    if value >= mid_cut:
        return "Medium"
    return "Low"


def _spatial_smooth(values, xs, ys, k=8, w_self=0.7):
    """Blend each value with the mean of its k nearest neighbours."""
    try:
        from scipy.spatial import KDTree
    except Exception:
        return values
    out = values.copy()
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return out
    coords = np.column_stack([xs[mask], ys[mask]])
    vals = values[mask]
    tree = KDTree(coords)
    _, nbr = tree.query(coords, k=min(k + 1, len(coords)))
    # vals has no NaN (mask excluded them), so neighbours are always valid ->
    # vectorize the blend instead of looping per plant. nbr[:, 0] is self.
    neigh_mean = vals[nbr[:, 1:]].mean(axis=1)
    out[mask] = w_self * vals + (1 - w_self) * neigh_mean
    return out


def _percentile_score(x, mask):
    """0-100 rank-percentile of x within the valid (mask) subset."""
    out = np.full(len(x), np.nan)
    if mask.sum() == 0:
        return out
    out[mask] = pd.Series(x[mask]).rank(pct=True).to_numpy() * 100
    return out


def _gmm_clusters(feature_arrays, mask, k=4):
    """GaussianMixture over the given feature arrays. Returns labels + conf."""
    n = len(feature_arrays[0])
    labels = np.full(n, -1)
    conf = np.full(n, np.nan)
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return labels, conf, None
    if mask.sum() < max(50, k * 5):
        return labels, conf, None
    X = np.column_stack([a[mask] for a in feature_arrays])
    Xs = StandardScaler().fit_transform(X)
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=42, n_init=3)
    lab = gmm.fit_predict(Xs)
    proba = gmm.predict_proba(Xs).max(axis=1)
    # Order clusters by overall vigour (mean of standardized features).
    order = np.argsort([Xs[lab == c].mean() for c in range(k)])
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[c] for c in lab])
    labels[mask] = lab
    conf[mask] = np.round(proba, 3)
    centroids = {remap[c]: X[lab == remap[c]].mean(axis=0) for c in range(k)}
    return labels, conf, centroids


def _anomaly_flag(ndvi, ndre, mask):
    try:
        from sklearn.ensemble import IsolationForest
    except Exception:
        return np.zeros(len(ndvi), dtype=bool)
    out = np.zeros(len(ndvi), dtype=bool)
    if mask.sum() < 30:
        return out
    X = np.column_stack([ndvi[mask], ndre[mask]])
    iso = IsolationForest(contamination=0.10, random_state=42)
    out[mask] = iso.fit_predict(X) == -1
    return out


def _level(score):
    if np.isnan(score):
        return None
    if score < LOW_CUT:
        return "low"
    if score < HIGH_CUT:
        return "med"
    return "high"


def _diagnose(biomass, nitrogen, structure=None):
    """Agronomic diagnosis + limiting factor from the axis scores."""
    b, n = _level(biomass), _level(nitrogen)
    if b is None or n is None:
        return None, None
    has_s = structure is not None and not np.isnan(structure)
    s = _level(structure) if has_s else None

    # Limiting factor = the axis with the lowest score, if it trails the rest.
    axes = {"Nitrogen": nitrogen, "Biomass": biomass}
    if has_s:
        axes["Structure"] = structure
    lo_name = min(axes, key=axes.get)
    others = [v for kk, v in axes.items() if kk != lo_name]
    limiting = lo_name if (min(others) - axes[lo_name]) >= LIMIT_MARGIN else "Balanced"

    # Diagnosis.
    if b == "high" and n == "high" and s in (None, "med", "high"):
        diag = "Healthy & vigorous"
    elif b == "low" and n == "low":
        diag = "Critical (low biomass & N)"
    elif s == "low" and b in ("high", "med") and n in ("med", "high"):
        diag = "Stunted (low canopy height)"
    elif n == "low" and b in ("high", "med"):
        diag = "Nitrogen deficient"
    elif b == "low" and n in ("high", "med"):
        diag = "Sparse canopy / early growth"
    else:
        diag = "Moderate"
    return diag, limiting


def _status(score):
    if np.isnan(score):
        return "Out of Boundary", STATUS_COLORS["Out of Boundary"]
    if score < 15:
        return "Critical", STATUS_COLORS["Critical"]
    if score < 40:
        return "Stressed", STATUS_COLORS["Stressed"]
    if score < 70:
        return "Moderate", STATUS_COLORS["Moderate"]
    return "Healthy", STATUS_COLORS["Healthy"]


def classify_health(df, noise_label="NOISE"):
    """Add the agronomic health-model columns to ``df`` and return it."""
    ndvi = df["ndvi"].to_numpy(dtype=float)
    ndre = df["ndre"].to_numpy(dtype=float)
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)
    valid = ~np.isnan(ndvi) & ~np.isnan(ndre)

    # 1. Spatial smoothing.
    ndvi_sm = _spatial_smooth(ndvi, xs, ys)
    ndre_sm = _spatial_smooth(ndre, xs, ys)
    df["ndvi_smoothed"] = np.round(ndvi_sm, 3)

    # 2. Agronomic axis scores (field-relative rank).
    biomass = _percentile_score(ndvi_sm, valid)     # NDVI  -> biomass
    nitrogen = _percentile_score(ndre_sm, valid)    # NDRE  -> nitrogen

    # 2a0. TDVI (if present) strengthens the biomass reading: NDVI saturates
    #      over dense canopy, TDVI does not, so averaging their ranks separates
    #      "healthy" from "very healthy" (mirrors GNDVI's role for nitrogen).
    if "tdvi" in df.columns:
        tdvi_sm = _spatial_smooth(df["tdvi"].to_numpy(dtype=float), xs, ys)
        tpct = _percentile_score(tdvi_sm, ~np.isnan(tdvi_sm))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            biomass = np.nanmean(np.vstack([biomass, tpct]), axis=0)

    # 2a. GNDVI (if present) strengthens the nitrogen reading: NDRE & GNDVI are
    #     both chlorophyll-sensitive -> average their ranks.
    if "gndvi" in df.columns:
        gndvi_sm = _spatial_smooth(df["gndvi"].to_numpy(dtype=float), xs, ys)
        gpct = _percentile_score(gndvi_sm, ~np.isnan(gndvi_sm))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            nitrogen = np.nanmean(np.vstack([nitrogen, gpct]), axis=0)

    df["biomass_score"] = np.round(biomass, 1)
    df["nitrogen_score"] = np.round(nitrogen, 1)

    # Absolute Low/Medium/High level + colour per axis (from raw NDVI/NDRE).
    df["biomass_level"] = [_abs_level(v, NDVI_HI, NDVI_MID) for v in ndvi_sm]
    df["nitrogen_level"] = [_abs_level(v, NDRE_HI, NDRE_MID) for v in ndre_sm]
    df["biomass_level_color"] = [LEVEL_COLORS.get(x) for x in df["biomass_level"]]
    df["nitrogen_level_color"] = [LEVEL_COLORS.get(x) for x in df["nitrogen_level"]]

    # 2b. Structure axis from the canopy height model (if present).
    structure = np.full(len(df), np.nan)
    has_struct = "plant_height_m" in df.columns
    if has_struct:
        h = df["plant_height_m"].to_numpy(dtype=float)
        structure = _percentile_score(h, ~np.isnan(h))
        df["structure_score"] = np.round(structure, 1)

    # 3. Unsupervised archetypes (NDVI, NDRE, + height if available).
    feats = [ndvi_sm, ndre_sm]
    if has_struct:
        feats.append(np.nan_to_num(structure, nan=np.nanmedian(structure)))
    clusters, conf, centroids = _gmm_clusters(feats, valid)
    df["health_cluster"] = clusters
    df["cluster_confidence"] = conf

    # 4. Anomaly flag.
    df["is_anomaly"] = _anomaly_flag(ndvi_sm, ndre_sm, valid)

    # 5. Overall score + status + diagnosis.
    #    With structure: biomass 40% / nitrogen 30% / structure 30%.
    #    Without:        biomass 55% / nitrogen 45%.
    two_axis = 0.55 * biomass + 0.45 * nitrogen
    if has_struct:
        three = 0.40 * biomass + 0.30 * nitrogen + 0.30 * structure
        score = np.where(np.isnan(structure), two_axis, three)
    else:
        score = two_axis
    score = np.where(valid, score, np.nan)
    # An outlier that is low on both spectral axes is pushed toward Critical.
    both_low = valid & (biomass < LOW_CUT) & (nitrogen < LOW_CUT)
    score = np.where(both_low & df["is_anomaly"].to_numpy(),
                     np.minimum(score, 12.0), score)
    df["health_score"] = np.round(score, 1)

    statuses, colors, diags, limits = [], [], [], []
    for i in range(len(df)):
        st, col = _status(score[i])
        statuses.append(st); colors.append(col)
        d, lim = _diagnose(biomass[i], nitrogen[i],
                           structure[i] if has_struct else None)
        diags.append(d); limits.append(lim)
    df["health_status"] = statuses
    df["health_color"] = colors
    df["health_diagnosis"] = diags
    df["limiting_factor"] = limits

    # NOISE override.
    if "sector_label" in df.columns:
        nmask = (df["sector_label"] == noise_label).to_numpy()
        for c in ["health_score", "biomass_score", "nitrogen_score",
                  "structure_score", "cluster_confidence"]:
            if c in df.columns:
                df.loc[nmask, c] = np.nan
        df.loc[nmask, "health_cluster"] = -1
        df.loc[nmask, "health_status"] = "NOISE"
        df.loc[nmask, "health_color"] = STATUS_COLORS["NOISE"]
        df.loc[nmask, ["health_diagnosis", "limiting_factor",
                       "biomass_level", "nitrogen_level",
                       "biomass_level_color", "nitrogen_level_color"]] = None

    _report(df, centroids)
    return df


def _report(df, centroids):
    print("[health_model] status distribution:")
    for k, v in df["health_status"].value_counts().items():
        print(f"    {k:<16} {v:>7,}")
    diag = df["health_diagnosis"].dropna()
    if len(diag):
        print("[health_model] diagnosis:")
        for k, v in diag.value_counts().items():
            print(f"    {k:<30} {v:>7,}")
    lim = df["limiting_factor"].dropna()
    if len(lim):
        print("[health_model] limiting factor:",
              ", ".join(f"{k}={v:,}" for k, v in lim.value_counts().items()))
    if centroids:
        has_h = len(next(iter(centroids.values()))) >= 3
        print("[health_model] GMM archetype centroids:")
        for c in sorted(centroids):
            cx = centroids[c]
            extra = f"  height_rank={cx[2]:.0f}" if has_h else ""
            print(f"    cluster {c}: NDVI={cx[0]:.3f}  NDRE={cx[1]:.3f}{extra}")
