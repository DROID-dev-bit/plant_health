#!/usr/bin/env python3
"""
Pineapple Plant Health Classifier
==================================
Classifies pineapple plant health using NDVI (early/mid-season) and NDRE
(mid/late-season) indices from UAV sensor data.

Usage:
    python health_classifier.py classify --index ndvi --value 0.72
    python health_classifier.py classify --index ndre --value 0.65
    python health_classifier.py auto    --ndvi 0.82 --ndre 0.71 --stage mid
    python health_classifier.py batch   --input field_map.csv --output report.json
    python health_classifier.py report  --input field_map.csv --output health_report.json
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional


# ── Thresholds ────────────────────────────────────────────────────────────────

NDVI_THRESHOLDS = [
    (-1.0, 0.2,  "Bare Soil",   "critical",
     "Replanting required — no viable crop detected."),
    ( 0.2, 0.6,  "Unhealthy",   "warning",
     "Immediate field scouting required. Check for disease, pest pressure, or water stress."),
    ( 0.6, 0.8,  "Healthy",     "good",
     "Maintain current protocols. Monitor during next flight."),
    ( 0.8, 1.0,  "Saturated",   "info",
     "NDVI saturated — switch to NDRE for accurate chlorophyll assessment."),
]

NDRE_THRESHOLDS = [
    (-1.0, 0.2,  "Bare Soil / Severe Defoliation", "critical",
     "Critical — severe biomass loss or bare soil. Immediate ground inspection needed."),
    ( 0.2, 0.6,  "Unhealthy",                      "warning",
     "Critical chlorophyll deficit. Intervention required: check N/K levels, irrigation, disease."),
    ( 0.6, 0.8,  "Optimal Health",                 "good",
     "Chlorophyll content is optimal. Maintain fertilization and irrigation schedule."),
    ( 0.8, 1.0,  "Peak Maturity / Ready for Harvest", "harvest",
     "Plant at peak maturity. Schedule harvest within the recommended window."),
]

NDVI_BASELINE_EARLY   = 0.72   # healthy early-season NDVI
NDRE_CRITICAL_MID     = 0.60   # minimum acceptable NDRE mid/late season

SEASON_ALIASES = {
    "early": "early", "e": "early",
    "mid":   "mid",   "m": "mid",
    "late":  "late",  "l": "late",
}

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    index:          str           # "ndvi" | "ndre"
    value:          float
    status:         str           # human label
    severity:       str           # critical | warning | good | info | harvest
    recommendation: str
    score:          float         # 0–100 normalised health score
    baseline_delta: Optional[float] = None  # vs NDVI_BASELINE or NDRE_CRITICAL


@dataclass
class AutoRouteResult:
    selected_index: str           # "ndvi" | "ndre"
    reason:         str
    ndvi_result:    ClassificationResult
    ndre_result:    Optional[ClassificationResult]
    primary_result: ClassificationResult


@dataclass
class PlantRecord:
    plant_id:   str
    lat:        Optional[float]
    lon:        Optional[float]
    ndvi:       Optional[float]
    ndre:       Optional[float]
    stage:      Optional[str]     # early | mid | late
    result:     Optional[ClassificationResult] = None
    auto_route: Optional[AutoRouteResult]      = None


# ── Core classifier ───────────────────────────────────────────────────────────

def _score_ndvi(value: float) -> float:
    """Map NDVI to 0–100 health score."""
    if value < 0.0:
        return 0.0
    if value < 0.2:
        return value / 0.2 * 15              # 0–15
    if value < 0.6:
        return 15 + (value - 0.2) / 0.4 * 45   # 15–60
    if value < 0.8:
        return 60 + (value - 0.6) / 0.2 * 35   # 60–95
    return 95 + (value - 0.8) / 0.2 * 5         # 95–100 (saturated)


def _score_ndre(value: float) -> float:
    """Map NDRE to 0–100 health score."""
    if value < 0.0:
        return 0.0
    if value < 0.2:
        return value / 0.2 * 10              # 0–10
    if value < 0.6:
        return 10 + (value - 0.2) / 0.4 * 55   # 10–65
    if value < 0.8:
        return 65 + (value - 0.6) / 0.2 * 30   # 65–95
    return 95 + (value - 0.8) / 0.2 * 5         # 95–100


def classify_ndvi(value: float) -> ClassificationResult:
    """Classify a single NDVI reading."""
    if math.isnan(value):
        raise ValueError("NDVI value is NaN.")
    for lo, hi, status, severity, rec in NDVI_THRESHOLDS:
        if lo <= value < hi:
            return ClassificationResult(
                index="ndvi", value=round(value, 4),
                status=status, severity=severity, recommendation=rec,
                score=round(_score_ndvi(value), 1),
                baseline_delta=round(value - NDVI_BASELINE_EARLY, 4),
            )
    # Exact 1.0 edge case
    if value == 1.0:
        lo, hi, status, severity, rec = NDVI_THRESHOLDS[-1]
        return ClassificationResult(
            index="ndvi", value=1.0, status=status, severity=severity,
            recommendation=rec, score=100.0,
            baseline_delta=round(1.0 - NDVI_BASELINE_EARLY, 4),
        )
    raise ValueError(f"NDVI value {value} out of expected range [-1, 1].")


def classify_ndre(value: float) -> ClassificationResult:
    """Classify a single NDRE reading."""
    if math.isnan(value):
        raise ValueError("NDRE value is NaN.")
    for lo, hi, status, severity, rec in NDRE_THRESHOLDS:
        if lo <= value < hi:
            return ClassificationResult(
                index="ndre", value=round(value, 4),
                status=status, severity=severity, recommendation=rec,
                score=round(_score_ndre(value), 1),
                baseline_delta=round(value - NDRE_CRITICAL_MID, 4),
            )
    if value == 1.0:
        lo, hi, status, severity, rec = NDRE_THRESHOLDS[-1]
        return ClassificationResult(
            index="ndre", value=1.0, status=status, severity=severity,
            recommendation=rec, score=100.0,
            baseline_delta=round(1.0 - NDRE_CRITICAL_MID, 4),
        )
    raise ValueError(f"NDRE value {value} out of expected range [-1, 1].")


def classify(index: Literal["ndvi", "ndre"], value: float) -> ClassificationResult:
    """Public single-index entry point."""
    index = index.lower()
    if index == "ndvi":
        return classify_ndvi(value)
    if index == "ndre":
        return classify_ndre(value)
    raise ValueError(f"Unknown index '{index}'. Use 'ndvi' or 'ndre'.")


# ── Auto-router ───────────────────────────────────────────────────────────────

def auto_route(
    ndvi: float,
    ndre:  Optional[float] = None,
    stage: Optional[str]   = None,
) -> AutoRouteResult:
    """
    Decide which index is authoritative and return the primary classification.

    Routing logic (in priority order):
      1. If stage is 'late' → use NDRE (if available)
      2. If NDVI ≥ 0.8 (saturated) → prefer NDRE for accurate reading
      3. If stage is 'mid' and NDRE available → use NDRE
      4. Default: use NDVI (early season or no NDRE)
    """
    ndvi_result = classify_ndvi(ndvi)
    ndre_result = classify_ndre(ndre) if ndre is not None and not math.isnan(ndre) else None

    stage_norm = SEASON_ALIASES.get((stage or "").lower())

    if stage_norm == "late" and ndre_result is not None:
        selected = "ndre"
        reason = "Late-season — NDRE is the authoritative chlorophyll indicator."
    elif ndvi_result.severity == "info" and ndre_result is not None:
        selected = "ndre"
        reason = "NDVI saturated (≥0.8) — switching to NDRE for accurate canopy assessment."
    elif stage_norm == "mid" and ndre_result is not None:
        selected = "ndre"
        reason = "Mid-season with NDRE available — using NDRE for canopy health."
    else:
        selected = "ndvi"
        reason = "Early season or no NDRE data — using NDVI as primary health index."

    primary = ndre_result if selected == "ndre" else ndvi_result

    return AutoRouteResult(
        selected_index=selected,
        reason=reason,
        ndvi_result=ndvi_result,
        ndre_result=ndre_result,
        primary_result=primary,
    )


# ── CSV batch processor ───────────────────────────────────────────────────────

REQUIRED_COLS = {"plant_id"}
OPTIONAL_COLS = {"lat", "lon", "ndvi", "ndre", "stage"}


def batch_process(csv_path: Path) -> list[PlantRecord]:
    """
    Process a CSV field map. Expected columns (all optional except plant_id):
        plant_id, lat, lon, ndvi, ndre, stage

    Returns a list of PlantRecord objects with classification results.
    """
    records: list[PlantRecord] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
        missing = REQUIRED_COLS - headers
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}. Found: {headers}")

        for row in reader:
            pid = row.get("plant_id", "").strip()
            if not pid:
                continue

            def _float(col: str) -> Optional[float]:
                v = row.get(col, "").strip()
                try:
                    f = float(v)
                    return None if math.isnan(f) else f
                except (ValueError, TypeError):
                    return None

            ndvi_val  = _float("ndvi")
            ndre_val  = _float("ndre")
            stage_val = row.get("stage", "").strip() or None

            rec = PlantRecord(
                plant_id=pid,
                lat=_float("lat"),
                lon=_float("lon"),
                ndvi=ndvi_val,
                ndre=ndre_val,
                stage=stage_val,
            )

            if ndvi_val is not None:
                ar = auto_route(ndvi_val, ndre_val, stage_val)
                rec.auto_route = ar
                rec.result = ar.primary_result
            elif ndre_val is not None:
                rec.result = classify_ndre(ndre_val)

            records.append(rec)

    return records


# ── Health score report generator ─────────────────────────────────────────────

def generate_report(records: list[PlantRecord]) -> dict:
    """Summarise batch results into an actionable farm report."""
    total = len(records)
    classified = [r for r in records if r.result is not None]

    severity_counts: dict[str, int] = {}
    status_counts:   dict[str, int] = {}
    scores:          list[float]    = []
    needs_harvest:   list[str]      = []
    needs_scouting:  list[str]      = []
    needs_replant:   list[str]      = []

    for rec in classified:
        res = rec.result
        severity_counts[res.severity] = severity_counts.get(res.severity, 0) + 1
        status_counts[res.status]     = status_counts.get(res.status, 0) + 1
        scores.append(res.score)

        if res.severity == "harvest":
            needs_harvest.append(rec.plant_id)
        elif res.severity == "warning":
            needs_scouting.append(rec.plant_id)
        elif res.severity == "critical":
            needs_replant.append(rec.plant_id)

    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Farm-level health grade
    if avg_score >= 80:
        farm_grade = "A — Excellent"
    elif avg_score >= 65:
        farm_grade = "B — Good"
    elif avg_score >= 50:
        farm_grade = "C — Fair"
    elif avg_score >= 30:
        farm_grade = "D — Poor"
    else:
        farm_grade = "F — Critical"

    recommendations = []
    crit = severity_counts.get("critical", 0)
    warn = severity_counts.get("warning",  0)
    harv = severity_counts.get("harvest",  0)

    if crit > 0:
        pct = crit / total * 100
        recommendations.append({
            "priority": "HIGH",
            "action": "Immediate replanting / ground inspection",
            "affected_plants": crit,
            "pct": round(pct, 1),
            "plant_ids_sample": needs_replant[:20],
        })
    if warn > 0:
        pct = warn / total * 100
        recommendations.append({
            "priority": "MEDIUM",
            "action": "Field scouting — check for disease, N-deficiency, water stress",
            "affected_plants": warn,
            "pct": round(pct, 1),
            "plant_ids_sample": needs_scouting[:20],
        })
    if harv > 0:
        pct = harv / total * 100
        recommendations.append({
            "priority": "HARVEST",
            "action": "Schedule harvest — plants at peak maturity",
            "affected_plants": harv,
            "pct": round(pct, 1),
            "plant_ids_sample": needs_harvest[:20],
        })

    return {
        "summary": {
            "total_plants":     total,
            "classified":       len(classified),
            "average_score":    round(avg_score, 2),
            "farm_health_grade": farm_grade,
        },
        "severity_breakdown": severity_counts,
        "status_breakdown":   status_counts,
        "recommendations":    recommendations,
        "plants": [
            {
                "plant_id":      r.plant_id,
                "lat":           r.lat,
                "lon":           r.lon,
                "ndvi":          r.ndvi,
                "ndre":          r.ndre,
                "stage":         r.stage,
                "selected_index": r.auto_route.selected_index if r.auto_route else (r.result.index if r.result else None),
                "status":        r.result.status        if r.result else "No Data",
                "severity":      r.result.severity      if r.result else "unknown",
                "score":         r.result.score         if r.result else None,
                "recommendation": r.result.recommendation if r.result else "Insufficient spectral data.",
            }
            for r in records
        ],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_result(res: ClassificationResult) -> None:
    SEV_ICONS = {
        "critical": "🔴", "warning": "🟡", "good": "🟢",
        "info": "🔵", "harvest": "🌿",
    }
    icon = SEV_ICONS.get(res.severity, "⚪")
    print(f"\n  Index       : {res.index.upper()}")
    print(f"  Value       : {res.value}")
    print(f"  Status      : {icon}  {res.status}")
    print(f"  Severity    : {res.severity.upper()}")
    print(f"  Health Score: {res.score} / 100")
    if res.baseline_delta is not None:
        sign = "+" if res.baseline_delta >= 0 else ""
        base = NDVI_BASELINE_EARLY if res.index == "ndvi" else NDRE_CRITICAL_MID
        print(f"  vs Baseline : {sign}{res.baseline_delta}  (baseline={base})")
    print(f"  Action      : {res.recommendation}")


def cmd_classify(args: argparse.Namespace) -> None:
    res = classify(args.index, args.value)
    if args.json:
        print(json.dumps(asdict(res), indent=2))
    else:
        _print_result(res)


def cmd_auto(args: argparse.Namespace) -> None:
    ar = auto_route(args.ndvi, args.ndre, args.stage)
    if args.json:
        d = {
            "selected_index":  ar.selected_index,
            "reason":          ar.reason,
            "ndvi_result":     asdict(ar.ndvi_result),
            "ndre_result":     asdict(ar.ndre_result) if ar.ndre_result else None,
            "primary_result":  asdict(ar.primary_result),
        }
        print(json.dumps(d, indent=2))
    else:
        print(f"\n  Auto-Router: using {ar.selected_index.upper()}")
        print(f"  Reason     : {ar.reason}")
        print("\n  ── NDVI ──")
        _print_result(ar.ndvi_result)
        if ar.ndre_result:
            print("\n  ── NDRE ──")
            _print_result(ar.ndre_result)
        print(f"\n  ★  Primary result ({ar.selected_index.upper()}):")
        _print_result(ar.primary_result)


def cmd_batch(args: argparse.Namespace) -> None:
    records = batch_process(Path(args.input))
    report  = generate_report(records)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[✓] Processed {report['summary']['total_plants']:,} plants → {out_path}")
    print(f"    Farm Health Grade : {report['summary']['farm_health_grade']}")
    print(f"    Average Score     : {report['summary']['average_score']} / 100")
    for rec in report["recommendations"]:
        print(f"    [{rec['priority']}] {rec['action']} — {rec['affected_plants']} plants ({rec['pct']}%)")


def cmd_report(args: argparse.Namespace) -> None:
    """Alias for batch that also pretty-prints the summary to stdout."""
    cmd_batch(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pineapple Plant Health Classifier — UAV NDVI/NDRE Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # classify
    pc = sub.add_parser("classify", help="Classify a single index value")
    pc.add_argument("--index", required=True, choices=["ndvi", "ndre"],
                    help="Which index to classify")
    pc.add_argument("--value", required=True, type=float,
                    help="Index value, e.g. 0.72")
    pc.add_argument("--json", action="store_true", help="Output as JSON")

    # auto
    pa = sub.add_parser("auto", help="Auto-route between NDVI and NDRE")
    pa.add_argument("--ndvi", required=True, type=float)
    pa.add_argument("--ndre", type=float, default=None)
    pa.add_argument("--stage", choices=["early", "mid", "late", "e", "m", "l"],
                    default=None, help="Growth season stage")
    pa.add_argument("--json", action="store_true")

    # batch
    pb = sub.add_parser("batch", help="Batch process a CSV field map")
    pb.add_argument("--input",  required=True, help="Input CSV path")
    pb.add_argument("--output", required=True, help="Output JSON report path")

    # report (alias)
    pr = sub.add_parser("report", help="Same as batch — generate health score report")
    pr.add_argument("--input",  required=True)
    pr.add_argument("--output", required=True)

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "classify": cmd_classify,
        "auto":     cmd_auto,
        "batch":    cmd_batch,
        "report":   cmd_report,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
