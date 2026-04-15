"""
Chicago Crime Prediction — FastAPI Backend
==========================================
Endpoints:
  GET  /api/nodes          → all grid cells with predicted crime risk
  GET  /api/crime-types    → list of crime type classes
  GET  /api/stats          → summary statistics
  POST /api/predict        → predict risk for a lat/lon + date/time

Install:
    pip install fastapi uvicorn pandas numpy

Run:
    uvicorn app:app --reload --port 8000
"""

import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PREDICTIONS_CSV  = "outputs/node_predictions.csv"
CRIMES_CSV       = "processed/crimes_processed.csv"

app = FastAPI(
    title="Crime Risk API",
    description="Serves GCN-based crime hotspot predictions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LOAD DATA AT STARTUP
# ─────────────────────────────────────────────
node_df          = None
crimes_df        = None
crime_types      = []
prob_cols        = []
LAT_MIN          = None
LAT_MAX          = None
LON_MIN          = None
LON_MAX          = None
BOUNDARY_PADDING_DEG = 0.01

# Temporal multipliers — computed from historical data
# shape: {grid_id: {crime_type: {hour: multiplier, month: multiplier, dow: multiplier}}}
temporal_index   = {}

@app.on_event("startup")
def load_data():
    global node_df, crimes_df, crime_types, prob_cols, temporal_index
    global LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

    if not os.path.exists(PREDICTIONS_CSV):
        raise RuntimeError(f"Missing {PREDICTIONS_CSV} — run model.py first")

    node_df   = pd.read_csv(PREDICTIONS_CSV)
    if node_df.empty:
        raise RuntimeError(f"{PREDICTIONS_CSV} is empty")

    LAT_MIN = float(node_df["lat"].min())
    LAT_MAX = float(node_df["lat"].max())
    LON_MIN = float(node_df["lon"].min())
    LON_MAX = float(node_df["lon"].max())

    prob_cols   = [c for c in node_df.columns if c.startswith("prob_")]
    crime_types = [c.replace("prob_", "") for c in prob_cols]

    print(f"✓ Loaded {len(node_df)} nodes with {len(crime_types)} crime types")
    print(
        "  Spatial bounds: "
        f"lat [{LAT_MIN:.6f}, {LAT_MAX:.6f}], "
        f"lon [{LON_MIN:.6f}, {LON_MAX:.6f}]"
    )

    # Build temporal index from historical crimes
    if os.path.exists(CRIMES_CSV):
        print("  Building temporal index from historical data...")
        crimes_df = pd.read_csv(
            CRIMES_CSV,
            usecols=["grid_id", "crime_type", "hour", "day_of_week", "month"],
            dtype={"grid_id": str, "crime_type": str}
        )
        _build_temporal_index()
        print("  ✓ Temporal index ready")
    else:
        print("  ⚠ crimes_processed.csv not found — temporal adjustment disabled")


def _build_temporal_index():
    """
    For each grid cell + crime type, compute how much more/less likely
    each hour, day-of-week, and month is compared to the average.
    Multiplier > 1 means higher than average risk at that time.
    """
    global temporal_index

    for grid_id, grp in crimes_df.groupby("grid_id"):
        temporal_index[str(grid_id)] = {}
        for crime, cgrp in grp.groupby("crime_type"):
            total = len(cgrp)
            if total < 10:
                continue  # not enough data for reliable stats

            # Hour multipliers (0–23)
            hour_counts  = cgrp["hour"].value_counts()
            hour_avg     = total / 24
            hour_mult    = {h: hour_counts.get(h, 0.1) / hour_avg for h in range(24)}

            # Day-of-week multipliers (0=Mon … 6=Sun)
            dow_counts   = cgrp["day_of_week"].value_counts()
            dow_avg      = total / 7
            dow_mult     = {d: dow_counts.get(d, 0.1) / dow_avg for d in range(7)}

            # Month multipliers (1–12)
            month_counts = cgrp["month"].value_counts()
            month_avg    = total / 12
            month_mult   = {m: month_counts.get(m, 0.1) / month_avg for m in range(1, 13)}

            temporal_index[str(grid_id)][crime] = {
                "hour":  hour_mult,
                "dow":   dow_mult,
                "month": month_mult,
            }


def apply_temporal_adjustment(
    risk_scores: dict,
    grid_id: str,
    hour: int,
    dow: int,
    month: int
) -> dict:
    """
    Adjust GCN base risk scores using historical temporal patterns.
    Final score = base_score * hour_mult * dow_mult * month_mult (clamped to 0–1)
    """
    if str(grid_id) not in temporal_index:
        return risk_scores  # no temporal data, return as-is

    adjusted = {}
    cell_idx = temporal_index[str(grid_id)]

    for crime, base_score in risk_scores.items():
        if crime in cell_idx:
            t = cell_idx[crime]
            h_mult = t["hour"].get(hour,  1.0)
            d_mult = t["dow"].get(dow,    1.0)
            m_mult = t["month"].get(month, 1.0)

            # Combined multiplier — normalised so it doesn't explode
            combined = (h_mult * d_mult * m_mult) ** (1/3)  # geometric mean
            adjusted[crime] = round(min(float(base_score) * combined, 1.0), 4)
        else:
            adjusted[crime] = base_score

    # Re-normalise so scores sum to 1
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}

    return adjusted


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


def find_nearest_node(lat: float, lon: float) -> pd.Series:
    dists = haversine_km(lat, lon, node_df["lat"].values, node_df["lon"].values)
    return node_df.iloc[dists.argmin()]


def within_dataset_bounds(lat: float, lon: float) -> bool:
    if any(v is None for v in (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)):
        return False

    lat_min = LAT_MIN - BOUNDARY_PADDING_DEG
    lat_max = LAT_MAX + BOUNDARY_PADDING_DEG
    lon_min = LON_MIN - BOUNDARY_PADDING_DEG
    lon_max = LON_MAX + BOUNDARY_PADDING_DEG
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def row_to_dict(row: pd.Series) -> dict:
    risk_scores = {ct: round(float(row[f"prob_{ct}"]), 4) for ct in crime_types}
    dominant    = max(risk_scores, key=risk_scores.get)
    return {
        "grid_id":         str(row.get("grid_id", "")),
        "lat":             round(float(row["lat"]), 6),
        "lon":             round(float(row["lon"]), 6),
        "crime_count":     int(row.get("crime_count", 0)),
        "predicted_crime": str(row.get("predicted_crime", dominant)),
        "dominant_crime":  dominant,
        "risk_score":      risk_scores[dominant],
        "risk_scores":     risk_scores,
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Crime Risk API is running"}


@app.get("/api/meta")
def get_meta():
    if any(v is None for v in (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)):
        raise HTTPException(500, "Dataset bounds are unavailable")

    return {
        "title": "Crime Risk Map",
        "bounds": {
            "lat_min": LAT_MIN,
            "lat_max": LAT_MAX,
            "lon_min": LON_MIN,
            "lon_max": LON_MAX,
            "padding_deg": BOUNDARY_PADDING_DEG,
        },
        "center": {
            "lat": (LAT_MIN + LAT_MAX) / 2,
            "lon": (LON_MIN + LON_MAX) / 2,
        },
    }


@app.get("/api/nodes")
def get_nodes(crime_type: Optional[str] = None):
    results = []
    for _, row in node_df.iterrows():
        d = row_to_dict(row)
        if crime_type:
            ct = crime_type.upper()
            if ct not in crime_types:
                raise HTTPException(400, f"Unknown crime type: {ct}. Valid: {crime_types}")
            d["risk_score"]   = d["risk_scores"].get(ct, 0.0)
            d["dominant_crime"] = ct
        results.append(d)
    return {"nodes": results, "count": len(results)}


@app.get("/api/crime-types")
def get_crime_types():
    return {"crime_types": crime_types}


@app.get("/api/stats")
def get_stats():
    crime_counts  = node_df["predicted_crime"].value_counts().to_dict()
    top_risk_rows = node_df.copy()
    top_risk_rows["max_risk"] = node_df[prob_cols].max(axis=1)
    top5 = top_risk_rows.nlargest(5, "max_risk")
    hotspots = [
        {
            "lat":   round(float(r["lat"]), 6),
            "lon":   round(float(r["lon"]), 6),
            "crime": str(r["predicted_crime"]),
            "risk":  round(float(r["max_risk"]), 4),
        }
        for _, r in top5.iterrows()
    ]
    return {
        "total_nodes":       len(node_df),
        "crime_type_counts": crime_counts,
        "top_hotspots":      hotspots,
        "avg_crime_count":   round(float(node_df["crime_count"].mean()), 1),
    }


class PredictRequest(BaseModel):
    lat:         float
    lon:         float
    hour:        Optional[int] = 12   # 0–23
    day_of_week: Optional[int] = 0    # 0=Mon … 6=Sun
    month:       Optional[int] = 6    # 1–12
    crime_type:  Optional[str] = None


@app.post("/api/predict")
def predict(req: PredictRequest):
    if not (-90 <= req.lat <= 90) or not (-180 <= req.lon <= 180):
        raise HTTPException(400, "Invalid coordinates")
    if not within_dataset_bounds(req.lat, req.lon):
        raise HTTPException(
            400,
            (
                "Coordinates are outside dataset bounds: "
                f"lat [{LAT_MIN:.4f}, {LAT_MAX:.4f}], "
                f"lon [{LON_MIN:.4f}, {LON_MAX:.4f}]"
            ),
        )

    node   = find_nearest_node(req.lat, req.lon)
    result = row_to_dict(node)

    # Apply temporal adjustment if time info provided
    result["risk_scores"] = apply_temporal_adjustment(
        result["risk_scores"],
        grid_id = result["grid_id"],
        hour    = req.hour,
        dow     = req.day_of_week,
        month   = req.month,
    )

    # Recompute dominant crime + risk score after temporal adjustment
    dominant              = max(result["risk_scores"], key=result["risk_scores"].get)
    result["dominant_crime"]  = dominant
    result["risk_score"]      = result["risk_scores"][dominant]
    result["predicted_crime"] = dominant

    if req.crime_type:
        ct = req.crime_type.upper()
        if ct not in crime_types:
            raise HTTPException(400, f"Unknown crime type: {ct}. Valid: {crime_types}")
        result["risk_score"]    = result["risk_scores"].get(ct, 0.0)
        result["dominant_crime"] = ct

    result["query"] = {
        "lat":         req.lat,
        "lon":         req.lon,
        "hour":        req.hour,
        "day_of_week": req.day_of_week,
        "month":       req.month,
    }

    result["temporal_adjusted"] = bool(temporal_index)
    return result