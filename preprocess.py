"""
Crime Dataset - Data Preprocessing Pipeline
===========================================
Step 1 of 4 in the Spatio-Temporal Crime Risk Prediction project.

What this script does:
    1. Loads a supported incident CSV (Chicago legacy or incident-reports schema)
    2. Cleans and filters records
    3. Engineers temporal + spatial features
    4. Builds a grid and assigns each incident to a cell
    5. Constructs the graph (nodes + edges) for GCN input
    6. Saves processed outputs ready for model training

Set a custom CSV path via environment variable:
        CRIME_INPUT_CSV=/path/to/file.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_CSV      = os.getenv("CRIME_INPUT_CSV", "")
INPUT_CANDIDATES = [
    "Police_Department_Incident_Reports__2018_to_Present_20260415.csv",
    "chicago_crimes.csv",
    "Crimes_-_2001_to_Present_20260318.csv",
]
OUTPUT_DIR     = "processed"            # folder where outputs are saved
GRID_SIZE_KM   = 1.5                     # grid cell size in km
EDGE_DIST_KM   = 2.5                     # max distance to connect two nodes
RARE_THRESHOLD = 1000                    # crime types below this -> "OTHER"
SPATIAL_QUANTILE_TRIM = 0.001            # trim extreme outliers before graphing

# Spatial bounds are inferred from the dataset in load_and_clean().
LAT_MIN, LAT_MAX = None, None
LON_MIN, LON_MAX = None, None

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1: LOAD & CLEAN
# ─────────────────────────────────────────────
def resolve_input_csv() -> str:
    """Resolve dataset path from env var or known filenames in project root."""
    if INPUT_CSV:
        if os.path.exists(INPUT_CSV):
            return INPUT_CSV
        raise FileNotFoundError(
            f"CRIME_INPUT_CSV points to a missing file: {INPUT_CSV}"
        )

    for candidate in INPUT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate

    candidate_str = ", ".join(INPUT_CANDIDATES)
    raise FileNotFoundError(
        "No input CSV found. Set CRIME_INPUT_CSV or place one of these files in the project root: "
        f"{candidate_str}"
    )


def detect_schema(columns) -> dict:
    """Map supported raw schemas to canonical columns used by this pipeline."""
    colset = set(columns)

    # Chicago Crimes 2001-to-present schema
    if {"Date", "Primary Type", "Latitude", "Longitude"}.issubset(colset):
        return {
            "name": "chicago_legacy",
            "date_col": "Date",
            "crime_col": "Primary Type",
            "lat_col": "Latitude",
            "lon_col": "Longitude",
        }

    # Incident reports schema (example: Police_Department_Incident_Reports...)
    if {"Incident Category", "Latitude", "Longitude"}.issubset(colset):
        if "Incident Datetime" in colset:
            date_col = "Incident Datetime"
        elif "Incident Date" in colset:
            date_col = "Incident Date"
        else:
            raise ValueError(
                "Incident reports schema detected but missing both 'Incident Datetime' and 'Incident Date'."
            )

        return {
            "name": "incident_reports",
            "date_col": date_col,
            "crime_col": "Incident Category",
            "lat_col": "Latitude",
            "lon_col": "Longitude",
        }

    raise ValueError(
        "Unsupported dataset schema. Required columns were not found. "
        "Expected either Chicago legacy columns (Date, Primary Type, Latitude, Longitude) "
        "or incident-report columns (Incident Category, Incident Datetime/Incident Date, Latitude, Longitude)."
    )


def load_and_clean(path: str) -> pd.DataFrame:
    global LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

    print("\n[1/5] Loading dataset...")
    raw_columns = pd.read_csv(path, nrows=0).columns.tolist()
    schema = detect_schema(raw_columns)

    usecols = [
        schema["date_col"],
        schema["crime_col"],
        schema["lat_col"],
        schema["lon_col"],
    ]

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    print(f"  Raw records: {len(df):,}")
    print(f"  Schema detected: {schema['name']}")

    # Canonical columns used by downstream pipeline.
    df = df.rename(
        columns={
            schema["date_col"]: "Date",
            schema["crime_col"]: "Primary Type",
            schema["lat_col"]: "Latitude",
            schema["lon_col"]: "Longitude",
        }
    )

    # Numeric coordinates + robust timestamp parsing.
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    raw_date = df["Date"].astype("string")
    preferred_fmt = None
    if schema["name"] == "incident_reports":
        preferred_fmt = "%Y/%m/%d %I:%M:%S %p" if schema["date_col"] == "Incident Datetime" else "%Y/%m/%d"
    elif schema["name"] == "chicago_legacy":
        preferred_fmt = "%m/%d/%Y %I:%M:%S %p"

    if preferred_fmt is not None:
        parsed = pd.to_datetime(raw_date, format=preferred_fmt, errors="coerce")
        missing_mask = parsed.isna()
        if missing_mask.any():
            parsed.loc[missing_mask] = pd.to_datetime(raw_date[missing_mask], errors="coerce")
        df["Date"] = parsed
    else:
        df["Date"] = pd.to_datetime(raw_date, errors="coerce")

    # Normalise crime categories and drop unusable rows.
    df["Primary Type"] = df["Primary Type"].astype("string").str.strip().str.upper()
    df = df[
        df["Primary Type"].notna() &
        (df["Primary Type"] != "") &
        (df["Primary Type"] != "<NA>")
    ]

    df = df.dropna(subset=["Latitude", "Longitude", "Date"])
    print(f"  After dropping rows missing key fields: {len(df):,}")

    if df.empty:
        raise ValueError("No valid rows available after cleaning key fields.")

    # Trim extreme coordinate outliers using quantiles.
    q = SPATIAL_QUANTILE_TRIM
    lat_lo, lat_hi = df["Latitude"].quantile([q, 1 - q])
    lon_lo, lon_hi = df["Longitude"].quantile([q, 1 - q])
    df = df[
        df["Latitude"].between(lat_lo, lat_hi) &
        df["Longitude"].between(lon_lo, lon_hi)
    ]
    print(f"  After spatial outlier trim: {len(df):,}")

    if df.empty:
        raise ValueError("No valid rows remained after spatial filtering.")

    # Spatial bounds are dataset-specific and used by griding + normalisation.
    LAT_MIN = float(df["Latitude"].min())
    LAT_MAX = float(df["Latitude"].max())
    LON_MIN = float(df["Longitude"].min())
    LON_MAX = float(df["Longitude"].max())

    print(
        "  Spatial bounds: "
        f"lat [{LAT_MIN:.6f}, {LAT_MAX:.6f}], "
        f"lon [{LON_MIN:.6f}, {LON_MAX:.6f}]"
    )

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 2: TEMPORAL FEATURE ENGINEERING
# ─────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/5] Engineering temporal features...")

    df["hour"]        = df["Date"].dt.hour
    df["day_of_week"] = df["Date"].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"]       = df["Date"].dt.month
    df["year"]        = df["Date"].dt.year

    # Cyclical encoding — prevents the model from treating hour 23 and hour 0 as far apart
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

    # Season: 0=Winter, 1=Spring, 2=Summer, 3=Fall
    df["season"] = df["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )

    # Time-of-day bucket for interpretability
    df["time_bucket"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 20, 23],
        labels=["Night", "Morning", "Afternoon", "Evening", "Late Night"]
    )

    print(f"  Temporal features added: hour, dow, month, season + cyclical encodings")
    return df


# ─────────────────────────────────────────────
# STEP 3: CRIME TYPE CONSOLIDATION
# ─────────────────────────────────────────────
def consolidate_crime_types(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/5] Consolidating rare crime types...")

    type_counts = df["Primary Type"].value_counts()
    rare_types  = type_counts[type_counts < RARE_THRESHOLD].index
    print(f"  Crime types before: {type_counts.shape[0]}")
    print(f"  Rare types (< {RARE_THRESHOLD} records) merged into OTHER: {len(rare_types)}")

    df["crime_type"] = df["Primary Type"].astype(str)
    df.loc[df["crime_type"].isin(rare_types), "crime_type"] = "OTHER"

    print(f"  Crime types after: {df['crime_type'].nunique()}")
    print(f"  Top 5: {df['crime_type'].value_counts().head().to_dict()}")

    # Encode labels (needed for classification target)
    le = LabelEncoder()
    df["crime_label"] = le.fit_transform(df["crime_type"])

    # Save label mapping
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    pd.DataFrame(label_map.items(), columns=["crime_type", "label"]).to_csv(
        f"{OUTPUT_DIR}/label_map.csv", index=False
    )
    print(f"  Label map saved → {OUTPUT_DIR}/label_map.csv")

    return df


# ─────────────────────────────────────────────
# STEP 4: SPATIAL GRID CONSTRUCTION
# ─────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised Haversine distance in kilometres."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def build_spatial_grid(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/5] Building spatial grid...")

    if any(v is None for v in (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)):
        raise RuntimeError("Spatial bounds are unset. Run load_and_clean() first.")

    # Degrees per km (approximate)
    lat_per_km = 1 / 110.574
    lon_per_km = 1 / (111.320 * np.cos(np.radians((LAT_MIN + LAT_MAX) / 2)))

    lat_step = GRID_SIZE_KM * lat_per_km
    lon_step = GRID_SIZE_KM * lon_per_km

    # Assign each crime to a grid cell
    df["grid_row"] = ((df["Latitude"]  - LAT_MIN) / lat_step).astype(int)
    df["grid_col"] = ((df["Longitude"] - LON_MIN) / lon_step).astype(int)
    df["grid_id"]  = df["grid_row"].astype(str) + "_" + df["grid_col"].astype(str)

    print(f"  Unique grid cells (nodes): {df['grid_id'].nunique()}")
    return df


# ─────────────────────────────────────────────
# STEP 5: AGGREGATE NODE FEATURES & BUILD GRAPH
# ─────────────────────────────────────────────
def build_graph(df: pd.DataFrame):
    print("\n[5/5] Building graph (nodes + edges)...")

    # ── Node features ──────────────────────────────────────────
    # Each node = one grid cell
    # Features: centroid lat/lon (normalised), crime count, dominant crime type

    node_df = (
        df.groupby("grid_id")
        .agg(
            lat        = ("Latitude",    "mean"),
            lon        = ("Longitude",   "mean"),
            crime_count= ("crime_label", "count"),
            # Most common crime type in this cell
            dominant_crime = ("crime_type", lambda x: x.value_counts().idxmax()),
        )
        .reset_index()
    )

    # Normalise coordinates to [0, 1]
    lat_range = max(LAT_MAX - LAT_MIN, 1e-8)
    lon_range = max(LON_MAX - LON_MIN, 1e-8)
    node_df["lat_norm"] = (node_df["lat"] - LAT_MIN) / lat_range
    node_df["lon_norm"] = (node_df["lon"] - LON_MIN) / lon_range

    # Z-normalise crime count
    node_df["crime_count_z"] = (
        (node_df["crime_count"] - node_df["crime_count"].mean()) /
        (node_df["crime_count"].std() + 1e-8)
    )
    # Log transform (helps reduce skew)
    node_df["log_crime_count"] = np.log1p(node_df["crime_count"])
    # Encode dominant crime label per node
    le = LabelEncoder()
    node_df["dominant_label"] = le.fit_transform(node_df["dominant_crime"])

    node_df = node_df.reset_index(drop=True)
    node_df["node_idx"] = node_df.index     # integer index for PyTorch Geometric

    n_nodes = len(node_df)
    print(f"  Nodes: {n_nodes}")

    # ── Edge construction via KD-Tree ───────────────────────────
    # Build KD-Tree on (lat, lon) in degrees — fast O(n log n) neighbour search
    tree = cKDTree(node_df[["lat", "lon"]].values)

    # Find all pairs within EDGE_DIST_KM
    lat_per_km = 1 / 110.574
    radius_deg = EDGE_DIST_KM * lat_per_km   # approximate degree radius

    edges = []
    print(f"  Building edges (threshold = {EDGE_DIST_KM} km)...")
    for i, row in tqdm(node_df.iterrows(), total=n_nodes):
        neighbours = tree.query_ball_point([row["lat"], row["lon"]], r=radius_deg)
        for j in neighbours:
            if j != i:
                dist_km = haversine_km(row["lat"], row["lon"],
                                       node_df.at[j, "lat"], node_df.at[j, "lon"])
                if dist_km <= EDGE_DIST_KM:
                    weight = 1.0 / (dist_km + 1e-6)
                    edges.append({"src": i, "dst": j, "weight": weight, "dist_km": dist_km})

    edge_df = pd.DataFrame(edges, columns=["src", "dst", "weight", "dist_km"])
    if not edge_df.empty:
        edge_df = edge_df.drop_duplicates(subset=["src", "dst"])
    print(f"  Edges: {len(edge_df):,}  |  Avg degree: {len(edge_df)/max(n_nodes, 1):.2f}")

    # ── Save outputs ────────────────────────────────────────────
    crimes_out = f"{OUTPUT_DIR}/crimes_processed.csv"
    nodes_out  = f"{OUTPUT_DIR}/nodes.csv"
    edges_out  = f"{OUTPUT_DIR}/edges.csv"

    df.to_csv(crimes_out, index=False)
    node_df.to_csv(nodes_out, index=False)
    edge_df.to_csv(edges_out, index=False)

    print(f"\n  ✓ crimes_processed.csv  → {crimes_out}")
    print(f"  ✓ nodes.csv             → {nodes_out}")
    print(f"  ✓ edges.csv             → {edges_out}")

    return node_df, edge_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Crime Risk — Preprocessing Pipeline")
    print("=" * 55)

    input_csv = resolve_input_csv()
    print(f"  Input CSV : {input_csv}")

    df = load_and_clean(input_csv)
    df = add_temporal_features(df)
    df = consolidate_crime_types(df)
    df = build_spatial_grid(df)
    node_df, edge_df = build_graph(df)

    print("\n" + "=" * 55)
    print("  Preprocessing complete!")
    print(f"  Records  : {len(df):,}")
    print(f"  Nodes    : {len(node_df):,}")
    print(f"  Edges    : {len(edge_df):,}")
    print(f"  Outputs  : ./{OUTPUT_DIR}/")
    print("=" * 55)


if __name__ == "__main__":
    main()