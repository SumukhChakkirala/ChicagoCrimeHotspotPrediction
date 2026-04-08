"""
Chicago Crime Dataset - Data Preprocessing Pipeline
=====================================================
Step 1 of 4 in the Spatio-Temporal Crime Risk Prediction project.

What this script does:
  1. Loads the Chicago Crime CSV
  2. Cleans and filters records
  3. Engineers temporal + spatial features
  4. Builds a 2.2km grid and assigns each crime to a cell
  5. Constructs the graph (nodes + edges) for GCN input
  6. Saves processed outputs ready for model training

Requirements:
    pip install pandas numpy scikit-learn scipy geopandas tqdm

Dataset:
    Download from: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2
    Export as CSV, rename to: chicago_crimes.csv
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
INPUT_CSV      = r"C:\Users\karan\OneDrive\Desktop\p\moneypal\6thsem\mlLab\project\Crimes_-_2001_to_Present_20260318.csv"   # path to your downloaded CSV
OUTPUT_DIR     = "processed"            # folder where outputs are saved
GRID_SIZE_KM   = 1.5                    # grid cell size in km
EDGE_DIST_KM   = 2.5                    # max distance to connect two nodes
RARE_THRESHOLD = 1000                   # crime types below this → "OTHER"

# Chicago bounding box (WGS84)
LAT_MIN, LAT_MAX = 41.6, 42.1
LON_MIN, LON_MAX = -87.9, -87.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1: LOAD & CLEAN
# ─────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(
        path,
        dtype={
            "Primary Type": "category",
            "Location Description": "category",
            "Arrest": bool,
            "Domestic": bool,
        },
        low_memory=False,
    )
    print(f"  Raw records: {len(df):,}")

    # Drop rows missing spatial coordinates
    df = df.dropna(subset=["Latitude", "Longitude"])
    print(f"  After dropping null coordinates: {len(df):,}")

    # Apply Chicago bounding box filter
    df = df[
        (df["Latitude"].between(LAT_MIN, LAT_MAX)) &
        (df["Longitude"].between(LON_MIN, LON_MAX))
    ]
    print(f"  After bounding box filter: {len(df):,}")

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df.dropna(subset=["Date"])

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
    node_df["lat_norm"] = (node_df["lat"] - LAT_MIN) / (LAT_MAX - LAT_MIN)
    node_df["lon_norm"] = (node_df["lon"] - LON_MIN) / (LON_MAX - LON_MIN)

    # Z-normalise crime count
    node_df["crime_count_z"] = (
        (node_df["crime_count"] - node_df["crime_count"].mean()) /
        (node_df["crime_count"].std() + 1e-8)
    )

    # Encode dominant crime label per node
    le = LabelEncoder()
    node_df["dominant_label"] = le.fit_transform(node_df["dominant_crime"])

    node_df = node_df.reset_index(drop=True)
    node_df["node_idx"] = node_df.index     # integer index for PyTorch Geometric

    n_nodes = len(node_df)
    print(f"  Nodes: {n_nodes}")

    # ── Edge construction via KD-Tree ───────────────────────────
    # Convert lat/lon to radians for BallTree-style query
    coords_rad = np.radians(node_df[["lat", "lon"]].values)

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

    edge_df = pd.DataFrame(edges).drop_duplicates(subset=["src", "dst"])
    print(f"  Edges: {len(edge_df):,}  |  Avg degree: {len(edge_df)/n_nodes:.2f}")

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
    print("  Chicago Crime — Preprocessing Pipeline")
    print("=" * 55)

    df = load_and_clean(INPUT_CSV)
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