import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "ghi","dni","temperature","cloud_pct","clearness","elevation",
    "slope","aspect","ndvi","road_km","grid_km","wind_speed",
    "precipitation","humidity","land_score"
]

TARGET_COL = "suitability_score"

WEIGHTS = {
    "ghi":0.35,"slope":0.20,"land":0.15,
    "ndvi":0.10,"cloud":0.10,"grid":0.05,"road":0.05
}

GHI_MIN, GHI_MAX = 2.0, 7.5
SLOPE_MAX = 30.0
GRID_NEAR, GRID_FAR = 1.0, 50.0
ROAD_NEAR, ROAD_FAR = 0.5, 20.0
NDVI_MIN, NDVI_MAX = -0.2, 0.9

FEATURE_DEFAULTS = {
    "ghi":4.5,"dni":4.0,"temperature":27.0,"cloud_pct":30.0,
    "clearness":0.55,"elevation":300.0,"slope":5.0,"aspect":180.0,
    "ndvi":0.30,"road_km":5.0,"grid_km":10.0,"wind_speed":10.0,
    "precipitation":2.0,"humidity":40.0,"land_score":0.50
}

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def basic_checks(df):
    print(df.shape)
    print(df.dtypes)
    print(df.isnull().sum())
    print("duplicates:", df.duplicated().sum())


def plot_score_distribution(df):
    plt.figure(figsize=(8,5))
    plt.hist(df[TARGET_COL], bins=20)
    plt.title("Score Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_score_dist.png")
    plt.show()


def plot_feature_distributions(df):
    cols = [c for c in FEATURE_COLS if c in df.columns]
    df[cols].hist(figsize=(12,8))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_features.png")
    plt.show()


def plot_correlation_heatmap(df):
    cols = [c for c in FEATURE_COLS if c in df.columns] + [TARGET_COL]
    corr = df[cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", vmin=-1, vmax=1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_corr.png")
    plt.show()


def plot_feature_vs_target(df):
    cols = ["ghi","slope","cloud_pct","ndvi","grid_km","land_score"]
    for c in cols:
        if c not in df.columns:
            continue
        plt.figure()
        plt.scatter(df[c], df[TARGET_COL])
        plt.xlabel(c)
        plt.ylabel(TARGET_COL)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"eda_{c}.png")
        plt.show()


def run_eda(df):
    plot_score_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_feature_vs_target(df)


def preprocess(df):
    df = df.copy()

    for col in FEATURE_COLS:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    out_cols = ["ghi","dni","temperature","elevation",
                "wind_speed","precipitation","humidity",
                "road_km","grid_km"]

    for col in out_cols:
        if col not in df:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
        df[col] = df[col].clip(low, high)

    if "land_use" in df.columns:
        m = {v:i for i,v in enumerate(sorted(df["land_use"].unique()))}
        df["land_use_code"] = df["land_use"].map(m)

    return df


def normalise(v, mn, mx):
    if mx <= mn:
        return 0.5
    return float(np.clip((v - mn)/(mx - mn), 0, 1))


def compute_score(row):
    ghi = normalise(row.get("ghi",4.5), GHI_MIN, GHI_MAX)

    slope = row.get("slope",5.0)
    slope_s = 0 if slope >= SLOPE_MAX else 1 - slope/SLOPE_MAX

    land = row.get("land_score",0.5)

    ndvi = np.clip(row.get("ndvi",0.3), NDVI_MIN, NDVI_MAX)
    ndvi_s = normalise(NDVI_MAX - ndvi, 0, NDVI_MAX - NDVI_MIN)

    cloud = normalise(100 - row.get("cloud_pct",30), 0, 100)

    grid = normalise(GRID_FAR - row.get("grid_km",10), 0, GRID_FAR - GRID_NEAR)
    road = normalise(ROAD_FAR - row.get("road_km",5), 0, ROAD_FAR - ROAD_NEAR)

    total = (
        WEIGHTS["ghi"]*ghi +
        WEIGHTS["slope"]*slope_s +
        WEIGHTS["land"]*land +
        WEIGHTS["ndvi"]*ndvi_s +
        WEIGHTS["cloud"]*cloud +
        WEIGHTS["grid"]*grid +
        WEIGHTS["road"]*road
    )

    score = total * 100

    if row.get("lat",20) >= 0:
        a = row.get("aspect",180)
        if 135 <= a <= 225:
            score = min(100, score * 1.05)

    return round(score, 2)


def get_rank(score):
    if score >= 80: return "Excellent"
    if score >= 65: return "Good"
    if score >= 50: return "Moderate"
    if score >= 35: return "Poor"
    return "Unsuitable"


def build_dataset(raw_rows):
    rows = []
    for r in raw_rows:
        d = dict(r)
        d[TARGET_COL] = compute_score(d)
        d["rank"] = get_rank(d[TARGET_COL])
        rows.append(d)

    df = pd.DataFrame(rows)

    for c, v in FEATURE_DEFAULTS.items():
        if c not in df.columns:
            df[c] = v

    return df