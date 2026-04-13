import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from utils.feature_engineering import FEATURE_COLS, TARGET_COL

warnings.filterwarnings("ignore")


NUMERIC_COLS = [
    "ghi","dni","temperature","cloud_pct","clearness",
    "elevation","slope","aspect","ndvi",
    "road_km","grid_km","wind_speed","precipitation","humidity"
]

CATEGORICAL_COLS = ["land_score"]


def build_preprocessor():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("num", num_pipe, FEATURE_COLS)
    ])


def build_pipeline(model):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model)
    ])


def split_data(df, test_size=0.2, random_state=42):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def train_pipeline(name, pipeline, X_train, y_train, X_test, y_test, cv=5):
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    t = round(time.time() - t0, 2)

    y_pred = np.clip(pipeline.predict(X_test), 0, 100)

    r2 = round(r2_score(y_test, y_pred), 4)
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")

    return {
        "Model": name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "CV_R2_Mean": round(cv_scores.mean(), 4),
        "CV_R2_Std": round(cv_scores.std(), 4),
        "Train_Time_sec": t,
        "pipeline_obj": pipeline,
        "y_pred": y_pred,
    }


def train_all_pipelines(X_train, X_test, y_train, y_test):
    models = [
        ("Ridge", Ridge(alpha=1.0)),
        ("Lasso", Lasso(alpha=0.1, max_iter=5000)),
        ("SVR", SVR(kernel="rbf", C=10, epsilon=0.1)),
        ("RF", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("GB", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("XGB", XGBRegressor(n_estimators=100, random_state=42, tree_method="hist", verbosity=0)),
    ]

    results = []
    for name, model in models:
        pipe = build_pipeline(model)
        results.append(train_pipeline(name, pipe, X_train, y_train, X_test, y_test))

    return results


def tune_pipeline_grid(name, model, param_grid, X_train, y_train, cv=5, scoring="r2"):
    pipe = build_pipeline(model)

    search = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    search.fit(X_train, y_train)

    return {
        "best_params": search.best_params_,
        "best_cv_score": round(search.best_score_, 4),
        "best_pipeline": search.best_estimator_,
        "search_obj": search,
    }


def tune_pipeline_random(name, model, param_dist, X_train, y_train,
                         n_iter=30, cv=5, scoring="r2", random_state=42):
    pipe = build_pipeline(model)

    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=n_iter, cv=cv,
        scoring=scoring, n_jobs=-1, random_state=random_state
    )
    search.fit(X_train, y_train)

    return {
        "best_params": search.best_params_,
        "best_cv_score": round(search.best_score_, 4),
        "best_pipeline": search.best_estimator_,
        "search_obj": search,
    }


def get_best_pipeline(results, metric="R2"):
    if metric in ("MAE", "RMSE"):
        return min(results, key=lambda x: x[metric])
    return max(results, key=lambda x: x[metric])


def save_pipeline(pipeline, metrics, save_dir="models"):
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, p / "best_pipeline.pkl")

    summary = {
        "model_name": metrics.get("Model"),
        "R2": metrics.get("R2"),
        "MAE": metrics.get("MAE"),
        "RMSE": metrics.get("RMSE"),
        "CV_R2_Mean": metrics.get("CV_R2_Mean"),
        "CV_R2_Std": metrics.get("CV_R2_Std"),
        "feature_cols": FEATURE_COLS
    }

    with open(p / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return p / "best_pipeline.pkl"


def load_pipeline(save_dir="models"):
    p = Path(save_dir)

    pipe = joblib.load(p / "best_pipeline.pkl")

    summary = {}
    if (p / "model_summary.json").exists():
        with open(p / "model_summary.json") as f:
            summary = json.load(f)

    return pipe, summary


def predict_location(pipeline, feature_dict):
    df = pd.DataFrame([{c: feature_dict.get(c, 0.0) for c in FEATURE_COLS}])
    pred = pipeline.predict(df)[0]
    return round(float(np.clip(pred, 0, 100)), 2)