import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import learning_curve
from utils.feature_engineering import FEATURE_COLS

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_metrics(result):
    print(result["Model"])
    print("R2:", result["R2"], "MAE:", result["MAE"], "RMSE:", result["RMSE"])
    print("CV:", result["CV_R2_Mean"], "±", result["CV_R2_Std"])
    print("Time:", result["Train_Time_sec"])


def plot_actual_vs_predicted(results_list, y_test,
                              title="Actual vs Predicted",
                              save_as="step7_actual_vs_pred.png"):

    n = len(results_list)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, r in zip(axes, results_list):
        y_pred = r["y_pred"]
        err = np.abs(np.array(y_test) - y_pred)

        sc = ax.scatter(y_test, y_pred, c=err, cmap="RdYlGn_r",
                        alpha=0.7, s=60, edgecolors="white")

        lo = min(float(y_test.min()), float(y_pred.min())) - 2
        hi = max(float(y_test.max()), float(y_pred.max())) + 2
        ax.plot([lo, hi], [lo, hi], "r--")

        ax.set_title(r["Model"])
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / save_as, dpi=120)
    plt.show()


def plot_residuals(results_list, y_test,
                   title="Residuals",
                   save_as="step7_residuals.png"):

    n = len(results_list)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, r in zip(axes, results_list):
        res = np.array(y_test) - r["y_pred"]
        ax.scatter(r["y_pred"], res, alpha=0.7)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_title(r["Model"])

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / save_as, dpi=120)
    plt.show()


def plot_learning_curve(pipeline, X_train, y_train,
                         model_name="Pipeline", cv=5,
                         save_as="step7_learning_curve.png"):

    sizes, train_s, val_s = learning_curve(
        pipeline, X_train, y_train,
        cv=cv, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sizes, train_s.mean(axis=1), label="train")
    ax.plot(sizes, val_s.mean(axis=1), label="val")

    ax.legend()
    ax.set_title(model_name)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / save_as, dpi=120)
    plt.show()


def print_comparison_table(results_list, sort_by="R2"):
    rows = []
    for r in results_list:
        rows.append({
            "Model": r["Model"],
            "R2": r["R2"],
            "MAE": r["MAE"],
            "RMSE": r["RMSE"],
            "CV_R2_Mean": r["CV_R2_Mean"],
            "CV_R2_Std": r["CV_R2_Std"],
            "Train_Time_sec": r["Train_Time_sec"],
        })

    df = pd.DataFrame(rows)
    asc = sort_by in ("MAE", "RMSE")
    df = df.sort_values(sort_by, ascending=asc)

    print(df)
    return df


def plot_model_comparison(results_list,
                           title="Comparison",
                           save_as="step8_pipeline_comparison.png"):

    names = [r["Model"] for r in results_list]
    r2 = [r["R2"] for r in results_list]
    mae = [r["MAE"] for r in results_list]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, r2)
    ax.set_title("R2 Comparison")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / save_as, dpi=120)
    plt.show()


def plot_feature_importance(pipeline, model_name="Pipeline",
                             save_as="step8_feature_importance.png"):

    if "model" not in pipeline.named_steps:
        return

    m = pipeline.named_steps["model"]
    if not hasattr(m, "feature_importances_"):
        return

    imp = pd.Series(m.feature_importances_, index=FEATURE_COLS)
    imp = imp.sort_values()

    imp.plot(kind="barh")
    plt.title(model_name)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / save_as, dpi=120)
    plt.show()