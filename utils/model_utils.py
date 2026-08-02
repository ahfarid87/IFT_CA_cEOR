"""
model_utils.py
===============
Reusable training / evaluation / visualization / persistence helpers for the
XGBoost CA and IFT regressors. Kept dataset-agnostic so both notebooks import
identical, tested code (DRY - "Avoid duplicate code" per project spec).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    mape_mask = y_true != 0
    # RMSE computed in log1p-space (on non-negative values only) - used to
    # build a multiplicative/asymmetric confidence interval that can never
    # go below 0, regardless of whether the model itself was log-trained.
    y_true_nn = np.clip(y_true, 0, None)
    y_pred_nn = np.clip(y_pred, 0, None)
    log_rmse = float(np.sqrt(mean_squared_error(np.log1p(y_true_nn), np.log1p(y_pred_nn))))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(
            mean_absolute_percentage_error(y_true[mape_mask], y_pred[mape_mask]) * 100
        ) if mape_mask.any() else float("nan"),
        "Log_RMSE": log_rmse,
        "Residual_Std": float(np.std(resid)),
        "Residual_Mean": float(np.mean(resid)),
        "Residual_Min": float(np.min(resid)),
        "Residual_Max": float(np.max(resid)),
        "N": int(len(y_true)),
    }


def print_metrics_table(metrics_by_split: dict, title: str = "Performance"):
    df = pd.DataFrame(metrics_by_split).T[
        ["RMSE", "MAE", "MAPE", "R2", "Residual_Std", "Residual_Mean"]
    ]
    print(f"\n=== {title} ===")
    print(df.round(4).to_string())
    return df


# --------------------------------------------------------------------------- #
# Hyperparameter search space (aligned with the published papers' ranges)     #
# --------------------------------------------------------------------------- #
XGB_PARAM_DISTRIBUTIONS = {
    "n_estimators": [50, 100, 150, 200, 300, 400],
    "max_depth": [3, 5, 7, 9, 11, 13, 15],
    "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.05, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.001, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
    "min_child_weight": [1, 3, 5, 7],
}


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: Optional[np.ndarray] = None,
    n_iter: int = 60,
    n_splits: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> xgb.XGBRegressor:
    """
    Hyperparameter tuning via RandomizedSearchCV with (group-aware) K-Fold CV,
    mirroring the tenfold nested cross-validation reported in the papers.
    Using a randomized search over the same style of grid keeps runtime
    tractable while covering the full space reported in the manuscripts.
    """
    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
    )

    if groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        cv_splits = list(cv.split(X_train, y_train, groups=groups))
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_splits = list(cv.split(X_train, y_train))

    search = RandomizedSearchCV(
        base,
        param_distributions=XGB_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=cv_splits,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print("Best CV MSE:", -search.best_score_)
    print("Best params:", search.best_params_)
    return search.best_estimator_


def train_final_xgboost(
    X_train, y_train, X_val, y_val, best_params: dict, random_state: int = 42
) -> xgb.XGBRegressor:
    """Refit the tuned model with early stopping against the validation set."""
    params = dict(best_params)
    params.update(
        objective="reg:squarederror",
        random_state=random_state,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=30,
    )
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    return model


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #
def plot_actual_vs_predicted(y_true, y_pred, title, save_path=None, unit=""):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "k--", lw=1.5, label="Ideal fit")
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
    ax.set_xlabel(f"Actual {unit}")
    ax.set_ylabel(f"Predicted {unit}")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.show()


def plot_residuals(y_true, y_pred, title, save_path=None, unit=""):
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_pred, resid, alpha=0.6, edgecolor="k", linewidth=0.3)
    axes[0].axhline(0, color="red", ls="--")
    axes[0].set_xlabel(f"Predicted {unit}")
    axes[0].set_ylabel(f"Residual {unit}")
    axes[0].set_title(f"{title}: Residuals vs Predicted")

    axes[1].hist(resid, bins=25, edgecolor="k", alpha=0.75)
    axes[1].axvline(0, color="red", ls="--")
    axes[1].set_xlabel(f"Residual {unit}")
    axes[1].set_title(f"{title}: Residual Distribution")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.show()


def plot_learning_curve(model: xgb.XGBRegressor, title, save_path=None):
    results = model.evals_result()
    if not results:
        return
    keys = list(results.keys())
    metric = list(results[keys[0]].keys())[0]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, k in enumerate(keys):
        label = "Train" if i == 0 else "Validation"
        ax.plot(results[k][metric], label=label)
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{title}: Learning Curve")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.show()


def plot_feature_importance(model: xgb.XGBRegressor, feature_names, title, save_path=None, top_n=20):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(np.array(feature_names)[order][::-1], importances[order][::-1])
    ax.set_xlabel("XGBoost Feature Importance (gain-normalized)")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.show()


def plot_shap_summary(model: xgb.XGBRegressor, X: pd.DataFrame, title, save_path=None, max_display=20):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values, X, show=False, max_display=max_display)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    return shap_values


def plot_pdp_1d(model, X, feature, title, save_path=None, unit=""):
    from sklearn.inspection import PartialDependenceDisplay
    fig, ax = plt.subplots(figsize=(6, 4.5))
    PartialDependenceDisplay.from_estimator(model, X, [feature], ax=ax)
    ax.set_title(f"{title}: PDP - {feature}")
    ax.set_ylabel(f"Predicted {unit}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    plt.show()


# --------------------------------------------------------------------------- #
# Persistence                                                                 #
# --------------------------------------------------------------------------- #
def save_model_bundle(
    model: xgb.XGBRegressor,
    preprocessor,
    out_dir: str,
    prefix: str,
    training_config: dict,
    metrics: dict,
):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, f"xgb_{prefix}.pkl"))
    preprocessor.save(out_dir, prefix)

    with open(os.path.join(out_dir, f"{prefix}_training_configuration.json"), "w") as f:
        json.dump(training_config, f, indent=2, default=str)

    with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(out_dir, f"{prefix}_feature_list.json"), "w") as f:
        json.dump(list(preprocessor.feature_names_out_), f, indent=2)

    print(f"Saved model bundle to: {out_dir}")
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith(prefix) or fn.startswith(f"xgb_{prefix}"):
            print("  -", fn)
