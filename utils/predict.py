"""
predict.py
==========
Generic "load trained bundle -> predict from a dict of raw inputs" helper,
reused by both CA and IFT prediction functions and by the future
FastAPI/Streamlit backend. This is the single source of truth for how a
saved model bundle (xgb_*.pkl + *_preprocessor.pkl) is loaded and queried,
so the notebook, the optimizer, and the web API can never drift apart.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np

from .preprocessing import TabularPreprocessor  # noqa: F401 (needed for joblib unpickle)


class ModelBundle:
    """Loads a trained XGBoost model + its preprocessor from a directory."""

    def __init__(self, model_dir: str, prefix: str):
        self.model_dir = model_dir
        self.prefix = prefix
        self.model = joblib.load(os.path.join(model_dir, f"xgb_{prefix}.pkl"))
        self.preprocessor = TabularPreprocessor.load(model_dir, prefix)
        with open(os.path.join(model_dir, f"{prefix}_metrics.json")) as f:
            self.metrics = json.load(f)
        with open(os.path.join(model_dir, f"{prefix}_feature_metadata.json")) as f:
            self.feature_metadata = json.load(f)

    def predict(self, input_dict: dict) -> dict:
        """
        Predict from a dict of raw (un-transformed) feature values, e.g.:
            {"Rock Type": "R1", "Porosity (%)": 20, "Permeability (mD)": 50,
             "Salinity (ppm)": 30000, "NPs": "NP6", "NPs Size (nm)": 20,
             "NPs Conc. (wt%)": 0.3, "Chemical Additive": "chem4",
             "Additive Conc. (wt%)": 0.1, "Viscosity (cP)": 1.0,
             "Oil Viscosity (cP)": 20, "Oil SG": 0.86, "API": 33,
             "Temp.": 50, "Sample State": "S3"}

        Returns predicted value plus an approximate confidence interval
        based on the validation-set RMSE (assuming ~Gaussian residuals):
        pred +/- 1.96 * RMSE  gives an approximate 95% interval.
        """
        X = self.preprocessor.transform_single(input_dict)
        y_scaled = self.model.predict(X)
        y = self.preprocessor.inverse_transform_target(y_scaled)
        pred = float(y[0])

        rmse = self.metrics.get("Validation", {}).get("RMSE") or self.metrics.get("Testing", {}).get("RMSE")
        ci95 = 1.96 * rmse if rmse else None

        return {
            "prediction": pred,
            "ci_95_lower": pred - ci95 if ci95 is not None else None,
            "ci_95_upper": pred + ci95 if ci95 is not None else None,
            "validation_rmse": rmse,
        }
