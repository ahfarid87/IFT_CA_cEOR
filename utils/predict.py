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

    # Hard floor applied to any negative value that slips through, regardless
    # of cause (e.g. a stale pickled model saved before physical_min/
    # physical_max existed on PreprocessConfig). Neither IFT nor Contact
    # Angle can physically be negative.
    NEGATIVE_FLOOR = 0.01

    def predict(self, input_dict: dict) -> dict:
        """
        Predict from a dict of raw (un-transformed) feature values, e.g.:
            {"Rock Type": "R1", "Porosity (%)": 20, "Permeability (mD)": 50,
             "Salinity (ppm)": 30000, "NPs": "NP6", "NPs Size (nm)": 20,
             "NPs Conc. (wt%)": 0.3, "Chemical Additive": "chem4",
             "Additive Conc. (wt%)": 0.1, "Viscosity (cP)": 1.0,
             "Oil Viscosity (cP)": 20, "Oil SG": 0.86, "API": 33,
             "Temp.": 50, "Sample State": "S3"}

        Returns the predicted value plus an approximate 95% confidence
        interval built in log1p-space (using the validation-set Log_RMSE)
        and then inverted back to real units. The preprocessor already
        clips to [physical_min, physical_max], and as a final safety net
        below, any value that is still negative (e.g. from an older saved
        model whose config predates physical_min/physical_max) is floored
        at NEGATIVE_FLOOR (0.01) rather than shown as a physically
        impossible negative IFT/Contact Angle.
        """
        X = self.preprocessor.transform_single(input_dict)
        y_scaled = self.model.predict(X)
        y = self.preprocessor.inverse_transform_target(y_scaled)
        pred = float(y[0])   # normally already clipped to [physical_min, physical_max]

        # getattr(..., default) guards against an older pickled preprocessor
        # whose PreprocessConfig predates these fields.
        phys_min = getattr(self.preprocessor.cfg, "physical_min", 0.0) or 0.0
        phys_max = getattr(self.preprocessor.cfg, "physical_max", None)

        split_metrics = self.metrics.get("Validation") or self.metrics.get("Testing") or {}
        rmse = split_metrics.get("RMSE")
        log_rmse = split_metrics.get("Log_RMSE")

        ci_lower = ci_upper = None
        if log_rmse is not None:
            log_pred = np.log1p(max(pred - phys_min, 0.0))
            ci_lower = float(np.expm1(max(log_pred - 1.96 * log_rmse, -50.0))) + phys_min
            ci_upper = float(np.expm1(log_pred + 1.96 * log_rmse)) + phys_min
            ci_lower = max(phys_min, ci_lower)
            if phys_max is not None:
                ci_upper = min(phys_max, ci_upper)
                ci_lower = min(ci_lower, ci_upper)

        def _floor_negative(x):
            if x is None:
                return None
            return self.NEGATIVE_FLOOR if x < 0 else x

        pred = _floor_negative(pred)
        ci_lower = _floor_negative(ci_lower)
        ci_upper = _floor_negative(ci_upper)
	pred = max(0.01, pred)

	if ci_lower is not None:
    		ci_lower = max(0.01, ci_lower)

	if ci_upper is not None:
    		ci_upper = max(0.01, ci_upper)
        return {
            "prediction": pred,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "validation_rmse": rmse,
        }
