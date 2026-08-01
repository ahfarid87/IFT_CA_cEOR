"""
preprocessing.py
=================
Reusable, production-quality preprocessing pipeline shared by the Contact
Angle (CA) and Interfacial Tension (IFT) XGBoost models.

The design follows the methodology reported in:
  - Kandiel, Mahmoud & Ibrahim (2026) "A robust machine learning framework
    for predicting contact angle in nano-assisted chemical EOR",
    Scientific Reports 16:14676.
  - Kandiel, Mahmoud & Ibrahim, "Machine Learning-Based Interfacial Tension
    Prediction for Nanoparticles-Assisted Enhanced Oil Recovery" (manuscript).

Both papers use:
  * log-transform of heavy-tailed / log-normal features
    (ln(1+x) so that zeros are handled safely)
  * Min-Max normalization of continuous features (and, for CA, of the target)
  * One-hot encoding (OHE) of categorical features (NP type, chemical
    additive, rock type, sample state)
  * median imputation for missing continuous values
  * percentile-based outlier trimming (1st / 99th percentile)

This module implements a single reusable `TabularPreprocessor` class that
can be configured per-dataset (CA vs IFT) and is fully serializable with
`joblib`, so that the exact same transformation can be replayed at
inference time inside a web calculator / FastAPI backend.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# --------------------------------------------------------------------------- #
# Category dictionaries (from the published papers)                          #
# --------------------------------------------------------------------------- #

# NPs Type encoding used identically in both the CA and IFT papers.
NP_TYPE_MAP = {
    "NP0": "None",
    "NP1": "SiO2",
    "NP2": "Al2O3",
    "NP3": "TiO2",
    "NP4": "Fe3O4",
    "NP5": "NiO",
    "NP6": "ZrO2",
    "NP7": "CuO",
    "NP8": "CN",
    "NP9": "MgO",
    "NP10": "ZnO",
}

# Chemical additive encoding used identically in both papers.
CHEM_ADDITIVE_MAP = {
    "chem0": "None",
    "chem1": "Anionic Surfactant",
    "chem2": "Cationic Surfactant",
    "chem3": "Nonionic Surfactant",
    "chem4": "Biosurfactant",
    "chem5": "Alcohols / Solvents",
    "chem6": "Anionic Polymer",
    "chem7": "Nonionic Polymer",
}

# Rock type encoding (CA model only).
ROCK_TYPE_MAP = {
    "R1": "Limestone",
    "R2": "Sandstone",
    "R3": "Glass",
    "R4": "Dolomite",
    "R5": "Carbonate",
}

# Sample / aging state encoding (CA model only).
# Confirmed against the paper's text ("oil-aged samples (S1)... high-salinity
# nanofluid-aged samples (S3)... low-salinity (S4)") and the full mapping
# supplied by the user.
SAMPLE_STATE_MAP = {
    "S1": "Oil Aged",
    "S2": "Nano Aged",
    "S3": "HS Nano Aged",
    "S4": "LS Nano Aged",
    "S5": "Surfactant Aged",
}


def enforce_domain_rules(input_dict: dict) -> dict:
    """
    Enforce physically-required consistency between a categorical "type"
    field and its associated continuous "amount" field(s):

      * NPs == "NP0" ("None")     -> NPs Size (nm) = 0, NPs Conc. (wt%) = 0
      * Chemical Additive == "chem0" ("None") -> Additive Conc. (wt%) = 0

    This is applied at every inference entry point (TabularPreprocessor.
    transform_single, and therefore also every optimizer trial that calls
    it) so a formulation can never be scored as "0.3 wt% of no nanoparticle."
    Training data is left untouched - this only affects inference-time rows.
    """
    row = dict(input_dict)
    if row.get("NPs") == "NP0":
        row["NPs Size (nm)"] = 0.0
        row["NPs Conc. (wt%)"] = 0.0
    if row.get("Chemical Additive") == "chem0":
        row["Additive Conc. (wt%)"] = 0.0
    return row


@dataclass
class PreprocessConfig:
    """Configuration describing how a given dataset should be preprocessed."""

    target_col: str
    log_features: list = field(default_factory=list)     # ln(1+x) transform
    continuous_features: list = field(default_factory=list)
    categorical_features: list = field(default_factory=list)
    scale_target: bool = False           # Min-Max scale the target too (CA paper does this)
    outlier_trim: bool = True
    outlier_low_pct: float = 1.0
    outlier_high_pct: float = 99.0
    group_col: Optional[str] = None      # e.g. "Publication" for group-based CV


class TabularPreprocessor:
    """
    A single, serializable preprocessing object that:
      1. Cleans and validates raw input data
      2. Trims outliers (percentile based) - fit time only
      3. Imputes missing continuous values (median)
      4. Log-transforms skewed features -> ln(1+x)
      5. Min-Max scales continuous features (and optionally the target)
      6. One-hot encodes categorical features

    Call `fit_transform` once on training data, then `transform` on any new
    data (validation/test/inference) to guarantee identical treatment.
    """

    def __init__(self, config: PreprocessConfig):
        self.cfg = config
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler() if config.scale_target else None
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names_out_: list = []
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Cleaning                                                            #
    # ------------------------------------------------------------------ #
    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        # drop fully empty rows / duplicate rows
        df = df.dropna(how="all")
        df = df.drop_duplicates()
        return df

    def _trim_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Percentile-based trimming on the target column (fit-time only)."""
        if not self.cfg.outlier_trim or self.cfg.target_col not in df.columns:
            return df
        lo = np.percentile(df[self.cfg.target_col].dropna(), self.cfg.outlier_low_pct)
        hi = np.percentile(df[self.cfg.target_col].dropna(), self.cfg.outlier_high_pct)
        mask = df[self.cfg.target_col].between(lo, hi)
        return df.loc[mask].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Fit / transform                                                     #
    # ------------------------------------------------------------------ #
    def fit(self, df_raw: pd.DataFrame):
        df = self._basic_clean(df_raw)
        df = self._trim_outliers(df)

        cont = self.cfg.continuous_features
        cat = self.cfg.categorical_features
        logf = self.cfg.log_features

        X_cont = df[cont].copy()
        # median impute BEFORE log transform (robust to right-skew)
        X_cont[:] = self.imputer.fit_transform(X_cont)
        # log-transform designated skewed features
        for c in logf:
            X_cont[c] = np.log1p(X_cont[c].clip(lower=0))
        # scale all continuous features to [0, 1]
        X_cont_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_cont), columns=cont, index=df.index
        )

        X_cat = df[cat].astype(str)
        X_cat_ohe = pd.DataFrame(
            self.ohe.fit_transform(X_cat),
            columns=self.ohe.get_feature_names_out(cat),
            index=df.index,
        )

        y = df[self.cfg.target_col].astype(float)
        if self.target_scaler is not None:
            y_arr = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
            y = pd.Series(y_arr, index=df.index, name=self.cfg.target_col)

        X = pd.concat([X_cont_scaled, X_cat_ohe], axis=1)
        self.feature_names_out_ = list(X.columns)
        self._fitted = True

        extra = {}
        if self.cfg.group_col and self.cfg.group_col in df.columns:
            extra["groups"] = df[self.cfg.group_col].values

        return X, y, extra

    def fit_transform(self, df_raw: pd.DataFrame):
        return self.fit(df_raw)

    def transform(self, df_raw: pd.DataFrame, has_target: bool = True):
        """Apply an already-fitted preprocessor to new data."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit() before transform().")
        df = self._basic_clean(df_raw)

        cont = self.cfg.continuous_features
        cat = self.cfg.categorical_features
        logf = self.cfg.log_features

        X_cont = df[cont].copy()
        X_cont[:] = self.imputer.transform(X_cont)
        for c in logf:
            X_cont[c] = np.log1p(X_cont[c].clip(lower=0))
        X_cont_scaled = pd.DataFrame(
            self.scaler.transform(X_cont), columns=cont, index=df.index
        )

        X_cat = df[cat].astype(str)
        X_cat_ohe = pd.DataFrame(
            self.ohe.transform(X_cat),
            columns=self.ohe.get_feature_names_out(cat),
            index=df.index,
        )

        X = pd.concat([X_cont_scaled, X_cat_ohe], axis=1)
        X = X.reindex(columns=self.feature_names_out_, fill_value=0.0)

        if has_target and self.cfg.target_col in df.columns:
            y = df[self.cfg.target_col].astype(float)
            if self.target_scaler is not None:
                y_arr = self.target_scaler.transform(y.values.reshape(-1, 1)).ravel()
                y = pd.Series(y_arr, index=df.index, name=self.cfg.target_col)
            return X, y
        return X

    # ------------------------------------------------------------------ #
    # Single-record inference helper (for the web calculator)             #
    # ------------------------------------------------------------------ #
    def transform_single(self, input_dict: dict) -> pd.DataFrame:
        """
        Build a one-row DataFrame from a dict of raw feature values (as a
        user would submit them from a web form) and run it through the
        fitted transform pipeline. Missing keys are treated as NaN and
        will be median-imputed.
        """
        input_dict = enforce_domain_rules(input_dict)
        cont = self.cfg.continuous_features
        cat = self.cfg.categorical_features
        row = {}
        for c in cont:
            row[c] = input_dict.get(c, np.nan)
        for c in cat:
            row[c] = input_dict.get(c, "None")
        df_row = pd.DataFrame([row])
        return self.transform(df_row, has_target=False)

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.target_scaler is None:
            return np.asarray(y_scaled)
        return self.target_scaler.inverse_transform(
            np.asarray(y_scaled).reshape(-1, 1)
        ).ravel()

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #
    def save(self, out_dir: str, prefix: str):
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self, os.path.join(out_dir, f"{prefix}_preprocessor.pkl"))
        meta = {
            "target_col": self.cfg.target_col,
            "log_features": self.cfg.log_features,
            "continuous_features": self.cfg.continuous_features,
            "categorical_features": self.cfg.categorical_features,
            "scale_target": self.cfg.scale_target,
            "feature_names_out": self.feature_names_out_,
        }
        with open(os.path.join(out_dir, f"{prefix}_feature_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(out_dir: str, prefix: str) -> "TabularPreprocessor":
        return joblib.load(os.path.join(out_dir, f"{prefix}_preprocessor.pkl"))
