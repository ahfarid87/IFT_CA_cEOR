# Nano-cEOR Contact Angle (CA) & Interfacial Tension (IFT) Prediction + Optimization Platform

Rebuilt, production-oriented XGBoost pipelines for the two published nano-cEOR
ML frameworks, plus a shared Bayesian-optimization engine for formulation
screening, structured so they can plug directly into a Streamlit or
FastAPI/React online calculator (in the same spirit as the earlier
activated-carbon adsorption calculator).

```
EOR_Prediction_Project/
├── data/
│   ├── CA_Data.xlsx            # raw CA dataset (426 rows, 16 columns)
│   └── IFT_raw_data.xlsx       # raw IFT dataset (474 rows, 12 columns)
├── utils/                      # shared, reusable modules (no duplicate code)
│   ├── preprocessing.py        # TabularPreprocessor (log -> MinMax -> OHE), category maps
│   ├── model_utils.py          # tuning / training / metrics / plots / persistence
│   ├── optimization.py         # FormulationOptimizer + MultiObjectiveFormulationOptimizer (Optuna)
│   └── predict.py              # ModelBundle: load + predict from a saved model dir
├── CA/
│   ├── CA_Pipeline.ipynb       # full CA notebook (data -> EDA -> preprocessing -> XGBoost -> SHAP -> save -> optimize)
│   ├── models/                 # saved artifacts (created by running the notebook)
│   └── figures/                # saved publication-quality figures
├── IFT/
│   ├── IFT_Pipeline.ipynb      # full IFT notebook, mirrors CA notebook + multi-objective demo
│   ├── models/
│   └── figures/
├── deployment/
│   ├── app_streamlit.py        # reference Streamlit calculator (Prediction + Optimization modes)
│   └── api_fastapi.py          # reference FastAPI backend (same functionality, for a React front end)
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
cd CA  && jupyter notebook CA_Pipeline.ipynb     # run all cells
cd ../IFT && jupyter notebook IFT_Pipeline.ipynb  # run all cells (run CA first if you want the
                                                   # multi-objective Pareto-front section to work)
```

Both notebooks have already been run once end-to-end on your real data during
development to confirm the pipeline is bug-free; you should re-run them
yourself (especially the hyperparameter search, `mu.tune_xgboost(...,
n_iter=...)`) with a larger `n_iter` / more CV folds for your final,
publication-grade model — the shipped run used a reduced search budget so it
would execute quickly.

## What each dataset contains

**CA** (`Contact Angle (Degree)`, target): Sample State (S1-S5), Rock Type
(R1-R5), Porosity (%), Permeability (mD), Salinity (ppm), NPs (NP0-NP10),
NPs Size (nm), NPs Conc. (wt%), Chemical Additive (chem0-chem7), Additive
Conc. (wt%), Viscosity (cP), Oil Viscosity (cP), Oil SG, API, Temp. (°C).

**IFT** (`IFT (mN/m)`, target): Salinity (ppm), NPs (0-10, mapped to NP0-NP10
for consistency with the CA encoding), NPs Size (nm), NPs Conc. (wt%),
Chemical Additive (0-7, mapped to chem0-chem7), Additive Conc. (wt%),
Viscosity (cP), Oil Viscosity (cP), Oil SG, API, Temp.C. **No rock/porosity/
permeability columns** — IFT is a bulk fluid-fluid property in this dataset.

Category dictionaries (`NP_TYPE_MAP`, `CHEM_ADDITIVE_MAP`, `ROCK_TYPE_MAP`,
`SAMPLE_STATE_MAP`) live once in `utils/preprocessing.py` and are shared by
both notebooks and the deployment code, so labels never drift out of sync.

## Preprocessing (identical logic, driven by `PreprocessConfig`)

1. Basic cleaning (strip headers, drop empty/duplicate rows)
2. 1st/99th-percentile outlier trim on the target
3. Median imputation of missing continuous values
4. `ln(1+x)` log-transform of heavy-tailed features (permeability, salinity,
   NP size, viscosities, additive concentration as applicable)
5. Min-Max scaling of continuous features **and** the target to `[0, 1]`
6. One-hot encoding of categorical features

The fitted `TabularPreprocessor` is pickled alongside the model, so a single
`transform_single(input_dict)` call reproduces the exact training-time
transformation for a brand-new formulation submitted through the calculator.

## Model

XGBoost only (`xgboost.XGBRegressor`), tuned via `RandomizedSearchCV` over the
same hyperparameter ranges reported in the papers, refit with early stopping
against the validation split. Evaluated with R², RMSE, MAE, MAPE, residual
diagnostics, SHAP, PDPs, feature importance, learning curves, and 10-fold CV
stability — all reproduced as saved PNGs in `CA/figures/` and `IFT/figures/`.

## Optimization module (`utils/optimization.py`)

`FormulationOptimizer.minimize()` / `.maximize()` run **Bayesian optimization
(Optuna, TPE sampler)** over a mixed space of:

* `VariableSpec(name, "fixed", value=...)` — reservoir/operating conditions
  the engineer cannot change
* `VariableSpec(name, "bounded", low=..., high=..., step=...)` — continuous
  formulation variables to search
* `VariableSpec(name, "categorical", choices=[...])` — NP type / chemical
  additive / rock type choices

An optional `constraints(row_dict) -> bool` callback lets you encode rules
such as the NP:additive synergistic ratio window (1:1 to 1.5:1) reported in
the CA paper.

`MultiObjectiveFormulationOptimizer` runs NSGA-II across **two** trained
models at once (e.g. CA + IFT) and returns the non-dominated Pareto front of
formulations — used for "minimize CA and minimize IFT simultaneously."

Why Optuna over GA / Differential Evolution / plain Bayesian-Optimization:
it natively handles mixed continuous/categorical/fixed spaces in one API,
has first-class multi-objective (NSGA-II) support, returns a full trial
history for convergence plots "for free," and is fast enough (a few hundred
trials against a trained tree ensemble = sub-second) to re-run on every user
click inside a web calculator.

## Prediction function

```python
from utils.predict import ModelBundle
bundle = ModelBundle("CA/models", "ca")
bundle.predict({
    "Sample State": "S3", "Rock Type": "R1", "Porosity (%)": 18,
    "Permeability (mD)": 50, "Salinity (ppm)": 30000, "NPs": "NP6",
    "NPs Size (nm)": 20, "NPs Conc. (wt%)": 0.3, "Chemical Additive": "chem4",
    "Additive Conc. (wt%)": 0.1, "Viscosity (cP)": 1.0, "Oil Viscosity (cP)": 20,
    "Oil SG": 0.86, "API": 33, "Temp.": 50,
})
# -> {"prediction": ..., "ci_95_lower": ..., "ci_95_upper": ..., "validation_rmse": ...}
```

Same pattern for IFT via `ModelBundle("IFT/models", "ift")`.

## Deploying the online calculator

Two ready-to-adapt entry points are provided in `deployment/`:

* **`app_streamlit.py`** — a single-file Streamlit app with a sidebar for
  Property (CA / IFT / Both) and Mode (Prediction / Optimization), mirroring
  the earlier activated-carbon adsorption calculator's structure. Deploy on
  Streamlit Community Cloud by pointing it at this file (make sure `utils/`,
  `CA/models/`, and `IFT/models/` are included in the deployed repo).
* **`api_fastapi.py`** — the same functionality behind a JSON API
  (`/predict/ca`, `/predict/ift`, `/optimize/ca`, `/optimize/ift`,
  `/optimize/multi`), for a React (or any JS) front end. Run with
  `uvicorn deployment.api_fastapi:app --reload`.

## Suggested future improvements

1. **Uncertainty quantification** — replace the Gaussian-residual 95% CI in
   `ModelBundle.predict` with quantile-regression XGBoost (`objective:
   "reg:quantileerror"`) or a conformal-prediction wrapper for calibrated
   intervals.
2. **Batch prediction** — add a `predict_batch(df)` method to `ModelBundle`
   for CSV upload/screening of many candidate formulations at once.
3. **Explainable AI dashboard** — embed the SHAP force/waterfall plot for a
   single prediction directly in the Streamlit app so users see *why* a
   given formulation scored the way it did.
4. **Group-aware CV in production retraining** — wire a "Publication /
   Source" column into `groups=` in `mu.tune_xgboost(...)` once available,
   to guard against literature-source leakage exactly as the CA paper does.
5. **Physics-informed constraints** — extend the optimizer's `constraints`
   callback with the paper's operational thresholds (permeability > 0.1 mD,
   salinity 30,000-80,000 ppm optimal window) as selectable presets in the
   calculator UI.
