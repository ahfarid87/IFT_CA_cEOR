"""
api_fastapi.py
===============
Minimal FastAPI backend exposing the CA / IFT models for prediction and
optimization, suitable as the backend for a React (or any JS) front-end
calculator. Run with:

    uvicorn deployment.api_fastapi:app --reload --port 8000

Endpoints:
    POST /predict/ca            -> {"prediction": ..., "ci_95_lower": ..., "ci_95_upper": ...}
    POST /predict/ift           -> same shape
    POST /optimize/ca           -> single-objective optimization on the CA model
    POST /optimize/ift          -> single-objective optimization on the IFT model
    POST /optimize/multi        -> multi-objective (CA + IFT) Pareto front
"""

import os
import sys
from typing import Dict, List, Literal, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.predict import ModelBundle
from utils.optimization import FormulationOptimizer, MultiObjectiveFormulationOptimizer, VariableSpec

CA_DIR = os.path.join(os.path.dirname(__file__), "..", "CA", "models")
IFT_DIR = os.path.join(os.path.dirname(__file__), "..", "IFT", "models")

app = FastAPI(title="Nano-cEOR CA/IFT Prediction & Optimization API")

_ca_bundle: Optional[ModelBundle] = None
_ift_bundle: Optional[ModelBundle] = None


def get_ca_bundle() -> ModelBundle:
    global _ca_bundle
    if _ca_bundle is None:
        _ca_bundle = ModelBundle(CA_DIR, "ca")
    return _ca_bundle


def get_ift_bundle() -> ModelBundle:
    global _ift_bundle
    if _ift_bundle is None:
        _ift_bundle = ModelBundle(IFT_DIR, "ift")
    return _ift_bundle


class PredictRequest(BaseModel):
    inputs: Dict[str, Union[float, str]]


class VariableSpecIn(BaseModel):
    name: str
    kind: Literal["fixed", "bounded", "categorical"]
    value: Optional[Union[float, str]] = None
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[str]] = None


class OptimizeRequest(BaseModel):
    variables: List[VariableSpecIn]
    direction: Literal["minimize", "maximize"] = "minimize"
    n_trials: int = 200


class MultiOptimizeRequest(BaseModel):
    variables: List[VariableSpecIn]
    n_trials: int = 300


def _to_variable_specs(vs: List[VariableSpecIn]) -> List[VariableSpec]:
    return [VariableSpec(**v.dict()) for v in vs]


@app.post("/predict/ca")
def predict_ca(req: PredictRequest):
    return get_ca_bundle().predict(req.inputs)


@app.post("/predict/ift")
def predict_ift(req: PredictRequest):
    return get_ift_bundle().predict(req.inputs)


@app.post("/optimize/ca")
def optimize_ca(req: OptimizeRequest):
    bundle = get_ca_bundle()
    opt = FormulationOptimizer(bundle.model, bundle.preprocessor,
                                bundle.feature_metadata["continuous_features"],
                                bundle.feature_metadata["categorical_features"])
    result = opt.optimize(_to_variable_specs(req.variables), direction=req.direction, n_trials=req.n_trials)
    return {"best_formulation": result["best_formulation"], "best_value": result["best_value"]}


@app.post("/optimize/ift")
def optimize_ift(req: OptimizeRequest):
    bundle = get_ift_bundle()
    opt = FormulationOptimizer(bundle.model, bundle.preprocessor,
                                bundle.feature_metadata["continuous_features"],
                                bundle.feature_metadata["categorical_features"])
    result = opt.optimize(_to_variable_specs(req.variables), direction=req.direction, n_trials=req.n_trials)
    return {"best_formulation": result["best_formulation"], "best_value": result["best_value"]}


@app.post("/optimize/multi")
def optimize_multi(req: MultiOptimizeRequest):
    ca_bundle, ift_bundle = get_ca_bundle(), get_ift_bundle()
    mo = MultiObjectiveFormulationOptimizer(
        ca_bundle.model, ca_bundle.preprocessor, "minimize",
        ift_bundle.model, ift_bundle.preprocessor, "minimize",
    )
    result = mo.optimize(_to_variable_specs(req.variables), n_trials=req.n_trials)
    return {"pareto_front": result["pareto_front"].to_dict(orient="records")}


@app.get("/health")
def health():
    return {"status": "ok"}
