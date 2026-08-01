"""
optimization.py
================
A general-purpose formulation optimizer built on top of a trained XGBoost
model + its TabularPreprocessor. Supports:

  * fixed variables      (reservoir/operating conditions the engineer can't change)
  * bounded continuous variables (e.g. NP concentration between 0.05 and 1.0 wt%)
  * categorical variables (e.g. NP type, chemical additive, rock type)
  * single-objective minimization or maximization
  * multi-objective optimization across TWO trained models at once
    (e.g. simultaneously minimize Contact Angle AND minimize IFT)

Bayesian Optimization (Optuna, TPE sampler) is used as the search engine.
Optuna is preferred here over Bayesian-Optimization/GA/Differential Evolution
because it natively supports **mixed integer / categorical / continuous**
search spaces in one unified `Trial` API, has built-in multi-objective
(NSGA-II) support, gives us a convergence/optimization history for free,
and is lightweight enough to embed inside a Streamlit/FastAPI backend
for the online calculator, where re-optimizing on every user click needs
to be fast (a few hundred trials against a trained tree ensemble is
sub-second).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class VariableSpec:
    """Describes one input variable's role in the optimization problem."""

    name: str
    kind: str                      # "fixed" | "bounded" | "categorical"
    value: Optional[object] = None       # for kind == "fixed"
    low: Optional[float] = None          # for kind == "bounded"
    high: Optional[float] = None         # for kind == "bounded"
    step: Optional[float] = None         # optional discretization step
    choices: Optional[list] = field(default=None)  # for kind == "categorical"


class FormulationOptimizer:
    """
    Wraps one (or two, for multi-objective) trained model(s) + preprocessor(s)
    and exposes a simple `.minimize()` / `.maximize()` / `.multi_objective()`
    API that a Streamlit/FastAPI calculator can call directly.
    """

    def __init__(self, model, preprocessor, continuous_features, categorical_features):
        self.model = model
        self.preprocessor = preprocessor
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features

    # ------------------------------------------------------------------ #
    def _suggest_row(self, trial: optuna.Trial, variables: list[VariableSpec]) -> dict:
        row = {}
        for v in variables:
            if v.kind == "fixed":
                row[v.name] = v.value
            elif v.kind == "bounded":
                if v.step:
                    row[v.name] = trial.suggest_float(v.name, v.low, v.high, step=v.step)
                else:
                    row[v.name] = trial.suggest_float(v.name, v.low, v.high)
            elif v.kind == "categorical":
                row[v.name] = trial.suggest_categorical(v.name, v.choices)
            else:
                raise ValueError(f"Unknown variable kind: {v.kind}")
        return row

    def _predict_raw(self, row: dict) -> float:
        X = self.preprocessor.transform_single(row)
        y_scaled = self.model.predict(X)
        y = self.preprocessor.inverse_transform_target(y_scaled)
        return float(y[0])

    # ------------------------------------------------------------------ #
    # Single-objective                                                    #
    # ------------------------------------------------------------------ #
    def optimize(
        self,
        variables: list[VariableSpec],
        direction: str = "minimize",
        n_trials: int = 200,
        constraints: Optional[Callable[[dict], bool]] = None,
        seed: int = 42,
    ) -> dict:
        """
        Runs Bayesian optimization (Optuna TPE) to find the formulation that
        minimizes or maximizes the model's predicted output.

        `constraints` (optional): a function(row_dict) -> bool. Trials that
        violate the constraint are pruned (assigned a very unfavorable score),
        e.g. NP:additive concentration ratio bounds.
        """
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction=direction, sampler=sampler)

        def objective(trial):
            row = self._suggest_row(trial, variables)
            if constraints is not None and not constraints(row):
                # Heavily penalize infeasible points instead of crashing the study
                return 1e6 if direction == "minimize" else -1e6
            return self._predict_raw(row)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        history = [
            {"trial": t.number, "value": t.value, **t.params}
            for t in study.trials
            if t.value is not None
        ]
        return {
            "best_formulation": study.best_params,
            "best_value": study.best_value,
            "direction": direction,
            "history": history,
            "study": study,
        }

    def minimize(self, variables, **kwargs):
        return self.optimize(variables, direction="minimize", **kwargs)

    def maximize(self, variables, **kwargs):
        return self.optimize(variables, direction="maximize", **kwargs)


class MultiObjectiveFormulationOptimizer:
    """
    Simultaneously optimizes across TWO models sharing (a subset of) the
    same input variables - e.g. minimize Contact Angle AND minimize IFT
    at once - using Optuna's NSGA-II multi-objective sampler. Returns the
    non-dominated Pareto front of formulations.
    """

    def __init__(
        self,
        model_a, preproc_a, direction_a: str,
        model_b, preproc_b, direction_b: str,
    ):
        self.model_a, self.preproc_a, self.direction_a = model_a, preproc_a, direction_a
        self.model_b, self.preproc_b, self.direction_b = model_b, preproc_b, direction_b

    def _predict(self, model, preproc, row: dict) -> float:
        X = preproc.transform_single(row)
        y_scaled = model.predict(X)
        return float(preproc.inverse_transform_target(y_scaled)[0])

    def optimize(
        self,
        variables: list[VariableSpec],
        n_trials: int = 300,
        constraints: Optional[Callable[[dict], bool]] = None,
        seed: int = 42,
    ) -> dict:
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
        study = optuna.create_study(
            directions=[self.direction_a, self.direction_b], sampler=sampler
        )

        def _suggest_row(trial):
            row = {}
            for v in variables:
                if v.kind == "fixed":
                    row[v.name] = v.value
                elif v.kind == "bounded":
                    row[v.name] = trial.suggest_float(v.name, v.low, v.high, step=v.step) \
                        if v.step else trial.suggest_float(v.name, v.low, v.high)
                elif v.kind == "categorical":
                    row[v.name] = trial.suggest_categorical(v.name, v.choices)
            return row

        def objective(trial):
            row = _suggest_row(trial)
            if constraints is not None and not constraints(row):
                pen_a = 1e6 if self.direction_a == "minimize" else -1e6
                pen_b = 1e6 if self.direction_b == "minimize" else -1e6
                return pen_a, pen_b
            val_a = self._predict(self.model_a, self.preproc_a, row)
            val_b = self._predict(self.model_b, self.preproc_b, row)
            return val_a, val_b

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        pareto = []
        for t in study.best_trials:
            pareto.append({"trial": t.number, "objective_a": t.values[0], "objective_b": t.values[1], **t.params})

        pareto_df = pd.DataFrame(pareto).sort_values("objective_a").reset_index(drop=True)
        return {"pareto_front": pareto_df, "study": study}
