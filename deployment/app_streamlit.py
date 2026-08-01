"""
app_streamlit.py
=================
Reference Streamlit front-end for the CA / IFT nano-cEOR calculator,
structured the same way as the team's earlier activated-carbon adsorption
calculator (Prediction Mode + Optimization Mode).

All dropdowns DISPLAY the human-readable label (e.g. "ZrO2", "Biosurfactant",
"Limestone") but still SEND the underlying code (e.g. "NP6", "chem4", "R1")
to the model, via Streamlit's `format_func`. The category dictionaries live
once in `utils/preprocessing.py` (NP_TYPE_MAP, CHEM_ADDITIVE_MAP,
ROCK_TYPE_MAP, SAMPLE_STATE_MAP) so labels can never drift out of sync
between the notebooks and this app.

Run locally with:
    streamlit run deployment/app_streamlit.py

Deploy on Streamlit Community Cloud the same way the adsorption calculator
was deployed - point it at this file, with `CA/models/` and `IFT/models/`
(and the `utils/` package) included in the repo.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.predict import ModelBundle
from utils.optimization import FormulationOptimizer, MultiObjectiveFormulationOptimizer, VariableSpec
from utils.preprocessing import NP_TYPE_MAP, CHEM_ADDITIVE_MAP, ROCK_TYPE_MAP, SAMPLE_STATE_MAP

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CA_DIR = os.path.join(ROOT, "CA", "models")
IFT_DIR = os.path.join(ROOT, "IFT", "models")

ALL_NPS = [f"NP{i}" for i in range(11)]
ALL_CHEMS = [f"chem{i}" for i in range(8)]
ROCK_TYPES = ["R1", "R2", "R3", "R4", "R5"]
SAMPLE_STATES = ["S1", "S2", "S3", "S4", "S5"]

# format_func helpers: display "Label (code)" while the widget still returns
# the raw code that the model/preprocessor expects.
fmt_np = lambda code: f"{NP_TYPE_MAP[code]} ({code})"
fmt_chem = lambda code: f"{CHEM_ADDITIVE_MAP[code]} ({code})"
fmt_rock = lambda code: f"{ROCK_TYPE_MAP[code]} ({code})"
fmt_state = lambda code: f"{SAMPLE_STATE_MAP[code]} ({code})"

st.set_page_config(page_title="Nano-cEOR CA / IFT Calculator", layout="wide")


@st.cache_resource
def load_bundle(model_dir, prefix):
    return ModelBundle(model_dir, prefix)


def sidebar_mode():
    st.sidebar.title("Nano-cEOR Calculator")
    property_choice = st.sidebar.radio("Property", ["Contact Angle (CA)", "Interfacial Tension (IFT)", "Both (multi-objective)"])
    mode = st.sidebar.radio("Mode", ["Prediction", "Optimization"])
    return property_choice, mode


def ca_inputs(prefix="ca"):
    c1, c2, c3 = st.columns(3)
    with c1:
        rock = st.selectbox("Rock Type", ROCK_TYPES, format_func=fmt_rock, key=f"{prefix}_rock")
        sample_state = st.selectbox("Sample State", SAMPLE_STATES, format_func=fmt_state, key=f"{prefix}_state")
        porosity = st.number_input("Porosity (%)", 5.0, 60.0, 20.0, key=f"{prefix}_por")
        permeability = st.number_input("Permeability (mD)", 0.01, 3000.0, 50.0, key=f"{prefix}_perm")
    with c2:
        salinity = st.number_input("Salinity (ppm)", 0.0, 200000.0, 30000.0, key=f"{prefix}_sal")
        temp = st.number_input("Temperature (C)", 15.0, 100.0, 25.0, key=f"{prefix}_temp")
        oil_visc = st.number_input("Oil Viscosity (cP)", 0.5, 300.0, 20.0, key=f"{prefix}_oilvisc")
        oil_sg = st.number_input("Oil SG", 0.6, 1.0, 0.86, key=f"{prefix}_oilsg")
    with c3:
        api = st.number_input("API Gravity", 10.0, 75.0, 33.0, key=f"{prefix}_api")
        viscosity = st.number_input("Nanofluid Viscosity (cP)", 0.3, 20.0, 1.0, key=f"{prefix}_visc")
        nps = st.selectbox("NP Type", ALL_NPS, format_func=fmt_np, key=f"{prefix}_nps")
        np_size = st.number_input("NP Size (nm)", 0.0, 100.0, 20.0, key=f"{prefix}_npsize")
    c4, c5 = st.columns(2)
    with c4:
        np_conc = st.number_input("NP Concentration (wt%)", 0.0, 2.0, 0.3, key=f"{prefix}_npconc")
    with c5:
        chem = st.selectbox("Chemical Additive", ALL_CHEMS, format_func=fmt_chem, key=f"{prefix}_chem")
        chem_conc = st.number_input("Additive Concentration (wt%)", 0.0, 2.0, 0.1, key=f"{prefix}_chemconc")

    return {
        "Rock Type": rock, "Sample State": sample_state, "Porosity (%)": porosity,
        "Permeability (mD)": permeability, "Salinity (ppm)": salinity, "NPs": nps,
        "NPs Size (nm)": np_size, "NPs Conc. (wt%)": np_conc, "Chemical Additive": chem,
        "Additive Conc. (wt%)": chem_conc, "Viscosity (cP)": viscosity,
        "Oil Viscosity (cP)": oil_visc, "Oil SG": oil_sg, "API": api, "Temp.": temp,
    }


def ift_inputs(prefix="ift"):
    c1, c2, c3 = st.columns(3)
    with c1:
        salinity = st.number_input("Salinity (ppm)", 0.0, 200000.0, 30000.0, key=f"{prefix}_sal")
        temp = st.number_input("Temperature (C)", 15.0, 100.0, 25.0, key=f"{prefix}_temp")
        oil_visc = st.number_input("Oil Viscosity (cP)", 0.5, 300.0, 20.0, key=f"{prefix}_oilvisc")
    with c2:
        oil_sg = st.number_input("Oil SG", 0.6, 1.0, 0.86, key=f"{prefix}_oilsg")
        api = st.number_input("API Gravity", 10.0, 75.0, 33.0, key=f"{prefix}_api")
        viscosity = st.number_input("Aqueous Viscosity (cP)", 0.3, 20.0, 1.0, key=f"{prefix}_visc")
    with c3:
        nps = st.selectbox("NP Type", ALL_NPS, format_func=fmt_np, key=f"{prefix}_nps")
        np_size = st.number_input("NP Size (nm)", 0.0, 100.0, 20.0, key=f"{prefix}_npsize")
        np_conc = st.number_input("NP Concentration (wt%)", 0.0, 2.0, 0.3, key=f"{prefix}_npconc")
    c4, c5 = st.columns(2)
    with c4:
        chem = st.selectbox("Chemical Additive", ALL_CHEMS, format_func=fmt_chem, key=f"{prefix}_chem")
    with c5:
        chem_conc = st.number_input("Additive Concentration (wt%)", 0.0, 2.0, 0.1, key=f"{prefix}_chemconc")

    return {
        "Salinity (ppm)": salinity, "NPs": nps, "NPs Size (nm)": np_size,
        "NPs Conc. (wt%)": np_conc, "Chemical Additive": chem, "Additive Conc. (wt%)": chem_conc,
        "Viscosity (cP)": viscosity, "Oil Viscosity (cP)": oil_visc, "Oil SG": oil_sg,
        "API": api, "Temp.C": temp,
    }


def prediction_mode(property_choice):
    if property_choice in ("Contact Angle (CA)", "Both (multi-objective)"):
        st.header("Contact Angle Prediction")
        inputs = ca_inputs()
        if st.button("Predict Contact Angle"):
            bundle = load_bundle(CA_DIR, "ca")
            result = bundle.predict(inputs)
            st.success(f"Predicted CA: **{result['prediction']:.1f} deg** "
                       f"(95% CI: {result['ci_95_lower']:.1f} - {result['ci_95_upper']:.1f} deg)")

    if property_choice in ("Interfacial Tension (IFT)", "Both (multi-objective)"):
        st.header("Interfacial Tension Prediction")
        inputs = ift_inputs()
        if st.button("Predict IFT"):
            bundle = load_bundle(IFT_DIR, "ift")
            result = bundle.predict(inputs)
            st.success(f"Predicted IFT: **{result['prediction']:.2f} mN/m** "
                       f"(95% CI: {result['ci_95_lower']:.2f} - {result['ci_95_upper']:.2f} mN/m)")


def build_variable_specs(fixed_dict, bounded_dict, categorical_dict):
    variables = []
    for name, value in fixed_dict.items():
        variables.append(VariableSpec(name, "fixed", value=value))
    for name, (lo, hi) in bounded_dict.items():
        variables.append(VariableSpec(name, "bounded", low=lo, high=hi))
    for name, choices in categorical_dict.items():
        variables.append(VariableSpec(name, "categorical", choices=choices))
    return variables


# Friendly-label lookup used to decorate the optimizer's output (which is
# expressed in codes, since that's what the model was trained on).
_LABEL_MAPS = {
    "NPs": NP_TYPE_MAP,
    "Chemical Additive": CHEM_ADDITIVE_MAP,
    "Rock Type": ROCK_TYPE_MAP,
    "Sample State": SAMPLE_STATE_MAP,
}


def humanize_formulation(formulation: dict) -> dict:
    """Return a copy of an optimizer result dict with '<code> (<Label>)' values
    for any categorical field, so the UI never shows a bare 'NP6' or 'chem4'."""
    out = {}
    for k, v in formulation.items():
        if k in _LABEL_MAPS and v in _LABEL_MAPS[k]:
            out[k] = f"{v} ({_LABEL_MAPS[k][v]})"
        elif isinstance(v, float):
            out[k] = round(v, 4)
        else:
            out[k] = v
    return out


def humanize_pareto_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, label_map in _LABEL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(lambda c: f"{c} ({label_map.get(c, c)})")
    return df


def optimization_mode(property_choice):
    st.header("Formulation Optimization")
    objective = st.selectbox("Objective", ["Minimize", "Maximize"])
    optimize_vars = st.multiselect(
        "Variables to optimize",
        ["NPs", "NPs Size (nm)", "NPs Conc. (wt%)", "Chemical Additive", "Additive Conc. (wt%)",
         "Salinity (ppm)", "Temp."],
        default=["NPs", "NPs Conc. (wt%)", "Chemical Additive"],
    )
    n_trials = st.slider("Optimization trials", 50, 500, 200, step=50)

    st.subheader("Fixed conditions")
    fixed_inputs = ca_inputs(prefix="fixed") if property_choice != "Interfacial Tension (IFT)" else ift_inputs(prefix="fixed")

    if st.button("Run Optimization"):
        bounded = {}
        categorical = {}
        fixed = {}
        default_bounds = {
            "NPs Size (nm)": (0, 100), "NPs Conc. (wt%)": (0, 1.5),
            "Additive Conc. (wt%)": (0, 1.0), "Salinity (ppm)": (0, 180000),
            "Temp.": (15, 90),
        }
        for k, v in fixed_inputs.items():
            if k in optimize_vars:
                if k in ("NPs", "Chemical Additive"):
                    categorical[k] = ALL_NPS if k == "NPs" else ALL_CHEMS
                elif k in default_bounds:
                    bounded[k] = default_bounds[k]
            else:
                fixed[k] = v

        variables = build_variable_specs(fixed, bounded, categorical)
        direction = "minimize" if objective == "Minimize" else "maximize"

        if property_choice == "Both (multi-objective)":
            ca_bundle = load_bundle(CA_DIR, "ca")
            ift_bundle = load_bundle(IFT_DIR, "ift")
            mo = MultiObjectiveFormulationOptimizer(
                ca_bundle.model, ca_bundle.preprocessor, "minimize",
                ift_bundle.model, ift_bundle.preprocessor, "minimize",
            )
            res = mo.optimize(variables, n_trials=n_trials)
            st.write("Pareto-optimal formulations (top 15):")
            st.dataframe(humanize_pareto_df(res["pareto_front"].head(15)))
        else:
            model_dir, prefix = (CA_DIR, "ca") if property_choice == "Contact Angle (CA)" else (IFT_DIR, "ift")
            bundle = load_bundle(model_dir, prefix)
            opt = FormulationOptimizer(bundle.model, bundle.preprocessor,
                                        bundle.feature_metadata["continuous_features"],
                                        bundle.feature_metadata["categorical_features"])
            res = opt.optimize(variables, direction=direction, n_trials=n_trials)
            st.success(f"Best predicted value: {res['best_value']:.3f}")
            st.write("Best formulation:")
            st.json(humanize_formulation(res["best_formulation"]))
            hist_df = pd.DataFrame(res["history"])
            st.line_chart(hist_df.set_index("trial")["value"].cummin() if direction == "minimize"
                           else hist_df.set_index("trial")["value"].cummax())


def main():
    property_choice, mode = sidebar_mode()
    st.title("Nanoparticle-Assisted EOR — CA / IFT Calculator")
    st.caption("Prediction and formulation-optimization tool built on validated XGBoost models "
               "(Scientific Reports 2026 CA framework + companion IFT framework).")
    if mode == "Prediction":
        prediction_mode(property_choice)
    else:
        optimization_mode(property_choice)


if __name__ == "__main__":
    main()
