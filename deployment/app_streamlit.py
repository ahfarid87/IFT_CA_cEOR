"""
app_streamlit.py
=================
Reference Streamlit front-end for the CA / IFT nano-cEOR calculator,
structured the same way as the team's earlier activated-carbon adsorption
calculator (Prediction Mode + Optimization Mode).

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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CA_DIR = os.path.join(ROOT, "CA", "models")
IFT_DIR = os.path.join(ROOT, "IFT", "models")

# ---------- Display Name -> Model Code ----------
NP_MAP = {
    "None": "NP0",
    "SiO₂": "NP1",
    "Al₂O₃": "NP2",
    "TiO₂": "NP3",
    "Fe₃O₄": "NP4",
    "NiO": "NP5",
    "ZrO₂": "NP6",
    "CuO": "NP7",
    "Carbon Nanotubes (CN)": "NP8",
    "MgO": "NP9",
    "ZnO": "NP10",
}

CHEM_MAP = {
    "Anionic Surfactant": "chem1",
    "Cationic Surfactant": "chem2",
    "Nonionic Surfactant": "chem3",
    "Biosurfactant": "chem4",
    "Alcohol / Solvent": "chem5",
    "Anionic Polymer": "chem6",
    "Nonionic Polymer": "chem7",
}

ROCK_MAP = {
    "Limestone": "R1",
    "Sandstone": "R2",
    "Glass": "R3",
    "Dolomite": "R4",
    "Carbonate": "R5",
}

SAMPLE_STATE_MAP = {
    "Oil Aged": "S1",
    "Nano Aged": "S2",
    "HS Nano": "S3",
    "LS Nano": "S4",
    "Surfactant Aged": "S5",
}

ALL_NPS = list(NP_MAP.keys())
ALL_CHEMS = list(CHEM_MAP.keys())
ROCK_TYPES = list(ROCK_MAP.keys())
SAMPLE_STATES = list(SAMPLE_STATE_MAP.keys())

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
	rock_name = st.selectbox("Rock Type", ROCK_TYPES, key=f"{prefix}_rock")
	sample_name = st.selectbox("Sample State", SAMPLE_STATES, key=f"{prefix}_state")

	rock = ROCK_MAP[rock_name]
	sample_state = SAMPLE_STATE_MAP[sample_name]        
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
        np_name = st.selectbox("Nanoparticle", ALL_NPS, key=f"{prefix}_nps")
	nps = NP_MAP[np_name]
        np_size = st.number_input("NP Size (nm)", 0.0, 100.0, 20.0, key=f"{prefix}_npsize")
    c4, c5 = st.columns(2)
    with c4:
        np_conc = st.number_input("NP Concentration (wt%)", 0.0, 2.0, 0.3, key=f"{prefix}_npconc")
    with c5:
        chem_name = st.selectbox("Chemical Additive", ALL_CHEMS, key=f"{prefix}_chem")
	chem = CHEM_MAP[chem_name]
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
        np_name = st.selectbox("Nanoparticle", ALL_NPS, key=f"{prefix}_nps")
	nps = NP_MAP[np_name]
        np_size = st.number_input("NP Size (nm)", 0.0, 100.0, 20.0, key=f"{prefix}_npsize")
        np_conc = st.number_input("NP Concentration (wt%)", 0.0, 2.0, 0.3, key=f"{prefix}_npconc")
    c4, c5 = st.columns(2)
    with c4:
        chem_name = st.selectbox("Chemical Additive", ALL_CHEMS, key=f"{prefix}_chem")
	chem = CHEM_MAP[chem_name]
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
            st.dataframe(res["pareto_front"].head(15))
        else:
            model_dir, prefix = (CA_DIR, "ca") if property_choice == "Contact Angle (CA)" else (IFT_DIR, "ift")
            bundle = load_bundle(model_dir, prefix)
            opt = FormulationOptimizer(bundle.model, bundle.preprocessor,
                                        bundle.feature_metadata["continuous_features"],
                                        bundle.feature_metadata["categorical_features"])
            res = opt.optimize(variables, direction=direction, n_trials=n_trials)
            st.success(f"Best predicted value: {res['best_value']:.3f}")
            st.write("Best formulation:")
            st.json(res["best_formulation"])
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
