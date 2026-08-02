"""
app_streamlit.py
=================
Reference Streamlit front-end for the CA / IFT nano-cEOR calculator,
structured the same way as the team's earlier activated-carbon adsorption
calculator (Prediction Mode + Optimization Mode).

Design notes
------------
* All categorical dropdowns DISPLAY the human-readable label (e.g. "ZrO2",
  "Biosurfactant", "Limestone") but still SEND the underlying code (e.g.
  "NP6", "chem4", "R1") to the model, via Streamlit's `format_func`. The
  category dictionaries live once in `utils/preprocessing.py` so labels can
  never drift out of sync between the notebooks and this app.

* Oil SG is NOT collected from the user - it is a deterministic function of
  API gravity (SG = 141.5 / (API + 131.5)) and is computed internally right
  before the row is sent to the model, for both CA and IFT.

* Optimization Mode only allows optimizing FORMULATION variables (NP type,
  NP size, NP concentration, chemical additive type, additive concentration)
  - i.e. the things an engineer is actually designing. Reservoir/operating
  conditions (rock type, porosity, permeability, salinity, temperature,
  fluid properties) are always treated as fixed context. Whichever
  formulation variable is selected for optimization is automatically
  removed from the "fixed conditions" form (no more redundant/ignored
  input box for a variable that's about to be searched over).

Run locally with:
    streamlit run deployment/app_streamlit.py
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

# The only variables the optimizer is allowed to search over - i.e. the
# formulation the engineer is designing, not the reservoir/operating context.
OPTIMIZABLE_FIELDS = ["NPs", "NPs Size (nm)", "NPs Conc. (wt%)", "Chemical Additive", "Additive Conc. (wt%)"]
OPTIMIZABLE_BOUNDS = {
    "NPs Size (nm)": (0.0, 100.0),
    "NPs Conc. (wt%)": (0.0, 1.5),
    "Additive Conc. (wt%)": (0.0, 1.0),
}

# --------------------------------------------------------------------------- #
# Branding                                                                     #
# --------------------------------------------------------------------------- #
# Drop the actual institutional logo files here (not included in this repo -
# official KFUPM / CPG marks are trademarked and must be sourced directly
# from the institution, not generated or scraped). Any PNG/JPG/SVG placed at
# these paths will render automatically; if a file is missing, that logo
# slot is simply skipped (no broken-image icon).
LOGO_DIR = os.path.join(ROOT, "deployment", "assets")
KFUPM_LOGO_PATH = os.path.join(LOGO_DIR, "kfupm_logo.png")
CPG_LOGO_PATH = os.path.join(LOGO_DIR, "cpg_logo.png")
COPYRIGHT_NOTICE = "© 2026 Ahmed Ibrahim. All rights reserved."

st.set_page_config(page_title="Nano-cEOR CA / IFT Calculator", layout="wide")


@st.cache_resource
def load_bundle(model_dir, prefix):
    return ModelBundle(model_dir, prefix)


def sg_from_api(api: float) -> float:
    """Standard oilfield correlation: SG = 141.5 / (API + 131.5)."""
    return 141.5 / (api + 131.5)


def sidebar_mode():
    st.sidebar.title("Nano-cEOR Calculator")
    property_choice = st.sidebar.radio("Property", ["Contact Angle (CA)", "Interfacial Tension (IFT)", "Both (multi-objective)"])
    mode = st.sidebar.radio("Mode", ["Prediction", "Optimization"])
    st.sidebar.markdown("---")
    st.sidebar.caption(COPYRIGHT_NOTICE)
    return property_choice, mode


# --------------------------------------------------------------------------- #
# Reusable field groups                                                       #
# --------------------------------------------------------------------------- #
def render_reservoir_fields(prefix: str) -> dict:
    """Rock type / sample state / porosity / permeability - CA model only, always fixed."""
    c1, c2 = st.columns(2)
    with c1:
        rock = st.selectbox("Rock Type", ROCK_TYPES, format_func=fmt_rock, key=f"{prefix}_rock")
        sample_state = st.selectbox("Sample State", SAMPLE_STATES, format_func=fmt_state, key=f"{prefix}_state")
    with c2:
        porosity = st.number_input("Porosity (%)", 5.0, 60.0, 20.0, key=f"{prefix}_por")
        permeability = st.number_input("Permeability (mD)", 0.01, 3000.0, 50.0, key=f"{prefix}_perm")
    return {"Rock Type": rock, "Sample State": sample_state,
            "Porosity (%)": porosity, "Permeability (mD)": permeability}


def render_fluid_fields(prefix: str, temp_key: str = "Temp.") -> dict:
    """Salinity / temperature / oil & aqueous properties - always fixed (not optimizable).
    Oil SG is computed from API, not collected from the user."""
    c1, c2, c3 = st.columns(3)
    with c1:
        salinity = st.number_input("Salinity (ppm)", 0.0, 200000.0, 30000.0, key=f"{prefix}_sal")
        temp = st.number_input("Temperature (C)", 15.0, 100.0, 25.0, key=f"{prefix}_temp")
    with c2:
        oil_visc = st.number_input("Oil Viscosity (cP)", 0.5, 300.0, 20.0, key=f"{prefix}_oilvisc")
        api = st.number_input("Oil API Gravity", 10.0, 75.0, 33.0, key=f"{prefix}_api")
    with c3:
        viscosity = st.number_input("Aqueous / Nanofluid Viscosity (cP)", 0.3, 20.0, 1.0, key=f"{prefix}_visc")
        oil_sg = sg_from_api(api)
        st.metric("Oil SG (from API)", f"{oil_sg:.4f}")

    return {
        "Salinity (ppm)": salinity, temp_key: temp, "Oil Viscosity (cP)": oil_visc,
        "API": api, "Oil SG": oil_sg, "Viscosity (cP)": viscosity,
    }


def render_formulation_fields(skip: set, prefix: str) -> dict:
    """NP type/size/conc + chemical additive type/conc - each one is omitted
    if it's in `skip` (because it's being optimized instead of held fixed).

    Domain rule enforced live in the UI: selecting "None" for NP type or
    Chemical Additive locks the corresponding amount field(s) at 0, since
    "0.3 wt% of no nanoparticle" is not a physically meaningful input. The
    same rule is also enforced server-side (utils.preprocessing.
    enforce_domain_rules) so it holds even if a widget is skipped/optimized.
    """
    values = {}
    shown_any = False
    cols = st.columns(2)
    with cols[0]:
        nps_value = None
        if "NPs" not in skip:
            nps_value = st.selectbox("NP Type", ALL_NPS, format_func=fmt_np, key=f"{prefix}_nps")
            values["NPs"] = nps_value
            shown_any = True
        np_is_none = (nps_value == "NP0")
        if "NPs Size (nm)" not in skip:
            if np_is_none:
                st.number_input("NP Size (nm)", value=0.0, disabled=True, key=f"{prefix}_npsize",
                                 help="Locked at 0 - NP Type is set to None.")
                values["NPs Size (nm)"] = 0.0
            else:
                values["NPs Size (nm)"] = st.number_input("NP Size (nm)", 0.0, 100.0, 20.0, key=f"{prefix}_npsize")
            shown_any = True
        if "NPs Conc. (wt%)" not in skip:
            if np_is_none:
                st.number_input("NP Concentration (wt%)", value=0.0, disabled=True, key=f"{prefix}_npconc",
                                 help="Locked at 0 - NP Type is set to None.")
                values["NPs Conc. (wt%)"] = 0.0
            else:
                values["NPs Conc. (wt%)"] = st.number_input("NP Concentration (wt%)", 0.0, 2.0, 0.3, key=f"{prefix}_npconc")
            shown_any = True
    with cols[1]:
        chem_value = None
        if "Chemical Additive" not in skip:
            chem_value = st.selectbox("Chemical Additive", ALL_CHEMS, format_func=fmt_chem, key=f"{prefix}_chem")
            values["Chemical Additive"] = chem_value
            shown_any = True
        chem_is_none = (chem_value == "chem0")
        if "Additive Conc. (wt%)" not in skip:
            if chem_is_none:
                st.number_input("Additive Concentration (wt%)", value=0.0, disabled=True, key=f"{prefix}_chemconc",
                                 help="Locked at 0 - Chemical Additive is set to None.")
                values["Additive Conc. (wt%)"] = 0.0
            else:
                values["Additive Conc. (wt%)"] = st.number_input("Additive Concentration (wt%)", 0.0, 2.0, 0.1, key=f"{prefix}_chemconc")
            shown_any = True
    if not shown_any:
        st.caption("All formulation variables below are being optimized (see above).")
    return values


def ca_inputs(prefix="ca") -> dict:
    st.markdown("**Reservoir**")
    reservoir = render_reservoir_fields(prefix)
    st.markdown("**Fluid & operating conditions**")
    fluid = render_fluid_fields(prefix, temp_key="Temp.")
    st.markdown("**Nanofluid formulation**")
    formulation = render_formulation_fields(skip=set(), prefix=prefix)
    return {**reservoir, **fluid, **formulation}


def ift_inputs(prefix="ift") -> dict:
    st.markdown("**Fluid & operating conditions**")
    fluid = render_fluid_fields(prefix, temp_key="Temp.C")
    st.markdown("**Nanofluid formulation**")
    formulation = render_formulation_fields(skip=set(), prefix=prefix)
    return {**fluid, **formulation}


# --------------------------------------------------------------------------- #
# Prediction mode                                                             #
# --------------------------------------------------------------------------- #
def prediction_mode(property_choice):
    if property_choice in ("Contact Angle (CA)", "Both (multi-objective)"):
        st.header("Contact Angle Prediction")
        inputs = ca_inputs()
        if st.button("Predict Contact Angle"):
            bundle = load_bundle(CA_DIR, "ca")
            result = bundle.predict(inputs)
	    result = {k: (0.01 if isinstance(v, (int, float)) and v < 0 else v) for k, v in result.items()}
            st.success(f"Predicted CA: **{result['prediction']:.1f} deg** "
                       f"(95% CI: {result['ci_95_lower']:.1f} - {result['ci_95_upper']:.1f} deg)")

    if property_choice in ("Interfacial Tension (IFT)", "Both (multi-objective)"):
        st.header("Interfacial Tension Prediction")
        inputs = ift_inputs()
        if st.button("Predict IFT"):
            bundle = load_bundle(IFT_DIR, "ift")
            result = bundle.predict(inputs)
	    result = {k: (0.01 if isinstance(v, (int, float)) and v < 0 else v) for k, v in result.items()}
            st.success(f"Predicted IFT: **{result['prediction']:.2f} mN/m** "
                       f"(95% CI: {result['ci_95_lower']:.2f} - {result['ci_95_upper']:.2f} mN/m)")


# --------------------------------------------------------------------------- #
# Optimization mode                                                           #
# --------------------------------------------------------------------------- #
def build_variable_specs(fixed_dict, bounded_dict, categorical_dict):
    variables = []
    for name, value in fixed_dict.items():
        variables.append(VariableSpec(name, "fixed", value=value))
    for name, (lo, hi) in bounded_dict.items():
        variables.append(VariableSpec(name, "bounded", low=lo, high=hi))
    for name, choices in categorical_dict.items():
        variables.append(VariableSpec(name, "categorical", choices=choices))
    return variables


_LABEL_MAPS = {
    "NPs": NP_TYPE_MAP, "Chemical Additive": CHEM_ADDITIVE_MAP,
    "Rock Type": ROCK_TYPE_MAP, "Sample State": SAMPLE_STATE_MAP,
}


def humanize_formulation(formulation: dict) -> dict:
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
        "Variables to optimize (formulation only)",
        OPTIMIZABLE_FIELDS,
        default=["NPs", "NPs Conc. (wt%)", "Chemical Additive"],
        help="Reservoir and operating conditions (rock type, porosity, permeability, "
             "salinity, temperature, fluid properties) are always held fixed below - "
             "only the nanofluid formulation itself can be optimized.",
    )
    n_trials = st.slider("Optimization trials", 50, 500, 200, step=50)
    skip = set(optimize_vars)

    st.subheader("Fixed reservoir / operating conditions")
    st.caption("Any variable selected above for optimization is removed from this form.")

    fixed = {}
    if property_choice in ("Contact Angle (CA)", "Both (multi-objective)"):
        st.markdown("**Reservoir**")
        fixed.update(render_reservoir_fields("optfixed"))

    st.markdown("**Fluid & operating conditions**")
    if property_choice == "Interfacial Tension (IFT)":
        fixed.update(render_fluid_fields("optfixed", temp_key="Temp.C"))
    elif property_choice == "Both (multi-objective)":
        fluid = render_fluid_fields("optfixed", temp_key="Temp.")
        fluid["Temp.C"] = fluid["Temp."]   # same physical temperature fed to both models
        fixed.update(fluid)
    else:  # CA only
        fixed.update(render_fluid_fields("optfixed", temp_key="Temp."))

    st.markdown("**Fixed formulation variables** *(not selected for optimization)*")
    fixed.update(render_formulation_fields(skip=skip, prefix="optfixed"))

    if st.button("Run Optimization"):
        bounded = {k: OPTIMIZABLE_BOUNDS[k] for k in optimize_vars if k in OPTIMIZABLE_BOUNDS}
        categorical = {}
        if "NPs" in optimize_vars:
            categorical["NPs"] = ALL_NPS
        if "Chemical Additive" in optimize_vars:
            categorical["Chemical Additive"] = ALL_CHEMS

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


def render_header_logos():
    """Show institutional logos if present at deployment/assets/*.png; skip
    silently otherwise. See LOGO_DIR note above for where to add the files."""
    kfupm_exists = os.path.exists(KFUPM_LOGO_PATH)
    cpg_exists = os.path.exists(CPG_LOGO_PATH)
    if not (kfupm_exists or cpg_exists):
        return
    cols = st.columns([1, 1, 4])
    if kfupm_exists:
        with cols[0]:
            st.image(KFUPM_LOGO_PATH, width=120)
    if cpg_exists:
        with cols[1]:
            st.image(CPG_LOGO_PATH, width=120)


def render_footer():
    st.markdown("---")
    st.caption(COPYRIGHT_NOTICE)


def main():
    property_choice, mode = sidebar_mode()
    render_header_logos()
    st.title("Nanoparticle-Assisted EOR — CA / IFT Calculator")
    st.caption("Prediction and formulation-optimization tool built on validated XGBoost models "
               "(Scientific Reports 2026 CA framework + companion IFT framework).")
    if mode == "Prediction":
        prediction_mode(property_choice)
    else:
        optimization_mode(property_choice)
    render_footer()


if __name__ == "__main__":
    main()
