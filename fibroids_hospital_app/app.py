# Import libraries
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from fpdf import FPDF
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import tempfile
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Uterine Fibroids Risk Tool", layout="wide")

# Auth — load config and render login page

base_dir = os.path.dirname(__file__)

with open(os.path.join(base_dir, "config.yaml")) as f:
    config = yaml.load(f, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

name, authentication_status, username = authenticator.login(
    location="main",
    fields={
        "Form name": "🩺 Uterine Fibroids Risk Tool",
        "Username": "Username",
        "Password": "Password",
        "Login": "Sign In"
    }
)

# ── Not logged in ─────────────────────────────────────────────────────────────
if authentication_status is False:
    st.error("Incorrect username or password. Please try again.")
    st.stop()

if authentication_status is None:
    st.info("Please enter your credentials to access the system.")
    st.stop()

# ── Logged in ─────────────────────────────────────────────────────────────────
role = config["credentials"]["usernames"][username]["role"]

with st.sidebar:
    st.markdown(f"**Logged in as:** {name}")
    st.markdown(f"**Role:** {role.title()}")
    st.markdown("---")
    authenticator.logout("Sign Out", location="sidebar")

st.title("🩺 Uterine Fibroids Risk Prediction")

# Load model and preprocessor
@st.cache_resource
def load_assets():
    try:
        # FIX 2: Load pkl files relative to app.py
        model = joblib.load(os.path.join(base_dir, 'uf_fibroids_final_model.pkl'))
        preprocessor = joblib.load(os.path.join(base_dir, 'uf_preprocessor.pkl'))
        return model, preprocessor
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

model, preprocessor = load_assets()

# Patient input form (both roles see this)
st.sidebar.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 15, 49, 35)
    race = st.selectbox("Race/Ethnicity", ['Black', 'Indian', 'White', 'Hispanic', 'Asian', 'Other'])
    bmi = st.slider("BMI", 15.0, 50.0, 27.0, step=0.1)
    parity = st.slider("Parity (number of births)", 0, 5, 1)
    menarche_age = st.slider("Age at menarche", 10, 16, 12)

with col2:
    hypertension = st.radio("Hypertension", [0, 1], horizontal=True, format_func=lambda x: "Yes" if x else "No")
    pcos = st.radio("PCOS", [0, 1], horizontal=True, format_func=lambda x: "Yes" if x else "No")
    vitamin_d_def = st.radio("Vitamin D Deficient", [0, 1], horizontal=True, format_func=lambda x: "Yes" if x else "No")
    diet_quality = st.slider("Diet Quality (0-10)", 0.0, 10.0, 5.0)
    physical_activity = st.slider("Physical Activity (hours/week)", 0.0, 20.0, 5.0)
    smoking = st.radio("Smoking", [0, 1], horizontal=True, format_func=lambda x: "Yes" if x else "No")
    family_history = st.radio("Family History of Fibroids", [0, 1], horizontal=True, format_func=lambda x: "Yes" if x else "No")
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

input_df = pd.DataFrame([{
    'Age': age, 'Race': race, 'BMI': bmi, 'Parity': parity,
    'Menarche_Age': menarche_age, 'Hypertension': hypertension,
    'PCOS': pcos, 'Vitamin_D_Deficient': vitamin_d_def,
    'Diet_Quality': diet_quality, 'Physical_Activity': physical_activity,
    'Smoking': smoking, 'Family_History': family_history,
    'Stress_Level': stress_level
}])

# Prediction (both roles see this)
proba = model.predict_proba(input_df)[0][1]
risk_pct = round(proba * 100, 1)

if risk_pct >= 70:
    level, color = "HIGH RISK", "🔴"
elif risk_pct >= 40:
    level, color = "MODERATE RISK", "🟠"
else:
    level, color = "LOW RISK", "🟢"

st.subheader("Prediction Result")
st.metric("Estimated Risk of Uterine Fibroids", f"{risk_pct}%", f"{color} {level}")

# NURSE VIEW ends here
if role == "nurse":
    st.info("For detailed model explanation and PDF report, please consult the attending doctor.")
    st.markdown("---")
    st.caption("App ready for research / demonstration use.")
    st.stop()

# DOCTOR VIEW — SHAP + PDF
st.subheader("SHAP Explanation (Why this prediction?)")

try:
    X_trans = preprocessor.transform(input_df)
    feature_names = list(preprocessor.get_feature_names_out())
    classifier = model.named_steps['classifier']

    coefficients = classifier.coef_[0]
    shap_vals = coefficients * X_trans[0]

    top_n = 12
    indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    top_shap = shap_vals[indices]
    top_names = [feature_names[i] for i in indices]
    top_data = X_trans[0][indices]

    def clean_name(n):
        return n.replace("num__", "").replace("cat__", "").replace("_", " ").title()

    top_labels = [
        f"{clean_name(fn)}  =  {fv:.2f}" if isinstance(fv, float) else f"{clean_name(fn)}  =  {fv}"
        for fn, fv in zip(top_names, top_data)
    ]

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#f0f0f0")
    ax.set_facecolor("#f0f0f0")

    colors = ["#e03232" if v > 0 else "#3278e0" for v in top_shap]
    bars = ax.barh(range(top_n), top_shap, color=colors, edgecolor="white", height=0.6)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_labels, fontsize=13)
    ax.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars, top_shap)):
        offset = max(np.abs(top_shap)) * 0.02
        ax.text(
            val + (offset if val >= 0 else -offset),
            i, f"{val:+.4f}",
            va='center',
            ha='left' if val >= 0 else 'right',
            fontsize=11, fontweight='bold', color="#222222"
        )

    ax.axvline(0, color="#444444", linewidth=1.2, linestyle="--")
    ax.set_xlabel("SHAP Value  (positive = increases risk,  negative = decreases risk)", fontsize=13)
    ax.set_title(
        f"Top {top_n} Features Driving This Prediction  |  Predicted risk: {proba:.1%}",
        fontsize=15, fontweight="bold", pad=15
    )

    legend_elements = [
        Patch(facecolor="#e03232", label="Increases risk"),
        Patch(facecolor="#3278e0", label="Decreases risk"),
    ]
    ax.legend(handles=legend_elements, fontsize=12, loc="lower right")
    plt.tight_layout(pad=2.0)

    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
    plt.savefig(tmp_path, dpi=180, bbox_inches='tight', pad_inches=0.5, facecolor="#f0f0f0")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)

    st.image(tmp_path, use_container_width=True)
    os.unlink(tmp_path)

except Exception as e:
    st.warning(f"SHAP generation failed: {str(e)}")
    st.exception(e)

# PDF Report (doctor only)
def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Uterine Fibroids Risk Assessment Report", ln=1, align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
    pdf.cell(0, 10, f"Reviewed by: {name}", ln=1)
    pdf.cell(0, 10, f"Estimated Risk: {risk_pct}% - {level}", ln=1)
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Patient Details:", ln=1)
    pdf.set_font("Helvetica", "", 11)

    for col, val in input_df.iloc[0].items():
        pdf.cell(0, 8, f"- {col.replace('_', ' ')}: {val}", ln=1)

    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 8, "Note: This is a research prototype. Not a substitute for clinical evaluation or imaging.")

    # FIX 3: Typo — NamedTemporementFile -> NamedTemporaryFile
    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf').name
    pdf.output(pdf_file)
    return pdf_file

if st.button("Generate & Download PDF Report"):
    try:
        pdf_path = generate_pdf()
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name=f"uf_risk_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )
        os.unlink(pdf_path)
    except Exception as e:
        st.error(f"PDF failed: {str(e)}")

st.markdown("---")
st.caption("App ready for research / demonstration use.")