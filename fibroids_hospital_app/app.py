# Import libraries
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="Uterine Fibroids Risk Tool", layout="wide")
st.title("🩺 Uterine Fibroids Risk Prediction")
st.markdown("**Master's Project — Aga Khan Inspired Logistic Regression Model**")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('uf_fibroids_final_model.pkl')
    preprocessor = joblib.load('uf_preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

# Sidebar for patient information
st.sidebar.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 15, 49, 35)
    race = st.selectbox("Race/Ethnicity", ['Black', 'Indian', 'White', 'Hispanic', 'Asian', 'Other'])
    bmi = st.slider("BMI", 15.0, 50.0, 27.0, step=0.1)
    parity = st.slider("Parity (Number of births)", 0, 5, 1)
    menarche = st.slider("Age at Menarche", 10, 16, 12)

with col2:
    hypertension = st.radio("Hypertension", [0, 1], horizontal=True)
    pcos = st.radio("PCOS", [0, 1], horizontal=True)
    vit_d = st.radio("Vitamin D Deficient", [0, 1], horizontal=True)
    diet = st.slider("Diet Quality (0-10)", 0.0, 10.0, 5.0)
    activity = st.slider("Physical Activity (hrs/week)", 0.0, 20.0, 5.0)
    smoking = st.radio("Smoking", [0, 1], horizontal=True)
    family_hist = st.radio("Family History of Fibroids", [0, 1], horizontal=True)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

# Prediction
input_df = pd.DataFrame([{
    'Age': age, 'Race': race, 'BMI': bmi, 'Parity': parity,
    'Menarche_Age': menarche, 'Hypertension': hypertension,
    'PCOS': pcos, 'Vitamin_D_Deficient': vit_d,
    'Diet_Quality': diet, 'Physical_Activity': activity,
    'Smoking': smoking, 'Family_History': family_hist,
    'Stress_Level': stress
}])

probability = model.predict_proba(input_df)[0][1]
risk_percent = round(probability * 100, 1)

if risk_percent >= 70:
    risk_level = "🔴 HIGH RISK"
elif risk_percent >= 40:
    risk_level = "🟠 MODERATE RISK"
else:
    risk_level = "🟢 LOW RISK"

st.subheader("Prediction Result")
st.metric("Risk of Uterine Fibroids", f"{risk_percent}%", risk_level)

# SHAP Force Plot
st.subheader("SHAP Explanation (Why this risk?)")

explainer = shap.LinearExplainer(model.named_steps['classifier'],
                                 preprocessor.transform(input_df))
shap_values = explainer.shap_values(preprocessor.transform(input_df))

fig = shap.force_plot(explainer.expected_value, shap_values[0],
                      preprocessor.transform(input_df),
                      feature_names=preprocessor.get_feature_names_out(),
                      matplotlib=True, show=False)

with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
    plt.savefig(tmp.name, bbox_inches='tight')
    shap_path = tmp.name

st.image(shap_path, use_column_width=True)

# Downloadab
def create_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Uterine Fibroids Risk Assessment Report", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", ln=1)
    
    pdf.cell(200, 10, txt=f"Patient Risk: {risk_percent}% — {risk_level}", ln=1)
    pdf.cell(200, 10, txt="Key Risk Factors:", ln=1)
    
    # Top factors from coefficients
    coef_df = pd.read_csv('logistic_regression_coefficients.csv')
    top_features = coef_df.head(8)
    for _, row in top_features.iterrows():
        pdf.cell(200, 10, txt=f"• {row['Feature']}: {row['Odds Ratio']:.2f}x", ln=1)
    
    pdf.cell(200, 10, txt="SHAP Force Plot attached below (explanation)", ln=1)
    
    pdf.image(shap_path, x=10, y=None, w=180)
    
    pdf.output("fibroids_risk_report.pdf")
    return "fibroids_risk_report.pdf"

if st.button("📄 Generate & Download Full Report (PDF)"):
    report_path = create_pdf()
    with open(report_path, "rb") as f:
        st.download_button(
            label="Click to Download PDF Report",
            data=f,
            file_name=f"fibroids_risk_report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
    os.unlink(report_path)  # clean up

st.success("App ready for hospital use! Run with: streamlit run app.py")