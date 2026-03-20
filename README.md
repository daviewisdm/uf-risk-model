# Uterine Fibroids Risk Prediction Tool

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg)](https://streamlit.io/)
  

## Overview

This project develops a simple, interpretable machine learning model to estimate the **risk of uterine fibroids** in women of reproductive age (15–49 years) using basic clinical and demographic data.

**Goal**: Create a low-cost, explainable screening support tool suitable for resource-limited settings (e.g., Kenyan public hospitals / clinics).

**Model**: Logistic Regression trained on **synthetic data** (2,000 records) engineered from real epidemiological evidence.

**Key features**:
- Instant risk probability + category (Low / Moderate / High)
- SHAP force plot for individual-level explanation
- One-click downloadable PDF report
- Hospital-friendly Streamlit web app

**Important Disclaimer**  
🔴 This is an **academic research prototype**.  
It has **not** been trained or validated on real patient data.  
**Never use it for clinical decisions** — always rely on ultrasound, clinical exam, and specialist opinion.

[Screenshot placeholder: App landing page showing prediction result and SHAP explanation]

## Project Structure
```
uf-risk-model/
│
├── app/
│   ├── app.py                  # Main Streamlit app
│   ├── shap_plot.py            # SHAP visualisation logic
│   └── pdf_report.py           # PDF generation logic
│
├── model/
│   ├── uf_fibroids_final_model.pkl
│   └── uf_preprocessor.pkl
│
├── notebooks/
│   └── UF_prevalence_prediction.ipynb   # Full EDA → modelling workflow
│
├── data/
│   ├── synthetic_fibroids_data.csv      # Generated training data
│   └── logistic_regression_coefficients.csv  # Coefficients & odds ratios
│
├── scripts/
│   └── synthetic_data.py       # Script to generate synthetic dataset
│
├── docs/
│   └── USER_MANUAL.md          # End-user guide
│
├── requirements.txt
└── README.md
```

## Quick Start – Run the Demo App
1. Clone/Fork the repo
```bash
git clone https://github.com/<your-username>/uf-risk-model.git
cd uf-risk-model
```
2. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn joblib shap matplotlib fpdf2
```
3. Navigate to the app folder

```bash
cd fibroids_hospital_app

```
4. Launch
   
```bash
streamlit run app.py
```
5. Open http://localhost:8501 in your browser

### Features
- Real-time risk scoring (0–100%)
- Color-coded risk level (🟢 Low / 🟠 Moderate / 🔴 High)
- SHAP force plot explaining which factors drove the prediction for this patient
- PDF report export (patient details + risk + disclaimer)

### Model Performance Highlights
- Algorithm: Logistic Regression (selected for interpretability & balanced performance)
- AUC-ROC: [insert your value, e.g. 0.82]
- F1-Score: [insert value]
- Recall (positive class): [insert value – key for screening]
- Top predictors (by odds ratio):
**Top Predictors by Odds Ratio**

| Rank | Feature                  | Odds Ratio |
|------|--------------------------|------------|
| 1    | Race_Black               | 4.87       |
| 2    | Race_Indian              | 2.56       |
| 3    | Family_History           | 2.52       |
| 4    | Vitamin_D_Deficient      | 1.55       |
| 5    | BMI                      | 1.30       |
| ...  | Hypertension, PCOS, etc. | ...        |

### Data Generation
synthetic_data.py generates realistic synthetic records:

- Age distribution: 15–49
- Race: Weighted toward Kenyan context (higher Black & Indian)
- Probabilistic target using published risk factors

Output: synthetic_fibroids_data.csv

### Limitations
- Synthetic data only – real-world validation pending
- No imaging (ultrasound/MRI) features included
- Static SHAP plot may have minor label cutoff on narrow screens
- PDF uses basic Helvetica font

### Reproducing the Model
1. Open notebooks/UF_prevalence_prediction.ipynb
2. Run all cells sequentially (EDA → preprocessing → training → evaluation → saving)
3. Saved artifacts (uf_fibroids_final_model.pkl, uf_preprocessor.pkl) power the app

### Future Work Ideas
1. Partner with hospitals for real anonymized data
2. Add ultrasound radiomics or CNN features
3. Implement federated learning for privacy
4. Deploy on secure server with authentication
5. Build patient-facing RAG chatbot for fibroid education

#### Kibet W. (2026)
