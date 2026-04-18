# 🫀 Blood Pressure Prediction & Lifestyle Impact Analyzer

> An end-to-end machine learning application that predicts systolic and diastolic blood pressure from lifestyle and anthropometric features, quantifies the impact of behavioural interventions, and delivers AI-powered personalised health recommendations — built on real-world NHANES survey data.

---

## 📌 Overview

This project combines clinical epidemiology, machine learning, and interactive data visualization to create a tool that:

- **Predicts** a user's Systolic (SBP) and Diastolic (DBP) blood pressure from inputs such as age, BMI, waist circumference, sodium intake, physical activity, and sleep duration.
- **Simulates** the expected BP change after lifestyle modifications (what-if analysis).
- **Explains** which features drive predictions using SHAP importance.
- **Generates** personalised, plain-English health plans via Google Gemini AI.
- **Exports** a clinical-style PDF report of the session.

---

## 🗂️ Project Structure

```
my_mini_bp_project/
│
├── app.py                          # Streamlit web application (main entry point)
│
├── data/
│   ├── DEMO_L.xpt                  # NHANES demographics (age, gender)
│   ├── BPXO_L.xpt                  # Oscillometric blood pressure readings
│   ├── BMX_L.xpt                   # Body measurements (BMI, waist)
│   ├── DR1TOT_L.xpt                # 24-hour dietary recall (sodium, potassium)
│   ├── PAQ_L.xpt                   # Physical activity questionnaire
│   ├── SLQ_L.xpt                   # Sleep questionnaire
│   ├── BPQ_L.xpt                   # BP medication questionnaire
│   ├── processed_bp_full.csv       # Merged & engineered dataset (n ≈ 4,518)
│   └── processed_bp_lifestyle.csv  # Lifestyle-feature subset
│
├── notebooks/
│   ├── 01_load_data.py             # Raw XPT ingestion
│   ├── 02_feature_engineering.py   # Merging, cleaning, derived features
│   ├── 03_model_training.py        # XGBoost SBP model with hyperparameter search
│   ├── 04_model_training_DBP.py    # XGBoost DBP model
│   ├── 05_what_if_analysis.py      # Counterfactual lifestyle simulation
│   ├── 06_visualization.py         # Matplotlib / Plotly diagnostic plots
│   ├── bp_pipeline.py              # End-to-end retraining pipeline
│   └── eda.py                      # Exploratory data analysis
│
├── models/
│   ├── sbp_model.pkl               # Trained SBP XGBoost model
│   ├── dbp_model.pkl               # Trained DBP XGBoost model
│   └── *.png                       # Saved diagnostic figures (SHAP, calibration, etc.)
│
├── results/
│   ├── sbp.png                     # SBP prediction vs actual
│   └── dbp.png                     # DBP prediction vs actual
│
└── requirements.txt
```

---

## 🧠 Machine Learning Pipeline

### Data Source
All raw data is sourced from the **[NHANES 2021–2023 (Cycle L)](https://wwwn.cdc.gov/nchs/nhanes/)** public-use survey files published by the US Centers for Disease Control and Prevention (CDC).

### Feature Engineering
Features engineered from six NHANES modules:

| Feature | Source |
|---|---|
| Age, Gender | `DEMO_L` |
| BMI, Waist Circumference | `BMX_L` |
| Systolic / Diastolic BP (averaged over 3 readings) | `BPXO_L` |
| 24-h Sodium, Potassium intake | `DR1TOT_L` |
| Sedentary time (min/day) | `PAQ_L` |
| Sleep duration (hours) | `SLQ_L` |
| BP medication status | `BPQ_L` |

Polynomial and interaction terms added: `BMI²`, `Age²`, `BMI×Age`, `Waist×Age`, `log(Sodium)`, `Na/K ratio`, Waist-Hip Ratio (WHR).

### Model
- **Algorithm:** XGBoost Regressor (two separate models for SBP and DBP)
- **Regularization:** Heavily constrained (`max_depth=2`, `reg_alpha=5`, `reg_lambda=5`, `min_child_weight=10`) to prevent overfitting on tabular health data
- **Tuning:** `RandomizedSearchCV` over 60 iterations, 5-fold CV, scored on R²
- **Dataset:** n ≈ 4,518 (full cohort including medicated participants, with `OnMedication` as a feature)

### Model Performance (10-fold CV on full dataset)

| Target | CV R² | Test MAE |
|---|---|---|
| SBP | reported in training logs | see `results/sbp.png` |
| DBP | reported in training logs | see `results/dbp.png` |

> Med-naive subset (n=270) was evaluated separately for comparison; high CV variance confirmed the full-cohort model as the primary artefact.

---

## 🖥️ Application Features

The Streamlit app (`app.py`) is organized into three tabs:

### 1. 🔮 BP Prediction & What-If Analysis
- Sidebar inputs: age, gender, BMI, waist, sodium intake, physical activity, sleep, medication status
- Real-time SBP / DBP prediction with BP category classification (Normal / Elevated / Stage 1 / Stage 2 Hypertension / Crisis)
- **What-If Lifestyle Sliders:** simulate the expected BP reduction from changes in sodium intake, exercise, sleep, and weight — with a bar chart showing per-factor impact
- Optional logging of the current reading to session history

### 2. 📈 BP History & Trends
- Interactive Plotly chart of all readings logged during the session
- Trend lines for both SBP and DBP

### 3. 📚 Education
- DASH diet food-group table
- BP category reference guide
- General lifestyle guidance

### 🤖 AI-Powered Recommendations (Gemini)
After prediction, a **"Generate AI Health Plan"** button calls the **Google Gemini API** with the user's full profile and predicted BP values. The model returns a structured six-section clinical advisory covering risk context, prioritised interventions, dietary changes, activity plan, monitoring guidance, and red-flag warnings — written in plain English.

### 📄 PDF Export
Generates a formatted clinical report (patient summary, measurements, AI recommendations) via **ReportLab**.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A Google Gemini API key (free tier available at [aistudio.google.com](https://aistudio.google.com))

### Installation

```bash
git clone https://github.com/<your-username>/bp-prediction-app.git
cd bp-prediction-app

pip install -r requirements.txt
pip install xgboost google-generativeai reportlab
```

### Run the App

```bash
streamlit run app.py
```

> **Note:** Set your Gemini API key in `app.py` at the `GEMINI_API_KEY` constant, or export it as an environment variable:
> ```bash
> export GEMINI_API_KEY="your_key_here"
> ```

### Retrain Models (Optional)

```bash
python notebooks/02_feature_engineering.py   # regenerates processed CSVs
python notebooks/03_model_training.py        # trains & saves sbp_model.pkl
python notebooks/04_model_training_DBP.py    # trains & saves dbp_model.pkl
```

---

## 📦 Dependencies

```
streamlit
pandas
numpy
matplotlib
plotly
scikit-learn
xgboost
joblib
reportlab
google-generativeai
```

---

## ⚠️ Disclaimer

This application is intended for **educational and research purposes only**. Predictions are derived from population-level survey data and are not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider regarding blood pressure and cardiovascular health.

---

## 📊 Dataset Citation

> National Center for Health Statistics. *National Health and Nutrition Examination Survey, 2021–2023.* Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/nhanes/

---

## 🙋 Author

**[Sukumar Erigadindla]**  
[LinkedIn](https://linkedin.com/in/sukumar-erugadindla) · [GitHub](https://github.com/Sukumar5705)

---

## 📄 License

This project is released under the [MIT License](LICENSE).
