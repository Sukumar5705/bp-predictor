import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
from datetime import datetime

# ── Page config (must be first) ────────────────────────────────────────────
st.set_page_config(
    page_title="BP Prediction & Lifestyle Impact",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

import plotly.graph_objects as go
import plotly.express as px

# ── Optional imports ────────────────────────────────────────────────────────
try:
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                    Table, TableStyle)
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("ReportLab not installed. PDF export disabled. Run: pip install reportlab")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Gemini API key ──────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyAncporonzFYY2I_0xaPKaVlnp2w8VXmtc"

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .ai-card {
        background:#212222;
        border-left: 4px solid #9333ea;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .alert-critical { background:#fee2e2; border-left:4px solid #dc2626; padding:1rem; border-radius:8px; }
    .alert-warning  { background:#fef3c7; border-left:4px solid #f59e0b; padding:1rem; border-radius:8px; }
    .alert-good     { background:#d1fae5; border-left:4px solid #10b981; padding:1rem; border-radius:8px; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
for key, default in [
    ("results_ready", False),
    ("bp_history", []),
    ("gemini_configured", False),
    ("patient_name", ""),
    ("recommendations", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    sbp = joblib.load("models/sbp_model.pkl")
    dbp = joblib.load("models/dbp_model.pkl")
    return sbp, dbp

try:
    sbp_model, dbp_model = load_models()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    MODEL_ERROR = str(e)

# ── Feature engineering ──────────────────────────────────────────────────────
BASE_FEATURES = [
    'Age', 'Gender',
    'BMI', 'Waist',
    'BMI_sq', 'Age_sq',
    'BMI_Age', 'Waist_Age',
    'Age_BMI', 'Age_Waist', 'BMI_Waist',
    'log_sodium', 'Na_K_ratio',
    'WHR',
]
MODEL_FEATURES = BASE_FEATURES + ['OnMedication']


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sodium_map    = {0: 1500, 1: 2300, 2: 3500}
    potassium_map = {0: 3000, 1: 2500, 2: 2000}
    df['Sodium']    = df['DietCategory'].map(sodium_map).clip(500, 5000)
    df['Potassium'] = df['DietCategory'].map(potassium_map).clip(500, 5000)
    df['BMI_sq']     = df['BMI'] ** 2
    df['Age_sq']     = df['Age'] ** 2
    df['BMI_Age']    = df['BMI']   * df['Age']
    df['Waist_Age']  = df['Waist'] * df['Age']
    df['Age_BMI']    = df['Age']   * df['BMI']
    df['Age_Waist']  = df['Age']   * df['Waist']
    df['BMI_Waist']  = df['BMI']   * df['Waist']
    df['log_sodium'] = np.log1p(df['Sodium'])
    df['Na_K_ratio'] = df['Sodium'] / (df['Potassium'] + 1)
    df['WHR']        = df['Waist'] / df['BMI']
    return df[MODEL_FEATURES]


# ── Model metrics (v2 full-cohort) ───────────────────────────────────────────
sbp_test_r2, sbp_cv_r2, sbp_cv_std = 0.196, 0.214, 0.032
sbp_mae,     sbp_rmse               = 11.30, 14.82
dbp_test_r2, dbp_cv_r2, dbp_cv_std = 0.138, 0.152, 0.018
dbp_mae,     dbp_rmse               = 7.75,   9.93

# ── Gemini helpers ───────────────────────────────────────────────────────────
def configure_gemini() -> bool:
    if not GEMINI_AVAILABLE:
        return False
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        st.session_state.gemini_configured = True
        return True
    except Exception:
        st.session_state.gemini_configured = False
        return False

if GEMINI_AVAILABLE and not st.session_state.gemini_configured:
    configure_gemini()


def bp_category(sbp: float, dbp: float) -> str:
    if sbp >= 180 or dbp >= 120: return "Hypertensive Crisis"
    if sbp >= 140 or dbp >= 90:  return "Stage 2 Hypertension"
    if sbp >= 130 or dbp >= 80:  return "Stage 1 Hypertension"
    if sbp >= 120:                return "Elevated"
    return "Normal"


def generate_ai_recommendations(profile: dict, sbp_before: float, dbp_before: float,
                                  sbp_after: float, dbp_after: float) -> str:
    if not GEMINI_AVAILABLE:
        return "⚠️ Google Gemini not installed. Run: `pip install google-generativeai`"
    if not st.session_state.gemini_configured:
        return "⚠️ Gemini API not configured. Check your API key."

    cat_before = bp_category(sbp_before, dbp_before)
    cat_after  = bp_category(sbp_after,  dbp_after)

    prompt = def create_recommendation_prompt(profile, sbp_before, dbp_before, sbp_after, dbp_after):
    cat_before = bp_category(sbp_before, dbp_before)
    cat_after  = bp_category(sbp_after,  dbp_after)
    sbp_delta  = sbp_after - sbp_before
    dbp_delta  = dbp_after - dbp_before

    return f"""You are a clinical cardiovascular health advisor. Write in plain English that any adult can understand — no jargon. Be direct, specific, and use the patient's actual numbers throughout. Keep every section concise.

            ---
            PATIENT DATA

            Age: {profile.get('Age')} | Gender: {"Male" if profile.get('Gender',1)==1 else "Female"}
            BMI: {profile.get('BMI_before',25):.1f} → {profile.get('BMI_after',25):.1f}
            Waist: {profile.get('Waist_before',90):.1f} cm → {profile.get('Waist_after',90):.1f} cm
            Diet: {['Low Sodium','Moderate Sodium','High Sodium'][profile.get('Diet_before',1)]} → {['Low Sodium','Moderate Sodium','High Sodium'][profile.get('Diet_after',1)]}
            Activity: {profile.get('Activity_before',0.5):.1f} → {profile.get('Activity_after',0.5):.1f} (scale 0–2)
            Sleep: {['Poor','Good','Excessive'][profile.get('Sleep_before',1)]} → {['Poor','Good','Excessive'][profile.get('Sleep_after',1)]}

            BP Before: {sbp_before:.0f}/{dbp_before:.0f} mmHg ({cat_before})
            BP After:  {sbp_after:.0f}/{dbp_after:.0f} mmHg ({cat_after})
            Change:    {sbp_delta:+.0f}/{dbp_delta:+.0f} mmHg

            ---
            INSTRUCTIONS

            Write exactly 6 sections. No extra sections. Each must directly reference the patient's numbers above.

            ## 1. What Your BP Means Right Now
            In 3–4 sentences: explain what {sbp_before:.0f}/{dbp_before:.0f} mmHg ({cat_before}) means for their heart health. If the proposed changes bring it to {sbp_after:.0f}/{dbp_after:.0f} mmHg, say clearly whether that is a meaningful improvement and why.

            ## 2. Your 3 Most Important Changes (in order of impact)
            For each change, write:
            - What to do (one specific action, not vague advice)
            - Why it helps their BP (one sentence, mechanism in plain words)
            - How much BP reduction to expect (give a mmHg range from clinical evidence)
            - How long before they see results (realistic weeks/months)
            Base the order on which of their factors — BMI {profile.get('BMI_before',25):.1f}, diet ({['low','moderate','high'][profile.get('Diet_before',1)]} sodium), activity level {profile.get('Activity_before',0.5):.1f} — has the highest potential to lower their BP.

            ## 3. What to Eat This Week
            - Daily sodium target in mg (based on their current {['low','moderate','high'][profile.get('Diet_before',1)]} sodium diet)
            - 5 specific foods to add (with the reason each lowers BP — e.g. "spinach: high in potassium which helps kidneys remove sodium")
            - 5 specific foods to cut (with why each raises BP)
            - One sample day of meals (breakfast / lunch / dinner / snack) that fits the targets

            ## 4. Exercise Plan Starting This Week
            Based on their current activity level of {profile.get('Activity_before',0.5):.1f}/2.0:
            - Week 1–2: exact activity (type, duration, how many times per week)
            - Week 3–4: progression step
            - Month 2–3: goal state
            - One safety note specific to {cat_before}

            ## 5. Sleep & Stress — Quick Wins
            If their sleep is {['poor','good','excessive'][profile.get('Sleep_before',1)]}, give 3 specific changes they can make tonight. For stress, give one technique with step-by-step instructions (not just "try meditation").

            ## 6. When to Call a Doctor Immediately
            List exactly 4 warning signs specific to someone with {cat_before}. Use numbers where possible (e.g. "BP reading above 180/120"). End with one sentence on how often they should check their BP at home.

            ---
            ⚕️ Reminder: This is educational information only. Always follow the advice of your personal doctor or healthcare provider.
            """
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating recommendations: {e}"


# ── Recommendation engine (original) ─────────────────────────────────────────
def generate_recommendations(age, bmi, activity, sleep, diet, sbp, dbp):
    recs = []
    if sbp >= 140 or dbp >= 90:
        recs.append("High blood pressure detected. Lifestyle modification and medical consultation are recommended.")
    if bmi >= 25:
        recs.append("Consider weight reduction strategies to improve blood pressure control.")
    if activity < 1.0:
        recs.append("Increase physical activity to at least 150 minutes of moderate exercise per week.")
    if sleep == 0:
        recs.append("Improve sleep quality to 7–8 hours per night for better cardiovascular health.")
    if diet == 2:
        recs.append("Reduce sodium intake to help manage blood pressure.")
    if not recs:
        recs.append("Maintain your current healthy lifestyle habits to sustain optimal blood pressure.")
    return recs


def render_bp_history_chart():
    if len(st.session_state.bp_history) == 0:
        st.info("No BP readings logged yet. Add readings below to see trends.")
        return

    df = pd.DataFrame(st.session_state.bp_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['sbp'],
        mode='lines+markers', name='Systolic',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=7)
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['dbp'],
        mode='lines+markers', name='Diastolic',
        line=dict(color='#ef4444', width=2),
        marker=dict(size=7)
    ))
    fig.add_hline(y=140, line_dash="dash", line_color="#f97316", annotation_text="Stage 2 (140)")
    fig.add_hline(y=130, line_dash="dot",  line_color="#eab308", annotation_text="Stage 1 (130)")
    fig.add_hline(y=120, line_dash="dot",  line_color="#22c55e", annotation_text="Normal (120)")
    fig.update_layout(
        title="Blood Pressure Over Time",
        xaxis_title="Date / Time",
        yaxis_title="BP (mmHg)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg SBP",    f"{df['sbp'].mean():.1f} mmHg")
    col2.metric("Avg DBP",    f"{df['dbp'].mean():.1f} mmHg")
    col3.metric("SBP Range",  f"{df['sbp'].min():.0f}–{df['sbp'].max():.0f}")
    col4.metric("Readings",   str(len(df)))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                               APP LAYOUT                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

st.markdown('<h1 class="main-header">🫀 Blood Pressure Prediction & Lifestyle Impact</h1>',
            unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666;'>Predict BP change from lifestyle modifications · AI-powered recommendations · Trend monitoring</p>",
            unsafe_allow_html=True)

if not MODELS_LOADED:
    st.error(f"⚠️ Could not load ML models: {MODEL_ERROR}")
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_predict, tab_history, tab_education = st.tabs([
    "📊 Prediction & Analysis",
    "📈 BP History & Trends",
    "📚 Education"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction & Analysis
# ════════════════════════════════════════════════════════════════════════════════
with tab_predict:

    # ── About ────────────────────────────────────────────────────────────────
    with st.expander("🧠 About This Model", expanded=False):
        st.write(
            "This system uses XGBoost (v2, full NHANES cohort n=4,518, 10-fold CV) "
            "to predict systolic and diastolic blood pressure from lifestyle factors. "
            "After prediction, Google Gemini AI generates personalised health recommendations."
        )

    st.markdown("---")

    # ── Input columns ────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("### 👤 Current Lifestyle (Before)")
        patient_name = st.text_input("Patient Name")
        age  = st.number_input("Age", 18, 80, 45)

        height_cm = st.number_input("Height (cm)", 120.0, 220.0, 170.0)
        weight_kg = st.number_input("Weight (kg)", 30.0, 150.0, 70.0)
        height_m  = height_cm / 100
        bmi       = round(weight_kg / (height_m ** 2), 2)
        st.info(f"**Calculated BMI:** {bmi}")

        waist = st.number_input("Waist Circumference (cm)", 60.0, 140.0, 100.0)
        st.caption("💡 Measure at navel level.")

        diet = st.selectbox(
            "Diet Type",
            options=[0, 1, 2],
            format_func=lambda x: ["🥗 Low Salt", "🍽️ Moderate Salt", "🧂 High Salt"][x]
        )
        activity = st.slider("Physical Activity Level", 0.0, 2.0, 0.5,
                              help="0 = sedentary, 1 = moderate (150 min/wk), 2 = very active")
        sleep = st.selectbox(
            "Sleep Quality",
            options=[0, 1, 2],
            format_func=lambda x: ["😴 Poor", "😊 Good", "🛌 Excessive"][x]
        )

    with col_right:
        st.markdown("### 🎯 Lifestyle After Change")

        weight_new = st.number_input("Weight after change (kg)", 30.0, 150.0, 65.0)
        bmi_new    = round(weight_new / (height_m ** 2), 2)
        st.info(f"**BMI After:** {bmi_new}")

        waist_new = st.number_input("Waist after change (cm)", 60.0, 140.0, 94.0)

        diet_new = st.selectbox(
            "Diet after change",
            options=[0, 1, 2],
            index=0,
            format_func=lambda x: ["🥗 Low Salt", "🍽️ Moderate Salt", "🧂 High Salt"][x]
        )
        activity_new = st.slider("Activity after change", 0.0, 2.0, 1.2)
        sleep_new = st.selectbox(
            "Sleep after change",
            options=[0, 1, 2],
            index=1,
            format_func=lambda x: ["😴 Poor", "😊 Good", "🛌 Excessive"][x]
        )

        # Optional: log a current reading to history
        st.markdown("#### 📝 Log Today's BP Reading")
        c1, c2, c3 = st.columns(3)
        with c1:
            log_sbp = st.number_input("Systolic", 70, 250, 120, key="log_sbp")
        with c2:
            log_dbp = st.number_input("Diastolic", 40, 150, 80, key="log_dbp")
        with c3:
            st.write("")
            st.write("")
            if st.button("➕ Add Reading"):
                st.session_state.bp_history.append({
                    'sbp': log_sbp, 'dbp': log_dbp,
                    'timestamp': datetime.now().isoformat()
                })
                st.success("Reading logged!")

    # ── Predict button ───────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔮 Predict BP Change", type="primary"):
        if not MODELS_LOADED:
            st.error("Models not loaded.")
        else:
            st.session_state.results_ready = True
            st.session_state.patient_name  = patient_name

            before_raw = pd.DataFrame([{
                'Age': age, 'Gender': 1,
                'BMI': bmi, 'Waist': waist,
                'DietCategory': diet, 'OnMedication': 0
            }])
            after_raw = pd.DataFrame([{
                'Age': age, 'Gender': 1,
                'BMI': bmi_new, 'Waist': waist_new,
                'DietCategory': diet_new, 'OnMedication': 0
            }])

            before_feat = transform_features(before_raw)
            after_feat  = transform_features(after_raw)

            st.session_state.before = before_feat
            st.session_state.after  = after_feat
            st.session_state.before_raw_base = {
                'Age': age, 'Gender': 1,
                'BMI': bmi, 'Waist': waist,
                'DietCategory': diet, 'OnMedication': 0
            }

            st.session_state.sbp_before = float(sbp_model.predict(before_feat)[0])
            st.session_state.sbp_after  = float(sbp_model.predict(after_feat)[0])
            st.session_state.dbp_before = float(dbp_model.predict(before_feat)[0])
            st.session_state.dbp_after  = float(dbp_model.predict(after_feat)[0])

            # Store profile for Gemini
            st.session_state.gemini_profile = {
                'Age': age, 'Gender': 1,
                'BMI_before': bmi,     'BMI_after': bmi_new,
                'Waist_before': waist, 'Waist_after': waist_new,
                'Diet_before': diet,   'Diet_after': diet_new,
                'Activity_before': activity, 'Activity_after': activity_new,
                'Sleep_before': sleep, 'Sleep_after': sleep_new,
            }

    # ════════════════════════════════════════════════════════════════════════
    # RESULTS SECTION
    # ════════════════════════════════════════════════════════════════════════
    if st.session_state.results_ready:
        sbp_before = st.session_state.sbp_before
        sbp_after  = st.session_state.sbp_after
        dbp_before = st.session_state.dbp_before
        dbp_after  = st.session_state.dbp_after
        before     = st.session_state.before
        after      = st.session_state.after

        st.markdown("---")
        st.markdown("## 📊 Prediction Results")

        # ── BP Summary metrics ───────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Before Changes")
            st.metric("Systolic BP",  f"{sbp_before:.1f} mmHg")
            st.metric("Diastolic BP", f"{dbp_before:.1f} mmHg")
            st.caption(f"Category: **{bp_category(sbp_before, dbp_before)}**")
        with col2:
            st.markdown("#### After Changes")
            st.metric("Systolic BP",  f"{sbp_after:.1f} mmHg",
                      delta=f"{sbp_after - sbp_before:+.1f}")
            st.metric("Diastolic BP", f"{dbp_after:.1f} mmHg",
                      delta=f"{dbp_after - dbp_before:+.1f}")
            st.caption(f"Category: **{bp_category(sbp_after, dbp_after)}**")
        with col3:
            st.markdown("#### Risk Level")
            if sbp_after >= 130 or dbp_after >= 85:
                st.markdown('<div class="alert-critical">🚨 <b>High Risk</b><br>Consult a doctor.</div>',
                            unsafe_allow_html=True)
            elif sbp_after >= 120 or dbp_after >= 80:
                st.markdown('<div class="alert-warning">⚠️ <b>Elevated</b><br>Lifestyle changes needed.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-good">✅ <b>Normal</b><br>Maintain habits.</div>',
                            unsafe_allow_html=True)

        # ── Before / After bar chart ─────────────────────────────────────
        st.markdown("### 📉 Before vs After Comparison")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=['Before', 'After'], y=[sbp_before, sbp_after],
            name='Systolic', marker_color='#3b82f6'
        ))
        fig_bar.add_trace(go.Bar(
            x=['Before', 'After'], y=[dbp_before, dbp_after],
            name='Diastolic', marker_color='#ef4444'
        ))
        fig_bar.add_hline(y=140, line_dash="dash", line_color="#f97316",
                          annotation_text="Stage 2 Threshold (140)")
        fig_bar.add_hline(y=120, line_dash="dot",  line_color="#22c55e",
                          annotation_text="Normal Threshold (120)")
        fig_bar.update_layout(
            barmode='group', height=380,
            yaxis_title="Blood Pressure (mmHg)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── BP History trend chart (inline) ──────────────────────────────
        if len(st.session_state.bp_history) > 0:
            st.markdown("### 📈 Your BP Trend Over Time")
            render_bp_history_chart()

        # ── Model performance ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📉 Model Performance (v2 — Full NHANES cohort, n=4,518)")

        col_s, col_d = st.columns(2)
        with col_s:
            st.markdown("**SBP Model**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Test R²",  f"{sbp_test_r2:.3f}")
            m2.metric("CV R²",    f"{sbp_cv_r2:.3f} ± {sbp_cv_std:.3f}")
            m3.metric("MAE",      f"{sbp_mae:.2f}")
            m4.metric("RMSE",     f"{sbp_rmse:.2f}")
        with col_d:
            st.markdown("**DBP Model**")
            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Test R²",  f"{dbp_test_r2:.3f}")
            m6.metric("CV R²",    f"{dbp_cv_r2:.3f} ± {dbp_cv_std:.3f}")
            m7.metric("MAE",      f"{dbp_mae:.2f}")
            m8.metric("RMSE",     f"{dbp_rmse:.2f}")

        st.caption("CV R² computed with 10-fold cross-validation on full NHANES cohort "
                   "(resolves the negative CV R² from the earlier n=270 pipeline).")

        # ── Interpretation & Insights ────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 Interpretation & Insights")

        bmi_change      = bmi - bmi_new
        activity_change = activity_new - activity
        diet_change     = diet - diet_new
        sleep_change    = sleep_new - sleep

        insights = []
        if diet_change > 0:
            insights.append("✅ Reduced dietary salt has contributed positively to BP improvement.")
        elif diet_change < 0:
            insights.append("⚠️ Increased dietary salt may negatively affect blood pressure.")
        if bmi_change > 0:
            insights.append("✅ Weight reduction supports better blood pressure control.")
        elif bmi_change < 0:
            insights.append("⚠️ Weight gain may raise blood pressure.")
        if activity_change > 0.3:
            insights.append("✅ Increased physical activity shows a positive impact on BP.")
        elif activity_change <= 0:
            insights.append("⚠️ Low or reduced activity limits BP improvement.")
        if sleep_change > 0:
            insights.append("✅ Improved sleep quality supports healthier BP regulation.")
        elif sleep_change < 0:
            insights.append("⚠️ Poorer sleep quality can negatively influence BP.")

        for ins in insights:
            st.write(ins)
        st.info("**Overall:** Lifestyle modifications collectively influence BP, with dietary "
                "and weight-related changes often showing stronger model-level impact.")

        # ── Basic recommendations ────────────────────────────────────────
        recs = generate_recommendations(age, bmi_new, activity_new, sleep_new,
                                         diet_new, sbp_after, dbp_after)
        st.session_state.recommendations = recs

        st.markdown("### 🩺 Personalised Lifestyle Recommendations")
        for r in recs:
            st.write(f"• {r}")

        # ── What-if analysis ─────────────────────────────────────────────
        st.markdown("---")
        show_impact = st.checkbox("🔎 Show which lifestyle change had the most impact")
        if show_impact:
            st.markdown("### 🧪 Impact of Individual Lifestyle Changes")
            base = st.session_state.before_raw_base
            effects = {}
            diet_v   = pd.DataFrame([{**base, 'DietCategory': diet_new}])
            weight_v = pd.DataFrame([{**base, 'BMI': bmi_new, 'Waist': waist_new}])
            effects['Diet Change']   = float(sbp_model.predict(transform_features(diet_v))[0]   - sbp_before)
            effects['Weight & Waist Change'] = float(sbp_model.predict(transform_features(weight_v))[0] - sbp_before)
            effects['Activity Change (recommendation only)'] = 0.0
            effects['Sleep Change (recommendation only)']    = 0.0

            for k, v in sorted(effects.items(), key=lambda x: x[1]):
                color = "🟢" if v < 0 else ("🔴" if v > 0 else "⚪")
                st.write(f"{color} **{k}** → {v:+.2f} mmHg")

            model_e = {k: v for k, v in effects.items() if 'only' not in k}
            if model_e:
                best = min(model_e, key=model_e.get)
                st.success(f"✅ Most impactful: **{best}**")

            # Scenario bar
            fig_eff = px.bar(
                x=list(effects.keys()), y=list(effects.values()),
                labels={'x': 'Lifestyle Factor', 'y': 'SBP Change (mmHg)'},
                title="Individual Factor Impact on Systolic BP",
                color=list(effects.values()),
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_eff, use_container_width=True)

        # ── Feature importance ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Feature Importance (Model Insight)")
        importance = sbp_model.feature_importances_
        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
        ax_imp.barh(MODEL_FEATURES, importance, color='#667eea')
        ax_imp.set_title("Feature Importance for SBP Prediction (15 features incl. OnMedication)")
        ax_imp.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig_imp)

        # ════════════════════════════════════════════════════════════════
        # 🤖  GEMINI AI RECOMMENDATIONS  (featured after prediction)
        # ════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("## 🤖 AI-Powered Personalised Recommendations")
        st.markdown(
            "_Powered by Google Gemini — analyses your full profile and BP predictions "
            "to generate a tailored health plan._"
        )

        if not GEMINI_AVAILABLE:
            st.error("Google Gemini not installed. Run: `pip install google-generativeai`")
        elif not st.session_state.gemini_configured:
            st.warning("Gemini API not configured. Check `GEMINI_API_KEY` in the code.")
        else:
            if st.button("✨ Generate AI Health Plan", type="primary", key="ai_btn"):
                with st.spinner("Gemini AI is analysing your data and creating a personalised plan… (~30s)"):
                    ai_text = generate_ai_recommendations(
                        st.session_state.gemini_profile,
                        sbp_before, dbp_before,
                        sbp_after, dbp_after
                    )
                    st.markdown(
                        f'<div class="ai-card">{ai_text}</div>',
                        unsafe_allow_html=True
                    )

        # ── PDF report ───────────────────────────────────────────────────
        st.markdown("---")
        if REPORTLAB_AVAILABLE:
            if st.button("📄 Download PDF Report"):
                if not os.path.exists("results"):
                    os.makedirs("results")

                # Save matplotlib charts for PDF
                fig1, ax1 = plt.subplots()
                ax1.bar(["Before", "After"], [sbp_before, sbp_after], color=['#3b82f6','#60a5fa'])
                ax1.set_title("SBP Before vs After"); ax1.set_ylabel("mmHg")
                sbp_path = "results/sbp.png"; fig1.savefig(sbp_path, bbox_inches='tight')

                fig2, ax2 = plt.subplots()
                ax2.bar(["Before", "After"], [dbp_before, dbp_after], color=['#ef4444','#f87171'])
                ax2.set_title("DBP Before vs After"); ax2.set_ylabel("mmHg")
                dbp_path = "results/dbp.png"; fig2.savefig(dbp_path, bbox_inches='tight')

                pdf_path = "results/BP_Lifestyle_Report.pdf"
                styles = getSampleStyleSheet()
                story  = []

                story.append(Paragraph(
                    "<font color='darkblue'><b>🏥 Lifestyle Health Assessment Report</b></font>",
                    styles['Title']
                ))
                story.append(Spacer(1, 12))
                story.append(Paragraph(
                    "<font color='grey'><b>Department:</b> Preventive Cardiology</font>",
                    styles['Normal']
                ))
                story.append(Spacer(1, 12))

                pt_data = [
                    ["Patient Name", st.session_state.get("patient_name", "N/A")],
                    ["Age", str(age)],
                    ["Date", datetime.now().strftime("%Y-%m-%d")]
                ]
                pt_table = Table(pt_data, colWidths=[150, 250])
                pt_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                story.append(Paragraph("<b>Patient Information</b>", styles['Heading2']))
                story.append(Spacer(1, 8)); story.append(pt_table); story.append(Spacer(1, 15))

                meas_data = [
                    ["Metric", "Before", "After"],
                    ["Systolic BP",  f"{sbp_before:.1f}", f"{sbp_after:.1f}"],
                    ["Diastolic BP", f"{dbp_before:.1f}", f"{dbp_after:.1f}"]
                ]
                meas_table = Table(meas_data, colWidths=[150, 100, 100])
                meas_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                story.append(Paragraph("<b>Clinical Measurements</b>", styles['Heading2']))
                story.append(Spacer(1, 8)); story.append(meas_table); story.append(Spacer(1, 15))

                if sbp_after >= 130 or dbp_after >= 85:
                    risk_label, risk_score, rc = "High Risk", 85, "red"
                elif sbp_after >= 120 or dbp_after >= 80:
                    risk_label, risk_score, rc = "Elevated", 60, "orange"
                else:
                    risk_label, risk_score, rc = "Normal", 20, "green"

                story.append(Paragraph(
                    f"<font color='{rc}'><b>Risk Level: {risk_label}</b></font>",
                    styles['Heading2']
                ))
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>Risk Score:</b> {risk_score}%", styles['Normal']))
                story.append(Spacer(1, 15))

                story.append(Paragraph("<b>Blood Pressure Trends</b>", styles['Heading2']))
                story.append(Spacer(1, 8))
                story.append(Image(sbp_path, width=400, height=250))
                story.append(Spacer(1, 10))
                story.append(Image(dbp_path, width=400, height=250))
                story.append(Spacer(1, 15))

                story.append(Paragraph(
                    "<font color='darkgreen'><b>Medical Recommendations</b></font>",
                    styles['Heading2']
                ))
                story.append(Spacer(1, 8))
                for r in st.session_state.recommendations:
                    story.append(Paragraph(f"• {r}", styles['Normal']))
                    story.append(Spacer(1, 5))
                story.append(Spacer(1, 15))

                story.append(Paragraph(
                    "<font color='darkred'><b>Medical Advisory</b></font>",
                    styles['Heading2']
                ))
                story.append(Spacer(1, 8))
                advisory = ("Patient shows high risk of hypertension. Immediate medical consultation is strongly advised."
                            if risk_label == "High Risk"
                            else "Patient is within manageable BP range. Maintain healthy lifestyle habits.")
                story.append(Paragraph(advisory, styles['Normal']))
                story.append(Spacer(1, 15))
                story.append(Paragraph(
                    "<i>This report is for decision support only and does not replace professional medical diagnosis.</i>",
                    styles['Italic']
                ))

                doc = SimpleDocTemplate(pdf_path)
                doc.build(story)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Click to Download Report",
                        data=f,
                        file_name="BP_Lifestyle_Report.pdf",
                        mime="application/pdf"
                    )
        else:
            st.info("Install `reportlab` to enable PDF export.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — BP History & Trends
# ════════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("## 📈 Blood Pressure History & Trends")

    # Manual add
    st.markdown("### ➕ Log a Reading")
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        h_sbp = st.number_input("Systolic (mmHg)", 70, 250, 120, key="h_sbp")
    with c2:
        h_dbp = st.number_input("Diastolic (mmHg)", 40, 150, 80, key="h_dbp")
    with c3:
        h_date = st.date_input("Date", datetime.now().date(), key="h_date")
    with c4:
        h_time = st.time_input("Time", datetime.now().time(), key="h_time")
        if st.button("Add", key="add_hist"):
            ts = datetime.combine(h_date, h_time).isoformat()
            st.session_state.bp_history.append({'sbp': h_sbp, 'dbp': h_dbp, 'timestamp': ts})
            st.success("Added!")

    st.markdown("---")
    render_bp_history_chart()

    if len(st.session_state.bp_history) > 0:
        # Table
        df_h = pd.DataFrame(st.session_state.bp_history)
        df_h['timestamp'] = pd.to_datetime(df_h['timestamp'])
        df_disp = df_h[['timestamp', 'sbp', 'dbp']].copy()
        df_disp['timestamp'] = df_disp['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        df_disp.columns = ['Date & Time', 'Systolic', 'Diastolic']
        st.dataframe(df_disp.sort_values('Date & Time', ascending=False), use_container_width=True)

        csv = df_disp.to_csv(index=False)
        st.download_button("⬇️ Download History as CSV", csv,
                           "bp_history.csv", "text/csv")

        if st.button("🗑️ Clear History"):
            st.session_state.bp_history = []
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Education
# ════════════════════════════════════════════════════════════════════════════════
with tab_education:
    st.markdown("## 📚 Blood Pressure Education")

    st.markdown("### 🩺 Understanding Blood Pressure")
    st.write("""
    Blood pressure is measured in mmHg:
    - **Systolic (top number):** Pressure when the heart beats
    - **Diastolic (bottom number):** Pressure when the heart rests between beats
    """)

    st.markdown("### 📊 BP Categories (AHA 2017 Guidelines)")
    cats = pd.DataFrame({
        'Category':  ['Normal', 'Elevated', 'Stage 1 Hypertension', 'Stage 2 Hypertension', 'Hypertensive Crisis'],
        'Systolic':  ['< 120', '120–129', '130–139', '≥ 140', '≥ 180'],
        'Diastolic': ['< 80',  '< 80',    '80–89',   '≥ 90',  '≥ 120'],
        'Action': [
            'Maintain healthy lifestyle',
            'Lifestyle changes recommended',
            'Lifestyle changes + possible medication',
            'Medication + lifestyle changes',
            'Seek emergency medical care immediately'
        ]
    })
    st.dataframe(cats, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ⚠️ Risk Factors")
        st.markdown("**Modifiable:** High sodium diet · Obesity · Physical inactivity · Smoking · Chronic stress")
        st.markdown("**Non-modifiable:** Age (>65) · Family history · Chronic disease · Race/ethnicity")
    with c2:
        st.markdown("### 🥗 DASH Diet Basics")
        dash = pd.DataFrame({
            'Food Group': ['Vegetables', 'Fruits', 'Whole Grains', 'Low-fat Dairy', 'Lean Meat', 'Nuts/Legumes'],
            'Servings/Day': ['4–5', '4–5', '6–8', '2–3', '≤ 6 oz', '4–5/week'],
        })
        st.dataframe(dash, use_container_width=True)

    st.markdown("### 📏 Proper Measurement Tips")
    st.write("""
    1. No caffeine or exercise 30 min before  
    2. Empty bladder; sit quietly 5 minutes  
    3. Back supported, feet flat on floor  
    4. Upper arm at heart level  
    5. Don't talk during measurement  
    6. Take 2–3 readings 1 min apart and average them  
    7. Measure at the same time each day
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#888;padding:0.5rem;font-size:0.85rem;'>
    ⚕️ <b>Medical Disclaimer:</b> For educational purposes only. Not a substitute for professional medical advice.<br>
    🤖 <b>AI Disclaimer:</b> AI recommendations should be verified with your healthcare provider.
</div>
""", unsafe_allow_html=True)