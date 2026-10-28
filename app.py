import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

# ----------------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioSense AI — Clinical Heart Disease Risk Platform",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Modern Clinical Dark Theme Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-main: #090d16;
        --card-bg: rgba(18, 26, 45, 0.72);
        --card-border: rgba(255, 255, 255, 0.09);
        --card-hover: rgba(26, 38, 66, 0.9);
        --accent-cyan: #00f2fe;
        --accent-blue: #4facfe;
        --accent-pink: #ff4b72;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-primary);
    }

    .stApp {
        background: radial-gradient(circle at 15% 10%, #101c38 0%, #070b14 60%, #04060a 100%);
    }

    /* Top navigation header */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 1.4rem;
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }

    .brand-title {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .brand-title h2 {
        font-size: 1.4rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 30%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-subtitle {
        font-size: 0.78rem;
        color: #94a3b8;
    }

    .heart-pulse {
        font-size: 1.7rem;
        display: inline-block;
        animation: pulse 1.6s infinite ease-in-out;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.18); }
    }

    .top-nav-badges {
        display: flex;
        gap: 0.6rem;
        align-items: center;
    }

    .badge-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
    }

    .badge-chip-cyan {
        background: rgba(56, 189, 248, 0.12);
        border-color: rgba(56, 189, 248, 0.35);
        color: #38bdf8;
    }

    /* Mobile responsiveness for Top Navigation Header only */
    @media (max-width: 768px) {
        .top-nav {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.85rem;
            padding: 1rem 1.1rem;
        }

        .brand-title {
            width: 100%;
            gap: 0.75rem;
        }

        .brand-title h2 {
            font-size: 1.25rem;
            white-space: nowrap;
        }

        .brand-subtitle {
            font-size: 0.75rem;
            line-height: 1.35;
        }

        .top-nav-badges {
            width: 100%;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .badge-chip {
            font-size: 0.72rem;
            padding: 0.22rem 0.65rem;
            white-space: nowrap;
        }
    }

    /* Form section cards */
    .med-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        transition: border 0.2s ease, box-shadow 0.2s ease;
    }

    .med-card:hover {
        border-color: rgba(79, 172, 254, 0.3);
        box-shadow: 0 12px 35px rgba(0, 242, 254, 0.08);
    }

    .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.9rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 0.5rem;
    }

    .card-title {
        font-size: 0.98rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: #cbd5e1;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Result Displays */
    #prediction-result {
        scroll-margin-top: 4.5rem;
    }

    .result-container {
        border-radius: 20px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        margin-top: 1rem;
        border: 1px solid;
        animation: slideUp 0.4s ease-out;
    }

    .evaluated-glow {
        animation: cardPulse 0.8s ease-in-out;
    }

    @keyframes cardPulse {
        0% { transform: scale(0.98); box-shadow: 0 0 0 rgba(79, 172, 254, 0); }
        50% { transform: scale(1.02); box-shadow: 0 0 30px rgba(0, 242, 254, 0.45); }
        100% { transform: scale(1); }
    }

    .result-danger {
        background: radial-gradient(circle at 50% 0%, rgba(255, 75, 114, 0.25) 0%, rgba(26, 16, 28, 0.85) 100%);
        border-color: rgba(255, 75, 114, 0.45);
        box-shadow: 0 14px 40px rgba(255, 75, 114, 0.2);
    }

    .result-safe {
        background: radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.22) 0%, rgba(10, 28, 22, 0.85) 100%);
        border-color: rgba(16, 185, 129, 0.45);
        box-shadow: 0 14px 40px rgba(16, 185, 129, 0.18);
    }

    .prob-large {
        font-size: 3rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
        margin: 0.6rem 0;
    }

    .factor-pill {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    .factor-risk {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #fca5a5;
    }

    .factor-safe {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #6ee7b7;
    }

    /* Button Enhancements */
    .stButton>button, .stFormSubmitButton>button {
        width: 100%;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: #040915 !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 8px 24px rgba(0, 242, 254, 0.28) !important;
    }

    .stButton>button:hover, .stFormSubmitButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px rgba(0, 242, 254, 0.42) !important;
    }

    /* Preset Quick Action Pills */
    div[data-testid="stHorizontalBlock"] .stButton>button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #94a3b8 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        font-size: 0.82rem !important;
        padding: 0.4rem 0.8rem !important;
        box-shadow: none !important;
    }

    div[data-testid="stHorizontalBlock"] .stButton>button:hover {
        background: rgba(79, 172, 254, 0.18) !important;
        color: #38bdf8 !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
        transform: none !important;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Streamlit tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
        padding: 0.6rem 1rem;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    /* Footer styling */
    .clinical-footer {
        text-align: center;
        padding: 2rem 1rem 1rem;
        font-size: 0.8rem;
        color: #64748b;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Client-side Persistent Auto-Scroll Listener (triggers on every button click)
# ----------------------------------------------------------------------------
components.html("""
<script>
(function() {
    const parentDoc = window.parent.document;

    function scrollToPrediction() {
        try {
            const target = parentDoc.getElementById('prediction-result');
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        } catch(e) {
            console.warn("Scroll target error:", e);
        }
    }

    function attachScrollListeners() {
        try {
            const buttons = parentDoc.querySelectorAll('button');
            buttons.forEach(btn => {
                const text = btn.innerText || btn.textContent || "";
                if (text.includes('Run Clinical Stratification') || text.includes('Stratification')) {
                    if (!btn.dataset.hasCardioScroll) {
                        btn.dataset.hasCardioScroll = 'true';
                        btn.addEventListener('click', function() {
                            // 1. Immediate scroll upon click
                            setTimeout(scrollToPrediction, 50);
                            setTimeout(scrollToPrediction, 250);
                            // 2. Mark session storage to re-scroll after Streamlit completes rerun
                            try {
                                window.parent.sessionStorage.setItem('cardio_auto_scroll', 'true');
                            } catch(e) {}
                        });
                    }
                }
            });
        } catch(e) {}
    }

    // Check if coming back from form submission rerun
    try {
        if (window.parent.sessionStorage.getItem('cardio_auto_scroll') === 'true') {
            window.parent.sessionStorage.removeItem('cardio_auto_scroll');
            setTimeout(scrollToPrediction, 100);
            setTimeout(scrollToPrediction, 350);
            setTimeout(scrollToPrediction, 600);
        }
    } catch(e) {}

    attachScrollListeners();

    // Observe DOM mutations to ensure listener stays attached even if Streamlit re-renders buttons
    try {
        const observer = new MutationObserver(function() {
            attachScrollListeners();
            if (window.parent.sessionStorage.getItem('cardio_auto_scroll') === 'true') {
                window.parent.sessionStorage.removeItem('cardio_auto_scroll');
                setTimeout(scrollToPrediction, 120);
                setTimeout(scrollToPrediction, 400);
            }
        });
        observer.observe(parentDoc.body, { childList: true, subtree: true });
    } catch(e) {}
})();
</script>
""", height=0, width=0)

# ----------------------------------------------------------------------------
# Artifact Loader
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("LR_heart_Model.pkl")
    scaler = joblib.load("Scaler.pkl")
    columns = joblib.load("Columns.pkl")
    return model, scaler, columns

model, scaler, model_columns = load_artifacts()

# ----------------------------------------------------------------------------
# Session State for Quick Patient Presets
# ----------------------------------------------------------------------------
preset_defaults = {
    "age": 45,
    "sex": "Male",
    "resting_bp": 120,
    "cholesterol": 200,
    "max_hr": 150,
    "fasting_bs": "No",
    "oldpeak": 0.0,
    "chest_pain": "ATA (Atypical Angina)",
    "resting_ecg": "Normal",
    "exercise_angina": "No",
    "st_slope": "Up"
}

for key, val in preset_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ----------------------------------------------------------------------------
# Header Bar
# ----------------------------------------------------------------------------
st.markdown("""
<div class="top-nav">
    <div class="brand-title">
        <span class="heart-pulse">🫀</span>
        <div>
            <h2>CardioSense AI</h2>
            <div class="brand-subtitle">Clinical Decision Support & Risk Stratification System</div>
        </div>
    </div>
    <div class="top-nav-badges">
        <span class="badge-chip">● Model Calibrated</span>
        <span class="badge-chip badge-chip-cyan">11 Clinical Biomarkers</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Main Navigation Tabs
# ----------------------------------------------------------------------------
tab_assess, tab_metrics, tab_model = st.tabs([
    "🏥 Patient Risk Assessment",
    "📖 Clinical Biomarkers Guide",
    "🧠 Model Diagnostics & Weights"
])

# ============================================================================
# TAB 1: Patient Risk Assessment
# ============================================================================
with tab_assess:

    st.markdown("##### ⚡ Quick Load Clinical Profiles")
    preset_cols = st.columns([1, 1, 1, 3])
    
    with preset_cols[0]:
        if st.button("🟢 Healthy Athlete Profile"):
            st.session_state.update({
                "age": 28, "sex": "Female", "resting_bp": 110, "cholesterol": 175,
                "max_hr": 182, "fasting_bs": "No", "oldpeak": 0.0,
                "chest_pain": "ATA (Atypical Angina)", "resting_ecg": "Normal",
                "exercise_angina": "No", "st_slope": "Up"
            })
            st.rerun()

    with preset_cols[1]:
        if st.button("🔴 High Risk Cardiac Case"):
            st.session_state.update({
                "age": 62, "sex": "Male", "resting_bp": 155, "cholesterol": 290,
                "max_hr": 115, "fasting_bs": "Yes", "oldpeak": 2.5,
                "chest_pain": "ASY (Asymptomatic)", "resting_ecg": "ST (ST-T wave abnormality)",
                "exercise_angina": "Yes", "st_slope": "Flat"
            })
            st.rerun()

    with preset_cols[2]:
        if st.button("🔄 Reset to Default"):
            st.session_state.update(preset_defaults)
            st.rerun()

    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    # Assessment Form Layout
    col_input, col_output = st.columns([1.1, 0.9], gap="large")

    with col_input:
        with st.form("clinical_form"):

            # Section 1: Demographics
            st.markdown("""
            <div class="med-card">
                <div class="card-header">
                    <span class="card-title">👤 Patient Demographics</span>
                    <span style="font-size: 0.75rem; color: #64748b;">Step 1 of 3</span>
                </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Patient Age (Years)", min_value=18, max_value=100, value=st.session_state.age)
            with c2:
                sex = st.radio("Biological Sex", options=["Male", "Female"], 
                               index=0 if st.session_state.sex == "Male" else 1, horizontal=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Section 2: Hemodynamics
            st.markdown("""
            <div class="med-card">
                <div class="card-header">
                    <span class="card-title">🩺 Hemodynamic & Metabolic Panel</span>
                    <span style="font-size: 0.75rem; color: #64748b;">Step 2 of 3</span>
                </div>
            """, unsafe_allow_html=True)
            h1, h2 = st.columns(2)
            with h1:
                resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, 
                                             value=st.session_state.resting_bp, help="Target optimal: 90–120 mm Hg")
                cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, max_value=700, 
                                              value=st.session_state.cholesterol, help="Target optimal: < 200 mg/dL")
                max_hr = st.number_input("Maximum Heart Rate (MaxHR)", min_value=60, max_value=220, 
                                         value=st.session_state.max_hr, help="Estimated max target: 220 - Age")
            with h2:
                fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL?", options=["No", "Yes"], 
                                      index=0 if st.session_state.fasting_bs == "No" else 1, horizontal=True)
                oldpeak = st.number_input("ST Depression (Oldpeak in mm)", min_value=-3.0, max_value=7.0, 
                                          value=float(st.session_state.oldpeak), step=0.1, format="%.1f",
                                          help="Exercise-induced ST segment depression relative to rest")
            st.markdown("</div>", unsafe_allow_html=True)

            # Section 3: Electrocardiogram & Symptoms
            st.markdown("""
            <div class="med-card">
                <div class="card-header">
                    <span class="card-title">📋 Cardiac Symptoms & Stress ECG</span>
                    <span style="font-size: 0.75rem; color: #64748b;">Step 3 of 3</span>
                </div>
            """, unsafe_allow_html=True)
            e1, e2 = st.columns(2)
            with e1:
                chest_pain_opts = [
                    "ATA (Atypical Angina)", 
                    "NAP (Non-Anginal Pain)", 
                    "TA (Typical Angina)", 
                    "ASY (Asymptomatic)"
                ]
                chest_pain = st.selectbox(
                    "Chest Pain Type",
                    options=chest_pain_opts,
                    index=chest_pain_opts.index(st.session_state.chest_pain) if st.session_state.chest_pain in chest_pain_opts else 0
                )
                resting_ecg_opts = [
                    "Normal", 
                    "ST (ST-T wave abnormality)", 
                    "LVH (Left Ventricular Hypertrophy)"
                ]
                resting_ecg = st.selectbox(
                    "Resting ECG Assessment",
                    options=resting_ecg_opts,
                    index=resting_ecg_opts.index(st.session_state.resting_ecg) if st.session_state.resting_ecg in resting_ecg_opts else 0
                )
            with e2:
                exercise_angina = st.radio("Exercise-Induced Angina?", options=["No", "Yes"], 
                                           index=0 if st.session_state.exercise_angina == "No" else 1, horizontal=True)
                st_slope_opts = ["Up", "Flat", "Down"]
                st_slope = st.selectbox(
                    "Peak Exercise ST Slope",
                    options=st_slope_opts,
                    index=st_slope_opts.index(st.session_state.st_slope) if st.session_state.st_slope in st_slope_opts else 0,
                    help="Up = Normal cardiac recovery; Flat/Down = Potential ischemia"
                )
            st.markdown("</div>", unsafe_allow_html=True)

            evaluate_btn = st.form_submit_button("⚡ Run Clinical Stratification")

    # =========================================================================
    # Prediction Pipeline & Dynamic Results Column
    # =========================================================================
    with col_output:
        # Default or calculated evaluation
        chest_pain_code = chest_pain.split(" ")[0]
        resting_ecg_code = resting_ecg.split(" ")[0]
        sex_val = "M" if sex == "Male" else "F"
        fasting_bs_val = 1 if fasting_bs == "Yes" else 0
        exercise_angina_val = "Y" if exercise_angina == "Yes" else "N"

        # Handle zero BP/Cholesterol gracefully
        calc_bp = 132.9 if resting_bp <= 0 else resting_bp
        calc_chol = 246.6 if cholesterol <= 0 else cholesterol

        row = {
            "Age": age,
            "RestingBP": calc_bp,
            "Cholesterol": calc_chol,
            "FastingBS": fasting_bs_val,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "Sex_M": 1 if sex_val == "M" else 0,
            "ChestPainType_ATA": 1 if chest_pain_code == "ATA" else 0,
            "ChestPainType_NAP": 1 if chest_pain_code == "NAP" else 0,
            "ChestPainType_TA": 1 if chest_pain_code == "TA" else 0,
            "RestingECG_Normal": 1 if resting_ecg_code == "Normal" else 0,
            "RestingECG_ST": 1 if resting_ecg_code == "ST" else 0,
            "ExerciseAngina_Y": 1 if exercise_angina_val == "Y" else 0,
            "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
            "ST_Slope_Up": 1 if st_slope == "Up" else 0,
        }

        input_encoded = pd.DataFrame([row])[model_columns]
        input_scaled = scaler.transform(input_encoded)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        risk_pct = probability * 100

        # Risk Stratification Tier
        if risk_pct >= 60:
            tier_class = "result-danger"
            tier_badge = "🔴 HIGH PROBABILITY DETECTED"
            tier_color = "#ff4b72"
            tier_msg = "The model flags elevated likelihood of underlying coronary heart disease."
        elif risk_pct >= 30:
            tier_class = "result-danger"
            tier_badge = "🟡 BORDERLINE / MODERATE RISK"
            tier_color = "#f59e0b"
            tier_msg = "Intermediate risk profile. Further diagnostic workup advised."
        else:
            tier_class = "result-safe"
            tier_badge = "🟢 LOW RISK / FAVORABLE"
            tier_color = "#10b981"
            tier_msg = "Cardiovascular biomarkers fall within low risk baseline."

        # Risk Result Card
        glow_class = "evaluated-glow" if evaluate_btn else ""
        st.markdown(f"""
<div id="prediction-result" class="result-container {tier_class} {glow_class}">
    <div style="font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em; color: {tier_color};">
        {tier_badge}
    </div>
    <div class="prob-large" style="color: {tier_color};">
        {risk_pct:.1f}%
    </div>
    <div style="font-size: 0.88rem; color: #cbd5e1; margin-bottom: 0.8rem;">
        Estimated Probability of Coronary Artery Disease
    </div>
    <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
        {tier_msg}
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

        # Feature impact logic from coefficients
        risk_factors = []
        safe_factors = []

        if exercise_angina == "Yes":
            risk_factors.append("Exercise Angina Induced (+Risk)")
        else:
            safe_factors.append("No Exercise Angina (Protective)")

        if st_slope == "Flat":
            risk_factors.append("Flat ST Slope (Ischemic Indicator)")
        elif st_slope == "Up":
            safe_factors.append("Upsloping ST Segment (Normal Response)")

        if chest_pain_code == "ASY":
            risk_factors.append("Asymptomatic Presentation (Silent Ischemia Risk)")
        elif chest_pain_code in ["ATA", "NAP"]:
            safe_factors.append(f"{chest_pain_code} Chest Pain (Lower CAD Odds)")

        if oldpeak >= 1.0:
            risk_factors.append(f"ST Depression {oldpeak:.1f} mm (Significant)")
        else:
            safe_factors.append("ST Depression < 1.0 mm (Normal)")

        if fasting_bs == "Yes":
            risk_factors.append("Fasting Glucose > 120 mg/dL")

        if max_hr >= 145:
            safe_factors.append(f"Cardiovascular Reserve MaxHR {max_hr} bpm")
        elif max_hr < 120:
            risk_factors.append(f"Attenuated MaxHR {max_hr} bpm")

        rf_pills = "".join([f"<span class='factor-pill factor-risk'>▲ {rf}</span>" for rf in risk_factors]) if risk_factors else "<span style='font-size:0.8rem; color:#64748b;'>No critical acute risk flags detected.</span>"
        sf_pills = "".join([f"<span class='factor-pill factor-safe'>▼ {sf}</span>" for sf in safe_factors]) if safe_factors else "<span style='font-size:0.8rem; color:#64748b;'>None.</span>"

        st.markdown(f"""
<div class="med-card">
    <div class="card-header">
        <span class="card-title">🔍 Individual Biomarker Impact</span>
        <span style="font-size: 0.75rem; color: #64748b;">Key Contributors</span>
    </div>
    <div style="font-weight:600; font-size:0.85rem; margin-bottom:0.4rem; color:#f1f5f9;">Factors Driving Elevated Risk:</div>
    <div style="margin-bottom:0.8rem;">{rf_pills}</div>
    <div style="font-weight:600; font-size:0.85rem; margin-bottom:0.4rem; color:#f1f5f9;">Protective / Low-Risk Factors:</div>
    <div>{sf_pills}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

        # Clinical Recommendations Card (unindented HTML prevents markdown code block interpretation)
        if risk_pct >= 50:
            rec_html = """<ul style="font-size: 0.83rem; color: #cbd5e1; margin: 0; padding-left: 1.2rem; line-height: 1.6;">
<li><b>Priority Cardiology Review:</b> Schedule 12-lead resting ECG and stress echocardiography.</li>
<li><b>Lipid & Metabolic Re-evaluation:</b> Comprehensive panel (LDL, HDL, HbA1c, hs-CRP).</li>
<li><b>Lifestyle & Monitoring:</b> Ambulatory BP tracking and supervised exercise program.</li>
</ul>"""
        else:
            rec_html = """<ul style="font-size: 0.83rem; color: #cbd5e1; margin: 0; padding-left: 1.2rem; line-height: 1.6;">
<li><b>Routine Cardiovascular Screening:</b> Annual checkup with primary physician.</li>
<li><b>Primary Prevention:</b> Maintain Mediterranean diet and 150+ mins aerobic exercise/week.</li>
<li><b>Lipid Monitoring:</b> Repeat standard lipid profile every 1–2 years.</li>
</ul>"""

        st.markdown(f"""
<div class="med-card">
    <div class="card-header">
        <span class="card-title">📋 Clinical Recommendations</span>
        <span style="font-size: 0.75rem; color: #64748b;">Protocol</span>
    </div>
    {rec_html}
</div>
""", unsafe_allow_html=True)

        with st.expander("🔬 View Encoded Feature Vector for Debugging"):
            st.dataframe(input_encoded, use_container_width=True)

        # Auto-scroll smoothly to prediction result upon clicking "Run Clinical Stratification"
        if evaluate_btn:
            # Dynamic nonce ensures Streamlit renders a fresh iframe execution on every click
            nonce = int(time.time() * 1000)
            components.html(
                f"""
                <script>
                    (function() {{
                        // Nonce: {nonce}
                        function doScroll() {{
                            try {{
                                const elem = window.parent.document.getElementById('prediction-result');
                                if (elem) {{
                                    elem.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                                }}
                            }} catch (e) {{}}
                        }}
                        setTimeout(doScroll, 80);
                        setTimeout(doScroll, 250);
                        setTimeout(doScroll, 500);
                    }})();
                </script>
                """,
                height=0,
                width=0
            )



# ============================================================================
# TAB 2: Clinical Biomarkers Guide
# ============================================================================
with tab_metrics:
    st.markdown("### 🩺 Clinical Feature Reference Guide")
    st.markdown("Understanding the physiological meaning of the 11 clinical features measured in this platform:")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("""
        #### 1. Chest Pain Types (ChestPainType)
        - **TA (Typical Angina):** Substernal chest discomfort provoked by physical exertion or emotional stress, relieved by rest or nitroglycerin.
        - **ATA (Atypical Angina):** Discomfort meeting 2 of the 3 typical angina criteria. Frequently seen in female patients.
        - **NAP (Non-Anginal Pain):** Sharp or fleeting chest pain unlikely to stem from epicardial coronary narrowing.
        - **ASY (Asymptomatic):** Absence of acute angina; counter-intuitively associated with severe multi-vessel disease (silent ischemia).

        #### 2. ST Depression / Oldpeak
        - Measured in millimeters on exercise stress testing.
        - **Normal:** < 0.5 mm depression.
        - **Borderline:** 0.5 mm – 1.0 mm.
        - **Clinically Significant:** $\ge 1.0$ mm horizontal or downsloping ST depression strongly indicates exercise-induced myocardial ischemia.
        """)

    with g2:
        st.markdown("""
        #### 3. Peak Exercise ST Slope
        - **Up (Upsloping):** Rapid return of the ST segment to baseline. Normal healthy physiological response during peak heart rate.
        - **Flat (Horizontal):** ST segment stays flat below baseline. High predictive value for myocardial hypoperfusion.
        - **Down (Downsloping):** ST segment continues downward. Associated with multi-vessel disease and adverse cardiac outcomes.

        #### 4. MaxHR (Maximum Heart Rate Achieved)
        - In healthy individuals, the heart reaches estimated age-adjusted maximum ($220 - \text{Age}$).
        - **Chronotropic Incompetence:** Inability to achieve $\ge 85\%$ of age-predicted maximum heart rate during peak exercise is an independent marker of CAD and mortality.
        """)

# ============================================================================
# TAB 3: Model Diagnostics & Weights
# ============================================================================
with tab_model:
    st.markdown("### 🧠 Model Architecture & Learned Weights")
    st.markdown("""
    The deployed risk engine uses a **Regularized Logistic Regression classifier** trained on 918 patients 
    with a leak-free preprocessing pipeline (`StandardScaler` normalization).
    """)

    # Extract coefficients
    coef_df = pd.DataFrame({
        "Feature": model_columns,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    col_chart, col_stats = st.columns([1.2, 0.8])
    with col_chart:
        st.markdown("##### Learned Feature Impact on Log-Odds:")
        st.bar_chart(coef_df.set_index("Feature"))

    with col_stats:
        st.markdown("##### Validation Metrics (20% Holdout):")
        st.markdown("""
        - **Algorithm:** Logistic Regression (L2 Penalty)
        - **Classification Accuracy:** **86.96%**
        - **F1 Score:** **0.8846**
        - **Target Variable:** `HeartDisease` (0 = Normal, 1 = Disease)
        - **Features:** 15 one-hot encoded variables
        - **Latency:** < 5 ms inference speed
        """)

# ----------------------------------------------------------------------------
# Professional Footer
# ----------------------------------------------------------------------------
st.markdown("""
<div class="clinical-footer">
    <b>CardioSense AI — Clinical Machine Learning Interface</b> &nbsp;•&nbsp; 
    Powered by Scikit-Learn & Streamlit &nbsp;•&nbsp; 
    <i>Disclaimer: For educational, research, and portfolio demonstration purposes only. Not certified as a medical diagnostic device.</i>
</div>
""", unsafe_allow_html=True)
