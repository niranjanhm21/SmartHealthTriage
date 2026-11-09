"""
SmartHealth Triage: Explainable Decision-Support & Risk Stratification Dashboard.
Clinical Informatics Web Application.
"""

import os
import sys
from typing import List, Dict, Any

# Ensure project root is on Python sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from src.inference import (
    predict_triage_assessment,
    get_available_symptoms,
    get_emergency_red_flags
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartHealth Triage | Clinical Decision-Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM MEDICAL-INFORMATICS STYLING (CSS)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Base background */
.stApp {
    background-color: #0c1222;
    color: #e2e8f0;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}

/* Header Container */
.header-box {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}

.header-title {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(120deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    padding-bottom: 6px;
}

.header-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
}

/* Disclaimer Banner */
.disclaimer-banner {
    background: linear-gradient(90deg, rgba(234, 179, 8, 0.12) 0%, rgba(245, 158, 11, 0.08) 100%);
    border-left: 4px solid #f59e0b;
    border-top: 1px solid rgba(245, 158, 11, 0.25);
    border-right: 1px solid rgba(245, 158, 11, 0.25);
    border-bottom: 1px solid rgba(245, 158, 11, 0.25);
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 24px;
    font-size: 0.88rem;
    color: #fef08a;
    display: flex;
    align-items: center;
    gap: 12px;
}

/* Triage Urgency Badges */
.triage-badge-container {
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.triage-badge-low {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.1) 100%);
    border: 1px solid #10b981;
}

.triage-badge-moderate {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.1) 100%);
    border: 1px solid #f59e0b;
}

.triage-badge-high {
    background: linear-gradient(135deg, rgba(249, 115, 22, 0.15) 0%, rgba(234, 88, 12, 0.1) 100%);
    border: 1px solid #f97316;
}

.triage-badge-emergency {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(185, 28, 28, 0.2) 100%);
    border: 2px solid #ef4444;
    animation: pulse-emergency 2s infinite;
}

@keyframes pulse-emergency {
    0% {
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6);
    }
    70% {
        box-shadow: 0 0 0 16px rgba(239, 68, 68, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);
    }
}

.triage-pill {
    display: inline-block;
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 9999px;
    width: fit-content;
}

.pill-low {
    background-color: #064e3b;
    color: #6ee7b7;
    border: 1px solid #10b981;
}

.pill-moderate {
    background-color: #451a03;
    color: #fcd34d;
    border: 1px solid #f59e0b;
}

.pill-high {
    background-color: #431407;
    color: #fdba74;
    border: 1px solid #f97316;
}

.pill-emergency {
    background-color: #7f1d1d;
    color: #fca5a5;
    border: 1px solid #ef4444;
}

/* Safety Override High-Visibility Alert */
.override-alert-box {
    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
    border: 2px solid #ef4444;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 24px;
    color: #ffffff;
    box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.4);
    animation: pulse-emergency 2.5s infinite;
}

.override-title {
    font-size: 1.15rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    gap: 10px;
    color: #fee2e2;
    margin-bottom: 8px;
}

/* Card components */
.clinical-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
}

.condition-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}

.condition-card:hover {
    border-color: #38bdf8;
    background: rgba(30, 41, 59, 0.95);
    transform: translateY(-1px);
}

.condition-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f8fafc;
    text-transform: capitalize;
}

.patient-stat {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}

.patient-stat-num {
    font-size: 1.25rem;
    font-weight: 700;
    color: #38bdf8;
}

.patient-stat-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Custom scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0f172a;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #475569;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. CLINICAL RED-FLAG DEFINITIONS
# -----------------------------------------------------------------------------
RED_FLAG_OPTIONS = [
    {
        "id": "severe_chest_pain",
        "label": "Severe Crushing Chest Pain / Pressure",
        "description": "Pain radiating to jaw, neck, shoulder, or left arm (suspected ACS / acute MI)."
    },
    {
        "id": "difficulty_breathing",
        "label": "Severe Shortness of Breath at Rest",
        "description": "Respiratory exhaustion, inability to speak complete sentences, stridor."
    },
    {
        "id": "hemoptysis",
        "label": "Coughing up Blood / Coffee-Ground Vomitus",
        "description": "Active pulmonary hemorrhage or acute upper GI bleeding."
    },
    {
        "id": "sudden_paralysis",
        "label": "Sudden Weakness / Facial Droop / Paralysis",
        "description": "Acute focal neurological deficit or hemiparesis (suspected stroke / CVA)."
    },
    {
        "id": "loss_of_consciousness",
        "label": "Syncope / Sudden Fainting / Unresponsiveness",
        "description": "Transient loss of consciousness or acute stupor / mental confusion."
    },
    {
        "id": "cyanosis",
        "label": "Cyanosis (Bluish Lips / Fingers / Face)",
        "description": "Severe arterial hypoxemia or peripheral hypoperfusion."
    },
    {
        "id": "severe_abdominal_trauma",
        "label": "Rigid Board-Like Abdomen / Severe Agony",
        "description": "Acute peritonitis, ruptured viscus, or internal hemorrhage."
    }
]


# -----------------------------------------------------------------------------
# 4. SIDEBAR CLINICAL INTAKE CONTROLS
# -----------------------------------------------------------------------------
available_symptoms = get_available_symptoms()
available_symptoms_sorted = sorted(available_symptoms)

with st.sidebar:
    st.markdown("### 📋 Clinical Intake Form")
    st.caption("Enter patient demographics, reported complaints, and red flags.")

    # Preset Quick-Load Scenarios with persistent session state synchronization
    st.markdown("##### ⚡ Quick Clinical Presets")
    
    if "symptoms_preset" not in st.session_state:
        st.session_state["symptoms_preset"] = [s for s in ["cough", "fever"] if s in available_symptoms_sorted]

    col_pre1, col_pre2 = st.columns(2)
    if col_pre1.button("🫀 Chest Agony", use_container_width=True, help="Load acute cardiac symptoms"):
        cardiac_syms = ["sharp chest pain", "chest tightness", "palpitations", "shortness of breath", "sweating"]
        st.session_state["symptoms_preset"] = [s for s in cardiac_syms if s in available_symptoms]
        for rf in RED_FLAG_OPTIONS:
            st.session_state[f"rf_{rf['id']}"] = (rf["id"] == "severe_chest_pain")
        st.rerun()

    if col_pre2.button("🫁 Pneumonia", use_container_width=True, help="Load respiratory symptoms"):
        pneu_syms = ["cough", "shortness of breath", "fever", "sharp chest pain", "chills"]
        st.session_state["symptoms_preset"] = [s for s in pneu_syms if s in available_symptoms]
        for rf in RED_FLAG_OPTIONS:
            st.session_state[f"rf_{rf['id']}"] = False
        st.rerun()
        
    col_pre3, col_pre4 = st.columns(2)
    if col_pre3.button("🤧 Mild Allergy", use_container_width=True, help="Load allergic rhinitis symptoms"):
        allergy_syms = ["nasal congestion", "sneezing", "cough", "itchy eyes"]
        st.session_state["symptoms_preset"] = [s for s in allergy_syms if s in available_symptoms]
        for rf in RED_FLAG_OPTIONS:
            st.session_state[f"rf_{rf['id']}"] = False
        st.rerun()

    if col_pre4.button("🔄 Reset Form", use_container_width=True, help="Clear all selections"):
        st.session_state["symptoms_preset"] = []
        for rf in RED_FLAG_OPTIONS:
            st.session_state[f"rf_{rf['id']}"] = False
        st.rerun()

    st.markdown("---")

    # Demographics
    st.markdown("##### 👤 Patient Demographics")
    age = st.slider("Patient Age (Years)", min_value=1, max_value=100, value=48, step=1)
    sex = st.radio("Biological Sex", options=["Male", "Female", "Other / Unspecified"], horizontal=True)

    st.markdown("---")

    # Symptoms Selection
    st.markdown("##### 🔍 Reported Symptoms (178 Features)")
    current_default = [s for s in st.session_state["symptoms_preset"] if s in available_symptoms_sorted]
    selected_symptoms = st.multiselect(
        "Select all symptoms reported by patient:",
        options=available_symptoms_sorted,
        default=current_default,
        help="Search and multi-select from the standardized 178 symptom vocabulary."
    )

    col_dur, col_sev = st.columns(2)
    with col_dur:
        duration = st.selectbox(
            "Symptom Duration",
            options=["< 24 hours", "1 – 3 days", "4 – 7 days", "> 1 week"],
            index=1
        )
    with col_sev:
        severity = st.selectbox(
            "Self-Reported Severity",
            options=["Mild (Manageable)", "Moderate (Disruptive)", "Severe (Incapacitating)"],
            index=1
        )

    st.markdown("---")

    # Emergency Red Flags
    st.markdown("##### 🚨 Emergency Red-Flag Checklist")
    st.caption("Deterministic triggers that immediately override ML triage to EMERGENCY status.")
    
    selected_red_flags = []
    for rf in RED_FLAG_OPTIONS:
        flag_key = f"rf_{rf['id']}"
        flag_val = st.session_state.get(flag_key, False)
        checked = st.checkbox(
            rf["label"],
            value=flag_val,
            key=flag_key,
            help=rf["description"]
        )
        if checked:
            selected_red_flags.append(rf["id"])

    st.markdown("---")
    
    # Run Assessment Button
    run_triage = st.button("🩺 Run Clinical Triage Assessment", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD CONTENT
# -----------------------------------------------------------------------------

# Header
st.markdown("""
<div class="header-box">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1 class="header-title">SmartHealth Triage</h1>
            <p class="header-subtitle">Explainable Decision-Support & Clinical Risk Stratification System</p>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid #0284c7; color: #38bdf8; font-size: 0.8rem; font-weight: 700; padding: 6px 12px; border-radius: 8px;">
                Scikit-Learn • TreeSHAP • Safety Engine v1.0
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Prominent Academic Disclaimer Banner
st.markdown("""
<div class="disclaimer-banner">
    <div style="font-size: 1.4rem;">⚠️</div>
    <div>
        <strong>Academic Research Prototype — Not a Medical Diagnosis.</strong><br/>
        This system is developed for educational demonstration and research.
        It does not replace professional medical evaluation or clinical judgment.
        Its predictions and urgency assessments have not been clinically validated.
        <strong>If experiencing a possible medical emergency, call local
        emergency services (112 in India) immediately.</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# Run Inference
if not selected_symptoms and not selected_red_flags and not run_triage:
    st.info("👈 Please select patient symptoms or test an intake preset from the sidebar, then click **'Run Clinical Triage Assessment'**.")
    
    # Clinical demo walkthrough guide
    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        st.markdown("""
        <div class="clinical-card">
            <h4 style="color: #38bdf8; margin-top: 0;">1. Symptom Vectorization</h4>
            <p style="color: #94a3b8; font-size: 0.88rem;">
                Maps patient-reported symptoms into high-dimensional (178 binary features) sparse clinical state space verified against medical taxonomies.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_g2:
        st.markdown("""
        <div class="clinical-card">
            <h4 style="color: #818cf8; margin-top: 0;">2. Dual Decision Engine</h4>
            <p style="color: #94a3b8; font-size: 0.88rem;">
                Computes Top-3 probabilistic disease differential while enforcing deterministic clinical red-flag overrides for zero safety compromise.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_g3:
        st.markdown("""
        <div class="clinical-card">
            <h4 style="color: #34d399; margin-top: 0;">3. TreeSHAP Attribution</h4>
            <p style="color: #94a3b8; font-size: 0.88rem;">
                Delivers full local feature attribution, revealing present symptoms supporting the condition and absent symptoms arguing against it.
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Run the full inference pipeline
    with st.spinner("Executing ML Inference & TreeSHAP Attribution..."):
        assessment = predict_triage_assessment(
            reported_symptoms=selected_symptoms,
            user_red_flags=selected_red_flags,
            top_k=3
        )

    triage_level = assessment["triage_level"]
    safety_override = assessment["safety_override"]
    primary_condition = assessment["primary_condition"]
    primary_conf = assessment["primary_confidence"]
    top_conditions = assessment["top_conditions"]
    shap_exp = assessment["shap_explanation"]
    red_flag_alerts = assessment["red_flag_alerts"]
    guidance = assessment["clinical_guidance"]

    # -------------------------------------------------------------------------
    # 5.1 PATIENT INTAKE SNAPSHOT
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Patient Assessment & Stratification")
    
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    with col_p1:
        st.markdown(f"""
        <div class="patient-stat">
            <div class="patient-stat-num">{age} <span style="font-size: 0.85rem; color: #94a3b8;">yrs</span></div>
            <div class="patient-stat-label">Patient Age</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"""
        <div class="patient-stat">
            <div class="patient-stat-num">{sex.split()[0]}</div>
            <div class="patient-stat-label">Biological Sex</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p3:
        st.markdown(f"""
        <div class="patient-stat">
            <div class="patient-stat-num">{len(selected_symptoms)}</div>
            <div class="patient-stat-label">Symptoms Count</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p4:
        st.markdown(f"""
        <div class="patient-stat">
            <div class="patient-stat-num">{duration}</div>
            <div class="patient-stat-label">Duration</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p5:
        rf_color = "#ef4444" if len(selected_red_flags) > 0 else "#10b981"
        st.markdown(f"""
        <div class="patient-stat" style="border-color: {rf_color};">
            <div class="patient-stat-num" style="color: {rf_color};">{len(selected_red_flags)}</div>
            <div class="patient-stat-label">Red Flags Active</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5.2 SAFETY OVERRIDE ALERT (IF ACTIVE)
    # -------------------------------------------------------------------------
    if safety_override:
        flags_html = "".join([f"<li><strong>{rf}</strong></li>" for rf in red_flag_alerts])
        st.markdown(f"""
        <div class="override-alert-box">
            <div class="override-title">
                <span>🚨</span> CLINICAL SAFETY OVERRIDE ACTIVATED (DETERMINISTIC RED-FLAG ENGINE)
            </div>
            <div style="font-size: 0.95rem; line-height: 1.5; color: #fee2e2;">
                <strong>Immediate Emergency Department Care Required.</strong> 
                The patient intake triggered one or more acute clinical red flags. Regardless of the machine learning model's statistical probability distribution, 
                this case has been deterministically escalated to <strong>EMERGENCY PRIORITY</strong> to prevent missed life-threatening presentations.
            </div>
            <div style="margin-top: 12px; font-size: 0.9rem; background: rgba(0,0,0,0.25); padding: 10px 16px; border-radius: 8px;">
                <span style="color: #fca5a5; font-weight: 700;">Triggered Red-Flag Symptoms:</span>
                <ul style="margin: 6px 0 0 0; padding-left: 20px; color: #ffffff;">
                    {flags_html}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5.3 TRIAGE LEVEL BADGE & RECOMMENDATION
    # -------------------------------------------------------------------------
    badge_class = {
        "LOW": "triage-badge-low",
        "MODERATE": "triage-badge-moderate",
        "HIGH": "triage-badge-high",
        "EMERGENCY": "triage-badge-emergency"
    }.get(triage_level, "triage-badge-moderate")

    pill_class = {
        "LOW": "pill-low",
        "MODERATE": "pill-moderate",
        "HIGH": "pill-high",
        "EMERGENCY": "pill-emergency"
    }.get(triage_level, "pill-moderate")

    timeframe_text = {
        "LOW": "Routine Care • Outpatient or Self-Limiting Management",
        "MODERATE": "Evaluate within 48–72 Hours • Primary Care Clinic",
        "HIGH": "Evaluate within 12–24 Hours • Same-Day Urgent Care",
        "EMERGENCY": "IMMEDIATE Clinical Intervention • Call 911 / Emergency Room"
    }.get(triage_level, "Clinical Evaluation Recommended")

    st.markdown(f"""
    <div class="triage-badge-container {badge_class}">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span class="triage-pill {pill_class}">TRIAGE LEVEL: {triage_level}</span>
                <span style="font-weight: 700; font-size: 1.15rem; color: #f8fafc;">{timeframe_text}</span>
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8;">
                Safety Status: <strong style="color: {'#ef4444' if safety_override else '#10b981'};">{'OVERRIDDEN BY RULE ENGINE' if safety_override else 'ML-DERIVED ALIGNMENT'}</strong>
            </div>
        </div>
        <div style="font-size: 0.95rem; color: #e2e8f0; margin-top: 8px; line-height: 1.5;">
            {guidance}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5.4 TOP-3 PROBABLE HEALTH CONDITIONS & DIFFERENTIAL
    # -------------------------------------------------------------------------
    col_left, col_right = st.columns([1.1, 1.2])

    with col_left:
        st.markdown("#### 🎯 Top-3 Probable Health Conditions")
        st.caption("Scikit-Learn Multi-Class Inference with Softmax Probabilities")

        rank_labels = ["Primary Diagnostic Match", "Secondary Differential", "Tertiary Differential"]
        rank_colors = ["#38bdf8", "#818cf8", "#94a3b8"]

        for i, cond_info in enumerate(top_conditions):
            c_name = cond_info["condition"]
            c_prob = cond_info["probability"]
            c_pct = cond_info["confidence_percent"]

            st.markdown(f"""
            <div class="condition-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <div>
                        <span style="font-size: 0.75rem; font-weight: 700; color: {rank_colors[i]}; text-transform: uppercase; letter-spacing: 0.05em;">
                            #{i+1} • {rank_labels[i]}
                        </span>
                        <div class="condition-title">{c_name}</div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 1.3rem; font-weight: 800; color: {rank_colors[i]};">{c_pct:.1f}%</span>
                        <div style="font-size: 0.7rem; color: #94a3b8;">Confidence</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for visual weight
            st.progress(float(min(max(c_prob, 0.0), 1.0)))

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("#### 🩺 Recommended Clinical Pathway")
        st.markdown(f"""
        <div class="clinical-card" style="font-size: 0.9rem; line-height: 1.6;">
            <strong>Primary Assessment:</strong> <span style="color: #38bdf8; text-transform: capitalize;">{primary_condition}</span> ({primary_conf:.1f}% match).<br/>
            <strong>Triage Urgency:</strong> <span style="font-weight: 700;">{triage_level}</span>.<br/>
            <strong>Recommended Setting:</strong> 
            {"Emergency Department / Call 911" if triage_level == "EMERGENCY" else "Urgent Care Clinic (12–24h)" if triage_level == "HIGH" else "Primary Care Clinic (48–72h)" if triage_level == "MODERATE" else "Home Supportive Care / Primary Care follow-up"}.<br/>
            <strong>Clinical Monitoring:</strong> If experiencing worsening respiratory distress, chest discomfort, or confusion, seek immediate acute emergency evaluation.
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5.5 SHAP EXPLAINABILITY (LOCAL FEATURE ATTRIBUTION)
    # -------------------------------------------------------------------------
    with col_right:
        st.markdown("#### 🔬 Explainability Section (TreeSHAP)")
        st.caption(f"Feature attribution explaining model logic for: **{primary_condition.title()}**")

        pos_contributors = shap_exp.get("positive_contributors", [])
        neg_contributors = shap_exp.get("negative_contributors", [])

        # Build data for Altair horizontal bar chart
        chart_data = []
        for item in pos_contributors[:5]:
            chart_data.append({
                "Symptom": item["symptom"].title(),
                "SHAP Impact": item["shap_value"],
                "Contribution": "Supports Disease (+)",
                "Present": "Yes (Reported)" if item["is_present"] else "No (Absent)"
            })
            
        for item in neg_contributors[:5]:
            chart_data.append({
                "Symptom": item["symptom"].title(),
                "SHAP Impact": item["shap_value"],
                "Contribution": "Argues Against (-)",
                "Present": "Yes (Reported)" if item["is_present"] else "No (Absent)"
            })

        if chart_data:
            shap_df = pd.DataFrame(chart_data)
            
            # Altair interactive horizontal bar chart
            chart = alt.Chart(shap_df).mark_bar(cornerRadius=4).encode(
                y=alt.Y('Symptom:N', sort=alt.EncodingSortField(field='SHAP Impact', order='ascending'), title='Symptom Feature'),
                x=alt.X('SHAP Impact:Q', title='SHAP Value (Contribution to Log-Odds)'),
                color=alt.Color(
                    'Contribution:N',
                    scale=alt.Scale(
                        domain=['Supports Disease (+)', 'Argues Against (-)'],
                        range=['#10b981', '#f43f5e']
                    ),
                    legend=alt.Legend(title="Attribution Direction", orient="top")
                ),
                tooltip=['Symptom', 'SHAP Impact', 'Contribution', 'Present']
            ).properties(
                height=320
            ).configure_axis(
                gridColor='#1e293b',
                labelColor='#94a3b8',
                titleColor='#cbd5e1'
            ).configure_view(
                strokeOpacity=0
            )

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No active symptoms to calculate SHAP feature contributions.")

        # Textual breakdown of explainability
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("##### 🟢 Top Supporting Symptoms")
            st.caption("Present symptoms that drove confidence higher:")
            if pos_contributors:
                for p in pos_contributors[:4]:
                    status_badge = "✅ Present" if p["is_present"] else "⭕ Absent"
                    st.markdown(f"- **{p['symptom']}** ({status_badge}) `+{p['shap_value']:.4f}`")
            else:
                st.caption("None identified.")

        with col_e2:
            st.markdown("##### 🔴 Counter / Absent Symptoms")
            st.caption("Features that lowered confidence:")
            if neg_contributors:
                for n in neg_contributors[:4]:
                    status_badge = "✅ Present" if n["is_present"] else "⭕ Absent"
                    st.markdown(f"- **{n['symptom']}** ({status_badge}) `{n['shap_value']:.4f}`")
            else:
                st.caption("None identified.")

    # -------------------------------------------------------------------------
    # 5.6 SYSTEM SPECIFICATIONS & AUDIT TRAIL
    # -------------------------------------------------------------------------
    with st.expander("🔍 Clinical Methodology & Safety Architecture Details"):
        st.markdown(f"""
        **System Pipeline Architecture:**
        1. **Feature Input Vector:** 178 binary clinical indicators extracted from standardized clinical records.
        2. **Multi-Class Classifier:** Scikit-Learn Multiclass Engine (`models/best_model.joblib`), trained and validated with Top-1/Top-3 benchmarking.
        3. **Explainability Engine:** TreeSHAP (`src/explainability.py`) computing exact Shapley feature attributions for additive feature importance.
        4. **Clinical Safety Layer:** Deterministic rule engine (`src/safety_layer.py`) acting as a fail-safe circuit breaker. Any presence of acute emergency red flags (e.g. crushing chest pain, acute hypoxia, sudden focal paralysis) forces an immediate escalation to **EMERGENCY** priority, superseding model confidence scores.
        
        **Full Medical Disclaimer:**
        {assessment['clinical_disclaimer']}
        """)

# -----------------------------------------------------------------------------
# 6. FOOTER
# -----------------------------------------------------------------------------
st.markdown("<hr style='border-color: #1e293b; margin-top: 40px;'/>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; padding-bottom: 20px;">
    SmartHealth Triage System • Advanced Medical Informatics & Explainable AI Research • Academic Prototype
</div>
""", unsafe_allow_html=True)
