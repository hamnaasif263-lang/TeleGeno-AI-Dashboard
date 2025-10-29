import streamlit as st
import pandas as pd
import numpy as np
import json
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple

# --- Safe optional imports (do not execute shell commands inside this file) ---
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except Exception:
    mp = None
    _HAS_MEDIAPIPE = False

from datetime import datetime
import time

# Define constants
SNP_EFFECTS = {
    "CYP2C19": {
        "rs4244285": {"AA": "Poor", "AG": "Intermediate", "GG": "Normal"},
        "Metabolite_Impact": {
            "Poor": "Consider alternative medication",
            "Intermediate": "Monitor closely",
            "Normal": "Standard dosing"
        }
    },
    "APOE": {
        "rs429358": {"AA": "Low Risk", "AG": "Medium Risk", "GG": "High Risk"},
        "Risk_Impact": {
            "Low Risk": "No specific action needed",
            "Medium Risk": "Regular monitoring advised",
            "High Risk": "Intensive monitoring required"
        }
    }
}

STATUS_COLORS = {
    "Normal": "#2ecc71",
    "Intermediate": "#f1c40f",
    "Poor": "#e74c3c",
    "Low Risk": "#2ecc71",
    "Medium Risk": "#f1c40f",
    "High Risk": "#e74c3c",
    "Unknown": "#95a5a6"
}

MED_RECOMMENDATIONS = {
    "CYP2C19": {
        "Poor": "Avoid clopidogrel. Consider prasugrel or ticagrelor.",
        "Intermediate": "Use caution with clopidogrel; consider monitoring or alternative.",
        "Normal": "Standard dosing acceptable."
    },
    "APOE": {
        "High Risk": "Assess cognitive risk; consider specialist referral.",
        "Medium Risk": "Lifestyle modification and monitoring.",
        "Low Risk": "Routine care."
    }
}

EMERGENCY_STATUS_COLORS = {
    "Critical": "#ff4444",
    "Warning": "#0b0b0b",   # changed to dark/black for readable text
    "Normal": "#00C851"
}

EMERGENCY_RECOMMENDATIONS = {
    "Critical": {
        "BP": "Immediate medical attention required. Call emergency services.",
        "Heart": "Emergency cardiac evaluation needed. Keep patient calm and still.",
        "Both": "Critical condition - call emergency services immediately."
    },
    "Warning": {
        "BP": "Urgent medical consultation recommended. Monitor closely.",
        "Heart": "Cardiac monitoring advised. Seek medical attention.",
        "Both": "Multiple concerns - urgent medical evaluation needed."
    },
    "Normal": "Vital signs within normal range. Continue monitoring."
}

# Page setup
st.set_page_config(page_title="Telemedicine PGx Dashboard", layout="wide")
st.title("Pharmacogenomic Analysis & Risk Assessment Dashboard")
st.markdown("**Simulation only — not medical advice.**")

# --- Dark theme toggle and stronger CSS injection (all-black look) ---
st.sidebar.header("UI / Theme")
dark_mode = st.sidebar.checkbox("Dark theme (black)", value=True)

def _inject_dark_css():
    st.markdown(
        """
        <style>
        /* App background and main containers */
        .stApp, .block-container, .main { background-color: #000000 !important; color: #ffffff !important; }
        /* Sidebar */
        .css-1d391kg, .sidebar .block-container { background-color:#000000 !important; color:#fff !important; border-right: 1px solid #111; }
        /* Inputs and controls */
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div, textarea {
            background: #0b0b0b !important; color: #fff !important; border: 1px solid #222 !important;
        }
        /* Buttons */
        .stButton>button { background-color: #111 !important; color: #fff !important; border: 1px solid #333 !important; }
        /* DataFrame / table */
        .stDataFrame table { background-color: #0a0a0a !important; color: #fff !important; }
        /* Expander and examples */
        .stExpander { background: #0b0b0b !important; color: #fff !important; border: 1px solid #222 !important; }
        /* Capsule patient name */
        .patient-capsule { display:inline-block; padding:8px 16px; border-radius:999px; background:#111; color:#fff; font-weight:600; border:1px solid #333; }
        /* Plotly background */
        .js-plotly-plot .plotly { background-color: #000000 !important; }
        /* small text, notes */
        .stMarkdown p, .stMarkdown span { color: #ddd !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

if dark_mode:
    _inject_dark_css()

# Sidebar setup
st.sidebar.header("Input Methods & Examples")

# Sidebar organization - clean version
st.sidebar.header("Input Methods")
input_type = st.sidebar.radio(
    "Choose input",
    ["JSON SNP Input", "Simulated VCF", "Demographics + History (AI)"]
)

# Add separator
st.sidebar.markdown("---")

# Patient info section
st.sidebar.header("Patient Info")
patient_name = st.sidebar.text_input("Patient name", value="John Doe")

# Add separator
st.sidebar.markdown("---")

# Emergency assessment checkbox
emergency_tab = st.sidebar.checkbox("🚨 Emergency Assessment")

# Example JSON data
sample_json = '{"rs4244285": "AA", "rs4986893": "AG", "rs429358": "AG"}'
sample_vcf = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
1\t12345\trs4244285\tA\tG\t.\t.\t.
1\t67890\trs429358\tC\tT\t.\t.\t."""

# Show examples in expandable sections
with st.sidebar.expander("Example JSON"):
    st.code(sample_json)
with st.sidebar.expander("Example VCF"):
    st.code(sample_vcf)

# --- Patient name input (always at top) ---
# st.sidebar.header("Patient Info")
# patient_name = st.sidebar.text_input("Patient name", value="John Doe")

# --- Emergency tab (always last) ---
# emergency_tab = st.sidebar.checkbox("🚨 Emergency Assessment")

# --- JSON / VCF example data (unchanged) ---
# sample_json = '{"rs4244285": "AA", "rs4986893": "AG", "rs429358": "AG"}'
# sample_vcf = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n1\t12345\trs4244285\tA\tG\t.\t.\t.\n1\t67890\trs429358\tC\tT\t.\t.\t."

# --- Helper functions (unchanged logic) ---
@st.cache_data
def parse_json_input(json_data: str) -> Dict[str, str] | None:
    try:
        parsed = json.loads(json_data)
        if isinstance(parsed, dict):
            return {k: str(v).upper() for k, v in parsed.items()}
        return None
    except Exception:
        return None

@st.cache_data
def parse_vcf_simulator(vcf_content: str) -> Dict[str, str]:
    lines = vcf_content.splitlines()
    snps = {}
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            snp_id = parts[2]
            for gene in SNP_EFFECTS:
                if snp_id in SNP_EFFECTS[gene]:
                    snps[snp_id] = random.choice(["AA", "AG", "GG"])
    return snps

def analyze_snps(snps: Dict[str, str]) -> Tuple[list, dict]:
    results = []
    status = {gene: "Unknown" for gene in SNP_EFFECTS.keys()}
    severity_order = ["Unknown", "Low Risk", "Normal", "Medium Risk", "Intermediate", "High Risk", "Poor"]
    for gene, mapping in SNP_EFFECTS.items():
        gene_effects = []
        for snp_id, genotype_map in mapping.items():
            if snp_id in ("Metabolite_Impact", "Risk_Impact"):
                continue
            if snp_id in snps:
                genotype = snps[snp_id]
                effect = genotype_map.get(genotype, "Unknown")
                results.append({
                    "Gene": gene,
                    "SNP ID": snp_id,
                    "Genotype": genotype,
                    "Effect": effect
                })
                gene_effects.append(effect)
        if gene_effects:
            worst = max(gene_effects, key=lambda e: severity_order.index(e) if e in severity_order else 0)
            status[gene] = worst
    return results, status

def mock_ai_prediction(age: int, weight: float, smoking: bool, family_history: bool, temperature: float, bp_sys: int, bp_dia: int) -> float:
    base = 0.05 + (age - 20) / 160 + (weight - 50) / 500
    base += 0.15 if smoking else 0
    base += 0.2 if family_history else 0
    base += max(0, (temperature - 36.5) / 6)  # fever increases score slightly
    base += max(0, (bp_sys - 120) / 400)
    score = min(max(base + random.uniform(-0.05, 0.15), 0.0), 1.0)
    return score

def predict_metabolizer_from_clinical(age: int, temperature: float, bp_sys: int, bp_dia: int, weight: float, smoking: bool) -> str:
    score = (age / 120) * 0.25
    score += max(0, (temperature - 36.5) / 4) * 0.25
    score += max(0, (bp_sys - 120) / 80) * 0.2
    score += (weight - 60) / 200 * 0.15
    score += 0.15 if smoking else 0
    score = min(max(score + random.uniform(-0.05, 0.1), 0.0), 1.0)
    if score > 0.7:
        return "Poor"
    if score > 0.4:
        return "Intermediate"
    return "Normal"

def results_to_df(results_list: list) -> pd.DataFrame:
    return pd.DataFrame(results_list) if results_list else pd.DataFrame(columns=["Gene", "SNP ID", "Genotype", "Effect"])

def recommend_meds(status: dict) -> dict:
    meds = {}
    for gene, stt in status.items():
        meds[gene] = MED_RECOMMENDATIONS.get(gene, {}).get(stt, "No specific medication guidance")
    return meds

def process_emergency_vitals(frame):
    """Simulate vital sign detection from video frame"""
    # Simulate processing delay
    time.sleep(0.5)
    
    # Generate mock vital signs
    heart_rate = random.randint(60, 130)
    systolic = random.randint(100, 180)
    diastolic = random.randint(60, 100)
    
    return {
        "heart_rate": heart_rate,
        "bp_systolic": systolic,
        "bp_diastolic": diastolic,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

# --- Main content area and patient capsule UI ---
st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
            f"<h2 style='margin:0;color:#fff'>Patient Dashboard</h2>"
            f"<div class='patient-capsule'>{patient_name}</div>"
            f"</div>", unsafe_allow_html=True)

# Prepare holders
snps_input = {}
results = []
status = {}

# Input flows
if input_type == "Demographics + History (AI)":
    st.subheader("Demographics + Clinical Measurements")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 100, 55)
        weight = st.number_input("Weight (kg)", 30, 200, 75)
        smoking = st.checkbox("Smoking history")
    with col2:
        family_history = st.checkbox("Family history (cardio/cognitive)")
        temperature = st.number_input("Temperature (°C)", 34.0, 42.0, 36.6, step=0.1)
        bp_sys = st.number_input("Systolic BP (mmHg)", 80, 220, 120)
    with col3:
        bp_dia = st.number_input("Diastolic BP (mmHg)", 40, 140, 80)
        # quick metabolizer preview button
        if st.button("Generate AI Prediction & Metabolizer"):
            ai_score = mock_ai_prediction(age, weight, smoking, family_history, temperature, bp_sys, bp_dia)
            metabolizer_pred = predict_metabolizer_from_clinical(age, temperature, bp_sys, bp_dia, weight, smoking)
            st.markdown(f"<div style='padding:12px;border-radius:8px;background:#0d0d0d;border:1px solid #222'>"
                        f"<strong style='color:#fff'>AI Risk Score:</strong> <span style='color:#9fe7a4'>{ai_score:.1%}</span><br>"
                        f"<strong style='color:#fff'>Model Confidence:</strong> <span style='color:#9fe7a4'>{random.uniform(0.65,0.95):.1%}</span><br>"
                        f"<strong style='color:#fff'>Predicted CYP2C19 Metabolizer:</strong> <span style='color:#ffd36b'>{metabolizer_pred}</span>"
                        f"</div>", unsafe_allow_html=True)
            # medication guidance for metabolizer
            med_for_met = MED_RECOMMENDATIONS.get("CYP2C19", {}).get(metabolizer_pred, "No guidance")
            st.subheader("Medication suggestion (based on predicted metabolizer)")
            st.markdown(f"<div style='padding:12px;border-radius:8px;background:{STATUS_COLORS.get(metabolizer_pred,'#222')};color:#000'>"
                        f"<strong>{metabolizer_pred}</strong><br>{med_for_met}</div>", unsafe_allow_html=True)
            # APOE style guidance from AI (use risk thresholds)
            st.subheader("AI-based recommendations")
            if ai_score > 0.7:
                st.error("High predicted risk. Recommend urgent clinical review and specialist referral.")
            elif ai_score > 0.4:
                st.warning("Moderate risk. Recommend closer monitoring and prevention.")
            else:
                st.success("Low risk. Routine care recommended.")
else:
    # SNP inputs
    if input_type == "JSON SNP Input":
        json_data = st.text_area("Enter JSON SNP object (e.g. {\"rs4244285\":\"AA\"})", height=160)
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Analyze JSON Data"):
                parsed = parse_json_input(json_data)
                if parsed:
                    snps_input = parsed
                    results, status = analyze_snps(snps_input)
                    st.success("JSON analyzed.")
                else:
                    st.error("Invalid JSON format. Provide a simple SNP->genotype dict.")
        with col2:
            st.info("Tips: use genotypes AA/AG/GG. Example available in sidebar.")
    else:
        uploaded_file = st.file_uploader("Upload VCF-like file", type=['vcf', 'txt'])
        col1, col2 = st.columns([1,3])
        with col1:
            if st.button("Use example VCF"):
                snps_input = parse_vcf_simulator(sample_vcf)
                results, status = analyze_snps(snps_input)
                st.success("Example VCF simulated.")
            if uploaded_file:
                content = uploaded_file.getvalue().decode(errors="ignore")
                snps_input = parse_vcf_simulator(content)
                results, status = analyze_snps(snps_input)
                st.success("VCF parsed (simulated genotypes).")
        with col2:
            st.info("Upload a VCF or text file. Parser is simulated and will return genotypes for known SNPs.")

# If simulate button set in session state, create random SNPs
if st.session_state.get("_simulate"):
    all_snps = [s for gene in SNP_EFFECTS for s in SNP_EFFECTS[gene] if s not in ("Metabolite_Impact", "Risk_Impact")]
    snps_input = {snp: random.choice(["AA", "AG", "GG"]) for snp in all_snps}
    results, status = analyze_snps(snps_input)
    st.success("Simulated SNPs generated.")
    st.session_state["_simulate"] = False

if emergency_tab:
    st.header("🚨 Emergency Assessment")

    if not _HAS_CV2:
        st.warning(
            "OpenCV (cv2) is not installed. Emergency camera and automatic vitals are disabled.\n"
            "Install opencv-python (and mediapipe if needed) in your virtual environment and restart the app."
        )
        st.stop()  # stop rendering the rest of the dashboard when emergency mode active
    else:
        col_cam, col_assess = st.columns([2, 1])

        with col_cam:
            st.subheader("Patient Camera Feed")
            camera = st.camera_input("Capture patient", key="emergency_camera")

            vitals = None
            if camera:
                bytes_data = camera.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                vitals = process_emergency_vitals(cv2_img)

                st.subheader("Vital Signs")
                col_hr, col_bp = st.columns(2)

                hr = vitals["heart_rate"]
                sys_bp = vitals["bp_systolic"]
                dia_bp = vitals["bp_diastolic"]

                with col_hr:
                    st.metric("Heart Rate", f"{hr} bpm")
                with col_bp:
                    st.metric("Blood Pressure", f"{sys_bp}/{dia_bp} mmHg")

        with col_assess:
            st.subheader("Quick Assessment")
            age = st.number_input("Patient Age", 1, 120, 50, key="emer_age")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="emer_gender")
            medical_history = st.multiselect(
                "Medical History",
                ["Hypertension", "Diabetes", "Heart Disease", "Asthma", "None"],
                key="emer_history"
            )
            chief_complaint = st.text_area("Chief Complaint/Symptoms", height=100, key="emer_complaint")

            # If camera not used yet, show hint only
            if not camera:
                st.info("Capture the patient photo above to run automated (simulated) vitals and assessment.")
            else:
                # determine statuses
                hr_status = (
                    "Critical" if hr > 130 or hr < 40 else
                    "Warning"  if hr > 100 or hr < 50 else
                    "Normal"
                )
                bp_status = (
                    "Critical" if sys_bp > 180 or dia_bp > 120 else
                    "Warning"  if sys_bp > 140 or dia_bp > 90 else
                    "Normal"
                )

                # overall status: only escalate to Emergency (Critical) when truly critical
                if hr_status == "Critical" or bp_status == "Critical":
                    overall = "Critical"
                elif hr_status == "Warning" or bp_status == "Warning":
                    overall = "Warning"
                else:
                    overall = "Normal"

                # Build non-dosage emergency treatment suggestions (clinician-level)
                emergency_treatments = []
                if overall == "Critical":
                    if sys_bp > 180 or dia_bp > 120:
                        emergency_treatments.append(
                            "Possible hypertensive emergency — urgent clinician-led BP lowering with IV antihypertensive agents in monitored setting."
                        )
                    if hr > 130:
                        emergency_treatments.append(
                            "Severe tachycardia — cardiac monitoring, consider urgent ECG and clinician evaluation for rhythm control."
                        )
                    if hr < 40:
                        emergency_treatments.append(
                            "Severe bradycardia — clinician assessment; pacing or IV chronotropic support may be required."
                        )
                    if sys_bp < 90:
                        emergency_treatments.append(
                            "Hypotension — rapid assessment for shock; IV fluids and vasopressors as clinically indicated."
                        )
                    # add metabolizer-based medication caution
                    # predict metabolizer from clinical features
                    clinical_met = predict_metabolizer_from_clinical(age, vitals.get("temperature", 36.6) if vitals else 36.6,
                                                                     sys_bp, dia_bp, weight if 'weight' in locals() else 70,
                                                                     True if "Hypertension" in medical_history else False)
                    pgx_met = infer_metabolizer_from_genotypes(results) if 'results' in locals() else "Unknown"
                    emergency_treatments.append(
                        f"Metabolizer (clinical): {clinical_met}. PGx inference: {pgx_met}. "
                        "Consider metabolizer status before giving drugs metabolized by CYP2C19 (avoid clopidogrel if Poor)."
                    )
                elif overall == "Warning":
                    emergency_treatments.append(
                        "Cardiac monitoring advised. Urgent outpatient or ED evaluation recommended."
                    )
                    # metabolizer caution shown for moderate concerns too
                    clinical_met = predict_metabolizer_from_clinical(age, vitals.get("temperature", 36.6) if vitals else 36.6,
                                                                     sys_bp, dia_bp, weight if 'weight' in locals() else 70,
                                                                     True if "Hypertension" in medical_history else False)
                    emergency_treatments.append(f"Metabolizer (clinical): {clinical_met} — consider when selecting meds.")
                else:
                    emergency_treatments.append("Vitals within expected ranges. Continue routine monitoring.")

                # Show single card depending on overall status
                # use EMERGENCY_STATUS_COLORS and ensure readable text color
                content_color = EMERGENCY_STATUS_COLORS.get(overall, "#111111")
                text_color = "#000" if content_color in ("#ffbb33", "#ff4444", "#00C851") and overall != "Warning" else "#fff"

                st.markdown(
                    f"<div style='padding:18px;border-radius:10px;background:{content_color};color:{text_color};'>"
                    f"<h3 style='margin:0'>{overall} Status</h3>"
                    f"<p style='margin:8px 0 0 0'>{' '.join(emergency_treatments)}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # Only show emergency call/action when Critical
                if overall == "Critical":
                    st.markdown("### Emergency Actions")
                    c1, c2 = st.columns([1,1])
                    with c1:
                        if st.button("🚑 Call Emergency"):
                            st.error("Simulated: Emergency services notified")
                    with c2:
                        report = (
                            f"Emergency Assessment Report\n"
                            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Patient: {patient_name}\nAge: {age}\nGender: {gender}\n\n"
                            f"Vital Signs:\n- Heart Rate: {hr} bpm ({hr_status})\n"
                            f"- Blood Pressure: {sys_bp}/{dia_bp} mmHg ({bp_status})\n\n"
                            f"Medical History: {', '.join(medical_history)}\n\n"
                            f"Chief Complaint:\n{chief_complaint}\n\n"
                            f"Overall Status: {overall}\nRecommendations:\n- " +
                            "\n- ".join(emergency_treatments)
                        )
                        st.download_button(
                            "📋 Download Emergency Report",
                            report,
                            file_name=f"emergency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                elif overall == "Warning":
                    # show only monitoring / urgent evaluation options (no 'Call Emergency' button)
                    st.markdown("### Recommended next steps")
                    if st.button("📄 Generate Evaluation Report"):
                        report = (
                            f"Urgent Evaluation Report\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Patient: {patient_name}\nAge: {age}\nGender: {gender}\n\n"
                            f"Vitals: HR {hr} bpm, BP {sys_bp}/{dia_bp} mmHg\n\n"
                            f"Recommendations:\n- {''.join(emergency_treatments)}"
                        )
                        st.download_button(
                            "📥 Download Report",
                            report,
                            file_name=f"urgent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                else:
                    # Normal -> show acknowledgement only (no further action unless user chooses)
                    st.success("Patient vitals are within normal limits. Continue routine monitoring.")
                    if st.button("📄 Generate Summary (optional)"):
                        report = (
                            f"Normal Assessment Summary\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            f"Patient: {patient_name}\nAge: {age}\nGender: {gender}\n\n"
                            f"Vitals: HR {hr} bpm, BP {sys_bp}/{dia_bp} mmHg\n\n"
                            f"Recommendations: Routine monitoring and follow-up as needed."
                        )
                        st.download_button(
                            "📥 Download Summary",
                            report,
                            file_name=f"normal_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )

# --- Display function updated (keeps charts, black styling honored) ---
def display_results(results: list, status: dict):
    st.header("Pharmacogenomic Analysis")
    df = results_to_df(results)
    if df.empty:
        st.info("No SNPs found for analysis.")
        return

    st.subheader("Summary table")
    genes = sorted(df["Gene"].unique().tolist())
    effects = sorted(df["Effect"].unique().tolist())
    gene_filter = st.multiselect("Genes", options=genes, default=genes)
    effect_filter = st.multiselect("Effects", options=effects, default=effects)
    df_filtered = df[df["Gene"].isin(gene_filter) & df["Effect"].isin(effect_filter)]
    st.dataframe(df_filtered, use_container_width=True, height=300)

    plotly_template = "plotly_dark" if dark_mode else "plotly"
    st.subheader("Visualizations")
    colA, colB = st.columns(2)
    with colA:
        genotype_counts = df_filtered["Genotype"].value_counts().reset_index()
        genotype_counts.columns = ["Genotype", "Count"]
        fig_bar = px.bar(genotype_counts, x="Genotype", y="Count", color="Genotype",
                         title="Genotype counts", template=plotly_template)
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)
    with colB:
        effect_counts = df_filtered["Effect"].value_counts().reset_index()
        effect_counts.columns = ["Effect", "Count"]
        fig_pie = px.pie(effect_counts, names="Effect", values="Count", title="Effect distribution",
                         color="Effect", color_discrete_map=STATUS_COLORS, template=plotly_template)
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    treemap_df = df_filtered.groupby(["Gene", "Effect"]).size().reset_index(name="count")
    fig_tree = px.treemap(treemap_df, path=["Gene", "Effect"], values="count", color="count",
                         title="Gene -> Effect treemap", template=plotly_template)
    fig_tree.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_tree, use_container_width=True)

    # medication recommendations for detected genes
    st.subheader("Medication Recommendations")
    meds = recommend_meds(status)
    cols = st.columns(len(meds))
    for i, (gene, rec) in enumerate(meds.items()):
        bg = STATUS_COLORS.get(status.get(gene, "Unknown"), "#95a5a6")
        cols[i].markdown(f"""
        <div style="padding:10px;border-radius:8px;background:{bg};color:black">
        <strong>{gene}</strong><br>
        <small>Status: {status.get(gene,'Unknown')}</small><hr style="opacity:0.6">
        {rec}
        </div>
        """, unsafe_allow_html=True)

    csv = df.to_csv(index=False)
    st.download_button("Download results (CSV)", csv, file_name="pgx_results.csv", mime="text/csv")

# If results were computed earlier, display them
try:
    if results:
        display_results(results, status)
except Exception as e:
    st.error(f"Display error: {e}")

# Footer / notes
st.markdown("---")
st.markdown("*This is a simulation dashboard for educational purposes only.*")
