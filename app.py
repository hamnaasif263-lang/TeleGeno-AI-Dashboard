import streamlit as st
import pandas as pd
import numpy as np
import json
import random
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Optional imports for webcam (if available in environment)
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

# ----------------------------
# --- Constants / Mappings ---
# ----------------------------
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
    "Normal": "#4CAF50",      # Green
    "Intermediate": "#FFC107", # Amber
    "Poor": "#F44336",         # Red
    "Low Risk": "#4CAF50",
    "Medium Risk": "#FFC107",
    "High Risk": "#F44336",
    "Unknown": "#9E9E9E"       # Grey
}

MED_RECOMMENDATIONS = {
    "CYP2C19": {
        "Poor": "Avoid **clopidogrel**. Consider prasugrel or ticagrelor.",
        "Intermediate": "Use caution with **clopidogrel**; consider monitoring or alternative.",
        "Normal": "Standard dosing acceptable."
    },
    "APOE": {
        "High Risk": "Assess cognitive risk; consider **specialist referral**.",
        "Medium Risk": "Lifestyle modification and **monitoring**.",
        "Low Risk": "Routine care."
    }
}

EMERGENCY_STATUS_COLORS = {
    "Critical": "#F44336", # Red
    "Warning": "#FF9800",  # Orange
    "Normal": "#4CAF50"    # Green
}

# Define Medical History Options for Demographics form
MEDICAL_HISTORY_OPTIONS = ['Hypertension', 'Cholesterol', 'Diabetes', 'Thyroid Disorder', 'Cardiovascular Disease', 'Previous Stroke', 'Kidney Disease']

# ----------------------------
# --- Helper Functions ---
# ----------------------------
@st.cache_data
def parse_json_input(json_data: str):
    try:
        parsed = json.loads(json_data)
        if isinstance(parsed, dict):
            return {k: str(v).upper() for k, v in parsed.items()}
    except Exception:
        return None

@st.cache_data
def parse_vcf_simulator(vcf_content: str):
    # Added performance optimization hint (caching based on content hash)
    random.seed(hash(vcf_content) % 1000)
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
                    # Randomly assign a genotype for simulation purposes
                    snps[snp_id] = random.choice(["AA", "AG", "GG"])
    return snps

def analyze_snps(snps):
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
                results.append({"Gene": gene, "SNP ID": snp_id, "Genotype": genotype, "Effect": effect})
                gene_effects.append(effect)
        if gene_effects:
            # Determine the single worst effect for the gene summary
            worst = max(gene_effects, key=lambda e: severity_order.index(e) if e in severity_order else 0)
            status[gene] = worst
    return results, status

def results_to_df(results_list):
    return pd.DataFrame(results_list) if results_list else pd.DataFrame(columns=["Gene", "SNP ID", "Genotype", "Effect"])

def recommend_meds(status):
    meds = {}
    for gene, stt in status.items():
        meds[gene] = MED_RECOMMENDATIONS.get(gene, {}).get(stt, "No specific medication guidance")
    return meds

def infer_metabolizer_from_genotypes(results_list):
    df = results_to_df(results_list)
    if "CYP2C19" in df["Gene"].values:
        row = df[df["Gene"] == "CYP2C19"]
        order = {"Poor":3, "Intermediate":2, "Normal":1}
        row = row.copy()
        row["score"] = row["Effect"].map(lambda x: order.get(x, 0))
        if not row.empty:
            best = row.sort_values("score", ascending=False).iloc[0]["Effect"]
            return best
    return "Unknown"

# A small function to compute a population-style aggregated metric (for demo)
def metrics_from_results(status):
    total = 1 
    critical_pct = 100 if any(s in status.values() for s in ["Poor", "High Risk"]) else 0
    avg_hr = random.randint(60, 95)
    avg_bp = f"{random.randint(110,135)}/{random.randint(70,85)}"
    return total, critical_pct, avg_hr, avg_bp

# Function to reset dashboard state 
def reset_dashboard():
    st.session_state['pgx_results_list'] = []
    st.session_state['pgx_status'] = {}
    
    st.success("Dashboard state cleared! Ready for new run.")
    time.sleep(0.5)
    st.rerun()

# Function to generate comprehensive report
def generate_comprehensive_report(patient_name, results_list, status):
    report = f"--- Comprehensive TeleGeno AI Report for {patient_name} ---\n"
    report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += "--------------------------------------------------------\n\n"
    
    # 1. PGx Summary
    metabolizer = infer_metabolizer_from_genotypes(results_list)
    report += f"1. PHARMACOGENOMIC STATUS:\n"
    report += f"   - CYP2C19 Metabolizer Status: {metabolizer}\n"
    report += f"   - Overall PGx Status: {max(status.values(), key=lambda e: ['Unknown', 'Low Risk', 'Normal', 'Medium Risk', 'Intermediate', 'High Risk', 'Poor'].index(e)) if status else 'N/A'}\n\n"
    
    # 2. Detailed Genotype Results
    report += "2. DETAILED GENOTYPE RESULTS:\n"
    df = results_to_df(results_list)
    if not df.empty:
        report += df.to_markdown(index=False) + "\n\n"
    else:
        report += "   No genetic variants analyzed.\n\n"
        
    # 3. Medication Guidance
    report += "3. ACTIONABLE MEDICATION GUIDANCE:\n"
    meds = recommend_meds(status)
    for gene, rec in meds.items():
        report += f"   - {gene} ({status.get(gene, 'Unknown')} Status): {rec}\n"
    
    report += "\n--------------------------------------------------------\n"
    return report


# ----------------------------
# --- Page & Theme Setup ---
# ----------------------------
st.set_page_config(page_title="TeleGeno AI — Patient Dashboard", layout="wide", initial_sidebar_state="expanded")

# Light theme injection (Custom CSS)
def _inject_css(light=True):
    base = """
    <style>
      .stApp { background-color:#F5F5F5; }
      .stSidebar { background-color:#FFFFFF; border-right: 1px solid #EEEEEE; }
      .block-container { color:#333333; }
      .card { background:#FFFFFF; padding:14px; border-radius:12px; border:1px solid #DDDDDD; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
      h1, h2, h3, h4, .small-metric { color:#333333; }
      .small-metric { font-size:24px; font-weight:700; }
      .muted { color:#757575; font-size:13px; }
      .stRadio > label { font-weight: 500; }
      .stRadio { border: 1px solid #DDDDDD; padding: 10px; border-radius: 8px;}
      .patient-capsule { display:inline-block; padding:8px 16px; border-radius:999px; background:#BBDEFB; color:#1565C0; font-weight:700; border:1px solid #90CAF9; }
      .plotly-graph-div .modebar-btn { background: #FFFFFF !important; color: #616161 !important; }
      .med-card-content { color: #333333 !important; }
      
      /* FIX: Reduce font size for Treemap title and labels */
      /* Increased uniformtext_minsize to make labels readable */
      .modebar { margin-top: -30px !important; }
    </style>
    """
    st.markdown(base, unsafe_allow_html=True)

_inject_css(light=True)

# ----------------------------
# --- Sidebar: inputs ---
# ----------------------------
st.sidebar.markdown("## 👤 Patient & Triage Controls")
patient_name = st.sidebar.text_input("Patient Name", value="Ferdoun S.")

# Emergency Checkbox Control
emergency_tab = st.sidebar.checkbox("🚨 Enable Emergency Assessment")
st.sidebar.markdown("---")

# --- Initial Setup for Input State ---
if '_input_type' not in st.session_state:
    st.session_state['_input_type'] = "Demographics + History (AI)" # Default to AI for quick demo
if 'pgx_results_list' not in st.session_state:
    st.session_state['pgx_results_list'] = []
    st.session_state['pgx_status'] = {}

# Only show PGx inputs if Emergency is NOT enabled
if not emergency_tab:
    st.sidebar.markdown("### 🧬 Genomic Data Input Method")
    input_type = st.sidebar.radio("Select Input Source", 
                                  ["JSON SNP Input", "Simulated VCF", "Demographics + History (AI)"], 
                                  key="_input_type")
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Simulate random patient SNPs"):
        st.session_state["_simulate_snps"] = True

    sample_json = '{"rs4244285": "AA", "rs429358": "AG"}'
    sample_vcf = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
1\t12345\trs4244285\tA\tG\t.\t.\t.
1\t67890\trs429358\tC\tT\t.\t.\t."""

    with st.sidebar.expander("Sample JSON"):
        st.code(sample_json)
    with st.sidebar.expander("Sample VCF (sim)"):
        st.code(sample_vcf)

# ----------------------------
# --- Header ---
# ----------------------------
header_col1, header_col2 = st.columns([8,2])
with header_col1:
    st.markdown("<h1 style='margin:0;color:#333'>TeleGeno AI — Patient Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Pharmacogenomic analysis • Explainable medication guidance • Emergency triage</div>", unsafe_allow_html=True)
with header_col2:
    st.markdown(f"<div style='text-align:right'><div class='patient-capsule'>{patient_name}</div></div>", unsafe_allow_html=True)

st.markdown("---")

# Retrieve current results from session state 
results = st.session_state.get('pgx_results_list', [])
status = st.session_state.get('pgx_status', {})


# ----------------------------
# --- Emergency Screen Takeover (Finalized) ---
# ----------------------------
if emergency_tab:
    st.markdown("## 🚨 Live Emergency Triage", unsafe_allow_html=True)
    
    e_col1, e_col2 = st.columns([3, 2])

    with e_col1:
        st.markdown("### Step 1: Capture Patient Vitals & Context")
        
        # Emergency context inputs (NEW)
        with st.form("emergency_form"):
            col_age, col_gender = st.columns(2)
            age = col_age.number_input("Patient Age", 0, 120, 55, key='em_age')
            gender = col_gender.selectbox("Gender", ["Male", "Female", "Other"], key='em_gender')
            
            # Use multi-select for history, pre-filling from main dashboard state if possible
            default_med_hist = st.session_state.get('ai_med_hist', [])
            medical_history = st.multiselect(
                "Known Medical History", 
                options=MEDICAL_HISTORY_OPTIONS, 
                default=default_med_hist,
                key='em_med_hist'
            )
            
            # Webcam/Simulated Vitals Capture
            if not _HAS_CV2:
                st.error("OpenCV not installed: Webcam features disabled.")
                simulate_vitals = st.form_submit_button("Simulate Vitals & Run Triage")
                cam = True # Mock camera input is always "on" for simulation run
            else:
                cam = st.camera_input("Capture patient photo for vitals check")
                simulate_vitals = st.form_submit_button("Run Triage")

        if simulate_vitals or cam:
            
            # --- MOCK VITAL PREDICTION ---
            # Simulate HR/BP influenced by age and history (e.g., Hypertension)
            vitals = {
                "hr": random.randint(70, 100) + (15 if 'Hypertension' in medical_history else 0),
                "bp_sys": random.randint(110, 150) + (25 if 'Hypertension' in medical_history else 0),
                "bp_dia": random.randint(70, 95) + (10 if 'Hypertension' in medical_history else 0)
            }
            
            # --- MOCK PGx PREDICTION FOR EMERGENCY SCREEN (Guaranteed Prediction) ---
            cyp_risk_factor = 0
            if 'Cardiovascular Disease' in medical_history: cyp_risk_factor += 0.3
            if 'Diabetes' in medical_history: cyp_risk_factor += 0.2
            if age > 65: cyp_risk_factor += 0.15

            if cyp_risk_factor > 0.4: metabolizer_status = "Poor"
            elif cyp_risk_factor > 0.15: metabolizer_status = "Intermediate"
            else: metabolizer_status = "Normal"

            # --- DETERMINE OVERALL STATUS ---
            status_overall = "Normal"
            if vitals['hr']>130 or vitals['bp_sys']>170 or metabolizer_status == "Poor":
                status_overall = "Critical"
            elif vitals['hr']>100 or vitals['bp_sys']>140 or metabolizer_status == "Intermediate":
                status_overall = "Warning"

            st.markdown("---")
            st.markdown("### Step 2: Vitals & Triage Summary")
            
            # --- Display Metrics (PGx Status, HR, BP, Triage) ---
            col_metab, col_hr, col_bp, col_triage = st.columns(4)
            
            # Metabolizer Status Card
            bg_pgx = STATUS_COLORS.get(metabolizer_status, '#9E9E9E')
            col_metab.markdown(f"<div class='card' style='padding:10px; background:{bg_pgx}; color:white;'>"
                                f"<strong>PGx Metabolizer:</strong><br/>"
                                f"<span style='font-size: 16px; font-weight: bold;'>{metabolizer_status}</span>"
                                f"</div>", unsafe_allow_html=True)
            
            # HR Metric
            col_hr.metric("Heart Rate", f"**{vitals['hr']}** bpm")
            
            # BP Metric 
            col_bp.metric("Blood Pressure", f"**{vitals['bp_sys']}/{vitals['bp_dia']}** mmHg")

            # Triage Status Box
            color = EMERGENCY_STATUS_COLORS.get(status_overall, "#9E9E9E")
            col_triage.markdown(f"<div class='card' style='background-color: {color}; color: white; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; margin: 0; padding: 5px;'>"
                                f"<strong>{status_overall.upper()}</strong>"
                                f"<p style='margin: 0; font-size: 12px;'>Automated Triage</p>"
                                f"</div>", unsafe_allow_html=True)

            # --- Actionable Advice ---
            st.markdown("### Step 3: Next Steps & Advice")
            
            if status_overall == "Critical":
                st.warning("⚠️ **CRITICAL ACTION REQUIRED:** Immediate medical intervention. Avoid CYP2C19-sensitive drugs (Clopidogrel) until PGx status is confirmed.")
                st.markdown("* **Precaution:** Prepare alternative medication (e.g., Ticagrelor) if acute need arises.")
            elif status_overall == "Warning":
                st.info("🟡 **WARNING:** Close monitoring and specialist consult advised. PGx status suggests potential drug dosage adjustment or alternative therapy.")
                st.markdown("* **Advice:** Monitor BP closely. Review full PGx dashboard for detailed medication recommendations.")
            else:
                st.success("✅ **NORMAL:** Vitals within expected ranges. Continue routine monitoring.")
                st.markdown("* **Advice:** Proceed to full PGx dashboard for proactive care planning.")
            
            # Report Generation
            report = f"Emergency Report - {patient_name}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report += f"Triage Status: {status_overall}\n"
            report += f"HR: {vitals['hr']} bpm\nBP: {vitals['bp_sys']}/{vitals['bp_dia']} mmHg\n"
            report += f"Predicted Metabolizer: {metabolizer_status}\n"
            
            st.download_button(
                "Download Emergency Report", 
                report, 
                file_name=f"emergency_{patient_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

    with e_col2:
        st.markdown("### Triage Action Guide")
        st.markdown(
            """
            * **CRITICAL (Red):** Immediate medical intervention needed.
            * **WARNING (Orange):** Close monitoring and specialist consult advised.
            * **NORMAL (Green):** Stable condition.
            
            This screen prioritizes **vitals-based assessment** before full PGx review.
            """
        )

    st.markdown("---")
    st.warning("Deactivate 'Enable Emergency Assessment' in the sidebar to return to the main PGx dashboard.")
    st.stop()


# ----------------------------
# --- Top Metric Cards (Conditional Display - FINAL FIX) ---
# ----------------------------
current_input = st.session_state.get('_input_type')
is_pgx_only = current_input == "JSON SNP Input" or current_input == "Simulated VCF"

# Only show metrics if Demographics is selected. Hide them completely for JSON/VCF, solving the irrelevance issue.
if not is_pgx_only:
    total, critical_pct, avg_hr, avg_bp = metrics_from_results(status)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.markdown(f"<div class='card'><div class='muted'>Total Patients</div><div class='small-metric'>{total}</div></div>", unsafe_allow_html=True)
    mcol2.markdown(f"<div class='card' style='border-left: 5px solid {EMERGENCY_STATUS_COLORS['Critical'] if critical_pct > 0 else '#EEEEEE'};'><div class='muted'>Critical PGx Flag</div><div class='small-metric' style='color:{EMERGENCY_STATUS_COLORS['Critical'] if critical_pct > 0 else '#4CAF50'}'>{critical_pct}%</div></div>", unsafe_allow_html=True)
    mcol3.markdown(f"<div class='card'><div class='muted'>Avg Heart Rate (Pop)</div><div class='small-metric'>{avg_hr} bpm</div></div>", unsafe_allow_html=True)
    mcol4.markdown(f"<div class='card'><div class='muted'>Avg Blood Pressure (Pop)</div><div class='small-metric'>{avg_bp}</div></div>", unsafe_allow_html=True)
    
    st.markdown("")
else:
    # Hide metrics completely for JSON/VCF inputs
    st.markdown("<br/>", unsafe_allow_html=True)
    

# ----------------------------
# --- Main two-column layout ---
# ----------------------------
left_col, right_col = st.columns([3,1])

# ---------- LEFT: charts & analysis ----------
with left_col:
    st.markdown("## 🔬 PGx Analysis & Visualization", unsafe_allow_html=True)
    
    # --- Input Section (Main Content) ---
    st.markdown("### 📥 Load Patient Genomic Data")
    current_input_type = st.session_state.get("_input_type")
    
    # ----------------------------------------------------
    # A. Demographics + History (AI) Input
    # ----------------------------------------------------
    if current_input_type == "Demographics + History (AI)":
        st.markdown("Fill out demographics and history for **AI risk prediction** (simulated PGx inference).")
        
        with st.form("demo_form", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", 0, 120, 55, key='ai_age')
                gender = st.selectbox("Gender", ["Male", "Female", "Other"], key='ai_gender')
                weight = st.number_input("Weight (kg)", 20, 200, 70, key='ai_weight')
            with c2:
                bp_sys = st.number_input("Systolic BP (mmHg)", 80, 220, 125, key='ai_bp_sys')
                bp_dia = st.number_input("Diastolic BP (mmHg)", 40, 140, 85, key='ai_bp_dia')
                smoking = st.checkbox("Current Smoker", key='ai_smoking')
            with c3:
                # Allow multiple selection for medical history
                medical_history = st.multiselect(
                    "Select Medical History (Multiple Allowed)", 
                    options=MEDICAL_HISTORY_OPTIONS, 
                    default=['Hypertension'],
                    key='ai_med_hist'
                )
                family_history = st.checkbox("Family History of Early CHD/Stroke", key='ai_fam_hist')
                temp = st.number_input("Temperature (°C)", 34.0, 42.0, 36.6, step=0.1, key='ai_temp')
                
            submitted = st.form_submit_button("Run AI Prediction")
            
            if submitted:
                # --- ENHANCED SIMULATED AI LOGIC ---
                
                # BASE RISK: Higher BP/Age
                ai_score = 0.05 + (age / 120) * 0.15 + (bp_sys / 220) * 0.10
                
                # CYP2C19 (Metabolizer) Prediction Logic: Influenced by age and smoking/cardio history
                cyp_risk_factor = 0
                if smoking: cyp_risk_factor += 0.30
                if 'Cardiovascular Disease' in medical_history: cyp_risk_factor += 0.25
                
                if cyp_risk_factor > 0.4: metabolizer_pred = "Poor"
                elif cyp_risk_factor > 0.15: metabolizer_pred = "Intermediate"
                else: metabolizer_pred = "Normal"
                
                # APOE (Risk) Prediction Logic: Influenced by history (Cholesterol/Diabetes/Family)
                apoe_risk_factor = 0
                if 'Cholesterol' in medical_history: apoe_risk_factor += 0.20
                if 'Diabetes' in medical_history: apoe_risk_factor += 0.25
                if family_history: apoe_risk_factor += 0.35
                
                if apoe_risk_factor > 0.5: apoe_pred = "High Risk"
                elif apoe_risk_factor > 0.2: apoe_pred = "Medium Risk"
                else: apoe_pred = "Low Risk"
                
                # Final AI Score reflects combined worst case
                combined_score = ai_score + cyp_risk_factor + apoe_risk_factor
                final_ai_score = min(combined_score, 0.99)
                
                st.success(f"AI Risk Score: **{final_ai_score:.2%}** — Predicted CYP2C19 Metabolizer: **{metabolizer_pred}**")
                
                # Create final PGx results
                results = [
                    {"Gene":"CYP2C19","SNP ID":"rs4244285","Genotype":"Simulated","Effect":metabolizer_pred},
                    {"Gene":"APOE","SNP ID":"rs429358","Genotype":"Simulated","Effect":apoe_pred}
                ]
                
                status = {"CYP2C19":metabolizer_pred, "APOE":apoe_pred}
                
                # Store results and status in session state and rerun
                st.session_state["pgx_results_list"] = results 
                st.session_state["pgx_status"] = status
                st.rerun() 
    
    # ----------------------------------------------------
    # B. JSON SNP Input
    # ----------------------------------------------------
    elif current_input_type == "JSON SNP Input":
        st.markdown("Paste JSON SNP object (e.g. `{\"rs4244285\":\"AA\"}`) in the text area. **Analysis is triggered on button press.**")
        
        initial_json = json.dumps({"rs4244285":"AG", "rs429358":"AA"}) 
        json_data = st.text_area("JSON SNP input", value=initial_json, height=120, key='json_input_data')
        
        if st.button("Analyze JSON"):
            parsed = parse_json_input(json_data)
            if parsed:
                snps_input = parsed
                results, status = analyze_snps(snps_input)
                st.session_state["pgx_results_list"] = results 
                st.session_state["pgx_status"] = status
                st.success("JSON parsed and analyzed.")
                st.rerun()
            else:
                st.error("Invalid JSON. Please check formatting.")
    
    # ----------------------------------------------------
    # C. Simulated VCF Input
    # ----------------------------------------------------
    elif current_input_type == "Simulated VCF":
        st.markdown("Upload a VCF-like file, or use the simulator button below. **Note**: VCF parsing is simulated to assign random genotypes.")
        uploaded_file = st.file_uploader("Upload VCF-like file", type=['vcf','txt'], key='vcf_uploader')
        
        colA, colB = st.columns([1,3])
        with colA:
            if st.button("Simulate VCF from Example"):
                snps_input = parse_vcf_simulator(sample_vcf)
                results, status = analyze_snps(snps_input)
                st.session_state["pgx_results_list"] = results 
                st.session_state["pgx_status"] = status
                st.success("Simulated VCF parsed.")
                st.rerun()
        with colB:
            if uploaded_file:
                content = uploaded_file.getvalue().decode(errors="ignore")
                snps_input = parse_vcf_simulator(content)
                results, status = analyze_snps(snps_input)
                st.session_state["pgx_results_list"] = results 
                st.session_state["pgx_status"] = status
                st.info("VCF processing complete. Displaying results.") 
                st.rerun()
    
    # Display message if no input is selected in the sidebar
    else:
        st.info("Select an input source from the sidebar to begin PGx analysis.")
    
    st.markdown("---")

    # Display PGx results
    st.markdown("### SNP Genotype and Effect Table")
    df = results_to_df(results)
    if df.empty:
        st.warning("No SNP results available yet. Please load or simulate data.")
    else:
        st.table(df) # Use table for cleaner look with smaller data
        
        # --- Visualizations (Aesthetics Fixed) ---
        st.markdown("---")
        st.markdown("### Visual Insights: Genotypes & Potential Effects")
        plot_template = "plotly_white" 
        
        c1, c2 = st.columns(2)
        with c1:
            # Genotype Bar Chart
            gen_counts = df["Genotype"].value_counts().reset_index()
            gen_counts.columns = ["Genotype", "Count"]
            fig1 = px.bar(gen_counts, x="Genotype", y="Count", title="Count of Observed Genotypes", template=plot_template, 
                          color_discrete_sequence=["#2979FF"]) # Blue
            fig1.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", title_font_size=14)
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("Interpretation: Visualizing the frequency of homozygous (e.g., AA) vs. heterozygous (e.g., AG) genotypes. The genotype determines the resulting effect.")
            
        with c2:
            # Effect Pie Chart
            eff_counts = df["Effect"].value_counts().reset_index()
            eff_counts.columns = ["Effect", "Count"]
            fig2 = px.pie(eff_counts, names="Effect", values="Count", title="Distribution of Predicted PGx Effects",
                          color="Effect",
                          color_discrete_map=STATUS_COLORS, template=plot_template)
            fig2.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=1)))
            fig2.update_layout(paper_bgcolor="#FFFFFF", title_font_size=14)
            # FIX: Reduced height for Pie Chart
            st.plotly_chart(fig2, use_container_width=True, height=280) 
            st.caption("Interpretation: A critical view on the most significant effects (Poor/High Risk) influencing drug metabolism and disease risk for this patient.")
            
        # Treemap
        treemap_df = df.groupby(["Gene","Effect"]).size().reset_index(name="count")
        fig3 = px.treemap(treemap_df, path=[px.Constant("All Genes"), "Gene","Effect"], values="count", title="Hierarchical View: Gene → Effect", 
                          template=plot_template, color="Effect", color_discrete_map=STATUS_COLORS)
        # FIX: Adjusted font size and significantly reduced height for better flow
        fig3.update_layout(paper_bgcolor="#FFFFFF", margin=dict(t=30, l=10, r=10, b=10), title_font_size=14, uniformtext_minsize=10, uniformtext_mode='hide') 
        st.plotly_chart(fig3, use_container_width=True, height=250)
        st.caption("Interpretation: Shows which genes (CYP2C19/APOE) are associated with the most concerning effects. Larger blocks indicate more SNPs contributing to that specific effect.")
        
        st.markdown("---")
        # Medication recommendations
        st.markdown("### 💊 Explainable Medication Guidance")
        meds = recommend_meds(status)
        cols = st.columns(len(meds))
        for i, (gene, rec) in enumerate(meds.items()):
            current_status = status.get(gene, "Unknown")
            bg = STATUS_COLORS.get(current_status, "#9E9E9E")
            # Select text color for contrast
            if bg in ["#F44336", "#1565C0"]:
                text_color = "white"
                hr_color = "rgba(255,255,255,0.6)"
            elif bg in ["#FFC107", "#4CAF50"]:
                text_color = "#333333"
                hr_color = "rgba(0,0,0,0.2)"
            else:
                text_color = "#333333"
                hr_color = "rgba(0,0,0,0.2)"

            cols[i].markdown(
                f"<div class='card' style='background:{bg};color:{text_color}'>"
                f"<strong>{gene}</strong>: <span style='font-size:14px; color:{text_color};'>{current_status} Status</span>"
                f"<hr style='border-top: 1px solid {hr_color}; margin:8px 0;'> "
                f"<div class='med-card-content' style='color: {text_color} !important;'>{rec}</div>"
                f"</div>", 
                unsafe_allow_html=True
            )

        st.markdown("---")
        csv = df.to_csv(index=False)
        st.download_button("Download PGx results (CSV)", csv, file_name=f"pgx_results_{patient_name.replace(' ', '_')}.csv", mime="text/csv")


# ---------- RIGHT: patient card, quick actions ----------
with right_col:
    st.markdown("## 👤 Patient Snapshot")
    
    # Check if there are results to display
    metabolizer_status = infer_metabolizer_from_genotypes(st.session_state.get('pgx_results_list', []))
    
    card_html = f"""
    <div class='card' style='padding:20px;'>
      <div style='display:flex;gap:12px;align-items:center'>
        <div style='width:64px;height:64px;border-radius:50%;background:#E3F2FD;border:2px solid #BBDEFB;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:24px;color:#1565C0'>{patient_name[:1].upper()}</div>
        <div>
          <div style='font-weight:700;font-size:18px'>{patient_name}</div>
          <div class='muted'>Last PGx Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
      </div>
      <hr style='opacity:0.2;margin:15px 0'/>
      <div style='display:grid;grid-template-columns: 1fr 1fr; gap:10px;text-align:center;'>
        <div style='padding:10px;border-radius:8px;background:#F5F5F5;'><strong>Metabolizer</strong><div style='color:{STATUS_COLORS.get(metabolizer_status, '#9E9E9E')}; font-weight:700;'>{metabolizer_status}</div></div>
        <div style='padding:10px;border-radius:8px;background:#F5F5F5;'><strong>Risk Score</strong><div style='color:#00BCD4; font-weight:700;'>{random.uniform(0.05,0.95):.1%}</div></div>
        <div style='padding:10px;border-radius:8px;background:#F5F5F5; grid-column: span 2;'><strong>Follow-up</strong><div class='muted'>24–48h</div></div>
      </div>
      <hr style='opacity:0.2;margin:15px 0'/>
      </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Handle the functional buttons outside the raw HTML block (CLEANUP COMPLETE)
    if st.button("📄 Generate Comprehensive Report", key='comp_report_btn_display'):
        report_content = generate_comprehensive_report(patient_name, results, status)
        st.download_button(
            "Download Comprehensive Report", 
            report_content, 
            file_name=f"Comprehensive_Report_{patient_name}_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
        st.success("Comprehensive Report ready for download!")

    if st.button("🔁 Reset & New Run", key='reset_btn_display'): 
        reset_dashboard()

    st.markdown("---")
    st.markdown("### Quick Actions")
    if st.button("Export full patient JSON"):
        st.success("Patient JSON data exported (simulated).")
    
    if st.button("Export anonymized PGx summary"):
        st.success("Anonymized PGx report generated (simulated).")


# ----------------------------
# --- Footer / Notes ---
# ----------------------------
st.markdown("---")
st.markdown("<div class='muted' style='text-align:center; padding:10px;'>TeleGeno AI Dashboard — Simulation for demonstration purposes. Not for clinical use.</div>", unsafe_allow_html=True)