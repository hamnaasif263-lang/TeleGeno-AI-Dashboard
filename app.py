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

# ----------------------------
# --- Page & Theme Setup ---
# ----------------------------
# Set page config for a general light theme look (Streamlit default)
st.set_page_config(page_title="TeleGeno AI — Patient Dashboard", layout="wide", initial_sidebar_state="expanded")
# Note: HEADER_IMAGE_PATH is likely invalid in the execution environment, so we'll rely on the title.

# Light theme injection for sleek look
def _inject_css(light=True):
    # This CSS overrides Streamlit defaults to achieve a cleaner, lighter look
    base = """
    <style>
      /* Main App Background - Slightly off-white */
      .stApp { background-color:#F5F5F5; }
      /* Sidebar Background - White */
      .stSidebar { background-color:#FFFFFF; border-right: 1px solid #EEEEEE; }
      /* Main Content Containers */
      .block-container { color:#333333; }
      
      /* Card Styling - White with a subtle shadow/border */
      .card { background:#FFFFFF; padding:14px; border-radius:12px; border:1px solid #DDDDDD; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
      
      /* Typography */
      h1, h2, h3, h4, .small-metric { color:#333333; }
      .small-metric { font-size:24px; font-weight:700; }
      .muted { color:#757575; font-size:13px; }
      
      /* Sidebar specific styles */
      .sidebar .stRadio > label { font-weight: 500; }
      
      /* Custom Elements */
      .patient-capsule { display:inline-block; padding:8px 16px; border-radius:999px; background:#BBDEFB; color:#1565C0; font-weight:700; border:1px solid #90CAF9; }
      
      /* Adjust Plotly to blend with light theme */
      .plotly-graph-div .modebar-btn { background: #FFFFFF !important; color: #616161 !important; }
    
      /* Medication Guidance Cards */
      .med-card-content { color: #333333 !important; }
    </style>
    """
    st.markdown(base, unsafe_allow_html=True)

_inject_css(light=True)

# ----------------------------
# --- Helper Functions (No changes in logic) ---
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
    # Produce single summary for CYP2C19 if present, else Unknown
    df = results_to_df(results_list)
    if "CYP2C19" in df["Gene"].values:
        row = df[df["Gene"] == "CYP2C19"]
        # pick worst effect (Poor > Intermediate > Normal)
        order = {"Poor":3, "Intermediate":2, "Normal":1}
        row = row.copy()
        row["score"] = row["Effect"].map(lambda x: order.get(x, 0))
        if not row.empty:
            best = row.sort_values("score", ascending=False).iloc[0]["Effect"]
            return best
    return "Unknown"

# A small function to compute a population-style aggregated metric (for demo)
def metrics_from_results(status):
    total = 1  # single patient demo; in real app this would be dataset length
    # Based on the worst status for the single patient (e.g., if APOE is High Risk)
    critical_pct = 100 if any(s in status.values() for s in ["Poor", "High Risk"]) else 0
    avg_hr = random.randint(60, 95)
    avg_bp = f"{random.randint(110,135)}/{random.randint(70,85)}"
    return total, critical_pct, avg_hr, avg_bp

# ----------------------------
# --- Sidebar: inputs ---
# ----------------------------
st.sidebar.markdown("## 🧬 TeleGeno AI — Patient Inputs")
patient_name = st.sidebar.text_input("Patient Name", value="Ferdoun S.")

# Emergency Checkbox Control
emergency_tab = st.sidebar.checkbox("🚨 Enable Emergency Assessment")
st.sidebar.markdown("---")

# Only show PGx inputs if Emergency is NOT enabled
if not emergency_tab:
    st.sidebar.markdown("### Genomic Data Input Method")
    input_type = st.sidebar.radio("Select Input Type", 
                                  ["JSON SNP Input", "Simulated VCF", "Demographics + History (AI)"], 
                                  key="_input_type")
    st.sidebar.markdown("---")
    
    # Logic to simulate random SNPs
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

# ----------------------------
# --- Emergency Screen Takeover ---
# ----------------------------
if emergency_tab:
    # Use the entire screen for emergency
    st.markdown("## 🚨 Live Emergency Triage", unsafe_allow_html=True)
    
    # Use two-columns for better layout on the full page
    e_col1, e_col2 = st.columns([2, 1])

    with e_col1:
        if not _HAS_CV2:
            st.error("OpenCV not installed: Webcam emergency features are disabled. Install `opencv-python` to enable.")
            st.button("Generate Simulated Vitals & Report")
        else:
            st.markdown("### Step 1: Capture Patient Vitals via Webcam")
            cam = st.camera_input("Capture patient photo for simulated vitals check")
            
            if cam:
                bytes_data = cam.getvalue()
                img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                # mock vitals extraction
                vitals = {
                    "hr": random.randint(60,120),
                    "bp_sys": random.randint(100,170),
                    "bp_dia": random.randint(60,110)
                }
                
                # quick emergency summary
                status_overall = "Normal"
                if vitals['hr']>130 or vitals['bp_sys']>180:
                    status_overall = "Critical"
                elif vitals['hr']>100 or vitals['bp_sys']>140:
                    status_overall = "Warning"
                
                st.markdown("---")
                st.markdown("### Step 2: Vitals & Triage Summary")
                
                v1, v2, v3 = st.columns(3)
                v1.metric("Heart Rate", f"**{vitals['hr']}** bpm", delta_color="normal")
                v2.metric("Blood Pressure", f"**{vitals['bp_sys']}/{vitals['bp_dia']}** mmHg", delta_color="off")
                
                color = EMERGENCY_STATUS_COLORS.get(status_overall, "#9E9E9E")
                v3.markdown(f"<div class='card' style='background-color: {color}; color: white;'>"
                            f"<strong style='font-size: 20px;'>{status_overall.upper()}</strong>"
                            f"<p style='margin: 0; font-size: 12px;'>Automated Triage</p>"
                            f"</div>", unsafe_allow_html=True)
                
                # Report generation (assuming current results from session state if available)
                report = f"Emergency Report - {patient_name}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                report += f"Triage Status: {status_overall}\n"
                report += f"HR: {vitals['hr']} bpm\nBP: {vitals['bp_sys']}/{vitals['bp_dia']} mmHg\n"
                
                # Add PGx info if available
                current_results = st.session_state.get('pgx_results_list', [])
                current_status = st.session_state.get('pgx_status', {})
                if current_status:
                    metabolizer = infer_metabolizer_from_genotypes(current_results)
                    report += f"Metabolizer Status (CYP2C19): {metabolizer}\n"
                    for gene, rec in recommend_meds(current_status).items():
                        report += f"PGx Guidance ({gene}): {rec}\n"
                
                st.markdown("### Step 3: Finalize Report")
                st.info("Based on the vitals, immediate action may be required. Review PGx guidance if available.")
                st.download_button(
                    "Download Emergency Report", 
                    report, 
                    file_name=f"emergency_{patient_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                )
            else:
                st.info("Awaiting webcam input to perform live triage...")
    
    with e_col2:
        st.markdown("### Quick Guide")
        st.markdown(
            """
            * **CRITICAL (Red):** Immediate medical intervention needed.
            * **WARNING (Orange):** Close monitoring and specialist consult advised.
            * **NORMAL (Green):** Stable condition.
            
            This screen prioritizes **vitals-based assessment** before full PGx review.
            """
        )

    st.markdown("---")
    st.warning("Deactivate 'Enable Emergency Assessment' in the sidebar to return to the main dashboard.")
    st.stop() # Stops execution of the rest of the main dashboard when emergency is activest.markdown("---")
    st.warning("Deactivate 'Enable Emergency Assessment' in the sidebar to return to the main PGx dashboard.")
    # THIS LINE MUST BE st.stop()
    st.stop()


# ----------------------------
# --- Main Dashboard (Not Emergency) ---
# ----------------------------

# ----------------------------
# --- Top Metric Cards ---
# ----------------------------
total, critical_pct, avg_hr, avg_bp = metrics_from_results(st.session_state.get('pgx_status', {}))

mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.markdown(f"<div class='card'><div class='muted'>Total Patients</div><div class='small-metric'>{total}</div></div>", unsafe_allow_html=True)
mcol2.markdown(f"<div class='card' style='border-left: 5px solid {EMERGENCY_STATUS_COLORS['Critical'] if critical_pct > 0 else '#EEEEEE'};'><div class='muted'>Critical PGx Flag</div><div class='small-metric' style='color:{EMERGENCY_STATUS_COLORS['Critical'] if critical_pct > 0 else '#4CAF50'}'>{critical_pct}%</div></div>", unsafe_allow_html=True)
mcol3.markdown(f"<div class='card'><div class='muted'>Avg Heart Rate (Pop)</div><div class='small-metric'>{avg_hr} bpm</div></div>", unsafe_allow_html=True)
mcol4.markdown(f"<div class='card'><div class='muted'>Avg Blood Pressure (Pop)</div><div class='small-metric'>{avg_bp}</div></div>", unsafe_allow_html=True)

st.markdown("")

# ----------------------------
# --- Main two-column layout ---
# ----------------------------
left_col, right_col = st.columns([3,1])

# ---------- LEFT: charts & analysis ----------
with left_col:
    st.markdown("## 🔬 PGx Analysis & Visualization", unsafe_allow_html=True)
    
    # Input flows + analysis
    snps_input = {}
    results = []
    status = {}

    if st.session_state.get("_simulate_snps"):
        all_snps = [s for gene in SNP_EFFECTS for s in SNP_EFFECTS[gene] if s not in ("Metabolite_Impact", "Risk_Impact")]
        snps_input = {snp: random.choice(["AA", "AG", "GG"]) for snp in all_snps}
        results, status = analyze_snps(snps_input)
        st.session_state["_simulate_snps"] = False
        st.session_state["pgx_results_list"] = results # Save to session state
        st.session_state["pgx_status"] = status
    
    # Check session state for existing data
    if "pgx_results_list" in st.session_state and not st.session_state["pgx_results_list"]:
        # Only process input if no data is present OR the input type is the demographics AI
        
        # --- Input Section ---
        st.markdown("### 📥 Load Patient Genomic Data")
        if st.session_state.get("_input_type") == "Demographics + History (AI)":
            with st.form("demo_form", clear_on_submit=False):
                st.markdown("Fill out demographics for **AI risk prediction** (simulated PGx inference).")
                c1, c2, c3 = st.columns(3)
                with c1:
                    age = st.number_input("Age", 0, 120, 55)
                    weight = st.number_input("Weight (kg)", 20, 200, 70)
                    smoking = st.checkbox("Smoking")
                with c2:
                    family_history = st.checkbox("Family history")
                    temp = st.number_input("Temperature (°C)", 34.0, 42.0, 36.6, step=0.1)
                    bp_sys = st.number_input("Systolic BP", 80, 220, 120)
                with c3:
                    bp_dia = st.number_input("Diastolic BP", 40, 140, 80)
                submitted = st.form_submit_button("Run AI Prediction")
                if submitted:
                    ai_score = random.uniform(0.02, 0.95)
                    metabolizer_pred = "Poor" if ai_score>0.7 else ("Intermediate" if ai_score>0.4 else "Normal")
                    st.success(f"AI Risk Score: **{ai_score:.2%}** — Predicted metabolizer: **{metabolizer_pred}**")
                    # create demo results summary
                    results = [{"Gene":"CYP2C19","SNP ID":"rs4244285","Genotype":"AA","Effect":metabolizer_pred}]
                    status = {"CYP2C19":metabolizer_pred}
                    st.session_state["pgx_results_list"] = results 
                    st.session_state["pgx_status"] = status
                    st.rerun() # Rerun to refresh dashboard metrics immediately
        
        elif st.session_state.get("_input_type") == "JSON SNP Input":
            st.markdown("Paste JSON SNP object (e.g. `{\"rs4244285\":\"AA\"}`) in the text area.")
            json_data = st.text_area("JSON SNP input", value=sample_json, height=120)
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
                    st.error("Invalid JSON.")
        
        else: # Simulated VCF
            uploaded_file = st.file_uploader("Upload VCF-like file (Simulated Parsing)", type=['vcf','txt'])
            colA, colB = st.columns([1,3])
            with colA:
                if st.button("Simulate example VCF"):
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
                    st.success("Uploaded VCF parsed (simulated).")
                    st.rerun()
    
    # Retrieve current results from session state
    results = st.session_state.get('pgx_results_list', [])
    status = st.session_state.get('pgx_status', {})
    
    # Display PGx results
    st.markdown("### SNP Genotype and Effect Table")
    df = results_to_df(results)
    if df.empty:
        st.info("No SNP results to display. Please use an input method from the sidebar or click 'Simulate random patient SNPs'.")
    else:
        # st.dataframe(df, use_container_width=True, height=220)
        st.table(df) # Use table for cleaner look with smaller data
        
        # --- Visualizations ---
        st.markdown("---")
        st.markdown("### Visual Insights: Genotypes & Potential Effects")
        plot_template = "plotly_white" # Change to white for light theme
        
        c1, c2 = st.columns(2)
        with c1:
            # Genotype Bar Chart
            gen_counts = df["Genotype"].value_counts().reset_index()
            gen_counts.columns = ["Genotype", "Count"]
            fig1 = px.bar(gen_counts, x="Genotype", y="Count", title="Count of Observed Genotypes", template=plot_template, 
                          color_discrete_sequence=["#2979FF"]) # Blue
            fig1.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", title_font_size=14)
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("Interpretation: Visualizing the frequency of homozygous (e.g., AA) vs. heterozygous (e.g., AG) genotypes.")
            
        with c2:
            # Effect Pie Chart
            eff_counts = df["Effect"].value_counts().reset_index()
            eff_counts.columns = ["Effect", "Count"]
            fig2 = px.pie(eff_counts, names="Effect", values="Count", title="Distribution of Predicted PGx Effects",
                          color="Effect",
                          color_discrete_map=STATUS_COLORS, template=plot_template)
            fig2.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=1)))
            fig2.update_layout(paper_bgcolor="#FFFFFF", title_font_size=14)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Interpretation: A critical view on the most significant effects (Poor/High Risk) influencing drug metabolism and disease risk.")
            
        # Treemap
        treemap_df = df.groupby(["Gene","Effect"]).size().reset_index(name="count")
        fig3 = px.treemap(treemap_df, path=[px.Constant("All Genes"), "Gene","Effect"], values="count", title="Hierarchical View: Gene → Effect", 
                          template=plot_template, color="Effect", color_discrete_map=STATUS_COLORS)
        fig3.update_layout(paper_bgcolor="#FFFFFF", margin=dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(fig3, use_container_width=True, height=350)
        st.caption("Interpretation: Shows which genes (CYP2C19/APOE) are associated with the most concerning effects. Larger blocks indicate more SNPs contributing to that effect.")
        
        st.markdown("---")
        # Medication recommendations
        st.markdown("### 💊 Explainable Medication Guidance")
        meds = recommend_meds(status)
        cols = st.columns(len(meds))
        for i, (gene, rec) in enumerate(meds.items()):
            bg = STATUS_COLORS.get(status.get(gene, "Unknown"), "#9E9E9E")
            text_color = "white" if bg in ["#F44336", "#1976D2"] else "#333333" # Ensure contrast
            cols[i].markdown(
                f"<div class='card' style='background:{bg};color:{text_color}'>"
                f"<strong>{gene}</strong>: <span style='font-size:14px;'>{status.get(gene,'Unknown')} Status</span>"
                f"<hr style='border-top: 1px solid rgba(255,255,255,0.6); margin:8px 0;'> "
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
        <div style='padding:10px;border-radius:8px;background:#F5F5F5;'><strong>Follow-up</strong><div class='muted'>24–48h</div></div>
        <div style='padding:10px;border-radius:8px;background:#F5F5F5;'><strong>Prescriber</strong><div class='muted'>Dr. A. Khan</div></div>
      </div>
      <hr style='opacity:0.2;margin:15px 0'/>
      <div style='display:flex;gap:10px;flex-direction:column;'>
        <a href="#"><button style='width:100%;padding:10px;border-radius:8px;background:#03A9F4;color:#fff;border:none;font-weight:600;cursor:pointer;'>📄 Generate Comprehensive Report</button></a>
        <a href="#"><button style='width:100%;padding:10px;border-radius:8px;background:#EEEEEE;color:#333;border:1px solid #DDDDDD;font-weight:600;cursor:pointer;'>🔁 Reset & New Run</button></a>
      </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Quick Actions")
    if st.button("Export patient JSON"):
        # Add logic to serialize and download patient data
        st.success("Patient JSON data exported (simulated).")
    
    if st.button("Export anonymized PGx report"):
        # Add logic to generate and download anonymized report
        st.success("Anonymized PGx report generated (simulated).")

# ----------------------------
# --- Footer / Notes ---
# ----------------------------
st.markdown("---")
st.markdown("<div class='muted' style='text-align:center; padding:10px;'>TeleGeno AI Dashboard — Simulation for demonstration purposes. Not for clinical use.</div>", unsafe_allow_html=True)