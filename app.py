# app.py
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
    "Warning": "#ffbb33",
    "Normal": "#00C851"
}

# ----------------------------
# --- Page & Theme Setup ---
# ----------------------------
st.set_page_config(page_title="MetaTele AI — TeleGeno Dashboard", layout="wide", initial_sidebar_state="expanded")
# header image path from uploaded developer asset; if missing, header will show title only
HEADER_IMAGE_PATH = "/mnt/data/afa34478-4816-428b-ab5a-f837ff66e2c8.png"

# dark theme injection for sleek look
def _inject_css(dark=True):
    base = """
    <style>
      .stApp, .block-container { background-color:#080808 !important; color:#e8e8e8 !important; }
      .sidebar .block-container { background-color:#0b0b0b !important; color:#e8e8e8 !important; }
      .patient-capsule { display:inline-block; padding:8px 16px; border-radius:999px; background:#111; color:#fff; font-weight:700; border:1px solid #222; }
      .card { background:#0b0b0b; padding:14px; border-radius:12px; border:1px solid #1a1a1a; }
      .small-metric { font-size:22px; font-weight:700; }
      .muted { color:#a6a6a6; font-size:13px; }
      .plotly-graph-div .modebar-btn { background: #111 !important; }
      .header-logo { height:48px; object-fit:contain; border-radius:6px; }
    </style>
    """
    st.markdown(base, unsafe_allow_html=True)

_inject_css()

# ----------------------------
# --- Sidebar: inputs ---
# ----------------------------
st.sidebar.markdown("### TeleGeno AI — Inputs & Controls")
patient_name = st.sidebar.text_input("Patient Name", value="John Doe")
input_type = st.sidebar.radio("Input Type", ["JSON SNP Input", "Simulated VCF", "Demographics + History (AI)"])
st.sidebar.markdown("---")
emergency_tab = st.sidebar.checkbox("Enable Emergency (Webcam)")

sample_json = '{"rs4244285": "AA", "rs429358": "AG"}'
sample_vcf = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
1\t12345\trs4244285\tA\tG\t.\t.\t.
1\t67890\trs429358\tC\tT\t.\t.\t."""

with st.sidebar.expander("Sample JSON"):
    st.code(sample_json)
with st.sidebar.expander("Sample VCF (sim)"):
    st.code(sample_vcf)

st.sidebar.markdown("---")
if st.sidebar.button("Simulate random patient SNPs"):
    st.session_state["_simulate_snps"] = True

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
    critical_pct = 0
    avg_hr = random.randint(60, 95)
    avg_bp = f"{random.randint(110,135)}/{random.randint(70,85)}"
    return total, critical_pct, avg_hr, avg_bp

# ----------------------------
# --- Header ---
# ----------------------------
header_col1, header_col2 = st.columns([8,2])
with header_col1:
    # show logo if available
    try:
        st.image(HEADER_IMAGE_PATH, width=160, caption=None, output_format="auto")
    except Exception:
        st.markdown("<h1 style='margin:0;color:#fff'>MetaTele AI — TeleGeno Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Pharmacogenomic analysis • Emergency webcam triage • Explainable medication guidance</div>", unsafe_allow_html=True)
with header_col2:
    st.markdown(f"<div style='text-align:right'><div class='patient-capsule'>{patient_name}</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# --- Top Metric Cards ---
# ----------------------------
total, critical_pct, avg_hr, avg_bp = metrics_from_results({})

mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.markdown(f"<div class='card'><div class='small-metric'>{total}</div><div class='muted'>Total Patients</div></div>", unsafe_allow_html=True)
mcol2.markdown(f"<div class='card'><div class='small-metric'>{critical_pct}%</div><div class='muted'>Critical Now</div></div>", unsafe_allow_html=True)
mcol3.markdown(f"<div class='card'><div class='small-metric'>{avg_hr} bpm</div><div class='muted'>Avg Heart Rate</div></div>", unsafe_allow_html=True)
mcol4.markdown(f"<div class='card'><div class='small-metric'>{avg_bp}</div><div class='muted'>Avg Blood Pressure</div></div>", unsafe_allow_html=True)

st.markdown("")

# ----------------------------
# --- Main two-column layout ---
# ----------------------------
left_col, right_col = st.columns([3,1])

# ---------- LEFT: charts & analysis ----------
with left_col:
    # Input flows + analysis
    snps_input = {}
    results = []
    status = {}

    if st.session_state.get("_simulate_snps"):
        all_snps = [s for gene in SNP_EFFECTS for s in SNP_EFFECTS[gene] if s not in ("Metabolite_Impact", "Risk_Impact")]
        snps_input = {snp: random.choice(["AA", "AG", "GG"]) for snp in all_snps}
        results, status = analyze_snps(snps_input)
        st.session_state["_simulate_snps"] = False

    st.markdown("## Inputs & PGx Analysis", unsafe_allow_html=True)
    if input_type == "Demographics + History (AI)":
        with st.form("demo_form"):
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
                st.success(f"AI Risk Score: {ai_score:.2%} — Predicted metabolizer: {metabolizer_pred}")
                # create demo results summary
                results = [{"Gene":"CYP2C19","SNP ID":"rs4244285","Genotype":"AA","Effect":metabolizer_pred}]
                status = {"CYP2C19":metabolizer_pred}
    elif input_type == "JSON SNP Input":
        st.markdown("Paste JSON SNP object (e.g. {\"rs4244285\":\"AA\"})")
        json_data = st.text_area("JSON SNP input", value=sample_json, height=120)
        if st.button("Analyze JSON"):
            parsed = parse_json_input(json_data)
            if parsed:
                snps_input = parsed
                results, status = analyze_snps(snps_input)
                st.success("JSON parsed and analyzed.")
            else:
                st.error("Invalid JSON.")
    else:
        uploaded_file = st.file_uploader("Upload VCF-like file", type=['vcf','txt'])
        colA, colB = st.columns([1,3])
        with colA:
            if st.button("Simulate example VCF"):
                snps_input = parse_vcf_simulator(sample_vcf)
                results, status = analyze_snps(snps_input)
                st.success("Simulated VCF parsed.")
        with colB:
            if uploaded_file:
                content = uploaded_file.getvalue().decode(errors="ignore")
                snps_input = parse_vcf_simulator(content)
                results, status = analyze_snps(snps_input)
                st.success("Uploaded VCF parsed (simulated).")

    # Display PGx results
    st.markdown("### Pharmacogenomic Summary")
    df = results_to_df(results)
    if df.empty:
        st.info("No SNP results to display. Use an input method or simulate.")
    else:
        st.dataframe(df, use_container_width=True, height=220)
        # plots
        plot_template = "plotly_dark"
        c1, c2 = st.columns(2)
        with c1:
            gen_counts = df["Genotype"].value_counts().reset_index()
            gen_counts.columns = ["Genotype", "Count"]
            fig1 = px.bar(gen_counts, x="Genotype", y="Count", title="Genotype Counts", template=plot_template)
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            eff_counts = df["Effect"].value_counts().reset_index()
            eff_counts.columns = ["Effect", "Count"]
            fig2 = px.pie(eff_counts, names="Effect", values="Count", title="Effect Distribution",
                          color_discrete_map=STATUS_COLORS, template=plot_template)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        # treemap
        treemap_df = df.groupby(["Gene","Effect"]).size().reset_index(name="count")
        fig3 = px.treemap(treemap_df, path=["Gene","Effect"], values="count", title="Gene → Effect Treemap", template=plot_template)
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True, height=420)

        # medication recommendations
        st.markdown("#### Medication Guidance")
        meds = recommend_meds(status)
        cols = st.columns(len(meds))
        for i, (gene, rec) in enumerate(meds.items()):
            bg = STATUS_COLORS.get(status.get(gene, "Unknown"), "#95a5a6")
            cols[i].markdown(f"<div style='padding:10px;border-radius:8px;background:{bg};color:#000'>"
                             f"<strong>{gene}</strong><br/><small>Status: {status.get(gene,'Unknown')}</small><hr style='opacity:0.6'>{rec}</div>", unsafe_allow_html=True)

        csv = df.to_csv(index=False)
        st.download_button("Download PGx results (CSV)", csv, file_name="pgx_results.csv", mime="text/csv")

# ---------- RIGHT: patient card, emergency, quick actions ----------
with right_col:
    st.markdown("## Patient Card")
    card_html = f"""
    <div class='card'>
      <div style='display:flex;gap:12px;align-items:center'>
        <div style='width:72px;height:72px;border-radius:10px;background:#111;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:28px'>{patient_name[:1].upper()}</div>
        <div>
          <div style='font-weight:700;font-size:16px'>{patient_name}</div>
          <div class='muted'>Last seen: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
      </div>
      <hr style='opacity:0.08;margin:12px 0'/>
      <div style='display:flex;gap:8px;flex-wrap:wrap'>
        <div style='padding:8px;border-radius:8px;background:#0f0f0f;min-width:120px'><strong>Metabolizer</strong><div style='color:#ffd36b'>{infer_metabolizer_from_genotypes(results)}</div></div>
        <div style='padding:8px;border-radius:8px;background:#0f0f0f;min-width:120px'><strong>Risk Score</strong><div style='color:#9fe7a4'>{random.uniform(0.05,0.95):.1%}</div></div>
        <div style='padding:8px;border-radius:8px;background:#0f0f0f;min-width:120px'><strong>Follow-up</strong><div class='muted'>24–48h</div></div>
      </div>
      <hr style='opacity:0.08;margin:12px 0'/>
      <div style='display:flex;gap:8px'>
        <form action="#">
        </form>
        <a><button style='padding:8px 10px;border-radius:8px;background:#111;color:#fff;border:1px solid #222'>📄 Patient Summary</button></a>
        <a><button style='padding:8px 10px;border-radius:8px;background:#0b7dda;color:#fff;border:1px solid #066ca0'>🔁 Re-run</button></a>
      </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("### Emergency Assessment")
    if emergency_tab:
        if not _HAS_CV2:
            st.warning("OpenCV not installed — webcam emergency features require opencv-python. Continue without webcam.")
            st.button("Generate simulated emergency report")
        else:
            cam = st.camera_input("Capture patient photo")
            if cam:
                bytes_data = cam.getvalue()
                img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                # mock vitals extraction
                vitals = {
                    "hr": random.randint(60,120),
                    "bp_sys": random.randint(100,170),
                    "bp_dia": random.randint(60,110)
                }
                st.metric("Heart Rate", f"{vitals['hr']} bpm")
                st.metric("Blood Pressure", f"{vitals['bp_sys']}/{vitals['bp_dia']} mmHg")
                # quick emergency summary
                status_overall = "Normal"
                if vitals['hr']>130 or vitals['bp_sys']>180:
                    status_overall = "Critical"
                elif vitals['hr']>100 or vitals['bp_sys']>140:
                    status_overall = "Warning"
                color = EMERGENCY_STATUS_COLORS.get(status_overall, "#95a5a6")
                st.markdown(f"<div class='card'><strong style='color:{color}'>{status_overall}</strong><p class='muted'>Automated emergency triage (simulated)</p></div>", unsafe_allow_html=True)
                # allow download of short report
                report = f"Emergency Report - {patient_name}\nTime: {datetime.now()}\nHR: {vitals['hr']} bpm\nBP: {vitals['bp_sys']}/{vitals['bp_dia']}\nMetabolizer: {infer_metabolizer_from_genotypes(results)}\nNotes: Simulated"
                st.download_button("Download Emergency Report", report, file_name=f"emergency_{patient_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    else:
        st.info("Enable Emergency mode from sidebar to use webcam triage (optional).")

    st.markdown("---")
    st.markdown("### Quick actions")
    st.button("Export patient JSON")
    st.button("Export anonymized PGx report")

# ----------------------------
# --- Footer / Notes ---
# ----------------------------
st.markdown("---")
st.markdown("<div class='muted'>Simulation dashboard for demo and development. Not for clinical use.</div>", unsafe_allow_html=True)
