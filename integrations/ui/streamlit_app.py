# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from integrations.reports.quantstats_report import write_quantstats_report

st.set_page_config(layout="wide", page_title="FXorcist Dashboard")
st.title("FXorcist — Model & Report Dashboard")

artifacts = list(Path("integrations/artifacts").glob("*"))
artifact_names = [a.name for a in artifacts]
choice = st.selectbox("Artifact", artifact_names)
if choice.endswith(".html"):
    st.markdown(f"### HTML Report: {choice}")
    file_path = Path("integrations/artifacts") / choice
    st.markdown(f"Open the HTML file locally: `{file_path}`")
else:
    st.write("Select an HTML report to view (QuantStats).")
# small interactive: qos slider to test gating threshold if cate exists
cate_path = st.text_input("Path to cate bundle (joblib)", "integrations/artifacts/cate_bundle.joblib")
if Path(cate_path).exists():
    st.success("Found cate bundle")
    from integrations.fxorcist_integration.policy.policy_overlay import apply_cate_gate
    # Simplified control: threshold slider
    min_tau = st.slider("min_tau", -0.01, 0.05, 0.0, 0.001)
    st.write("Use `apply_cate_gate` in your backtester with min_tau:", min_tau)
else:
    st.info("Place cate bundle at integrations/artifacts/cate_bundle.joblib to enable gating controls.")