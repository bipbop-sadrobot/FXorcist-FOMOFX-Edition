import sys
sys.path.insert(0, '.')
import argparse
import json
import time
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

# --- Local imports ---
from forex_ai_dashboard.utils.explainability import ModelExplainer
from forex_ai_dashboard.utils.narrative_report import NarrativeReportGenerator
from forex_ai_dashboard.utils.logger import logger
from memory_system.core import analyze_memory_trends, generate_insights_report
from memory_system.store.sqlite_store import SQLiteStore

# This file contains the code for the Streamlit dashboard.
# It allows users to upload data, analyze it, and visualize the results.
# The dashboard is divided into several pages, each with a specific purpose.


# ===============================================================
# CLI MODE (Batch Analyzer)
# ===============================================================
def cli_mode(args):
    import matplotlib.pyplot as plt
    import numpy as np

    # Load dataset
    if args.data:
        df = pd.read_csv(args.data)
        memory_data = list(zip(df["timestamp"], df["usage"]))
    else:
        # Dummy synthetic data if no file provided
        memory_data = [(time.time() + i*60, 1000 + i*10 + (i % 3) * 50) for i in range(20)]

    # Run analysis
    trend_report = analyze_memory_trends(memory_data)
    insights_report = generate_insights_report(trend_report)
    report_dict = json.loads(insights_report)

    # Export report
    if args.report:
        with open(args.report, "w") as f:
            if args.format == "json":
                json.dump(report_dict, f, indent=2)
            else:
                f.write(str(report_dict))
        print(f"✅ Report saved to {args.report}")

    # Visualization
    if args.visualize:
        df = pd.DataFrame(memory_data, columns=["timestamp", "memory_usage"])
        plt.figure(figsize=(10, 4))
        plt.plot(df["timestamp"], df["memory_usage"], marker="o")
        plt.title("Memory Usage Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Usage")
        plt.tight_layout()
        plt.savefig("memory_usage.png")
        print("📊 Chart saved to memory_usage.png")

    # Print to console
    print(json.dumps(report_dict, indent=2))


# ===============================================================
# DASHBOARD MODE (Streamlit)
# ===============================================================
def dashboard_mode():
    # --- Dashboard Config ---
    if "page_config_set" not in st.session_state:
        st.set_page_config(page_title="Forex AI Dashboard", layout="wide")
        st.session_state.page_config_set = True

    # --- Initialize Session State ---
    defaults = {
        "model": None,
        "feature_names": None,
        "data": None,
        "dark_mode": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # --- Sidebar ---
    st.sidebar.title("⚙️ Dashboard Controls")

    uploaded_file = st.sidebar.file_uploader("Upload CSV data", type=["csv"])
    if uploaded_file:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.session_state.feature_names = list(st.session_state.data.columns)

    uploaded_model = st.sidebar.file_uploader("Upload Model (pkl)", type=["pkl"])
    if uploaded_model:
        st.session_state.model = "Loaded_Model_Object"  # TODO: load properly

    st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode, key="dark_mode")

    page = st.sidebar.radio("📑 Pages", [
        "Memory Analysis",
        "Model Explainability",
        "Performance Metrics",
        "What-If Analysis",
        "Trading Insights",
        "Dataset Analysis",
    ])

    # Utility
    def export_report(text):
        return BytesIO(text.encode("utf-8"))

    # --- Memory Analysis ---
    if page == "Memory Analysis":
        st.title("🧠 Memory Analysis Insights")
        try:
            num_entries = 15
            memory_data = [(time.time() + i, i * 15.0) for i in range(num_entries)]

            trend_report = analyze_memory_trends(memory_data)
            insights_report = generate_insights_report(trend_report)
            report_dict = json.loads(insights_report)

            st.write("### Memory Trend Analysis")
            st.json(report_dict)

            if report_dict["potential_memory_leak"]:
                st.error("🚨 Potential Memory Leak Detected!")
            elif report_dict["sustained_downward_trend"]:
                st.warning("⚠️ Sustained downward trend observed")

            df = pd.DataFrame(memory_data, columns=["timestamp", "memory_usage"])
            fig = px.line(df, x="timestamp", y="memory_usage", title="Memory Usage Over Time")
            st.plotly_chart(fig, use_container_width=True)

            st.download_button("📥 Download Report",
                               export_report(json.dumps(report_dict, indent=2)),
                               "memory_report.json")

        except Exception as e:
            logger.exception(f"Memory analysis error: {e}")
            st.error(f"❌ Error during memory analysis: {e}")

    # --- Model Explainability ---
    elif page == "Model Explainability":
        st.title("🔍 Model Explainability")
        if st.session_state.model and st.session_state.data is not None:
            try:
                explainer = ModelExplainer(st.session_state.model, st.session_state.feature_names)
                report_gen = NarrativeReportGenerator(st.session_state.model, st.session_state.feature_names)

                with st.spinner("Generating explanations..."):
                    explanation = explainer.explain(st.session_state.data)
                    report = report_gen.generate_report(st.session_state.data)

                st.subheader("Feature Importance")
                st.plotly_chart(report["plots"]["feature_importance"], use_container_width=True)

                feature_choice = st.selectbox("🔎 Drill down on feature", st.session_state.feature_names)
                fig_hist = px.histogram(st.session_state.data, x=feature_choice, title=f"Distribution of {feature_choice}")
                st.plotly_chart(fig_hist, use_container_width=True)

                st.write(report["summary"])

                st.subheader("Instance-level Explanations")
                instance_idx = st.selectbox("Select instance", range(len(st.session_state.data)))
                instance = st.session_state.data.iloc[[instance_idx]]
                instance_report = report_gen.generate_instance_report(instance)
                st.write(instance_report["summary"])

            except Exception as e:
                logger.exception(f"Explainability error: {e}")
                st.error(f"❌ Explainability failed: {e}")
        else:
            st.warning("Please upload a model and dataset first")

    # --- Performance Metrics ---
    elif page == "Performance Metrics":
        st.title("📈 Model Performance Metrics")
        if st.session_state.data is not None:
            metrics = {
                "RMSE": 0.123,
                "Sharpe Ratio": 1.85,
                "Max Drawdown": -0.12,
                "Accuracy": "92%"
            }
            st.json(metrics)
            df = st.session_state.data.copy()
            if "actual" in df.columns and "predicted" in df.columns:
                fig = px.line(df, y=["actual", "predicted"], title="Predictions vs Actual")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Upload data with 'actual' and 'predicted' columns")

    # --- What-If Analysis ---
    elif page == "What-If Analysis":
        st.title("🧪 What-If Analysis")
        if st.session_state.data is not None and st.session_state.feature_names:
            sliders = {}
            st.write("Adjust features to simulate new predictions:")
            for feature in st.session_state.feature_names[:5]:
                val = float(st.session_state.data[feature].mean())
                sliders[feature] = st.slider(f"{feature}",
                                             float(st.session_state.data[feature].min()),
                                             float(st.session_state.data[feature].max()),
                                             val)
            st.json(sliders)
            st.info("⚡ Connect this to model.predict() to see hypothetical predictions.")
        else:
            st.warning("Please upload data to enable What-If analysis")

    # --- Trading Insights ---
    elif page == "Trading Insights":
        st.title("💹 Trading Insights")
        query = st.text_input("Ask a question (e.g., 'Show me top features for EUR/USD')")
        if query:
            st.write(f"🔎 Interpreted query: {query}")
            st.success("💡 (Prototype) Natural language queries could route to explainability/metrics functions")

        st.write("📡 Realtime Forex Data (placeholder)")
        st.info("Connect to AlphaVantage/Oanda API for live streaming predictions here.")

    # --- Dataset Analysis ---
    elif page == "Dataset Analysis":
        st.title("📊 Dataset Analysis")
        uploaded_file = st.file_uploader("Upload dataset", type=["csv", "parquet", "xls", "xlsx", "json"])
        if uploaded_file is not None:
            try:
                file_extension = uploaded_file.name.split(".")[-1].lower()
                if file_extension == "csv":
                    df = pd.read_csv(uploaded_file)
                elif file_extension == "parquet":
                    df = pd.read_parquet(uploaded_file)
                elif file_extension in ["xls", "xlsx"]:
                    df = pd.read_excel(uploaded_file)
                elif file_extension == "json":
                    df = pd.read_json(uploaded_file)
                else:
                    raise ValueError("Unsupported file format")

                # Analyze the dataset
                from forex_ai_dashboard.pipeline import analyze_dataset
                # The analyze_dataset function expects a file path and a file object.
                # We pass the filename for reporting purposes and the file object for reading the data.
                # The file object is created using io.BytesIO to handle different file types.
                analysis_results = analyze_dataset(
                    file_path=uploaded_file.name,
                    file_obj=io.BytesIO(uploaded_file.getvalue()),
                    config_path=None,
                    backend="pandas",
                    workers=1,
                    sample=None,
                    plots=False,
                    report_format="json",
                    outdir="reports"
                )

                st.write("### Analysis Results")
                st.json(analysis_results)

            except Exception as e:
                st.error(f"Error during analysis: {e}")

    st.sidebar.markdown("---")
    st.sidebar.write("📥 Export full session report")
    if st.button("Generate Session Report"):
        session_report = {
            "model": str(st.session_state.model),
            "features": st.session_state.feature_names,
            "dark_mode": st.session_state.dark_mode,
            "page": page,
        }
        st.download_button("Download Session Report",
                           export_report(json.dumps(session_report, indent=2)),
                           "session_report.json")


# ===============================================================
# MAIN ENTRY
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forex AI Dashboard & CLI Analyzer")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of Streamlit dashboard")
    parser.add_argument("--data", type=str, help="Path to CSV data file")
    parser.add_argument("--report", type=str, help="Where to save the analysis report")
    parser.add_argument("--format", type=str, choices=["json", "text"], default="json")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization in CLI mode")
    args, unknown = parser.parse_known_args()

    if args.cli:
        cli_mode(args)
    else:
        from streamlit.runtime import Runtime
        if not Runtime.exists():
            import streamlit.web.cli as stcli
            import sys
            sys.argv = ["streamlit", "run", sys.argv[0]] + unknown
            sys.exit(stcli.main())
