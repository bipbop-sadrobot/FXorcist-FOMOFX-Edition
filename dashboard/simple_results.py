#!/usr/bin/env python3
"""
Simple dashboard to display training results
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
import glob

st.set_page_config(
    page_title="Training Results Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("🎉 Massive Scale Training Results Dashboard")
st.markdown("---")

# Find the latest training results
logs_dir = Path("logs")
result_files = list(logs_dir.glob("comprehensive_training_results_*.json"))

if result_files:
    # Get the most recent file
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)

    st.success(f"📁 Loading results from: {latest_file.name}")

    try:
        with open(latest_file, 'r') as f:
            results = json.load(f)

        # Display key metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Best Model", results['stages']['model_comparison']['best_model'])

        with col2:
            total_time = results.get('total_time_seconds', 0)
            st.metric("Training Time", ".2f")

        with col3:
            st.metric("Success", "✅" if results.get('success') else "❌")

        st.markdown("---")

        # Configuration
        st.subheader("⚙️ Training Configuration")
        config = results.get('configuration', {})
        st.json(config)

        # Data preparation results
        st.subheader("📊 Data Preparation")
        data_prep = results['stages'].get('data_preparation', {})
        if data_prep.get('success'):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Features", data_prep.get('feature_count', 0))
            with col2:
                st.metric("Training Samples", data_prep.get('train_samples', 0))
            with col3:
                st.metric("Test Samples", data_prep.get('test_samples', 0))

        # Model training results
        st.subheader("🤖 Model Training")
        training = results['stages'].get('model_training', {})
        models = training.get('models', {})

        if models:
            st.write("**Trained Models:**")
            for model_name, model_info in models.items():
                st.write(f"- {model_name}")

        # Model comparison
        st.subheader("⚖️ Model Comparison")
        comparison = results['stages'].get('model_comparison', {})
        st.write(f"**Best Model:** {comparison.get('best_model', 'N/A')}")
        st.write(f"**Available Models:** {comparison.get('available_models', [])}")

        # Interpretability
        st.subheader("🔍 Model Interpretability")
        interp = results['stages'].get('interpretability', {})
        st.write(f"**Models Interpreted:** {interp.get('models_interpreted', [])}")

        # Raw results
        with st.expander("📄 Raw Results JSON"):
            st.json(results)

    except Exception as e:
        st.error(f"Error loading results: {e}")

else:
    st.warning("No training result files found in logs directory")

    # Show available files
    st.subheader("Available Files in logs/")
    all_files = list(logs_dir.glob("*"))
    if all_files:
        for file in sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
            st.write(f"- {file.name} ({file.stat().st_mtime})")
    else:
        st.write("No files found")