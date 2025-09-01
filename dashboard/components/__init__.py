"""
Dashboard components module.
Contains modular components for the Forex AI dashboard visualization and analysis.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from forex_ai_dashboard.pipeline.evaluation_metrics import EvaluationMetrics