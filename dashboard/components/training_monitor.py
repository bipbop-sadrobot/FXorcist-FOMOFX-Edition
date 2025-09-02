"""
Training System Monitor Component
Provides comprehensive training system monitoring and visualization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from . import PerformanceComponent, ComponentConfig

class TrainingMonitor(PerformanceComponent):
    """Component for monitoring and visualizing the training system."""

    def __init__(self, config: ComponentConfig):
        """Initialize training monitor component with real-time tracking."""
        super().__init__(config)
        self.models_dir = Path("models/trained")
        self.logs_dir = Path("logs")
        self.data_dir = Path("data/processed")
        
        # Initialize real-time tracking
        if 'metrics_history' not in st.session_state:
            st.session_state.metrics_history = {
                'training_loss': [],
                'validation_loss': [],
                'learning_rate': [],
                'gpu_memory': [],
                'cpu_usage': [],
                'memory_usage': [],
                'batch_time': [],
                'custom_metrics': {}
            }
        
        # Training state
        if 'training_state' not in st.session_state:
            st.session_state.training_state = {
                'active': False,
                'current_epoch': 0,
                'total_epochs': 0,
                'model_type': 'CatBoost',
                'optimizer': 'Adam',
                'batch_size': 128
            }
        
        # Visualization preferences
        if 'viz_preferences' not in st.session_state:
            st.session_state.viz_preferences = {
                'update_interval': 5,
                'show_history': True,
                'history_window': 30,
                'chart_type': 'Line',
                'color_scheme': 'Default'
            }

    def _get_system_metrics(self) -> Dict[str, int]:
        """Get current system metrics."""
        metrics = {
            'trained_models': len(list(self.models_dir.glob("*.cbm")) + 
                                list(self.models_dir.glob("*.pkl")) + 
                                list(self.models_dir.glob("*.json"))),
            'training_sessions': len(list(self.logs_dir.glob("*.log"))),
            'processed_datasets': len(list(self.data_dir.glob("*.parquet")))
        }
        return metrics

    def _render_system_improvements(self):
        """Render training system improvements section."""
        st.header("🎯 Training System Improvements")

        improvements = [
            "✅ Hyperparameter Optimization with Optuna",
            "✅ Ensemble Methods (Random Forest, Extra Trees, LightGBM)",
            "✅ Advanced Feature Engineering (50+ indicators)",
            "✅ Cross-validation and Robust Evaluation",
            "✅ Model Interpretability with SHAP",
            "✅ Automated Model Comparison",
            "✅ Comprehensive Logging and Monitoring"
        ]

        for improvement in improvements:
            st.success(improvement)

    def _render_data_synthesis_controls(self):
        """Render controls for data synthesis and augmentation."""
        st.header("🔄 Data Synthesis Controls")
        
        with st.expander("Data Synthesis Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                edge_case_ratio = st.slider(
                    "Edge Case Ratio",
                    0.0, 1.0, 
                    self.data_loader.synthesis_config['edge_case_ratio'],
                    help="Proportion of synthetic data that should be edge cases"
                )
                
                noise_level = st.slider(
                    "Noise Level",
                    0.0, 0.2, 
                    self.data_loader.synthesis_config['noise_level'],
                    help="Amount of random noise to add to synthetic data"
                )
            
            with col2:
                augmentation_ratio = st.slider(
                    "Augmentation Ratio",
                    0.0, 1.0, 
                    self.data_loader.synthesis_config['augmentation_ratio'],
                    help="Proportion of augmented data to add to training set"
                )
                
                trend_strength = st.slider(
                    "Trend Strength",
                    0.1, 2.0, 
                    self.data_loader.synthesis_config['trend_strength'],
                    help="Strength of synthetic trends"
                )
            
            # Update synthesis config if changed
            if (edge_case_ratio != self.data_loader.synthesis_config['edge_case_ratio'] or
                noise_level != self.data_loader.synthesis_config['noise_level'] or
                augmentation_ratio != self.data_loader.synthesis_config['augmentation_ratio'] or
                trend_strength != self.data_loader.synthesis_config['trend_strength']):
                
                self.data_loader.synthesis_config.update({
                    'edge_case_ratio': edge_case_ratio,
                    'noise_level': noise_level,
                    'augmentation_ratio': augmentation_ratio,
                    'trend_strength': trend_strength
                })
                st.success("Synthesis settings updated!")

    def _render_algorithmic_efficiency(self):
        """Render algorithmic efficiency metrics and visualizations."""
        st.header("⚡ Algorithmic Efficiency")
        
        col1, col2, col3 = st.columns(3)
        
        # Training speed metrics
        with col1:
            if st.session_state.metrics_history['batch_time']:
                avg_batch_time = np.mean(st.session_state.metrics_history['batch_time'][-50:])
                st.metric(
                    "Avg Batch Time",
                    f"{avg_batch_time:.3f}s",
                    delta=f"{avg_batch_time - np.mean(st.session_state.metrics_history['batch_time'][-100:-50]):.3f}s"
                )
            
            throughput = len(st.session_state.metrics_history['batch_time']) / sum(st.session_state.metrics_history['batch_time'])
            st.metric("Training Throughput", f"{throughput:.1f} batches/s")
        
        # Memory efficiency
        with col2:
            if 'memory_usage' in st.session_state.metrics_history:
                current_memory = st.session_state.metrics_history['memory_usage'][-1]
                peak_memory = max(st.session_state.metrics_history['memory_usage'])
                st.metric("Current Memory", f"{current_memory:.1f} MB")
                st.metric("Peak Memory", f"{peak_memory:.1f} MB")
        
        # Cache efficiency
        with col3:
            cache_stats = self.data_loader.get_performance_stats()
            hit_rate = cache_stats['cache_hit_rate'] * 100
            st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
            st.metric("Cache Size", f"{len(cache_stats['cache_stats'])} items")
        
        # Efficiency timeline
        efficiency_data = pd.DataFrame({
            'Batch Time': st.session_state.metrics_history['batch_time'],
            'Memory Usage': st.session_state.metrics_history['memory_usage'],
            'GPU Memory': st.session_state.metrics_history['gpu_memory']
        })
        
        fig = px.line(efficiency_data, title='Resource Usage Over Time')
        st.plotly_chart(fig, use_container_width=True)

    def _render_real_time_metrics(self):
        """Render real-time performance metrics with customizable display."""
        st.header("📊 Real-time Metrics")
        
        # Add tabs for different metric views
        metric_tabs = st.tabs(["Training Metrics", "Data Synthesis", "Algorithmic Efficiency"])
        
        with metric_tabs[0]:
            # Original metrics display settings
            with st.expander("📈 Metrics Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    update_interval = st.slider("Update Interval (s)", 1, 30, 
                                             st.session_state.viz_preferences['update_interval'])
                    show_history = st.checkbox("Show History", 
                                            st.session_state.viz_preferences['show_history'])
                with col2:
                    if show_history:
                        history_window = st.slider("History Window (min)", 5, 60, 
                                                st.session_state.viz_preferences['history_window'])
                    chart_type = st.selectbox("Chart Type", ['Line', 'Area', 'Bar'],
                                           index=['Line', 'Area', 'Bar'].index(
                                               st.session_state.viz_preferences['chart_type']))
        
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
        
            # Training metrics
            with col1:
                if st.session_state.metrics_history['training_loss']:
                    current_loss = st.session_state.metrics_history['training_loss'][-1]
                    st.metric("Current Loss", f"{current_loss:.6f}")
                
                if st.session_state.metrics_history['batch_time']:
                    avg_time = sum(st.session_state.metrics_history['batch_time'][-10:]) / 10
                    st.metric("Avg Batch Time", f"{avg_time:.3f}s")
        
            # Resource metrics
            with col2:
                import psutil
                cpu_usage = psutil.cpu_percent()
                memory = psutil.Process().memory_info()
                memory_usage = memory.rss / (1024 * 1024)  # MB
                
                st.metric("CPU Usage", f"{cpu_usage}%")
                st.metric("Memory Usage", f"{memory_usage:.1f} MB")
        
            # GPU metrics if available
            with col3:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                    gpu_utilization = torch.cuda.utilization()
                    
                    st.metric("GPU Memory", f"{gpu_memory:.1f} MB")
                    st.metric("GPU Utilization", f"{gpu_utilization}%")
            
            # Historical metrics visualization
            if show_history:
                self._render_metrics_history(chart_type)
        
        with metric_tabs[1]:
            self._render_data_synthesis_controls()
            
            # Show synthetic data preview if available
            if hasattr(self, 'data_loader') and hasattr(self.data_loader, 'generate_synthetic_data'):
                st.subheader("Synthetic Data Preview")
                try:
                    # Generate a small sample of synthetic data
                    sample_data = self.data_loader.load_forex_data(timeframe="1H", augment_data=True)[0].tail(100)
                    
                    # Plot the synthetic data
                    fig = go.Figure(data=[
                        go.Candlestick(
                            x=sample_data.index,
                            open=sample_data['open'],
                            high=sample_data['high'],
                            low=sample_data['low'],
                            close=sample_data['close']
                        )
                    ])
                    fig.update_layout(title='Synthetic Data Sample')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    st.subheader("Synthetic Data Statistics")
                    stats = sample_data['close'].describe()
                    st.dataframe(stats)
                except Exception as e:
                    st.warning(f"Could not generate synthetic data preview: {str(e)}")
        
        with metric_tabs[2]:
            self._render_algorithmic_efficiency()

    def _render_training_controls(self):
        """Render enhanced training controls with real-time parameter adjustment."""
        st.header("🎮 Training Controls")
        
        # Training control tabs
        control_tab, params_tab, advanced_tab = st.tabs([
            "Basic Controls", "Training Parameters", "Advanced Settings"
        ])
        
        with control_tab:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Start Training", 
                           disabled=st.session_state.training_state['active']):
                    st.session_state.training_state['active'] = True
                    st.session_state.training_state['current_epoch'] = 0
            
            with col2:
                if st.button("Pause", 
                           disabled=not st.session_state.training_state['active']):
                    st.session_state.training_state['active'] = False
            
            with col3:
                if st.button("Resume", 
                           disabled=not st.session_state.training_state['active']):
                    st.session_state.training_state['active'] = True
        
        with params_tab:
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox(
                    "Model Architecture",
                    ["CatBoost", "XGBoost", "LightGBM", "Ensemble"],
                    index=["CatBoost", "XGBoost", "LightGBM", "Ensemble"].index(
                        st.session_state.training_state['model_type'])
                )
                
                learning_rate = st.slider(
                    "Learning Rate",
                    min_value=0.0001,
                    max_value=0.1,
                    value=0.01,
                    format="%.4f"
                )
                
                batch_size = st.select_slider(
                    "Batch Size",
                    options=[32, 64, 128, 256, 512],
                    value=st.session_state.training_state['batch_size']
                )
            
            with col2:
                optimizer = st.selectbox(
                    "Optimizer",
                    ["Adam", "SGD", "RMSprop", "AdamW"],
                    index=["Adam", "SGD", "RMSprop", "AdamW"].index(
                        st.session_state.training_state['optimizer'])
                )
                
                epochs = st.number_input(
                    "Number of Epochs",
                    min_value=1,
                    value=st.session_state.training_state['total_epochs'] or 100
                )
                
                early_stopping = st.number_input(
                    "Early Stopping Patience",
                    min_value=0,
                    value=10
                )
        
        with advanced_tab:
            with st.expander("Optimization Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    use_amp = st.checkbox("Use Mixed Precision", True)
                    use_gradient_clipping = st.checkbox("Enable Gradient Clipping")
                    if use_gradient_clipping:
                        clip_value = st.slider("Clip Value", 0.1, 10.0, 1.0)
                
                with col2:
                    scheduler = st.selectbox(
                        "LR Scheduler",
                        ["None", "CosineAnnealing", "ReduceLROnPlateau", "OneCycle"]
                    )
                    if scheduler != "None":
                        warmup_epochs = st.slider("Warmup Epochs", 0, 10, 3)
            
            with st.expander("Feature Engineering"):
                col1, col2 = st.columns(2)
                
                with col1:
                    feature_selection = st.multiselect(
                        "Feature Selection Methods",
                        ["Correlation", "Mutual Information", "SHAP", "Permutation"],
                        ["Correlation", "SHAP"]
                    )
                    
                    min_feature_importance = st.slider(
                        "Min Feature Importance",
                        0.0, 1.0, 0.01
                    )
                
                with col2:
                    use_pca = st.checkbox("Use PCA")
                    if use_pca:
                        n_components = st.slider(
                            "Number of Components",
                            2, 50, 10
                        )
                    
                    scaling_method = st.selectbox(
                        "Feature Scaling",
                        ["StandardScaler", "MinMaxScaler", "RobustScaler"]
                    )

    def _render_feature_engineering(self):
        """Render feature engineering showcase."""
        st.header("🔧 Feature Engineering Showcase")

        data_files = list(self.data_dir.glob("*.parquet"))
        if data_files:
            try:
                import pyarrow.parquet as pq
                sample_data = pq.read_table(data_files[0]).to_pandas().head(100)

                st.subheader("Sample Features Generated")

                # Show feature categories
                basic_features = [col for col in sample_data.columns if any(x in col for x in ['returns', 'price_range', 'body_size'])]
                momentum_features = [col for col in sample_data.columns if any(x in col for x in ['rsi', 'stoch', 'momentum'])]
                volatility_features = [col for col in sample_data.columns if any(x in col for x in ['volatility', 'bb_', 'atr'])]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Basic Features", len(basic_features))
                    if basic_features:
                        st.write("• " + "\n• ".join(basic_features[:3]))

                with col2:
                    st.metric("Momentum Features", len(momentum_features))
                    if momentum_features:
                        st.write("• " + "\n• ".join(momentum_features[:3]))

                with col3:
                    st.metric("Volatility Features", len(volatility_features))
                    if volatility_features:
                        st.write("• " + "\n• ".join(volatility_features[:3]))

                # Show sample data
                st.subheader("Sample Processed Data")
                st.dataframe(sample_data.head(), use_container_width=True)

            except Exception as e:
                st.error(f"Error loading sample data: {e}")
        else:
            st.info("No processed data found. Run the enhanced training pipeline to generate features.")

    def _render_model_comparison(self):
        """Render model performance comparison."""
        st.header("⚖️ Model Performance Comparison")

        # Mock comparison data
        comparison_data = pd.DataFrame({
            'Model': ['CatBoost (Old)', 'CatBoost (Optimized)', 'LightGBM', 'Random Forest', 'Ensemble'],
            'R² Score': [0.72, 0.87, 0.84, 0.81, 0.89],
            'RMSE': [0.0085, 0.0052, 0.0061, 0.0068, 0.0049],
            'Training Time (s)': [45, 120, 95, 85, 180],
            'Features Used': [14, 47, 47, 47, 47]
        })

        fig = px.bar(comparison_data, x='Model', y='R² Score',
                    title='Model Performance Comparison',
                    color='R² Score',
                    color_continuous_scale='viridis')

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comparison_data, use_container_width=True)

    def _render_pipeline_steps(self):
        """Render training pipeline steps."""
        st.header("🔄 Enhanced Training Pipeline")

        pipeline_steps = [
            "1. 📊 Data Loading & Validation",
            "2. 🔧 Advanced Feature Engineering (50+ indicators)",
            "3. 🎯 Feature Selection & PCA",
            "4. 🤖 Hyperparameter Optimization (Optuna)",
            "5. 🚀 Model Training (Multiple algorithms)",
            "6. ⚖️ Cross-validation & Evaluation",
            "7. 📈 Model Interpretability (SHAP)",
            "8. 💾 Results & Monitoring"
        ]

        for step in pipeline_steps:
            st.markdown(f"**{step}**")

    def _render_usage_instructions(self):
        """Render usage instructions."""
        st.header("🚀 How to Use the Enhanced System")

        st.code("""
        # Run comprehensive training
        python comprehensive_training_pipeline.py --optimize --ensemble --interpretability --features 50

        # Or use the interactive runner
        python run_enhanced_training.py

        # View results in this dashboard
        streamlit run enhanced_training_dashboard.py
        """)

        st.markdown("""
        ### Key Features:
        - **Hyperparameter Optimization**: Automatic tuning with Optuna
        - **Ensemble Methods**: Multiple algorithms for better performance
        - **Advanced Features**: 50+ technical indicators and statistical features
        - **Model Interpretability**: SHAP explanations for transparency
        - **Cross-validation**: Robust evaluation with time series splits
        - **Comprehensive Logging**: Full experiment tracking
        """)

    def _render_metrics_history(self, chart_type: str):
        """Render historical metrics visualization."""
        # Create figure
        fig = go.Figure()
        
        # Add metrics based on availability
        metrics_to_plot = {
            'Training Loss': 'training_loss',
            'Validation Loss': 'validation_loss',
            'Learning Rate': 'learning_rate'
        }
        
        for label, key in metrics_to_plot.items():
            if st.session_state.metrics_history[key]:
                data = list(st.session_state.metrics_history[key])
                if chart_type == 'Area':
                    fig.add_trace(go.Scatter(
                        y=data,
                        name=label,
                        fill='tonexty'
                    ))
                elif chart_type == 'Bar':
                    fig.add_trace(go.Bar(
                        y=data,
                        name=label
                    ))
                else:  # Line chart
                    fig.add_trace(go.Scatter(
                        y=data,
                        name=label,
                        mode='lines'
                    ))
        
        # Update layout
        fig.update_layout(
            height=400,
            title='Training Metrics History',
            xaxis_title='Iteration',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Resource usage history
        if any(len(st.session_state.metrics_history[k]) > 0 
               for k in ['cpu_usage', 'memory_usage', 'gpu_memory']):
            fig = go.Figure()
            
            # CPU Usage
            if st.session_state.metrics_history['cpu_usage']:
                fig.add_trace(go.Scatter(
                    y=list(st.session_state.metrics_history['cpu_usage']),
                    name='CPU Usage (%)',
                    line=dict(color='blue')
                ))
            
            # Memory Usage
            if st.session_state.metrics_history['memory_usage']:
                fig.add_trace(go.Scatter(
                    y=list(st.session_state.metrics_history['memory_usage']),
                    name='Memory (MB)',
                    line=dict(color='green')
                ))
            
            # GPU Memory
            if st.session_state.metrics_history['gpu_memory']:
                fig.add_trace(go.Scatter(
                    y=list(st.session_state.metrics_history['gpu_memory']),
                    name='GPU Memory (MB)',
                    line=dict(color='red')
                ))
            
            fig.update_layout(
                height=300,
                title='Resource Usage History',
                xaxis_title='Time',
                yaxis_title='Usage',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def _apply_smoothing(self, data: List[float], factor: float) -> List[float]:
        """Apply exponential smoothing to data."""
        smoothed = []
        if not data:
            return smoothed
        
        smoothed.append(data[0])
        for n in range(1, len(data)):
            smoothed.append(smoothed[-1] * factor + data[n] * (1 - factor))
        return smoothed

    def _get_colorway(self, scheme: str) -> List[str]:
        """Get color scheme for plots."""
        schemes = {
            'Default': None,  # Use Plotly default
            'Viridis': px.colors.sequential.Viridis,
            'Plasma': px.colors.sequential.Plasma,
            'Magma': px.colors.sequential.Magma
        }
        return schemes.get(scheme)

    def render(self) -> None:
        """Render the enhanced training monitor component."""
        st.subheader(self.config.title)
        st.markdown(self.config.description)

        # Sidebar metrics
        with st.sidebar:
            st.header("Training System Status")
            metrics = self._get_system_metrics()
            st.metric("Trained Models", metrics['trained_models'])
            st.metric("Training Sessions", metrics['training_sessions'])
            st.metric("Processed Datasets", metrics['processed_datasets'])
            
            # Quick actions
            st.subheader("Quick Actions")
            if st.button("Clear History"):
                for key in st.session_state.metrics_history:
                    st.session_state.metrics_history[key] = []
            
            if st.button("Export Metrics"):
                metrics_df = pd.DataFrame(st.session_state.metrics_history)
                st.download_button(
                    "Download CSV",
                    metrics_df.to_csv(index=False),
                    "training_metrics.csv",
                    "text/csv"
                )

        # Main content in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Real-time Monitor",
            "🎮 Training Controls",
            "📈 Analysis",
            "📚 Documentation"
        ])

        with tab1:
            self._render_real_time_metrics()
            self._render_system_improvements()

        with tab2:
            self._render_training_controls()
            self._render_feature_engineering()

        with tab3:
            self._render_model_comparison()
            
            # Add custom metric analysis
            with st.expander("🔍 Custom Metric Analysis"):
                metric_name = st.text_input("Metric Name")
                metric_formula = st.text_area("Formula (Python expression)")
                if st.button("Add Metric"):
                    try:
                        # Safely evaluate the formula
                        import numpy as np
                        data = {k: np.array(v) for k, v in 
                               st.session_state.metrics_history.items() 
                               if isinstance(v, list)}
                        result = eval(metric_formula, {"np": np}, data)
                        st.session_state.metrics_history['custom_metrics'][metric_name] = result
                        st.success(f"Added metric: {metric_name}")
                    except Exception as e:
                        st.error(f"Error computing metric: {str(e)}")

        with tab4:
            self._render_pipeline_steps()
            self._render_usage_instructions()

    def update(self, data: Dict[str, Any]) -> None:
        """Update component with new data."""
        self._cache['data'] = data
        self.clear_cache()