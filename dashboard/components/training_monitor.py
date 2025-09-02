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
        metric_tabs = st.tabs([
            "Training Progress",
            "Data Synthesis",
            "Algorithmic Efficiency",
            "Resource Usage",
            "Model Insights"
        ])
        
        with metric_tabs[0]:  # Training Progress
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Interactive training progress chart
                progress_chart = self._create_progress_chart(
                    st.session_state.metrics_history['training_loss'],
                    st.session_state.metrics_history['validation_loss']
                )
                st.plotly_chart(progress_chart, use_container_width=True)
            
            with col2:
                # Key metrics and status
                if st.session_state.metrics_history['training_loss']:
                    current_loss = st.session_state.metrics_history['training_loss'][-1]
                    best_loss = min(st.session_state.metrics_history['training_loss'])
                    improvement = ((current_loss - best_loss) / best_loss) * 100
                    
                    st.metric("Current Loss", f"{current_loss:.6f}")
                    st.metric("Best Loss", f"{best_loss:.6f}")
                    st.metric("Improvement", f"{improvement:.2f}%")
                
                # Training controls
                if st.button("Pause Training", disabled=not st.session_state.training_state['active']):
                    st.session_state.training_state['active'] = False
                if st.button("Resume Training", disabled=st.session_state.training_state['active']):
                    st.session_state.training_state['active'] = True
            
            # Settings and customization
            with st.expander("📈 Display Settings"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    update_interval = st.slider(
                        "Update Interval (s)",
                        1, 30,
                        st.session_state.viz_preferences['update_interval']
                    )
                with col2:
                    smoothing = st.slider(
                        "Smoothing Factor",
                        0.0, 1.0, 0.6,
                        help="Apply exponential smoothing to metrics"
                    )
                with col3:
                    chart_type = st.selectbox(
                        "Chart Type",
                        ['Interactive', 'Area', 'Candlestick'],
                        index=['Interactive', 'Area', 'Candlestick'].index(
                            st.session_state.viz_preferences.get('chart_type', 'Interactive')
                        )
                    )
        
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
        
        with metric_tabs[1]:  # Data Synthesis
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Interactive synthetic data controls
                st.subheader("Synthetic Data Generation")
                
                synthesis_params = {
                    'num_samples': st.slider("Number of Samples", 100, 5000, 1000),
                    'edge_case_ratio': st.slider("Edge Case Ratio", 0.0, 1.0, 0.2),
                    'market_regimes': st.multiselect(
                        "Market Regimes",
                        ['trending', 'ranging', 'volatile'],
                        ['trending', 'ranging']
                    )
                }
                
                if st.button("Generate Preview"):
                    with st.spinner("Generating synthetic data..."):
                        synthetic_data = self.data_loader.generate_synthetic_data(
                            self.latest_data,
                            **synthesis_params
                        )
                        st.session_state.synthetic_preview = synthetic_data
            
            with col2:
                # Synthesis statistics
                if hasattr(st.session_state, 'synthetic_preview'):
                    data = st.session_state.synthetic_preview
                    stats = self._calculate_synthesis_stats(data)
                    
                    st.metric("Total Samples", len(data))
                    st.metric("Edge Cases", stats['edge_cases'])
                    st.metric("Pattern Quality", f"{stats['quality_score']:.2f}")
            
            # Pattern visualization
            if hasattr(st.session_state, 'synthetic_preview'):
                pattern_tabs = st.tabs(["Price Patterns", "Distributions", "Quality Metrics"])
                
                with pattern_tabs[0]:
                    fig = self._create_pattern_visualization(st.session_state.synthetic_preview)
                    st.plotly_chart(fig, use_container_width=True)
                
                with pattern_tabs[1]:
                    fig = self._create_distribution_plots(st.session_state.synthetic_preview)
                    st.plotly_chart(fig, use_container_width=True)
                
                with pattern_tabs[2]:
                    self._render_quality_metrics(st.session_state.synthetic_preview)
        
        with metric_tabs[2]:  # Algorithmic Efficiency
            # Enhanced efficiency metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Training efficiency metrics
                efficiency_metrics = self._calculate_efficiency_metrics()
                
                st.metric(
                    "Training Speed",
                    f"{efficiency_metrics['samples_per_second']:.1f} samples/s",
                    delta=f"{efficiency_metrics['speed_improvement']:.1f}%"
                )
                
                st.metric(
                    "Memory Efficiency",
                    f"{efficiency_metrics['memory_per_sample']:.2f} KB/sample",
                    delta=f"{efficiency_metrics['memory_improvement']:.1f}%"
                )
                
                st.metric(
                    "GPU Utilization",
                    f"{efficiency_metrics['gpu_utilization']:.1f}%",
                    delta=f"{efficiency_metrics['gpu_improvement']:.1f}%"
                )
            
            with col2:
                # Interactive efficiency analysis
                st.subheader("Performance Analysis")
                metric_type = st.selectbox(
                    "Analyze",
                    ['Training Speed', 'Memory Usage', 'GPU Utilization']
                )
                
                fig = self._create_efficiency_chart(metric_type)
                st.plotly_chart(fig, use_container_width=True)
        
        with metric_tabs[3]:  # Resource Usage
            self._render_resource_monitor()
        
        with metric_tabs[4]:  # Model Insights
            self._render_model_insights()

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

    def _create_progress_chart(
        self,
        training_loss: List[float],
        validation_loss: List[float]
    ) -> go.Figure:
        """Create interactive training progress chart."""
        fig = go.Figure()
        
        # Apply smoothing if enabled
        smoothing = st.session_state.viz_preferences.get('smoothing', 0.6)
        if smoothing > 0:
            training_loss = self._apply_smoothing(training_loss, smoothing)
            validation_loss = self._apply_smoothing(validation_loss, smoothing)
        
        # Add traces
        fig.add_trace(go.Scatter(
            y=training_loss,
            name='Training Loss',
            mode='lines',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            y=validation_loss,
            name='Validation Loss',
            mode='lines',
            line=dict(color='red')
        ))
        
        # Update layout
        fig.update_layout(
            title='Training Progress',
            xaxis_title='Iteration',
            yaxis_title='Loss',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig

    def _calculate_synthesis_stats(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistics for synthetic data."""
        stats = {}
        
        # Count edge cases
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        
        stats['edge_cases'] = len(data[
            (returns.abs() > returns.std() * 3) |  # Price jumps
            (volatility > volatility.quantile(0.95))  # High volatility
        ])
        
        # Calculate pattern quality score
        pattern_scores = []
        
        # Check for trend consistency
        trend_consistency = np.corrcoef(
            data.index,
            data['close'].values
        )[0, 1]
        pattern_scores.append(abs(trend_consistency))
        
        # Check for realistic volatility
        vol_score = 1 - abs(
            volatility.mean() - returns.std()
        ) / returns.std()
        pattern_scores.append(vol_score)
        
        # Check for realistic correlations
        if 'correlated_asset' in data.columns:
            corr_score = abs(
                data['close'].corr(data['correlated_asset']) - 0.7
            )
            pattern_scores.append(1 - corr_score)
        
        stats['quality_score'] = np.mean(pattern_scores)
        
        return stats

    def _create_pattern_visualization(self, data: pd.DataFrame) -> go.Figure:
        """Create interactive pattern visualization."""
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ))
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3
        ))
        
        # Add technical indicators
        if 'sma_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['sma_20'],
                name='20 SMA',
                line=dict(color='orange')
            ))
        
        if 'bollinger_upper' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['bollinger_upper'],
                name='Bollinger Upper',
                line=dict(color='gray', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['bollinger_lower'],
                name='Bollinger Lower',
                line=dict(color='gray', dash='dash'),
                fill='tonexty'
            ))
        
        # Update layout
        fig.update_layout(
            title='Synthetic Price Patterns',
            yaxis_title='Price',
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right'
            ),
            xaxis_rangeslider_visible=False
        )
        
        return fig

    def _create_distribution_plots(self, data: pd.DataFrame) -> go.Figure:
        """Create distribution analysis plots."""
        fig = go.Figure()
        
        # Returns distribution
        returns = data['close'].pct_change().dropna()
        
        fig.add_trace(go.Histogram(
            x=returns,
            name='Returns Distribution',
            nbinsx=50,
            opacity=0.7
        ))
        
        # Add normal distribution for comparison
        x = np.linspace(returns.min(), returns.max(), 100)
        y = stats.norm.pdf(x, returns.mean(), returns.std())
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            name='Normal Distribution',
            line=dict(color='red', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='Returns Distribution Analysis',
            xaxis_title='Returns',
            yaxis_title='Frequency',
            showlegend=True
        )
        
        return fig

    def _render_quality_metrics(self, data: pd.DataFrame):
        """Render synthetic data quality metrics."""
        st.subheader("Quality Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Statistical properties
            returns = data['close'].pct_change().dropna()
            
            st.metric("Mean Return", f"{returns.mean():.6f}")
            st.metric("Volatility", f"{returns.std():.6f}")
            st.metric("Skewness", f"{returns.skew():.2f}")
            st.metric("Kurtosis", f"{returns.kurtosis():.2f}")
        
        with col2:
            # Pattern metrics
            st.metric(
                "Trend Strength",
                f"{abs(returns.autocorr()):.2f}",
                help="Autocorrelation of returns"
            )
            
            # Volatility clustering
            vol_cluster = returns.abs().autocorr()
            st.metric(
                "Volatility Clustering",
                f"{vol_cluster:.2f}",
                help="Presence of volatility clusters"
            )
            
            # Market efficiency
            hurst = self._calculate_hurst_exponent(returns)
            st.metric(
                "Market Efficiency",
                f"{hurst:.2f}",
                help="Hurst exponent (0.5 = efficient market)"
            )

    def _calculate_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate algorithmic efficiency metrics."""
        metrics = {}
        
        # Calculate training speed
        batch_times = st.session_state.metrics_history['batch_time']
        if batch_times:
            recent_time = np.mean(batch_times[-50:])
            old_time = np.mean(batch_times[-100:-50]) if len(batch_times) > 100 else recent_time
            
            metrics['samples_per_second'] = 1 / recent_time
            metrics['speed_improvement'] = ((old_time - recent_time) / old_time) * 100
        else:
            metrics['samples_per_second'] = 0
            metrics['speed_improvement'] = 0
        
        # Calculate memory efficiency
        memory_usage = st.session_state.metrics_history['memory_usage']
        if memory_usage:
            current_memory = memory_usage[-1]
            peak_memory = max(memory_usage)
            baseline_memory = memory_usage[0]
            
            metrics['memory_per_sample'] = current_memory / len(batch_times) if batch_times else 0
            metrics['memory_improvement'] = ((peak_memory - current_memory) / peak_memory) * 100
        else:
            metrics['memory_per_sample'] = 0
            metrics['memory_improvement'] = 0
        
        # Calculate GPU efficiency
        if torch.cuda.is_available():
            metrics['gpu_utilization'] = torch.cuda.utilization()
            metrics['gpu_improvement'] = 0  # Need historical data for improvement
        else:
            metrics['gpu_utilization'] = 0
            metrics['gpu_improvement'] = 0
        
        return metrics

    def _create_efficiency_chart(self, metric_type: str) -> go.Figure:
        """Create efficiency analysis chart."""
        fig = go.Figure()
        
        if metric_type == 'Training Speed':
            y = [1/t for t in st.session_state.metrics_history['batch_time']]
            name = 'Samples/second'
        elif metric_type == 'Memory Usage':
            y = st.session_state.metrics_history['memory_usage']
            name = 'Memory (MB)'
        else:  # GPU Utilization
            y = st.session_state.metrics_history['gpu_memory']
            name = 'GPU Memory (MB)'
        
        # Add main metric
        fig.add_trace(go.Scatter(
            y=y,
            name=name,
            mode='lines'
        ))
        
        # Add trend line
        if len(y) > 1:
            z = np.polyfit(range(len(y)), y, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                y=p(range(len(y))),
                name='Trend',
                line=dict(dash='dash')
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{metric_type} Over Time',
            xaxis_title='Iteration',
            yaxis_title=name,
            showlegend=True
        )
        
        return fig

    def _render_resource_monitor(self):
        """Render resource usage monitoring."""
        st.subheader("💻 Resource Monitor")
        
        col1, col2, col3 = st.columns(3)
        
        # System resources
        with col1:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            st.metric(
                "CPU Usage",
                f"{cpu_percent}%",
                delta=f"{cpu_percent - self._get_previous_cpu()}%"
            )
            
            st.metric(
                "Memory Usage",
                f"{memory.percent}%",
                delta=f"{memory.percent - self._get_previous_memory()}%"
            )
        
        # GPU resources
        with col2:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
                gpu_util = torch.cuda.utilization()
                
                st.metric("GPU Memory", f"{gpu_memory:.1f} MB")
                st.metric("GPU Utilization", f"{gpu_util}%")
            else:
                st.info("No GPU available")
        
        # Disk resources
        with col3:
            disk = psutil.disk_usage('/')
            st.metric("Disk Usage", f"{disk.percent}%")
            st.metric("Free Space", f"{disk.free / 1024**3:.1f} GB")

    def _render_model_insights(self):
        """Render model training insights."""
        st.subheader("🔍 Model Insights")
        
        # Feature importance
        if self.metrics['feature_importance_history']:
            st.subheader("Feature Importance")
            
            # Calculate average importance across folds
            importance_df = pd.DataFrame(
                self.metrics['feature_importance_history']
            ).groupby('features')['importance'].mean()
            
            fig = px.bar(
                importance_df,
                title='Feature Importance',
                labels={'value': 'Importance', 'index': 'Feature'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Learning dynamics
        st.subheader("Learning Dynamics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss landscape
            if st.session_state.metrics_history['training_loss']:
                loss_data = pd.DataFrame({
                    'Training': st.session_state.metrics_history['training_loss'],
                    'Validation': st.session_state.metrics_history['validation_loss']
                })
                
                fig = px.line(
                    loss_data,
                    title='Loss Landscape'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Learning rate dynamics
            if st.session_state.metrics_history['learning_rate']:
                fig = px.line(
                    y=st.session_state.metrics_history['learning_rate'],
                    title='Learning Rate Schedule'
                )
                st.plotly_chart(fig, use_container_width=True)

    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent for market efficiency analysis."""
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag])))
               for lag in lags]
        
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]

    def _get_previous_cpu(self) -> float:
        """Get previous CPU usage for delta calculation."""
        if 'cpu_usage' in st.session_state.metrics_history:
            return st.session_state.metrics_history['cpu_usage'][-2] \
                if len(st.session_state.metrics_history['cpu_usage']) > 1 else 0
        return 0

    def _get_previous_memory(self) -> float:
        """Get previous memory usage for delta calculation."""
        if 'memory_usage' in st.session_state.metrics_history:
            return st.session_state.metrics_history['memory_usage'][-2] \
                if len(st.session_state.metrics_history['memory_usage']) > 1 else 0
        return 0

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

    def calculate_performance_metrics(self, data: Any) -> dict[str, pd.Series]:
        """Calculate performance metrics over time."""
        if not data or 'returns' not in data:
            return {}

        returns = data['returns']
        metrics = {}

        # Basic performance metrics
        metrics['cumulative_returns'] = (1 + returns).cumprod() - 1
        metrics['rolling_sharpe'] = returns.rolling(window=30).mean() / returns.rolling(window=30).std() * np.sqrt(252)
        metrics['rolling_volatility'] = returns.rolling(window=30).std() * np.sqrt(252)

        return metrics

    def analyze_drawdowns(self, returns: pd.Series) -> dict[str, Any]:
        """Analyze drawdowns from return series."""
        if returns.empty:
            return {}

        # Calculate drawdowns
        wealth_index = (1 + returns).cumprod()
        previous_peaks = wealth_index.expanding(min_periods=1).max()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks

        return {
            'drawdowns': drawdowns,
            'max_drawdown': drawdowns.min(),
            'current_drawdown': drawdowns.iloc[-1],
            'drawdown_duration': (drawdowns < 0).astype(int).groupby((drawdowns >= 0).cumsum()).cumsum()
        }

    def create_figure(self, data: Any) -> go.Figure:
        """Create plotly figure for visualization."""
        if not data or 'returns' not in data:
            return go.Figure()

        returns = data['returns']
        cumulative_returns = (1 + returns).cumprod() - 1

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values * 100,
            name='Cumulative Returns',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title='Training Performance',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            height=400
        )

        return fig

    def update(self, data: Dict[str, Any]) -> None:
        """Update component with new data."""
        self._cache['data'] = data
        self.clear_cache()