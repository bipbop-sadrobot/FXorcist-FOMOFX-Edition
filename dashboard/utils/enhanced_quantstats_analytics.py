"""
Enhanced QuantStats Analytics Module
Provides comprehensive portfolio analytics with advanced features including:
- HTML report generation
- Decomposition analysis
- Causal inference
- Advanced visualizations
- Export functionality
"""

import pandas as pd
import numpy as np
import quantstats as qs
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta
import tempfile
import os
from scipy import stats
from sklearn.decomposition import PCA
from econml.dml import CausalForestDML
import shap

logger = logging.getLogger(__name__)

class EnhancedQuantStatsAnalytics:
    """Enhanced analytics class with advanced features."""

    def __init__(self):
        """Initialize enhanced quantstats analytics."""
        super().__init__()
        qs.extend_pandas()
        
        # Default settings
        self.risk_free_rate = 0.02
        self.compounding = True
        self.periods_per_year = 252

    def generate_html_report(self, returns: pd.Series, 
                           benchmark_returns: Optional[pd.Series] = None,
                           output_dir: Optional[str] = None) -> str:
        """
        Generate comprehensive HTML report using quantstats.
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns
            output_dir: Optional output directory for the report
            
        Returns:
            Path to generated HTML report
        """
        try:
            # Create temporary directory if no output_dir specified
            if output_dir is None:
                output_dir = tempfile.mkdtemp()
            
            output_file = os.path.join(output_dir, 'quantstats_report.html')
            
            # Generate report
            qs.reports.html(returns, 
                          benchmark=benchmark_returns,
                          output=output_file,
                          title='Portfolio Analysis Report',
                          rf=self.risk_free_rate)
            
            return output_file
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return ""

    def perform_returns_decomposition(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Perform returns decomposition analysis.
        
        Args:
            returns: Portfolio returns
            
        Returns:
            Dictionary containing decomposition results
        """
        try:
            # Convert returns to numpy array for PCA
            returns_array = returns.values.reshape(-1, 1)
            
            # Perform PCA
            pca = PCA(n_components=min(3, len(returns_array)))
            pca_result = pca.fit_transform(returns_array)
            
            # Calculate component contributions
            components = pd.DataFrame(
                pca_result,
                columns=[f'Component_{i+1}' for i in range(pca.n_components_)],
                index=returns.index
            )
            
            # Calculate explained variance
            explained_variance = pd.Series(
                pca.explained_variance_ratio_,
                index=[f'Component_{i+1}' for i in range(pca.n_components_)]
            )
            
            return {
                'components': components,
                'explained_variance': explained_variance,
                'cumulative_variance': explained_variance.cumsum()
            }
        except Exception as e:
            logger.error(f"Error in returns decomposition: {e}")
            return {}

    def perform_causal_analysis(self, returns: pd.Series, 
                              features: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform causal inference analysis using EconML.
        
        Args:
            returns: Portfolio returns
            features: Feature matrix for analysis
            
        Returns:
            Dictionary containing causal analysis results
        """
        try:
            # Prepare data
            Y = returns.values
            T = features.iloc[:, 0].values  # Treatment variable
            X = features.iloc[:, 1:].values  # Covariates
            
            # Initialize and fit causal forest
            est = CausalForestDML(
                n_estimators=100,
                min_samples_leaf=10,
                max_depth=5
            )
            est.fit(Y, T, X=X)
            
            # Calculate treatment effects
            te_pred = est.effect(X)
            
            # Generate SHAP values for interpretation
            explainer = shap.TreeExplainer(est)
            shap_values = explainer.shap_values(X)
            
            return {
                'treatment_effects': pd.Series(te_pred, index=returns.index),
                'shap_values': shap_values,
                'feature_importance': pd.Series(
                    np.abs(shap_values).mean(0),
                    index=features.columns[1:]
                )
            }
        except Exception as e:
            logger.error(f"Error in causal analysis: {e}")
            return {}

    def create_advanced_visualizations(self, 
                                    returns: pd.Series,
                                    metrics: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Create advanced visualization suite.
        
        Args:
            returns: Portfolio returns
            metrics: Dictionary of calculated metrics
            
        Returns:
            Dictionary of Plotly figures
        """
        figures = {}
        
        try:
            # Create rolling correlation heatmap
            if 'rolling_correlations' in metrics:
                corr_data = metrics['rolling_correlations']
                figures['correlation_heatmap'] = px.imshow(
                    corr_data,
                    title='Rolling Correlations Heatmap',
                    color_continuous_scale='RdBu'
                )
            
            # Create underwater plot (drawdown visualization)
            if 'drawdowns' in metrics:
                dd_data = metrics['drawdowns']
                figures['underwater_plot'] = go.Figure()
                figures['underwater_plot'].add_trace(
                    go.Scatter(
                        x=dd_data.index,
                        y=dd_data.values * 100,
                        fill='tozeroy',
                        name='Drawdown',
                        fillcolor='rgba(255,0,0,0.3)',
                        line=dict(color='red')
                    )
                )
                figures['underwater_plot'].update_layout(
                    title='Underwater Plot (Drawdowns)',
                    yaxis_title='Drawdown (%)',
                    xaxis_title='Date'
                )
            
            # Create rolling beta plot
            if 'rolling_beta' in metrics:
                beta_data = metrics['rolling_beta']
                figures['rolling_beta'] = go.Figure()
                figures['rolling_beta'].add_trace(
                    go.Scatter(
                        x=beta_data.index,
                        y=beta_data.values,
                        name='Rolling Beta',
                        line=dict(color='blue')
                    )
                )
                figures['rolling_beta'].update_layout(
                    title='Rolling Beta',
                    yaxis_title='Beta',
                    xaxis_title='Date'
                )
            
            # Create return decomposition visualization
            decomp_result = self.perform_returns_decomposition(returns)
            if decomp_result and 'components' in decomp_result:
                components = decomp_result['components']
                figures['return_decomposition'] = go.Figure()
                
                for col in components.columns:
                    figures['return_decomposition'].add_trace(
                        go.Scatter(
                            x=components.index,
                            y=components[col],
                            name=col,
                            stackgroup='one'
                        )
                    )
                
                figures['return_decomposition'].update_layout(
                    title='Return Decomposition Analysis',
                    yaxis_title='Component Contribution',
                    xaxis_title='Date'
                )
            
        except Exception as e:
            logger.error(f"Error creating advanced visualizations: {e}")
        
        return figures

    def export_analysis_results(self, 
                              returns: pd.Series,
                              metrics: Dict[str, Any],
                              output_dir: Optional[str] = None) -> str:
        """
        Export comprehensive analysis results.
        
        Args:
            returns: Portfolio returns
            metrics: Dictionary of calculated metrics
            output_dir: Optional output directory
            
        Returns:
            Path to exported results
        """
        try:
            if output_dir is None:
                output_dir = tempfile.mkdtemp()
            
            # Create results dictionary
            results = {
                'basic_metrics': {
                    k: v for k, v in metrics.items() 
                    if isinstance(v, (int, float, str))
                },
                'returns_summary': returns.describe().to_dict(),
                'rolling_metrics': {
                    k: v.to_dict() for k, v in metrics.items()
                    if isinstance(v, pd.Series) and k.startswith('rolling_')
                },
                'decomposition': self.perform_returns_decomposition(returns)
            }
            
            # Export to JSON
            import json
            output_file = os.path.join(output_dir, 'analysis_results.json')
            with open(output_file, 'w') as f:
                json.dump(results, f, default=str, indent=2)
            
            return output_file
        except Exception as e:
            logger.error(f"Error exporting analysis results: {e}")
            return ""

    def generate_enhanced_tearsheet(self, 
                                  returns: pd.Series,
                                  benchmark_returns: Optional[pd.Series] = None,
                                  features: Optional[pd.DataFrame] = None,
                                  title: str = "Enhanced Portfolio Analysis") -> Dict[str, Any]:
        """
        Generate enhanced tearsheet with all advanced features.
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns
            features: Optional feature matrix for causal analysis
            title: Report title
            
        Returns:
            Dictionary containing all tearsheet components
        """
        try:
            # Initialize tearsheet dictionary
            tearsheet = {
                'title': title,
                'generated_at': datetime.now(),
                'analysis_period': {
                    'start': returns.index.min(),
                    'end': returns.index.max(),
                    'total_days': len(returns)
                }
            }
            
            # Calculate base metrics
            metrics = self.calculate_comprehensive_metrics(returns, benchmark_returns)
            
            # Add decomposition analysis
            decomp_results = self.perform_returns_decomposition(returns)
            metrics.update({'decomposition': decomp_results})
            
            # Add causal analysis if features provided
            if features is not None:
                causal_results = self.perform_causal_analysis(returns, features)
                metrics.update({'causal_analysis': causal_results})
            
            # Create advanced visualizations
            advanced_charts = self.create_advanced_visualizations(returns, metrics)
            
            # Generate HTML report
            html_report = self.generate_html_report(returns, benchmark_returns)
            
            # Export detailed results
            exported_results = self.export_analysis_results(returns, metrics)
            
            # Update tearsheet with all components
            tearsheet.update({
                'metrics': metrics,
                'charts': advanced_charts,
                'html_report': html_report,
                'exported_results': exported_results
            })
            
            return tearsheet
            
        except Exception as e:
            logger.error(f"Error generating enhanced tearsheet: {e}")
            return {'error': str(e)}

    def calculate_comprehensive_metrics(self, 
                                     returns: pd.Series,
                                     benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics including advanced indicators.
        Inherits from base class and adds additional metrics.
        """
        # Get base metrics
        metrics = super().calculate_comprehensive_metrics(returns, benchmark_returns)
        
        try:
            # Add advanced metrics
            metrics.update({
                # Advanced risk metrics
                'omega_ratio': qs.stats.omega(returns),
                'sortino_ratio': qs.stats.sortino(returns),
                'kappa_three': qs.stats.kappa(returns, threshold=0.0),
                'gain_to_pain_ratio': qs.stats.gain_to_pain_ratio(returns),
                
                # Additional drawdown metrics
                'avg_drawdown': qs.stats.avg_drawdown(returns),
                'avg_drawdown_days': qs.stats.avg_drawdown_days(returns),
                
                # Risk-adjusted returns
                'treynor_ratio': qs.stats.treynor_ratio(returns, benchmark_returns) if benchmark_returns is not None else None,
                'adjusted_sortino': qs.stats.adjusted_sortino(returns),
                
                # Consistency metrics
                'outlier_loss_ratio': qs.stats.outlier_loss_ratio(returns),
                'recovery_factor': qs.stats.recovery_factor(returns),
                
                # Advanced timing metrics
                'tail_ratio': qs.stats.tail_ratio(returns),
                'common_sense_ratio': qs.stats.common_sense_ratio(returns)
            })
            
        except Exception as e:
            logger.warning(f"Error calculating additional metrics: {e}")
        
        return metrics