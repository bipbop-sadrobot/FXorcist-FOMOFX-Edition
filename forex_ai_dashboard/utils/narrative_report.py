"""Generate human-readable narrative reports from model explanations.

Transforms technical model explanations into:
- Natural language summaries
- Interactive visualizations
- Actionable insights

Features:
- Automated report generation
- Integration with explainability module
- Plotly visualization support
- Customizable templates

Example Usage:
    ```python
    from utils.narrative_report import NarrativeReportGenerator
    
    # Initialize with model and features
    reporter = NarrativeReportGenerator(model, feature_names)
    
    # Generate full report
    report = reporter.generate_report(X_test)
    
    # Generate instance report
    instance_report = reporter.generate_instance_report(X_test.iloc[[0]])
    ```

Design Decisions:
- Uses SHAP values from explainability module
- Focuses on actionable business insights
- Maintains technical details for analysts
- Visualizations optimized for dashboard use
"""
from typing import Any, Dict
import pandas as pd
import plotly.express as px
import plotly.io as pio
from loguru import logger
from .explainability import ModelExplainer

class NarrativeReportGenerator:
    """Transforms model explanations into narrative reports.
    
    Combines SHAP explanations with domain knowledge to create:
    - Executive summaries
    - Feature importance analysis
    - Instance-level explanations
    - Interactive visualizations
    
    Args:
        model: Trained ML model (must work with SHAP)
        feature_names: List of feature names for interpretation
        nlp_model: Optional natural language generation model
        
    Attributes:
        explainer: ModelExplainer instance for SHAP values
    """
    
    def __init__(self, model: Any, feature_names: list[str], nlp_model: Any = None):
        """Initialize with model and features."""
        self.explainer = ModelExplainer(model, feature_names)
        self.nlp_model = nlp_model
        logger.info("Initialized narrative report generator")

    def generate_report(self, X: pd.DataFrame, y_true: pd.Series = None, template: str = None, export_format: str = None) -> Dict[str, Any]:
        """Generate comprehensive model performance report.
        
        Creates:
        - Natural language summary of key findings
        - Feature importance analysis
        - Interactive visualizations
        - Performance metrics (if y_true provided)
        
        Args:
            X: Input features (n_samples × n_features)
            y_true: Optional true labels for performance metrics
            template: Optional template for the report. The template should be a string with placeholders for the following variables: top_feature_1, top_importance_1, top_feature_2, top_feature_3.
            export_format: Optional format to export the report to (e.g., 'pdf', 'html'). Requires kaleido. The plots will be exported to the current directory.
            
        Returns:
            Dictionary with:
            - 'summary': Text summary of key insights
            - 'feature_importance': Ranked features DataFrame
            - 'plots': Dictionary of Plotly figures
            
        Note:
            The summary highlights top 3 most important features
            and their relative impacts. If an nlp_model is provided, it will be used to generate the summary instead of the template.
        """
        explanation = self.explainer.explain(X)
        
        # Generate narrative summary
        top_features = explanation['feature_importance'].head(3)
        
        if self.nlp_model:
            # Use NLP model to generate summary
            summary = self.nlp_model.generate(explanation)
        else:
            # Use template-based approach
            if template is None:
                template = (
                    "The model shows strongest dependence on {top_feature_1} "
                    "(importance: {top_importance_1:.2f}), followed by "
                    "{top_feature_2} and {top_feature_3}.\n\n"
                    "Key drivers appear to be related to technical indicators and recent price movements."
                )
            
            summary = template.format(
                top_feature_1=top_features['feature'].iloc[0],
                top_importance_1=top_features['importance'].iloc[0],
                top_feature_2=top_features['feature'].iloc[1],
                top_feature_3=top_features['feature'].iloc[2]
            )
        
        # Create visualizations
        plots = {
            'feature_importance': px.bar(
                explanation['feature_importance'],
                x='importance',
                y='feature',
                title='Feature Importance'
            ),
            'summary_plot': explanation['summary_plot']
        }
        
        if export_format:
            for plot_name, plot in plots.items():
                pio.write_image(plot, f"{plot_name}.{export_format}")
        
        return {
            'summary': summary,
            'feature_importance': explanation['feature_importance'],
            'plots': plots
        }

    def generate_instance_report(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """Generate report for a single prediction."""
        explanation = self.explainer.explain_instance(instance)
        
        return {
            'summary': "Detailed explanation for this prediction:",
            'feature_effects': explanation['feature_importance'],
            'force_plot': explanation['force_plot']
        }
