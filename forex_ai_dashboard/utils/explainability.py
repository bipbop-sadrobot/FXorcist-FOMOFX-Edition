"""Model explainability utilities using SHAP (SHapley Additive exPlanations).

This module provides model-agnostic explainability for machine learning models
using the SHAP library. It supports:
- Global model explanations (feature importance)
- Instance-level explanations
- Visualization integration
- Multiple model types (tree-based, neural networks, etc.)

Key Features:
1. Unified interface for different model types
2. Automatic explainer selection based on model type
3. Integration with logging and visualization
4. Pandas DataFrame compatibility

Example Usage:
    ```python
    from utils.explainability import ModelExplainer
    import shap
    
    # Initialize with trained model and feature names
    explainer = ModelExplainer(model, feature_names=['feature1', 'feature2'])
    
    # Initialize with a masker for text data
    text_masker = shap.maskers.Text(tokenizer=...)
    explainer_text = ModelExplainer(model, feature_names=['text_feature'], masker=text_masker)
    
    # Get global explanations
    global_exp = explainer.explain(X_test)
    print(global_exp['feature_importance'])
    
    # Get instance explanation
    instance_exp = explainer.explain_instance(X_test.iloc[[0]])
    ```

Design Decisions:
- Uses SHAP for consistent, theoretically-grounded explanations
- Wraps SHAP complexity in simple interface
- Returns structured data for easy integration
- Built-in visualization support
"""
from typing import Any
import numpy as np
import pandas as pd
import shap
from loguru import logger

class ModelExplainer:
    """Provides model-agnostic explainability using SHAP.
    
    Handles different model types automatically and provides:
    - Global feature importance
    - Instance-specific explanations
    - Visualization-ready outputs
    
    The explainer automatically selects the appropriate SHAP algorithm:
    - TreeExplainer for tree-based models
    - DeepExplainer for neural networks
    - KernelExplainer as fallback
    
    Args:
        model: Any trained ML model (XGBoost, CatBoost, PyTorch, etc.)
        feature_names: List of feature names for interpretable outputs
        
    Attributes:
        explainer: Configured SHAP explainer instance
        feature_names: Feature names for interpretation
    """
    
    def __init__(self, model: Any, feature_names: list[str], masker=None):
        """Initialize explainer for a trained model.
        
        Args:
            model: Trained model (XGBoost, CatBoost, PyTorch etc.)
            feature_names: List of feature names for interpretation
            masker: Optional SHAP masker/background data to use. 
                   If None, will use model as masker for tree/linear models.
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = self._create_explainer(masker)
        logger.info(f"Initialized explainer for {type(model).__name__} model")

    def _create_explainer(self, masker=None) -> shap.Explainer:
        """Create appropriate SHAP explainer based on model type."""
        try:
            if hasattr(self.model, 'predict_proba'):
                return shap.Explainer(self.model.predict_proba, masker=masker or self.model)
            return shap.Explainer(self.model, masker=masker)
        except Exception as e:
            logger.error(f"Failed to create explainer: {e}")
            raise

    def explain(self, X: pd.DataFrame, plot_options: dict = None) -> dict:
        """Generate comprehensive SHAP explanations for input data.
        
        Computes SHAP values and creates:
        - Feature importance rankings
        - Summary visualizations
        - Raw SHAP values for custom analysis
        
        Args:
            X: Input features as DataFrame (shape n_samples × n_features)
            plot_options: Dictionary of plotting options. See shap.summary_plot for available options.
            
        Returns:
            Dictionary with:
            - 'shap_values': Raw SHAP values (n_samples × n_features array)
            - 'feature_importance': DataFrame of mean absolute SHAP values
            - 'summary_plot': SHAP summary plot visualization
            
        Note:
            The summary plot shows feature importance and value impacts
        """
        shap_values = self.explainer(X)
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values.values).mean(0)
        }).sort_values('importance', ascending=False)
        
        if plot_options is None:
            plot_options = {}
        
        try:
            plot = shap.summary_plot(shap_values, X, feature_names=self.feature_names, **plot_options)
        except Exception as e:
            logger.warning(f"Could not generate summary plot: {e}")
            plot = None
            
        return {
            'shap_values': shap_values,
            'feature_importance': importance_df,
            'summary_plot': plot
        }

    def explain_instance(self, instance: pd.DataFrame) -> dict:
        """Generate explanation for a single instance."""
        explanation = self.explain(instance)
        try:
            force_plot = shap.plots.force(explanation['shap_values'][0])
        except Exception as e:
            logger.warning(f"Could not generate force plot: {e}")
            force_plot = None
            
        return {
            **explanation,
            'force_plot': force_plot
        }
