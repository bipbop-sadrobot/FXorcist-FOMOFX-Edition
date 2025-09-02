#!/usr/bin/env python3
"""
Model Interpretability Module
Uses SHAP and other techniques to explain model predictions and feature importance.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelInterpreter:
    """Comprehensive model interpretability using SHAP and other techniques."""

    def __init__(self):
        self.shap_available = self._check_shap_availability()
        self.interpretation_results = {}

    def _check_shap_availability(self) -> bool:
        """Check if SHAP is available."""
        try:
            import shap
            return True
        except ImportError:
            logger.warning("SHAP not available. Install with: pip install shap")
            return False

    def explain_model(self,
                     model: Any,
                     X_train: pd.DataFrame,
                     X_test: pd.DataFrame,
                     y_test: pd.Series,
                     model_name: str = "model",
                     max_evals: int = 1000) -> Dict[str, Any]:
        """
        Generate comprehensive model explanations.

        Args:
            model: Trained model to explain
            X_train: Training features
            X_test: Test features
            y_test: Test targets
            model_name: Name for the model
            max_evals: Maximum evaluations for SHAP

        Returns:
            Dictionary containing various explanations
        """
        logger.info(f"Generating explanations for {model_name}")

        explanations = {
            'model_name': model_name,
            'feature_importance': {},
            'shap_values': None,
            'partial_dependence': {},
            'feature_interactions': {},
            'prediction_explanations': {},
            'timestamp': datetime.now().isoformat()
        }

        # Feature importance
        explanations['feature_importance'] = self._get_feature_importance(model, X_train)

        # SHAP explanations
        if self.shap_available:
            explanations['shap_values'] = self._calculate_shap_values(model, X_train, X_test, max_evals)
            explanations['shap_summary'] = self._shap_summary_plot(explanations['shap_values'], X_test)

        # Partial dependence plots
        explanations['partial_dependence'] = self._calculate_partial_dependence(model, X_train, X_test)

        # Feature interactions
        explanations['feature_interactions'] = self._analyze_feature_interactions(model, X_train)

        # Individual prediction explanations
        explanations['prediction_explanations'] = self._explain_predictions(model, X_test, y_test)

        self.interpretation_results[model_name] = explanations

        return explanations

    def _get_feature_importance(self, model: Any, X: pd.DataFrame) -> Dict[str, float]:
        """Extract feature importance from the model."""
        importance_dict = {}

        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance_dict = dict(zip(X.columns, model.feature_importances_))

            elif hasattr(model, 'coef_'):
                # Linear models
                coef = model.coef_
                if len(coef.shape) > 1:
                    # Multi-output case
                    importance_dict = {f'feature_{i}': np.abs(coef[:, i]).mean()
                                     for i in range(coef.shape[1])}
                else:
                    importance_dict = dict(zip(X.columns, np.abs(coef)))

            elif hasattr(model, 'feature_importance'):
                # LightGBM
                importance_dict = dict(zip(X.columns, model.feature_importance()))

            else:
                logger.warning("Could not extract feature importance from model")
                return {}

            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(),
                                        key=lambda x: x[1], reverse=True))

            return importance_dict

        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            return {}

    def _calculate_shap_values(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, max_evals: int):
        """Calculate SHAP values for model explanations."""
        if not self.shap_available:
            return None

        try:
            import shap

            # Choose appropriate explainer based on model type
            if hasattr(model, 'predict_proba'):
                # Classification-like models
                explainer = shap.TreeExplainer(model)
            else:
                # Regression models
                explainer = shap.TreeExplainer(model)

            # Calculate SHAP values
            if len(X_test) > max_evals:
                # Sample for efficiency
                sample_indices = np.random.choice(len(X_test), size=max_evals, replace=False)
                X_sample = X_test.iloc[sample_indices]
            else:
                X_sample = X_test

            shap_values = explainer.shap_values(X_sample)

            return {
                'values': shap_values,
                'base_values': explainer.expected_value,
                'data': X_sample,
                'feature_names': X_sample.columns.tolist()
            }

        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")
            return None

    def _shap_summary_plot(self, shap_data: Dict, X: pd.DataFrame) -> Dict:
        """Generate SHAP summary statistics."""
        if shap_data is None:
            return {}

        try:
            shap_values = shap_data['values']
            feature_names = shap_data['feature_names']

            # Calculate mean absolute SHAP values for each feature
            if isinstance(shap_values, list):
                # Multi-class case
                mean_shap = np.abs(shap_values[0]).mean(axis=0)
            else:
                # Single output case
                mean_shap = np.abs(shap_values).mean(axis=0)

            # Create summary
            summary = {
                'mean_abs_shap': dict(zip(feature_names, mean_shap)),
                'top_features': sorted(zip(feature_names, mean_shap),
                                      key=lambda x: x[1], reverse=True)[:10]
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating SHAP summary: {str(e)}")
            return {}

    def _calculate_partial_dependence(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict:
        """Calculate partial dependence for key features."""
        pdp_results = {}

        try:
            # Get top features by importance
            importance = self._get_feature_importance(model, X_train)
            top_features = list(importance.keys())[:5]  # Top 5 features

            for feature in top_features:
                if feature in X_test.columns:
                    pdp_results[feature] = self._single_feature_pdp(model, X_test, feature)

        except Exception as e:
            logger.error(f"Error calculating partial dependence: {str(e)}")

        return pdp_results

    def _single_feature_pdp(self, model: Any, X: pd.DataFrame, feature: str, n_points: int = 20) -> Dict:
        """Calculate partial dependence for a single feature."""
        try:
            feature_values = X[feature].values
            feature_min, feature_max = np.min(feature_values), np.max(feature_values)

            # Create grid of feature values
            grid = np.linspace(feature_min, feature_max, n_points)

            partial_dependence = []

            for value in grid:
                # Create copy of data with feature set to current value
                X_copy = X.copy()
                X_copy[feature] = value

                # Get predictions
                predictions = model.predict(X_copy)
                partial_dependence.append(np.mean(predictions))

            return {
                'feature_values': grid.tolist(),
                'partial_dependence': partial_dependence,
                'feature_range': [feature_min, feature_max]
            }

        except Exception as e:
            logger.error(f"Error calculating PDP for {feature}: {str(e)}")
            return {}

    def _analyze_feature_interactions(self, model: Any, X: pd.DataFrame) -> Dict:
        """Analyze feature interactions."""
        interactions = {}

        try:
            # Get feature importance
            importance = self._get_feature_importance(model, X)
            top_features = list(importance.keys())[:5]

            # Calculate correlation matrix for top features
            corr_matrix = X[top_features].corr()

            # Find strongest interactions
            interactions['correlations'] = corr_matrix.to_dict()

            # Calculate mutual information between features
            from sklearn.feature_selection import mutual_info_regression

            mi_matrix = {}
            for i, feature1 in enumerate(top_features):
                mi_matrix[feature1] = {}
                for feature2 in top_features[i+1:]:
                    mi = mutual_info_regression(X[[feature1]], X[feature2])[0]
                    mi_matrix[feature1][feature2] = mi

            interactions['mutual_information'] = mi_matrix

        except Exception as e:
            logger.error(f"Error analyzing feature interactions: {str(e)}")

        return interactions

    def _explain_predictions(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, n_samples: int = 5) -> Dict:
        """Explain individual predictions."""
        explanations = {}

        try:
            # Sample some predictions to explain
            sample_indices = np.random.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)

            for idx in sample_indices:
                sample = X_test.iloc[idx:idx+1]
                true_value = y_test.iloc[idx]
                predicted_value = model.predict(sample)[0]

                explanation = {
                    'true_value': true_value,
                    'predicted_value': predicted_value,
                    'prediction_error': abs(true_value - predicted_value),
                    'feature_contributions': {}
                }

                # Get feature contributions if available
                if hasattr(model, 'feature_importances_'):
                    # For tree-based models, approximate feature contributions
                    feature_contribs = {}
                    for feature in X_test.columns:
                        # Create baseline prediction
                        baseline = sample.copy()
                        baseline[feature] = X_test[feature].mean()

                        baseline_pred = model.predict(baseline)[0]
                        feature_contribs[feature] = predicted_value - baseline_pred

                    explanation['feature_contributions'] = dict(
                        sorted(feature_contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                    )

                explanations[f'sample_{idx}'] = explanation

        except Exception as e:
            logger.error(f"Error explaining predictions: {str(e)}")

        return explanations

    def generate_interpretability_report(self, model_name: str, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive interpretability report."""
        if model_name not in self.interpretation_results:
            return f"No interpretation results available for {model_name}"

        results = self.interpretation_results[model_name]

        report = f"""
MODEL INTERPRETABILITY REPORT - {model_name.upper()}
{'='*60}

1. FEATURE IMPORTANCE:
{'-'*30}
"""

        # Feature importance
        if results.get('feature_importance'):
            for i, (feature, importance) in enumerate(results['feature_importance'].items()):
                report += ".4f"
                if i >= 9:  # Show top 10
                    break

        report += "\n\n2. SHAP ANALYSIS:\n"
        report += "-"*30 + "\n"

        if results.get('shap_summary') and results['shap_summary'].get('top_features'):
            report += "Top features by mean |SHAP value|:\n"
            for feature, shap_value in results['shap_summary']['top_features'][:5]:
                report += ".4f"

        report += "\n\n3. PREDICTION EXPLANATIONS:\n"
        report += "-"*30 + "\n"

        if results.get('prediction_explanations'):
            for sample_name, explanation in list(results['prediction_explanations'].items())[:3]:
                report += f"\n{sample_name.upper()}:\n"
                report += ".6f"
                report += ".6f"
                report += ".6f"

                if explanation.get('feature_contributions'):
                    report += "  Top contributing features:\n"
                    for feature, contrib in list(explanation['feature_contributions'].items())[:3]:
                        report += ".6f"

        report += "\n\n4. FEATURE INTERACTIONS:\n"
        report += "-"*30 + "\n"

        if results.get('feature_interactions') and results['feature_interactions'].get('correlations'):
            corr = results['feature_interactions']['correlations']
            features = list(corr.keys())[:3]
            report += "Correlation matrix (top features):\n"
            for f1 in features:
                for f2 in features:
                    if f1 != f2 and f1 in corr and f2 in corr[f1]:
                        report += ".3f"

        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Interpretability report saved to {save_path}")

        return report

    def compare_models_interpretability(self, model_names: List[str]) -> str:
        """Compare interpretability across multiple models."""
        if not all(name in self.interpretation_results for name in model_names):
            return "Not all models have interpretation results available"

        report = f"""
MODEL INTERPRETABILITY COMPARISON
{'='*50}

Models compared: {', '.join(model_names)}
"""

        # Compare feature importance rankings
        report += "\n1. FEATURE IMPORTANCE COMPARISON:\n"
        report += "-"*40 + "\n"

        all_features = set()
        for name in model_names:
            if self.interpretation_results[name].get('feature_importance'):
                all_features.update(self.interpretation_results[name]['feature_importance'].keys())

        # Get top 5 features for each model
        for name in model_names:
            report += f"\n{name.upper()} - Top 5 features:\n"
            importance = self.interpretation_results[name].get('feature_importance', {})
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

            for feature, imp in top_features:
                report += ".4f"

        # Compare SHAP summaries
        report += "\n\n2. SHAP VALUE COMPARISON:\n"
        report += "-"*40 + "\n"

        for name in model_names:
            shap_summary = self.interpretation_results[name].get('shap_summary', {})
            if shap_summary.get('top_features'):
                report += f"\n{name.upper()} - Top SHAP features:\n"
                for feature, shap_val in shap_summary['top_features'][:3]:
                    report += ".4f"

        return report

    def save_interpretation_results(self, output_dir: str = "interpretation_results"):
        """Save all interpretation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save each model's interpretation
        for model_name, results in self.interpretation_results.items():
            results_file = output_path / f"{model_name}_interpretation_{timestamp}.json"

            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, dict):
                    serializable_results[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, np.ndarray):
                            serializable_results[key][sub_key] = sub_value.tolist()
                        else:
                            serializable_results[key][sub_key] = sub_value
                else:
                    serializable_results[key] = value

            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Interpretation results saved to {output_path}")
        return str(output_path)