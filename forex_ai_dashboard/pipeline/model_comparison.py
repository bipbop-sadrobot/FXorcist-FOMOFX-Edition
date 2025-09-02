#!/usr/bin/env python3
"""
Model Comparison and Evaluation Framework
Comprehensive evaluation with cross-validation, statistical tests, and automated model selection.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_validate
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelComparator:
    """Comprehensive model comparison and evaluation framework."""

    def __init__(self, cv_splits: int = 5, test_size: float = 0.2):
        self.cv_splits = cv_splits
        self.test_size = test_size
        self.tscv = TimeSeriesSplit(n_splits=cv_splits)
        self.results = {}

    def evaluate_models(self,
                       models: Dict[str, Any],
                       X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: pd.Series,
                       y_test: pd.Series) -> Dict[str, Dict]:
        """
        Evaluate multiple models with comprehensive metrics.

        Args:
            models: Dictionary of model_name -> model_instance
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets

        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating {len(models)} models")

        results = {}

        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}")

            try:
                # Train model
                model.fit(X_train, y_train)

                # Generate predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                # Calculate comprehensive metrics
                metrics = self._calculate_metrics(y_train, y_test, y_pred_train, y_pred_test)

                # Cross-validation scores
                cv_scores = self._cross_validate_model(model, X_train, y_train)

                # Feature importance (if available)
                feature_importance = self._get_feature_importance(model, X_train.columns)

                # Model characteristics
                model_info = self._get_model_info(model)

                results[model_name] = {
                    'metrics': metrics,
                    'cv_scores': cv_scores,
                    'feature_importance': feature_importance,
                    'model_info': model_info,
                    'predictions': {
                        'train': y_pred_train,
                        'test': y_pred_test
                    }
                }

                logger.info(".4f")

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}

        # Statistical comparison
        if len(results) > 1:
            self._statistical_comparison(results, y_test)

        self.results = results
        return results

    def _calculate_metrics(self,
                          y_train: pd.Series,
                          y_test: pd.Series,
                          y_pred_train: np.ndarray,
                          y_pred_test: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""

        # Test set metrics
        metrics = {
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_mape': mean_absolute_percentage_error(y_test, y_pred_test),
            'test_r2': r2_score(y_test, y_pred_test),
            'test_explained_variance': explained_variance_score(y_test, y_pred_test),
        }

        # Training set metrics
        metrics.update({
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_r2': r2_score(y_train, y_pred_train),
        })

        # Additional metrics
        metrics.update(self._calculate_additional_metrics(y_test, y_pred_test))

        return metrics

    def _calculate_additional_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate additional performance metrics."""
        additional_metrics = {}

        # Directional accuracy
        true_direction = np.sign(y_true - y_true.shift(1))
        pred_direction = np.sign(y_pred - y_true.shift(1))
        directional_accuracy = np.mean(true_direction == pred_direction)
        additional_metrics['directional_accuracy'] = directional_accuracy

        # Profit factor (simplified)
        returns = y_true.pct_change()
        pred_returns = pd.Series(y_pred).pct_change()

        # Win rate
        wins = np.sum((y_pred[1:] - y_true[:-1]) * (y_true[1:] - y_true[:-1]) > 0)
        total = len(y_pred) - 1
        win_rate = wins / total if total > 0 else 0
        additional_metrics['win_rate'] = win_rate

        # Sharpe ratio (simplified)
        returns_std = np.std(returns)
        if returns_std > 0:
            sharpe_ratio = np.mean(returns) / returns_std * np.sqrt(252)
            additional_metrics['sharpe_ratio'] = sharpe_ratio

        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        additional_metrics['max_drawdown'] = max_drawdown

        # Information ratio
        tracking_error = np.std(returns - pred_returns)
        if tracking_error > 0:
            information_ratio = np.mean(returns - pred_returns) / tracking_error
            additional_metrics['information_ratio'] = information_ratio

        return additional_metrics

    def _cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[float]]:
        """Perform time series cross-validation."""
        cv_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

        try:
            cv_results = cross_validate(
                model, X, y,
                cv=self.tscv,
                scoring=cv_metrics,
                return_train_score=False
            )

            # Convert negative scores to positive
            cv_scores = {
                'mse': [-x for x in cv_results['test_neg_mean_squared_error']],
                'mae': [-x for x in cv_results['test_neg_mean_absolute_error']],
                'r2': cv_results['test_r2']
            }

            # Add summary statistics
            for metric, scores in cv_scores.items():
                cv_scores[f'{metric}_mean'] = np.mean(scores)
                cv_scores[f'{metric}_std'] = np.std(scores)
                cv_scores[f'{metric}_min'] = np.min(scores)
                cv_scores[f'{metric}_max'] = np.max(scores)

            return cv_scores

        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
            return {'error': str(e)}

    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance if available."""
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                # For linear models
                return dict(zip(feature_names, np.abs(model.coef_)))
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return None

    def _get_model_info(self, model: Any) -> Dict[str, Any]:
        """Get model information and parameters."""
        info = {
            'model_type': type(model).__name__,
            'model_module': type(model).__module__
        }

        # Get model parameters
        try:
            params = model.get_params()
            info['parameters'] = params
        except:
            info['parameters'] = 'Not available'

        return info

    def _statistical_comparison(self, results: Dict, y_test: pd.Series):
        """Perform statistical comparison between models."""
        logger.info("Performing statistical model comparison")

        # Extract test predictions for each model
        model_predictions = {}
        for model_name, result in results.items():
            if 'predictions' in result and 'test' in result['predictions']:
                model_predictions[model_name] = result['predictions']['test']

        if len(model_predictions) < 2:
            return

        # Calculate prediction errors
        model_errors = {}
        for model_name, predictions in model_predictions.items():
            errors = y_test - predictions
            model_errors[model_name] = errors

        # Perform pairwise t-tests
        model_names = list(model_errors.keys())
        statistical_tests = {}

        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                errors1, errors2 = model_errors[model1], model_errors[model2]

                # Paired t-test
                t_stat, p_value = stats.ttest_rel(np.abs(errors1), np.abs(errors2))

                statistical_tests[f'{model1}_vs_{model2}'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'model1_better': np.mean(np.abs(errors1)) < np.mean(np.abs(errors2))
                }

        # Store statistical comparison results
        for model_name in results:
            if 'metrics' in results[model_name]:
                results[model_name]['statistical_comparison'] = statistical_tests

    def select_best_model(self, criterion: str = 'test_r2') -> Tuple[str, Dict]:
        """
        Select the best model based on specified criterion.

        Args:
            criterion: Metric to use for selection (e.g., 'test_r2', 'test_rmse')

        Returns:
            Tuple of (best_model_name, best_model_results)
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_models first.")

        best_model = None
        best_score = -float('inf') if 'r2' in criterion else float('inf')
        best_results = None

        for model_name, result in self.results.items():
            if 'metrics' not in result:
                continue

            score = result['metrics'].get(criterion)
            if score is None:
                continue

            if 'r2' in criterion:
                # Higher is better for R²
                if score > best_score:
                    best_score = score
                    best_model = model_name
                    best_results = result
            else:
                # Lower is better for error metrics
                if score < best_score:
                    best_score = score
                    best_model = model_name
                    best_results = result

        if best_model is None:
            raise ValueError(f"No valid results found for criterion: {criterion}")

        logger.info(f"Selected best model: {best_model} (criterion: {criterion}, score: {best_score:.4f})")

        return best_model, best_results

    def generate_comparison_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive comparison report."""
        if not self.results:
            return "No evaluation results available."

        report = ".1f" \
                "MODEL COMPARISON REPORT\n" \
                "=" * 50 + "\n\n"

        # Summary table
        report += "MODEL PERFORMANCE SUMMARY:\n"
        report += "-" * 80 + "\n"
        report += "<25"
        report += "-" * 80 + "\n"

        for model_name, result in self.results.items():
            if 'metrics' not in result:
                continue

            metrics = result['metrics']
            report += "<25"
            report += ".4f"
            report += ".4f"
            report += ".4f"
            report += ".4f"
            report += ".4f"
            report += "\n"

        report += "\n"

        # Cross-validation results
        report += "CROSS-VALIDATION RESULTS:\n"
        report += "-" * 50 + "\n"

        for model_name, result in self.results.items():
            if 'cv_scores' not in result or 'error' in result['cv_scores']:
                continue

            cv_scores = result['cv_scores']
            report += f"\n{model_name.upper()}:\n"
            report += ".4f"
            report += ".4f"
            report += ".4f"
            report += ".4f"

        # Best model recommendation
        try:
            best_model, best_results = self.select_best_model('test_r2')
            report += f"\n\n🏆 RECOMMENDED MODEL: {best_model.upper()}\n"
            report += ".4f"
        except:
            report += "\n\n⚠️  Could not determine best model automatically."

        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Comparison report saved to {save_path}")

        return report

    def get_model_rankings(self, metric: str = 'test_r2') -> List[Tuple[str, float]]:
        """Get models ranked by specified metric."""
        rankings = []

        for model_name, result in self.results.items():
            if 'metrics' in result and metric in result['metrics']:
                score = result['metrics'][metric]
                rankings.append((model_name, score))

        # Sort by metric (higher better for R², lower better for errors)
        reverse = 'r2' in metric
        rankings.sort(key=lambda x: x[1], reverse=reverse)

        return rankings

    def save_results(self, output_dir: str = "evaluation_results"):
        """Save all evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = output_path / f"model_evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for model_name, result in self.results.items():
                serializable_results[model_name] = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        serializable_results[model_name][key] = value.tolist()
                    elif isinstance(value, dict):
                        serializable_results[model_name][key] = {}
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, np.ndarray):
                                serializable_results[model_name][key][sub_key] = sub_value.tolist()
                            else:
                                serializable_results[model_name][key][sub_key] = sub_value
                    else:
                        serializable_results[model_name][key] = value

            json.dump(serializable_results, f, indent=2, default=str)

        # Save comparison report
        report_file = output_path / f"model_comparison_report_{timestamp}.txt"
        report = self.generate_comparison_report(str(report_file))

        logger.info(f"Evaluation results saved to {output_path}")

        return str(output_path)