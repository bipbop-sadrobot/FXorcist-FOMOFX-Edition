import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from scipy import stats
# import empyrical as ep  # For financial metrics - replaced with manual calculations  # For financial metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import psutil
import torch

# Ensure logs directory exists
Path("logs").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("logs/evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Financial metrics calculation functions
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio."""
    if returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return excess_returns.mean() / returns.std() * np.sqrt(252)

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio."""
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.expanding(min_periods=1).max()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns.min()

def calculate_annual_return(returns: pd.Series) -> float:
    """Calculate annual return."""
    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252  # Assuming 252 trading days per year
    if years == 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1

def calculate_annual_volatility(returns: pd.Series) -> float:
    """Calculate annual volatility."""
    return returns.std() * np.sqrt(252)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    timestamp: datetime
    model_name: str
    model_version: str
    metrics: dict[str, float]
    resource_usage: dict[str, float]
    predictions: Optional[pd.Series] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EvaluationMetrics':
        return cls(**data)

class ModelEvaluator:
    """Comprehensive model evaluation system."""
    
    def __init__(self, save_dir: Path = Path('evaluation_results')):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_metrics = {}
    
    def calculate_financial_metrics(
        self,
        returns: pd.Series,
        predictions: pd.Series,
        risk_free_rate: float = 0.02
    ) -> dict[str, float]:
        """Calculate trading-specific metrics."""
        # Convert predictions to position signals (-1, 0, 1)
        signals = np.sign(predictions)
        strategy_returns = returns * signals.shift(1)  # Avoid lookahead bias
        
        try:
            metrics = {
                'sharpe_ratio': calculate_sharpe_ratio(strategy_returns, risk_free_rate),
                'sortino_ratio': calculate_sortino_ratio(strategy_returns, risk_free_rate),
                'max_drawdown': calculate_max_drawdown(strategy_returns),
                'annual_return': calculate_annual_return(strategy_returns),
                'annual_volatility': calculate_annual_volatility(strategy_returns),
                'value_at_risk': strategy_returns.quantile(0.05) if len(strategy_returns) > 0 else 0.0
            }
        except Exception as e:
            logger.error(f"Error calculating financial metrics: {str(e)}")
            metrics = {}
        
        return metrics
    
    def calculate_prediction_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> dict[str, float]:
        """Calculate prediction accuracy metrics."""
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'directional_accuracy': np.mean(np.sign(y_true) == np.sign(y_pred))
            }
            
            # Information Coefficient (Spearman rank correlation)
            metrics['ic'] = stats.spearmanr(y_true, y_pred)[0]
            
            # Information Ratio (IC / IC std)
            rolling_ic = y_true.rolling(20).corr(y_pred)
            metrics['ir'] = rolling_ic.mean() / rolling_ic.std()
            
        except Exception as e:
            logger.error(f"Error calculating prediction metrics: {str(e)}")
            metrics = {}
        
        return metrics
    
    def calculate_resource_metrics(self) -> dict[str, float]:
        """Calculate system resource usage metrics."""
        try:
            metrics = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024 ** 3)
            }
            
            # Add GPU metrics if available
            if torch.cuda.is_available():
                metrics.update({
                    'gpu_memory_used': torch.cuda.memory_allocated() / (1024 ** 3),
                    'gpu_memory_reserved': torch.cuda.memory_reserved() / (1024 ** 3)
                })
                
        except Exception as e:
            logger.error(f"Error calculating resource metrics: {str(e)}")
            metrics = {}
        
        return metrics
    
    def evaluate_model(
        self,
        model_name: str,
        model_version: str,
        y_true: pd.Series,
        y_pred: pd.Series,
        returns: Optional[pd.Series] = None
    ) -> EvaluationMetrics:
        """Perform comprehensive model evaluation."""
        logger.info(f"Starting evaluation for {model_name} v{model_version}")
        
        try:
            # Calculate all metrics
            prediction_metrics = self.calculate_prediction_metrics(y_true, y_pred)
            resource_metrics = self.calculate_resource_metrics()
            
            # Add financial metrics if returns provided
            if returns is not None:
                financial_metrics = self.calculate_financial_metrics(returns, y_pred)
                all_metrics = {**prediction_metrics, **financial_metrics}
            else:
                all_metrics = prediction_metrics
            
            # Create evaluation result
            result = EvaluationMetrics(
                timestamp=datetime.now(),
                model_name=model_name,
                model_version=model_version,
                metrics=all_metrics,
                resource_usage=resource_metrics,
                predictions=y_pred
            )
            
            # Save results
            self._save_evaluation(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            raise
    
    def _save_evaluation(self, result: EvaluationMetrics):
        """Save evaluation results to disk."""
        try:
            save_path = self.save_dir / f"{result.model_name}_{result.timestamp:%Y%m%d_%H%M%S}.json"
            with open(save_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            logger.info(f"Evaluation results saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {str(e)}")
    
    def compare_to_baseline(
        self,
        current_metrics: EvaluationMetrics,
        baseline_name: str = 'simple_ma'
    ) -> dict[str, float]:
        """Compare model performance to baseline."""
        if baseline_name not in self.baseline_metrics:
            logger.warning(f"Baseline {baseline_name} not found")
            return {}
        
        baseline = self.baseline_metrics[baseline_name]
        improvements = {}
        
        for metric, value in current_metrics.metrics.items():
            if metric in baseline.metrics:
                pct_change = ((value - baseline.metrics[metric]) / 
                            abs(baseline.metrics[metric])) * 100
                improvements[f"{metric}_improvement"] = pct_change
        
        return improvements
    
    def detect_model_drift(
        self,
        current_metrics: EvaluationMetrics,
        historical_window: int = 20
    ) -> dict[str, bool]:
        """Detect potential model drift using statistical tests."""
        drift_indicators = {}
        
        try:
            # Load historical metrics
            historical_metrics = self._load_historical_metrics(
                current_metrics.model_name,
                historical_window
            )
            
            if not historical_metrics:
                return {}
            
            # Check for significant changes in key metrics
            for metric in ['mse', 'directional_accuracy', 'ic']:
                if metric in current_metrics.metrics:
                    historical_values = [m.metrics.get(metric) for m in historical_metrics]
                    current_value = current_metrics.metrics[metric]
                    
                    # Perform statistical test
                    z_score = (current_value - np.mean(historical_values)) / np.std(historical_values)
                    drift_indicators[f"{metric}_drift"] = abs(z_score) > 2.0
            
        except Exception as e:
            logger.error(f"Error detecting model drift: {str(e)}")
        
        return drift_indicators
    
    def _load_historical_metrics(
        self,
        model_name: str,
        window: int
    ) -> list[EvaluationMetrics]:
        """Load historical evaluation results."""
        results = []
        try:
            files = sorted(self.save_dir.glob(f"{model_name}_*.json"))[-window:]
            for f in files:
                with open(f, 'r') as file:
                    data = json.load(file)
                    results.append(EvaluationMetrics.from_dict(data))
        except Exception as e:
            logger.error(f"Error loading historical metrics: {str(e)}")
        return results

if __name__ == "__main__":
    # Example usage
    try:
        # Create evaluator
        evaluator = ModelEvaluator()
        
        # Load some test data
        df = pd.read_parquet('data/processed/ingested_forex_1min_aug2025.parquet')
        
        # Generate some test predictions
        y_true = df['returns']
        y_pred = df['returns'].rolling(20).mean()  # Simple MA strategy
        
        # Evaluate
        results = evaluator.evaluate_model(
            model_name="test_model",
            model_version="1.0.0",
            y_true=y_true,
            y_pred=y_pred,
            returns=df['returns']
        )
        
        # Check for drift
        drift = evaluator.detect_model_drift(results)
        
        logger.info("\nEvaluation Results:")
        logger.info(f"Metrics: {results.metrics}")
        logger.info(f"Resource Usage: {results.resource_usage}")
        logger.info(f"Drift Indicators: {drift}")
        
    except Exception as e:
        logger.error("Evaluation failed", exc_info=True)
        raise