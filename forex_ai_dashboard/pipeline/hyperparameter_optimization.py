#!/usr/bin/env python3
"""
Hyperparameter Optimization Module for Forex AI
Uses Optuna for automated hyperparameter tuning with advanced optimization strategies.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import mlflow
import mlflow.optuna

# Import model classes
from forex_ai_dashboard.models.catboost_model import CatBoostModel

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using Optuna."""

    def __init__(self,
                 study_name: str = "forex_ai_optimization",
                 storage_path: str = "optuna_studies",
                 n_trials: int = 100,
                 timeout: int = 3600):
        """
        Initialize the hyperparameter optimizer.

        Args:
            study_name: Name for the Optuna study
            storage_path: Path to store study results
            n_trials: Number of optimization trials
            timeout: Timeout in seconds for optimization
        """
        self.study_name = study_name
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.n_trials = n_trials
        self.timeout = timeout

        # Create or load study
        storage_url = f"sqlite:///{self.storage_path}/{study_name}.db"
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            sampler=TPESampler(),
            pruner=MedianPruner(),
            direction="maximize",  # Maximize R² score
            load_if_exists=True
        )

        # Setup MLflow integration
        mlflow.set_experiment(f"optuna_{study_name}")

    def optimize_catboost(self,
                         X_train: pd.DataFrame,
                         X_test: pd.DataFrame,
                         y_train: pd.Series,
                         y_test: pd.Series,
                         base_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize CatBoost hyperparameters.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            base_params: Base parameters to keep fixed

        Returns:
            Best parameters and optimization results
        """
        logger.info("Starting CatBoost hyperparameter optimization")

        def objective(trial):
            """Objective function for CatBoost optimization."""
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'loss_function': 'RMSE',
                'random_seed': 42,
                'verbose': False,
                'early_stopping_rounds': 50
            }

            # Add base parameters if provided
            if base_params:
                params.update(base_params)

            # Train model
            model = CatBoostModel(**params)
            model.train(X_train, y_train, X_test, y_test)

            # Evaluate
            y_pred = model.predict(X_test)
            r2_score = model.evaluate(y_test, y_pred)['r2']

            # Log to MLflow
            with mlflow.start_run():
                mlflow.log_params(params)
                mlflow.log_metric("r2_score", r2_score)
                mlflow.log_metric("trial_number", trial.number)

            return r2_score

        # Run optimization
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        # Get best parameters
        best_params = self.study.best_params
        best_score = self.study.best_value

        logger.info(".4f")
        logger.info(f"Best parameters: {best_params}")

        # Save optimization results
        results = {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(self.study.trials),
            'optimization_history': [
                {
                    'trial': t.number,
                    'params': t.params,
                    'value': t.value
                }
                for t in self.study.trials if t.value is not None
            ],
            'timestamp': datetime.now().isoformat()
        }

        self._save_results(results, 'catboost')

        return results

    def optimize_lightgbm(self,
                         X_train: pd.DataFrame,
                         X_test: pd.DataFrame,
                         y_train: pd.Series,
                         y_test: pd.Series) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters.
        """
        logger.info("Starting LightGBM hyperparameter optimization")

        def objective(trial):
            """Objective function for LightGBM optimization."""
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'random_state': 42,
                'verbosity': -1
            }

            try:
                import lightgbm as lgb

                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )

                y_pred = model.predict(X_test)
                r2_score = r2_score(y_test, y_pred)

                # Log to MLflow
                with mlflow.start_run():
                    mlflow.log_params(params)
                    mlflow.log_metric("r2_score", r2_score)

                return r2_score

            except ImportError:
                logger.warning("LightGBM not available, skipping trial")
                return -float('inf')

        # Run optimization
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = self.study.best_params
        best_score = self.study.best_value

        results = {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(self.study.trials),
            'timestamp': datetime.now().isoformat()
        }

        self._save_results(results, 'lightgbm')
        return results

    def optimize_xgboost(self,
                        X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters.
        """
        logger.info("Starting XGBoost hyperparameter optimization")

        def objective(trial):
            """Objective function for XGBoost optimization."""
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': 'gbtree',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'random_state': 42
            }

            try:
                from xgboost import XGBRegressor
                from sklearn.metrics import r2_score

                model = XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

                y_pred = model.predict(X_test)
                r2_score = r2_score(y_test, y_pred)

                # Log to MLflow
                with mlflow.start_run():
                    mlflow.log_params(params)
                    mlflow.log_metric("r2_score", r2_score)

                return r2_score

            except ImportError:
                logger.warning("XGBoost not available, skipping trial")
                return -float('inf')

        # Run optimization
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = self.study.best_params
        best_score = self.study.best_value

        results = {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(self.study.trials),
            'timestamp': datetime.now().isoformat()
        }

        self._save_results(results, 'xgboost')
        return results

    def optimize_ensemble(self,
                         X_train: pd.DataFrame,
                         X_test: pd.DataFrame,
                         y_train: pd.Series,
                         y_test: pd.Series) -> Dict[str, Any]:
        """
        Optimize ensemble model hyperparameters.
        """
        logger.info("Starting Ensemble hyperparameter optimization")

        def objective(trial):
            """Objective function for ensemble optimization."""
            from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
            from sklearn.metrics import r2_score

            # Random Forest parameters
            rf_params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
                'random_state': 42
            }

            # Extra Trees parameters
            et_params = {
                'n_estimators': trial.suggest_int('et_n_estimators', 50, 300),
                'max_depth': trial.suggest_int('et_max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('et_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('et_min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('et_max_features', ['sqrt', 'log2', None]),
                'random_state': 42
            }

            # Ensemble weights
            rf_weight = trial.suggest_float('rf_weight', 0.1, 0.9)
            et_weight = 1.0 - rf_weight

            # Train models
            rf_model = RandomForestRegressor(**rf_params)
            et_model = ExtraTreesRegressor(**et_params)

            rf_model.fit(X_train, y_train)
            et_model.fit(X_train, y_train)

            # Ensemble predictions
            rf_pred = rf_model.predict(X_test)
            et_pred = et_model.predict(X_test)
            ensemble_pred = rf_weight * rf_pred + et_weight * et_pred

            r2_score = r2_score(y_test, ensemble_pred)

            # Log to MLflow
            with mlflow.start_run():
                mlflow.log_params({**rf_params, **et_params, 'rf_weight': rf_weight, 'et_weight': et_weight})
                mlflow.log_metric("r2_score", r2_score)

            return r2_score

        # Run optimization
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = self.study.best_params
        best_score = self.study.best_value

        results = {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(self.study.trials),
            'timestamp': datetime.now().isoformat()
        }

        self._save_results(results, 'ensemble')
        return results

    def _save_results(self, results: Dict, model_type: str):
        """Save optimization results to file."""
        results_path = self.storage_path / f"{model_type}_optimization_results.json"

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Optimization results saved to {results_path}")

    def get_study_summary(self) -> Dict:
        """Get summary of the optimization study."""
        return {
            'study_name': self.study_name,
            'n_trials': len(self.study.trials),
            'best_score': self.study.best_value,
            'best_params': self.study.best_params,
            'completed_trials': len([t for t in self.study.trials if t.state == optuna.TrialState.COMPLETE]),
            'pruned_trials': len([t for t in self.study.trials if t.state == optuna.TrialState.PRUNED])
        }

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot optimization history
            optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=ax1)
            ax1.set_title("Optimization History")

            # Plot parameter importance
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax2)
            ax2.set_title("Parameter Importances")

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Optimization plots saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib/seaborn not available for plotting")

def main():
    """Example usage of hyperparameter optimization."""
    # This would be called from the main training pipeline
    optimizer = HyperparameterOptimizer(
        study_name="forex_ai_optimization",
        n_trials=50,
        timeout=1800  # 30 minutes
    )

    # Load your data here
    # X_train, X_test, y_train, y_test = load_your_data()

    # Example optimization calls
    # catboost_results = optimizer.optimize_catboost(X_train, X_test, y_train, y_test)
    # lightgbm_results = optimizer.optimize_lightgbm(X_train, X_test, y_train, y_test)
    # xgboost_results = optimizer.optimize_xgboost(X_train, X_test, y_train, y_test)
    # ensemble_results = optimizer.optimize_ensemble(X_train, X_test, y_train, y_test)

    # Print summary
    summary = optimizer.get_study_summary()
    print("Optimization Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()