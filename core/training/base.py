"""
Base training pipeline implementation for FXorcist.
Provides core functionality for model training and evaluation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..models.model_hierarchy import ModelHierarchy
from ..pipeline.data_ingestion import load_data
from ..pipeline.feature_engineering import FeatureGenerator
from ..pipeline.evaluation_metrics import EvaluationMetrics

logger = logging.getLogger(__name__)

class BaseTrainer:
    """Base training pipeline implementation."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the training pipeline.

        Args:
            config: Training configuration dictionary containing:
                - data_dir: Path to data directory
                - models_dir: Path to models directory
                - batch_size: Training batch size
                - epochs: Number of training epochs
                - learning_rate: Initial learning rate
                - validation_split: Validation data ratio
                - early_stopping_patience: Early stopping patience
                - model_type: Type of model to train
        """
        self.config = config
        self.data_dir = Path(config['data_dir'])
        self.models_dir = Path(config['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.feature_generator = FeatureGenerator()
        self.model = ModelHierarchy(config['model_type'])
        self.metrics = EvaluationMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.training_history = []

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare data for training.

        Returns:
            Tuple containing (X_train, X_val, y_train, y_val)
        """
        logger.info("Loading and preparing data...")
        
        # Load raw data
        df = load_data(self.data_dir)
        
        # Generate features
        features_df = self.feature_generator.generate_features(df)
        
        # Prepare target variable (next period returns)
        target = features_df['close'].pct_change().shift(-1).dropna()
        features = features_df.drop(['close'], axis=1).dropna()
        
        # Align features and target
        features = features[:-1]  # Remove last row since we don't have next period return
        target = target[:-1]
        
        # Split data using time series split
        ts_split = TimeSeriesSplit(
            n_splits=5,
            test_size=int(len(features) * self.config['validation_split'])
        )
        
        # Get the last split for final training
        for train_idx, val_idx in ts_split.split(features):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        return X_train_scaled, X_val_scaled, y_train.values, y_val.values

    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Train for one epoch.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Training loss for the epoch
        """
        return self.model.train_epoch(X_train, y_train, self.config['batch_size'])

    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Dictionary of validation metrics
        """
        predictions = self.model.predict(X_val)
        
        return {
            'val_loss': mean_squared_error(y_val, predictions, squared=False),
            'val_mae': mean_absolute_error(y_val, predictions),
            'val_r2': r2_score(y_val, predictions)
        }

    def check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if training should stop early.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            return False
        
        self.early_stopping_counter += 1
        return self.early_stopping_counter >= self.config['early_stopping_patience']

    def save_checkpoint(self) -> None:
        """Save model checkpoint and training state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.models_dir / f"model_{timestamp}.pkl"
        self.model.save(model_path)
        
        # Save training history
        history_path = self.models_dir / f"history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Saved checkpoint: {model_path}")

    def train(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Returns:
            Dictionary containing training results and metrics
        """
        logger.info("Starting training pipeline...")
        
        try:
            # Prepare data
            X_train, X_val, y_train, y_val = self.prepare_data()
            
            # Training loop
            for epoch in range(self.config['epochs']):
                self.current_epoch = epoch
                
                # Train one epoch
                train_loss = self.train_epoch(X_train, y_train)
                
                # Validate
                val_metrics = self.validate(X_val, y_val)
                
                # Log metrics
                metrics = {'train_loss': train_loss, **val_metrics}
                self.training_history.append({
                    'epoch': epoch,
                    **metrics
                })
                
                logger.info(
                    f"Epoch {epoch}/{self.config['epochs']} - "
                    f"train_loss: {train_loss:.4f} - "
                    f"val_loss: {val_metrics['val_loss']:.4f}"
                )
                
                # Check early stopping
                if self.check_early_stopping(val_metrics['val_loss']):
                    logger.info("Early stopping triggered")
                    break
                
                # Save checkpoint
                if epoch % 5 == 0:
                    self.save_checkpoint()
            
            # Final evaluation
            final_metrics = self.validate(X_val, y_val)
            
            # Save final model
            self.save_checkpoint()
            
            return {
                'status': 'success',
                'epochs_completed': self.current_epoch + 1,
                'final_metrics': final_metrics,
                'training_history': self.training_history
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'epochs_completed': self.current_epoch
            }

    def cleanup(self) -> None:
        """Cleanup resources after training."""
        # Implement cleanup logic if needed
        pass