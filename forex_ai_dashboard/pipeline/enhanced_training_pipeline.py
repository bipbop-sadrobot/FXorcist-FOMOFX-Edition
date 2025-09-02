#!/usr/bin/env python3
"""
Enhanced Training Pipeline with Memory System Integration
Combines traditional ML training with memory system, federated learning,
and anomaly detection for improved performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime
import sys
import os
from typing import Tuple, Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from forex_ai_dashboard.models.catboost_model import CatBoostModel
from memory_system.memory import IntegratedMemorySystem
from memory_system.anomaly import AnomalyDetector
from memory_system.federated import FederatedMemory
from memory_system.event_bus import EventBus
from memory_system.metadata import SharedMetadata

# Import intuition-driven analyzers
from .intuition_driven_analyzer import IntuitionDrivenAnalyzer
from .temporal_correlation_analyzer import TemporalCorrelationAnalyzer
from .cross_pair_interaction_analyzer import CrossPairInteractionAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/enhanced_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedTrainingPipeline:
    """Enhanced training pipeline with memory system integration."""

    def __init__(self):
        # Initialize memory system components
        self.event_bus = EventBus()
        self.metadata = SharedMetadata()
        self.memory = IntegratedMemorySystem(self.event_bus, self.metadata)
        self.anomaly_detector = AnomalyDetector(self.memory)
        self.federated_memory = FederatedMemory(self.event_bus, self.metadata)

        # Initialize intuition-driven analyzers
        self.intuition_analyzer = IntuitionDrivenAnalyzer()
        self.temporal_analyzer = TemporalCorrelationAnalyzer()
        self.cross_pair_analyzer = CrossPairInteractionAnalyzer()

        # Training state
        self.model = None
        self.training_history = []
        self.anomaly_history = []
        self.intuition_insights = {}

    def load_training_data(self, data_path: str = "data/processed/eurusd_features_2024.parquet") -> pd.DataFrame:
        """Load and validate training data."""
        logger.info(f"Loading training data from {data_path}")

        if not Path(data_path).exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")

        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # Basic validation
        if len(df) == 0:
            raise ValueError("No data found in training file")

        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    def prepare_features_and_target(self, df: pd.DataFrame, target_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series, list]:
        """Prepare features and target for training with memory integration."""
        logger.info(f"Preparing features and target with {target_horizon}-step ahead prediction")

        # Create target: future price movement
        df = df.copy()
        df['target'] = df['close'].shift(-target_horizon)

        # Remove rows with NaN target
        df = df.dropna(subset=['target'])

        # Feature columns (exclude target-related columns)
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol', 'target']]

        # Add memory-based features if available
        memory_features = self._extract_memory_features(df)
        if memory_features:
            for feature_name, values in memory_features.items():
                df[feature_name] = values
                feature_cols.append(feature_name)

        X = df[feature_cols]
        y = df['target']

        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        logger.info(f"Feature columns: {feature_cols[:10]}...")

        return X, y, feature_cols

    def _extract_memory_features(self, df: pd.DataFrame) -> Dict[str, list]:
        """Extract memory-based features from historical data."""
        memory_features = {}

        try:
            # Get recent memory entries
            recent_memory = self.memory.recall(top_k=100)

            if recent_memory:
                # Calculate memory-based indicators
                memory_errors = [entry.get('error', 0) for entry in recent_memory if isinstance(entry, dict)]
                memory_predictions = [entry.get('prediction', 0) for entry in recent_memory if isinstance(entry, dict)]

                if memory_errors:
                    # Memory error trend
                    memory_features['memory_error_trend'] = [np.mean(memory_errors[-10:])] * len(df)

                    # Memory confidence score
                    memory_features['memory_confidence'] = [1.0 / (1.0 + np.std(memory_errors[-20:]))] * len(df)

                if memory_predictions:
                    # Memory prediction volatility
                    memory_features['memory_prediction_volatility'] = [np.std(memory_predictions[-20:])] * len(df)

        except Exception as e:
            logger.warning(f"Failed to extract memory features: {e}")

        return memory_features

    def detect_anomalies_in_training_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Detect and handle anomalies in training data."""
        logger.info("Detecting anomalies in training data")

        try:
            # Convert to memory format for anomaly detection
            for i, (_, row) in enumerate(X.iterrows()):
                memory_entry = {
                    "model": "training_data",
                    "prediction": y.iloc[i],
                    "target": y.iloc[i],
                    "error": 0.0,
                    "features": row.to_dict(),
                    "ts": datetime.now().timestamp()
                }
                self.memory.add_record(memory_entry)

            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies()

            if anomalies.get('anomalies'):
                logger.info(f"Detected {len(anomalies['anomalies'])} anomalies in training data")

                # Remove anomalous samples
                anomaly_indices = [anom['index'] for anom in anomalies['anomalies']]
                X_clean = X.drop(X.index[anomaly_indices])
                y_clean = y.drop(y.index[anomaly_indices])

                logger.info(f"Removed {len(anomaly_indices)} anomalous samples")
                self.anomaly_history.append(anomalies)

                return X_clean, y_clean

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")

        return X, y

    def train_with_memory_integration(self, X_train, X_test, y_train, y_test, model_params: dict = None) -> dict:
        """Train model with memory system integration."""
        logger.info("Starting enhanced training with memory integration")

        # Default parameters
        default_params = {
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'loss_function': 'RMSE',
            'random_seed': 42,
            'early_stopping_rounds': 50
        }

        if model_params:
            default_params.update(model_params)

        # Initialize model
        self.model = CatBoostModel(**default_params)

        logger.info("Starting model training")
        start_time = datetime.now()

        # Train model
        self.model.train(X_train, y_train, X_test, y_test)

        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(".2f")

        # Evaluate on test set
        logger.info("Evaluating model on test set")
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'training_time': training_time
        }

        logger.info("Model evaluation metrics:")
        logger.info(".6f")
        logger.info(".6f")
        logger.info(".6f")
        logger.info(".6f")

        # Store training results in memory
        self._store_training_results_in_memory(metrics, y_test, y_pred)

        return metrics

    def _store_training_results_in_memory(self, metrics: dict, y_true: pd.Series, y_pred: np.ndarray):
        """Store training results in memory system."""
        try:
            # Store overall metrics
            metrics_entry = {
                "model": "enhanced_training",
                "prediction": metrics['r2'],  # Use R² as prediction score
                "target": 1.0,  # Target is perfect prediction
                "error": 1.0 - metrics['r2'],  # Error is 1 - R²
                "features": {
                    "rmse": metrics['rmse'],
                    "mae": metrics['mae'],
                    "training_time": metrics['training_time']
                },
                "ts": datetime.now().timestamp()
            }
            self.memory.add_record(metrics_entry)

            # Store individual predictions
            for i, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
                prediction_entry = {
                    "model": "enhanced_training",
                    "prediction": float(pred_val),
                    "target": float(true_val),
                    "error": float(abs(true_val - pred_val)),
                    "features": {"sample_index": i},
                    "ts": datetime.now().timestamp()
                }
                self.memory.add_record(prediction_entry)

            logger.info("Training results stored in memory system")

        except Exception as e:
            logger.error(f"Failed to store training results in memory: {e}")

    def perform_federated_training_round(self):
        """Perform federated learning training round."""
        try:
            logger.info("Performing federated training round")
            result = self.federated_memory.train_round()

            if result:
                logger.info(f"Federated round completed: {result}")
                return result
            else:
                logger.warning("Federated training round returned no results")
                return None

        except Exception as e:
            logger.error(f"Federated training failed: {e}")
            return None

    def get_memory_insights(self) -> dict:
        """Get insights from memory system."""
        try:
            insights = self.memory.generate_insights_report()
            return insights
        except Exception as e:
            logger.error(f"Failed to get memory insights: {e}")
            return {}

    def perform_intuition_driven_analysis(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform comprehensive intuition-driven analysis."""
        logger.info("Performing intuition-driven analysis")

        try:
            # Run intuition-driven analysis
            intuition_results = self.intuition_analyzer.analyze_with_intuition(data_dict)

            # Store insights for later use
            self.intuition_insights = intuition_results

            # Extract key recommendations
            recommendations = {}
            for pair in data_dict.keys():
                if pair in intuition_results.get('recency_weighted_insights', {}):
                    recs = self.intuition_analyzer.get_intuition_recommendations(
                        pair, intuition_results['recency_weighted_insights'][pair]
                    )
                    recommendations[pair] = recs

            logger.info(f"Generated recommendations for {len(recommendations)} pairs")
            return {
                'intuition_results': intuition_results,
                'recommendations': recommendations
            }

        except Exception as e:
            logger.error(f"Intuition-driven analysis failed: {e}")
            return {}

    def save_model_and_results(self, metrics: dict, model_path: str = "models/trained/enhanced_model"):
        """Save trained model and results."""
        logger.info(f"Saving enhanced model to {model_path}")

        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        # Save model
        if self.model:
            self.model.save(f"{model_path}.cbm")

        # Save metrics and memory insights
        import json
        results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'memory_insights': self.get_memory_insights(),
            'anomaly_history': self.anomaly_history,
            'model_params': self.model.params if self.model else {}
        }

        with open(f"{model_path}_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Enhanced model and results saved successfully")

def main():
    """Main enhanced training function."""
    try:
        logger.info("Starting Enhanced Training Pipeline")

        # Initialize enhanced pipeline
        pipeline = EnhancedTrainingPipeline()

        # Load training data
        data_path = "data/processed/eurusd_features_2024.parquet"
        df = pipeline.load_training_data(data_path)

        # Prepare features and target
        X, y, feature_cols = pipeline.prepare_features_and_target(df, target_horizon=1)

        # Detect and handle anomalies
        X_clean, y_clean = pipeline.detect_anomalies_in_training_data(X, y)

        # Split data
        test_size = 0.2
        val_size = 0.1
        train_size = 1 - test_size - val_size

        logger.info(".1f")

        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_clean, y_clean, test_size=test_size, shuffle=False
        )

        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(train_size + val_size), shuffle=False
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Train model with memory integration
        model_params = {
            'iterations': 300,
            'learning_rate': 0.1,
            'depth': 4
        }

        metrics = pipeline.train_with_memory_integration(
            X_train, X_test, y_train, y_test, model_params
        )

        # Perform federated training round
        federated_result = pipeline.perform_federated_training_round()

        # Save model and results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/trained/enhanced_forex_{timestamp}"
        pipeline.save_model_and_results(metrics, model_path)

        # Print final summary
        print("\n" + "="*60)
        print("ENHANCED FOREX AI TRAINING SUMMARY")
        print("="*60)
        print(f"Data source: {data_path}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features used: {len(feature_cols)}")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")
        print(f"Memory records: {len(pipeline.memory.records)}")
        print(f"Anomalies detected: {len(pipeline.anomaly_history)}")
        print(f"Federated rounds: {'Completed' if federated_result else 'Not performed'}")
        print(f"Model saved as: {model_path}.cbm")
        print("="*60)

        logger.info("Enhanced training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Enhanced training failed: {str(e)}", exc_info=True)
        print(f"\n❌ Enhanced training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()