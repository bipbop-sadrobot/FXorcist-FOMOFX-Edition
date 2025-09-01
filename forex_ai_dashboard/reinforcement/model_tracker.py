from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from forex_ai_dashboard.utils.logger import logger
from forex_ai_dashboard.reinforcement.memory import EventBus, SharedMetadata, IntegratedMemorySystem, PredictionRecord
import threading
import queue

class ModelTracker:
    """Tracks model predictions and performance using IntegratedMemorySystem"""
   
    def __init__(self):
        self.event_bus = EventBus()
        self.metadata = SharedMetadata()
        self.memory = IntegratedMemorySystem(self.event_bus, self.metadata)
        self.current_model_version = "unknown"
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._process_log_queue, daemon=True)
        self.log_thread.start()

    def _process_log_queue(self):
        """
        Processes log messages from the queue asynchronously.
        """
        while True:
            try:
                log_message = self.log_queue.get()
                logger.info(log_message)
                self.log_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing log queue: {e}")

    def set_model_version(self, version: str):
        """
        Set the current model version being tracked.
        """
        self.current_model_version = version
        self.log_queue.put(f"Tracking model version: {version}")

    def record_prediction(self,
                        features: Dict[str, float],
                        prediction: float,
                        timestamp: datetime = None) -> str:
        """
        Record a new prediction in the IntegratedMemorySystem.
        """
        if timestamp is None:
            timestamp = datetime.now()
           
        record = PredictionRecord(
            timestamp=timestamp,
            model_version=self.current_model_version,
            features=features,
            prediction=prediction,
            actual=None,
            error_metrics=None,
            feature_importance=None
        )

        # Add the prediction data to the memory
        self.memory.add_record(record)

        # Log the prediction
        self.log_queue.put(f"Recorded prediction for model {self.current_model_version}")
        return f"Recorded prediction for model {self.current_model_version}"

    def record_outcome(self,
                      timestamp: datetime,
                      actual: float,
                      error_metrics: Dict[str, float] = None,
                      feature_importance: Dict[str, float] = None):
        """
        Update the latest prediction record with actual outcome and metrics.
        """
        # Create a PredictionRecord for the outcome
        record = PredictionRecord(
            timestamp=timestamp,
            model_version=self.current_model_version,
            features={},  # Features are already in the prediction record
            prediction=0.0,  # Prediction is not relevant for outcome
            actual=actual,
            error_metrics=error_metrics,
            feature_importance=feature_importance
        )

        self.memory.add_record(record)

        # Log the outcome
        self.log_queue.put(f"Updated record with outcome for model {self.current_model_version}")
       
    def get_performance_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get performance metrics for the current model version"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
       
        # TODO: Implement get_performance_metrics using IntegratedMemorySystem
        metrics = {
            "model_version": self.current_model_version,
            "time_period": f"{days} days",
            "prediction_count": 0,
            "average_error": None,
            "error_metrics": {}
        }
       
        return metrics

    def check_for_drift(self, threshold: float = 0.1, span: int = 5) -> bool:
        """Check if model performance has drifted beyond threshold using EWMA."""
        if len(self.memory.df_cache) < 2 or 'error_RMSE' not in self.memory.df_cache.columns:
            return False

        # Calculate EWMA for RMSE
        self.memory.df_cache['ewma_error'] = self.memory.df_cache['error_RMSE'].ewm(span=span).mean()

        # Check if the latest EWMA error exceeds the threshold
        if self.memory.df_cache['ewma_error'].iloc[-1] > threshold:
            logger.warning(f"Model drift detected - EWMA error {self.memory.df_cache['ewma_error'].iloc[-1]} exceeds threshold {threshold}")
            return True
        return False

    def auto_retrain_pipeline(self):
        """Initiate retraining pipeline based on memory insights"""
        from forex_ai_dashboard.pipeline import model_training
       
        insights = self.memory.generate_insights_report()
        resource_allocation = self.memory.prioritize_resources()
       
        logger.info(f"Initiating auto-retrain with insights: {insights}")
        logger.info(f"Resource allocation: {resource_allocation}")
       
        # Dynamic training configuration
        training_config = {
            'optimization_target': 'error',
            'compute_budget': int(resource_allocation.get('cpu', 30))
        }
           
        model_training.train_new_version(**training_config)
       
        # Update production model after validation
        if model_training.validate_new_model():
            self.set_model_version(training_config['model_architecture'])
