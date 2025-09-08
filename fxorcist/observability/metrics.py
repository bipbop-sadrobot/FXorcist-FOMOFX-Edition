from prometheus_client import (
    Counter, 
    Histogram, 
    Gauge, 
    start_http_server, 
    REGISTRY, 
    CollectorRegistry
)
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class PrometheusMetricsConfig:
    """Enhanced configuration for Prometheus metrics."""
    def __init__(
        self, 
        port: int = 9090, 
        namespace: str = 'fxorcist',
        subsystem: str = 'backtest',
        financial_metrics: bool = True
    ):
        """
        Initialize Prometheus metrics configuration.
        
        :param port: Port to expose metrics on
        :param namespace: Metrics namespace
        :param subsystem: Metrics subsystem
        :param financial_metrics: Enable additional financial-specific metrics
        """
        self.port = port
        self.namespace = namespace
        self.subsystem = subsystem
        self.financial_metrics = financial_metrics

class PrometheusMetrics:
    """Advanced Prometheus metrics tracking with financial insights."""
    
    def __init__(
        self, 
        config: Optional[PrometheusMetricsConfig] = None,
        logging_level: int = logging.INFO
    ):
        """
        Initialize Prometheus metrics tracking.
        
        :param config: Prometheus metrics configuration
        :param logging_level: Logging level for the metrics
        """
        logger.setLevel(logging_level)
        
        # Use default config if not provided
        self.config = config or PrometheusMetricsConfig()
        
        # Create a custom registry to allow multiple metric collectors
        self.registry = CollectorRegistry()
        
        # Define metrics with namespace and subsystem
        self._setup_metrics()

    def _setup_metrics(self):
        """Set up Prometheus metrics with namespace and subsystem."""
        namespace, subsystem = self.config.namespace, self.config.subsystem
        
        # Standard metrics
        self.BACKTEST_COUNT = Counter(
            'backtest_total', 
            'Total backtests run', 
            ['strategy'], 
            namespace=namespace, 
            subsystem=subsystem,
            registry=self.registry
        )
        
        self.BACKTEST_DURATION = Histogram(
            'backtest_duration_seconds', 
            'Backtest duration', 
            ['strategy'], 
            namespace=namespace, 
            subsystem=subsystem,
            registry=self.registry
        )
        
        self.BACKTEST_ERROR_COUNT = Counter(
            'backtest_errors_total', 
            'Backtest errors', 
            ['strategy', 'error_type'], 
            namespace=namespace, 
            subsystem=subsystem,
            registry=self.registry
        )
        
        self.CURRENT_ACTIVE_BACKTESTS = Gauge(
            'active_backtests', 
            'Currently running backtests', 
            ['strategy'], 
            namespace=namespace, 
            subsystem=subsystem,
            registry=self.registry
        )
        
        # Financial-specific metrics (if enabled)
        if self.config.financial_metrics:
            self.STRATEGY_SHARPE_RATIO = Gauge(
                'strategy_sharpe_ratio', 
                'Sharpe ratio for trading strategies', 
                ['strategy'], 
                namespace=namespace, 
                subsystem=subsystem,
                registry=self.registry
            )
            
            self.STRATEGY_MAX_DRAWDOWN = Gauge(
                'strategy_max_drawdown', 
                'Maximum drawdown for trading strategies', 
                ['strategy'], 
                namespace=namespace, 
                subsystem=subsystem,
                registry=self.registry
            )
            
            self.STRATEGY_WIN_RATE = Gauge(
                'strategy_win_rate', 
                'Win rate for trading strategies', 
                ['strategy'], 
                namespace=namespace, 
                subsystem=subsystem,
                registry=self.registry
            )

    def observe_backtest(
        self, 
        strategy: str, 
        duration: float, 
        success: bool = True, 
        error_type: Optional[str] = None,
        financial_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Record metrics for a backtest trial.
        
        :param strategy: Name of the trading strategy
        :param duration: Time taken for the backtest
        :param success: Whether the backtest completed successfully
        :param error_type: Type of error if the backtest failed
        :param financial_metrics: Optional financial performance metrics
        """
        try:
            self.BACKTEST_COUNT.labels(strategy=strategy).inc()
            self.BACKTEST_DURATION.labels(strategy=strategy).observe(duration)
            
            if not success:
                self.BACKTEST_ERROR_COUNT.labels(
                    strategy=strategy, 
                    error_type=error_type or 'unknown'
                ).inc()
            
            # Log financial metrics if provided and enabled
            if self.config.financial_metrics and financial_metrics:
                if 'sharpe_ratio' in financial_metrics:
                    self.STRATEGY_SHARPE_RATIO.labels(strategy=strategy).set(
                        financial_metrics['sharpe_ratio']
                    )
                
                if 'max_drawdown' in financial_metrics:
                    self.STRATEGY_MAX_DRAWDOWN.labels(strategy=strategy).set(
                        financial_metrics['max_drawdown']
                    )
                
                if 'win_rate' in financial_metrics:
                    self.STRATEGY_WIN_RATE.labels(strategy=strategy).set(
                        financial_metrics['win_rate']
                    )
        except Exception as e:
            logger.error(f"Failed to record backtest metrics: {e}")

    def track_active_backtests(
        self, 
        strategy: str, 
        active: bool = True
    ):
        """
        Track number of active backtests for a strategy.
        
        :param strategy: Name of the trading strategy
        :param active: Whether to increment or decrement active backtests
        """
        try:
            if active:
                self.CURRENT_ACTIVE_BACKTESTS.labels(strategy=strategy).inc()
            else:
                self.CURRENT_ACTIVE_BACKTESTS.labels(strategy=strategy).dec()
        except Exception as e:
            logger.error(f"Failed to track active backtests: {e}")

    def start_server(self):
        """
        Start Prometheus metrics server.
        """
        try:
            # Register the custom registry
            REGISTRY.register(self.registry)
            
            start_http_server(
                self.config.port, 
                registry=self.registry
            )
            logger.info(f"Prometheus metrics server running on port {self.config.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {e}")