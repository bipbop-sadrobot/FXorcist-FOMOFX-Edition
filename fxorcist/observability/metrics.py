from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import logging

logger = logging.getLogger(__name__)

# Metrics definitions
BACKTEST_COUNT = Counter('fxorcist_backtest_total', 'Total backtests run', ['strategy'])
BACKTEST_DURATION = Histogram('fxorcist_backtest_duration_seconds', 'Backtest duration', ['strategy'])
BACKTEST_ERROR_COUNT = Counter('fxorcist_backtest_errors_total', 'Backtest errors', ['strategy'])
CURRENT_ACTIVE_BACKTESTS = Gauge('fxorcist_active_backtests', 'Currently running backtests')

class PrometheusMetrics:
    @staticmethod
    def observe_backtest(strategy: str, duration: float, success: bool = True):
        """
        Record metrics for a backtest trial.
        
        :param strategy: Name of the trading strategy
        :param duration: Time taken for the backtest
        :param success: Whether the backtest completed successfully
        """
        BACKTEST_COUNT.labels(strategy=strategy).inc()
        BACKTEST_DURATION.labels(strategy=strategy).observe(duration)
        if not success:
            BACKTEST_ERROR_COUNT.labels(strategy=strategy).inc()

    @staticmethod
    def start_server(port: int = 9090):
        """
        Start Prometheus metrics server.
        
        :param port: Port to expose metrics on
        """
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server running on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {e}")

    @staticmethod
    def track_active_backtests(active: bool = True):
        """
        Track number of active backtests.
        
        :param active: Whether to increment or decrement active backtests
        """
        if active:
            CURRENT_ACTIVE_BACKTESTS.inc()
        else:
            CURRENT_ACTIVE_BACKTESTS.dec()