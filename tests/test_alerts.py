import pytest
import os
import tempfile
from fxorcist.observability.alerts import AlertManager, AlertConfig

@pytest.fixture
def alert_manager():
    """
    Fixture to create an AlertManager with a temporary log directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = AlertConfig(
            log_dir=tmpdir,
            smtp_server=None,  # Disable email for testing
            email_from=None,
            email_to=None
        )
        manager = AlertManager(config=config)
        yield manager

def test_alert_manager_initialization(alert_manager):
    """
    Test AlertManager initialization.
    """
    assert alert_manager.config is not None
    assert os.path.exists(alert_manager.config.log_dir)

def test_log_governance_event(alert_manager):
    """
    Test logging a governance event.
    """
    event_details = {
        "strategy": "test_strategy",
        "performance": {
            "sharpe": 1.5,
            "return": 0.2
        }
    }
    
    alert_manager.log_governance_event(
        event_type="performance_review",
        details=event_details,
        log_level="INFO"
    )
    
    # Check if log file was created
    log_files = os.listdir(alert_manager.config.log_dir)
    assert any("governance" in filename for filename in log_files)

def test_register_threshold_alert(alert_manager):
    """
    Test registering a threshold alert.
    """
    alert_manager.register_threshold_alert(
        metric_name="sharpe_ratio",
        threshold=1.0,
        condition="above",
        severity="warning"
    )
    
    # Check if log file was created
    log_files = os.listdir(alert_manager.config.log_dir)
    assert any("governance" in filename for filename in log_files)

def test_webhook_registration(alert_manager):
    """
    Test registering and calling a webhook.
    """
    # Mock webhook function
    webhook_calls = []
    def test_webhook(event_data):
        webhook_calls.append(event_data)
    
    alert_manager.register_webhook("test_webhook", test_webhook)
    
    event_details = {
        "type": "test_event",
        "data": {"key": "value"}
    }
    
    alert_manager.log_governance_event(
        event_type="webhook_test",
        details=event_details,
        log_level="INFO"
    )
    
    assert len(webhook_calls) > 0
    assert webhook_calls[0]["type"] == "webhook_test"

def test_email_alert_configuration():
    """
    Test email alert configuration.
    """
    config = AlertConfig(
        smtp_server="smtp.test.com",
        email_from="sender@test.com",
        email_to="recipient@test.com"
    )
    
    manager = AlertManager(config=config)
    
    assert manager.config.smtp_server == "smtp.test.com"
    assert manager.config.email_from == "sender@test.com"
    assert manager.config.email_to == "recipient@test.com"

def test_governance_event_log_levels(alert_manager):
    """
    Test logging events with different log levels.
    """
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    for level in log_levels:
        alert_manager.log_governance_event(
            event_type=f"test_{level.lower()}",
            details={"level": level},
            log_level=level
        )
    
    # Check if log files were created
    log_files = os.listdir(alert_manager.config.log_dir)
    assert any("governance" in filename for filename in log_files)