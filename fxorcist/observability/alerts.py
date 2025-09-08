import smtplib
from email.message import EmailMessage
import logging
import json
import os
from typing import Optional, Dict, Any, Union, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class AlertConfig:
    """Configuration for alerting system."""
    def __init__(
        self,
        smtp_server: Optional[str] = None,
        email_from: Optional[str] = None,
        email_to: Optional[str] = None,
        log_dir: Optional[str] = None,
        webhook_url: Optional[str] = None
    ):
        """
        Initialize alert configuration.
        
        :param smtp_server: SMTP server address
        :param email_from: Sender email address
        :param email_to: Recipient email address
        :param log_dir: Directory to store governance logs
        :param webhook_url: Optional webhook URL for additional notifications
        """
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER')
        self.email_from = email_from or os.getenv('EMAIL_FROM')
        self.email_to = email_to or os.getenv('EMAIL_TO')
        self.log_dir = log_dir or os.path.join(os.getcwd(), 'governance_logs')
        self.webhook_url = webhook_url or os.getenv('ALERT_WEBHOOK_URL')
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

class AlertManager:
    """Advanced alerting and governance system."""
    
    def __init__(
        self, 
        config: Optional[AlertConfig] = None,
        logging_level: int = logging.INFO
    ):
        """
        Initialize AlertManager.
        
        :param config: Alert configuration
        :param logging_level: Logging level for the alert manager
        """
        logger.setLevel(logging_level)
        
        # Use default config if not provided
        self.config = config or AlertConfig()
        
        # Setup additional notification methods
        self._notification_hooks: Dict[str, Callable] = {}

    def send_email_alert(
        self, 
        subject: str, 
        body: str, 
        attachments: Optional[List[str]] = None
    ):
        """
        Send email alert with optional attachments.
        
        :param subject: Email subject
        :param body: Email body text
        :param attachments: Optional list of file paths to attach
        """
        if not all([self.config.smtp_server, self.config.email_from, self.config.email_to]):
            logger.warning("SMTP configuration incomplete. Skipping email alert.")
            return

        try:
            msg = EmailMessage()
            msg.set_content(body)
            msg["Subject"] = subject
            msg["From"] = self.config.email_from
            msg["To"] = self.config.email_to

            # Add attachments
            if attachments:
                for filepath in attachments:
                    with open(filepath, 'rb') as f:
                        msg.add_attachment(
                            f.read(), 
                            maintype='application', 
                            subtype='octet-stream', 
                            filename=os.path.basename(filepath)
                        )

            with smtplib.SMTP(self.config.smtp_server) as server:
                server.send_message(msg)
            
            logger.info(f"Alert sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def register_webhook(
        self, 
        name: str, 
        webhook_func: Callable[[Dict[str, Any]], None]
    ):
        """
        Register a custom webhook for notifications.
        
        :param name: Name of the webhook
        :param webhook_func: Function to call for webhook notification
        """
        self._notification_hooks[name] = webhook_func

    def log_governance_event(
        self, 
        event_type: str, 
        details: Dict[str, Any], 
        log_level: str = "INFO"
    ):
        """
        Log comprehensive governance event with multiple notification channels.
        
        :param event_type: Type of governance event
        :param details: Event details dictionary
        :param log_level: Logging level
        """
        # Prepare event data
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details,
            "log_level": log_level
        }

        # Log to file
        log_filename = os.path.join(
            self.config.log_dir, 
            f"governance_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        
        try:
            with open(log_filename, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to write governance log: {e}")

        # Log to standard logging
        log_func = {
            "DEBUG": logger.debug,
            "INFO": logger.info,
            "WARNING": logger.warning,
            "ERROR": logger.error,
            "CRITICAL": logger.critical
        }.get(log_level.upper(), logger.info)

        log_func(f"Governance Event [{event_type}]: {details}")

        # Send email if critical
        if log_level.upper() in ["ERROR", "CRITICAL"]:
            self.send_email_alert(
                subject=f"Governance Alert: {event_type}",
                body=json.dumps(event_data, indent=2)
            )

        # Call registered webhooks
        for name, webhook_func in self._notification_hooks.items():
            try:
                webhook_func(event_data)
            except Exception as e:
                logger.error(f"Webhook {name} failed: {e}")

    def register_threshold_alert(
        self, 
        metric_name: str, 
        threshold: float, 
        condition: str = "above",
        severity: str = "warning"
    ):
        """
        Register a threshold-based alert with comprehensive logging.
        
        :param metric_name: Name of the metric to monitor
        :param threshold: Threshold value
        :param condition: 'above' or 'below'
        :param severity: 'warning' or 'critical'
        """
        event_details = {
            "metric": metric_name,
            "threshold": threshold,
            "condition": condition,
            "severity": severity
        }
        
        self.log_governance_event(
            event_type="threshold_alert_registered", 
            details=event_details,
            log_level="INFO"
        )