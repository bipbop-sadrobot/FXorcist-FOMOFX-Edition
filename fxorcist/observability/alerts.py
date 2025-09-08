import smtplib
from email.message import EmailMessage
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(
        self, 
        smtp_server: Optional[str] = None, 
        email_from: Optional[str] = None, 
        email_to: Optional[str] = None
    ):
        """
        Initialize AlertManager for sending notifications.
        
        :param smtp_server: SMTP server address
        :param email_from: Sender email address
        :param email_to: Recipient email address
        """
        self.smtp_server = smtp_server
        self.email_from = email_from
        self.email_to = email_to

    def send_alert(self, subject: str, body: str):
        """
        Send email alert.
        
        :param subject: Email subject
        :param body: Email body text
        """
        if not all([self.smtp_server, self.email_from, self.email_to]):
            logger.warning("SMTP configuration incomplete. Skipping email alert.")
            return

        try:
            msg = EmailMessage()
            msg.set_content(body)
            msg["Subject"] = subject
            msg["From"] = self.email_from
            msg["To"] = self.email_to

            with smtplib.SMTP(self.smtp_server) as server:
                server.send_message(msg)
            
            logger.info(f"Alert sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def register_threshold_alert(
        self, 
        metric_name: str, 
        threshold: float, 
        condition: str = "above",
        severity: str = "warning"
    ):
        """
        Register a threshold-based alert (mock implementation).
        
        :param metric_name: Name of the metric to monitor
        :param threshold: Threshold value
        :param condition: 'above' or 'below'
        :param severity: 'warning' or 'critical'
        """
        logger.info(
            f"Alert registered: {metric_name} {condition} {threshold} "
            f"(Severity: {severity})"
        )

    def log_governance_event(
        self, 
        event_type: str, 
        details: dict, 
        log_level: str = "INFO"
    ):
        """
        Log governance-related events.
        
        :param event_type: Type of governance event
        :param details: Event details dictionary
        :param log_level: Logging level
        """
        log_func = {
            "INFO": logger.info,
            "WARNING": logger.warning,
            "ERROR": logger.error,
            "CRITICAL": logger.critical
        }.get(log_level.upper(), logger.info)

        log_func(f"Governance Event [{event_type}]: {details}")