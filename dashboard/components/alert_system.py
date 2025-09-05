"""
Real-time alert system for the Forex AI dashboard.
Provides customizable alerts, notifications, and automated monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import json
from pathlib import Path
import logging
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertRule:
    """Represents a single alert rule."""

    def __init__(self,
                 name: str,
                 condition: str,
                 threshold: float,
                 symbol: Optional[str] = None,
                 timeframe: str = "1H",
                 enabled: bool = True,
                 notification_methods: List[str] = None,
                 cooldown_minutes: int = 60):
        self.name = name
        self.condition = condition  # e.g., "price > threshold", "volatility > threshold"
        self.threshold = threshold
        self.symbol = symbol
        self.timeframe = timeframe
        self.enabled = enabled
        self.notification_methods = notification_methods or ["dashboard"]
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered = None

    def check_condition(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Check if alert condition is met."""
        if not self.enabled:
            return None

        # Check cooldown
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(minutes=self.cooldown_minutes)
            if datetime.now() < cooldown_end:
                return None

        try:
            # Parse condition and check
            if self.condition == "price_above":
                current_price = data['close'].iloc[-1]
                if current_price > self.threshold:
                    return {
                        'type': 'price_alert',
                        'message': f"{self.symbol} price above {self.threshold:.4f} (Current: {current_price:.4f})",
                        'severity': 'warning',
                        'data': {'price': current_price, 'threshold': self.threshold}
                    }

            elif self.condition == "price_below":
                current_price = data['close'].iloc[-1]
                if current_price < self.threshold:
                    return {
                        'type': 'price_alert',
                        'message': f"{self.symbol} price below {self.threshold:.4f} (Current: {current_price:.4f})",
                        'severity': 'warning',
                        'data': {'price': current_price, 'threshold': self.threshold}
                    }

            elif self.condition == "volatility_spike":
                volatility = data['returns'].rolling(20, min_periods=1).std().iloc[-1]
                if volatility > self.threshold:
                    return {
                        'type': 'volatility_alert',
                        'message': f"High volatility in {self.symbol}: {volatility:.4f}",
                        'severity': 'warning',
                        'data': {'volatility': volatility, 'threshold': self.threshold}
                    }

            elif self.condition == "prediction_confidence":
                # This would be checked against model predictions
                confidence = np.random.uniform(0.5, 0.95)  # Placeholder
                if confidence > self.threshold:
                    return {
                        'type': 'prediction_alert',
                        'message': f"High confidence prediction for {self.symbol}: {confidence:.1f}%",
                        'severity': 'info',
                        'data': {'confidence': confidence, 'threshold': self.threshold}
                    }

            elif self.condition == "model_performance":
                # Check model performance metrics
                performance = np.random.uniform(0.8, 0.99)  # Placeholder
                if performance < self.threshold:
                    return {
                        'type': 'performance_alert',
                        'message': f"Model performance degraded: {performance:.3f}",
                        'severity': 'error',
                        'data': {'performance': performance, 'threshold': self.threshold}
                    }

        except Exception as e:
            logger.error(f"Error checking alert condition {self.name}: {str(e)}")

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert rule to dictionary."""
        return {
            'name': self.name,
            'condition': self.condition,
            'threshold': self.threshold,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'enabled': self.enabled,
            'notification_methods': self.notification_methods,
            'cooldown_minutes': self.cooldown_minutes,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertRule':
        """Create alert rule from dictionary."""
        rule = cls(
            name=data['name'],
            condition=data['condition'],
            threshold=data['threshold'],
            symbol=data.get('symbol'),
            timeframe=data.get('timeframe', '1H'),
            enabled=data.get('enabled', True),
            notification_methods=data.get('notification_methods', ['dashboard']),
            cooldown_minutes=data.get('cooldown_minutes', 60)
        )
        if data.get('last_triggered'):
            rule.last_triggered = datetime.fromisoformat(data['last_triggered'])
        return rule

class NotificationManager:
    """Manages different notification methods."""

    def __init__(self):
        self.email_config = self._load_email_config()

    def _load_email_config(self) -> Dict[str, Any]:
        """Load email configuration."""
        config_file = Path("config/email_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'enabled': False,
            'smtp_server': '',
            'smtp_port': 587,
            'username': '',
            'password': '',
            'from_email': '',
            'to_emails': []
        }

    def send_dashboard_notification(self, alert: Dict[str, Any]):
        """Send notification to dashboard."""
        # This will be handled by the dashboard display
        pass

    def send_email_notification(self, alert: Dict[str, Any]):
        """Send email notification."""
        if not self.email_config.get('enabled', False):
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(self.email_config['to_emails'])
            msg['Subject'] = f"Forex AI Alert: {alert['type'].replace('_', ' ').title()}"

            body = f"""
            Forex AI Alert Notification

            Type: {alert['type']}
            Message: {alert['message']}
            Severity: {alert['severity']}
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            Details: {json.dumps(alert.get('data', {}), indent=2)}
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], self.email_config['to_emails'], text)
            server.quit()

            logger.info(f"Email alert sent: {alert['message']}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")

    def send_webhook_notification(self, alert: Dict[str, Any], webhook_url: str):
        """Send webhook notification."""
        # Placeholder for webhook implementation
        logger.info(f"Webhook alert: {alert['message']} to {webhook_url}")

class AlertSystem:
    """Main alert system for monitoring and notifications."""

    def __init__(self):
        self.rules: List[AlertRule] = []
        self.notifications: List[Dict[str, Any]] = []
        self.notification_manager = NotificationManager()
        self.is_monitoring = False
        self.monitor_thread = None

        self._load_alert_rules()

    def _load_alert_rules(self):
        """Load alert rules from file."""
        rules_file = Path("config/alert_rules.json")
        if rules_file.exists():
            try:
                with open(rules_file, 'r') as f:
                    rules_data = json.load(f)
                    self.rules = [AlertRule.from_dict(rule_data) for rule_data in rules_data]
            except Exception as e:
                logger.error(f"Error loading alert rules: {str(e)}")
                self._create_default_rules()

        if not self.rules:
            self._create_default_rules()

    def _create_default_rules(self):
        """Create default alert rules."""
        self.rules = [
            AlertRule(
                name="EURUSD Price Alert",
                condition="price_above",
                threshold=1.1000,
                symbol="EURUSD",
                notification_methods=["dashboard", "email"]
            ),
            AlertRule(
                name="High Volatility Alert",
                condition="volatility_spike",
                threshold=0.02,
                symbol="EURUSD",
                notification_methods=["dashboard"]
            ),
            AlertRule(
                name="Model Performance Alert",
                condition="model_performance",
                threshold=0.85,
                notification_methods=["dashboard", "email"]
            )
        ]
        self._save_alert_rules()

    def _save_alert_rules(self):
        """Save alert rules to file."""
        rules_file = Path("config/alert_rules.json")
        rules_file.parent.mkdir(parents=True, exist_ok=True)
        with open(rules_file, 'w') as f:
            json.dump([rule.to_dict() for rule in self.rules], f, indent=2)

    def add_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.rules.append(rule)
        self._save_alert_rules()
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        self._save_alert_rules()
        logger.info(f"Removed alert rule: {rule_name}")

    def update_rule(self, rule_name: str, updates: Dict[str, Any]):
        """Update an existing alert rule."""
        for rule in self.rules:
            if rule.name == rule_name:
                for key, value in updates.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                self._save_alert_rules()
                logger.info(f"Updated alert rule: {rule_name}")
                break

    def check_alerts(self, market_data: Dict[str, pd.DataFrame]):
        """Check all alert rules against current market data."""
        new_alerts = []

        for rule in self.rules:
            if rule.symbol and rule.symbol in market_data:
                data = market_data[rule.symbol]
                alert = rule.check_condition(data)

                if alert:
                    alert['timestamp'] = datetime.now()
                    alert['rule_name'] = rule.name
                    new_alerts.append(alert)
                    rule.last_triggered = datetime.now()

                    # Send notifications
                    self._send_notifications(alert, rule.notification_methods)

        if new_alerts:
            self.notifications.extend(new_alerts)
            # Keep only last 100 notifications
            self.notifications = self.notifications[-100:]

        return new_alerts

    def _send_notifications(self, alert: Dict[str, Any], methods: List[str]):
        """Send alert notifications via specified methods."""
        for method in methods:
            if method == "dashboard":
                self.notification_manager.send_dashboard_notification(alert)
            elif method == "email":
                self.notification_manager.send_email_notification(alert)
            elif method.startswith("webhook:"):
                webhook_url = method.split(":", 1)[1]
                self.notification_manager.send_webhook_notification(alert, webhook_url)

    def start_monitoring(self, market_data_callback: Callable):
        """Start real-time alert monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(market_data_callback,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Alert monitoring started")

    def stop_monitoring(self):
        """Stop alert monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Alert monitoring stopped")

    def _monitoring_loop(self, market_data_callback: Callable):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Get current market data
                market_data = market_data_callback()

                # Check alerts
                new_alerts = self.check_alerts(market_data)

                if new_alerts:
                    logger.info(f"Triggered {len(new_alerts)} alert(s)")

            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")

            # Wait before next check
            time.sleep(60)  # Check every minute

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        # Return alerts from last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        return [alert for alert in self.notifications
                if alert['timestamp'] > cutoff]

    def clear_alerts(self, alert_ids: Optional[List[str]] = None):
        """Clear alerts."""
        if alert_ids:
            self.notifications = [alert for alert in self.notifications
                                if alert.get('id') not in alert_ids]
        else:
            self.notifications.clear()

class AlertComponent:
    """Streamlit component for alert management."""

    def __init__(self):
        self.alert_system = AlertSystem()

    def render_alert_panel(self):
        """Render the alert management panel."""
        st.markdown("### 🔔 Alert Management")

        # Alert summary
        active_alerts = self.alert_system.get_active_alerts()
        col1, col2, col3 = st.columns(3)

        with col1:
            total_alerts = len(active_alerts)
            st.metric("Active Alerts", total_alerts)

        with col2:
            error_alerts = len([a for a in active_alerts if a.get('severity') == 'error'])
            st.metric("Critical Alerts", error_alerts)

        with col3:
            warning_alerts = len([a for a in active_alerts if a.get('severity') == 'warning'])
            st.metric("Warnings", warning_alerts)

        # Recent alerts
        if active_alerts:
            st.markdown("#### Recent Alerts")
            for alert in active_alerts[-5:]:  # Show last 5
                icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️', 'success': '✅'}.get(
                    alert.get('severity', 'info'), 'ℹ️'
                )

                with st.expander(f"{icon} {alert.get('message', '')}", expanded=False):
                    st.write(f"**Type:** {alert.get('type', '')}")
                    st.write(f"**Time:** {alert.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Rule:** {alert.get('rule_name', '')}")

                    if alert.get('data'):
                        st.json(alert['data'])

        # Alert rules management
        st.markdown("#### Alert Rules")

        # Display existing rules
        for rule in self.alert_system.rules:
            with st.expander(f"{'✅' if rule.enabled else '❌'} {rule.name}", expanded=False):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**Condition:** {rule.condition}")
                    st.write(f"**Threshold:** {rule.threshold}")
                    st.write(f"**Symbol:** {rule.symbol or 'All'}")
                    st.write(f"**Cooldown:** {rule.cooldown_minutes} minutes")

                with col2:
                    enabled = st.checkbox(
                        "Enabled",
                        value=rule.enabled,
                        key=f"enable_{rule.name}",
                        on_change=self._toggle_rule,
                        args=(rule.name,)
                    )

                    if st.button("Delete", key=f"delete_{rule.name}"):
                        self.alert_system.remove_rule(rule.name)
                        st.rerun()

        # Add new rule
        st.markdown("#### Add New Alert Rule")

        with st.form("new_alert_rule"):
            col1, col2 = st.columns(2)

            with col1:
                rule_name = st.text_input("Rule Name")
                condition = st.selectbox("Condition", [
                    "price_above", "price_below", "volatility_spike",
                    "prediction_confidence", "model_performance"
                ])
                threshold = st.number_input("Threshold", value=1.0, step=0.01)

            with col2:
                symbol = st.selectbox("Symbol", ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "All"])
                symbol = None if symbol == "All" else symbol
                notification_methods = st.multiselect(
                    "Notifications",
                    ["dashboard", "email"],
                    default=["dashboard"]
                )
                cooldown = st.number_input("Cooldown (minutes)", value=60, min_value=1)

            if st.form_submit_button("Add Rule"):
                if rule_name:
                    new_rule = AlertRule(
                        name=rule_name,
                        condition=condition,
                        threshold=threshold,
                        symbol=symbol,
                        notification_methods=notification_methods,
                        cooldown_minutes=cooldown
                    )
                    self.alert_system.add_rule(new_rule)
                    st.success(f"Added alert rule: {rule_name}")
                    st.rerun()
                else:
                    st.error("Please enter a rule name")

    def _toggle_rule(self, rule_name: str):
        """Toggle alert rule enabled/disabled."""
        for rule in self.alert_system.rules:
            if rule.name == rule_name:
                rule.enabled = not rule.enabled
                self.alert_system._save_alert_rules()
                break

    def get_alert_badge(self) -> Optional[Dict[str, Any]]:
        """Get alert badge information for header."""
        active_alerts = self.alert_system.get_active_alerts()
        if not active_alerts:
            return None

        critical_count = len([a for a in active_alerts if a.get('severity') == 'error'])
        warning_count = len([a for a in active_alerts if a.get('severity') == 'warning'])

        return {
            'total': len(active_alerts),
            'critical': critical_count,
            'warnings': warning_count,
            'latest': active_alerts[-1] if active_alerts else None
        }