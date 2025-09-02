#!/usr/bin/env python3
"""
Continuous Training Scheduler
Automatically runs training pipeline on a schedule and monitors performance.
"""

import schedule
import time
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/continuous_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousTrainingScheduler:
    """Scheduler for continuous model training and monitoring."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.logs_dir = self.project_root / "logs"
        self.models_dir = self.project_root / "models" / "trained"
        self.metrics_history = []

        # Training schedule configuration
        self.training_schedule = {
            'daily': True,      # Run daily training
            'weekly': True,     # Run comprehensive weekly training
            'monthly': True     # Run full monthly retraining
        }

        # Performance thresholds
        self.performance_thresholds = {
            'min_r2': 0.95,     # Minimum acceptable R² score
            'max_rmse': 0.001,  # Maximum acceptable RMSE
            'min_samples': 10000  # Minimum training samples
        }

    def load_metrics_history(self) -> List[Dict]:
        """Load historical training metrics."""
        metrics_file = self.logs_dir / "metrics_history.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading metrics history: {str(e)}")
        return []

    def save_metrics_history(self, metrics: List[Dict]):
        """Save training metrics history."""
        metrics_file = self.logs_dir / "metrics_history.json"
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metrics history: {str(e)}")

    def check_model_performance(self) -> Dict[str, bool]:
        """Check if current models meet performance thresholds."""
        logger.info("Checking model performance")

        performance_status = {
            'meets_thresholds': True,
            'needs_retraining': False,
            'issues': []
        }

        # Load latest metrics
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]

            for model_name, model_data in latest_metrics.get('results', {}).items():
                metrics = model_data.get('metrics', {})

                # Check R² score
                r2_score = metrics.get('r2', 0)
                if r2_score < self.performance_thresholds['min_r2']:
                    performance_status['issues'].append(
                        f"{model_name} R² ({r2_score:.4f}) below threshold ({self.performance_thresholds['min_r2']})"
                    )
                    performance_status['needs_retraining'] = True

                # Check RMSE
                rmse = metrics.get('rmse', float('inf'))
                if rmse > self.performance_thresholds['max_rmse']:
                    performance_status['issues'].append(
                        f"{model_name} RMSE ({rmse:.6f}) above threshold ({self.performance_thresholds['max_rmse']})"
                    )
                    performance_status['needs_retraining'] = True

        if performance_status['issues']:
            performance_status['meets_thresholds'] = False
            logger.warning("Performance issues detected:")
            for issue in performance_status['issues']:
                logger.warning(f"  • {issue}")
        else:
            logger.info("All models meet performance thresholds")

        return performance_status

    def run_daily_training(self):
        """Run daily incremental training."""
        logger.info("🚀 Starting daily training cycle")

        try:
            # Run the automated pipeline
            result = subprocess.run(
                [sys.executable, "automated_training_pipeline.py"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info("✅ Daily training completed successfully")
                self.update_metrics_history()
            else:
                logger.error(f"❌ Daily training failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("Daily training timed out")
        except Exception as e:
            logger.error(f"Error in daily training: {str(e)}")

    def run_weekly_training(self):
        """Run comprehensive weekly training."""
        logger.info("🚀 Starting weekly comprehensive training")

        try:
            # Run with extended parameters for weekly training
            env = dict(os.environ)
            env['TRAINING_MODE'] = 'comprehensive'

            result = subprocess.run(
                [sys.executable, "automated_training_pipeline.py"],
                cwd=self.project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            if result.returncode == 0:
                logger.info("✅ Weekly training completed successfully")
                self.update_metrics_history()
            else:
                logger.error(f"❌ Weekly training failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("Weekly training timed out")
        except Exception as e:
            logger.error(f"Error in weekly training: {str(e)}")

    def run_monthly_training(self):
        """Run full monthly retraining from scratch."""
        logger.info("🚀 Starting monthly full retraining")

        try:
            # Clear old models and retrain everything
            self.cleanup_old_models()

            env = dict(os.environ)
            env['TRAINING_MODE'] = 'full_retrain'

            result = subprocess.run(
                [sys.executable, "automated_training_pipeline.py"],
                cwd=self.project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=14400  # 4 hour timeout
            )

            if result.returncode == 0:
                logger.info("✅ Monthly retraining completed successfully")
                self.update_metrics_history()
            else:
                logger.error(f"❌ Monthly retraining failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("Monthly retraining timed out")
        except Exception as e:
            logger.error(f"Error in monthly retraining: {str(e)}")

    def update_metrics_history(self):
        """Update metrics history from latest training results."""
        # Look for the latest training summary
        summary_files = list(self.logs_dir.glob("training_summary_*.json"))
        if summary_files:
            latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_summary, 'r') as f:
                    summary_data = json.load(f)
                    self.metrics_history.append(summary_data)
                    self.save_metrics_history(self.metrics_history[-100:])  # Keep last 100 entries
                    logger.info(f"Updated metrics history with {latest_summary.name}")
            except Exception as e:
                logger.error(f"Error loading training summary: {str(e)}")

    def cleanup_old_models(self, keep_days: int = 30):
        """Clean up old model files."""
        logger.info(f"Cleaning up models older than {keep_days} days")

        cutoff_date = datetime.now() - timedelta(days=keep_days)

        for model_file in self.models_dir.glob("*"):
            if model_file.is_file():
                try:
                    file_date = datetime.fromtimestamp(model_file.stat().st_mtime)
                    if file_date < cutoff_date:
                        model_file.unlink()
                        logger.info(f"Removed old model: {model_file.name}")
                except Exception as e:
                    logger.warning(f"Error checking {model_file}: {str(e)}")

    def generate_performance_report(self) -> str:
        """Generate a performance report."""
        if not self.metrics_history:
            return "No training history available"

        latest = self.metrics_history[-1]
        report = f"""
📊 Forex AI Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Training Summary:
• Timestamp: {latest.get('timestamp', 'Unknown')}
• Models Trained: {', '.join(latest.get('models_trained', []))}

Performance Metrics:
"""

        for model_name, model_data in latest.get('results', {}).items():
            metrics = model_data.get('metrics', {})
            report += f"""
{model_name.upper()} Model:
  • R² Score: {metrics.get('r2', 'N/A'):.6f}
  • RMSE: {metrics.get('rmse', 'N/A'):.6f}
  • MAE: {metrics.get('mae', 'N/A'):.6f}
  • Training Time: {model_data.get('training_time', 'N/A'):.2f}s
"""

        return report

    def send_notification(self, message: str):
        """Send notification (placeholder for email/slack integration)."""
        logger.info(f"NOTIFICATION: {message}")
        # TODO: Implement actual notification system
        print(f"\n🔔 {message}")

    def start_scheduler(self):
        """Start the continuous training scheduler."""
        logger.info("🎯 Starting Continuous Training Scheduler")

        # Load existing metrics
        self.metrics_history = self.load_metrics_history()

        # Schedule daily training
        if self.training_schedule['daily']:
            schedule.every().day.at("02:00").do(self.run_daily_training)
            logger.info("📅 Scheduled daily training at 02:00")

        # Schedule weekly training
        if self.training_schedule['weekly']:
            schedule.every().week.do(self.run_weekly_training)
            logger.info("📅 Scheduled weekly training")

        # Schedule monthly training
        if self.training_schedule['monthly']:
            schedule.every(30).days.do(self.run_monthly_training)
            logger.info("📅 Scheduled monthly retraining")

        # Performance monitoring
        schedule.every(6).hours.do(self.monitor_performance)

        # Report generation
        schedule.every().day.at("08:00").do(self.send_daily_report)

        logger.info("✅ Scheduler started successfully")
        logger.info("Available commands:")
        logger.info("  • 'status' - Show current status")
        logger.info("  • 'report' - Generate performance report")
        logger.info("  • 'train_now' - Run training immediately")
        logger.info("  • 'quit' - Stop scheduler")

        # Interactive command loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

                # Check for user input (simplified)
                try:
                    # In a real implementation, you'd use proper input handling
                    pass
                except KeyboardInterrupt:
                    break

        except KeyboardInterrupt:
            logger.info("🛑 Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")

    def monitor_performance(self):
        """Monitor model performance and trigger alerts if needed."""
        logger.info("🔍 Monitoring model performance")

        performance_status = self.check_model_performance()

        if not performance_status['meets_thresholds']:
            alert_message = "⚠️ Model Performance Alert!\n" + "\n".join(performance_status['issues'])
            self.send_notification(alert_message)

            if performance_status['needs_retraining']:
                logger.info("🔄 Triggering immediate retraining due to performance issues")
                self.run_daily_training()

    def send_daily_report(self):
        """Send daily performance report."""
        report = self.generate_performance_report()
        self.send_notification(f"Daily Performance Report:\n{report}")

def main():
    """Main function to run the continuous training scheduler."""
    scheduler = ContinuousTrainingScheduler()

    print("\n" + "="*60)
    print("🤖 CONTINUOUS FOREX AI TRAINING SCHEDULER")
    print("="*60)
    print("This scheduler will:")
    print("  • Run daily training at 02:00")
    print("  • Run weekly comprehensive training")
    print("  • Run monthly full retraining")
    print("  • Monitor model performance")
    print("  • Send daily performance reports")
    print("="*60)

    try:
        scheduler.start_scheduler()
    except KeyboardInterrupt:
        print("\n👋 Scheduler stopped")
    except Exception as e:
        logger.error(f"Scheduler failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()