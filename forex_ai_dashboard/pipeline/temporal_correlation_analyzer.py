#!/usr/bin/env python3
"""
Temporal Correlation Analyzer for FXorcist-FOMOFX-Edition
Analyzes temporal patterns, monthly correlations, and time-based insights
for enhanced forex trading model performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/temporal_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TemporalCorrelationAnalyzer:
    """Advanced temporal correlation analyzer for forex data."""

    def __init__(self):
        self.monthly_patterns = {}
        self.seasonal_patterns = {}
        self.temporal_windows = {
            'short_term': 7,    # 1 week
            'medium_term': 30,  # 1 month
            'long_term': 90,    # 3 months
            'yearly': 365       # 1 year
        }

    def analyze_monthly_correlations(self, df: pd.DataFrame, pair_name: str) -> Dict[str, Any]:
        """Analyze monthly patterns and correlations."""
        if df.empty or 'timestamp' not in df.columns:
            return {}

        # Ensure timestamp is datetime
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp')

        # Extract temporal features
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour'] = df['timestamp'].dt.hour

        monthly_insights = {}

        # Monthly seasonality analysis
        if len(df) > 30:
            monthly_returns = df.groupby('day_of_month')['returns'].mean()
            monthly_volatility = df.groupby('day_of_month')['returns'].std()

            # Identify significant monthly patterns
            significant_days = monthly_returns[abs(monthly_returns) > monthly_returns.std() * 1.5]

            monthly_insights['monthly_seasonality'] = {
                'average_returns_by_day': monthly_returns.to_dict(),
                'volatility_by_day': monthly_volatility.to_dict(),
                'significant_days': significant_days.to_dict(),
                'best_days': monthly_returns.nlargest(5).to_dict(),
                'worst_days': monthly_returns.nsmallest(5).to_dict()
            }

        # Monthly correlation analysis
        if len(df) > 60:  # Need at least 2 months
            monthly_correlations = self._calculate_monthly_correlations(df)
            monthly_insights['monthly_correlations'] = monthly_correlations

        # Month-over-month performance
        mom_performance = self._analyze_month_over_month(df)
        monthly_insights['month_over_month'] = mom_performance

        return monthly_insights

    def _calculate_monthly_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlations between different months."""
        correlations = {}

        # Group by month and calculate correlations
        monthly_data = df.groupby(df['timestamp'].dt.to_period('M'))

        if len(monthly_data) >= 3:  # Need at least 3 months
            monthly_returns = []
            months = []

            for period, group in monthly_data:
                if len(group) > 5:  # Minimum data points
                    monthly_returns.append(group['returns'].mean())
                    months.append(str(period))

            if len(monthly_returns) >= 3:
                # Calculate autocorrelation
                autocorr = pd.Series(monthly_returns).autocorr(lag=1)
                correlations['monthly_autocorrelation'] = autocorr

                # Calculate trend
                if len(monthly_returns) >= 5:
                    trend = np.polyfit(range(len(monthly_returns)), monthly_returns, 1)[0]
                    correlations['monthly_trend'] = trend

                # Seasonal patterns
                seasonal = self._detect_seasonal_patterns(monthly_returns, months)
                correlations['seasonal_patterns'] = seasonal

        return correlations

    def _detect_seasonal_patterns(self, returns: List[float], months: List[str]) -> Dict[str, Any]:
        """Detect seasonal patterns in monthly returns."""
        patterns = {}

        if len(returns) < 12:  # Need at least a year
            return patterns

        # Convert to numpy array for analysis
        returns_array = np.array(returns)

        # Calculate seasonal decomposition (simplified)
        seasonal_component = np.zeros(len(returns_array))

        # Look for quarterly patterns
        for i in range(3, len(returns_array)):
            seasonal_component[i] = returns_array[i] - returns_array[i-3]

        # Identify strong seasonal effects
        seasonal_strength = np.std(seasonal_component) / np.std(returns_array)
        patterns['seasonal_strength'] = seasonal_strength

        # Find months with consistent patterns
        monthly_avg = {}
        for i, month in enumerate(months):
            month_name = datetime.strptime(month, '%Y-%m').strftime('%B')
            if month_name not in monthly_avg:
                monthly_avg[month_name] = []
            monthly_avg[month_name].append(returns[i])

        # Calculate average return by month name
        month_averages = {}
        for month, values in monthly_avg.items():
            if len(values) >= 2:  # Need at least 2 years of data
                month_averages[month] = np.mean(values)

        patterns['monthly_averages'] = month_averages

        # Identify best and worst months
        if month_averages:
            best_month = max(month_averages.items(), key=lambda x: x[1])
            worst_month = min(month_averages.items(), key=lambda x: x[1])

            patterns['best_month'] = {'month': best_month[0], 'return': best_month[1]}
            patterns['worst_month'] = {'month': worst_month[0], 'return': worst_month[1]}

        return patterns

    def _analyze_month_over_month(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze month-over-month performance changes."""
        mom_analysis = {}

        # Group by month
        monthly_groups = df.groupby(df['timestamp'].dt.to_period('M'))

        monthly_stats = []
        for period, group in monthly_groups:
            if len(group) > 5:  # Minimum data points
                stats = {
                    'month': str(period),
                    'avg_return': group['returns'].mean(),
                    'volatility': group['returns'].std(),
                    'total_return': group['returns'].sum(),
                    'max_drawdown': (group['close'].max() - group['close'].min()) / group['close'].max(),
                    'trading_days': len(group)
                }
                monthly_stats.append(stats)

        if len(monthly_stats) >= 2:
            # Calculate month-over-month changes
            returns = [stat['avg_return'] for stat in monthly_stats]
            volatilities = [stat['volatility'] for stat in monthly_stats]

            mom_changes = []
            for i in range(1, len(returns)):
                change = returns[i] - returns[i-1]
                mom_changes.append({
                    'from_month': monthly_stats[i-1]['month'],
                    'to_month': monthly_stats[i]['month'],
                    'return_change': change,
                    'volatility_change': volatilities[i] - volatilities[i-1]
                })

            mom_analysis['monthly_changes'] = mom_changes
            mom_analysis['average_mom_change'] = np.mean([change['return_change'] for change in mom_changes])

            # Identify significant changes
            significant_changes = [change for change in mom_changes
                                 if abs(change['return_change']) > np.std([c['return_change'] for c in mom_changes]) * 1.5]
            mom_analysis['significant_changes'] = significant_changes

        mom_analysis['monthly_stats'] = monthly_stats
        return mom_analysis

    def analyze_temporal_windows(self, df: pd.DataFrame, pair_name: str) -> Dict[str, Any]:
        """Analyze different temporal windows for patterns."""
        if df.empty:
            return {}

        temporal_analysis = {}

        for window_name, days in self.temporal_windows.items():
            if len(df) < days:
                continue

            # Rolling window analysis
            window_data = df.copy()
            window_data = window_data.set_index('timestamp')

            # Calculate rolling statistics
            rolling_returns = window_data['returns'].rolling(window=f'{days}D', min_periods=1).mean()
            rolling_volatility = window_data['returns'].rolling(window=f'{days}D', min_periods=1).std()

            # Calculate rolling correlations
            if 'close' in window_data.columns:
                rolling_corr = window_data['returns'].rolling(window=f'{days}D', min_periods=1).corr(window_data['close'].shift(1))

            temporal_analysis[window_name] = {
                'rolling_mean_return': rolling_returns.mean(),
                'rolling_volatility': rolling_volatility.mean(),
                'max_rolling_return': rolling_returns.max(),
                'min_rolling_return': rolling_returns.min(),
                'volatility_trend': rolling_volatility.iloc[-1] - rolling_volatility.iloc[0] if len(rolling_volatility) > 1 else 0
            }

        return temporal_analysis

    def detect_temporal_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect temporal anomalies in the data."""
        anomalies = []

        if df.empty or len(df) < 30:
            return anomalies

        # Ensure timestamp is datetime
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values('timestamp')

        # Calculate rolling statistics
        df['rolling_mean_30'] = df['returns'].rolling(window=30, min_periods=1).mean()
        df['rolling_std_30'] = df['returns'].rolling(window=30, min_periods=1).std()

        # Detect outliers
        df['z_score'] = (df['returns'] - df['rolling_mean_30']) / df['rolling_std_30']

        # Find significant anomalies
        anomaly_threshold = 3.0  # 3 standard deviations
        anomalous_points = df[abs(df['z_score']) > anomaly_threshold]

        for _, row in anomalous_points.iterrows():
            anomaly = {
                'timestamp': row['timestamp'].isoformat(),
                'return_value': row['returns'],
                'z_score': row['z_score'],
                'expected_return': row['rolling_mean_30'],
                'severity': 'extreme' if abs(row['z_score']) > 4.0 else 'significant'
            }
            anomalies.append(anomaly)

        return anomalies

    def generate_temporal_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate trading recommendations based on temporal analysis."""
        recommendations = {}

        # Monthly seasonality recommendations
        if 'monthly_seasonality' in analysis_results:
            monthly_data = analysis_results['monthly_seasonality']
            if 'best_days' in monthly_data and monthly_data['best_days']:
                best_day = max(monthly_data['best_days'].items(), key=lambda x: x[1])
                recommendations['monthly_timing'] = f"Consider increased exposure around day {best_day[0]} of each month"

        # Seasonal pattern recommendations
        if 'monthly_correlations' in analysis_results:
            corr_data = analysis_results['monthly_correlations']
            if 'seasonal_patterns' in corr_data:
                seasonal = corr_data['seasonal_patterns']
                if 'best_month' in seasonal:
                    best_month = seasonal['best_month']
                    recommendations['seasonal_timing'] = f"Historically strong performance in {best_month['month']}"

        # Temporal window recommendations
        if 'temporal_windows' in analysis_results:
            windows = analysis_results['temporal_windows']
            if 'short_term' in windows:
                short_term = windows['short_term']
                if short_term['rolling_mean_return'] > 0:
                    recommendations['short_term_trend'] = "Positive short-term momentum detected"
                else:
                    recommendations['short_term_trend'] = "Negative short-term momentum - exercise caution"

        return recommendations

def create_comprehensive_temporal_analysis():
    """Create comprehensive temporal analysis for demonstration."""
    analyzer = TemporalCorrelationAnalyzer()

    # Create sample data with temporal patterns
    dates = pd.date_range('2023-01-01', '2024-08-01', freq='D')
    np.random.seed(42)

    # Create synthetic data with monthly patterns
    monthly_pattern = np.sin(2 * np.pi * np.arange(len(dates)) / 30) * 0.002  # Monthly cycle
    weekly_pattern = np.sin(2 * np.pi * np.arange(len(dates)) / 7) * 0.001   # Weekly cycle
    noise = np.random.normal(0, 0.005, len(dates))

    returns = monthly_pattern + weekly_pattern + noise
    prices = 100 * (1 + returns).cumprod()

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates)),
        'returns': returns,
        'volatility': pd.Series(returns).rolling(5).std().fillna(0.01)
    })

    # Run comprehensive analysis
    monthly_insights = analyzer.analyze_monthly_correlations(df, 'EURUSD')
    temporal_windows = analyzer.analyze_temporal_windows(df, 'EURUSD')
    anomalies = analyzer.detect_temporal_anomalies(df)

    # Combine results
    analysis_results = {
        'monthly_seasonality': monthly_insights,
        'temporal_windows': temporal_windows,
        'anomalies': anomalies
    }

    recommendations = analyzer.generate_temporal_recommendations(analysis_results)

    print("=== Temporal Correlation Analysis Results ===")
    print(f"Monthly patterns analyzed: {len(monthly_insights)}")
    print(f"Temporal windows analyzed: {len(temporal_windows)}")
    print(f"Anomalies detected: {len(anomalies)}")
    print(f"Recommendations generated: {len(recommendations)}")

    if recommendations:
        print("\n=== Trading Recommendations ===")
        for key, recommendation in recommendations.items():
            print(f"{key}: {recommendation}")

    return analyzer, analysis_results, recommendations

if __name__ == "__main__":
    analyzer, results, recommendations = create_comprehensive_temporal_analysis()