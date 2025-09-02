#!/usr/bin/env python3
"""
Intuition-Driven Forex Analyzer for FXorcist-FOMOFX-Edition
Implements advanced temporal correlations, recency weighting, and cross-pair interactions
for enhanced model training and memory system integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from memory_system.memory import IntegratedMemorySystem
from memory_system.anomaly import AnomalyDetector
from memory_system.event_bus import EventBus
from memory_system.metadata import SharedMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/intuition_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntuitionDrivenAnalyzer:
    """Advanced intuition-driven forex analyzer with temporal and cross-pair insights."""

    def __init__(self):
        # Initialize memory system components
        self.event_bus = EventBus()
        self.metadata = SharedMetadata()
        self.memory = IntegratedMemorySystem(self.event_bus, self.metadata)
        self.anomaly_detector = AnomalyDetector(self.memory)

        # Analysis parameters
        self.recency_decay_factor = 0.95  # How much to weight recent data
        self.temporal_window_days = 30    # Days to analyze for correlations
        self.cross_pair_threshold = 0.7   # Correlation threshold for pair interactions

        # Major currency pairs for cross-analysis
        self.major_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'
        ]

        # Temporal patterns to analyze
        self.temporal_patterns = {
            'monthly_seasonality': 30,
            'weekly_patterns': 7,
            'daily_cycles': 1,
            'hourly_trends': 1/24
        }

    def analyze_with_intuition(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform intuition-driven analysis on forex data.

        Args:
            data_dict: Dictionary of currency pair dataframes

        Returns:
            Comprehensive analysis results
        """
        logger.info(f"Starting intuition-driven analysis for {len(data_dict)} currency pairs")

        results = {
            'recency_weighted_insights': {},
            'temporal_correlations': {},
            'cross_pair_interactions': {},
            'memory_enhanced_predictions': {},
            'federated_learning_insights': {},
            'anomaly_detected_patterns': []
        }

        # 1. Apply recency weighting to all data
        weighted_data = self._apply_recency_weighting(data_dict)

        # 2. Analyze temporal correlations
        temporal_insights = self._analyze_temporal_patterns(weighted_data)

        # 3. Identify cross-pair interactions
        cross_pair_insights = self._analyze_cross_pair_interactions(weighted_data)

        # 4. Generate memory-enhanced predictions
        memory_predictions = self._generate_memory_enhanced_predictions(weighted_data)

        # 5. Create federated learning insights
        federated_insights = self._create_federated_learning_insights(weighted_data)

        # 6. Detect anomalous patterns
        anomaly_patterns = self._detect_anomalous_patterns(weighted_data)

        # Update results
        results.update({
            'recency_weighted_insights': weighted_data,
            'temporal_correlations': temporal_insights,
            'cross_pair_interactions': cross_pair_insights,
            'memory_enhanced_predictions': memory_predictions,
            'federated_learning_insights': federated_insights,
            'anomaly_detected_patterns': anomaly_patterns
        })

        # Store insights in memory system
        self._store_insights_in_memory(results)

        logger.info("Intuition-driven analysis completed")
        return results

    def _apply_recency_weighting(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply recency weighting to prioritize newer data points."""
        logger.info("Applying recency weighting to data")

        weighted_data = {}

        for pair, df in data_dict.items():
            if df.empty or 'timestamp' not in df.columns:
                continue

            # Ensure timestamp is datetime
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp')

            # Calculate recency weights
            max_timestamp = df['timestamp'].max()
            df['days_old'] = (max_timestamp - df['timestamp']).dt.total_seconds() / (24 * 3600)

            # Apply exponential decay weighting
            df['recency_weight'] = np.exp(-self.recency_decay_factor * df['days_old'])

            # Normalize weights
            df['recency_weight'] = df['recency_weight'] / df['recency_weight'].max()

            # Apply weights to key indicators
            weight_columns = ['close', 'returns', 'volatility']
            for col in weight_columns:
                if col in df.columns:
                    df[f'{col}_weighted'] = df[col] * df['recency_weight']

            weighted_data[pair] = df
            logger.info(f"Applied recency weighting to {pair}: {len(df)} records")

        return weighted_data

    def _analyze_temporal_patterns(self, weighted_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze temporal patterns and correlations across months."""
        logger.info("Analyzing temporal patterns")

        temporal_insights = {}

        for pair, df in weighted_data.items():
            if df.empty:
                continue

            insights = {}

            # Monthly seasonality analysis
            if len(df) > 30:  # Need at least 30 days
                df['month'] = df['timestamp'].dt.month
                df['day_of_month'] = df['timestamp'].dt.day

                # Monthly average returns by day
                monthly_patterns = df.groupby('day_of_month')['returns'].mean()
                insights['monthly_seasonality'] = monthly_patterns.to_dict()

                # Identify strong monthly patterns
                strong_patterns = monthly_patterns[abs(monthly_patterns) > monthly_patterns.std()]
                insights['strong_monthly_patterns'] = strong_patterns.to_dict()

            # Weekly patterns
            if len(df) > 7:
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                weekly_patterns = df.groupby('day_of_week')['returns'].mean()
                insights['weekly_patterns'] = weekly_patterns.to_dict()

            # Hourly patterns (if available)
            if 'hour' in df.columns or len(df) > 24:
                if 'hour' not in df.columns:
                    df['hour'] = df['timestamp'].dt.hour
                hourly_patterns = df.groupby('hour')['returns'].mean()
                insights['hourly_patterns'] = hourly_patterns.to_dict()

            # Temporal correlation analysis
            if len(df) > self.temporal_window_days:
                # Rolling correlations over time windows
                rolling_corr = df['returns'].rolling(window=self.temporal_window_days).corr()
                insights['temporal_stability'] = rolling_corr.mean()
                insights['temporal_volatility'] = rolling_corr.std()

            temporal_insights[pair] = insights

        return temporal_insights

    def _analyze_cross_pair_interactions(self, weighted_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze interactions between currency pairs."""
        logger.info("Analyzing cross-pair interactions")

        cross_pair_insights = {}

        # Get available pairs
        available_pairs = list(weighted_data.keys())

        # Focus on major pairs
        major_available = [pair for pair in available_pairs if any(major in pair.upper() for major in self.major_pairs)]

        if len(major_available) < 2:
            logger.warning("Need at least 2 major pairs for cross-analysis")
            return cross_pair_insights

        # Calculate cross-correlations
        returns_data = {}
        for pair in major_available:
            df = weighted_data[pair]
            if 'returns' in df.columns and len(df) > 10:
                returns_data[pair] = df['returns'].dropna()

        if len(returns_data) >= 2:
            # Create correlation matrix
            returns_df = pd.DataFrame(returns_data).dropna()
            correlation_matrix = returns_df.corr()

            # Find strong correlations
            strong_correlations = {}
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    pair1 = correlation_matrix.columns[i]
                    pair2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]

                    if abs(corr_value) > self.cross_pair_threshold:
                        strong_correlations[f"{pair1}_{pair2}"] = {
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate',
                            'direction': 'positive' if corr_value > 0 else 'negative'
                        }

            cross_pair_insights['correlation_matrix'] = correlation_matrix.to_dict()
            cross_pair_insights['strong_correlations'] = strong_correlations

            # Identify currency group behaviors
            usd_pairs = [pair for pair in major_available if 'USD' in pair.upper()]
            if len(usd_pairs) >= 2:
                usd_returns = {pair: returns_data[pair] for pair in usd_pairs if pair in returns_data}
                if usd_returns:
                    usd_corr = pd.DataFrame(usd_returns).corr().mean().mean()
                    cross_pair_insights['usd_pair_cohesion'] = usd_corr

        return cross_pair_insights

    def _generate_memory_enhanced_predictions(self, weighted_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate predictions enhanced by memory system insights."""
        logger.info("Generating memory-enhanced predictions")

        memory_predictions = {}

        for pair, df in weighted_data.items():
            if df.empty:
                continue

            # Get recent memory entries for this pair
            memory_entries = self.memory.recall(query=pair, top_k=50)

            if memory_entries:
                # Extract patterns from memory
                recent_errors = []
                recent_predictions = []

                for entry in memory_entries:
                    if isinstance(entry, dict):
                        if 'error' in entry:
                            recent_errors.append(entry['error'])
                        if 'prediction' in entry:
                            recent_predictions.append(entry['prediction'])

                # Calculate memory-based confidence
                if recent_errors:
                    avg_error = np.mean(recent_errors)
                    error_volatility = np.std(recent_errors)

                    memory_predictions[pair] = {
                        'memory_confidence': 1.0 / (1.0 + avg_error),
                        'prediction_stability': 1.0 / (1.0 + error_volatility),
                        'recent_performance': avg_error,
                        'memory_sample_size': len(memory_entries)
                    }

                    # Add memory-enhanced features to dataframe
                    df_copy = df.copy()
                    df_copy['memory_confidence'] = memory_predictions[pair]['memory_confidence']
                    df_copy['prediction_stability'] = memory_predictions[pair]['prediction_stability']

                    weighted_data[pair] = df_copy

        return memory_predictions

    def _create_federated_learning_insights(self, weighted_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create insights for federated learning scenarios."""
        logger.info("Creating federated learning insights")

        federated_insights = {
            'pair_clusters': {},
            'temporal_synchronization': {},
            'cross_pair_learning_opportunities': []
        }

        # Identify pair clusters based on correlation
        if 'cross_pair_interactions' in self._analyze_cross_pair_interactions(weighted_data):
            corr_data = self._analyze_cross_pair_interactions(weighted_data)

            if 'strong_correlations' in corr_data:
                # Group pairs by correlation clusters
                clusters = {}
                for pair_combo, data in corr_data['strong_correlations'].items():
                    pair1, pair2 = pair_combo.split('_', 1)

                    # Simple clustering based on correlation strength
                    cluster_key = f"cluster_{data['strength']}"
                    if cluster_key not in clusters:
                        clusters[cluster_key] = []
                    clusters[cluster_key].extend([pair1, pair2])

                # Remove duplicates
                for cluster in clusters:
                    clusters[cluster] = list(set(clusters[cluster]))

                federated_insights['pair_clusters'] = clusters

        # Identify temporal synchronization opportunities
        temporal_sync = {}
        for pair, df in weighted_data.items():
            if 'recency_weight' in df.columns:
                # Calculate temporal synchronization score
                recent_weight = df['recency_weight'].tail(100).mean()
                temporal_sync[pair] = recent_weight

        federated_insights['temporal_synchronization'] = temporal_sync

        return federated_insights

    def _detect_anomalous_patterns(self, weighted_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Detect anomalous patterns using memory system."""
        logger.info("Detecting anomalous patterns")

        anomalous_patterns = []

        for pair, df in weighted_data.items():
            if df.empty:
                continue

            # Convert recent data to memory format for anomaly detection
            recent_data = df.tail(100)  # Last 100 records

            for _, row in recent_data.iterrows():
                memory_entry = {
                    "model": f"intuition_analysis_{pair}",
                    "prediction": row.get('close', 0),
                    "target": row.get('close', 0),
                    "error": 0.0,
                    "features": {
                        "returns": row.get('returns', 0),
                        "volatility": row.get('volatility', 0),
                        "recency_weight": row.get('recency_weight', 1.0)
                    },
                    "ts": row.get('timestamp').timestamp() if pd.notna(row.get('timestamp')) else datetime.now().timestamp()
                }
                self.memory.add_record(memory_entry)

            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies()

            if anomalies.get('anomalies'):
                for anomaly in anomalies['anomalies']:
                    anomalous_patterns.append({
                        'pair': pair,
                        'anomaly': anomaly,
                        'timestamp': datetime.now().isoformat(),
                        'severity': anomaly.get('score', 0)
                    })

        return anomalous_patterns

    def _store_insights_in_memory(self, results: Dict[str, Any]):
        """Store analysis insights in memory system."""
        try:
            # Store key insights
            insight_entry = {
                "model": "intuition_analyzer",
                "prediction": 0.0,  # Not a prediction, but an analysis
                "target": 0.0,
                "error": 0.0,
                "features": {
                    "pairs_analyzed": len(results.get('recency_weighted_insights', {})),
                    "temporal_patterns": len(results.get('temporal_correlations', {})),
                    "cross_pair_interactions": len(results.get('cross_pair_interactions', {})),
                    "anomalies_detected": len(results.get('anomaly_detected_patterns', []))
                },
                "ts": datetime.now().timestamp()
            }

            self.memory.add_record(insight_entry)
            logger.info("Analysis insights stored in memory system")

        except Exception as e:
            logger.error(f"Failed to store insights in memory: {e}")

    def get_intuition_recommendations(self, pair: str, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Get intuition-driven recommendations for a specific pair."""
        recommendations = {
            'recency_bias': 'high',  # Always prioritize recent data
            'temporal_focus': [],
            'cross_pair_opportunities': [],
            'memory_confidence': 0.0,
            'risk_assessment': 'medium'
        }

        # Get temporal insights
        if hasattr(self, '_temporal_correlations') and pair in self._temporal_correlations:
            temp_data = self._temporal_correlations[pair]
            if 'strong_monthly_patterns' in temp_data:
                recommendations['temporal_focus'] = list(temp_data['strong_monthly_patterns'].keys())

        # Get cross-pair opportunities
        if hasattr(self, '_cross_pair_interactions') and 'strong_correlations' in self._cross_pair_interactions:
            for pair_combo, data in self._cross_pair_interactions['strong_correlations'].items():
                if pair in pair_combo:
                    recommendations['cross_pair_opportunities'].append({
                        'pair': pair_combo.replace(pair, '').replace('_', ''),
                        'correlation': data['correlation'],
                        'strength': data['strength']
                    })

        # Get memory confidence
        memory_entries = self.memory.recall(query=pair, top_k=10)
        if memory_entries:
            errors = [entry.get('error', 0) for entry in memory_entries if isinstance(entry, dict)]
            if errors:
                recommendations['memory_confidence'] = 1.0 / (1.0 + np.mean(errors))

        return recommendations

def create_sample_analysis():
    """Create a sample analysis for demonstration."""
    analyzer = IntuitionDrivenAnalyzer()

    # Create sample data
    dates = pd.date_range('2024-01-01', '2024-08-01', freq='D')
    sample_data = {}

    for pair in ['EURUSD', 'GBPUSD', 'USDJPY']:
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, len(dates))
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

        sample_data[pair] = df

    # Run analysis
    results = analyzer.analyze_with_intuition(sample_data)

    print("=== Intuition-Driven Analysis Results ===")
    print(f"Pairs analyzed: {len(results['recency_weighted_insights'])}")
    print(f"Temporal patterns found: {len(results['temporal_correlations'])}")
    print(f"Cross-pair interactions: {len(results['cross_pair_interactions'])}")
    print(f"Anomalies detected: {len(results['anomaly_detected_patterns'])}")
    print(f"Memory records: {len(analyzer.memory.records)}")

    return analyzer, results

if __name__ == "__main__":
    analyzer, results = create_sample_analysis()