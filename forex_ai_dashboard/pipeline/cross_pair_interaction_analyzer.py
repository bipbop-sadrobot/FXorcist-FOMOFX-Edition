#!/usr/bin/env python3
"""
Cross-Pair Interaction Analyzer for FXorcist-FOMOFX-Edition
Analyzes interactions between currency pairs (e.g., EUR/USD and GBP/USD)
for enhanced model training and federated learning insights.
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/cross_pair_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CrossPairInteractionAnalyzer:
    """Advanced cross-pair interaction analyzer for forex data."""

    def __init__(self):
        self.correlation_threshold = 0.6
        self.major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
        self.currency_groups = {
            'commodity_currencies': ['CAD', 'AUD', 'NZD'],
            'safe_haven': ['USD', 'CHF', 'JPY'],
            'major_european': ['EUR', 'GBP']
        }

    def analyze_pair_interactions(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze interactions between currency pairs.

        Args:
            data_dict: Dictionary of currency pair dataframes

        Returns:
            Comprehensive cross-pair analysis
        """
        logger.info(f"Starting cross-pair analysis for {len(data_dict)} currency pairs")

        results = {
            'correlation_matrix': {},
            'strong_interactions': {},
            'currency_group_analysis': {},
            'lead_lag_relationships': {},
            'clustering_analysis': {},
            'federated_learning_opportunities': []
        }

        if len(data_dict) < 2:
            logger.warning("Need at least 2 currency pairs for cross-analysis")
            return results

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(data_dict)
        results['correlation_matrix'] = correlation_matrix

        # Identify strong interactions
        strong_interactions = self._identify_strong_interactions(correlation_matrix)
        results['strong_interactions'] = strong_interactions

        # Analyze currency groups
        currency_groups = self._analyze_currency_groups(data_dict)
        results['currency_group_analysis'] = currency_groups

        # Analyze lead-lag relationships
        lead_lag = self._analyze_lead_lag_relationships(data_dict)
        results['lead_lag_relationships'] = lead_lag

        # Perform clustering analysis
        clustering = self._perform_clustering_analysis(correlation_matrix)
        results['clustering_analysis'] = clustering

        # Identify federated learning opportunities
        fed_opportunities = self._identify_federated_opportunities(strong_interactions, clustering)
        results['federated_learning_opportunities'] = fed_opportunities

        logger.info("Cross-pair analysis completed")
        return results

    def _calculate_correlation_matrix(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between currency pairs."""
        logger.info("Calculating correlation matrix")

        # Extract returns for each pair
        returns_data = {}
        for pair, df in data_dict.items():
            if 'returns' in df.columns and len(df) > 10:
                returns_data[pair] = df['returns'].dropna()

        if len(returns_data) < 2:
            return {}

        # Create DataFrame with aligned dates
        returns_df = pd.DataFrame(returns_data).dropna()

        if returns_df.empty or len(returns_df.columns) < 2:
            return {}

        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()

        # Convert to nested dict format
        corr_dict = {}
        for col in correlation_matrix.columns:
            corr_dict[col] = correlation_matrix[col].to_dict()

        return corr_dict

    def _identify_strong_interactions(self, correlation_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Identify pairs with strong correlations."""
        strong_interactions = {
            'highly_correlated': [],
            'negatively_correlated': [],
            'weakly_correlated': []
        }

        if not correlation_matrix:
            return strong_interactions

        pairs_analyzed = set()

        for pair1 in correlation_matrix:
            for pair2 in correlation_matrix[pair1]:
                if pair1 == pair2 or f"{pair1}_{pair2}" in pairs_analyzed:
                    continue

                correlation = correlation_matrix[pair1][pair2]
                pair_key = f"{pair1}_{pair2}"

                interaction = {
                    'pair1': pair1,
                    'pair2': pair2,
                    'correlation': correlation,
                    'strength': self._classify_correlation_strength(correlation),
                    'direction': 'positive' if correlation > 0 else 'negative'
                }

                if abs(correlation) >= self.correlation_threshold:
                    if correlation > 0:
                        strong_interactions['highly_correlated'].append(interaction)
                    else:
                        strong_interactions['negatively_correlated'].append(interaction)
                else:
                    strong_interactions['weakly_correlated'].append(interaction)

                pairs_analyzed.add(pair_key)
                pairs_analyzed.add(f"{pair2}_{pair1}")

        return strong_interactions

    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength."""
        abs_corr = abs(correlation)

        if abs_corr >= 0.8:
            return 'very_strong'
        elif abs_corr >= 0.6:
            return 'strong'
        elif abs_corr >= 0.4:
            return 'moderate'
        elif abs_corr >= 0.2:
            return 'weak'
        else:
            return 'very_weak'

    def _analyze_currency_groups(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze behavior within currency groups."""
        currency_group_analysis = {}

        for group_name, currencies in self.currency_groups.items():
            group_pairs = []
            for pair in data_dict.keys():
                # Check if pair contains currencies from this group
                pair_currencies = self._extract_currencies_from_pair(pair)
                if any(curr in currencies for curr in pair_currencies):
                    group_pairs.append(pair)

            if len(group_pairs) >= 2:
                # Analyze group behavior
                group_analysis = self._analyze_group_behavior(data_dict, group_pairs, group_name)
                currency_group_analysis[group_name] = group_analysis

        return currency_group_analysis

    def _extract_currencies_from_pair(self, pair: str) -> List[str]:
        """Extract currency codes from pair name."""
        # Handle common formats: EURUSD, EUR/USD, EUR_USD
        pair = pair.replace('/', '').replace('_', '').upper()

        currencies = []
        for currency in self.major_currencies:
            if currency in pair:
                currencies.append(currency)

        return currencies

    def _analyze_group_behavior(self, data_dict: Dict[str, pd.DataFrame],
                              group_pairs: List[str], group_name: str) -> Dict[str, Any]:
        """Analyze behavior of a currency group."""
        group_returns = []

        for pair in group_pairs:
            if pair in data_dict and 'returns' in data_dict[pair].columns:
                returns = data_dict[pair]['returns'].dropna()
                if len(returns) > 0:
                    group_returns.append(returns)

        if not group_returns:
            return {}

        # Calculate group statistics
        group_df = pd.concat(group_returns, axis=1, keys=group_pairs).dropna()

        analysis = {
            'group_size': len(group_pairs),
            'avg_correlation': group_df.corr().mean().mean(),
            'group_volatility': group_df.std().mean(),
            'group_trend': group_df.mean().mean(),
            'pairs': group_pairs
        }

        # Identify group leader (pair with highest correlation to others)
        if len(group_pairs) > 1:
            correlations = group_df.corr()
            avg_correlations = correlations.mean()
            leader_pair = avg_correlations.idxmax()
            analysis['group_leader'] = leader_pair
            analysis['leadership_score'] = avg_correlations[leader_pair]

        return analysis

    def _analyze_lead_lag_relationships(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze lead-lag relationships between pairs."""
        lead_lag_analysis = {}

        pairs = list(data_dict.keys())
        if len(pairs) < 2:
            return lead_lag_analysis

        for i in range(len(pairs)):
            for j in range(i+1, len(pairs)):
                pair1, pair2 = pairs[i], pairs[j]

                if 'returns' in data_dict[pair1].columns and 'returns' in data_dict[pair2].columns:
                    returns1 = data_dict[pair1]['returns'].dropna()
                    returns2 = data_dict[pair2]['returns'].dropna()

                    # Align data by timestamp
                    combined = pd.concat([returns1, returns2], axis=1, keys=[pair1, pair2]).dropna()

                    if len(combined) > 30:  # Need sufficient data
                        # Calculate cross-correlation at different lags
                        max_lag = min(10, len(combined) // 4)  # Up to 10 periods or 1/4 of data
                        correlations = {}

                        for lag in range(-max_lag, max_lag + 1):
                            if lag < 0:
                                corr = combined[pair1].corr(combined[pair2].shift(-lag))
                            else:
                                corr = combined[pair1].shift(lag).corr(combined[pair2])

                            correlations[lag] = corr if not np.isnan(corr) else 0

                        # Find optimal lag
                        best_lag = max(correlations.items(), key=lambda x: abs(x[1]))

                        lead_lag_analysis[f"{pair1}_{pair2}"] = {
                            'optimal_lag': best_lag[0],
                            'max_correlation': best_lag[1],
                            'correlation_profile': correlations,
                            'leader': pair1 if best_lag[0] <= 0 else pair2,
                            'follower': pair2 if best_lag[0] <= 0 else pair1
                        }

        return lead_lag_analysis

    def _perform_clustering_analysis(self, correlation_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Perform clustering analysis on currency pairs."""
        clustering_results = {}

        if not correlation_matrix or len(correlation_matrix) < 3:
            return clustering_results

        # Convert correlation matrix to distance matrix
        pairs = list(correlation_matrix.keys())
        n_pairs = len(pairs)

        # Create distance matrix (1 - |correlation|)
        distance_matrix = np.zeros((n_pairs, n_pairs))

        for i in range(n_pairs):
            for j in range(n_pairs):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    corr = correlation_matrix[pairs[i]].get(pairs[j], 0)
                    distance_matrix[i, j] = 1 - abs(corr)

        # Perform clustering
        max_clusters = min(5, n_pairs - 1)
        best_score = -1
        best_n_clusters = 2

        for n_clusters in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(distance_matrix)

                # Calculate silhouette score
                if n_pairs > n_clusters:
                    score = silhouette_score(distance_matrix, clusters)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            except:
                continue

        # Final clustering with best number of clusters
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(distance_matrix)

        # Organize results
        cluster_groups = {}
        for i, pair in enumerate(pairs):
            cluster_id = int(clusters[i])
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(pair)

        clustering_results = {
            'n_clusters': best_n_clusters,
            'silhouette_score': best_score,
            'clusters': cluster_groups,
            'cluster_centers': kmeans.cluster_centers_.tolist() if hasattr(kmeans, 'cluster_centers_') else []
        }

        return clustering_results

    def _identify_federated_opportunities(self, strong_interactions: Dict[str, List],
                                       clustering: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities for federated learning."""
        opportunities = []

        # Opportunities based on strong correlations
        if 'highly_correlated' in strong_interactions:
            for interaction in strong_interactions['highly_correlated']:
                opportunity = {
                    'type': 'correlation_based',
                    'pairs': [interaction['pair1'], interaction['pair2']],
                    'correlation': interaction['correlation'],
                    'rationale': f"High correlation ({interaction['correlation']:.3f}) suggests shared market dynamics",
                    'federated_benefit': 'Model updates can be shared between correlated pairs'
                }
                opportunities.append(opportunity)

        # Opportunities based on clustering
        if 'clusters' in clustering:
            for cluster_id, pairs in clustering['clusters'].items():
                if len(pairs) >= 2:
                    opportunity = {
                        'type': 'cluster_based',
                        'pairs': pairs,
                        'cluster_id': cluster_id,
                        'rationale': f"Pairs in cluster {cluster_id} show similar behavior patterns",
                        'federated_benefit': f'Federated learning across {len(pairs)} related pairs'
                    }
                    opportunities.append(opportunity)

        return opportunities

    def generate_cross_pair_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate trading recommendations based on cross-pair analysis."""
        recommendations = {}

        # Correlation-based recommendations
        if 'strong_interactions' in analysis_results:
            interactions = analysis_results['strong_interactions']

            if 'highly_correlated' in interactions and interactions['highly_correlated']:
                top_corr = max(interactions['highly_correlated'],
                             key=lambda x: abs(x['correlation']))

                if top_corr['correlation'] > 0:
                    recommendations['correlation_strategy'] = (
                        f"Consider pairs trading between {top_corr['pair1']} and {top_corr['pair2']} "
                        f"(correlation: {top_corr['correlation']:.3f})"
                    )
                else:
                    recommendations['hedging_strategy'] = (
                        f"Use {top_corr['pair1']} to hedge {top_corr['pair2']} "
                        f"(correlation: {top_corr['correlation']:.3f})"
                    )

        # Clustering-based recommendations
        if 'clustering_analysis' in analysis_results:
            clustering = analysis_results['clustering_analysis']

            if 'clusters' in clustering:
                largest_cluster = max(clustering['clusters'].values(),
                                    key=len, default=[])

                if len(largest_cluster) >= 3:
                    recommendations['cluster_strategy'] = (
                        f"Focus on cluster with {len(largest_cluster)} pairs: "
                        f"{', '.join(largest_cluster[:3])}{'...' if len(largest_cluster) > 3 else ''}"
                    )

        # Lead-lag recommendations
        if 'lead_lag_relationships' in analysis_results:
            lead_lag = analysis_results['lead_lag_relationships']

            if lead_lag:
                best_lead_lag = max(lead_lag.values(),
                                  key=lambda x: abs(x['max_correlation']))

                if abs(best_lead_lag['max_correlation']) > 0.5:
                    recommendations['lead_lag_strategy'] = (
                        f"Monitor {best_lead_lag['leader']} for signals on {best_lead_lag['follower']} "
                        f"(lag: {best_lead_lag['optimal_lag']}, correlation: {best_lead_lag['max_correlation']:.3f})"
                    )

        return recommendations

def create_cross_pair_analysis_demo():
    """Create demonstration of cross-pair analysis."""
    analyzer = CrossPairInteractionAnalyzer()

    # Create synthetic data for major pairs
    dates = pd.date_range('2024-01-01', '2024-08-01', freq='D')
    np.random.seed(42)

    # Create correlated pairs
    base_returns = np.random.normal(0, 0.01, len(dates))

    # EURUSD: follows base with some independence
    eurusd_returns = base_returns * 0.7 + np.random.normal(0, 0.007, len(dates))

    # GBPUSD: highly correlated with EURUSD
    gbpusd_returns = eurusd_returns * 0.8 + np.random.normal(0, 0.005, len(dates))

    # USDJPY: negatively correlated with USD pairs
    usdjpy_returns = -base_returns * 0.6 + np.random.normal(0, 0.008, len(dates))

    # AUDUSD: commodity currency, less correlated
    audusd_returns = base_returns * 0.3 + np.random.normal(0, 0.009, len(dates))

    # Create dataframes
    data_dict = {}
    pairs_data = {
        'EURUSD': eurusd_returns,
        'GBPUSD': gbpusd_returns,
        'USDJPY': usdjpy_returns,
        'AUDUSD': audusd_returns
    }

    for pair, returns in pairs_data.items():
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

        data_dict[pair] = df

    # Run analysis
    results = analyzer.analyze_pair_interactions(data_dict)
    recommendations = analyzer.generate_cross_pair_recommendations(results)

    print("=== Cross-Pair Interaction Analysis Results ===")
    print(f"Pairs analyzed: {len(data_dict)}")
    print(f"Strong interactions found: {len(results.get('strong_interactions', {}).get('highly_correlated', []))}")
    print(f"Clusters identified: {results.get('clustering_analysis', {}).get('n_clusters', 0)}")
    print(f"Federated opportunities: {len(results.get('federated_learning_opportunities', []))}")

    if recommendations:
        print("\n=== Trading Recommendations ===")
        for key, recommendation in recommendations.items():
            print(f"{key}: {recommendation}")

    return analyzer, results, recommendations

if __name__ == "__main__":
    analyzer, results, recommendations = create_cross_pair_analysis_demo()