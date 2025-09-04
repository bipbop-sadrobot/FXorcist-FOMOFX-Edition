"""
Decomposed tests module for FXorcist.
Implements robust testing under various market conditions and perturbations.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit

@dataclass
class MarketRegime:
    """Represents a market regime with specific characteristics."""
    name: str
    volatility_range: Tuple[float, float]
    min_samples: int = 100
    
    def identify(self, returns: pd.Series) -> pd.Series:
        """Identify periods belonging to this regime."""
        vol = returns.rolling(20).std()
        in_regime = (vol >= self.volatility_range[0]) & (vol < self.volatility_range[1])
        return in_regime

@dataclass
class Perturbation:
    """Defines a market perturbation for testing robustness."""
    name: str
    apply_fn: Callable[[pd.DataFrame], pd.DataFrame]
    description: str

class DecomposedTestResult:
    """Container for decomposed test results."""
    def __init__(
        self,
        regime: str,
        perturbation: str,
        auc_score: float,
        precision: float,
        recall: float,
        n_samples: int
    ):
        self.regime = regime
        self.perturbation = perturbation
        self.auc_score = auc_score
        self.precision = precision
        self.recall = recall
        self.n_samples = n_samples
    
    def to_dict(self) -> Dict[str, float]:
        """Convert results to dictionary."""
        return {
            'regime': self.regime,
            'perturbation': self.perturbation,
            'auc': self.auc_score,
            'precision': self.precision,
            'recall': self.recall,
            'n_samples': self.n_samples
        }

def create_market_regimes() -> List[MarketRegime]:
    """Create standard market regime definitions."""
    return [
        MarketRegime('low_vol', (0.0, 0.001)),
        MarketRegime('medium_vol', (0.001, 0.002)),
        MarketRegime('high_vol', (0.002, float('inf')))
    ]

def create_perturbations() -> List[Perturbation]:
    """Create standard market perturbations."""
    def add_spread(df: pd.DataFrame, spread: float = 0.0001) -> pd.DataFrame:
        """Add bid-ask spread effect."""
        df = df.copy()
        df['adjusted_close'] = df['close'] - spread * np.sign(df.get('signal', 0))
        return df
    
    def add_slippage(df: pd.DataFrame, std: float = 0.0001) -> pd.DataFrame:
        """Add execution slippage."""
        df = df.copy()
        noise = np.random.normal(0, std, size=len(df))
        df['adjusted_close'] = df['close'] * (1 + noise)
        return df
    
    def add_feature_noise(df: pd.DataFrame, std: float = 0.01) -> pd.DataFrame:
        """Add noise to feature columns."""
        df = df.copy()
        feature_cols = [c for c in df.columns if c.startswith('feat_')]
        for col in feature_cols:
            df[col] = df[col] + np.random.normal(0, std * df[col].std(), size=len(df))
        return df
    
    return [
        Perturbation(
            'base',
            lambda x: x,
            'No perturbation baseline'
        ),
        Perturbation(
            'spread_1pip',
            lambda x: add_spread(x, 0.0001),
            'Add 1 pip spread effect'
        ),
        Perturbation(
            'slippage_1bps',
            lambda x: add_slippage(x, 0.0001),
            'Add 1 bps slippage effect'
        ),
        Perturbation(
            'feature_noise_1pct',
            lambda x: add_feature_noise(x, 0.01),
            'Add 1% noise to features'
        )
    ]

class DecomposedTester:
    """Main class for running decomposed tests."""
    
    def __init__(
        self,
        regimes: Optional[List[MarketRegime]] = None,
        perturbations: Optional[List[Perturbation]] = None,
        cv_splits: int = 5
    ):
        self.regimes = regimes or create_market_regimes()
        self.perturbations = perturbations or create_perturbations()
        self.cv_splits = cv_splits
        
        # Initialize classifiers for each component
        self.classifiers = {
            'mlp': MLPClassifier(
                hidden_layer_sizes=(10, 5),
                max_iter=1000,
                early_stopping=True
            ),
            'gbc': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            )
        }
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        regime: MarketRegime,
        perturbation: Perturbation
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for a specific regime and perturbation."""
        # Apply perturbation
        df_pert = perturbation.apply_fn(df)
        
        # Identify regime periods
        returns = df_pert['close'].pct_change()
        in_regime = regime.identify(returns)
        
        # Extract features and labels
        feature_cols = [c for c in df.columns if c.startswith('feat_')]
        X = df_pert[feature_cols].values[in_regime]
        y = (df_pert['signal'] > 0).astype(int).values[in_regime]
        
        # Create sample weights (optional)
        w = np.ones(len(X))
        return X, y, w
    
    def evaluate_classifier(
        self,
        clf,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate a classifier using time series CV."""
        cv = TimeSeriesSplit(n_splits=self.cv_splits)
        aucs, precs, recs = [], [], []
        
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = weights[train_idx] if weights is not None else None
            
            # Fit and predict
            clf.fit(X_train, y_train, sample_weight=w_train)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            
            # Compute metrics
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
            
            aucs.append(auc(fpr, tpr))
            precs.append(prec.mean())
            recs.append(rec.mean())
        
        return {
            'auc': np.mean(aucs),
            'precision': np.mean(precs),
            'recall': np.mean(recs)
        }
    
    def run_tests(self, df: pd.DataFrame) -> List[DecomposedTestResult]:
        """Run full decomposed tests across all regimes and perturbations."""
        results = []
        
        for regime in self.regimes:
            for pert in self.perturbations:
                # Prepare data for this combination
                X, y, w = self.prepare_data(df, regime, pert)
                
                if len(X) < regime.min_samples:
                    continue  # Skip if too few samples
                
                # Test each classifier
                for clf_name, clf in self.classifiers.items():
                    metrics = self.evaluate_classifier(clf, X, y, w)
                    
                    result = DecomposedTestResult(
                        regime=f"{regime.name}_{clf_name}",
                        perturbation=pert.name,
                        auc_score=metrics['auc'],
                        precision=metrics['precision'],
                        recall=metrics['recall'],
                        n_samples=len(X)
                    )
                    results.append(result)
        
        return results

def run_decomposed_analysis(
    df: pd.DataFrame,
    cv_splits: int = 5
) -> pd.DataFrame:
    """
    Run complete decomposed analysis and return results as DataFrame.
    
    Args:
        df: Input DataFrame with features and signals
        cv_splits: Number of CV splits for evaluation
    
    Returns:
        DataFrame with test results
    """
    tester = DecomposedTester(cv_splits=cv_splits)
    results = tester.run_tests(df)
    
    # Convert results to DataFrame
    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows)

if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Run decomposed tests')
    parser.add_argument('--csv', required=True, help='Path to features CSV')
    parser.add_argument('--output', default='decomposed_results.csv',
                       help='Output CSV path')
    
    args = parser.parse_args()
    
    # Load data and run analysis
    df = pd.read_csv(args.csv)
    results_df = run_decomposed_analysis(df)
    
    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"Results saved to: {args.output}")
    
    # Print summary
    summary = results_df.groupby(['regime', 'perturbation'])['auc'].mean()
    print("\nSummary of AUC scores by regime and perturbation:")
    print(summary.to_string())