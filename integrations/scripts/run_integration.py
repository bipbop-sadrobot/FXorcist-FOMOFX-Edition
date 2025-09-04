#!/usr/bin/env python3
"""
Main integration script for FXorcist ML components.
Demonstrates the full workflow of causal analysis, decomposed testing, and model selection.
"""
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path

from fxorcist_integration.causal.econml_effects import analyze_market_event
from fxorcist_integration.tests.decomposed_tests import run_decomposed_analysis
from fxorcist_integration.models.model_zoo import ModelZoo

def create_example_data(output_path: str, n_samples: int = 1000):
    """Create example forex data for demonstration."""
    np.random.seed(42)
    
    # Generate timestamps
    base_ts = pd.date_range('2025-01-01', periods=n_samples, freq='H')
    
    # Generate synthetic price data
    close = 1.0 + np.random.randn(n_samples).cumsum() * 0.001
    
    # Generate features
    feat_rsi = np.random.normal(50, 10, n_samples)
    feat_ma = close.rolling(20).mean()
    feat_vol = np.abs(np.random.normal(0, 0.001, n_samples))
    
    # Generate event and signal columns
    events = (np.random.rand(n_samples) > 0.95).astype(int)
    signals = np.where(
        feat_rsi > 70,
        1,
        np.where(feat_rsi < 30, -1, 0)
    )
    
    # Create labels (future returns)
    returns = pd.Series(close).pct_change().shift(-1)
    labels = (returns > 0).astype(int)
    
    # Combine into DataFrame
    df = pd.DataFrame({
        'timestamp': base_ts,
        'close': close,
        'feat_rsi': feat_rsi,
        'feat_ma': feat_ma,
        'feat_vol': feat_vol,
        'event': events,
        'signal': signals,
        'label': labels
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Created example data: {output_path}")
    return df

def run_full_analysis(
    data_path: str,
    output_dir: str,
    event_col: str = 'event',
    signal_col: str = 'signal',
    label_col: str = 'label'
):
    """Run complete analysis using all components."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"\nLoaded data from {data_path}")
    print(f"Shape: {df.shape}")
    
    # 1. Causal Analysis
    print("\n1. Running Causal Analysis...")
    causal_results = analyze_market_event(
        df,
        event_col=event_col,
        outcome_col='close',
        horizon=1
    )
    
    with open(output_dir / 'causal_effects.json', 'w') as f:
        json.dump(causal_results['summary'], f, indent=2)
    print("Saved causal effects to: causal_effects.json")
    
    # 2. Decomposed Tests
    print("\n2. Running Decomposed Tests...")
    decomp_results = run_decomposed_analysis(df)
    decomp_results.to_csv(output_dir / 'decomposed_results.csv', index=False)
    print("Saved decomposed test results to: decomposed_results.csv")
    
    # 3. Model Zoo Training
    print("\n3. Training Model Zoo...")
    zoo = ModelZoo()
    model_results = zoo.fit(df, label_col=label_col)
    
    # Save best model
    zoo.save(output_dir / 'best_model.joblib')
    
    with open(output_dir / 'model_results.json', 'w') as f:
        json.dump(model_results, f, indent=2)
    print("Saved model results to: model_results.json")
    
    # Generate example predictions
    predictions = zoo.predict(df)
    probabilities = zoo.predict_proba(df)
    
    results_df = pd.DataFrame({
        'timestamp': df['timestamp'],
        'true_label': df[label_col],
        'predicted': predictions,
        'probability': probabilities[:, 1]
    })
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    print("Saved predictions to: predictions.csv")
    
    # Print summary
    print("\nAnalysis Summary:")
    print("-" * 50)
    print("Causal Effects:")
    print(f"Average Treatment Effect: {causal_results['summary']['ate']:.4f}")
    
    print("\nDecomposed Tests:")
    print(decomp_results.groupby('regime')['auc'].mean())
    
    print("\nBest Model:")
    print(f"Model: {model_results['best_model']}")
    print(f"AUC: {model_results['auc']:.4f}")
    print(f"Accuracy: {model_results['accuracy']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Run FXorcist ML integration')
    parser.add_argument('--data', required=True, help='Path to input CSV or "example" to create demo data')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--event-col', default='event', help='Event column name')
    parser.add_argument('--signal-col', default='signal', help='Signal column name')
    parser.add_argument('--label-col', default='label', help='Label column name')
    
    args = parser.parse_args()
    
    # Create example data if requested
    if args.data.lower() == 'example':
        data_path = 'example_forex_data.csv'
        create_example_data(data_path)
    else:
        data_path = args.data
    
    # Run analysis
    run_full_analysis(
        data_path,
        args.output,
        event_col=args.event_col,
        signal_col=args.signal_col,
        label_col=args.label_col
    )

if __name__ == '__main__':
    main()