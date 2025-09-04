# run_model_zoo.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
from integrations.features.mlfinlab_tools import purged_time_series_split

def get_model_zoo() -> Dict:
    """Get dictionary of candidate models to evaluate"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    
    return {
        "rf": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "gbm": GradientBoostingClassifier(n_estimators=100),
        "logistic": LogisticRegression(max_iter=1000),
        "xgb": xgb.XGBClassifier(n_estimators=100, n_jobs=-1)
    }

def evaluate_models(X: pd.DataFrame,
                   y: pd.Series,
                   models: Dict,
                   cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Dict]:
    """
    Evaluate multiple models using purged CV
    Returns dict of metrics for each model
    """
    results = {}
    
    for name, model in models.items():
        cv_scores = []
        for train_ix, test_ix in cv_splits:
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X.iloc[train_ix])
            X_test = scaler.transform(X.iloc[test_ix])
            
            y_train = y.iloc[train_ix]
            y_test = y.iloc[test_ix]
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred)
            }
            cv_scores.append(metrics)
        
        # Average CV scores
        results[name] = {
            metric: np.mean([s[metric] for s in cv_scores])
            for metric in cv_scores[0].keys()
        }
    
    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with features and labels")
    ap.add_argument("--label", required=True, help="label column name")
    ap.add_argument("--save", default="integrations/artifacts/best_model.joblib")
    ap.add_argument("--report", default="integrations/artifacts/model_zoo_report.csv")
    args = ap.parse_args()
    
    # Load data
    df = pd.read_csv(args.csv, parse_dates=True, index_col=0)
    
    # Separate features and label
    y = df[args.label]
    X = df.drop(columns=[args.label])
    
    # Get CV splits
    cv_splits = purged_time_series_split(df.index, n_splits=5, purge=1)
    
    # Get and evaluate models
    models = get_model_zoo()
    results = evaluate_models(X, y, models, cv_splits)
    
    # Save results
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).T.to_csv(args.report)
    
    # Find and save best model
    best_model = max(results.items(), key=lambda x: x[1]["roc_auc"])[0]
    print(f"\nBest model: {best_model}")
    
    # Retrain best model on full dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    models[best_model].fit(X_scaled, y)
    
    # Save model and scaler
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": models[best_model],
        "scaler": scaler,
        "features": list(X.columns)
    }, args.save)
    
    print("\nModel Zoo Results:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:10} - ROC AUC: {metrics['roc_auc']:6.3f}, Accuracy: {metrics['accuracy']:6.3f}")