from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
import shap
from datetime import datetime, timezone
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Optional import for regime change detection
try:
    from ruptures import Binseg
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("Warning: ruptures package not available. Regime change detection will be disabled.")

class AnomalyDetector:
    def __init__(self, memory, n_estimators: int = 100, contamination: float = 0.05):
        self.memory = memory
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        if RUPTURES_AVAILABLE:
            self.regime_detector = Binseg(model="l2")  # L2 cost function for financial data
        else:
            self.regime_detector = None
        self.flash_crash_threshold = -4.0  # Standard deviations
        self.regime_change_threshold = 0.05  # p-value threshold
        
    def normalize_timezone(self, ts: float) -> float:
        """Convert timestamp to UTC."""
        dt = datetime.fromtimestamp(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    def detect_flash_crash(self, prices: np.ndarray) -> List[Dict[str, Any]]:
        """Detect flash crash patterns in price series."""
        if len(prices) < 2:
            return []
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Z-score of returns
        z_scores = stats.zscore(returns)
        
        # Detect extreme movements
        crashes = []
        for i, z in enumerate(z_scores):
            if z < self.flash_crash_threshold:
                # Check for V-shaped recovery
                if i + 5 < len(returns):
                    recovery = returns[i+1:i+6].mean()
                    if recovery > abs(returns[i]) * 0.5:  # 50% recovery threshold
                        crashes.append({
                            "index": i,
                            "severity": float(z),
                            "recovery_rate": float(recovery),
                            "timestamp": self.normalize_timezone(prices[i])
                        })
        return crashes

    def detect_regime_change(self, data: np.ndarray, min_size: int = 20) -> List[Dict[str, Any]]:
        """Detect regime changes in the time series."""
        if len(data) < min_size * 2 or self.regime_detector is None:
            return []

        # Detect change points
        change_points = self.regime_detector.fit(data).predict(n_bkps=3)
        
        regimes = []
        for cp in change_points:
            if cp < len(data) - min_size:
                # Validate regime change with statistical test
                before = data[max(0, cp-min_size):cp]
                after = data[cp:min(len(data), cp+min_size)]
                
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(before, after)
                
                if p_value < self.regime_change_threshold:
                    regimes.append({
                        "change_point": cp,
                        "confidence": float(1 - p_value),
                        "before_mean": float(before.mean()),
                        "after_mean": float(after.mean()),
                        "ks_statistic": float(ks_stat)
                    })
        return regimes

    def _extract_feature_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Extract and validate feature matrix with proper padding handling."""
        rows: List[List[float]] = []
        feature_names: List[str] = []
        timestamps: List[float] = []
        
        memory_entries = self.memory.recall()
        
        for r in memory_entries:
            # Handle both dict format and object format
            if hasattr(r, 'metadata'):
                feats = r.metadata.get("features") or {}
                ts = r.metadata.get("timestamp")
            else:
                # Handle dict format from IntegratedMemorySystem
                feats = r.get("features") or {}
                ts = r.get("ts") or r.get("timestamp")
            
            if ts:
                ts = self.normalize_timezone(float(ts))
            
            # Extract and validate features
            keys = sorted(feats.keys())
            vec = []
            names = []
            
            for k in keys:
                try:
                    v = float(feats[k])
                    # Validate value
                    if not np.isfinite(v):
                        continue
                    vec.append(v)
                    names.append(k)
                except (ValueError, TypeError):
                    continue
                    
            # Add derived features
            if ts:
                vec.append(np.sin(2 * np.pi * (ts % 86400) / 86400))  # Time of day
                vec.append(np.sin(2 * np.pi * (ts % 604800) / 604800))  # Day of week
                names.extend(["time_of_day", "day_of_week"])
            
            if vec:
                rows.append(vec)
                if not feature_names:
                    feature_names = names
                timestamps.append(ts if ts else 0)
        
        if not rows:
            return np.zeros((0, 1)), []
            
        # Intelligent padding based on feature type
        max_len = max(len(x) for x in rows)
        padded = []
        
        for row in rows:
            if len(row) < max_len:
                # Use feature-specific padding
                pad_row = []
                for i, val in enumerate(row):
                    pad_row.append(val)
                
                # Pad remaining features intelligently
                for i in range(len(row), max_len):
                    fname = feature_names[i] if i < len(feature_names) else ""
                    if "time" in fname.lower():
                        pad_row.append(0.0)  # Neutral for cyclical features
                    else:
                        pad_row.append(np.mean(row))  # Use mean for other features
                        
                padded.append(pad_row)
            else:
                padded.append(row)
                
        return np.array(padded, dtype=float), feature_names

    def detect_anomalies(self) -> Dict[str, Any]:
        """Enhanced anomaly detection with multiple detection types."""
        X, feature_names = self._extract_feature_matrix()
        if X.shape[0] < 10:
            return {"anomalies": [], "flash_crashes": [], "regime_changes": []}
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Core anomaly detection
        self.model.fit(X_scaled)
        scores = self.model.decision_function(X_scaled)
        labels = self.model.predict(X_scaled)  # -1 anomaly, 1 normal
        
        anomalies = []
        for i, (lab, sc) in enumerate(zip(labels, scores)):
            if lab == -1:
                anomaly = {
                    "index": i,
                    "score": float(sc),
                    "features": {
                        name: float(X[i, j]) 
                        for j, name in enumerate(feature_names)
                    }
                }
                anomalies.append(anomaly)

        # Add SHAP explainability
        if anomalies and X_scaled.shape[0] > 0:
            try:
                explainer = shap.Explainer(self.model.decision_function, X_scaled)
                shap_values = explainer(X_scaled)
                for i, anomaly in enumerate(anomalies):
                    anomaly_index = anomaly["index"]
                    anomaly["feature_importance"] = {
                        name: float(abs(shap_values[anomaly_index].values[j]))
                        for j, name in enumerate(feature_names)
                    }
            except Exception as e:
                print(f"SHAP explainability failed: {e}")
        
        # Detect flash crashes if price data available
        flash_crashes = []
        if "price" in feature_names:
            price_idx = feature_names.index("price")
            prices = X[:, price_idx]
            flash_crashes = self.detect_flash_crash(prices)
        
        # Detect regime changes
        regime_changes = self.detect_regime_change(X_scaled.mean(axis=1))
        
        return {
            "anomalies": anomalies,
            "flash_crashes": flash_crashes,
            "regime_changes": regime_changes,
            "metadata": {
                "feature_names": feature_names,
                "samples_analyzed": X.shape[0],
                "anomaly_rate": len(anomalies) / X.shape[0] if X.shape[0] > 0 else 0
            }
        }
