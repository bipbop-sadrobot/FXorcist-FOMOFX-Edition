from typing import List, Dict, Any
import numpy as np
from sklearn.ensemble import IsolationForest
import shap

class AnomalyDetector:
    def __init__(self, memory, n_estimators: int = 100, contamination: float = 0.05):
        self.memory = memory
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)

    def _extract_feature_matrix(self) -> np.ndarray:
        rows: List[List[float]] = []
        memory_entries = self.memory.recall()
        for r in memory_entries:
            feats = r.metadata.get("features") or {}
            # Convert features to a deterministic vector order
            keys = sorted(feats.keys())
            vec = [float(feats[k]) for k in keys] if keys else []
            # Ensure non-empty
            if not vec and ("prediction" in r.metadata):
                vec = [float(r.metadata["prediction"])]
            if vec:
                rows.append(vec)
        if not rows:
            return np.zeros((0, 1))
        # pad to equal length
        max_len = max(len(x) for x in rows)
        padded = [x + [0.0]*(max_len - len(x)) for x in rows]
        return np.array(padded, dtype=float)

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        X = self._extract_feature_matrix()
        if X.shape[0] < 10:
            return []  # need enough samples
        self.model.fit(X)
        scores = self.model.decision_function(X)
        labels = self.model.predict(X)  # -1 anomaly, 1 normal
        anomalies = []
        for i, (lab, sc) in enumerate(zip(labels, scores)):
            if lab == -1:
                anomalies.append({"index": i, "score": float(sc)})

        # Add explainability using SHAP
        if anomalies and X.shape[0] > 0:
            try:
                explainer = shap.Explainer(self.model.decision_function, X)
                shap_values = explainer(X)
                for i, anomaly in enumerate(anomalies):
                    anomaly_index = anomaly["index"]
                    anomaly["shap_values"] = shap_values[anomaly_index].values.tolist()
            except Exception as e:
                print(f"SHAP explainability failed: {e}")

        return anomalies
