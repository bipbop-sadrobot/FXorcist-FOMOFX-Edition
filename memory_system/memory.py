from collections import deque
from typing import Dict, Any, List, Optional
import statistics
from .event_bus import EventBus
from .metadata import SharedMetadata
from .utils import moving_average

class IntegratedMemorySystem:
    def __init__(self, event_bus: EventBus, metadata: SharedMetadata, max_records: int = 2000):
        self.event_bus = event_bus
        self.metadata = metadata
        self.records = deque(maxlen=max_records)
        self.meta_model_state: Dict[str, Any] = {"weights": {}}
        self._federated_tick = 0

    # ------------- Storage -------------
    def add_record(self, record: Dict[str, Any]):
        """record example: {model, prediction, target?, error?, features{...}, ts}"""
        self.records.append(record)
        self.event_bus.publish("memory_updated", record)

    def recall(self, query: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Recall records from memory, optionally filtered by query."""
        if not self.records:
            return []

        # Convert deque to list for easier handling
        records_list = list(self.records)

        if query is None:
            # Return most recent records
            return records_list[-top_k:] if len(records_list) > top_k else records_list

        # Simple text-based filtering
        filtered = []
        for record in records_list:
            # Search in various text fields
            searchable_text = ""
            if "model" in record:
                searchable_text += str(record["model"]) + " "
            if "features" in record and isinstance(record["features"], dict):
                searchable_text += " ".join(str(v) for v in record["features"].values())

            if query.lower() in searchable_text.lower():
                filtered.append(record)

        return filtered[-top_k:] if len(filtered) > top_k else filtered

    # ------------- Analysis -------------
    def analyze_memory_trends(self, window: int = 100) -> Dict[str, Any]:
        if not self.records:
            return {"trend": "no_data"}
        errors = [r.get("error") for r in self.records if r.get("error") is not None]
        if not errors:
            return {"trend": "no_errors"}
        mean_err = statistics.fmean(errors)
        stdev_err = statistics.pstdev(errors) if len(errors) > 1 else 0.0
        ma_err = moving_average(errors, min(window, len(errors)))
        return {
            "mean_error": mean_err,
            "stdev_error": stdev_err,
            "ma_error": ma_err,
            "n": len(errors)
        }

    def generate_insights_report(self) -> Dict[str, Any]:
        trend = self.analyze_memory_trends()
        by_model: Dict[str, List[float]] = {}
        for r in self.records:
            if "model" in r and "error" in r and r["error"] is not None:
                by_model.setdefault(r["model"], []).append(r["error"])
        model_summary = {m: {"mean_error": statistics.fmean(v)} for m, v in by_model.items() if v}
        return {
            "records": len(self.records),
            "trend": trend,
            "model_summary": model_summary,
        }

    # ------------- Meta-model (stub) -------------
    def train_meta_model(self):
        """Learn simple inverse-error weights as a placeholder."""
        summary = self.generate_insights_report()["model_summary"]
        if not summary:
            return self.meta_model_state
        weights = {}
        total = 0.0
        for m, s in summary.items():
            w = 1.0 / max(1e-6, s["mean_error"])
            weights[m] = w
            total += w
        if total > 0:
            for m in list(weights.keys()):
                weights[m] /= total
        self.meta_model_state["weights"] = weights
        return self.meta_model_state

    # ------------- Resource Prioritization -------------
    def prioritize_resources(self) -> Dict[str, Any]:
        trend = self.analyze_memory_trends()
        mean_err = trend.get("mean_error", 0.0) or 0.0
        ma_err = trend.get("ma_error", 0.0) or 0.0

        critical = mean_err > 0.2 or ma_err > 0.25
        retrain_freq = "high" if critical else "normal"
        sampling = "recent_bias" if critical else "uniform"
        return {
            "critical": critical,
            "recommendations": {
                "retrain_frequency": retrain_freq,
                "data_sampling": sampling,
                "meta_model": self.meta_model_state,
            }
        }

    # ------------- Federated trigger heuristic -------------
    def should_trigger_federated_round(self, period: int = 10) -> bool:
        self._federated_tick += 1
        return self._federated_tick % period == 0