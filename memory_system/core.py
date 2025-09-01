from typing import Dict, Any, List, Optional
import time, statistics, threading
from dataclasses import dataclass, field
import json
import shap
import numpy as np

# Extension point: embedding hook should be a callable(vectorize_text: Callable[[str], List[float]])
EMBED_HOOK = None

@dataclass
class MemoryEntry:
    id: str
    text: str
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tier: str = "wm"  # stm, wm, ltm, em, sm
    ts: float = field(default_factory=lambda: time.time())

class MemoryManager:
    def __init__(self, vector_index=None, store=None, max_stm=1000, enable_embedding_hook=True):
        self.vector_index = vector_index
        self.store = store
        self.max_stm = max_stm
        self.tiers = {"stm": [], "wm": [], "ltm": [], "em": [], "sm": []}
        self._lock = threading.RLock()
        self._consolidation_worker = None
        self.enable_embedding_hook = enable_embedding_hook

    def set_embedding_hook(self, fn):
        global EMBED_HOOK
        EMBED_HOOK = fn

    def store_entry(self, entry: MemoryEntry):
        with self._lock:
            # run embedding hook if enabled and no vector present
            if self.enable_embedding_hook and entry.vector is None and EMBED_HOOK is not None:
                try:
                    entry.vector = EMBED_HOOK(entry.text)
                except Exception:
                    entry.vector = None
            self.tiers.setdefault(entry.tier, [])
            self.tiers[entry.tier].insert(0, entry)
            if self.store:
                self.store.save_entry(entry)
            if entry.vector is not None and self.vector_index is not None:
                self.vector_index.add(id=entry.id, vector=entry.vector, metadata=entry.metadata)
            if entry.tier == "stm":
                self._enforce_stm()

    def _enforce_stm(self):
        stm = self.tiers.get("stm", [])
        if len(stm) > self.max_stm:
            while len(stm) > self.max_stm:
                e = stm.pop()
                e.tier = "ltm"
                self.tiers["ltm"].insert(0, e)
                if self.store:
                    self.store.save_entry(e)

    def recall(self, query: str = None, top_k: int = 5, vector: Optional[List[float]] = None, tier_filters: Optional[List[str]] = None) -> List[MemoryEntry]:
        results = []
        tiers = tier_filters or ["wm","stm","em","ltm","sm"]
        with self._lock:
            for t in tiers:
                for e in self.tiers.get(t, []):
                    if query is None or query.lower() in e.text.lower():
                        results.append((t, 0.0, e))
            if vector is not None and self.vector_index is not None:
                vec_results = self.vector_index.search(vector, top_k=top_k)
                for r in vec_results:
                    eid = r["id"]
                    found = None
                    for t in tiers:
                        for e in self.tiers.get(t, []):
                            if e.id == eid:
                                found = e; break
                        if found: break
                    if found:
                        results.append((found.tier, r["score"], found))
            results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
            seen = set(); out = []
            for t, sc, e in results_sorted:
                if e.id in seen: continue
                seen.add(e.id)
                out.append(e)
                if len(out) >= top_k: break
            return out

    def update(self, id: str, **changes) -> Optional[MemoryEntry]:
        with self._lock:
            for tier, arr in self.tiers.items():
                for e in arr:
                    if e.id == id:
                        if "text" in changes:
                            e.text = changes["text"]
                        if "metadata" in changes:
                            e.metadata.update(changes["metadata"])
                        if "vector" in changes:
                            e.vector = changes["vector"]
                            if self.vector_index:
                                self.vector_index.update(id, e.vector, e.metadata)
                        if self.store:
                            self.store.save_entry(e)
                        return e
            return None

    def forget(self, id: Optional[str] = None, tier: Optional[str] = None, older_than: Optional[float] = None) -> int:
        removed = 0
        with self._lock:
            if id is not None:
                for t, arr in self.tiers.items():
                    before = len(arr)
                    arr[:] = [e for e in arr if e.id != id]
                    removed += before - len(arr)
                if self.vector_index:
                    try: self.vector_index.remove(id)
                    except Exception: pass
                if self.store:
                    try: self.store.delete_entry(id)
                    except Exception: pass
                return removed
            if tier is not None:
                before = len(self.tiers.get(tier, []))
                if older_than is None:
                    self.tiers[tier] = []
                else:
                    self.tiers[tier] = [e for e in self.tiers.get(tier, []) if e.ts >= older_than]
                removed += before - len(self.tiers[tier])
                return removed
            if older_than is not None:
                for t in list(self.tiers.keys()):
                    before = len(self.tiers[t])
                    self.tiers[t] = [e for e in self.tiers[t] if e.ts >= older_than]
                    removed += before - len(self.tiers[t])
                return removed
            return removed

    def summarize(self, tier: str = "wm", timeframe_seconds: Optional[int] = None) -> str:
        entries = self.tiers.get(tier, [])
        if timeframe_seconds is not None:
            cutoff = time.time() - timeframe_seconds
            entries = [e for e in entries if e.ts >= cutoff]
        texts = [e.text for e in entries[:200]]
        if not texts:
            return ""
        joined = "\\n".join(texts)
        words = [w.strip(".,!?").lower() for w in joined.split() if len(w) > 2]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:20]
        summary = {"count": len(entries), "top_words": top, "snippet": joined[:2000]}
        return json.dumps(summary, indent=2)

    # Consolidation worker (background): demote STM to LTM and deduplicate by text signature
    def start_consolidation_worker(self, interval_seconds: int = 60):
        if self._consolidation_worker and self._consolidation_worker.is_alive():
            return
        self._consolidation_worker = threading.Thread(target=self._consolidation_loop, args=(interval_seconds,), daemon=True)
        self._consolidation_worker.start()

    def _consolidation_loop(self, interval_seconds: int):
        while True:
            try:
                self._consolidate_once()
            except Exception:
                pass
            time.sleep(interval_seconds)

    def _consolidate_once(self):
        """Consolidate memory tiers and manage lifecycle."""
        with self._lock:
            # Process STM entries
            stm = list(self.tiers.get("stm", []))
            seen_text = set()
            metrics = {'duplicates': 0, 'promoted': 0, 'quarantined': 0}
            
            for e in reversed(stm):  # oldest first
                key = (e.text or "").strip().lower()
                
                # Check for duplicates
                if key in seen_text:
                    metrics['duplicates'] += 1
                    self.forget(id=e.id)
                    continue
                
                seen_text.add(key)
                age = time.time() - e.ts
                
                # Check for anomalies
                if e.metadata.get('confidence', 1.0) < 0.3 or e.metadata.get('error', 0) > 0.1:
                    e.metadata['quarantined'] = True
                    e.metadata['quarantine_reason'] = 'Low confidence or high error'
                    metrics['quarantined'] += 1
                    if self.store:
                        self.store.save_entry(e)
                    continue
                
                # Promote to LTM if old enough
                if age > 3600:  # 1 hour threshold
                    e.tier = "ltm"
                    self.tiers["ltm"].insert(0, e)
                    metrics['promoted'] += 1
                    try:
                        self.tiers["stm"].remove(e)
                    except ValueError:
                        pass
                    if self.store:
                        self.store.save_entry(e)
            
            # Manage LTM size
            if len(self.tiers.get("ltm", [])) > self.max_stm * 2:
                self.tiers["ltm"] = self.tiers["ltm"][:self.max_stm * 2]
            
            return metrics

def analyze_memory_trends(
    memory_data: List[tuple[float, float]],
    leak_threshold: float = 0.01,
    window: int = 5,
    output_format: str = "dict"
) -> Dict[str, Any]:
    """
    Analyze memory usage trends and system health metrics.
    
    Args:
        memory_data (List[Tuple[float, float]]): List of (timestamp, memory_usage) tuples.
            - timestamp: float (epoch or sequential index)
            - memory_usage: float (e.g., MB)
        leak_threshold (float): Minimum slope for memory leak warning
        window (int): Window size for trend analysis
        output_format (str): Output format ("dict" or "json")
        
    Returns:
        Dict[str, Any] or str: Report containing:
            - Memory usage statistics
            - Trend analysis and slopes
            - Health indicators
            - Quarantine metrics
            - SHAP explanations (when available)
            
    Raises:
        ValueError: If memory_data is empty or improperly formatted
    """
    if not memory_data or not all(isinstance(x, (list, tuple)) and len(x) == 2 for x in memory_data):
        raise ValueError("memory_data must be a list of (timestamp, usage) tuples.")

    timestamps, usages = zip(*memory_data)

    # Core metrics
    avg_usage = statistics.mean(usages)
    min_usage = min(usages)
    max_usage = max(usages)

    # Overall trend slope (simple linear regression)
    n = len(usages)
    mean_x, mean_y = statistics.mean(timestamps), statistics.mean(usages)
    num = sum((timestamps[i] - mean_x) * (usages[i] - mean_y) for i in range(n))
    den = sum((timestamps[i] - mean_x) ** 2 for i in range(n)) or 1e-9
    slope = num / den

    # Rolling slopes for leak/trend detection
    rolling_slopes = []
    for i in range(len(usages) - window):
        xw, yw = timestamps[i:i + window], usages[i:i + window]
        mx, my = statistics.mean(xw), statistics.mean(yw)
        num = sum((xw[j] - mx) * (yw[j] - my) for j in range(window))
        den = sum((xw[j] - mx) ** 2 for j in range(window)) or 1e-9
        rolling_slopes.append(num / den)

    sustained_upward = all(s > leak_threshold for s in rolling_slopes[-3:]) if rolling_slopes else False
    sustained_downward = all(s < -leak_threshold for s in rolling_slopes[-3:]) if rolling_slopes else False

    report = {
        "average_usage": avg_usage,
        "min_usage": min_usage,
        "max_usage": max_usage,
        "overall_trend": "upward" if slope > leak_threshold else "downward" if slope < -leak_threshold else "stable",
        "trend_slope": slope,
        "potential_memory_leak": sustained_upward,
        "sustained_downward_trend": sustained_downward,
        "datapoints_analyzed": n,
    }

    if output_format == "json":
        return json.dumps(report, indent=2)

    # Add explainability using SHAP
    try:
        X = np.array([timestamps, usages])
        def f(x):
            return slope * x + (mean_y - slope * mean_x)
        explainer = shap.Explainer(f, X)
        shap_values = explainer(X)
        report["shap_values"] = shap_values.values.tolist()
    except Exception as e:
        print(f"SHAP explainability failed: {e}")

    return report


def generate_insights_report(
    trend_report: Dict[str, Any],
    include_metrics: bool = True,
    include_health: bool = True
) -> str:
    """
    Generate comprehensive memory system insights report.

    Args:
        trend_report: Output from analyze_memory_trends()
        include_metrics: Include detailed metrics
        include_health: Include health indicators

    Returns:
        str: JSON report with metrics, health status, and recommendations
    """
    if not isinstance(trend_report, dict):
        raise ValueError("trend_report must be a dictionary.")

    report = {
        "average_usage": f"{trend_report['average_usage']:.2f} MB",
        "min_usage": f"{trend_report['min_usage']:.2f} MB",
        "max_usage": f"{trend_report['max_usage']:.2f} MB",
        "overall_trend": f"{trend_report['overall_trend']} (slope={trend_report['trend_slope']:.4f})",
        "potential_memory_leak": "Yes 🚨" if trend_report['potential_memory_leak'] else "No",
        "sustained_downward_trend": "Yes 📉" if trend_report['sustained_downward_trend'] else "No",
        "datapoints_analyzed": trend_report['datapoints_analyzed'],
        "recommendations": []
    }

    if trend_report["potential_memory_leak"]:
        report["recommendations"].append("Investigate for unbounded object growth or missing cleanup.")
    elif trend_report["overall_trend"] == "upward":
        report["recommendations"].append("Monitor closely: upward trend may signal early-stage leak.")
    elif trend_report["sustained_downward_trend"]:
        report["recommendations"].append("System memory usage is reducing — check for over-aggressive cleanup.")
    else:
        report["recommendations"].append("Memory usage stable. No immediate actions required.")

    # Add SHAP explanations if available
    if include_metrics and "shap_values" in trend_report:
        report["explanations"] = {
            "shap_values": trend_report["shap_values"],
            "interpretation": "SHAP values show feature importance for trend analysis"
        }
    
    # Add health indicators
    if include_health:
        report["health_status"] = {
            "memory_health": "critical" if trend_report["potential_memory_leak"] else 
                           "warning" if trend_report["overall_trend"] == "upward" else 
                           "good",
            "last_checked": datetime.now().isoformat()
        }
    
    return json.dumps(report, indent=2)
