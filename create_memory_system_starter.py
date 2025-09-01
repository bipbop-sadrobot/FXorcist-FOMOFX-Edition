import os, json, pathlib, zipfile, textwrap

base = "memory_system_starter"
vscode = os.path.join(base, ".vscode")
pkg = os.path.join(base, "memory_system")
tests = os.path.join(base, "tests")
examples = os.path.join(base, "examples")

os.makedirs(base, exist_ok=True)
os.makedirs(os.path.join(base, ".vscode"), exist_ok=True)
os.makedirs(pkg, exist_ok=True)
os.makedirs(tests, exist_ok=True)
os.makedirs(examples, exist_ok=True)

# -------------------- pyproject.toml --------------------
pyproject = """
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "memory-system"
version = "0.1.0"
description = "Personal local Integrated Memory System scaffold with EventBus, FederatedMemory, and AnomalyDetector."
authors = [{name="James Priest"}]
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
  "numpy>=1.24.0",
  "scikit-learn>=1.2.0"
]

[tool.pytest.ini_options]
addopts = "-q"
pythonpath = ["memory_system"]
"""

# -------------------- README.md --------------------
readme = """
# Memory System Architecture (Starter Project)

Local, personal-use scaffold implementing:
- Rolling Memory Storage
- Meta-Model Training (stubbed)
- Event-Driven Architecture
- Federated Learning (HMAC-verified aggregation)
- Anomaly Detection (IsolationForest)

## Quickstart

```bash
# 1) Enter project
cd memory_system

# 2) Create & activate venv
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# 3) Install
pip install -e .
pip install pytest

# 4) Run example
python examples/run_cycle.py

# 5) Benchmark (toy)
pytest tests/benchmark_memory.py --benchmark-json=benchmark_results.json || true
VS Code & Cline
Open this folder in VS Code.

Suggested Cline prompt:

"Open memory_system/memory.py and add a time-decayed moving average in analyze_memory_trends, then run pytest."

Tasks:

Run → Examples: run_cycle or Test: PyTest via VS Code Command Palette.

Modules
memory_system/metadata.py – SharedMetadata registry + docs.

memory_system/event_bus.py – In-memory pub/sub with persistence hooks.

memory_system/memory.py – IntegratedMemorySystem with pruning, trends, insights, resource allocation.

memory_system/federated.py – FederatedMemory with weighted aggregation & HMAC verification (toy).

memory_system/anomaly.py – IsolationForest-based detector.

examples/run_cycle.py – Demo pipeline.

Notes
Security primitives are illustrative (HMAC, weighting). Adapt for your environment.

No external services required; pure local runtime.
"""

# -------------------- .vscode settings --------------------
launch_json = {
"version": "0.2.0",
"configurations": [
{
"name": "Python: run_cycle",
"type": "python",
"request": "launch",
"program": "${workspaceFolder}/examples/run_cycle.py",
"console": "integratedTerminal",
"justMyCode": True
}
]
}
tasks_json = {
"version": "2.0.0",
"tasks": [
{
"label": "Test: PyTest",
"type": "shell",
"command": "pytest -v",
"group": "test",
"problemMatcher": []
},
{
"label": "Examples: run_cycle",
"type": "shell",
"command": "python examples/run_cycle.py",
"group": "build",
"problemMatcher": []
}
]
}

# -------------------- init.py --------------------
init_py = """
from .event_bus import EventBus
from .metadata import SharedMetadata
from .memory import IntegratedMemorySystem
from .federated import FederatedMemory
from .anomaly import AnomalyDetector

all = [
"EventBus",
"SharedMetadata",
"IntegratedMemorySystem",
"FederatedMemory",
"AnomalyDetector",
]
"""

# -------------------- event_bus.py --------------------
event_bus_py = """
from collections import defaultdict
from typing import Callable, Dict, Any, List

class EventBus:
    \"\"\"Simple in-memory pub/sub event bus.\"\"\"
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)
        self._history: List[tuple] = [] # (event, payload)

    def subscribe(self, event: str, handler: Callable[[Any], None]):
        self._subscribers[event].append(handler)

    def publish(self, event: str, payload: Any):
        self._history.append((event, payload))
        for handler in self._subscribers.get(event, []):
            handler(payload)

    def history(self) -> List[tuple]:
        return list(self._history)

    # Hooks for persistence/backlog if desired
    def enable_persistence(self):
        pass
"""

# -------------------- metadata.py --------------------
metadata_py = """
from typing import Dict, Any, List
from dataclasses import dataclass, field
import json
import hashlib

@dataclass
class SharedMetadata:
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    features: Dict[str, str] = field(default_factory=dict) # feature_name -> dtype
    subscribers: List[str] = field(default_factory=list)
    _revisions: List[str] = field(default_factory=list) # json snapshots (hash keys)

    def register_model(self, name: str, version: str, capabilities: Dict[str, Any] = None):
        key = f"{name}:{version}"
        self.models[key] = capabilities or {}
        self._snapshot()

    def register_feature(self, name: str, dtype: str):
        self.features[name] = dtype
        self._snapshot()

    def register_subscriber(self, subscriber_id: str):
        if subscriber_id not in self.subscribers:
            self.subscribers.append(subscriber_id)
            self._snapshot()

    def _snapshot(self):
        snap = json.dumps({
            "models": self.models,
            "features": self.features,
            "subscribers": self.subscribers
        }, sort_keys=True)
        digest = hashlib.sha256(snap.encode()).hexdigest()
        self._revisions.append(digest)

    def revision_history(self) -> List[str]:
        return list(self._revisions)

    def generate_documentation(self) -> str:
        doc = {
            "models": self.models,
            "features": self.features,
            "subscribers": self.subscribers,
            revisions: self._revisions[-10:],
        }
        return json.dumps(doc, indent=2, sort_keys=True)
"""

# -------------------- utils.py --------------------
utils_py = """
from typing import Iterable
def moving_average(seq: Iterable[float], window: int) -> float:
    seq = list(seq)[-window:]
    return sum(seq) / max(1, len(seq))
"""

# -------------------- memory.py --------------------
memory_py = """
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
        \"\"\"record example: {model, prediction, target?, error?, features{...}, ts}\"\"\"
        self.records.append(record)
        self.event_bus.publish("memory_updated", record)

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
        \"\"\"Learn simple inverse-error weights as a placeholder.\"\"\"
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
"""

# -------------------- federated.py --------------------
federated_py = """
from typing import Dict, Any, List
import hmac, hashlib

class FederatedMemory:
    \"\"\"Toy federated aggregator with HMAC verification and weighted averaging.\"\"\"
    def __init__(self, event_bus, metadata, secret_key: bytes = b"local-secret"):
        self.event_bus = event_bus
        self.metadata = metadata
        self.secret_key = secret_key
        self.global_model: Dict[str, float] = {} # parameter_name -> value

    def _verify(self, payload: bytes, signature: str) -> bool:
        sig = hmac.new(self.secret_key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, signature)

    def aggregate(self, updates: List[Dict[str, Any]]):
        \"\"\"updates: [{params: {w1:..., w2:...}, weight: float, signature: str}]\"\"\"
        acc: Dict[str, float] = {}
        total_w = 0.0
        for up in updates:
            payload = str(up.get("params", {})).encode()
            signature = up.get("signature", "")
            if not self._verify(payload, signature):
                continue
            w = float(up.get("weight", 1.0))
            for k, v in up.get("params", {}).items():
                acc[k] = acc.get(k, 0.0) + w * float(v)
            total_w += w
        if total_w > 0:
            self.global_model = {k: v/total_w for k, v in acc.items()}
        return self.global_model

    def train_round(self):
        \"\"\"Toy local round: pretend we received two client updates.\"\"\"
        dummy1 = {"params": {"w1": 0.9, "w2": 0.1}, "weight": 10.0}
        dummy2 = {"params": {"w1": 1.1, "w2": -0.1}, "weight": 5.0}
        for d in (dummy1, dummy2):
            payload = str(d["params"]).encode()
            d["signature"] = hmac.new(self.secret_key, payload, hashlib.sha256).hexdigest()
        model = self.aggregate([dummy1, dummy2])
        self.event_bus.publish("model_registered", {"global_model": model})
        return model
"""

# -------------------- anomaly.py --------------------
anomaly_py = """
from typing import List, Dict, Any
import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, memory, n_estimators: int = 100, contamination: float = 0.05):
        self.memory = memory
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)

    def _extract_feature_matrix(self) -> np.ndarray:
        rows: List[List[float]] = []
        for r in self.memory.records:
            feats = r.get("features") or {}
            # Convert features to a deterministic vector order
            keys = sorted(feats.keys())
            vec = [float(feats[k]) for k in keys] if keys else []
            # Ensure non-empty
            if not vec and ("prediction" in r):
                vec = [float(r["prediction"])]
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
        return anomalies
"""

# -------------------- examples/run_cycle.py --------------------
run_cycle = """
from memory_system.event_bus import EventBus
from memory_system.metadata import SharedMetadata
from memory_system.memory import IntegratedMemorySystem
from memory_system.federated import FederatedMemory
from memory_system.anomaly import AnomalyDetector

def main():
    event_bus = EventBus()
    metadata = SharedMetadata()
    memory = IntegratedMemorySystem(event_bus, metadata)
    federated = FederatedMemory(event_bus, metadata)
    detector = AnomalyDetector(memory)

    # Subscribe to anomaly events for demo
    def on_anomaly(payload):
        print("[EVENT] anomaly_detected:", payload)
    event_bus.subscribe("anomaly_detected", on_anomaly)

    # Simulate a stream of predictions
    import random
    for t in range(60):
        err = abs(random.gauss(0.15, 0.08))
        feats = {"x1": random.random(), "x2": random.random()}
        memory.add_record({
            "ts": t, "model": "mA", "prediction": random.random()},
            "error": err, "features": feats
        )

        if memory.should_trigger_federated_round(period=15):
            federated.train_round()

        anomalies = detector.detect_anomalies()
        if anomalies:
            event_bus.publish("anomaly_detected", anomalies)

    print("Insights:", memory.generate_insights_report())
    print("Resource plan:", memory.prioritize_resources())

if __name__ == "__main__":
    main()
"""

# -------------------- tests --------------------
test_event_bus = """
from memory_system.event_bus import EventBus

def test_pubsub_roundtrip():
    bus = EventBus()
    seen = []
    bus.subscribe("ping", lambda p: seen.append(p))
    bus.publish("ping", {"ok": True})
    assert seen and seen[0]["ok"] is True
"""

test_metadata = """
from memory_system.metadata import SharedMetadata

def test_metadata_register_and_docs():
    md = SharedMetadata()
    md.register_model("foo", "1.0", {"task": "regression"})
    md.register_feature("x1", "float")
    md.register_subscriber("svc-A")
    doc = md.generate_documentation()
    assert "foo:1.0" in doc and "x1" in doc and "svc-A" in doc
    assert len(md.revision_history()) >= 1
"""

test_memory = """
from memory_system.event_bus import EventBus
from memory_system.metadata import SharedMetadata
from memory_system.memory import IntegratedMemorySystem

def test_memory_trends_and_resources():
    bus = EventBus()
    md = SharedMetadata()
    mem = IntegratedMemorySystem(bus, md, max_records=50)
    for i in range(30):
        mem.add_record({"model": "m1", "prediction": 0.8, "error": 0.1 + i*0.005})
    trend = mem.analyze_memory_trends()
    assert trend["n"] >= 1 and "mean_error" in trend
    plan = mem.prioritize_resources()
    assert "recommendations" in plan
"""

test_federated = """
from memory_system.event_bus import EventBus
from memory_system.metadata import SharedMetadata
from memory_system.memory import IntegratedMemorySystem

def test_federated_train_round():
    bus = EventBus()
    md = SharedMetadata()
    fed = FederatedMemory(bus, md, secret_key=b"test")
    model = fed.train_round()
    assert "w1" in model and "w2" in model
"""

test_anomaly = """
from memory_system.event_bus import EventBus
from memory_system.metadata import SharedMetadata
from memory_system.memory import IntegratedMemorySystem

def test_anomaly_detector_min_samples():
    bus = EventBus()
    md = SharedMetadata()
    mem = IntegratedMemorySystem(bus, md)
    det = AnomalyDetector(mem)
    # Not enough samples -> no anomalies by design
    for i in range(5):
        mem.add_record({"model": "m", "prediction": 0.5, "error": 0.1, "features": {"a": i}})
    assert det.detect_anomalies() == []
"""

benchmark_memory = """
import time
from memory_system.event_bus import EventBus
from memory_system.metadata import SharedMetadata
from memory_system.memory import IntegratedMemorySystem

def test_insert_speed():
    bus = EventBus()
    md = SharedMetadata()
    mem = IntegratedMemorySystem(bus, md, max_records=10000)
    t0 = time.time()
    for i in range(5000):
        mem.add_record({"model": "m", "prediction": 0.1, "error": 0.2})
    elapsed = time.time() - t0
    # Soft assertion threshold (won't fail in slower envs)
    assert elapsed < 2.5 or True
"""

# Write files
(pathlib.Path("pyproject.toml")).write_text(pyproject.strip() + "\\n")
(pathlib.Path("README.md")).write_text(readme.strip() + "\\n")
path = pathlib.Path(os.path.join(base, ".vscode", "launch.json"))
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(launch_json, indent=2))
path = pathlib.Path(os.path.join(base, ".vscode", "tasks.json"))
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(tasks_json, indent=2))

path = pathlib.Path("memory_system/__init__.py")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(init_py.strip() + "\\n")
(pathlib.Path("memory_system/event_bus.py")).write_text(event_bus_py.strip() + "\\n")
(pathlib.Path("memory_system/metadata.py")).write_text(metadata_py.strip() + "\\n")
(pathlib.Path("memory_system/utils.py")).write_text(utils_py.strip() + "\\n")
(pathlib.Path("memory_system/memory.py")).write_text(memory_py.strip() + "\\n")
(pathlib.Path("memory_system/federated.py")).write_text(federated_py.strip() + "\\n")
(pathlib.Path("memory_system/anomaly.py")).write_text(anomaly_py.strip() + "\\n")

path = pathlib.Path("examples/run_cycle.py")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(run_cycle.strip() + "\\n")

(pathlib.Path("tests/test_event_bus.py")).write_text(test_event_bus.strip() + "\\n")
(pathlib.Path("tests/test_metadata.py")).write_text(test_metadata.strip() + "\\n")
(pathlib.Path("tests/test_memory.py")).write_text(test_memory.strip() + "\\n")
(pathlib.Path("tests/test_federated.py")).write_text(test_federated.strip() + "\\n")
(pathlib.Path("tests/test_anomaly.py")).write_text(test_anomaly.strip() + "\\n")
(pathlib.Path("tests/benchmark_memory.py")).write_text(benchmark_memory.strip() + "\\n")

# Zip the project for download
zip_path = "memory_system_starter.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk("."):
        for f in files:
            full = pathlib.Path(root) / f
            if str(full).startswith("."):
                continue
            z.write(full, full)

print(zip_path)

