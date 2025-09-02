from typing import Dict, Any, List, Tuple
import hmac, hashlib, math, random, time

def _gaussian_noise(std: float) -> float:
    u1 = max(1e-12, random.random())
    u2 = random.random()
    z0 = ( (-2.0*math.log(u1)) ** 0.5 ) * math.cos(2.0*math.pi*u2)
    return z0 * std

class DPAccountant:
    def __init__(self):
        self.epsilon = 0.0
        self.delta = 0.0
        self.rounds = 0

    def add_round(self, eps: float, delta: float):
        # simple composition (advanced composition would be more precise)
        self.epsilon += eps
        self.delta = max(self.delta, delta)
        self.rounds += 1

    def summary(self):
        return {"epsilon": self.epsilon, "delta": self.delta, "rounds": self.rounds}

class FederatedMemory:
    def __init__(self, event_bus, metadata, secret_key: bytes = b"local-secret",
                 clip_norm: float = 1.0, dp_noise_std: float = 0.0, enable_dp: bool = False):
        self.event_bus = event_bus
        self.metadata = metadata
        self.secret_key = secret_key
        self.global_model: Dict[str, float] = {}
        self.clip_norm = clip_norm
        self.dp_noise_std = dp_noise_std
        self.enable_dp = enable_dp
        self.accountant = DPAccountant()

    def _verify(self, payload: bytes, signature: str) -> bool:
        sig = hmac.new(self.secret_key, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, signature)

    def _clip_params(self, params: Dict[str, float]) -> Dict[str, float]:
        clipped = {}
        norm = math.sqrt(sum(v*v for v in params.values())) + 1e-12
        factor = min(1.0, self.clip_norm / norm)
        for k, v in params.items():
            clipped[k] = float(v) * factor
        return clipped

    def aggregate(self, updates: List[Dict[str, Any]], eps: float = 0.1, delta: float = 1e-5):
        acc: Dict[str, float] = {}
        total_w = 0.0
        for up in updates:
            payload = str(up.get("params", {})).encode()
            signature = up.get("signature", "")
            if not self._verify(payload, signature):
                continue
            w = float(up.get("weight", 1.0))
            params = self._clip_params({k: float(v) for k, v in up.get("params", {}).items()})
            for k, v in params.items():
                acc[k] = acc.get(k, 0.0) + w * v
            total_w += w
        if total_w > 0:
            aggregated = {k: v/total_w for k, v in acc.items()}
        else:
            aggregated = self.global_model
        if self.enable_dp and self.dp_noise_std > 0:
            aggregated = {k: v + _gaussian_noise(self.dp_noise_std) for k, v in aggregated.items()}
            # update accountant with simple composition
            self.accountant.add_round(eps, delta)
        self.global_model = aggregated
        return {"model": self.global_model, "accounting": self.accountant.summary()}

    def train_round(self):
        """Simple training round simulation for demo purposes."""
        # Simulate a federated training round with dummy data
        dummy_updates = [
            {
                "params": {"weight1": random.random(), "weight2": random.random()},
                "weight": 1.0,
                "signature": hmac.new(self.secret_key, str({"weight1": 0.5, "weight2": 0.5}).encode(), hashlib.sha256).hexdigest()
            }
        ]
        result = self.aggregate(dummy_updates)
        self.event_bus.publish("federated_round_completed", result)
        return result
