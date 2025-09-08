from abc import ABC, abstractmethod
from typing import Dict, Any

class ExecutionModel(ABC):
    """Abstract base for slippage/commission models."""

    @abstractmethod
    def execute_order(
        self,
        order: Dict[str, Any],
        market_snapshot: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return fill with price, commission, slippage."""
        pass

class ZeroSlippage(ExecutionModel):
    def execute_order(self, order, market_snapshot):
        mid_price = market_snapshot["mid"]
        commission = 0.0
        return {
            "price": mid_price,
            "commission": commission,
            "slippage": 0.0,
            "filled": True
        }

class SimpleSlippage(ExecutionModel):
    def __init__(self, commission_pct: float = 0.00002, slippage_bps: float = 0.5):
        self.commission_pct = commission_pct
        self.slippage_bps = slippage_bps / 10000  # bps to decimal

    def execute_order(self, order, market_snapshot):
        mid_price = market_snapshot["mid"]
        spread = market_snapshot["ask"] - market_snapshot["bid"]
        sign = 1 if order["side"] > 0 else -1

        # Slippage = half spread + fixed bps
        slippage = (spread / 2) + (mid_price * self.slippage_bps * abs(sign))
        exec_price = mid_price + (sign * slippage)

        # Commission = notional * pct
        notional = abs(order["size"]) * exec_price
        commission = notional * self.commission_pct

        return {
            "price": exec_price,
            "commission": commission,
            "slippage": slippage,
            "filled": True
        }

class ImpactModel(ExecutionModel):
    def __init__(self, commission_pct: float = 0.00002, impact_coeff: float = 0.1):
        self.commission_pct = commission_pct
        self.impact_coeff = impact_coeff  # price impact per unit volume

    def execute_order(self, order, market_snapshot):
        mid_price = market_snapshot["mid"]
        avg_volume = market_snapshot.get("avg_volume", 1e6)  # default
        sign = 1 if order["side"] > 0 else -1

        # Price impact proportional to size/volume
        impact = self.impact_coeff * (abs(order["size"]) / avg_volume)
        exec_price = mid_price + (sign * mid_price * impact)

        notional = abs(order["size"]) * exec_price
        commission = notional * self.commission_pct

        return {
            "price": exec_price,
            "commission": commission,
            "slippage": impact * mid_price,
            "filled": True
        }