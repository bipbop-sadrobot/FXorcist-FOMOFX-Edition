"""
Models for simulating transaction costs in trading.

It is critical to distinguish between the three primary types of transaction costs:

1.  **Spread:** The difference between the bid and ask price. This is a guaranteed cost
    of execution for any market order and should be handled by the execution engine
    by buying at the 'ask' and selling at the 'bid'.

2.  **Slippage:** The potential for the price to move *after* an order is sent to the
    market but *before* it is filled. This is an unexpected, variable cost. The
    models in this module are designed to simulate this phenomenon.

3.  **Commission:** A fixed or variable fee charged by the broker for executing a
    trade.
"""

import numpy as np
from typing import Dict, Any


class SlippageModel:
    """
    Abstract base class for slippage models.

    It calculates the unfavorable price difference between the intended price
    and the actual fill price due to market movement during order execution.
    """
    def calculate_slippage(self, instrument: str, direction: str, market_context: Dict[str, Any], order_time: datetime = None) -> float:
        """
        Returns the unfavorable slippage amount in absolute price terms.

        Args:
            instrument: The trading instrument's symbol (e.g., 'EUR_USD').
            direction: The direction of the trade ('BUY' or 'SELL').
            market_context: A dictionary containing relevant market data, such as
                            'volatility', 'bid', 'ask'.

        Returns:
            The calculated slippage value. A positive value always represents
            an adverse price change.
        """
        raise NotImplementedError("Should implement calculate_slippage()")


class FixedAbsoluteSlippageModel(SlippageModel):
    """A simple model with a fixed, absolute slippage per trade."""
    def __init__(self, absolute_slippage: float = 0.00002):
        """
        Args:
            absolute_slippage: The fixed amount of adverse price change to apply
                               to each trade (e.g., 0.00002 for a forex pair).
        """
        if absolute_slippage < 0:
            raise ValueError("Absolute slippage cannot be negative.")
        self.absolute_slippage = absolute_slippage

    def calculate_slippage(self, instrument: str, direction: str, market_context: Dict[str, Any] = None) -> float:
        return self.absolute_slippage


class TimeAwareVolatilitySlippageModel(SlippageModel):
    """A realistic model incorporating time delay and market volatility."""
    def __init__(self, volatility_factor: float = 0.1, time_factor: float = 0.5, base_latency_ms: float = 50.0):
        """
        Args:
            volatility_factor: The fraction of current volatility to apply as base slippage
            time_factor: How much to increase slippage per ms of delay
            base_latency_ms: Assumed minimum network/execution latency in milliseconds
        """
        if volatility_factor < 0:
            raise ValueError("Volatility factor cannot be negative.")
        if time_factor < 0:
            raise ValueError("Time factor cannot be negative.")
        if base_latency_ms < 0:
            raise ValueError("Base latency cannot be negative.")
            
        self.volatility_factor = volatility_factor
        self.time_factor = time_factor
        self.base_latency_ms = base_latency_ms

    def calculate_slippage(self, instrument: str, direction: str, market_context: Dict[str, Any], order_time: datetime = None) -> float:
        # Get base volatility component
        volatility = market_context.get('volatility', 0.0)
        if volatility < 0:
            raise ValueError("Volatility data cannot be negative.")
        
        base_slippage = volatility * self.volatility_factor
        
        # Add time-based component if order time provided
        if order_time is not None:
            current_time = datetime.now()
            delay_ms = (current_time - order_time).total_seconds() * 1000
            # Apply minimum latency
            effective_delay = max(delay_ms, self.base_latency_ms)
            time_slippage = (effective_delay / 1000.0) * self.time_factor * volatility
        else:
            time_slippage = (self.base_latency_ms / 1000.0) * self.time_factor * volatility
            
        # Consider market impact based on order size relative to volume
        volume = market_context.get('volume', float('inf'))
        order_size = abs(float(market_context.get('order_units', 0)))
        if volume > 0:
            market_impact = min(1.0, order_size / volume) * base_slippage
        else:
            market_impact = 0.0
            
        return base_slippage + time_slippage + market_impact


class CommissionModel:
    """
    Abstract base class for commission models.

    Calculates the cost of a trade based on broker fees.
    """
    def calculate_commission(self, units: float, fill_price: float) -> float:
        """
        Calculates the commission for a given trade.

        Args:
            units: The number of units traded.
            fill_price: The price at which the trade was executed.

        Returns:
            The total commission cost for the trade.
        """
        raise NotImplementedError("Should implement calculate_commission()")


class OANDACommissionModel(CommissionModel):
    """A realistic model for brokers charging per units traded."""
    def __init__(self, commission_per_100k: float = 5.0):
        """
        Args:
            commission_per_100k: The commission fee per 100,000 units of the
                                 base currency traded (e.g., 5.0 for $5).
        """
        if commission_per_100k < 0:
            raise ValueError("Commission rate cannot be negative.")
        self.commission_per_100k = commission_per_100k

    def calculate_commission(self, units: float, fill_price: float) -> float:
        if units < 0:
            raise ValueError("Trade units cannot be negative.")
        return self.commission_per_100k * (abs(units) / 100000.0)


class PercentageCommissionModel(CommissionModel):
    """A model for brokers that charge a percentage of the total trade value."""
    def __init__(self, percentage_fee: float = 0.001):
        """
        Args:
            percentage_fee: The fee as a decimal (e.g., 0.001 for 0.1%).
        """
        if not 0 <= percentage_fee < 1:
            raise ValueError("Percentage fee must be a decimal between 0 and 1.")
        self.percentage_fee = percentage_fee

    def calculate_commission(self, units: float, fill_price: float) -> float:
        if units < 0 or fill_price < 0:
            raise ValueError("Trade units and fill price cannot be negative.")
        trade_value = abs(units) * fill_price
        return trade_value * self.percentage_fee