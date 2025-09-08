"""
Models realistic transaction costs including market impact.

Problems Solved:
- Ignoring bid-ask spreads
- Not modeling market impact
- Unrealistic fill assumptions
- Missing opportunity costs
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class MarketConditions:
    """Current market conditions affecting transaction costs."""
    volatility: float  # Daily volatility
    volume: float     # Average daily volume
    spread: float     # Current bid-ask spread
    liquidity_score: float  # 0-1 score of current liquidity

@dataclass
class TransactionCostConfig:
    """Configuration for transaction cost modeling."""
    # Commission structure
    fixed_commission: Decimal = Decimal('2.50')  # Fixed fee per trade
    percentage_commission: Decimal = Decimal('0.0001')  # 1bp commission
    min_commission: Decimal = Decimal('1.00')  # Minimum commission
    
    # Market impact parameters
    impact_coefficient: float = 0.1  # Base market impact coefficient
    impact_decay: float = 0.5  # Impact decay factor
    
    # Timing costs
    execution_delay_ms: int = 100  # Simulated latency
    price_drift: float = 0.0  # Expected price drift during execution
    
    # Opportunity costs
    partial_fill_threshold: float = 0.8  # Minimum fill ratio
    opportunity_cost_factor: float = 1.5  # Multiplier for missed trades

class RealisticTransactionCostModel:
    """Models realistic transaction costs including market impact."""
    
    def __init__(self, config: Optional[TransactionCostConfig] = None):
        self.config = config or TransactionCostConfig()
        self.logger = logging.getLogger(__name__)
        
        # Historical cost tracking
        self.cost_history = []
        self.market_impact_history = []
        self.fill_ratios = []
    
    def calculate_total_transaction_cost(
        self,
        order: Dict,
        market_data: Dict,
        market_conditions: Optional[MarketConditions] = None
    ) -> Dict:
        """Calculate comprehensive transaction costs."""
        
        # 1. Commission costs
        commission = self._calculate_commission(order)
        
        # 2. Bid-ask spread cost
        spread_cost = self._calculate_spread_cost(order, market_data)
        
        # 3. Market impact cost
        impact_cost = self._calculate_market_impact(
            order, market_data, market_conditions
        )
        
        # 4. Timing/slippage cost
        timing_cost = self._calculate_timing_cost(
            order, market_data, market_conditions
        )
        
        # 5. Opportunity cost (for partial fills)
        opportunity_cost = self._calculate_opportunity_cost(
            order, market_data, market_conditions
        )
        
        total_cost = (
            commission +
            spread_cost +
            Decimal(str(impact_cost)) +
            Decimal(str(timing_cost)) +
            Decimal(str(opportunity_cost))
        )
        
        # Record costs for analysis
        self._record_costs(order, {
            'commission': float(commission),
            'spread_cost': float(spread_cost),
            'market_impact': impact_cost,
            'timing_cost': timing_cost,
            'opportunity_cost': opportunity_cost,
            'total_cost': float(total_cost)
        })
        
        return {
            'commission': float(commission),
            'spread_cost': float(spread_cost),
            'market_impact': impact_cost,
            'timing_cost': timing_cost,
            'opportunity_cost': opportunity_cost,
            'total_cost': float(total_cost),
            'cost_bps': float(total_cost / Decimal(str(order['notional'])) * 10000)
        }
    
    def _calculate_commission(self, order: Dict) -> Decimal:
        """Calculate commission based on configured structure."""
        notional = Decimal(str(order['notional']))
        
        # Calculate percentage commission
        percentage_fee = notional * self.config.percentage_commission
        
        # Apply fixed commission
        total_commission = percentage_fee + self.config.fixed_commission
        
        # Apply minimum commission
        return max(total_commission, self.config.min_commission)
    
    def _calculate_spread_cost(self, order: Dict, market_data: Dict) -> Decimal:
        """Calculate cost due to bid-ask spread."""
        notional = Decimal(str(order['notional']))
        
        # Use actual spread if available, otherwise estimate
        if 'bid' in market_data and 'ask' in market_data:
            spread = Decimal(str(market_data['ask'])) - Decimal(str(market_data['bid']))
            mid_price = (Decimal(str(market_data['ask'])) + Decimal(str(market_data['bid']))) / 2
            spread_bps = spread / mid_price
        else:
            # Estimate spread based on volatility and volume
            volatility = market_data.get('volatility', 0.001)
            volume = market_data.get('volume', 1000000)
            spread_bps = Decimal(str(self._estimate_spread(volatility, volume)))
        
        # Half spread cost for each side of trade
        return notional * spread_bps / Decimal('2')
    
    def _calculate_market_impact(
        self,
        order: Dict,
        market_data: Dict,
        market_conditions: Optional[MarketConditions]
    ) -> float:
        """Calculate market impact based on order size and liquidity."""
        notional = float(order['notional'])
        avg_volume = market_data.get('volume', 1000000)
        
        # Adjust impact coefficient based on market conditions
        impact_coeff = self.config.impact_coefficient
        if market_conditions:
            # Increase impact in high volatility or low liquidity
            impact_coeff *= (1 + market_conditions.volatility) / market_conditions.liquidity_score
        
        # Participation rate (order size relative to average volume)
        participation_rate = notional / avg_volume
        
        # Square root impact model with decay
        impact = impact_coeff * (participation_rate ** self.config.impact_decay)
        
        # Record impact for analysis
        self.market_impact_history.append({
            'timestamp': order.get('timestamp', datetime.now()),
            'impact_bps': impact * 10000,
            'participation_rate': participation_rate
        })
        
        return notional * impact
    
    def _calculate_timing_cost(
        self,
        order: Dict,
        market_data: Dict,
        market_conditions: Optional[MarketConditions]
    ) -> float:
        """Calculate cost due to execution delay."""
        notional = float(order['notional'])
        
        # Base volatility from market data or conditions
        if market_conditions:
            volatility = market_conditions.volatility
        else:
            volatility = market_data.get('volatility', 0.001)
        
        # Convert delay to fraction of day
        delay_fraction = self.config.execution_delay_ms / (1000 * 60 * 60 * 24)
        
        # Expected cost due to volatility during delay
        volatility_cost = notional * volatility * np.sqrt(delay_fraction)
        
        # Add drift component
        drift_cost = notional * self.config.price_drift * delay_fraction
        
        return volatility_cost + drift_cost
    
    def _calculate_opportunity_cost(
        self,
        order: Dict,
        market_data: Dict,
        market_conditions: Optional[MarketConditions]
    ) -> float:
        """Calculate opportunity cost for partial fills or missed trades."""
        notional = float(order['notional'])
        
        # Estimate fill probability based on size and liquidity
        avg_volume = market_data.get('volume', 1000000)
        participation_rate = notional / avg_volume
        
        if market_conditions:
            liquidity_factor = market_conditions.liquidity_score
        else:
            liquidity_factor = 1.0
        
        # Probability of complete fill decreases with size and poor liquidity
        fill_probability = np.exp(-participation_rate / liquidity_factor)
        expected_fill_ratio = min(1.0, fill_probability)
        
        # Record fill ratio for analysis
        self.fill_ratios.append(expected_fill_ratio)
        
        # Calculate opportunity cost if fill ratio below threshold
        if expected_fill_ratio < self.config.partial_fill_threshold:
            unfilled_ratio = 1.0 - expected_fill_ratio
            opportunity_cost = (
                notional *
                unfilled_ratio *
                self.config.opportunity_cost_factor *
                market_data.get('volatility', 0.001)
            )
        else:
            opportunity_cost = 0.0
        
        return opportunity_cost
    
    def _estimate_spread(self, volatility: float, volume: float) -> float:
        """Estimate bid-ask spread based on volatility and volume."""
        # Simple spread model based on volatility and inverse square root of volume
        base_spread = 0.0001  # 1 bp minimum spread
        vol_component = volatility * 0.1
        liquidity_component = 1 / np.sqrt(volume) * 0.0001
        
        return base_spread + vol_component + liquidity_component
    
    def _record_costs(self, order: Dict, costs: Dict):
        """Record transaction costs for analysis."""
        self.cost_history.append({
            'timestamp': order.get('timestamp', datetime.now()),
            'notional': order['notional'],
            **costs
        })
    
    def analyze_costs(self, period: Optional[timedelta] = None) -> Dict:
        """Analyze transaction cost components over time."""
        if not self.cost_history:
            return {}
        
        # Convert to DataFrame for analysis
        costs_df = pd.DataFrame(self.cost_history)
        
        # Filter by period if specified
        if period:
            cutoff = datetime.now() - period
            costs_df = costs_df[costs_df['timestamp'] >= cutoff]
        
        # Calculate cost statistics
        total_notional = costs_df['notional'].sum()
        cost_components = {
            'commission': costs_df['commission'].sum(),
            'spread_cost': costs_df['spread_cost'].sum(),
            'market_impact': costs_df['market_impact'].sum(),
            'timing_cost': costs_df['timing_cost'].sum(),
            'opportunity_cost': costs_df['opportunity_cost'].sum()
        }
        
        # Calculate cost ratios
        cost_ratios = {
            k: v / total_notional * 10000  # Convert to bps
            for k, v in cost_components.items()
        }
        
        # Analyze market impact
        impact_df = pd.DataFrame(self.market_impact_history)
        if not impact_df.empty:
            impact_analysis = {
                'avg_impact_bps': impact_df['impact_bps'].mean(),
                'max_impact_bps': impact_df['impact_bps'].max(),
                'avg_participation': impact_df['participation_rate'].mean()
            }
        else:
            impact_analysis = {}
        
        # Analyze fill ratios
        if self.fill_ratios:
            fill_analysis = {
                'avg_fill_ratio': np.mean(self.fill_ratios),
                'min_fill_ratio': min(self.fill_ratios),
                'complete_fill_rate': sum(r >= 0.99 for r in self.fill_ratios) / len(self.fill_ratios)
            }
        else:
            fill_analysis = {}
        
        return {
            'total_costs': cost_components,
            'cost_ratios_bps': cost_ratios,
            'market_impact': impact_analysis,
            'fill_analysis': fill_analysis,
            'total_cost_bps': sum(cost_ratios.values()),
            'total_notional': total_notional
        }