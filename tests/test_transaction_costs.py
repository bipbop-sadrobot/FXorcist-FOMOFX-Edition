"""
Tests for transaction costs models and execution handling.

These tests verify that:
1. Slippage models calculate correctly
2. Commission models work as expected
3. Execution handler properly applies all costs
4. Edge cases and errors are handled appropriately
"""

import pytest
from datetime import datetime
import queue
from unittest.mock import Mock, MagicMock

from fxorcist.models.transaction_costs import (
    SlippageModel,
    FixedAbsoluteSlippageModel,
    VolatilitySlippageModel,
    CommissionModel,
    OANDACommissionModel,
    PercentageCommissionModel
)
from fxorcist.pipeline.execution_handler import (
    ExecutionHandler,
    OrderEvent,
    FillEvent
)


class TestSlippageModels:
    """Tests for slippage model implementations."""
    
    def test_fixed_slippage(self):
        """Test FixedAbsoluteSlippageModel calculations."""
        model = FixedAbsoluteSlippageModel(absolute_slippage=0.0001)
        slippage = model.calculate_slippage('EUR_USD', 'BUY', {})
        assert slippage == 0.0001
        
        # Test negative slippage validation
        with pytest.raises(ValueError):
            FixedAbsoluteSlippageModel(absolute_slippage=-0.0001)
    
    def test_volatility_slippage(self):
        """Test VolatilitySlippageModel calculations."""
        model = VolatilitySlippageModel(volatility_factor=0.1)
        
        # Test with valid volatility
        market_context = {'volatility': 0.002}
        slippage = model.calculate_slippage('EUR_USD', 'BUY', market_context)
        assert slippage == pytest.approx(0.0002)
        
        # Test with missing volatility
        assert model.calculate_slippage('EUR_USD', 'BUY', {}) == 0.0
        
        # Test negative volatility validation
        with pytest.raises(ValueError):
            model.calculate_slippage('EUR_USD', 'BUY', {'volatility': -0.001})
        
        # Test negative factor validation
        with pytest.raises(ValueError):
            VolatilitySlippageModel(volatility_factor=-0.1)


class TestCommissionModels:
    """Tests for commission model implementations."""
    
    def test_oanda_commission(self):
        """Test OANDACommissionModel calculations."""
        model = OANDACommissionModel(commission_per_100k=5.0)
        
        # Test standard case
        commission = model.calculate_commission(10000, 1.2000)
        assert commission == pytest.approx(0.5)  # 5.0 * (10000/100000)
        
        # Test larger position
        commission = model.calculate_commission(200000, 1.2000)
        assert commission == pytest.approx(10.0)  # 5.0 * (200000/100000)
        
        # Test validation
        with pytest.raises(ValueError):
            model.calculate_commission(-10000, 1.2000)
        with pytest.raises(ValueError):
            OANDACommissionModel(commission_per_100k=-5.0)
    
    def test_percentage_commission(self):
        """Test PercentageCommissionModel calculations."""
        model = PercentageCommissionModel(percentage_fee=0.001)  # 0.1%
        
        # Test standard case
        commission = model.calculate_commission(10000, 1.2000)
        assert commission == pytest.approx(12.0)  # 0.001 * 10000 * 1.2000
        
        # Test validation
        with pytest.raises(ValueError):
            model.calculate_commission(-10000, 1.2000)
        with pytest.raises(ValueError):
            model.calculate_commission(10000, -1.2000)
        with pytest.raises(ValueError):
            PercentageCommissionModel(percentage_fee=1.5)  # >100%


class TestExecutionHandler:
    """Tests for execution handler implementation."""
    
    @pytest.fixture
    def mock_data_handler(self):
        """Create a mock data handler for testing."""
        handler = MagicMock()
        handler.get_current_price.return_value = {
            'bid': 1.2000,
            'ask': 1.2002
        }
        handler.get_current_volatility.return_value = 0.0002
        handler.get_current_volume.return_value = 1000000
        return handler
    
    @pytest.fixture
    def execution_handler(self, mock_data_handler):
        """Create an execution handler with mock components."""
        event_queue = queue.Queue()
        return ExecutionHandler(
            data_handler=mock_data_handler,
            event_queue=event_queue,
            slippage_model=FixedAbsoluteSlippageModel(0.0001),
            commission_model=OANDACommissionModel(5.0)
        )
    
    def test_market_buy_execution(self, execution_handler):
        """Test execution of market buy order."""
        order = OrderEvent(
            timestamp=datetime.now(),
            instrument='EUR_USD',
            units=10000,
            direction='BUY'
        )
        
        execution_handler.execute_order(order)
        
        # Get the fill event from the queue
        fill_event = execution_handler.event_queue.get_nowait()
        
        # Verify fill price includes spread and slippage
        # Ask price (1.2002) + slippage (0.0001) = 1.2003
        assert fill_event.fill_price == pytest.approx(1.2003)
        
        # Verify commission calculation
        # 5.0 * (10000/100000) = 0.5
        assert fill_event.commission == pytest.approx(0.5)
        
        # Verify slippage tracking
        assert fill_event.slippage == 0.0001
    
    def test_market_sell_execution(self, execution_handler):
        """Test execution of market sell order."""
        order = OrderEvent(
            timestamp=datetime.now(),
            instrument='EUR_USD',
            units=-10000,  # Negative for sell
            direction='SELL'
        )
        
        execution_handler.execute_order(order)
        
        # Get the fill event from the queue
        fill_event = execution_handler.event_queue.get_nowait()
        
        # Verify fill price includes spread and slippage
        # Bid price (1.2000) - slippage (0.0001) = 1.1999
        assert fill_event.fill_price == pytest.approx(1.1999)
        
        # Verify commission calculation
        assert fill_event.commission == pytest.approx(0.5)
        
        # Verify slippage tracking
        assert fill_event.slippage == 0.0001
    
    def test_unsupported_order_type(self, execution_handler):
        """Test handling of unsupported order types."""
        order = OrderEvent(
            timestamp=datetime.now(),
            instrument='EUR_USD',
            units=10000,
            direction='BUY',
            order_type='LIMIT'  # Unsupported type
        )
        
        with pytest.raises(ValueError, match="Unsupported order type"):
            execution_handler.execute_order(order)


if __name__ == '__main__':
    pytest.main([__file__])