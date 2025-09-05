"""
Safe Aroon Indicator Implementation
Independent of ta library updates or deprecations.
"""

import pandas as pd


class SafeAroonIndicator:
    """
    A stable implementation of the Aroon indicator that is independent of ta library.
    
    The Aroon indicator measures the time between highs and lows and the 
    magnitude of the trend over a time period.
    """
    
    def __init__(self, high: pd.Series, low: pd.Series, window: int = 25):
        """
        Initialize the SafeAroonIndicator.

        Args:
            high: Series of high prices
            low: Series of low prices
            window: Look-back period for calculations (default: 25)
        """
        self.high = high
        self.low = low
        self.window = window

    def aroon_up(self) -> pd.Series:
        """
        Calculate Aroon Up.
        
        Returns:
            Series containing Aroon Up values
        """
        rolling_high = (
            self.high.rolling(self.window, min_periods=1)
            .apply(lambda x: x.argmax(), raw=True)
        )
        return 100 * (self.window - (self.window - rolling_high)) / self.window

    def aroon_down(self) -> pd.Series:
        """
        Calculate Aroon Down.
        
        Returns:
            Series containing Aroon Down values
        """
        rolling_low = (
            self.low.rolling(self.window, min_periods=1)
            .apply(lambda x: x.argmin(), raw=True)
        )
        return 100 * (self.window - (self.window - rolling_low)) / self.window
    
    def aroon_indicator(self) -> pd.Series:
        """
        Calculate Aroon Indicator (Aroon Up - Aroon Down).
        
        Returns:
            Series containing Aroon Indicator values
        """
        return self.aroon_up() - self.aroon_down()