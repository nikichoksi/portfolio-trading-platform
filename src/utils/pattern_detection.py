"""
Technical chart pattern detection.
Identifies common trading patterns like head and shoulders, triangles, flags, etc.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.signal import argrelextrema
from dataclasses import dataclass


@dataclass
class PatternSignal:
    """Pattern detection signal"""
    pattern_name: str
    pattern_type: str  # 'bullish', 'bearish', 'continuation', 'reversal'
    confidence: float  # 0-1
    start_idx: int
    end_idx: int
    description: str
    marker_positions: List[Tuple[int, float]]  # List of (index, price) for markers


class PatternDetector:
    """Detects technical chart patterns in price data"""

    def __init__(self, window: int = 5):
        """
        Initialize pattern detector.

        Args:
            window: Window size for local extrema detection
        """
        self.window = window

    def detect_all_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """
        Detect all patterns in price data.

        Args:
            df: DataFrame with OHLC data

        Returns:
            List of detected patterns
        """
        patterns = []

        if len(df) < 20:  # Need minimum data
            return patterns

        # Detect various patterns
        patterns.extend(self.detect_head_and_shoulders(df))
        patterns.extend(self.detect_double_top_bottom(df))
        patterns.extend(self.detect_triangles(df))
        patterns.extend(self.detect_flags_pennants(df))
        patterns.extend(self.detect_cup_and_handle(df))
        patterns.extend(self.detect_wedges(df))

        return patterns

    def find_peaks_valleys(self, prices: pd.Series, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find local peaks and valleys in price series"""
        peaks = argrelextrema(prices.values, np.greater, order=order)[0]
        valleys = argrelextrema(prices.values, np.less, order=order)[0]
        return peaks, valleys

    def detect_head_and_shoulders(self, df: pd.DataFrame) -> List[PatternSignal]:
        """
        Detect Head and Shoulders pattern.
        Bearish reversal pattern with three peaks.
        """
        patterns = []
        prices = df['Close']
        peaks, _ = self.find_peaks_valleys(prices)

        if len(peaks) < 3:
            return patterns

        # Look for three consecutive peaks where middle is highest
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]

            left_price = prices.iloc[left_shoulder]
            head_price = prices.iloc[head]
            right_price = prices.iloc[right_shoulder]

            # Head should be higher than shoulders
            if head_price > left_price and head_price > right_price:
                # Shoulders should be roughly equal (within 5%)
                if abs(left_price - right_price) / left_price < 0.05:
                    confidence = 1 - (abs(left_price - right_price) / left_price)

                    patterns.append(PatternSignal(
                        pattern_name="Head and Shoulders",
                        pattern_type="bearish",
                        confidence=confidence,
                        start_idx=left_shoulder,
                        end_idx=right_shoulder,
                        description="Bearish reversal pattern indicating potential downtrend",
                        marker_positions=[
                            (left_shoulder, left_price),
                            (head, head_price),
                            (right_shoulder, right_price)
                        ]
                    ))

        return patterns

    def detect_double_top_bottom(self, df: pd.DataFrame) -> List[PatternSignal]:
        """
        Detect Double Top (bearish) and Double Bottom (bullish) patterns.
        """
        patterns = []
        prices = df['Close']
        peaks, valleys = self.find_peaks_valleys(prices)

        # Double Top
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                peak1 = peaks[i]
                peak2 = peaks[i + 1]

                price1 = prices.iloc[peak1]
                price2 = prices.iloc[peak2]

                # Peaks should be close in price (within 3%)
                if abs(price1 - price2) / price1 < 0.03:
                    confidence = 1 - (abs(price1 - price2) / price1)

                    patterns.append(PatternSignal(
                        pattern_name="Double Top",
                        pattern_type="bearish",
                        confidence=confidence,
                        start_idx=peak1,
                        end_idx=peak2,
                        description="Bearish reversal pattern with two peaks at similar levels",
                        marker_positions=[(peak1, price1), (peak2, price2)]
                    ))

        # Double Bottom
        if len(valleys) >= 2:
            for i in range(len(valleys) - 1):
                valley1 = valleys[i]
                valley2 = valleys[i + 1]

                price1 = prices.iloc[valley1]
                price2 = prices.iloc[valley2]

                # Valleys should be close in price (within 3%)
                if abs(price1 - price2) / price1 < 0.03:
                    confidence = 1 - (abs(price1 - price2) / price1)

                    patterns.append(PatternSignal(
                        pattern_name="Double Bottom",
                        pattern_type="bullish",
                        confidence=confidence,
                        start_idx=valley1,
                        end_idx=valley2,
                        description="Bullish reversal pattern with two bottoms at similar levels",
                        marker_positions=[(valley1, price1), (valley2, price2)]
                    ))

        return patterns

    def detect_triangles(self, df: pd.DataFrame) -> List[PatternSignal]:
        """
        Detect triangle patterns (ascending, descending, symmetrical).
        """
        patterns = []
        prices = df['Close']

        if len(prices) < 30:
            return patterns

        # Use last 30 periods for triangle detection
        recent = prices.iloc[-30:]
        peaks, valleys = self.find_peaks_valleys(recent, order=3)

        if len(peaks) >= 2 and len(valleys) >= 2:
            # Calculate trendlines
            peak_prices = recent.iloc[peaks]
            valley_prices = recent.iloc[valleys]

            # Ascending Triangle: flat top, rising bottom
            peak_trend = np.polyfit(peaks, peak_prices.values, 1)[0]
            valley_trend = np.polyfit(valleys, valley_prices.values, 1)[0]

            if abs(peak_trend) < 0.1 and valley_trend > 0.1:
                patterns.append(PatternSignal(
                    pattern_name="Ascending Triangle",
                    pattern_type="bullish",
                    confidence=0.7,
                    start_idx=len(df) - 30,
                    end_idx=len(df) - 1,
                    description="Bullish continuation pattern with flat resistance and rising support",
                    marker_positions=[(len(df) - 30 + peaks[0], peak_prices.iloc[0]),
                                     (len(df) - 30 + valleys[-1], valley_prices.iloc[-1])]
                ))

            # Descending Triangle: falling top, flat bottom
            elif peak_trend < -0.1 and abs(valley_trend) < 0.1:
                patterns.append(PatternSignal(
                    pattern_name="Descending Triangle",
                    pattern_type="bearish",
                    confidence=0.7,
                    start_idx=len(df) - 30,
                    end_idx=len(df) - 1,
                    description="Bearish continuation pattern with falling resistance and flat support",
                    marker_positions=[(len(df) - 30 + peaks[0], peak_prices.iloc[0]),
                                     (len(df) - 30 + valleys[0], valley_prices.iloc[0])]
                ))

            # Symmetrical Triangle: converging trendlines
            elif peak_trend < -0.05 and valley_trend > 0.05:
                patterns.append(PatternSignal(
                    pattern_name="Symmetrical Triangle",
                    pattern_type="continuation",
                    confidence=0.65,
                    start_idx=len(df) - 30,
                    end_idx=len(df) - 1,
                    description="Continuation pattern with converging trendlines",
                    marker_positions=[(len(df) - 30 + peaks[0], peak_prices.iloc[0]),
                                     (len(df) - 30 + valleys[0], valley_prices.iloc[0])]
                ))

        return patterns

    def detect_flags_pennants(self, df: pd.DataFrame) -> List[PatternSignal]:
        """
        Detect flag and pennant patterns (continuation patterns).
        """
        patterns = []
        prices = df['Close']

        if len(prices) < 20:
            return patterns

        # Look for strong move followed by consolidation
        for i in range(10, len(prices) - 10):
            # Check for strong upward move (flagpole)
            pole_start = prices.iloc[i - 10]
            pole_end = prices.iloc[i]
            pole_move = (pole_end - pole_start) / pole_start

            if pole_move > 0.05:  # 5% move
                # Check for consolidation (flag)
                consolidation = prices.iloc[i:i + 10]
                if len(consolidation) == 10:
                    volatility = consolidation.std() / consolidation.mean()

                    if volatility < 0.02:  # Low volatility consolidation
                        patterns.append(PatternSignal(
                            pattern_name="Bull Flag",
                            pattern_type="bullish",
                            confidence=0.75,
                            start_idx=i - 10,
                            end_idx=i + 10,
                            description="Bullish continuation pattern after strong upward move",
                            marker_positions=[(i - 10, pole_start), (i, pole_end)]
                        ))

        return patterns

    def detect_cup_and_handle(self, df: pd.DataFrame) -> List[PatternSignal]:
        """
        Detect cup and handle pattern (bullish continuation).
        """
        patterns = []
        prices = df['Close']

        if len(prices) < 40:
            return patterns

        # Look for U-shaped recovery (cup) followed by small pullback (handle)
        for i in range(20, len(prices) - 20):
            left = prices.iloc[i - 20:i - 10].mean()
            bottom = prices.iloc[i - 5:i + 5].min()
            right = prices.iloc[i + 10:i + 15].mean()

            # Cup: U-shaped pattern
            if abs(left - right) / left < 0.05:  # Rims at similar height
                drop = (left - bottom) / left
                if 0.10 < drop < 0.30:  # 10-30% drop
                    # Check for handle (small pullback)
                    if i + 20 < len(prices):
                        handle = prices.iloc[i + 15:i + 20]
                        if len(handle) > 0 and handle.iloc[-1] < right * 0.95:
                            patterns.append(PatternSignal(
                                pattern_name="Cup and Handle",
                                pattern_type="bullish",
                                confidence=0.8,
                                start_idx=i - 20,
                                end_idx=i + 20,
                                description="Bullish continuation pattern with U-shaped cup and handle pullback",
                                marker_positions=[(i - 20, left), (i, bottom), (i + 15, right)]
                            ))

        return patterns

    def detect_wedges(self, df: pd.DataFrame) -> List[PatternSignal]:
        """
        Detect rising and falling wedge patterns.
        """
        patterns = []
        prices = df['Close']

        if len(prices) < 25:
            return patterns

        recent = prices.iloc[-25:]
        peaks, valleys = self.find_peaks_valleys(recent, order=3)

        if len(peaks) >= 2 and len(valleys) >= 2:
            peak_prices = recent.iloc[peaks]
            valley_prices = recent.iloc[valleys]

            peak_trend = np.polyfit(peaks, peak_prices.values, 1)[0]
            valley_trend = np.polyfit(valleys, valley_prices.values, 1)[0]

            # Rising Wedge: both lines rising, converging (bearish)
            if peak_trend > 0 and valley_trend > 0 and valley_trend > peak_trend:
                patterns.append(PatternSignal(
                    pattern_name="Rising Wedge",
                    pattern_type="bearish",
                    confidence=0.7,
                    start_idx=len(df) - 25,
                    end_idx=len(df) - 1,
                    description="Bearish reversal pattern with upward converging trendlines",
                    marker_positions=[(len(df) - 25 + peaks[0], peak_prices.iloc[0]),
                                     (len(df) - 25 + valleys[-1], valley_prices.iloc[-1])]
                ))

            # Falling Wedge: both lines falling, converging (bullish)
            elif peak_trend < 0 and valley_trend < 0 and valley_trend < peak_trend:
                patterns.append(PatternSignal(
                    pattern_name="Falling Wedge",
                    pattern_type="bullish",
                    confidence=0.7,
                    start_idx=len(df) - 25,
                    end_idx=len(df) - 1,
                    description="Bullish reversal pattern with downward converging trendlines",
                    marker_positions=[(len(df) - 25 + peaks[0], peak_prices.iloc[0]),
                                     (len(df) - 25 + valleys[0], valley_prices.iloc[0])]
                ))

        return patterns


def get_pattern_marker_style(pattern_type: str) -> Dict:
    """Get marker style for pattern type"""
    styles = {
        'bullish': {
            'color': 'green',
            'symbol': 'triangle-up',
            'size': 15
        },
        'bearish': {
            'color': 'red',
            'symbol': 'triangle-down',
            'size': 15
        },
        'continuation': {
            'color': 'blue',
            'symbol': 'diamond',
            'size': 12
        },
        'reversal': {
            'color': 'orange',
            'symbol': 'star',
            'size': 14
        }
    }
    return styles.get(pattern_type, {'color': 'gray', 'symbol': 'circle', 'size': 10})
