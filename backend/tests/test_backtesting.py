"""
Unit tests for backtesting logic

Tests that Python implementation matches JavaScript behavior
"""
import pytest
import math
from app.ml.uniswap_v3_adapter import (
    active_liquidity_for_candle,
    tokens_for_strategy,
    get_tick_from_price,
    round_to_nearest_tick
)
from app.core.backtesting import calc_fees, pivot_fee_data, backtest_indicators, calc_unbounded_fees


class TestLiquidityMath:
    """Test liquidity math functions"""

    def test_active_liquidity_for_candle_in_range(self):
        """Test active liquidity calculation when candle is inside position"""
        # Position range: ticks 0 to 1000
        # Candle range: ticks 200 to 800 (completely inside position)
        # Overlap = 600, Candle width = 600
        # 100% of candle's range is inside position → 100% active
        result = active_liquidity_for_candle(0, 1000, 200, 800)
        assert abs(result - 100.0) < 0.01, f"Expected 100%, got {result}%"

    def test_active_liquidity_for_candle_out_of_range(self):
        """Test active liquidity when price is out of range"""
        # Position range: ticks 0 to 1000
        # Candle range: ticks 1100 to 1200 (completely outside)
        result = active_liquidity_for_candle(0, 1000, 1100, 1200)
        assert result == 0, f"Expected 0%, got {result}%"

    def test_active_liquidity_for_candle_full_range(self):
        """Test active liquidity when candle covers full position"""
        # Position range: ticks 200 to 800 (width = 600)
        # Candle range: ticks 0 to 1000 (width = 1000, covers full position)
        # Overlap = 600, Candle width = 1000
        # 60% of candle's range is inside position → 60% active
        result = active_liquidity_for_candle(200, 800, 0, 1000)
        assert abs(result - 60.0) < 0.01, f"Expected 60%, got {result}%"

    def test_tokens_for_strategy_below_range(self):
        """Test token calculation when price is below range"""
        # Price below range - should be all token1
        # decimal = token1_decimals - token0_decimals = 6 - 18 = -12
        tokens = tokens_for_strategy(
            min_range=1500,
            max_range=2000,
            investment=10000,
            price=1000,
            decimal=-12
        )
        assert tokens[0] == 0, "Token0 should be 0 when price below range"
        assert tokens[1] > 0, "Token1 should be > 0 when price below range"

    def test_tokens_for_strategy_in_range(self):
        """Test token calculation when price is in range"""
        # Price in range - should have both tokens
        tokens = tokens_for_strategy(
            min_range=1500,
            max_range=2000,
            investment=10000,
            price=1750,
            decimal=-12
        )
        assert tokens[0] > 0, "Token0 should be > 0 when price in range"
        assert tokens[1] > 0, "Token1 should be > 0 when price in range"
        # Total value should equal investment
        total_value = tokens[0] + tokens[1] * 1750
        assert abs(total_value - 10000) < 1, f"Total value should be ~10000, got {total_value}"

    def test_tokens_for_strategy_above_range(self):
        """Test token calculation when price is above range"""
        # Price above range - should be all token0
        tokens = tokens_for_strategy(
            min_range=1500,
            max_range=2000,
            investment=10000,
            price=2500,
            decimal=-12
        )
        assert tokens[0] > 0, "Token0 should be > 0 when price above range"
        assert tokens[1] == 0, "Token1 should be 0 when price above range"

    def test_calc_unbounded_fees(self):
        """Test fee calculation from feeGrowthGlobal"""
        pool = {
            'token0': {'decimals': 18},
            'token1': {'decimals': 6}
        }

        # Simulate fee growth (Q128 format)
        current_fee0 = str(int(1.5 * (2 ** 128)))  # 1.5 in Q128
        prev_fee0 = str(int(1.0 * (2 ** 128)))     # 1.0 in Q128
        current_fee1 = str(int(0.8 * (2 ** 128)))
        prev_fee1 = str(int(0.5 * (2 ** 128)))

        fg0, fg1 = calc_unbounded_fees(current_fee0, prev_fee0, current_fee1, prev_fee1, pool)

        # Fee growth should be approximately 0.5 and 0.3 (adjusted for decimals)
        assert fg0 > 0, "Fee growth for token0 should be positive"
        assert fg1 > 0, "Fee growth for token1 should be positive"

    def test_round_to_nearest_tick(self):
        """Test tick rounding for fee tier"""
        # Fee tier 3000 (0.3%) has tick spacing of 60
        price = 1234.56
        rounded = round_to_nearest_tick(price, 3000, 18, 6)

        # Price should be rounded to valid tick
        assert isinstance(rounded, float)
        assert rounded > 0


class TestBacktesting:
    """Test backtesting functions"""

    def test_calc_fees_basic(self):
        """Test basic fee calculation with minimal data"""
        pool = {
            'token0': {'decimals': 18},
            'token1': {'decimals': 6}
        }

        # Minimal test data (2 hours)
        data = [
            {
                'periodStartUnix': 1700000000,
                'close': 1500.0,
                'low': 1480.0,
                'high': 1520.0,
                'feeGrowthGlobal0X128': str(int(1.0 * (2 ** 128))),
                'feeGrowthGlobal1X128': str(int(0.5 * (2 ** 128))),
                'pool': {
                    'totalValueLockedUSD': '1000000',
                    'totalValueLockedToken0': '500',
                    'totalValueLockedToken1': '333333'
                }
            },
            {
                'periodStartUnix': 1700003600,
                'close': 1505.0,
                'low': 1490.0,
                'high': 1525.0,
                'feeGrowthGlobal0X128': str(int(1.5 * (2 ** 128))),
                'feeGrowthGlobal1X128': str(int(0.8 * (2 ** 128))),
                'pool': {
                    'totalValueLockedUSD': '1000000',
                    'totalValueLockedToken0': '500',
                    'totalValueLockedToken1': '333333'
                }
            }
        ]

        result = calc_fees(
            data=data,
            pool=pool,
            base_id=0,
            liquidity=1000000,
            unbounded_liquidity=500000,
            min_price=1400,
            max_price=1600,
            custom_fee_divisor=1.0,
            leverage=1.0,
            investment=10000,
            token_ratio={'token0': 0.5, 'token1': 0.5},
            hedging={'type': None, 'amount': 0, 'leverage': 1}
        )

        assert len(result) == 2, "Should return 2 records"
        assert 'feeV' in result[0], "Should have feeV field"
        assert 'activeliquidity' in result[0], "Should have activeliquidity field"
        assert result[0]['feeV'] == 0, "First hour should have 0 fees"
        assert result[1]['feeV'] >= 0, "Second hour should have >= 0 fees"

    def test_pivot_fee_data(self):
        """Test pivoting hourly data to daily"""
        # Create 25 hours of data (spanning 2+ days depending on timezone)
        # Note: pivot_fee_data uses local timezone for date grouping
        from datetime import datetime

        # Start at local midnight to ensure clean day boundaries
        # Get today at midnight local time and convert to timestamp
        now = datetime.now()
        local_midnight = datetime(now.year, now.month, now.day, 0, 0, 0)
        base_timestamp = int(local_midnight.timestamp())

        data = []
        for i in range(25):
            data.append({
                'periodStartUnix': base_timestamp + (i * 3600),  # Hourly increments from local midnight
                'feeToken0': 0.1 * i,
                'feeToken1': 0.05 * i,
                'feeV': 0.15 * i,
                'feeUnb': 0.2 * i,
                'fgV': 0.01 * i,
                'feeUSD': 100 * i,
                'activeliquidity': 80.0,
                'amountV': 10000,
                'amountTR': 10100,
                'close': 1500.0
            })

        result = pivot_fee_data(
            data=data,
            base_id=0,
            investment=10000,
            leverage=1.0,
            token_ratio={'token0': 0.5, 'token1': 0.5}
        )

        # Should aggregate into 2 days (starting from local midnight)
        # Day 1: hours 0-23 (24 hours)
        # Day 2: hour 24 (1 hour)
        assert len(result) == 2, f"Expected 2 days, got {len(result)}"
        assert result[0]['count'] == 24, f"First day should have 24 hours, got {result[0]['count']}"
        assert result[1]['count'] == 1, f"Second day should have 1 hour, got {result[1]['count']}"

        # Verify total hours
        total_hours = sum(r['count'] for r in result)
        assert total_hours == 25, f"Expected 25 total hours, got {total_hours}"

    def test_backtest_indicators(self):
        """Test backtest indicator calculations"""
        # Sample daily data
        data = [
            {
                'feeToken0': 0.5,
                'feeToken1': 0.3,
                'feeV': 800,
                'feeUSD': 800,
                'activeliquidity': 85.0,
                'amountV': 10000,
                'amountTR': 10800
            },
            {
                'feeToken0': 0.6,
                'feeToken1': 0.4,
                'feeV': 900,
                'feeUSD': 900,
                'activeliquidity': 90.0,
                'amountV': 10100,
                'amountTR': 11000
            },
            {
                'feeToken0': 0.7,
                'feeToken1': 0.5,
                'feeV': 1000,
                'feeUSD': 1000,
                'activeliquidity': 95.0,
                'amountV': 10200,
                'amountTR': 11200
            }
        ]

        result = backtest_indicators(
            data=data,
            investment=10000,
            custom_calc=False,
            hedging={'amount': 0}
        )

        assert 'apr' in result, "Should have APR"
        assert 'feeroi' in result, "Should have fee ROI"
        assert 'confidence' in result, "Should have confidence"
        assert result['apr'] > 0, "APR should be positive"
        assert result['confidence'] in ['Low', 'Medium', 'High', 'Very High'], "Invalid confidence level"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
