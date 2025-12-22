"""
Backtesting Core Logic - Port from JavaScript src/helpers/uniswap/backtest.js

Critical: These functions must replicate EXACT JavaScript behavior for RL reward calculation
"""
import math
from typing import List, Dict, Any, Tuple
from datetime import datetime

from app.ml.uniswap_v3_adapter import (
    get_tick_from_price,
    active_liquidity_for_candle,
    tokens_for_strategy,
    calculate_fees as calculate_fees_whitepaper,
)
from uniswap_v3.constants import Q128


def calc_unbounded_fees(global_fee0: str, prev_global_fee0: str,
                       global_fee1: str, prev_global_fee1: str,
                       pool: dict) -> tuple:
    """Calculate fees per unit of unbounded liquidity."""
    decimal0 = pool['token0']['decimals']
    decimal1 = pool['token1']['decimals']

    fg0_0 = (int(global_fee0) / Q128) / (10 ** decimal0)
    fg0_1 = (int(prev_global_fee0) / Q128) / (10 ** decimal0)

    fg1_0 = (int(global_fee1) / Q128) / (10 ** decimal1)
    fg1_1 = (int(prev_global_fee1) / Q128) / (10 ** decimal1)

    fg0 = fg0_0 - fg0_1
    fg1 = fg1_0 - fg1_1

    return (fg0, fg1)


def calc_fees(data: List[Dict[str, Any]], pool: Dict[str, Any], base_id: int,
              liquidity: float, unbounded_liquidity: float, min_price: float,
              max_price: float, custom_fee_divisor: float, leverage: float,
              investment: float, token_ratio: Dict[str, float],
              hedging: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Port of calcFees from backtest.js (lines 4-79)

    Calculates fees earned per hour based on feeGrowthGlobal deltas and active liquidity.

    Args:
        data: List of hourly pool data records
        pool: Pool configuration with token decimals
        base_id: Base token ID (0 or 1)
        liquidity: Position liquidity
        unbounded_liquidity: Unbounded (V2-style) liquidity
        min_price: Minimum price of range
        max_price: Maximum price of range
        custom_fee_divisor: Fee divisor for custom calculations
        leverage: Leverage amount
        investment: Initial investment
        token_ratio: Token composition ratio
        hedging: Hedging configuration

    Returns:
        List of hourly fee data with calculated metrics
    """
    result = []

    for i, d in enumerate(data):
        # Calculate fee growth from previous period
        if i - 1 < 0:
            fg = [0, 0]
        else:
            prev = data[i - 1]
            fg = calc_unbounded_fees(
                d['feeGrowthGlobal0X128'],
                prev['feeGrowthGlobal0X128'],
                d['feeGrowthGlobal1X128'],
                prev['feeGrowthGlobal1X128'],
                pool
            )

        # Convert prices based on base token
        low = d['low'] if base_id == 0 else 1 / (d['low'] if d['low'] != 0 else 1)
        high = d['high'] if base_id == 0 else 1 / (d['high'] if d['high'] != 0 else 1)

        # Convert prices to ticks
        low_tick = get_tick_from_price(low, pool, base_id)
        high_tick = get_tick_from_price(high, pool, base_id)
        min_tick = get_tick_from_price(min_price, pool, base_id)
        max_tick = get_tick_from_price(max_price, pool, base_id)

        # Calculate active liquidity percentage
        active_liq = active_liquidity_for_candle(min_tick, max_tick, low_tick, high_tick)

        # Calculate token amounts at current price using investment
        close_price = 1 / d['close'] if base_id == 1 else d['close']
        decimal = pool['token1']['decimals'] - pool['token0']['decimals']
        tokens = tokens_for_strategy(
            min_price,
            max_price,
            investment,
            close_price,
            decimal
        )

        # Calculate fees per token
        fee_token0 = 0 if i == 0 else fg[0] * liquidity * active_liq / 100
        fee_token1 = 0 if i == 0 else fg[1] * liquidity * active_liq / 100

        # Unbounded fees
        fee_unb0 = 0 if i == 0 else fg[0] * unbounded_liquidity
        fee_unb1 = 0 if i == 0 else fg[1] * unbounded_liquidity

        # Calculate value metrics
        latest_rec = data[len(data) - 1]
        first_close = 1 / data[0]['close'] if base_id == 1 else data[0]['close']
        current_close = 1 / d['close'] if base_id == 1 else d['close']

        # Token ratio at first close
        token_ratio_first = tokens_for_strategy(
            min_price,
            max_price,
            investment,
            first_close,
            decimal
        )
        x0 = token_ratio_first[1]
        y0 = token_ratio_first[0]

        # Calculate impermanent loss hedge
        imp_loss_hedge = 0
        if hedging['type'] == 'long':
            imp_loss_hedge = hedging['amount'] * hedging['leverage'] * ((current_close - first_close) / first_close)
        elif hedging['type'] == 'short':
            imp_loss_hedge = hedging['amount'] * hedging['leverage'] * ((current_close - first_close) / first_close) * -1

        # Calculate fee and amount values based on base token
        if base_id == 0:
            fg_v = 0 if i == 0 else fg[0] + (fg[1] * d['close'])
            fee_v = 0 if i == 0 else fee_token0 + (fee_token1 * d['close'])
            fee_unb = 0 if i == 0 else fee_unb0 + (fee_unb1 * d['close'])
            amount_v = tokens[0] + (tokens[1] * d['close'])

            # Calculate USD value
            pool_tvl_usd = float(latest_rec['pool']['totalValueLockedUSD'])
            pool_tvl_token0 = float(latest_rec['pool']['totalValueLockedToken0'])
            pool_tvl_token1 = float(latest_rec['pool']['totalValueLockedToken1'])
            latest_close = float(latest_rec['close'])

            fee_usd = fee_v * pool_tvl_usd / ((pool_tvl_token1 * latest_close) + pool_tvl_token0)
            amount_tr = (investment + (amount_v - ((x0 * d['close']) + y0))) + imp_loss_hedge

        else:  # base_id == 1
            fg_v = 0 if i == 0 else (fg[0] / d['close']) + fg[1]
            fee_v = 0 if i == 0 else (fee_token0 / d['close']) + fee_token1
            fee_unb = 0 if i == 0 else fee_unb0 + (fee_unb1 * d['close'])
            amount_v = (tokens[1] / d['close']) + tokens[0]

            # Calculate USD value
            pool_tvl_usd = float(latest_rec['pool']['totalValueLockedUSD'])
            pool_tvl_token0 = float(latest_rec['pool']['totalValueLockedToken0'])
            pool_tvl_token1 = float(latest_rec['pool']['totalValueLockedToken1'])
            latest_close = float(latest_rec['close'])

            fee_usd = fee_v * pool_tvl_usd / (pool_tvl_token1 + (pool_tvl_token0 / latest_close))
            amount_tr = (investment + (amount_v - ((x0 * (1 / d['close'])) + y0))) + imp_loss_hedge

        # Parse date
        date = datetime.fromtimestamp(d['periodStartUnix'])

        # Build result record
        result.append({
            **d,
            'day': date.day,
            'month': date.month,
            'year': date.year,
            'fg0': fg[0],
            'fg1': fg[1],
            'activeliquidity': active_liq,
            'feeToken0': fee_token0,
            'feeToken1': fee_token1,
            'tokens': tokens,
            'fgV': fg_v,
            'feeV': fee_v / custom_fee_divisor,
            'feeUnb': fee_unb,
            'amountV': amount_v,
            'amountTR': amount_tr,
            'feeUSD': fee_usd,
            'close': d['close'],
            'baseClose': 1 / d['close'] if base_id == 1 else d['close']
        })

    return result


def pivot_fee_data(data: List[Dict[str, Any]], base_id: int,
                   investment: float, leverage: float,
                   token_ratio: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Port of pivotFeeData from backtest.js (lines 82-146)

    Aggregates hourly estimated fee data into daily summaries.

    Args:
        data: Hourly fee data from calc_fees()
        base_id: Base token ID (0 or 1)
        investment: Initial investment
        leverage: Leverage amount
        token_ratio: Token composition ratio

    Returns:
        List of daily aggregated fee data
    """
    if not data or len(data) == 0:
        return []

    def create_pivot_record(date: datetime, d: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'date': f"{date.month}/{date.day}/{date.year}",
            'day': date.day,
            'month': date.month,
            'year': date.year,
            'feeToken0': d['feeToken0'],
            'feeToken1': d['feeToken1'],
            'feeV': d['feeV'],
            'feeUnb': d['feeUnb'],
            'fgV': float(d['fgV']),
            'feeUSD': d['feeUSD'],
            'activeliquidity': d['activeliquidity'] if not math.isnan(d['activeliquidity']) else 0,
            'amountV': d['amountV'],
            'amountTR': d['amountTR'],
            'amountVLast': d['amountV'],
            'percFee': d['feeV'] / d['amountV'] if d['amountV'] != 0 else 0,
            'close': d['close'],
            'baseClose': 1 / d['close'] if base_id == 1 else d['close'],
            'count': 1
        }

    first_date = datetime.fromtimestamp(data[0]['periodStartUnix'])
    pivot = [create_pivot_record(first_date, data[0])]

    for i, d in enumerate(data):
        if i > 0:
            current_date = datetime.fromtimestamp(d['periodStartUnix'])
            current_tick = pivot[len(pivot) - 1]

            # Check if same day
            if (current_date.day == current_tick['day'] and
                current_date.month == current_tick['month'] and
                current_date.year == current_tick['year']):

                # Aggregate within same day
                current_tick['feeToken0'] += d['feeToken0']
                current_tick['feeToken1'] += d['feeToken1']
                current_tick['feeV'] += d['feeV']
                current_tick['feeUnb'] += d['feeUnb']
                current_tick['fgV'] = float(current_tick['fgV']) + float(d['fgV'])
                current_tick['feeUSD'] += d['feeUSD']
                current_tick['activeliquidity'] += d['activeliquidity']
                current_tick['amountVLast'] = d['amountV']
                current_tick['count'] += 1

                # If last record, finalize averages
                if i == (len(data) - 1):
                    current_tick['activeliquidity'] = current_tick['activeliquidity'] / current_tick['count']
                    current_tick['percFee'] = (current_tick['feeV'] / current_tick['amountV'] * 100) if current_tick['amountV'] != 0 else 0

            else:
                # Finalize previous day and start new day
                current_tick['activeliquidity'] = current_tick['activeliquidity'] / current_tick['count']
                current_tick['percFee'] = (current_tick['feeV'] / current_tick['amountV'] * 100) if current_tick['amountV'] != 0 else 0
                pivot.append(create_pivot_record(current_date, d))

    return pivot


def backtest_indicators(data: List[Dict[str, Any]], investment: float,
                       custom_calc: bool, hedging: Dict[str, Any]) -> Dict[str, Any]:
    """
    Port of backtestIndicators from backtest.js (lines 149-174)

    Computes final backtest performance metrics (APR, ROI, confidence).

    Args:
        data: Daily aggregated fee data from pivot_fee_data()
        investment: Initial investment
        custom_calc: Whether to use custom calculation
        hedging: Hedging configuration

    Returns:
        Dictionary with performance indicators (feeV, feeRoi, apr, etc.)
    """
    if not data or len(data) == 0:
        return {}

    fee_roi = 0
    token0_fee = 0
    token1_fee = 0
    fee_usd = 0
    active_liquidity = 0
    fee_v = 0

    for i, d in enumerate(data):
        fee_roi += d['feeV']
        token0_fee += d['feeToken0']
        token1_fee += d['feeToken1']
        fee_v += d['feeV']
        fee_usd += d['feeUSD']
        active_liquidity += d['activeliquidity']

    # Calculate ROI
    if custom_calc:
        fee_roi = fee_v / (investment + hedging['amount']) * 100
    else:
        fee_roi = fee_v / (data[0]['amountV'] + hedging['amount']) * 100

    # Calculate averages
    active_liquidity = active_liquidity / len(data)
    apr = fee_roi * 365 / len(data)

    # Asset value change
    asset = ((data[len(data) - 1]['amountV'] - data[0]['amountV']) / data[0]['amountV']) * 100
    total = fee_roi + asset

    # Confidence level
    if active_liquidity == 100:
        confidence = "Very High"
    elif active_liquidity > 80:
        confidence = "High"
    elif active_liquidity > 40:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        'feeV': fee_v,
        'feeroi': round(fee_roi, 2),
        'apr': round(apr, 2),
        'token0Fee': round(token0_fee, 6),
        'token1Fee': round(token1_fee, 6),
        'feeUSD': round(fee_v if custom_calc else fee_usd, 2),
        'activeliquidity': int(active_liquidity),
        'assetval': round(asset, 6),
        'total': round(total, 6),
        'confidence': confidence
    }
