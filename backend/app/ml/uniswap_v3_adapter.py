"""
Uniswap V3 Math Adapter for ML Backend

Unified adapter that provides all Uniswap V3 math functions for the ML environment.
This module replaces both whitepaper_math.py and liquidity_math.py with direct
imports from the uniswap_v3 module.

All calculations use Uniswap V3 whitepaper formulas with on-chain precision.

References:
- Whitepaper Section 6.2.1: Concentrated Liquidity
- Whitepaper Section 6.3: Fee Growth
- Whitepaper Section 6.4.1: Impermanent Loss
"""

import math
import sys
import os
from typing import Tuple, Dict, Any

# Add project root to path for uniswap_v3 module access
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from uniswap_v3 module
from uniswap_v3.constants import Q96, Q128, TICK_SPACINGS, MIN_TICK, MAX_TICK
from uniswap_v3.math.tick_math import (
    get_sqrt_ratio_at_tick,
    get_tick_at_sqrt_ratio,
    tick_to_price as _tick_to_price,
    price_to_tick as _price_to_tick,
    round_tick_to_spacing,
)
from uniswap_v3.math.liquidity_math import (
    get_liquidity_for_amounts,
    get_amounts_for_liquidity,
    get_liquidity_for_amount0,
    get_liquidity_for_amount1,
)
from uniswap_v3.math.fee_math import (
    calculate_fee_growth_delta,
    calculate_uncollected_fees,
    fee_growth_inside,
)


# ==============================================================================
# Price/Tick Conversion
# ==============================================================================

def price_to_tick(price: float, token0_decimals: int, token1_decimals: int) -> int:
    """
    Convert human-readable price to tick.

    tick = log₁.₀₀₀₁(price × 10^(token1_decimals - token0_decimals))

    Args:
        price: Human-readable price (token1/token0, e.g., USDC per WETH)
        token0_decimals: Token0 decimals (e.g., 18 for WETH)
        token1_decimals: Token1 decimals (e.g., 6 for USDT)

    Returns:
        Tick index
    """
    if price <= 0:
        raise ValueError("Price must be positive")

    # Adjust price for decimal difference
    # For WETH/USDT (token0=WETH 18, token1=USDT 6):
    # tick = log(price × 10^(6-18)) = log(price × 10^-12)
    decimal_diff = token1_decimals - token0_decimals
    adjusted_price = price * (10 ** decimal_diff)

    if adjusted_price <= 0:
        raise ValueError("Adjusted price must be positive")

    tick = math.log(adjusted_price) / math.log(1.0001)
    return round(tick)


def tick_to_price(tick: int, token0_decimals: int, token1_decimals: int) -> float:
    """
    Convert tick to human-readable price.

    price = 1.0001^tick / 10^(token1_decimals - token0_decimals)

    Args:
        tick: Tick index
        token0_decimals: Token0 decimals
        token1_decimals: Token1 decimals

    Returns:
        Human-readable price (token1/token0)
    """
    decimal_diff = token1_decimals - token0_decimals
    raw_price = 1.0001 ** tick
    return raw_price / (10 ** decimal_diff)


def round_to_nearest_tick(
    price: float,
    fee_tier: int,
    token0_decimals: int,
    token1_decimals: int
) -> float:
    """
    Round a price to the nearest valid tick for the given fee tier.

    Args:
        price: Price value to round
        fee_tier: Fee tier (100, 500, 3000, 10000)
        token0_decimals: Token0 decimals
        token1_decimals: Token1 decimals

    Returns:
        Rounded price aligned to valid tick spacing
    """
    if price <= 0:
        return price

    tick_spacing = TICK_SPACINGS.get(fee_tier, 60)
    tick = price_to_tick(price, token0_decimals, token1_decimals)
    rounded_tick = round_tick_to_spacing(tick, tick_spacing)
    return tick_to_price(rounded_tick, token0_decimals, token1_decimals)


def get_tick_from_price(price: float, pool_config: Dict[str, Any], base_selected: int = 0) -> int:
    """
    Convert price to tick index (compatibility wrapper for pool config).

    Args:
        price: Price value
        pool_config: Pool config with token0/token1 decimals
        base_selected: Which token is base (0 or 1)

    Returns:
        Tick index
    """
    if base_selected == 1:
        # Inverted price
        decimal0 = pool_config['token1']['decimals']
        decimal1 = pool_config['token0']['decimals']
        price = 1 / price if price > 0 else 0
    else:
        decimal0 = pool_config['token0']['decimals']
        decimal1 = pool_config['token1']['decimals']

    return price_to_tick(price, decimal0, decimal1)


# ==============================================================================
# Liquidity Calculation
# ==============================================================================

def calculate_liquidity(
    investment: float,
    price: float,
    pa: float,
    pb: float,
    token0_decimals: int = 18,
    token1_decimals: int = 6
) -> int:
    """
    Calculate liquidity L for a given investment and price range.

    Uses optimal token split to ensure L_from_x = L_from_y for maximum efficiency.

    Args:
        investment: Investment amount (USD)
        price: Current price (token1/token0)
        pa: Lower bound price
        pb: Upper bound price
        token0_decimals: Token0 decimals
        token1_decimals: Token1 decimals

    Returns:
        Liquidity L (integer)
    """
    if investment <= 0 or price <= 0 or pa <= 0 or pb <= 0:
        return 0
    if pa >= pb:
        pa, pb = pb, pa

    # Calculate optimal token split
    x_human, y_human = optimal_token_split(investment, price, pa, pb)

    # Convert to smallest units
    x_amount = int(x_human * (10 ** token0_decimals))
    y_amount = int(y_human * (10 ** token1_decimals))

    # Convert prices to sqrtPriceX96
    tick_current = price_to_tick(price, token0_decimals, token1_decimals)
    tick_lower = price_to_tick(pa, token0_decimals, token1_decimals)
    tick_upper = price_to_tick(pb, token0_decimals, token1_decimals)

    if tick_lower > tick_upper:
        tick_lower, tick_upper = tick_upper, tick_lower

    sqrt_price_x96 = get_sqrt_ratio_at_tick(tick_current)
    sqrt_price_lower = get_sqrt_ratio_at_tick(tick_lower)
    sqrt_price_upper = get_sqrt_ratio_at_tick(tick_upper)

    # Calculate liquidity using on-chain math
    return get_liquidity_for_amounts(
        sqrt_price_x96,
        sqrt_price_lower,
        sqrt_price_upper,
        x_amount,
        y_amount
    )


def optimal_token_split(
    investment: float,
    price: float,
    pa: float,
    pb: float
) -> Tuple[float, float]:
    """
    Calculate optimal token split for LP position.

    Returns (x, y) where x is token0 amount and y is token1 amount (in human units).
    This ensures L_from_x = L_from_y for maximum capital efficiency.

    Args:
        investment: Total investment (USD)
        price: Current price (token1/token0)
        pa: Lower bound price
        pb: Upper bound price

    Returns:
        (x, y) = (token0 amount, token1 amount) in human-readable units
    """
    sqrt_p = math.sqrt(price)
    sqrt_pa = math.sqrt(pa)
    sqrt_pb = math.sqrt(pb)

    if price <= pa:
        # Below range: all token0
        return investment / price, 0
    elif price >= pb:
        # Above range: all token1
        return 0, investment
    else:
        # In range: optimal split to ensure L_from_x = L_from_y
        # k = (√Pb - √P) / (√P × √Pb × (√P - √Pa))
        k = (sqrt_pb - sqrt_p) / (sqrt_p * sqrt_pb * (sqrt_p - sqrt_pa))
        y = investment / (k * price + 1)
        x = k * y
        return x, y


def tokens_for_strategy(
    min_range: float,
    max_range: float,
    investment: float,
    price: float,
    decimal: int
) -> Tuple[float, float]:
    """
    Calculate token amounts for a strategy position.

    Args:
        min_range: Lower price bound
        max_range: Upper price bound
        investment: Total investment (USD)
        price: Current price (token1/token0)
        decimal: Decimal difference (token1_decimals - token0_decimals)

    Returns:
        (amount0, amount1) - Token amounts for this position
    """
    return optimal_token_split(investment, price, min_range, max_range)


def liquidity_for_strategy(
    price: float,
    low: float,
    high: float,
    tokens0: float,
    tokens1: float,
    decimal0: int,
    decimal1: int
) -> int:
    """
    Calculate liquidity for a strategy position.

    Args:
        price: Current price
        low: Lower price bound
        high: Upper price bound
        tokens0: Amount of token0
        tokens1: Amount of token1
        decimal0: Token0 decimals
        decimal1: Token1 decimals

    Returns:
        Liquidity amount
    """
    if price <= 0 or low <= 0 or high <= 0:
        return 0
    if low >= high:
        low, high = high, low
    # Guard against zero-width range (would cause division by zero)
    if abs(high - low) / low < 0.001:  # Less than 0.1% range width
        return 0

    # Convert to smallest units
    amount0 = int(tokens0 * (10 ** decimal0))
    amount1 = int(tokens1 * (10 ** decimal1))

    # Convert prices to sqrtPriceX96
    tick_current = price_to_tick(price, decimal0, decimal1)
    tick_lower = price_to_tick(low, decimal0, decimal1)
    tick_upper = price_to_tick(high, decimal0, decimal1)

    if tick_lower > tick_upper:
        tick_lower, tick_upper = tick_upper, tick_lower

    sqrt_price_x96 = get_sqrt_ratio_at_tick(tick_current)
    sqrt_price_lower = get_sqrt_ratio_at_tick(tick_lower)
    sqrt_price_upper = get_sqrt_ratio_at_tick(tick_upper)

    return get_liquidity_for_amounts(
        sqrt_price_x96,
        sqrt_price_lower,
        sqrt_price_upper,
        amount0,
        amount1
    )


def get_token_amounts(
    liquidity: int,
    price: float,
    pa: float,
    pb: float,
    token0_decimals: int = 18,
    token1_decimals: int = 6
) -> Tuple[float, float]:
    """
    Calculate token amounts from liquidity (whitepaper formula).

    Args:
        liquidity: Position liquidity L
        price: Current price
        pa: Lower bound price
        pb: Upper bound price
        token0_decimals: Token0 decimals
        token1_decimals: Token1 decimals

    Returns:
        (x, y) = (token0, token1) amounts in human-readable units
    """
    if liquidity <= 0 or price <= 0 or pa <= 0 or pb <= 0:
        return 0.0, 0.0
    if pa >= pb:
        pa, pb = pb, pa

    # Convert prices to sqrtPriceX96
    tick_current = price_to_tick(price, token0_decimals, token1_decimals)
    tick_lower = price_to_tick(pa, token0_decimals, token1_decimals)
    tick_upper = price_to_tick(pb, token0_decimals, token1_decimals)

    if tick_lower > tick_upper:
        tick_lower, tick_upper = tick_upper, tick_lower

    sqrt_price_x96 = get_sqrt_ratio_at_tick(tick_current)
    sqrt_price_lower = get_sqrt_ratio_at_tick(tick_lower)
    sqrt_price_upper = get_sqrt_ratio_at_tick(tick_upper)

    # Get amounts using on-chain math
    amount0, amount1 = get_amounts_for_liquidity(
        sqrt_price_x96,
        sqrt_price_lower,
        sqrt_price_upper,
        liquidity
    )

    # Convert to human-readable units
    x = amount0 / (10 ** token0_decimals)
    y = amount1 / (10 ** token1_decimals)

    return x, y


# ==============================================================================
# Impermanent Loss Calculation
# ==============================================================================

def calculate_il(
    investment: float,
    mint_price: float,
    current_price: float,
    pa: float,
    pb: float,
    token0_decimals: int = 18,
    token1_decimals: int = 6
) -> Dict[str, float]:
    """
    Calculate Impermanent Loss using whitepaper formula (Section 6.4.1).

    IL = (LP_value - HODL_value) / HODL_value

    Args:
        investment: Initial investment (USD)
        mint_price: Price when position was minted
        current_price: Current market price
        pa: Lower bound price
        pb: Upper bound price
        token0_decimals: Token0 decimals
        token1_decimals: Token1 decimals

    Returns:
        Dict with il_pct, il_absolute, hodl_value, lp_value, etc.
    """
    if investment <= 0 or mint_price <= 0 or current_price <= 0:
        return {
            'il_pct': 0.0,
            'il_absolute': 0.0,
            'hodl_value': investment,
            'lp_value': investment,
        }
    if pa <= 0 or pb <= 0 or pa >= pb:
        return {
            'il_pct': 0.0,
            'il_absolute': 0.0,
            'hodl_value': investment,
            'lp_value': investment,
        }

    # Calculate liquidity at mint
    L = calculate_liquidity(investment, mint_price, pa, pb, token0_decimals, token1_decimals)

    if L <= 0:
        return {
            'il_pct': 0.0,
            'il_absolute': 0.0,
            'hodl_value': investment,
            'lp_value': investment,
        }

    # Initial token amounts at mint price
    x0, y0 = get_token_amounts(L, mint_price, pa, pb, token0_decimals, token1_decimals)

    # Current token amounts
    x1, y1 = get_token_amounts(L, current_price, pa, pb, token0_decimals, token1_decimals)

    # Value calculations
    hodl_value = x0 * current_price + y0
    lp_value = x1 * current_price + y1

    # IL calculation (negative when LP < HODL)
    if hodl_value <= 0:
        il_pct = 0.0
        il_absolute = 0.0
    else:
        il_pct = (lp_value - hodl_value) / hodl_value
        il_absolute = lp_value - hodl_value

    return {
        'il_pct': il_pct,
        'il_absolute': il_absolute,
        'hodl_value': hodl_value,
        'lp_value': lp_value,
        'L': L,
        'x0': x0, 'y0': y0,
        'x1': x1, 'y1': y1,
    }


# ==============================================================================
# Fee Calculation
# ==============================================================================

def calculate_fees(
    liquidity: int,
    fee_growth_start_0: int,
    fee_growth_end_0: int,
    fee_growth_start_1: int,
    fee_growth_end_1: int,
    active_percentage: float,
    token0_decimals: int = 18,
    token1_decimals: int = 6,
    current_price: float = 1.0
) -> float:
    """
    Calculate fees using whitepaper formula (Section 6.3, 6.4).

    f_u = L × Δf_g / Q128

    Args:
        liquidity: Position liquidity L
        fee_growth_start_0: Start feeGrowthGlobal0X128
        fee_growth_end_0: End feeGrowthGlobal0X128
        fee_growth_start_1: Start feeGrowthGlobal1X128
        fee_growth_end_1: End feeGrowthGlobal1X128
        active_percentage: Percentage of time in range (0-100)
        token0_decimals: Token0 decimals
        token1_decimals: Token1 decimals
        current_price: Current price for USD conversion

    Returns:
        Total fees earned (USD)
    """
    if liquidity <= 0 or active_percentage <= 0:
        return 0.0

    # Calculate fee growth deltas
    delta_fg0 = calculate_fee_growth_delta(fee_growth_end_0, fee_growth_start_0)
    delta_fg1 = calculate_fee_growth_delta(fee_growth_end_1, fee_growth_start_1)

    # Calculate token fees: f_u = L × Δf_g / Q128
    fees_token0 = (liquidity * delta_fg0) // Q128
    fees_token1 = (liquidity * delta_fg1) // Q128

    # Apply active ratio
    active_ratio = active_percentage / 100.0

    # Convert to human-readable units
    fees_0_human = fees_token0 / (10 ** token0_decimals) * active_ratio
    fees_1_human = fees_token1 / (10 ** token1_decimals) * active_ratio

    # Convert to USD
    fees_usd = fees_0_human * current_price + fees_1_human

    return fees_usd


def active_liquidity_for_candle(
    min_tick: float,
    max_tick: float,
    low_tick: float,
    high_tick: float
) -> float:
    """
    Estimate active liquidity percentage for a candle period.

    Args:
        min_tick: Position minimum tick
        max_tick: Position maximum tick
        low_tick: Candle low tick
        high_tick: Candle high tick

    Returns:
        Active liquidity percentage (0-100)
    """
    if high_tick <= min_tick or low_tick >= max_tick:
        return 0.0

    divider = high_tick - low_tick
    if divider == 0:
        divider = 1

    overlap = min(max_tick, high_tick) - max(min_tick, low_tick)
    ratio = (overlap / divider) * 100

    if math.isnan(ratio) or ratio < 0:
        return 0.0

    return ratio


# ==============================================================================
# Action to Range (for ML model)
# ==============================================================================

def action_to_range(
    action,
    current_price: float,
    volatility: float,
    fee_tier: int,
    token0_decimals: int,
    token1_decimals: int
) -> Tuple[float, float]:
    """
    Convert model action to actual price range using log-scale mapping.

    Args:
        action: [rebalance_confidence, min_multiplier, max_multiplier] in [-1, 1]
        current_price: Current pool price
        volatility: Absolute price volatility
        fee_tier: Pool fee tier for tick rounding
        token0_decimals: Token0 decimals
        token1_decimals: Token1 decimals

    Returns:
        (min_price, max_price) tuple
    """
    import numpy as np

    # Log-scale parameters
    min_width = 0.5   # Minimum width in standard deviations
    max_width = 5.0   # Maximum width in standard deviations
    width_ratio = max_width / min_width

    # Extract action components
    min_action = float(action[1]) if len(action) > 1 else 0.0
    max_action = float(action[2]) if len(action) > 2 else 0.0

    # Log-scale mapping: [-1, 1] -> [0.5σ, 5.0σ]
    min_multiplier = min_width * (width_ratio ** ((min_action + 1) / 2))
    max_multiplier = min_width * (width_ratio ** ((max_action + 1) / 2))

    # Calculate relative volatility (percentage)
    relative_volatility = volatility / current_price if current_price > 0 else 0.05

    # Calculate price range using percentage
    min_price = current_price * (1 - relative_volatility * min_multiplier)
    max_price = current_price * (1 + relative_volatility * max_multiplier)

    # Safety bounds
    if min_price <= 0:
        min_price = current_price * 0.5
    if max_price <= min_price:
        max_price = min_price * 1.5

    # Round to valid ticks
    min_price = round_to_nearest_tick(min_price, fee_tier, token0_decimals, token1_decimals)
    max_price = round_to_nearest_tick(max_price, fee_tier, token0_decimals, token1_decimals)

    return min_price, max_price


# ==============================================================================
# Backward Compatibility Aliases
# ==============================================================================

# Aliases for whitepaper_math.py functions
calculate_liquidity_whitepaper = calculate_liquidity
calculate_il_whitepaper = calculate_il
calculate_fees_whitepaper = calculate_fees
get_token_amounts_whitepaper = get_token_amounts

# Aliases for liquidity_math.py functions
tokens_for_strategy = tokens_for_strategy
liquidity_for_strategy = liquidity_for_strategy
active_liquidity_for_candle = active_liquidity_for_candle
round_to_nearest_tick = round_to_nearest_tick
action_to_range = action_to_range
