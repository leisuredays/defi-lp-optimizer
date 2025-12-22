"""
Math - Uniswap V3 수학 함수

- convert: tick ↔ price 변환
- calc: 유동성, 수수료, 토큰 계산
"""

from .convert import (
    tick_to_sqrt,
    tick_to_price,
    price_to_tick,
    snap_tick,
)
from .calc import (
    amounts_to_L,
    L_to_amounts,
    fee_inside,
    fee_delta,
    uncollected_fees,
    split_tokens,
)

# 하위 호환 별칭
get_sqrt_ratio_at_tick = tick_to_sqrt
round_tick_to_spacing = snap_tick
get_liquidity_for_amounts = amounts_to_L
get_amounts_for_liquidity = L_to_amounts
calculate_fee_growth_delta = fee_delta
fee_growth_inside = fee_inside
calculate_uncollected_fees = uncollected_fees
optimal_token_split = split_tokens
