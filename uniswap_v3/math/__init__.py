"""
Math layer for Uniswap V3 Calculator

온체인 수준 정밀도의 수학 함수들:
- tick_math: Tick ↔ Price 변환
- sqrt_price_math: sqrtPriceX96 관련 계산
- liquidity_math: 유동성 계산
- fee_math: 백서 기반 수수료 계산
- il_math: Impermanent Loss 계산
"""

from .tick_math import (
    get_tick_at_sqrt_ratio,
    get_sqrt_ratio_at_tick,
    tick_to_price,
    price_to_tick,
    round_tick_to_spacing,
)
from .sqrt_price_math import (
    sqrt_price_x96_to_price,
    price_to_sqrt_price_x96,
)
from .liquidity_math import (
    get_amount0_delta,
    get_amount1_delta,
    get_liquidity_for_amounts,
    get_amounts_for_liquidity,
)
from .fee_math import (
    fee_growth_inside,
    calculate_uncollected_fees,
    calculate_fee_growth_delta,
)
from .il_math import (
    calculate_il,
    calculate_il_simple,
    calculate_liquidity,
    get_token_amounts,
    optimal_token_split,
)
