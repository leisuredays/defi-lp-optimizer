"""
Impermanent Loss Math - IL 계산

Uniswap V3 집중화된 유동성의 Impermanent Loss 계산.
백서 Section 6.4.1 기반.

References:
- Uniswap V3 Whitepaper: Section 6.4.1
- IL = (LP_value - HODL_value) / HODL_value

핵심 공식:
    IL = (V_LP - V_HODL) / V_HODL

    여기서:
    - V_LP = 현재 가격에서 LP 포지션 가치
    - V_HODL = 초기 토큰을 그냥 보유했을 때 가치
"""

import math
from typing import Tuple, Dict

from .tick_math import get_sqrt_ratio_at_tick, price_to_tick
from .liquidity_math import get_liquidity_for_amounts, get_amounts_for_liquidity
from ..constants import Q96


def optimal_token_split(
    investment: float,
    price: float,
    pa: float,
    pb: float
) -> Tuple[float, float]:
    """
    LP 포지션을 위한 최적 토큰 분할 계산.

    L_from_x = L_from_y가 되도록 최적 분할을 계산합니다.

    Args:
        investment: 총 투자금 (USD)
        price: 현재 가격 (token1/token0)
        pa: 하한 가격
        pb: 상한 가격

    Returns:
        (x, y) = (token0 수량, token1 수량) in human-readable units
    """
    sqrt_p = math.sqrt(price)
    sqrt_pa = math.sqrt(pa)
    sqrt_pb = math.sqrt(pb)

    if price <= pa:
        # 범위 아래: 전부 token0
        return investment / price, 0.0
    elif price >= pb:
        # 범위 위: 전부 token1
        return 0.0, investment
    else:
        # 범위 내: 최적 분할 (L_from_x = L_from_y)
        # k = (√Pb - √P) / (√P × √Pb × (√P - √Pa))
        k = (sqrt_pb - sqrt_p) / (sqrt_p * sqrt_pb * (sqrt_p - sqrt_pa))
        y = investment / (k * price + 1)
        x = k * y
        return x, y


def calculate_liquidity(
    investment: float,
    price: float,
    pa: float,
    pb: float,
    token0_decimals: int = 18,
    token1_decimals: int = 6
) -> int:
    """
    투자금과 가격 범위로 유동성 L 계산.

    최적 토큰 분할을 사용하여 L_from_x = L_from_y가 되도록 합니다.

    Args:
        investment: 투자금 (USD)
        price: 현재 가격 (token1/token0)
        pa: 하한 가격
        pb: 상한 가격
        token0_decimals: Token0 decimals
        token1_decimals: Token1 decimals

    Returns:
        유동성 L (정수)
    """
    if investment <= 0 or price <= 0 or pa <= 0 or pb <= 0:
        return 0
    if pa >= pb:
        pa, pb = pb, pa

    # 최적 토큰 분할 계산
    x_human, y_human = optimal_token_split(investment, price, pa, pb)

    # 최소 단위로 변환
    x_amount = int(x_human * (10 ** token0_decimals))
    y_amount = int(y_human * (10 ** token1_decimals))

    # 가격을 sqrtPriceX96로 변환
    tick_current = price_to_tick(price, token0_decimals, token1_decimals)
    tick_lower = price_to_tick(pa, token0_decimals, token1_decimals)
    tick_upper = price_to_tick(pb, token0_decimals, token1_decimals)

    if tick_lower > tick_upper:
        tick_lower, tick_upper = tick_upper, tick_lower

    sqrt_price_x96 = get_sqrt_ratio_at_tick(tick_current)
    sqrt_price_lower = get_sqrt_ratio_at_tick(tick_lower)
    sqrt_price_upper = get_sqrt_ratio_at_tick(tick_upper)

    # Guard against zero-width range
    if abs(pb - pa) / pa < 0.001:  # 0.1% 미만 범위
        return 0

    # 유동성 계산
    L = get_liquidity_for_amounts(
        sqrt_price_x96, sqrt_price_lower, sqrt_price_upper,
        x_amount, y_amount
    )

    return L


def get_token_amounts(
    liquidity: int,
    price: float,
    pa: float,
    pb: float,
    token0_decimals: int = 18,
    token1_decimals: int = 6
) -> Tuple[float, float]:
    """
    유동성에서 토큰 수량 계산 (백서 공식).

    Args:
        liquidity: 포지션 유동성 L
        price: 현재 가격
        pa: 하한 가격
        pb: 상한 가격
        token0_decimals: Token0 decimals
        token1_decimals: Token1 decimals

    Returns:
        (x, y) = (token0, token1) 수량 (human-readable units)
    """
    if liquidity <= 0 or price <= 0 or pa <= 0 or pb <= 0:
        return 0.0, 0.0
    if pa >= pb:
        pa, pb = pb, pa

    # 가격을 tick으로 변환
    tick_current = price_to_tick(price, token0_decimals, token1_decimals)
    tick_lower = price_to_tick(pa, token0_decimals, token1_decimals)
    tick_upper = price_to_tick(pb, token0_decimals, token1_decimals)

    if tick_lower > tick_upper:
        tick_lower, tick_upper = tick_upper, tick_lower

    sqrt_price_x96 = get_sqrt_ratio_at_tick(tick_current)
    sqrt_price_lower = get_sqrt_ratio_at_tick(tick_lower)
    sqrt_price_upper = get_sqrt_ratio_at_tick(tick_upper)

    # 온체인 수학으로 수량 계산
    amount0, amount1 = get_amounts_for_liquidity(
        sqrt_price_x96, sqrt_price_lower, sqrt_price_upper, liquidity
    )

    # Human-readable 단위로 변환
    x = amount0 / (10 ** token0_decimals)
    y = amount1 / (10 ** token1_decimals)

    return x, y


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
    백서 공식으로 Impermanent Loss 계산 (Section 6.4.1).

    IL = (LP_value - HODL_value) / HODL_value

    Args:
        investment: 초기 투자금 (USD)
        mint_price: 포지션 생성 시 가격
        current_price: 현재 시장 가격
        pa: 하한 가격
        pb: 상한 가격
        token0_decimals: Token0 decimals
        token1_decimals: Token1 decimals

    Returns:
        Dict with il_pct, il_absolute, hodl_value, lp_value, L, x0, y0, x1, y1
    """
    # 입력 검증
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

    # mint 시점의 유동성 계산
    L = calculate_liquidity(investment, mint_price, pa, pb, token0_decimals, token1_decimals)

    if L <= 0:
        return {
            'il_pct': 0.0,
            'il_absolute': 0.0,
            'hodl_value': investment,
            'lp_value': investment,
        }

    # mint 가격에서의 초기 토큰 수량
    x0, y0 = get_token_amounts(L, mint_price, pa, pb, token0_decimals, token1_decimals)

    # 현재 가격에서의 토큰 수량
    x1, y1 = get_token_amounts(L, current_price, pa, pb, token0_decimals, token1_decimals)

    # 가치 계산
    hodl_value = x0 * current_price + y0  # HODL: 초기 토큰을 현재 가격으로
    lp_value = x1 * current_price + y1    # LP: 현재 토큰을 현재 가격으로

    # IL 계산 (LP < HODL 이면 음수)
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


def calculate_il_simple(
    price_ratio: float,
    range_factor: float = 0.0
) -> float:
    """
    간단한 IL 공식 (Uniswap V2 스타일).

    IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1

    Args:
        price_ratio: 현재가격 / 초기가격 (예: 1.5 = 50% 상승)
        range_factor: V3 범위 배수 (0 = V2 unbounded)

    Returns:
        IL 비율 (음수 = 손실)
    """
    if price_ratio <= 0:
        return 0.0

    # V2 unbounded IL
    il_v2 = 2 * math.sqrt(price_ratio) / (1 + price_ratio) - 1

    # V3 concentrated = V2 IL * range_multiplier
    if range_factor > 0:
        # 대략적인 배수: 좁은 범위일수록 IL 증폭
        multiplier = 1.0 / range_factor
        return il_v2 * multiplier

    return il_v2
