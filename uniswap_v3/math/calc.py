"""
유동성/수수료/토큰 계산 함수

L 계산, 토큰량 계산, 수수료 계산.
"""

import math
from typing import Tuple

from ..constants import Q96, Q128
from .convert import tick_to_sqrt


def amounts_to_L(
    sqrt_price: int,
    sqrt_lower: int,
    sqrt_upper: int,
    amount0: int,
    amount1: int
) -> int:
    """토큰 수량 → 유동성 L"""
    if sqrt_lower > sqrt_upper:
        sqrt_lower, sqrt_upper = sqrt_upper, sqrt_lower

    if sqrt_price <= sqrt_lower:
        return _L_from_amount0(sqrt_lower, sqrt_upper, amount0)
    elif sqrt_price < sqrt_upper:
        L0 = _L_from_amount0(sqrt_price, sqrt_upper, amount0)
        L1 = _L_from_amount1(sqrt_lower, sqrt_price, amount1)
        return min(L0, L1)
    else:
        return _L_from_amount1(sqrt_lower, sqrt_upper, amount1)


def L_to_amounts(
    sqrt_price: int,
    sqrt_lower: int,
    sqrt_upper: int,
    liquidity: int
) -> Tuple[int, int]:
    """유동성 L → 토큰 수량 (amount0, amount1)"""
    if sqrt_lower > sqrt_upper:
        sqrt_lower, sqrt_upper = sqrt_upper, sqrt_lower

    if sqrt_price <= sqrt_lower:
        return _amount0_from_L(sqrt_lower, sqrt_upper, liquidity), 0
    elif sqrt_price < sqrt_upper:
        amt0 = _amount0_from_L(sqrt_price, sqrt_upper, liquidity)
        amt1 = _amount1_from_L(sqrt_lower, sqrt_price, liquidity)
        return amt0, amt1
    else:
        return 0, _amount1_from_L(sqrt_lower, sqrt_upper, liquidity)


def fee_inside(
    tick_lower: int,
    tick_upper: int,
    current_tick: int,
    fee_global: int,
    fee_outside_lower: int,
    fee_outside_upper: int
) -> int:
    """범위 내 fee growth 계산"""
    # f_b (아래)
    if current_tick >= tick_lower:
        f_b = fee_outside_lower
    else:
        f_b = fee_global - fee_outside_lower

    # f_a (위)
    if current_tick >= tick_upper:
        f_a = fee_global - fee_outside_upper
    else:
        f_a = fee_outside_upper

    result = fee_global - f_b - f_a
    if result < 0:
        result += 2 ** 256
    return result


def fee_delta(current: int, previous: int) -> int:
    """fee growth 변화량"""
    delta = current - previous
    if delta < 0:
        delta += 2 ** 256
    return delta


def uncollected_fees(liquidity: int, fee_current: int, fee_last: int) -> int:
    """미수령 수수료 (Q128)"""
    delta = fee_current - fee_last
    if delta < 0:
        delta += 2 ** 256
    return liquidity * delta


def split_tokens(
    investment: float,
    price: float,
    price_lower: float,
    price_upper: float
) -> Tuple[float, float]:
    """투자금을 최적 토큰 비율로 분할 → (token0, token1)"""
    sqrt_p = math.sqrt(price)
    sqrt_pa = math.sqrt(price_lower)
    sqrt_pb = math.sqrt(price_upper)

    if price <= price_lower:
        return investment / price, 0.0
    elif price >= price_upper:
        return 0.0, investment
    else:
        k = (sqrt_pb - sqrt_p) / (sqrt_p * sqrt_pb * (sqrt_p - sqrt_pa))
        y = investment / (k * price + 1)
        x = k * y
        return x, y


# === 내부 헬퍼 함수 ===

def _L_from_amount0(sqrt_a: int, sqrt_b: int, amount0: int) -> int:
    if sqrt_a > sqrt_b:
        sqrt_a, sqrt_b = sqrt_b, sqrt_a
    if sqrt_b <= sqrt_a:
        return 0
    intermediate = sqrt_a * sqrt_b // Q96
    return amount0 * intermediate // (sqrt_b - sqrt_a)


def _L_from_amount1(sqrt_a: int, sqrt_b: int, amount1: int) -> int:
    if sqrt_a > sqrt_b:
        sqrt_a, sqrt_b = sqrt_b, sqrt_a
    if sqrt_b <= sqrt_a:
        return 0
    return amount1 * Q96 // (sqrt_b - sqrt_a)


def _amount0_from_L(sqrt_a: int, sqrt_b: int, liquidity: int) -> int:
    if sqrt_a > sqrt_b:
        sqrt_a, sqrt_b = sqrt_b, sqrt_a
    numerator = liquidity << 96
    diff = sqrt_b - sqrt_a
    return (numerator * diff // sqrt_b) // sqrt_a


def _amount1_from_L(sqrt_a: int, sqrt_b: int, liquidity: int) -> int:
    if sqrt_a > sqrt_b:
        sqrt_a, sqrt_b = sqrt_b, sqrt_a
    return liquidity * (sqrt_b - sqrt_a) // Q96
