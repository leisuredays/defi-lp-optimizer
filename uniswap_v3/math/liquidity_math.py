"""
Liquidity Math - 유동성 계산

Uniswap V3의 집중화된 유동성(Concentrated Liquidity) 계산.
특정 가격 범위에서의 토큰 수량과 유동성 간의 변환.

References:
- Uniswap V3 Core: contracts/libraries/LiquidityMath.sol
- Uniswap V3 Periphery: contracts/libraries/LiquidityAmounts.sol
- 백서 Section 6.2.1: Concentrated Liquidity

핵심 공식:
    L = Δy / (√P_upper - √P_lower)  # token1 기준
    L = Δx / (1/√P_lower - 1/√P_upper)  # token0 기준
"""

from typing import Tuple

from ..constants import Q96


def get_amount0_delta(
    sqrt_ratio_a_x96: int,
    sqrt_ratio_b_x96: int,
    liquidity: int,
    round_up: bool = True
) -> int:
    """유동성에서 amount0 변화량 계산

    두 가격 사이에서 주어진 유동성으로 얻을 수 있는 token0 양.

    공식: Δx = L * (√P_b - √P_a) / (√P_a * √P_b)
              = L * (1/√P_a - 1/√P_b)

    Args:
        sqrt_ratio_a_x96: 하한 sqrtPriceX96
        sqrt_ratio_b_x96: 상한 sqrtPriceX96
        liquidity: 유동성
        round_up: True면 올림, False면 내림

    Returns:
        amount0 (token0 수량, 최소 단위)
    """
    if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
        sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96

    numerator1 = liquidity << 96
    numerator2 = sqrt_ratio_b_x96 - sqrt_ratio_a_x96

    if round_up:
        return _div_rounding_up(
            _mul_div_rounding_up(numerator1, numerator2, sqrt_ratio_b_x96),
            sqrt_ratio_a_x96
        )
    else:
        return (numerator1 * numerator2 // sqrt_ratio_b_x96) // sqrt_ratio_a_x96


def get_amount1_delta(
    sqrt_ratio_a_x96: int,
    sqrt_ratio_b_x96: int,
    liquidity: int,
    round_up: bool = True
) -> int:
    """유동성에서 amount1 변화량 계산

    두 가격 사이에서 주어진 유동성으로 얻을 수 있는 token1 양.

    공식: Δy = L * (√P_b - √P_a)

    Args:
        sqrt_ratio_a_x96: 하한 sqrtPriceX96
        sqrt_ratio_b_x96: 상한 sqrtPriceX96
        liquidity: 유동성
        round_up: True면 올림, False면 내림

    Returns:
        amount1 (token1 수량, 최소 단위)
    """
    if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
        sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96

    if round_up:
        return _div_rounding_up(
            liquidity * (sqrt_ratio_b_x96 - sqrt_ratio_a_x96),
            Q96
        )
    else:
        return liquidity * (sqrt_ratio_b_x96 - sqrt_ratio_a_x96) // Q96


def get_liquidity_for_amount0(
    sqrt_ratio_a_x96: int,
    sqrt_ratio_b_x96: int,
    amount0: int
) -> int:
    """amount0에서 유동성 계산

    주어진 token0 양으로 얻을 수 있는 최대 유동성.

    공식: L = Δx * √P_a * √P_b / (√P_b - √P_a)

    Args:
        sqrt_ratio_a_x96: 하한 sqrtPriceX96
        sqrt_ratio_b_x96: 상한 sqrtPriceX96
        amount0: token0 수량

    Returns:
        유동성
    """
    if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
        sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96
    # Guard against division by zero (identical sqrt prices)
    if sqrt_ratio_b_x96 <= sqrt_ratio_a_x96:
        return 0

    intermediate = sqrt_ratio_a_x96 * sqrt_ratio_b_x96 // Q96
    return amount0 * intermediate // (sqrt_ratio_b_x96 - sqrt_ratio_a_x96)


def get_liquidity_for_amount1(
    sqrt_ratio_a_x96: int,
    sqrt_ratio_b_x96: int,
    amount1: int
) -> int:
    """amount1에서 유동성 계산

    주어진 token1 양으로 얻을 수 있는 최대 유동성.

    공식: L = Δy / (√P_b - √P_a)

    Args:
        sqrt_ratio_a_x96: 하한 sqrtPriceX96
        sqrt_ratio_b_x96: 상한 sqrtPriceX96
        amount1: token1 수량

    Returns:
        유동성
    """
    if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
        sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96
    # Guard against division by zero (identical sqrt prices)
    if sqrt_ratio_b_x96 <= sqrt_ratio_a_x96:
        return 0

    return amount1 * Q96 // (sqrt_ratio_b_x96 - sqrt_ratio_a_x96)


def get_liquidity_for_amounts(
    sqrt_ratio_x96: int,
    sqrt_ratio_a_x96: int,
    sqrt_ratio_b_x96: int,
    amount0: int,
    amount1: int
) -> int:
    """토큰 수량에서 유동성 계산

    현재 가격과 범위, 두 토큰 수량이 주어졌을 때
    민트 가능한 최대 유동성을 계산합니다.

    Args:
        sqrt_ratio_x96: 현재 sqrtPriceX96
        sqrt_ratio_a_x96: 하한 sqrtPriceX96
        sqrt_ratio_b_x96: 상한 sqrtPriceX96
        amount0: token0 수량
        amount1: token1 수량

    Returns:
        유동성 (두 제약 조건 중 작은 값)
    """
    if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
        sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96

    if sqrt_ratio_x96 <= sqrt_ratio_a_x96:
        # 가격이 범위 아래: token0만 사용
        return get_liquidity_for_amount0(sqrt_ratio_a_x96, sqrt_ratio_b_x96, amount0)

    elif sqrt_ratio_x96 < sqrt_ratio_b_x96:
        # 가격이 범위 내: 양쪽 토큰 사용, 작은 값 반환
        liquidity0 = get_liquidity_for_amount0(sqrt_ratio_x96, sqrt_ratio_b_x96, amount0)
        liquidity1 = get_liquidity_for_amount1(sqrt_ratio_a_x96, sqrt_ratio_x96, amount1)
        return min(liquidity0, liquidity1)

    else:
        # 가격이 범위 위: token1만 사용
        return get_liquidity_for_amount1(sqrt_ratio_a_x96, sqrt_ratio_b_x96, amount1)


def get_amounts_for_liquidity(
    sqrt_ratio_x96: int,
    sqrt_ratio_a_x96: int,
    sqrt_ratio_b_x96: int,
    liquidity: int
) -> Tuple[int, int]:
    """유동성에서 토큰 수량 계산

    현재 가격과 범위, 유동성이 주어졌을 때
    포지션이 보유한 토큰 수량을 계산합니다.

    Args:
        sqrt_ratio_x96: 현재 sqrtPriceX96
        sqrt_ratio_a_x96: 하한 sqrtPriceX96
        sqrt_ratio_b_x96: 상한 sqrtPriceX96
        liquidity: 유동성

    Returns:
        (amount0, amount1) 튜플
    """
    if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
        sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96

    if sqrt_ratio_x96 <= sqrt_ratio_a_x96:
        # 가격이 범위 아래: token0만 보유
        amount0 = get_amount0_delta(sqrt_ratio_a_x96, sqrt_ratio_b_x96, liquidity, False)
        amount1 = 0

    elif sqrt_ratio_x96 < sqrt_ratio_b_x96:
        # 가격이 범위 내: 양쪽 토큰 보유
        amount0 = get_amount0_delta(sqrt_ratio_x96, sqrt_ratio_b_x96, liquidity, False)
        amount1 = get_amount1_delta(sqrt_ratio_a_x96, sqrt_ratio_x96, liquidity, False)

    else:
        # 가격이 범위 위: token1만 보유
        amount0 = 0
        amount1 = get_amount1_delta(sqrt_ratio_a_x96, sqrt_ratio_b_x96, liquidity, False)

    return amount0, amount1


def _mul_div_rounding_up(a: int, b: int, denominator: int) -> int:
    """(a * b) / denominator 올림"""
    result = (a * b) // denominator
    if (a * b) % denominator > 0:
        result += 1
    return result


def _div_rounding_up(numerator: int, denominator: int) -> int:
    """numerator / denominator 올림"""
    result = numerator // denominator
    if numerator % denominator > 0:
        result += 1
    return result
