"""
Sqrt Price Math - sqrtPriceX96 관련 계산

Uniswap V3의 가격은 sqrtPriceX96 형식으로 저장됩니다.
sqrtPriceX96 = sqrt(price) * 2^96

References:
- Uniswap V3 Core: contracts/libraries/SqrtPriceMath.sol
"""

import math
from typing import Tuple

from ..constants import Q96, Q192


def sqrt_price_x96_to_price(
    sqrt_price_x96: int,
    decimal0: int = 18,
    decimal1: int = 18
) -> float:
    """sqrtPriceX96을 human-readable 가격으로 변환

    가격 = (sqrtPriceX96 / 2^96)^2 / 10^(decimal1 - decimal0)
         = sqrtPriceX96^2 / 2^192 / 10^(decimal1 - decimal0)

    Args:
        sqrt_price_x96: sqrtPriceX96 값
        decimal0: token0 소수점 자릿수
        decimal1: token1 소수점 자릿수

    Returns:
        가격 (token1/token0 기준, human-readable)
    """
    # 정밀도를 위해 단계별 계산
    sqrt_price = sqrt_price_x96 / Q96
    price_raw = sqrt_price ** 2

    # 소수점 조정: price = price_raw / 10^(decimal1 - decimal0)
    decimal_adjustment = 10 ** (decimal1 - decimal0)
    price = price_raw / decimal_adjustment

    return price


def sqrt_price_x96_to_price_int(
    sqrt_price_x96: int,
    decimal0: int = 18,
    decimal1: int = 18,
    precision: int = 18
) -> int:
    """sqrtPriceX96을 정수 가격으로 변환 (온체인 정밀도)

    결과는 precision 자릿수의 고정소수점 정수입니다.

    Args:
        sqrt_price_x96: sqrtPriceX96 값
        decimal0: token0 소수점 자릿수
        decimal1: token1 소수점 자릿수
        precision: 결과 정밀도 (소수점 자릿수)

    Returns:
        가격 (고정소수점 정수, precision 자릿수)
    """
    # price = sqrtPriceX96^2 / 2^192 * 10^precision / 10^(decimal1 - decimal0)
    decimal_diff = decimal1 - decimal0

    # 오버플로우 방지를 위한 순서 조정
    numerator = sqrt_price_x96 ** 2 * (10 ** precision)
    denominator = Q192 * (10 ** decimal_diff) if decimal_diff >= 0 else Q192

    if decimal_diff < 0:
        numerator *= 10 ** (-decimal_diff)

    return numerator // denominator


def price_to_sqrt_price_x96(
    price: float,
    decimal0: int = 18,
    decimal1: int = 18
) -> int:
    """Human-readable 가격을 sqrtPriceX96으로 변환

    sqrtPriceX96 = sqrt(price * 10^(decimal1 - decimal0)) * 2^96

    Args:
        price: 가격 (token1/token0 기준)
        decimal0: token0 소수점 자릿수
        decimal1: token1 소수점 자릿수

    Returns:
        sqrtPriceX96 값
    """
    if price <= 0:
        raise ValueError("가격은 양수여야 합니다")

    decimal_adjustment = 10 ** (decimal1 - decimal0)
    adjusted_price = price * decimal_adjustment
    sqrt_price = math.sqrt(adjusted_price)
    sqrt_price_x96 = int(sqrt_price * Q96)

    return sqrt_price_x96


def get_next_sqrt_price_from_amount0_rounding_up(
    sqrt_price_x96: int,
    liquidity: int,
    amount: int,
    add: bool
) -> int:
    """amount0 변화에 따른 다음 sqrtPriceX96 계산 (올림)

    amount0를 추가/제거할 때의 새 가격 계산.

    Args:
        sqrt_price_x96: 현재 sqrtPriceX96
        liquidity: 유동성
        amount: amount0 변화량
        add: True면 추가, False면 제거

    Returns:
        새로운 sqrtPriceX96
    """
    if amount == 0:
        return sqrt_price_x96

    numerator1 = liquidity << 96

    if add:
        product = amount * sqrt_price_x96
        if product // amount == sqrt_price_x96:
            denominator = numerator1 + product
            if denominator >= numerator1:
                return _mul_div_rounding_up(numerator1, sqrt_price_x96, denominator)

        return _div_rounding_up(numerator1, (numerator1 // sqrt_price_x96) + amount)
    else:
        product = amount * sqrt_price_x96
        assert product // amount == sqrt_price_x96
        assert numerator1 > product
        denominator = numerator1 - product
        return _mul_div_rounding_up(numerator1, sqrt_price_x96, denominator)


def get_next_sqrt_price_from_amount1_rounding_down(
    sqrt_price_x96: int,
    liquidity: int,
    amount: int,
    add: bool
) -> int:
    """amount1 변화에 따른 다음 sqrtPriceX96 계산 (내림)

    Args:
        sqrt_price_x96: 현재 sqrtPriceX96
        liquidity: 유동성
        amount: amount1 변화량
        add: True면 추가, False면 제거

    Returns:
        새로운 sqrtPriceX96
    """
    if add:
        quotient = (amount << 96) // liquidity
        return sqrt_price_x96 + quotient
    else:
        quotient = _div_rounding_up(amount << 96, liquidity)
        assert sqrt_price_x96 > quotient
        return sqrt_price_x96 - quotient


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
