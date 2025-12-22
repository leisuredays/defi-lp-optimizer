"""
Tick Math - Tick ↔ Price 변환

Uniswap V3의 틱 수학 함수들. 온체인 컨트랙트와 동일한 정밀도로 구현.

References:
- Uniswap V3 Core: contracts/libraries/TickMath.sol
- 백서 Section 6.1: Ticks and Tick Spacing

핵심 공식:
    price = 1.0001^tick
    tick = log₁.₀₀₀₁(price)
    sqrtPriceX96 = sqrt(price) * 2^96
"""

import math
from typing import Tuple

from ..constants import Q96, MIN_TICK, MAX_TICK, TICK_SPACINGS


# Uniswap V3 TickMath 상수
MIN_SQRT_RATIO: int = 4295128739
MAX_SQRT_RATIO: int = 1461446703485210103287273052203988822378723970342


def get_sqrt_ratio_at_tick(tick: int) -> int:
    """틱에서 sqrtPriceX96 계산

    Solidity TickMath.getSqrtRatioAtTick()과 동일한 구현.
    온체인 수준의 정밀도를 위해 정수 연산만 사용.

    Args:
        tick: 틱 인덱스 (-887272 ~ 887272)

    Returns:
        sqrtPriceX96 (Q64.96 형식)

    Raises:
        ValueError: 틱이 유효 범위를 벗어난 경우
    """
    if tick < MIN_TICK or tick > MAX_TICK:
        raise ValueError(f"틱이 유효 범위를 벗어났습니다: {tick} (범위: {MIN_TICK} ~ {MAX_TICK})")

    abs_tick = abs(tick)

    # 매직 넘버를 사용한 비트 연산 (Solidity 구현과 동일)
    ratio = 0x100000000000000000000000000000000 if abs_tick & 0x1 == 0 \
        else 0xfffcb933bd6fad37aa2d162d1a594001

    if abs_tick & 0x2:
        ratio = (ratio * 0xfff97272373d413259a46990580e213a) >> 128
    if abs_tick & 0x4:
        ratio = (ratio * 0xfff2e50f5f656932ef12357cf3c7fdcc) >> 128
    if abs_tick & 0x8:
        ratio = (ratio * 0xffe5caca7e10e4e61c3624eaa0941cd0) >> 128
    if abs_tick & 0x10:
        ratio = (ratio * 0xffcb9843d60f6159c9db58835c926644) >> 128
    if abs_tick & 0x20:
        ratio = (ratio * 0xff973b41fa98c081472e6896dfb254c0) >> 128
    if abs_tick & 0x40:
        ratio = (ratio * 0xff2ea16466c96a3843ec78b326b52861) >> 128
    if abs_tick & 0x80:
        ratio = (ratio * 0xfe5dee046a99a2a811c461f1969c3053) >> 128
    if abs_tick & 0x100:
        ratio = (ratio * 0xfcbe86c7900a88aedcffc83b479aa3a4) >> 128
    if abs_tick & 0x200:
        ratio = (ratio * 0xf987a7253ac413176f2b074cf7815e54) >> 128
    if abs_tick & 0x400:
        ratio = (ratio * 0xf3392b0822b70005940c7a398e4b70f3) >> 128
    if abs_tick & 0x800:
        ratio = (ratio * 0xe7159475a2c29b7443b29c7fa6e889d9) >> 128
    if abs_tick & 0x1000:
        ratio = (ratio * 0xd097f3bdfd2022b8845ad8f792aa5825) >> 128
    if abs_tick & 0x2000:
        ratio = (ratio * 0xa9f746462d870fdf8a65dc1f90e061e5) >> 128
    if abs_tick & 0x4000:
        ratio = (ratio * 0x70d869a156d2a1b890bb3df62baf32f7) >> 128
    if abs_tick & 0x8000:
        ratio = (ratio * 0x31be135f97d08fd981231505542fcfa6) >> 128
    if abs_tick & 0x10000:
        ratio = (ratio * 0x9aa508b5b7a84e1c677de54f3e99bc9) >> 128
    if abs_tick & 0x20000:
        ratio = (ratio * 0x5d6af8dedb81196699c329225ee604) >> 128
    if abs_tick & 0x40000:
        ratio = (ratio * 0x2216e584f5fa1ea926041bedfe98) >> 128
    if abs_tick & 0x80000:
        ratio = (ratio * 0x48a170391f7dc42444e8fa2) >> 128

    if tick > 0:
        ratio = (2**256 - 1) // ratio

    # Q128.128 -> Q64.96
    sqrt_price_x96 = (ratio >> 32) + (1 if ratio % (1 << 32) != 0 else 0)
    return sqrt_price_x96


def get_tick_at_sqrt_ratio(sqrt_price_x96: int) -> int:
    """sqrtPriceX96에서 틱 계산

    Solidity TickMath.getTickAtSqrtRatio()과 동일한 구현.

    Args:
        sqrt_price_x96: sqrtPriceX96 (Q64.96 형식)

    Returns:
        틱 인덱스

    Raises:
        ValueError: sqrtPriceX96이 유효 범위를 벗어난 경우
    """
    if sqrt_price_x96 < MIN_SQRT_RATIO or sqrt_price_x96 >= MAX_SQRT_RATIO:
        raise ValueError(
            f"sqrtPriceX96이 유효 범위를 벗어났습니다: {sqrt_price_x96}"
        )

    ratio = sqrt_price_x96 << 32

    r = ratio
    msb = 0

    # 최상위 비트 찾기
    f = (1 if r > 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF else 0) << 7
    msb |= f
    r >>= f

    f = (1 if r > 0xFFFFFFFFFFFFFFFF else 0) << 6
    msb |= f
    r >>= f

    f = (1 if r > 0xFFFFFFFF else 0) << 5
    msb |= f
    r >>= f

    f = (1 if r > 0xFFFF else 0) << 4
    msb |= f
    r >>= f

    f = (1 if r > 0xFF else 0) << 3
    msb |= f
    r >>= f

    f = (1 if r > 0xF else 0) << 2
    msb |= f
    r >>= f

    f = (1 if r > 0x3 else 0) << 1
    msb |= f
    r >>= f

    f = 1 if r > 0x1 else 0
    msb |= f

    if msb >= 128:
        r = ratio >> (msb - 127)
    else:
        r = ratio << (127 - msb)

    log_2 = (msb - 128) << 64

    # 로그 계산
    for i in range(14):
        r = (r * r) >> 127
        f = r >> 128
        log_2 |= f << (63 - i)
        r >>= f

    log_sqrt10001 = log_2 * 255738958999603826347141

    tick_low = (log_sqrt10001 - 3402992956809132418596140100660247210) >> 128
    tick_high = (log_sqrt10001 + 291339464771989622907027621153398088495) >> 128

    if tick_low == tick_high:
        return tick_low

    if get_sqrt_ratio_at_tick(tick_high) <= sqrt_price_x96:
        return tick_high
    else:
        return tick_low


def tick_to_price(tick: int, token0_decimals: int = 18, token1_decimals: int = 6) -> float:
    """틱을 human-readable 가격으로 변환

    price = 1.0001^tick × 10^(token0_decimals - token1_decimals)

    Args:
        tick: 틱 인덱스
        token0_decimals: token0 소수점 자릿수 (예: WETH = 18)
        token1_decimals: token1 소수점 자릿수 (예: USDT = 6)

    Returns:
        가격 (token1/token0, 예: USDT per WETH)

    Example:
        >>> tick_to_price(-196256, 18, 6)  # WETH/USDT
        3000.10
    """
    ratio = 1.0001 ** tick
    return ratio * (10 ** (token0_decimals - token1_decimals))


def price_to_tick(price: float, token0_decimals: int = 18, token1_decimals: int = 6) -> int:
    """Human-readable 가격을 틱으로 변환

    tick = log₁.₀₀₀₁(price × 10^(token1_decimals - token0_decimals))

    Args:
        price: 가격 (token1/token0, 예: USDT per WETH)
        token0_decimals: token0 소수점 자릿수 (예: WETH = 18)
        token1_decimals: token1 소수점 자릿수 (예: USDT = 6)

    Returns:
        틱 인덱스

    Example:
        >>> price_to_tick(3000, 18, 6)  # WETH/USDT $3000
        -196256
    """
    if price <= 0:
        raise ValueError("가격은 양수여야 합니다")

    ratio = price * (10 ** (token1_decimals - token0_decimals))
    tick = math.log(ratio) / math.log(1.0001)
    return int(tick)


def round_tick_to_spacing(tick: int, tick_spacing: int) -> int:
    """틱을 유효한 틱 간격으로 반올림

    각 풀의 fee tier에 따라 유효한 틱만 사용 가능합니다.
    가장 가까운 유효 틱으로 반올림합니다.

    Args:
        tick: 반올림할 틱
        tick_spacing: 틱 간격 (예: 60 for 0.3% fee)

    Returns:
        반올림된 틱 (가장 가까운 유효 틱)
    """
    # Python의 floor division을 사용하여 lower bound 계산
    lower = (tick // tick_spacing) * tick_spacing
    upper = lower + tick_spacing

    # 가장 가까운 틱 선택
    dist_lower = abs(tick - lower)
    dist_upper = abs(tick - upper)

    if dist_lower < dist_upper:
        return lower
    elif dist_lower > dist_upper:
        return upper
    else:
        # 같은 거리일 때: 양수는 올림(upper), 음수는 0 방향(upper)
        return upper


def get_tick_spacing_for_fee(fee_tier: int) -> int:
    """수수료 티어에 해당하는 틱 간격 반환

    Args:
        fee_tier: 수수료 티어 (100, 500, 3000, 10000)

    Returns:
        틱 간격
    """
    if fee_tier not in TICK_SPACINGS:
        raise ValueError(f"지원하지 않는 수수료 티어: {fee_tier}")
    return TICK_SPACINGS[fee_tier]
