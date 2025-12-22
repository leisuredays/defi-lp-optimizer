"""
Tick/Price 변환 함수

tick ↔ price ↔ sqrtPrice 변환.
"""

import math
from typing import Tuple

from ..constants import Q96, MIN_TICK, MAX_TICK, TICK_SPACINGS


# Uniswap V3 TickMath 상수
MIN_SQRT_RATIO: int = 4295128739
MAX_SQRT_RATIO: int = 1461446703485210103287273052203988822378723970342


def tick_to_sqrt(tick: int) -> int:
    """tick → sqrtPriceX96 (Q64.96)"""
    if tick < MIN_TICK or tick > MAX_TICK:
        raise ValueError(f"tick 범위 초과: {tick}")

    abs_tick = abs(tick)

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

    return (ratio >> 32) + (1 if ratio % (1 << 32) != 0 else 0)


def tick_to_price(tick: int, decimals0: int = 18, decimals1: int = 6) -> float:
    """tick → 가격 (token1/token0)"""
    return 1.0001 ** tick * (10 ** (decimals0 - decimals1))


def price_to_tick(price: float, decimals0: int = 18, decimals1: int = 6) -> int:
    """가격 → tick"""
    if price <= 0:
        raise ValueError("가격은 양수")
    ratio = price * (10 ** (decimals1 - decimals0))
    return int(math.log(ratio) / math.log(1.0001))


def snap_tick(tick: int, spacing: int) -> int:
    """tick을 유효한 간격으로 반올림"""
    lower = (tick // spacing) * spacing
    upper = lower + spacing
    return upper if abs(tick - upper) <= abs(tick - lower) else lower
