"""
ML Adapter - uniswap_v3.math 함수의 ML 환경용 래퍼

math 모듈의 저수준 함수를 ML 환경에서 사용하기 쉽게 래핑.
- math: tick/sqrtPrice 기반 정수 연산
- adapter: price/USD 기반 고수준 인터페이스
"""

import math as pymath
from typing import Tuple, Dict, Any

from ..constants import Q96, Q128, TICK_SPACINGS
from ..math import (
    # convert.py
    tick_to_sqrt,
    tick_to_price,
    price_to_tick,
    snap_tick,
    # calc.py
    amounts_to_L,
    L_to_amounts,
    fee_delta,
    split_tokens,
)


# ==============================================================================
# Price/Tick 변환 (decimals 처리)
# ==============================================================================

def snap_price(price: float, fee_tier: int, decimals0: int, decimals1: int) -> float:
    """가격을 유효한 tick spacing으로 반올림"""
    import math
    spacing = TICK_SPACINGS.get(fee_tier, 60)

    # 실용적인 최소/최대 tick 범위 (오버플로우 방지)
    # WETH/USDT의 경우 실제 tick 범위: 약 -400000 ~ -300000
    MIN_SAFE_TICK = -500000
    MAX_SAFE_TICK = 500000

    # 유효하지 않은 가격 처리 (0, 음수, NaN, Inf)
    if price <= 0 or not math.isfinite(price):
        rounded = snap_tick(MIN_SAFE_TICK, spacing)
        return tick_to_price(rounded, decimals0, decimals1)

    # ratio 계산 후 유효성 검사
    ratio = price * (10 ** (decimals1 - decimals0))
    if ratio <= 0 or not math.isfinite(ratio):
        rounded = snap_tick(MIN_SAFE_TICK, spacing)
        return tick_to_price(rounded, decimals0, decimals1)

    try:
        tick = int(math.log(ratio) / math.log(1.0001))
        # tick 범위 제한
        tick = max(MIN_SAFE_TICK, min(MAX_SAFE_TICK, tick))
        rounded = snap_tick(tick, spacing)
        return tick_to_price(rounded, decimals0, decimals1)
    except (ValueError, OverflowError):
        rounded = snap_tick(MIN_SAFE_TICK, spacing)
        return tick_to_price(rounded, decimals0, decimals1)


def get_tick_from_price(price: float, pool_config: Dict[str, Any], base_selected: int = 0) -> int:
    """pool_config을 사용한 가격 → tick 변환"""
    if base_selected == 1:
        d0 = pool_config['token1']['decimals']
        d1 = pool_config['token0']['decimals']
        price = 1 / price if price > 0 else 0
    else:
        d0 = pool_config['token0']['decimals']
        d1 = pool_config['token1']['decimals']
    return price_to_tick(price, d0, d1)


# ==============================================================================
# Liquidity 계산 (USD 인터페이스)
# ==============================================================================

def calc_liquidity(
    investment: float,
    price: float,
    pa: float,
    pb: float,
    decimals0: int = 18,
    decimals1: int = 6
) -> int:
    """투자금(USD)과 가격 범위로 유동성 L 계산"""
    if investment <= 0 or price <= 0 or pa <= 0 or pb <= 0:
        return 0
    if pa >= pb:
        pa, pb = pb, pa

    x, y = split_tokens(investment, price, pa, pb)
    x_amt = int(x * 10**decimals0)
    y_amt = int(y * 10**decimals1)

    sqrt_c = tick_to_sqrt(price_to_tick(price, decimals0, decimals1))
    sqrt_a = tick_to_sqrt(price_to_tick(pa, decimals0, decimals1))
    sqrt_b = tick_to_sqrt(price_to_tick(pb, decimals0, decimals1))
    if sqrt_a > sqrt_b:
        sqrt_a, sqrt_b = sqrt_b, sqrt_a

    return amounts_to_L(sqrt_c, sqrt_a, sqrt_b, x_amt, y_amt)


def get_token_amounts(
    L: int,
    price: float,
    pa: float,
    pb: float,
    decimals0: int = 18,
    decimals1: int = 6
) -> Tuple[float, float]:
    """유동성 L과 현재 가격으로 토큰 수량 계산 (human-readable)"""
    if L <= 0:
        return 0.0, 0.0

    sqrt_c = tick_to_sqrt(price_to_tick(price, decimals0, decimals1))
    sqrt_a = tick_to_sqrt(price_to_tick(pa, decimals0, decimals1))
    sqrt_b = tick_to_sqrt(price_to_tick(pb, decimals0, decimals1))
    if sqrt_a > sqrt_b:
        sqrt_a, sqrt_b = sqrt_b, sqrt_a

    amt0, amt1 = L_to_amounts(sqrt_c, sqrt_a, sqrt_b, L)
    return amt0 / 10**decimals0, amt1 / 10**decimals1


# ==============================================================================
# IL 계산
# ==============================================================================

def calc_il(
    investment: float,
    mint_price: float,
    current_price: float,
    pa: float,
    pb: float,
    decimals0: int = 18,
    decimals1: int = 6
) -> Dict[str, float]:
    """
    비영구적 손실(IL) 계산

    IL = (LP_value - HODL_value) / HODL_value
    음수 = 손실, 양수 = 이득
    """
    empty = {'il_pct': 0.0, 'il_usd': 0.0, 'hodl': investment, 'lp': investment}
    if investment <= 0 or mint_price <= 0 or current_price <= 0:
        return empty
    if pa <= 0 or pb <= 0 or pa >= pb:
        return empty

    L = calc_liquidity(investment, mint_price, pa, pb, decimals0, decimals1)
    if L <= 0:
        return empty

    x0, y0 = get_token_amounts(L, mint_price, pa, pb, decimals0, decimals1)
    x1, y1 = get_token_amounts(L, current_price, pa, pb, decimals0, decimals1)

    hodl = x0 * current_price + y0
    lp = x1 * current_price + y1

    if hodl <= 0:
        return {'il_pct': 0.0, 'il_usd': 0.0, 'hodl': hodl, 'lp': lp}

    il_pct = (lp - hodl) / hodl
    il_usd = lp - hodl

    return {'il_pct': il_pct, 'il_usd': il_usd, 'hodl': hodl, 'lp': lp, 'L': L}


# ==============================================================================
# Fee 계산
# ==============================================================================

def calc_fees(
    liquidity: int,
    fg_start_0: int,
    fg_end_0: int,
    fg_start_1: int,
    fg_end_1: int,
    active_pct: float,
    decimals0: int = 18,
    decimals1: int = 6,
    price: float = 1.0
) -> float:
    """
    수수료 계산 (USD)

    fees = L × Δfee_growth × active% / Q128
    """
    if liquidity <= 0 or active_pct <= 0:
        return 0.0

    delta0 = fee_delta(fg_end_0, fg_start_0)
    delta1 = fee_delta(fg_end_1, fg_start_1)

    fees0 = (liquidity * delta0) // Q128
    fees1 = (liquidity * delta1) // Q128

    ratio = active_pct / 100.0
    f0 = fees0 / 10**decimals0 * ratio
    f1 = fees1 / 10**decimals1 * ratio

    return f0 * price + f1


def active_ratio(min_tick: int, max_tick: int, low_tick: int, high_tick: int) -> float:
    """
    캔들 기간 동안 포지션 활성 비율 (0-100%)

    min_tick, max_tick: LP 포지션 범위
    low_tick, high_tick: 캔들의 가격 범위
    """
    if high_tick <= min_tick or low_tick >= max_tick:
        return 0.0

    divider = high_tick - low_tick
    if divider == 0:
        divider = 1

    overlap = min(max_tick, high_tick) - max(min_tick, low_tick)
    ratio = (overlap / divider) * 100

    return max(0.0, ratio) if not pymath.isnan(ratio) else 0.0


# ==============================================================================
# RL Action 변환
# ==============================================================================

def action_to_range(
    action: Tuple[float, float, float],
    current_price: float,
    fee_tier: int,
    decimals0: int,
    decimals1: int,
    volatility: float = 0.05
) -> Tuple[float, float]:
    """
    RL action → (min_price, max_price)

    action[0]: 범위 폭 (-1~1 → 10%~50%, 최소 ±5%)
    action[1]: 범위 중심 오프셋 (-1~1 → -10%~+10%)
    """
    # 입력 유효성 검사
    if current_price <= 0:
        raise ValueError(f"current_price must be positive, got {current_price}")

    # action 값 클리핑 (-1 ~ 1)
    action = [max(-1, min(1, a)) for a in action]

    width_pct = 0.10 + (action[0] + 1) / 2 * 0.40  # 10%~50% (minimum ±5%)
    offset_pct = action[1] * 0.10  # -10%~+10%

    center = current_price * (1 + offset_pct)
    half_width = center * width_pct / 2

    # 안전 범위: 최소 현재가격의 10%, 최대 현재가격의 10배
    min_price = max(center - half_width, current_price * 0.1)
    max_price = min(center + half_width, current_price * 10.0)

    # snap_price는 이제 항상 유효한 가격 반환
    min_price = snap_price(min_price, fee_tier, decimals0, decimals1)
    max_price = snap_price(max_price, fee_tier, decimals0, decimals1)

    # min >= max인 경우 안전하게 처리
    if min_price >= max_price:
        spacing = TICK_SPACINGS.get(fee_tier, 60)
        # snap_price 후에는 항상 유효한 가격이므로 안전하게 변환 가능
        try:
            min_tick = price_to_tick(min_price, decimals0, decimals1)
            max_price = tick_to_price(min_tick + spacing, decimals0, decimals1)
        except ValueError:
            # 최후의 안전장치: 기본 범위 사용
            min_price = current_price * 0.95
            max_price = current_price * 1.05
            min_price = snap_price(min_price, fee_tier, decimals0, decimals1)
            max_price = snap_price(max_price, fee_tier, decimals0, decimals1)

    return min_price, max_price


# ==============================================================================
# 편의 함수 (하위 호환)
# ==============================================================================

def tokens_for_strategy(
    min_range: float,
    max_range: float,
    investment: float,
    price: float,
    decimal: int = 0
) -> Tuple[float, float]:
    """split_tokens 래퍼 (하위 호환)"""
    return split_tokens(investment, price, min_range, max_range)


def liquidity_for_strategy(
    price: float,
    low: float,
    high: float,
    tokens0: float,
    tokens1: float,
    decimal0: int,
    decimal1: int
) -> int:
    """토큰 수량으로 유동성 계산"""
    if price <= 0 or low <= 0 or high <= 0:
        return 0
    if low >= high:
        low, high = high, low
    if abs(high - low) / low < 0.001:
        return 0

    sqrt_c = tick_to_sqrt(price_to_tick(price, decimal0, decimal1))
    sqrt_a = tick_to_sqrt(price_to_tick(low, decimal0, decimal1))
    sqrt_b = tick_to_sqrt(price_to_tick(high, decimal0, decimal1))
    if sqrt_a > sqrt_b:
        sqrt_a, sqrt_b = sqrt_b, sqrt_a

    amt0 = int(tokens0 * 10**decimal0)
    amt1 = int(tokens1 * 10**decimal1)

    return amounts_to_L(sqrt_c, sqrt_a, sqrt_b, amt0, amt1)
