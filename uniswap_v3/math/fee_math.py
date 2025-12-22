"""
Fee Math - 백서 기반 수수료 계산

Uniswap V3 백서 Section 6.3, 6.4의 공식을 정확하게 구현.
온체인 컨트랙트와 동일한 정밀도의 수수료 계산.

References:
- 백서 Section 6.3: Tick-Indexed State (feeGrowthOutside)
- 백서 Section 6.4.1: Position-Indexed State (uncollected fees)
- 참조 문서: /home/zekiya/liquidity/uniswap-v3-simulator/uniswap_v3_notes.md

핵심 공식:
    f_a(i) = f_g - f_o(i)  if i_c >= i else f_o(i)     # 틱 i 위 수수료
    f_b(i) = f_o(i)        if i_c >= i else f_g - f_o(i) # 틱 i 아래 수수료
    f_r = f_g - f_b(i_l) - f_a(i_u)                     # 범위 내 수수료
    f_u = l × (f_r(t_1) - f_r(t_0))                     # 미수령 수수료
"""

from typing import Tuple, NamedTuple

from ..constants import Q128


class FeeCalculationResult(NamedTuple):
    """수수료 계산 결과"""
    uncollected_fees_0: int  # token0 미수령 수수료 (최소 단위)
    uncollected_fees_1: int  # token1 미수령 수수료 (최소 단위)
    fee_growth_inside_0: int  # 현재 범위 내 fee growth token0
    fee_growth_inside_1: int  # 현재 범위 내 fee growth token1


def fee_growth_above(
    tick_idx: int,
    current_tick: int,
    fee_growth_global: int,
    fee_growth_outside: int
) -> int:
    """틱 위에서 발생한 수수료 성장률 (f_a)

    백서 Section 6.3 공식:
        f_a(i) = f_g - f_o(i)  if i_c >= i
        f_a(i) = f_o(i)        if i_c < i

    Args:
        tick_idx: 틱 인덱스 (i)
        current_tick: 현재 틱 (i_c)
        fee_growth_global: 전역 fee growth (f_g)
        fee_growth_outside: 틱의 fee growth outside (f_o)

    Returns:
        틱 위의 fee growth (f_a)
    """
    if current_tick >= tick_idx:
        return fee_growth_global - fee_growth_outside
    else:
        return fee_growth_outside


def fee_growth_below(
    tick_idx: int,
    current_tick: int,
    fee_growth_global: int,
    fee_growth_outside: int
) -> int:
    """틱 아래에서 발생한 수수료 성장률 (f_b)

    백서 Section 6.3 공식:
        f_b(i) = f_o(i)        if i_c >= i
        f_b(i) = f_g - f_o(i)  if i_c < i

    Args:
        tick_idx: 틱 인덱스 (i)
        current_tick: 현재 틱 (i_c)
        fee_growth_global: 전역 fee growth (f_g)
        fee_growth_outside: 틱의 fee growth outside (f_o)

    Returns:
        틱 아래의 fee growth (f_b)
    """
    if current_tick >= tick_idx:
        return fee_growth_outside
    else:
        return fee_growth_global - fee_growth_outside


def fee_growth_inside(
    tick_lower: int,
    tick_upper: int,
    current_tick: int,
    fee_growth_global: int,
    fee_growth_outside_lower: int,
    fee_growth_outside_upper: int
) -> int:
    """범위 내 fee growth 계산 (f_r)

    백서 Section 6.3 공식:
        f_r = f_g - f_b(i_l) - f_a(i_u)

    Args:
        tick_lower: 하한 틱 (i_l)
        tick_upper: 상한 틱 (i_u)
        current_tick: 현재 틱 (i_c)
        fee_growth_global: 전역 fee growth (f_g)
        fee_growth_outside_lower: 하한 틱의 fee growth outside (f_o(i_l))
        fee_growth_outside_upper: 상한 틱의 fee growth outside (f_o(i_u))

    Returns:
        범위 내 fee growth (f_r)
    """
    f_b = fee_growth_below(tick_lower, current_tick, fee_growth_global, fee_growth_outside_lower)
    f_a = fee_growth_above(tick_upper, current_tick, fee_growth_global, fee_growth_outside_upper)

    # 언더플로우 처리 (uint256 연산과 동일하게)
    # Solidity에서는 unchecked 블록에서 자연스럽게 랩어라운드됨
    result = fee_growth_global - f_b - f_a

    # Python에서 음수가 될 수 있으므로 256비트 랩어라운드 시뮬레이션
    if result < 0:
        result += 2 ** 256

    return result


def calculate_uncollected_fees(
    liquidity: int,
    fee_growth_inside_current: int,
    fee_growth_inside_last: int
) -> int:
    """미수령 수수료 계산 (f_u)

    백서 Section 6.4.1 공식:
        f_u = l × (f_r(t_1) - f_r(t_0))

    Args:
        liquidity: 포지션 유동성 (l)
        fee_growth_inside_current: 현재 범위 내 fee growth (f_r(t_1))
        fee_growth_inside_last: 마지막 업데이트 시 fee growth (f_r(t_0))

    Returns:
        미수령 수수료 (Q128 형식)
    """
    # fee growth 차이 계산 (언더플로우 처리)
    fee_growth_delta = fee_growth_inside_current - fee_growth_inside_last
    if fee_growth_delta < 0:
        fee_growth_delta += 2 ** 256

    # 미수령 수수료 = 유동성 × fee growth 차이
    return liquidity * fee_growth_delta


def calculate_uncollected_fees_both_tokens(
    liquidity: int,
    tick_lower: int,
    tick_upper: int,
    current_tick: int,
    fee_growth_global_0: int,
    fee_growth_global_1: int,
    fee_growth_outside_lower_0: int,
    fee_growth_outside_lower_1: int,
    fee_growth_outside_upper_0: int,
    fee_growth_outside_upper_1: int,
    fee_growth_inside_last_0: int,
    fee_growth_inside_last_1: int
) -> FeeCalculationResult:
    """두 토큰의 미수령 수수료 계산

    백서 Section 6.3, 6.4의 전체 수수료 계산 파이프라인.

    필수 데이터 (9개 변수):
    - Global: f_g,0, f_g,1, i_c (3개)
    - Lower tick: f_o,0(i_l), f_o,1(i_l) (2개)
    - Upper tick: f_o,0(i_u), f_o,1(i_u) (2개)
    - Position: l, f_r,0(t_0), f_r,1(t_0) (3개, 여기서는 2개만 사용)

    Args:
        liquidity: 포지션 유동성 (l)
        tick_lower: 하한 틱 (i_l)
        tick_upper: 상한 틱 (i_u)
        current_tick: 현재 틱 (i_c)
        fee_growth_global_0: token0 전역 fee growth (f_g,0)
        fee_growth_global_1: token1 전역 fee growth (f_g,1)
        fee_growth_outside_lower_0: 하한 틱 token0 outside (f_o,0(i_l))
        fee_growth_outside_lower_1: 하한 틱 token1 outside (f_o,1(i_l))
        fee_growth_outside_upper_0: 상한 틱 token0 outside (f_o,0(i_u))
        fee_growth_outside_upper_1: 상한 틱 token1 outside (f_o,1(i_u))
        fee_growth_inside_last_0: 마지막 업데이트 시 token0 inside (f_r,0(t_0))
        fee_growth_inside_last_1: 마지막 업데이트 시 token1 inside (f_r,1(t_0))

    Returns:
        FeeCalculationResult: 미수령 수수료 및 현재 fee growth inside
    """
    # Step 1: 현재 범위 내 fee growth 계산 (f_r(t_1))
    fee_growth_inside_0 = fee_growth_inside(
        tick_lower, tick_upper, current_tick,
        fee_growth_global_0,
        fee_growth_outside_lower_0,
        fee_growth_outside_upper_0
    )

    fee_growth_inside_1 = fee_growth_inside(
        tick_lower, tick_upper, current_tick,
        fee_growth_global_1,
        fee_growth_outside_lower_1,
        fee_growth_outside_upper_1
    )

    # Step 2: 미수령 수수료 계산 (f_u)
    uncollected_0_q128 = calculate_uncollected_fees(
        liquidity,
        fee_growth_inside_0,
        fee_growth_inside_last_0
    )

    uncollected_1_q128 = calculate_uncollected_fees(
        liquidity,
        fee_growth_inside_1,
        fee_growth_inside_last_1
    )

    # Q128 디코딩: 수수료를 토큰 최소 단위로 변환
    uncollected_0 = uncollected_0_q128 // Q128
    uncollected_1 = uncollected_1_q128 // Q128

    return FeeCalculationResult(
        uncollected_fees_0=uncollected_0,
        uncollected_fees_1=uncollected_1,
        fee_growth_inside_0=fee_growth_inside_0,
        fee_growth_inside_1=fee_growth_inside_1
    )


def decode_fee_growth(fee_growth_x128: int, decimals: int = 18) -> float:
    """Q128 인코딩된 fee growth를 human-readable 값으로 변환

    Args:
        fee_growth_x128: Q128 인코딩된 fee growth
        decimals: 토큰 소수점 자릿수

    Returns:
        Human-readable fee growth (토큰 단위)
    """
    return fee_growth_x128 / Q128 / (10 ** decimals)


def calculate_fee_growth_delta(
    fee_growth_current: int,
    fee_growth_previous: int
) -> int:
    """두 시점 간 fee growth 변화량 계산

    언더플로우를 고려한 차이 계산 (uint256 랩어라운드).

    Args:
        fee_growth_current: 현재 fee growth
        fee_growth_previous: 이전 fee growth

    Returns:
        fee growth 변화량
    """
    delta = fee_growth_current - fee_growth_previous
    if delta < 0:
        delta += 2 ** 256
    return delta


def estimate_fees_for_period(
    fee_growth_global_start_0: int,
    fee_growth_global_end_0: int,
    fee_growth_global_start_1: int,
    fee_growth_global_end_1: int,
    position_liquidity: int,
    total_liquidity: int,
    active_percentage: float = 100.0
) -> Tuple[int, int]:
    """기간 동안의 예상 수수료 계산 (단순화된 버전)

    이 함수는 전체 범위 유동성(unbounded)을 가정한 단순화된 계산입니다.
    정확한 계산을 위해서는 calculate_uncollected_fees_both_tokens를 사용하세요.

    Args:
        fee_growth_global_start_0: 시작 시점 token0 global fee growth
        fee_growth_global_end_0: 종료 시점 token0 global fee growth
        fee_growth_global_start_1: 시작 시점 token1 global fee growth
        fee_growth_global_end_1: 종료 시점 token1 global fee growth
        position_liquidity: 포지션 유동성
        total_liquidity: 풀 총 유동성
        active_percentage: 활성 유동성 비율 (0-100)

    Returns:
        (fees_token0, fees_token1) 튜플 (최소 단위)
    """
    # fee growth 변화량 계산
    fg_delta_0 = calculate_fee_growth_delta(fee_growth_global_end_0, fee_growth_global_start_0)
    fg_delta_1 = calculate_fee_growth_delta(fee_growth_global_end_1, fee_growth_global_start_1)

    # 활성 비율 적용
    active_ratio = active_percentage / 100.0

    # 수수료 = fee growth delta × 유동성 × 활성 비율
    fees_0 = int(fg_delta_0 * position_liquidity * active_ratio) // Q128
    fees_1 = int(fg_delta_1 * position_liquidity * active_ratio) // Q128

    return fees_0, fees_1
