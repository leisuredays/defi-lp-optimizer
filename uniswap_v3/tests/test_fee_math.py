"""
Fee Math 테스트

백서 Section 6.3, 6.4 기반 수수료 계산 함수들을 테스트합니다.
"""

import pytest

from ..math.fee_math import (
    fee_growth_above,
    fee_growth_below,
    fee_growth_inside,
    calculate_uncollected_fees,
    calculate_uncollected_fees_both_tokens,
    decode_fee_growth,
    calculate_fee_growth_delta
)
from ..constants import Q128


class TestFeeGrowthAbove:
    """fee_growth_above 테스트 (f_a)"""

    def test_current_tick_above_target(self):
        """현재 틱이 타겟 틱 위에 있을 때: f_a = f_g - f_o"""
        # i_c >= i 이면 f_a = f_g - f_o
        f_g = 1000
        f_o = 300
        result = fee_growth_above(tick_idx=100, current_tick=150,
                                   fee_growth_global=f_g, fee_growth_outside=f_o)
        assert result == f_g - f_o  # 700

    def test_current_tick_at_target(self):
        """현재 틱이 타겟 틱과 같을 때: f_a = f_g - f_o"""
        f_g = 1000
        f_o = 300
        result = fee_growth_above(tick_idx=100, current_tick=100,
                                   fee_growth_global=f_g, fee_growth_outside=f_o)
        assert result == f_g - f_o  # 700

    def test_current_tick_below_target(self):
        """현재 틱이 타겟 틱 아래에 있을 때: f_a = f_o"""
        f_g = 1000
        f_o = 300
        result = fee_growth_above(tick_idx=100, current_tick=50,
                                   fee_growth_global=f_g, fee_growth_outside=f_o)
        assert result == f_o  # 300


class TestFeeGrowthBelow:
    """fee_growth_below 테스트 (f_b)"""

    def test_current_tick_above_target(self):
        """현재 틱이 타겟 틱 위에 있을 때: f_b = f_o"""
        f_g = 1000
        f_o = 300
        result = fee_growth_below(tick_idx=100, current_tick=150,
                                   fee_growth_global=f_g, fee_growth_outside=f_o)
        assert result == f_o  # 300

    def test_current_tick_at_target(self):
        """현재 틱이 타겟 틱과 같을 때: f_b = f_o"""
        f_g = 1000
        f_o = 300
        result = fee_growth_below(tick_idx=100, current_tick=100,
                                   fee_growth_global=f_g, fee_growth_outside=f_o)
        assert result == f_o  # 300

    def test_current_tick_below_target(self):
        """현재 틱이 타겟 틱 아래에 있을 때: f_b = f_g - f_o"""
        f_g = 1000
        f_o = 300
        result = fee_growth_below(tick_idx=100, current_tick=50,
                                   fee_growth_global=f_g, fee_growth_outside=f_o)
        assert result == f_g - f_o  # 700


class TestFeeGrowthInside:
    """fee_growth_inside 테스트 (f_r)

    f_r = f_g - f_b(i_l) - f_a(i_u)
    """

    def test_current_tick_in_range(self):
        """현재 틱이 범위 내에 있을 때"""
        # tick_lower = 100, tick_upper = 200, current_tick = 150
        # f_b(100) = f_o_lower (i_c >= i_l)
        # f_a(200) = f_o_upper (i_c < i_u)
        f_g = 1000
        f_o_lower = 100
        f_o_upper = 200

        result = fee_growth_inside(
            tick_lower=100, tick_upper=200, current_tick=150,
            fee_growth_global=f_g,
            fee_growth_outside_lower=f_o_lower,
            fee_growth_outside_upper=f_o_upper
        )
        # f_b = f_o_lower = 100
        # f_a = f_o_upper = 200
        # f_r = 1000 - 100 - 200 = 700
        assert result == 700

    def test_current_tick_below_range(self):
        """현재 틱이 범위 아래에 있을 때"""
        # tick_lower = 100, tick_upper = 200, current_tick = 50
        f_g = 1000
        f_o_lower = 100
        f_o_upper = 200

        result = fee_growth_inside(
            tick_lower=100, tick_upper=200, current_tick=50,
            fee_growth_global=f_g,
            fee_growth_outside_lower=f_o_lower,
            fee_growth_outside_upper=f_o_upper
        )
        # f_b(100) = f_g - f_o_lower = 900 (i_c < i_l)
        # f_a(200) = f_o_upper = 200 (i_c < i_u)
        # f_r = 1000 - 900 - 200 = -100 (언더플로우 처리됨)
        # Python에서는 음수가 되므로 2^256 래핑
        expected = 2**256 - 100
        assert result == expected

    def test_current_tick_above_range(self):
        """현재 틱이 범위 위에 있을 때"""
        # tick_lower = 100, tick_upper = 200, current_tick = 250
        f_g = 1000
        f_o_lower = 100
        f_o_upper = 200

        result = fee_growth_inside(
            tick_lower=100, tick_upper=200, current_tick=250,
            fee_growth_global=f_g,
            fee_growth_outside_lower=f_o_lower,
            fee_growth_outside_upper=f_o_upper
        )
        # f_b(100) = f_o_lower = 100 (i_c >= i_l)
        # f_a(200) = f_g - f_o_upper = 800 (i_c >= i_u)
        # f_r = 1000 - 100 - 800 = 100
        assert result == 100


class TestCalculateUncollectedFees:
    """calculate_uncollected_fees 테스트 (f_u)

    f_u = l × (f_r(t_1) - f_r(t_0))
    """

    def test_basic_calculation(self):
        """기본 수수료 계산"""
        liquidity = 1000000
        f_r_current = 500 * Q128  # Q128 인코딩
        f_r_last = 100 * Q128

        result = calculate_uncollected_fees(liquidity, f_r_current, f_r_last)
        # f_u = 1000000 × (500 - 100) × Q128 (Q128 형식)
        expected = liquidity * (f_r_current - f_r_last)
        assert result == expected

    def test_zero_delta(self):
        """수수료 변화 없음"""
        liquidity = 1000000
        f_r = 100 * Q128

        result = calculate_uncollected_fees(liquidity, f_r, f_r)
        assert result == 0

    def test_underflow_handling(self):
        """언더플로우 처리 (fee growth가 래핑된 경우)"""
        liquidity = 1000000
        f_r_current = 100 * Q128
        f_r_last = 200 * Q128  # current < last (래핑 발생)

        result = calculate_uncollected_fees(liquidity, f_r_current, f_r_last)
        # 래핑 처리됨
        assert result > 0


class TestDecodeFeeGrowth:
    """decode_fee_growth 테스트"""

    def test_basic_decoding(self):
        """기본 디코딩"""
        # Q128 인코딩된 값
        fee_growth_x128 = Q128  # 1.0 토큰 (decimals 적용 전)
        result = decode_fee_growth(fee_growth_x128, decimals=18)
        # 1.0 / 10^18 = 1e-18
        assert abs(result - 1e-18) < 1e-30

    def test_large_value(self):
        """큰 값 디코딩"""
        fee_growth_x128 = 1000 * Q128
        result = decode_fee_growth(fee_growth_x128, decimals=6)  # USDC
        # 1000 / 10^6 = 0.001
        assert abs(result - 0.001) < 1e-10


class TestCalculateFeeGrowthDelta:
    """calculate_fee_growth_delta 테스트"""

    def test_normal_delta(self):
        """일반적인 델타 계산"""
        result = calculate_fee_growth_delta(1000, 500)
        assert result == 500

    def test_underflow_wraparound(self):
        """언더플로우 래핑"""
        result = calculate_fee_growth_delta(100, 200)
        expected = 2**256 - 100
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
