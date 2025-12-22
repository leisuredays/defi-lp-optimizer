"""
Tick Math 테스트

tick_math.py의 함수들을 테스트합니다.
온체인 값과 비교하여 정확도를 검증합니다.
"""

import pytest
import math

from ..math.tick_math import (
    get_sqrt_ratio_at_tick,
    get_tick_at_sqrt_ratio,
    tick_to_price,
    price_to_tick,
    round_tick_to_spacing,
    MIN_TICK,
    MAX_TICK,
    MIN_SQRT_RATIO,
    MAX_SQRT_RATIO
)


class TestGetSqrtRatioAtTick:
    """get_sqrt_ratio_at_tick 테스트"""

    def test_min_tick(self):
        """최소 틱에서의 sqrtPrice"""
        result = get_sqrt_ratio_at_tick(MIN_TICK)
        assert result == MIN_SQRT_RATIO

    def test_max_tick(self):
        """최대 틱에서의 sqrtPrice"""
        result = get_sqrt_ratio_at_tick(MAX_TICK)
        # MAX_SQRT_RATIO와 같거나 약간 작을 수 있음
        assert result <= MAX_SQRT_RATIO
        assert result > 0

    def test_tick_0(self):
        """틱 0에서의 sqrtPrice (price = 1)"""
        result = get_sqrt_ratio_at_tick(0)
        # sqrtPrice at tick 0 should be approximately 2^96
        expected = 2 ** 96  # 79228162514264337593543950336
        assert abs(result - expected) < 10  # 작은 오차 허용

    def test_positive_tick(self):
        """양수 틱 테스트"""
        result = get_sqrt_ratio_at_tick(100)
        assert result > 2 ** 96  # 틱 0보다 큼

    def test_negative_tick(self):
        """음수 틱 테스트"""
        result = get_sqrt_ratio_at_tick(-100)
        assert result < 2 ** 96  # 틱 0보다 작음

    def test_invalid_tick_too_low(self):
        """유효 범위를 벗어난 틱 (너무 낮음)"""
        with pytest.raises(ValueError):
            get_sqrt_ratio_at_tick(MIN_TICK - 1)

    def test_invalid_tick_too_high(self):
        """유효 범위를 벗어난 틱 (너무 높음)"""
        with pytest.raises(ValueError):
            get_sqrt_ratio_at_tick(MAX_TICK + 1)


class TestGetTickAtSqrtRatio:
    """get_tick_at_sqrt_ratio 테스트"""

    def test_min_sqrt_ratio(self):
        """최소 sqrtRatio에서의 틱"""
        result = get_tick_at_sqrt_ratio(MIN_SQRT_RATIO)
        assert result == MIN_TICK

    def test_sqrt_ratio_at_tick_0(self):
        """sqrtPrice 2^96에서의 틱"""
        sqrt_price = 2 ** 96
        result = get_tick_at_sqrt_ratio(sqrt_price)
        assert result == 0

    def test_roundtrip(self):
        """틱 -> sqrtPrice -> 틱 왕복 테스트"""
        for tick in [-50000, -1000, 0, 1000, 50000]:
            sqrt_price = get_sqrt_ratio_at_tick(tick)
            result_tick = get_tick_at_sqrt_ratio(sqrt_price)
            assert result_tick == tick

    def test_invalid_sqrt_ratio_too_low(self):
        """유효 범위를 벗어난 sqrtRatio (너무 낮음)"""
        with pytest.raises(ValueError):
            get_tick_at_sqrt_ratio(MIN_SQRT_RATIO - 1)


class TestTickToPrice:
    """tick_to_price 테스트"""

    def test_tick_0_same_decimals(self):
        """틱 0, 동일 소수점 (가격 = 1)"""
        result = tick_to_price(0, 18, 18)
        assert abs(result - 1.0) < 1e-10

    def test_tick_0_different_decimals(self):
        """틱 0, 다른 소수점 (USDC/WETH 예시)"""
        # USDC: 6 decimals, WETH: 18 decimals
        result = tick_to_price(0, 6, 18)
        # price = 1.0001^0 / 10^(6-18) = 1 / 10^(-12) = 10^12
        expected = 1e12
        assert abs(result - expected) < 1

    def test_positive_tick(self):
        """양수 틱 테스트"""
        result = tick_to_price(1000, 18, 18)
        expected = 1.0001 ** 1000
        assert abs(result - expected) / expected < 1e-6

    def test_negative_tick(self):
        """음수 틱 테스트"""
        result = tick_to_price(-1000, 18, 18)
        expected = 1.0001 ** (-1000)
        assert abs(result - expected) / expected < 1e-6


class TestPriceToTick:
    """price_to_tick 테스트"""

    def test_price_1_same_decimals(self):
        """가격 1, 동일 소수점"""
        result = price_to_tick(1.0, 18, 18)
        assert result == 0

    def test_roundtrip(self):
        """가격 -> 틱 -> 가격 왕복 테스트"""
        for price in [0.001, 0.1, 1.0, 10.0, 1000.0]:
            tick = price_to_tick(price, 18, 18)
            result_price = tick_to_price(tick, 18, 18)
            # 틱 반올림으로 인한 오차 허용
            assert abs(result_price - price) / price < 0.01

    def test_invalid_price_zero(self):
        """가격 0은 유효하지 않음"""
        with pytest.raises(ValueError):
            price_to_tick(0, 18, 18)

    def test_invalid_price_negative(self):
        """음수 가격은 유효하지 않음"""
        with pytest.raises(ValueError):
            price_to_tick(-1.0, 18, 18)


class TestRoundTickToSpacing:
    """round_tick_to_spacing 테스트

    가장 가까운 유효 틱으로 반올림합니다.
    """

    def test_already_aligned(self):
        """이미 정렬된 틱"""
        assert round_tick_to_spacing(60, 60) == 60
        assert round_tick_to_spacing(120, 60) == 120
        assert round_tick_to_spacing(-60, 60) == -60
        assert round_tick_to_spacing(0, 60) == 0

    def test_round_nearest_positive(self):
        """양수 틱 - 가장 가까운 틱으로"""
        # 65는 60과 120 중 60이 더 가까움
        assert round_tick_to_spacing(65, 60) == 60
        # 89는 60과 120 중 60이 더 가까움 (89-60=29 < 120-89=31)
        assert round_tick_to_spacing(89, 60) == 60
        # 91은 60과 120 중 120이 더 가까움 (91-60=31 > 120-91=29)
        assert round_tick_to_spacing(91, 60) == 120
        # 정확히 중간인 경우 올림
        assert round_tick_to_spacing(90, 60) == 120

    def test_round_nearest_negative(self):
        """음수 틱 - 가장 가까운 틱으로"""
        # -65는 -60과 -120 중 -60이 더 가까움
        assert round_tick_to_spacing(-65, 60) == -60
        # -1은 0과 -60 중 0이 더 가까움
        assert round_tick_to_spacing(-1, 60) == 0
        # -31은 0과 -60 중 -60이 더 가까움 (정확히 중간이면 0 방향)
        assert round_tick_to_spacing(-31, 60) == -60
        # -29는 0과 -60 중 0이 더 가까움
        assert round_tick_to_spacing(-29, 60) == 0

    def test_different_spacings(self):
        """다양한 틱 간격 테스트"""
        # 500 fee tier -> spacing 10
        # 15는 10과 20 중 20이 더 가까움 (5 == 5, 올림)
        assert round_tick_to_spacing(15, 10) == 20
        # 14는 10과 20 중 10이 더 가까움
        assert round_tick_to_spacing(14, 10) == 10
        # -15는 -10과 -20 중 -10이 더 가까움
        assert round_tick_to_spacing(-15, 10) == -10
        # -16은 -10과 -20 중 -20이 더 가까움
        assert round_tick_to_spacing(-16, 10) == -20

        # 10000 fee tier -> spacing 200
        # 250은 200과 400 중 200이 더 가까움
        assert round_tick_to_spacing(250, 200) == 200
        # 350은 200과 400 중 400이 더 가까움
        assert round_tick_to_spacing(350, 200) == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
