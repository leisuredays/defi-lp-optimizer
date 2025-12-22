"""
Liquidity Math 테스트

유동성 계산 함수들을 테스트합니다.
"""

import pytest

from ..math.liquidity_math import (
    get_amount0_delta,
    get_amount1_delta,
    get_liquidity_for_amount0,
    get_liquidity_for_amount1,
    get_liquidity_for_amounts,
    get_amounts_for_liquidity
)
from ..math.tick_math import get_sqrt_ratio_at_tick
from ..constants import Q96


class TestGetAmountDeltas:
    """get_amount0_delta, get_amount1_delta 테스트"""

    def test_amount0_delta_basic(self):
        """amount0 변화량 기본 테스트"""
        sqrt_a = get_sqrt_ratio_at_tick(0)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        liquidity = 10**18

        result = get_amount0_delta(sqrt_a, sqrt_b, liquidity)
        assert result > 0

    def test_amount1_delta_basic(self):
        """amount1 변화량 기본 테스트"""
        sqrt_a = get_sqrt_ratio_at_tick(0)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        liquidity = 10**18

        result = get_amount1_delta(sqrt_a, sqrt_b, liquidity)
        assert result > 0

    def test_amount_deltas_swap_order(self):
        """sqrt 순서가 바뀌어도 결과 동일"""
        sqrt_a = get_sqrt_ratio_at_tick(0)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        liquidity = 10**18

        result1 = get_amount0_delta(sqrt_a, sqrt_b, liquidity)
        result2 = get_amount0_delta(sqrt_b, sqrt_a, liquidity)
        assert result1 == result2

    def test_amount_deltas_zero_liquidity(self):
        """유동성 0일 때"""
        sqrt_a = get_sqrt_ratio_at_tick(0)
        sqrt_b = get_sqrt_ratio_at_tick(100)

        result0 = get_amount0_delta(sqrt_a, sqrt_b, 0)
        result1 = get_amount1_delta(sqrt_a, sqrt_b, 0)
        assert result0 == 0
        assert result1 == 0


class TestGetLiquidityForAmounts:
    """get_liquidity_for_amount0, get_liquidity_for_amount1, get_liquidity_for_amounts 테스트"""

    def test_liquidity_for_amount0(self):
        """amount0에서 유동성 계산"""
        sqrt_a = get_sqrt_ratio_at_tick(0)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        amount0 = 10**18

        liquidity = get_liquidity_for_amount0(sqrt_a, sqrt_b, amount0)
        assert liquidity > 0

    def test_liquidity_for_amount1(self):
        """amount1에서 유동성 계산"""
        sqrt_a = get_sqrt_ratio_at_tick(0)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        amount1 = 10**18

        liquidity = get_liquidity_for_amount1(sqrt_a, sqrt_b, amount1)
        assert liquidity > 0

    def test_liquidity_for_amounts_below_range(self):
        """가격이 범위 아래일 때: token0만 사용"""
        sqrt_current = get_sqrt_ratio_at_tick(-200)
        sqrt_a = get_sqrt_ratio_at_tick(0)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        amount0 = 10**18
        amount1 = 10**18

        liquidity = get_liquidity_for_amounts(sqrt_current, sqrt_a, sqrt_b, amount0, amount1)
        expected = get_liquidity_for_amount0(sqrt_a, sqrt_b, amount0)
        assert liquidity == expected

    def test_liquidity_for_amounts_above_range(self):
        """가격이 범위 위일 때: token1만 사용"""
        sqrt_current = get_sqrt_ratio_at_tick(200)
        sqrt_a = get_sqrt_ratio_at_tick(0)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        amount0 = 10**18
        amount1 = 10**18

        liquidity = get_liquidity_for_amounts(sqrt_current, sqrt_a, sqrt_b, amount0, amount1)
        expected = get_liquidity_for_amount1(sqrt_a, sqrt_b, amount1)
        assert liquidity == expected

    def test_liquidity_for_amounts_in_range(self):
        """가격이 범위 내일 때: 둘 다 사용, 작은 값 반환"""
        sqrt_current = get_sqrt_ratio_at_tick(50)
        sqrt_a = get_sqrt_ratio_at_tick(0)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        amount0 = 10**18
        amount1 = 10**18

        liquidity = get_liquidity_for_amounts(sqrt_current, sqrt_a, sqrt_b, amount0, amount1)

        liq0 = get_liquidity_for_amount0(sqrt_current, sqrt_b, amount0)
        liq1 = get_liquidity_for_amount1(sqrt_a, sqrt_current, amount1)
        expected = min(liq0, liq1)

        assert liquidity == expected


class TestGetAmountsForLiquidity:
    """get_amounts_for_liquidity 테스트"""

    def test_amounts_below_range(self):
        """가격이 범위 아래일 때: token0만 반환"""
        sqrt_current = get_sqrt_ratio_at_tick(-200)
        sqrt_a = get_sqrt_ratio_at_tick(0)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        liquidity = 10**18

        amount0, amount1 = get_amounts_for_liquidity(sqrt_current, sqrt_a, sqrt_b, liquidity)
        assert amount0 > 0
        assert amount1 == 0

    def test_amounts_above_range(self):
        """가격이 범위 위일 때: token1만 반환"""
        sqrt_current = get_sqrt_ratio_at_tick(200)
        sqrt_a = get_sqrt_ratio_at_tick(0)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        liquidity = 10**18

        amount0, amount1 = get_amounts_for_liquidity(sqrt_current, sqrt_a, sqrt_b, liquidity)
        assert amount0 == 0
        assert amount1 > 0

    def test_amounts_in_range(self):
        """가격이 범위 내일 때: 둘 다 반환"""
        sqrt_current = get_sqrt_ratio_at_tick(50)
        sqrt_a = get_sqrt_ratio_at_tick(0)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        liquidity = 10**18

        amount0, amount1 = get_amounts_for_liquidity(sqrt_current, sqrt_a, sqrt_b, liquidity)
        assert amount0 > 0
        assert amount1 > 0

    def test_roundtrip(self):
        """유동성 -> 토큰 -> 유동성 왕복 테스트"""
        sqrt_current = get_sqrt_ratio_at_tick(50)
        sqrt_a = get_sqrt_ratio_at_tick(0)
        sqrt_b = get_sqrt_ratio_at_tick(100)
        original_liquidity = 10**18

        # 유동성 -> 토큰
        amount0, amount1 = get_amounts_for_liquidity(sqrt_current, sqrt_a, sqrt_b, original_liquidity)

        # 토큰 -> 유동성
        result_liquidity = get_liquidity_for_amounts(sqrt_current, sqrt_a, sqrt_b, amount0, amount1)

        # 반올림으로 인한 작은 차이 허용
        assert abs(result_liquidity - original_liquidity) < original_liquidity * 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
