"""
백서 공식 테스트

Uniswap V3 백서 Section 6.3, 6.4.1의 공식을 검증합니다.
"""

import pytest
import math
import sys
import os

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.uniswap_v3_adapter import (
    calculate_liquidity as calculate_liquidity_whitepaper,
    get_token_amounts as get_token_amounts_whitepaper,
    calculate_il as calculate_il_whitepaper,
    calculate_fees as calculate_fees_whitepaper,
    price_to_tick,
    tick_to_price
)


class TestLiquidityCalculation:
    """유동성 계산 테스트"""

    def test_liquidity_basic(self):
        """기본 유동성 계산 테스트"""
        # $10,000 투자, 현재 가격 $3,000, 범위 ±20%
        investment = 10000
        P0 = 3000
        Pa = P0 * 0.8  # 2400
        Pb = P0 * 1.2  # 3600

        L = calculate_liquidity_whitepaper(investment, P0, Pa, Pb)

        assert L > 0, "유동성은 양수여야 함"

        # 계산된 유동성으로 초기 토큰 양 검증
        x0, y0 = get_token_amounts_whitepaper(L, P0, Pa, Pb)

        # 초기 가치 = x0 * P0 + y0 ≈ investment
        initial_value = x0 * P0 + y0
        assert abs(initial_value - investment) < investment * 0.01, \
            f"초기 가치 오차: {initial_value} vs {investment}"

    def test_liquidity_narrow_range(self):
        """좁은 범위에서 유동성이 높아지는지 테스트"""
        investment = 10000
        P0 = 3000

        # 넓은 범위: ±50%
        L_wide = calculate_liquidity_whitepaper(investment, P0, P0 * 0.5, P0 * 1.5)

        # 좁은 범위: ±10%
        L_narrow = calculate_liquidity_whitepaper(investment, P0, P0 * 0.9, P0 * 1.1)

        assert L_narrow > L_wide, "좁은 범위의 유동성이 더 높아야 함"

    def test_liquidity_invalid_inputs(self):
        """잘못된 입력 처리 테스트"""
        assert calculate_liquidity_whitepaper(0, 3000, 2400, 3600) == 0.0
        assert calculate_liquidity_whitepaper(10000, 0, 2400, 3600) == 0.0
        assert calculate_liquidity_whitepaper(10000, 3000, 3600, 2400) == 0.0  # Pa >= Pb


class TestTokenAmounts:
    """토큰 양 계산 테스트 (백서 공식 6.29, 6.30)"""

    def test_token_amounts_in_range(self):
        """범위 내 가격에서 토큰 양 테스트"""
        L = 1000
        P = 3000
        Pa = 2400
        Pb = 3600

        x, y = get_token_amounts_whitepaper(L, P, Pa, Pb)

        assert x > 0, "token0 양은 양수여야 함"
        assert y > 0, "token1 양은 양수여야 함"

    def test_token_amounts_below_range(self):
        """가격이 범위 아래일 때 테스트"""
        L = 1000
        P = 2000  # Pa(2400) 아래
        Pa = 2400
        Pb = 3600

        x, y = get_token_amounts_whitepaper(L, P, Pa, Pb)

        assert x > 0, "token0 양은 양수여야 함"
        assert y == 0, "token1 양은 0이어야 함"

    def test_token_amounts_above_range(self):
        """가격이 범위 위일 때 테스트"""
        L = 1000
        P = 4000  # Pb(3600) 위
        Pa = 2400
        Pb = 3600

        x, y = get_token_amounts_whitepaper(L, P, Pa, Pb)

        assert x == 0, "token0 양은 0이어야 함"
        assert y > 0, "token1 양은 양수여야 함"


class TestILCalculation:
    """IL 계산 테스트 (백서 Section 6.4.1)"""

    def test_il_no_price_change(self):
        """가격 변화 없을 때 IL = 0"""
        result = calculate_il_whitepaper(
            investment=10000,
            mint_price=3000,
            current_price=3000,
            Pa=2400,
            Pb=3600
        )

        assert abs(result['il_pct']) < 0.001, "가격 변화 없으면 IL ≈ 0"
        assert abs(result['hodl_value'] - result['lp_value']) < 1, \
            "가격 변화 없으면 HODL ≈ LP"

    def test_il_price_increase(self):
        """가격 상승 시 IL 테스트"""
        result = calculate_il_whitepaper(
            investment=10000,
            mint_price=3000,
            current_price=3600,  # 20% 상승
            Pa=2400,
            Pb=4200
        )

        # IL은 가격이 변하면 음수 (LP < HODL)
        assert result['il_pct'] < 0, "가격 상승 시 IL < 0 (손실)"
        assert result['lp_value'] < result['hodl_value'], "LP < HODL"

    def test_il_price_decrease(self):
        """가격 하락 시 IL 테스트"""
        result = calculate_il_whitepaper(
            investment=10000,
            mint_price=3000,
            current_price=2400,  # 20% 하락
            Pa=1800,
            Pb=3600
        )

        # IL은 가격이 변하면 음수 (LP < HODL)
        assert result['il_pct'] < 0, "가격 하락 시 IL < 0 (손실)"
        assert result['lp_value'] < result['hodl_value'], "LP < HODL"

    def test_il_symmetry(self):
        """IL은 가격 변화 방향에 관계없이 비슷해야 함"""
        # 20% 상승
        result_up = calculate_il_whitepaper(
            investment=10000,
            mint_price=3000,
            current_price=3600,
            Pa=2400,
            Pb=4200
        )

        # 20% 하락
        result_down = calculate_il_whitepaper(
            investment=10000,
            mint_price=3000,
            current_price=2400,
            Pa=1800,
            Pb=3600
        )

        # IL 크기는 대략 비슷해야 함 (범위 비대칭으로 완전히 같지는 않음)
        il_up = abs(result_up['il_pct'])
        il_down = abs(result_down['il_pct'])

        assert abs(il_up - il_down) < 0.05, \
            f"IL 대칭성: 상승={il_up:.4f}, 하락={il_down:.4f}"


class TestFeeCalculation:
    """수수료 계산 테스트 (백서 Section 6.3, 6.4)"""

    def test_fees_basic(self):
        """기본 수수료 계산 테스트"""
        Q128 = 2 ** 128

        # feeGrowthGlobal이 증가했을 때 수수료 계산
        fees = calculate_fees_whitepaper(
            position_liquidity=1000000,  # 유동성
            fee_growth_global_start_0=0,
            fee_growth_global_end_0=Q128 * 100,  # 100 token0 per L
            fee_growth_global_start_1=0,
            fee_growth_global_end_1=Q128 * 50,  # 50 token1 per L
            active_percentage=100,
            token0_decimals=6,
            token1_decimals=18,
            current_price=3000
        )

        assert fees > 0, "수수료는 양수여야 함"

    def test_fees_zero_activity(self):
        """활성 유동성이 0일 때 수수료 = 0"""
        Q128 = 2 ** 128

        fees = calculate_fees_whitepaper(
            position_liquidity=1000000,
            fee_growth_global_start_0=0,
            fee_growth_global_end_0=Q128 * 100,
            fee_growth_global_start_1=0,
            fee_growth_global_end_1=Q128 * 50,
            active_percentage=0,  # 활성 비율 0
            token0_decimals=6,
            token1_decimals=18,
            current_price=3000
        )

        assert fees == 0.0, "활성 비율 0이면 수수료 = 0"

    def test_fees_proportional_to_liquidity(self):
        """수수료가 유동성에 비례하는지 테스트"""
        Q128 = 2 ** 128

        fees_L1 = calculate_fees_whitepaper(
            position_liquidity=1000,
            fee_growth_global_start_0=0,
            fee_growth_global_end_0=Q128 * 100,
            fee_growth_global_start_1=0,
            fee_growth_global_end_1=0,
            active_percentage=100,
            token0_decimals=6,
            token1_decimals=18,
            current_price=3000
        )

        fees_L2 = calculate_fees_whitepaper(
            position_liquidity=2000,  # 2배 유동성
            fee_growth_global_start_0=0,
            fee_growth_global_end_0=Q128 * 100,
            fee_growth_global_start_1=0,
            fee_growth_global_end_1=0,
            active_percentage=100,
            token0_decimals=6,
            token1_decimals=18,
            current_price=3000
        )

        # 유동성 2배 → 수수료 2배
        assert abs(fees_L2 / fees_L1 - 2.0) < 0.01, \
            f"수수료가 유동성에 비례해야 함: {fees_L2} vs {fees_L1 * 2}"


class TestTickConversion:
    """틱 변환 테스트"""

    def test_price_tick_roundtrip(self):
        """가격 → 틱 → 가격 왕복 테스트"""
        original_price = 3000

        tick = price_to_tick(original_price, token0_decimals=6, token1_decimals=18)
        recovered_price = tick_to_price(tick, token0_decimals=6, token1_decimals=18)

        # 틱은 이산적이므로 약간의 오차 허용
        assert abs(recovered_price - original_price) / original_price < 0.01, \
            f"가격 왕복 오차: {original_price} → {tick} → {recovered_price}"

    def test_tick_ordering(self):
        """높은 가격 → 높은 틱"""
        tick_low = price_to_tick(2000, token0_decimals=6, token1_decimals=18)
        tick_high = price_to_tick(4000, token0_decimals=6, token1_decimals=18)

        assert tick_high > tick_low, "높은 가격 → 높은 틱"


class TestIntegration:
    """통합 테스트"""

    def test_full_position_lifecycle(self):
        """전체 포지션 라이프사이클 테스트"""
        # 1. 포지션 생성
        investment = 10000
        mint_price = 3000
        Pa = 2400
        Pb = 3600

        # 2. 유동성 계산
        L = calculate_liquidity_whitepaper(investment, mint_price, Pa, Pb)
        assert L > 0

        # 3. 초기 토큰 양
        x0, y0 = get_token_amounts_whitepaper(L, mint_price, Pa, Pb)
        initial_value = x0 * mint_price + y0
        assert abs(initial_value - investment) < investment * 0.01

        # 4. 가격 변화
        new_price = 3300

        # 5. 새 토큰 양
        x1, y1 = get_token_amounts_whitepaper(L, new_price, Pa, Pb)
        new_value = x1 * new_price + y1

        # 6. IL 계산
        il_result = calculate_il_whitepaper(investment, mint_price, new_price, Pa, Pb)
        assert il_result['L'] == L
        assert abs(il_result['lp_value'] - new_value) < 1

        # 7. HODL vs LP 검증
        hodl_value = x0 * new_price + y0
        assert abs(il_result['hodl_value'] - hodl_value) < 1

        print(f"\n=== 포지션 라이프사이클 테스트 ===")
        print(f"투자금: ${investment:,.0f}")
        print(f"Mint 가격: ${mint_price:,.0f}")
        print(f"현재 가격: ${new_price:,.0f}")
        print(f"범위: [${Pa:,.0f}, ${Pb:,.0f}]")
        print(f"유동성 L: {L:,.2f}")
        print(f"초기 토큰: {x0:.6f} ETH + ${y0:,.2f} USDC")
        print(f"현재 토큰: {x1:.6f} ETH + ${y1:,.2f} USDC")
        print(f"HODL 가치: ${hodl_value:,.2f}")
        print(f"LP 가치: ${new_value:,.2f}")
        print(f"IL: {il_result['il_pct'] * 100:.4f}%")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
