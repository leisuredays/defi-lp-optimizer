#!/usr/bin/env python3
"""
Backtest - DB 또는 CSV 데이터를 사용한 백테스트

Usage:
    # DB에서 데이터 로드 (기본)
    python backtest.py --pool WETH/USDT --range 10,20,30

    # 특정 기간 지정
    python backtest.py --pool WETH/USDT --range 20 --from 2024-01-01 --to 2024-12-31

    # CSV에서 데이터 로드
    python backtest.py --csv /path/to/data.csv --range 10
"""

import sys
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import matplotlib.pyplot as plt

from uniswap_v3.math import (
    get_sqrt_ratio_at_tick,
    round_tick_to_spacing,
    price_to_tick,
    tick_to_price,
    get_liquidity_for_amounts,
    get_amounts_for_liquidity,
    calculate_fee_growth_delta,
    optimal_token_split,
)
from uniswap_v3.constants import Q128, TICK_SPACINGS

# Import from pool_data_manager
from pool_data_manager import PoolDataDB, get_pool_config, POOL_REGISTRY


def load_from_db(
    pool_name: str,
    chain: str = "ethereum",
    from_date: str = None,
    to_date: str = None
) -> Optional[pd.DataFrame]:
    """DB에서 데이터 로드"""
    pool_config = get_pool_config(pool_name, chain)
    if not pool_config:
        print(f"❌ 풀 없음: {pool_name} ({chain})")
        return None

    db = PoolDataDB()

    from_ts = None
    to_ts = None

    if from_date:
        from_ts = int(datetime.strptime(from_date, '%Y-%m-%d').timestamp())
    if to_date:
        to_ts = int(datetime.strptime(to_date, '%Y-%m-%d').timestamp())

    # 데이터 범위 확인
    data_range = db.get_data_range(pool_config.pool_id, chain)
    if not data_range:
        print(f"❌ DB에 데이터 없음: {pool_name}")
        return None

    # 기본값: 전체 데이터
    if not from_ts:
        from_ts = data_range[0]
    if not to_ts:
        to_ts = data_range[1]

    rows = db.get_hour_data(pool_config.pool_id, chain, from_ts, to_ts)

    if not rows:
        print(f"❌ 조회된 데이터 없음")
        return None

    df = pd.DataFrame(rows)

    # 컬럼명 정리
    df = df.rename(columns={
        'period_start_unix': 'timestamp',
    })

    # timestamp를 datetime으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    print(f"✅ DB에서 {len(df):,}개 레코드 로드")
    print(f"   기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    return df


def run_backtest_with_timeseries(
    df: pd.DataFrame,
    investment_usd: float = 10000,
    range_pct: float = 0.10,
    token0_decimals: int = 18,
    token1_decimals: int = 6,
    fee_tier: int = 3000,
    invert_price: bool = False,
) -> Dict[str, Any]:
    """백테스트 실행 (시계열 데이터 포함)"""
    tick_spacing = TICK_SPACINGS[fee_tier]

    def get_price(tick):
        p = tick_to_price(tick, token0_decimals, token1_decimals)
        return 1/p if invert_price and p != 0 else p

    start_row = df.iloc[0]
    end_row = df.iloc[-1]

    start_tick = int(start_row['tick'])
    end_tick = int(end_row['tick'])

    P0 = get_price(start_tick)
    P1 = get_price(end_tick)

    Pa = P0 * (1 - range_pct)
    Pb = P0 * (1 + range_pct)

    tick_lower = round_tick_to_spacing(
        price_to_tick(Pb, token0_decimals, token1_decimals), tick_spacing
    )
    tick_upper = round_tick_to_spacing(
        price_to_tick(Pa, token0_decimals, token1_decimals), tick_spacing
    )
    if tick_lower > tick_upper:
        tick_lower, tick_upper = tick_upper, tick_lower

    sqrt_price_lower = get_sqrt_ratio_at_tick(tick_lower)
    sqrt_price_upper = get_sqrt_ratio_at_tick(tick_upper)
    sqrt_price_initial = get_sqrt_ratio_at_tick(start_tick)

    x_human, y_human = optimal_token_split(investment_usd, P0, Pa, Pb)
    x_amount = int(x_human * (10 ** token0_decimals))
    y_amount = int(y_human * (10 ** token1_decimals))

    L = get_liquidity_for_amounts(
        sqrt_price_initial, sqrt_price_lower, sqrt_price_upper,
        x_amount, y_amount
    )

    if L == 0:
        return {"error": "유동성이 0입니다"}

    hodl_token0 = x_amount
    hodl_token1 = y_amount

    # 시계열 데이터 수집
    timestamps = []
    prices = []
    hodl_values = []
    lp_values = []
    cumulative_fees = []
    il_pcts = []

    total_fees_usd = 0.0
    active_hours = 0

    prev_fg0 = int(df.iloc[0]['fee_growth_global_0_x128'])
    prev_fg1 = int(df.iloc[0]['fee_growth_global_1_x128'])

    for i in range(1, len(df)):
        row = df.iloc[i]
        current_tick = int(row['tick'])
        current_price = get_price(current_tick)

        in_range = tick_lower <= current_tick <= tick_upper

        if in_range:
            active_hours += 1
            curr_fg0 = int(row['fee_growth_global_0_x128'])
            curr_fg1 = int(row['fee_growth_global_1_x128'])

            delta_fg0 = calculate_fee_growth_delta(curr_fg0, prev_fg0)
            delta_fg1 = calculate_fee_growth_delta(curr_fg1, prev_fg1)

            fee0_raw = (L * delta_fg0) // Q128
            fee1_raw = (L * delta_fg1) // Q128

            # invert인 경우 token0이 스테이블코인이므로 fee 계산 조정
            if invert_price:
                fee0_usd = fee0_raw / 10**token0_decimals
                fee1_usd = (fee1_raw / 10**token1_decimals) * current_price
            else:
                fee0_usd = (fee0_raw / 10**token0_decimals) * current_price
                fee1_usd = fee1_raw / 10**token1_decimals

            total_fees_usd += fee0_usd + fee1_usd

        prev_fg0 = int(row['fee_growth_global_0_x128'])
        prev_fg1 = int(row['fee_growth_global_1_x128'])

        # 시계열 데이터 수집
        sqrt_price_current = get_sqrt_ratio_at_tick(current_tick)
        amt0, amt1 = get_amounts_for_liquidity(
            sqrt_price_current, sqrt_price_lower, sqrt_price_upper, L
        )
        if invert_price:
            lp_val = (amt0 / 10**token0_decimals) + (amt1 / 10**token1_decimals) * current_price
            hodl_val = (hodl_token0 / 10**token0_decimals) + (hodl_token1 / 10**token1_decimals) * current_price
        else:
            lp_val = (amt0 / 10**token0_decimals) * current_price + (amt1 / 10**token1_decimals)
            hodl_val = (hodl_token0 / 10**token0_decimals) * current_price + (hodl_token1 / 10**token1_decimals)
        il_pct = ((lp_val - hodl_val) / hodl_val * 100) if hodl_val > 0 else 0

        timestamps.append(row['timestamp'])
        prices.append(current_price)
        hodl_values.append(hodl_val)
        lp_values.append(lp_val)
        cumulative_fees.append(total_fees_usd)
        il_pcts.append(il_pct)

    sqrt_price_final = get_sqrt_ratio_at_tick(end_tick)
    amt0, amt1 = get_amounts_for_liquidity(
        sqrt_price_final, sqrt_price_lower, sqrt_price_upper, L
    )

    if invert_price:
        final_lp_value = (amt0 / 10**token0_decimals) + (amt1 / 10**token1_decimals) * P1
        final_hodl_value = (hodl_token0 / 10**token0_decimals) + (hodl_token1 / 10**token1_decimals) * P1
    else:
        final_lp_value = (amt0 / 10**token0_decimals) * P1 + (amt1 / 10**token1_decimals)
        final_hodl_value = (hodl_token0 / 10**token0_decimals) * P1 + (hodl_token1 / 10**token1_decimals)

    il_usd = final_lp_value - final_hodl_value
    il_pct_final = (il_usd / final_hodl_value) * 100 if final_hodl_value > 0 else 0

    net_pnl_usd = (final_lp_value + total_fees_usd) - investment_usd
    net_pnl_pct = (net_pnl_usd / investment_usd) * 100

    vs_hodl_usd = (final_lp_value + total_fees_usd) - final_hodl_value
    vs_hodl_pct = (vs_hodl_usd / final_hodl_value) * 100 if final_hodl_value > 0 else 0

    total_hours = len(df) - 1
    active_pct = (active_hours / total_hours) * 100 if total_hours > 0 else 0
    period_days = total_hours / 24
    apr_pct = (net_pnl_pct / period_days) * 365 if period_days > 0 else 0

    return {
        "range_pct": range_pct * 100,
        "price_lower": Pa,
        "price_upper": Pb,
        "start_price": P0,
        "end_price": P1,
        "investment_usd": investment_usd,
        "total_fees_usd": total_fees_usd,
        "final_lp_value": final_lp_value,
        "final_hodl_value": final_hodl_value,
        "il_pct": il_pct_final,
        "vs_hodl_usd": vs_hodl_usd,
        "active_pct": active_pct,
        "apr_pct": apr_pct,
        # 시계열 데이터
        "timestamps": timestamps,
        "prices": prices,
        "hodl_values": hodl_values,
        "lp_values": lp_values,
        "cumulative_fees": cumulative_fees,
        "il_pcts": il_pcts,
    }


def create_detailed_chart(results: List[Dict], output_path: str = None, pool_name: str = "", show: bool = False):
    """6패널 상세 차트 생성"""
    colors = ['blue', 'orange', 'green', 'red', 'purple']

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.patch.set_facecolor('white')

    first = results[0]
    timestamps = first['timestamps']
    P0 = first['start_price']
    P1 = first['end_price']

    # 1. Price with LP Ranges
    ax1 = axes[0, 0]
    ax1.plot(timestamps, first['prices'], 'k-', lw=1, label='Price')
    for i, r in enumerate(results):
        ax1.fill_between(timestamps, r['price_lower'], r['price_upper'],
                        alpha=0.15, color=colors[i], label=f"±{r['range_pct']:.0f}%")
    ax1.set_ylabel('Price (USD)')
    ax1.set_title('Price with LP Ranges', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(alpha=0.3)

    # 2. LP + Fees vs HODL
    ax2 = axes[0, 1]
    ax2.plot(timestamps, first['hodl_values'], 'k--', lw=2, label='HODL')
    for i, r in enumerate(results):
        lp_plus_fees = [lp + fee for lp, fee in zip(r['lp_values'], r['cumulative_fees'])]
        ax2.plot(timestamps, lp_plus_fees, color=colors[i], lw=1.5,
                label=f"LP+Fees ±{r['range_pct']:.0f}%")
    ax2.set_ylabel('Value (USD)')
    ax2.set_title('LP + Fees vs HODL', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(alpha=0.3)

    # 3. Impermanent Loss
    ax3 = axes[1, 0]
    for i, r in enumerate(results):
        ax3.plot(timestamps, r['il_pcts'], color=colors[i], lw=1.5,
                label=f"±{r['range_pct']:.0f}%: {r['il_pct']:.1f}%")
    ax3.axhline(y=0, color='black', linestyle='-', lw=0.5)
    ax3.set_ylabel('IL (%)')
    ax3.set_title('Impermanent Loss Over Time', fontweight='bold')
    ax3.legend(loc='lower left', fontsize=8)
    ax3.grid(alpha=0.3)

    # 4. Fee Accumulation
    ax4 = axes[1, 1]
    for i, r in enumerate(results):
        ax4.plot(timestamps, r['cumulative_fees'], color=colors[i], lw=2,
                label=f"±{r['range_pct']:.0f}%: ${r['total_fees_usd']:,.0f}")
    ax4.set_ylabel('Cumulative Fees (USD)')
    ax4.set_title('Fee Accumulation (ETH→USDT at earn time)', fontweight='bold')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(alpha=0.3)

    # 5. LP Value vs HODL (Before Fees)
    ax5 = axes[2, 0]
    ax5.plot(timestamps, first['hodl_values'], 'k--', lw=2, label='HODL')
    for i, r in enumerate(results):
        ax5.plot(timestamps, r['lp_values'], color=colors[i], lw=1.5,
                label=f"LP ±{r['range_pct']:.0f}%")
    ax5.set_ylabel('Value (USD)')
    ax5.set_title('LP Value vs HODL (Before Fees)', fontweight='bold')
    ax5.legend(loc='upper left', fontsize=8)
    ax5.grid(alpha=0.3)

    # 6. Final Value Comparison
    ax6 = axes[2, 1]
    categories = ['HODL'] + [f"±{r['range_pct']:.0f}%\nLP+Fees" for r in results]
    values = [first['final_hodl_value']] + [r['final_lp_value'] + r['total_fees_usd'] for r in results]
    bar_colors = ['gray'] + colors[:len(results)]

    bars = ax6.bar(categories, values, color=bar_colors, edgecolor='black')
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, val + max(values)*0.01,
                f'${val:,.0f}', ha='center', fontweight='bold', fontsize=9)
    ax6.set_ylabel('Final Value (USD)')
    ax6.set_title('Final Value Comparison', fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')

    # 요약 텍스트
    summary_lines = [f"Backtest | Investment: ${first['investment_usd']:,.0f} | P0: ${P0:,.0f} → P1: ${P1:,.0f}"]
    for r in results:
        summary_lines.append(
            f"±{r['range_pct']:.0f}%: Active {r['active_pct']:.0f}% | IL {r['il_pct']:.1f}% | "
            f"Fees ${r['total_fees_usd']:,.0f}|vsHODL${r['vs_hodl_usd']:+,.0f}"
        )

    fig.text(0.5, 0.01, '\n'.join(summary_lines), ha='center', fontsize=9,
            fontfamily='monospace', bbox=dict(facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ 차트 저장: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def run_backtest(
    df: pd.DataFrame,
    investment_usd: float = 10000,
    range_pct: float = 0.10,
    token0_decimals: int = 18,
    token1_decimals: int = 6,
    fee_tier: int = 3000,
    invert_price: bool = False,
) -> Dict[str, Any]:
    """백테스트 실행"""
    tick_spacing = TICK_SPACINGS[fee_tier]

    def get_price(tick):
        p = tick_to_price(tick, token0_decimals, token1_decimals)
        return 1/p if invert_price and p != 0 else p

    # 시작/종료 데이터
    start_row = df.iloc[0]
    end_row = df.iloc[-1]

    start_tick = int(start_row['tick'])
    end_tick = int(end_row['tick'])

    P0 = get_price(start_tick)
    P1 = get_price(end_tick)
    price_change_pct = ((P1 - P0) / P0) * 100

    # 가격 범위 설정
    Pa = P0 * (1 - range_pct)
    Pb = P0 * (1 + range_pct)

    # 틱으로 변환
    tick_lower = round_tick_to_spacing(
        price_to_tick(Pb, token0_decimals, token1_decimals), tick_spacing
    )
    tick_upper = round_tick_to_spacing(
        price_to_tick(Pa, token0_decimals, token1_decimals), tick_spacing
    )
    if tick_lower > tick_upper:
        tick_lower, tick_upper = tick_upper, tick_lower

    sqrt_price_lower = get_sqrt_ratio_at_tick(tick_lower)
    sqrt_price_upper = get_sqrt_ratio_at_tick(tick_upper)
    sqrt_price_initial = get_sqrt_ratio_at_tick(start_tick)

    # 최적 토큰 분할
    x_human, y_human = optimal_token_split(investment_usd, P0, Pa, Pb)
    x_amount = int(x_human * (10 ** token0_decimals))
    y_amount = int(y_human * (10 ** token1_decimals))

    # 유동성 계산
    L = get_liquidity_for_amounts(
        sqrt_price_initial, sqrt_price_lower, sqrt_price_upper,
        x_amount, y_amount
    )

    if L == 0:
        return {"error": "유동성이 0입니다"}

    # HODL 수량 저장
    hodl_token0 = x_amount
    hodl_token1 = y_amount

    # 수수료 누적 계산
    total_fees_usd = 0.0
    active_hours = 0

    prev_fg0 = int(df.iloc[0]['fee_growth_global_0_x128'])
    prev_fg1 = int(df.iloc[0]['fee_growth_global_1_x128'])

    for i in range(1, len(df)):
        row = df.iloc[i]
        current_tick = int(row['tick'])
        current_price = get_price(current_tick)

        in_range = tick_lower <= current_tick <= tick_upper

        if in_range:
            active_hours += 1

            curr_fg0 = int(row['fee_growth_global_0_x128'])
            curr_fg1 = int(row['fee_growth_global_1_x128'])

            delta_fg0 = calculate_fee_growth_delta(curr_fg0, prev_fg0)
            delta_fg1 = calculate_fee_growth_delta(curr_fg1, prev_fg1)

            fee0_raw = (L * delta_fg0) // Q128
            fee1_raw = (L * delta_fg1) // Q128

            if invert_price:
                fee0_usd = fee0_raw / 10**token0_decimals
                fee1_usd = (fee1_raw / 10**token1_decimals) * current_price
            else:
                fee0_usd = (fee0_raw / 10**token0_decimals) * current_price
                fee1_usd = fee1_raw / 10**token1_decimals

            total_fees_usd += fee0_usd + fee1_usd

        prev_fg0 = int(row['fee_growth_global_0_x128'])
        prev_fg1 = int(row['fee_growth_global_1_x128'])

    # 최종 포지션 가치 계산
    sqrt_price_final = get_sqrt_ratio_at_tick(end_tick)
    amt0, amt1 = get_amounts_for_liquidity(
        sqrt_price_final, sqrt_price_lower, sqrt_price_upper, L
    )

    if invert_price:
        final_lp_value = (amt0 / 10**token0_decimals) + (amt1 / 10**token1_decimals) * P1
        final_hodl_value = (hodl_token0 / 10**token0_decimals) + (hodl_token1 / 10**token1_decimals) * P1
    else:
        final_lp_value = (amt0 / 10**token0_decimals) * P1 + (amt1 / 10**token1_decimals)
        final_hodl_value = (hodl_token0 / 10**token0_decimals) * P1 + (hodl_token1 / 10**token1_decimals)

    # IL 계산
    il_usd = final_lp_value - final_hodl_value
    il_pct = (il_usd / final_hodl_value) * 100 if final_hodl_value > 0 else 0

    # 성과 지표
    net_pnl_usd = (final_lp_value + total_fees_usd) - investment_usd
    net_pnl_pct = (net_pnl_usd / investment_usd) * 100

    vs_hodl_usd = (final_lp_value + total_fees_usd) - final_hodl_value
    vs_hodl_pct = (vs_hodl_usd / final_hodl_value) * 100 if final_hodl_value > 0 else 0

    total_hours = len(df) - 1
    active_pct = (active_hours / total_hours) * 100 if total_hours > 0 else 0

    period_days = total_hours / 24
    apr_pct = (net_pnl_pct / period_days) * 365 if period_days > 0 else 0

    return {
        "period_days": round(period_days, 1),
        "range_pct": range_pct * 100,
        "investment_usd": investment_usd,
        "start_price": round(P0, 2),
        "end_price": round(P1, 2),
        "price_change_pct": round(price_change_pct, 2),
        "price_lower": round(Pa, 2),
        "price_upper": round(Pb, 2),
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "total_fees_usd": round(total_fees_usd, 2),
        "final_lp_value": round(final_lp_value, 2),
        "final_hodl_value": round(final_hodl_value, 2),
        "il_usd": round(il_usd, 2),
        "il_pct": round(il_pct, 2),
        "net_pnl_usd": round(net_pnl_usd, 2),
        "net_pnl_pct": round(net_pnl_pct, 2),
        "vs_hodl_usd": round(vs_hodl_usd, 2),
        "vs_hodl_pct": round(vs_hodl_pct, 2),
        "apr_pct": round(apr_pct, 2),
        "active_hours": active_hours,
        "total_hours": total_hours,
        "active_pct": round(active_pct, 2),
        "start_date": str(start_row['timestamp']),
        "end_date": str(end_row['timestamp']),
    }


def print_results(results: List[Dict[str, Any]], pool_name: str = ""):
    """결과 출력"""
    if not results:
        return

    first = results[0]

    print("\n" + "=" * 70)
    print(f"BACKTEST RESULTS - {pool_name}")
    print("=" * 70)

    print(f"\n📅 기간: {first['start_date']} ~ {first['end_date']}")
    print(f"   ({first['period_days']} days, {first['total_hours']} hours)")

    print(f"\n💰 투자금: ${first['investment_usd']:,.0f}")

    print(f"\n📊 가격 변동:")
    print(f"   시작: ${first['start_price']:,.2f}")
    print(f"   종료: ${first['end_price']:,.2f}")
    print(f"   변화: {first['price_change_pct']:+.2f}%")

    print("\n" + "-" * 70)
    print(f"{'Range':<10} {'Fees':>12} {'IL':>10} {'Net PnL':>12} {'vs HODL':>12} {'APR':>10} {'Active':>10}")
    print("-" * 70)

    for r in results:
        print(f"±{r['range_pct']:.0f}%{'':<5} "
              f"${r['total_fees_usd']:>10,.0f} "
              f"{r['il_pct']:>+9.1f}% "
              f"${r['net_pnl_usd']:>+10,.0f} "
              f"${r['vs_hodl_usd']:>+10,.0f} "
              f"{r['apr_pct']:>+9.1f}% "
              f"{r['active_pct']:>9.0f}%")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="DB 또는 CSV 데이터로 백테스트 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DB에서 전체 기간 백테스트
  python backtest.py --pool WETH/USDT --range 10,20,30

  # 특정 기간 지정
  python backtest.py --pool WETH/USDT --range 20 --from 2024-01-01 --to 2024-12-31

  # CSV 파일 사용
  python backtest.py --csv data.csv --range 10
        """
    )

    # 데이터 소스
    parser.add_argument("--pool", type=str, help="풀 이름 (DB 사용시)")
    parser.add_argument("--chain", type=str, default="ethereum", help="체인")
    parser.add_argument("--csv", type=str, help="CSV 파일 경로 (CSV 사용시)")

    # 기간
    parser.add_argument("--from", dest="from_date", type=str, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", type=str, help="종료일 (YYYY-MM-DD)")

    # 백테스트 설정
    parser.add_argument("--range", type=str, default="10", help="가격 범위 %% (쉼표 구분, 기본: 10)")
    parser.add_argument("--investment", type=float, default=10000, help="투자금 USD")
    parser.add_argument("--chart", type=str, nargs='?', const='__show__', default=None,
                        help="차트 표시 (파일 경로 지정시 저장, 미지정시 창으로 표시)")
    parser.add_argument("--no-chart", action="store_true", help="차트 표시 안함")
    parser.add_argument("--invert", action="store_true", help="가격 반전 (WETH/USDC 등 token0이 스테이블코인인 경우)")

    args = parser.parse_args()

    # 데이터 로드
    if args.csv:
        print(f"📂 CSV 로드: {args.csv}")
        df = pd.read_csv(args.csv)
        pool_name = "CSV Data"
        token0_dec = 18
        token1_dec = 6
        fee_tier = 3000
    elif args.pool:
        pool_config = get_pool_config(args.pool, args.chain)
        if not pool_config:
            print(f"❌ 풀 없음: {args.pool}")
            return

        df = load_from_db(args.pool, args.chain, args.from_date, args.to_date)
        if df is None:
            return

        pool_name = f"{args.pool} ({args.chain})"
        token0_dec = pool_config.token0_decimals
        token1_dec = pool_config.token1_decimals
        fee_tier = pool_config.fee_tier
    else:
        parser.print_help()
        print("\n❌ --pool 또는 --csv 중 하나를 지정하세요")
        return

    # 범위 파싱
    ranges = [float(r.strip()) / 100 for r in args.range.split(',')]

    # 백테스트 실행
    results = []
    chart_results = []

    # 차트 옵션 처리
    show_chart = args.chart == '__show__'
    save_chart = args.chart and args.chart != '__show__'
    need_chart = (args.chart is not None) and not args.no_chart

    for range_pct in ranges:
        if need_chart:
            # 차트용 시계열 데이터 수집
            result = run_backtest_with_timeseries(
                df=df,
                investment_usd=args.investment,
                range_pct=range_pct,
                token0_decimals=token0_dec,
                token1_decimals=token1_dec,
                fee_tier=fee_tier,
                invert_price=args.invert,
            )
            if "error" not in result:
                chart_results.append(result)

        # 기본 백테스트
        result = run_backtest(
            df=df,
            investment_usd=args.investment,
            range_pct=range_pct,
            token0_decimals=token0_dec,
            token1_decimals=token1_dec,
            fee_tier=fee_tier,
            invert_price=args.invert,
        )

        if "error" in result:
            print(f"❌ ±{range_pct*100:.0f}%: {result['error']}")
        else:
            results.append(result)

    print_results(results, pool_name)

    # 차트 생성
    if chart_results and need_chart:
        output_path = args.chart if save_chart else None
        create_detailed_chart(chart_results, output_path, pool_name, show=show_chart)


if __name__ == "__main__":
    main()
