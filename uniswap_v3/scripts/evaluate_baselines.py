#!/usr/bin/env python
"""
Evaluate ML model against baseline strategies on test data.

Baselines:
- HODL (50/50 ETH/USDT)
- Fixed-range LP: ±10%, ±20%, ±30%, ±40%, ±50%, Full range

Usage:
  python evaluate_with_baselines.py --model fold_01.zip --fold 1
  python evaluate_with_baselines.py --model fold_01_1000k.zip --fold 1
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from uniswap_v3.ml import UniswapV3LPEnv
from uniswap_v3.math import (
    tick_to_sqrt, snap_tick, price_to_tick, tick_to_price,
    amounts_to_L, L_to_amounts, fee_delta, split_tokens
)
from uniswap_v3.constants import Q128, TICK_SPACINGS


def load_data_from_db(db_path: Path) -> pd.DataFrame:
    """Load all data from SQLite database."""
    conn = sqlite3.connect(db_path)
    pool_info = pd.read_sql('SELECT * FROM pools LIMIT 1', conn)
    df = pd.read_sql('SELECT * FROM pool_hour_data ORDER BY period_start_unix', conn)
    conn.close()

    df = df.rename(columns={
        'period_start_unix': 'periodStartUnix',
        'fee_growth_global_0_x128': 'feeGrowthGlobal0X128',
        'fee_growth_global_1_x128': 'feeGrowthGlobal1X128',
    })

    df['protocol_id'] = 0
    df['fee_tier'] = pool_info['fee_tier'].iloc[0]
    df['token0_symbol'] = pool_info['token0_symbol'].iloc[0]
    df['token1_symbol'] = pool_info['token1_symbol'].iloc[0]
    df['token0_decimals'] = pool_info['token0_decimals'].iloc[0]
    df['token1_decimals'] = pool_info['token1_decimals'].iloc[0]

    return df.reset_index(drop=True)


def create_env(data: pd.DataFrame, episode_length: int = None):
    """Create environment from data slice."""
    if episode_length is None:
        episode_length = len(data) - 10

    pool_config = {
        'protocol': int(data['protocol_id'].iloc[0]),
        'feeTier': int(data['fee_tier'].iloc[0]),
        'token0': {'decimals': int(data['token0_decimals'].iloc[0])},
        'token1': {'decimals': int(data['token1_decimals'].iloc[0])}
    }

    env = UniswapV3LPEnv(
        historical_data=data,
        pool_config=pool_config,
        initial_investment=10000,
        episode_length_hours=episode_length,
        obs_dim=28
    )
    return Monitor(env)


def evaluate_hodl(data: pd.DataFrame, initial_investment: float = 10000,
                  token0_decimals: int = 18, token1_decimals: int = 6,
                  invert_price: bool = True) -> dict:
    """Evaluate HODL strategy (50/50 split)."""
    def get_price(tick):
        p = tick_to_price(tick, token0_decimals, token1_decimals)
        return 1/p if invert_price and p != 0 else p

    start_tick = int(data['tick'].iloc[0])
    end_tick = int(data['tick'].iloc[-1])

    start_price = get_price(start_tick)
    end_price = get_price(end_tick)

    # 50/50 split: half in ETH, half in USDT
    usdt_amount = initial_investment / 2
    eth_amount = (initial_investment / 2) / start_price

    # Final value
    final_value = usdt_amount + (eth_amount * end_price)
    net_return = final_value - initial_investment

    return {
        'strategy': 'HODL (50/50)',
        'start_price': start_price,
        'end_price': end_price,
        'price_change_pct': (end_price - start_price) / start_price * 100,
        'initial_value': initial_investment,
        'final_value': final_value,
        'net_return': net_return,
        'vs_hodl': 0,  # HODL is the reference
        'return_pct': net_return / initial_investment * 100,
        'fees': 0,
        'il': 0,
        'gas': 0,
        'rebalances': 0,
        'in_range_pct': 0
    }


def evaluate_fixed_range_lp(data: pd.DataFrame, range_pct: float,
                            initial_investment: float = 10000,
                            fee_tier: int = 3000,
                            token0_decimals: int = 18,
                            token1_decimals: int = 6,
                            invert_price: bool = True,
                            gas_cost_usd: float = 10.0) -> dict:
    """
    Evaluate fixed-range LP strategy using accurate Uniswap V3 math.

    Rebalances when price touches the LP boundary (out of range).
    Uses the same fee calculation as backtest.py with actual on-chain fee growth data.
    """
    tick_spacing = TICK_SPACINGS[fee_tier]
    pool_fee_rate = fee_tier / 1_000_000  # e.g., 3000 → 0.003
    slippage_rate = 0.001  # 0.1%

    def get_price(tick):
        p = tick_to_price(tick, token0_decimals, token1_decimals)
        return 1/p if invert_price and p != 0 else p

    def calc_swap_cost(old_tokens, new_tokens, price):
        """Calculate swap cost for rebalancing.

        Args:
            old_tokens: (token0_amount, token1_amount) - WETH in ETH units, USDT in USD
            new_tokens: (token0_amount, token1_amount) - same units
            price: USD per ETH (e.g., 1500)
        """
        delta0 = new_tokens[0] - old_tokens[0]  # ETH units
        delta1 = new_tokens[1] - old_tokens[1]  # USD units

        if delta0 > 0 and delta1 < 0:
            # Need more WETH, have excess USDT → swap USDT to WETH
            # delta1 is already in USD
            swap_amount_usd = abs(delta1)
        elif delta0 < 0 and delta1 > 0:
            # Need more USDT, have excess WETH → swap WETH to USDT
            # delta0 is in ETH, convert to USD
            swap_amount_usd = abs(delta0) * price
        else:
            return 0.0

        return swap_amount_usd * (pool_fee_rate + slippage_rate)

    start_row = data.iloc[0]
    end_row = data.iloc[-1]

    start_tick = int(start_row['tick'])
    end_tick = int(end_row['tick'])

    P0 = get_price(start_tick)
    P1 = get_price(end_tick)

    # Full range doesn't need rebalancing
    if range_pct >= 1.0:
        Pa = P0 * 0.0001
        Pb = P0 * 10000
        strategy_name = "Full Range LP"
        rebalance_on_boundary = False
    else:
        Pa = P0 * (1 - range_pct)
        Pb = P0 * (1 + range_pct)
        strategy_name = f"Fixed ±{int(range_pct*100)}% LP"
        rebalance_on_boundary = True

    # Current position state
    current_price = P0
    current_Pa = Pa
    current_Pb = Pb

    # Convert to ticks
    tick_lower = snap_tick(price_to_tick(current_Pb, token0_decimals, token1_decimals), tick_spacing)
    tick_upper = snap_tick(price_to_tick(current_Pa, token0_decimals, token1_decimals), tick_spacing)
    if tick_lower > tick_upper:
        tick_lower, tick_upper = tick_upper, tick_lower

    sqrt_price_lower = tick_to_sqrt(tick_lower)
    sqrt_price_upper = tick_to_sqrt(tick_upper)
    sqrt_price_initial = tick_to_sqrt(start_tick)

    # Calculate optimal token split and liquidity
    x_human, y_human = split_tokens(initial_investment, P0, current_Pa, current_Pb)
    x_amount = int(x_human * (10 ** token0_decimals))
    y_amount = int(y_human * (10 ** token1_decimals))

    L = amounts_to_L(sqrt_price_initial, sqrt_price_lower, sqrt_price_upper, x_amount, y_amount)

    if L == 0:
        return {'strategy': strategy_name, 'error': 'Zero liquidity'}

    # Track HODL amounts for current position (resets on each rebalance)
    hodl_token0 = x_amount
    hodl_token1 = y_amount
    position_mint_price = P0  # Price when current position was minted

    # Track current position value
    position_value = initial_investment

    # Accumulate metrics
    total_fees_usd = 0.0
    total_gas_usd = 0.0
    total_il_usd = 0.0  # Accumulated IL across all positions
    in_range_hours = 0
    rebalances = 0

    prev_fg0 = int(data.iloc[0]['feeGrowthGlobal0X128'])
    prev_fg1 = int(data.iloc[0]['feeGrowthGlobal1X128'])

    for i in range(1, len(data)):
        row = data.iloc[i]
        current_tick = int(row['tick'])
        current_price = get_price(current_tick)

        in_range = tick_lower <= current_tick <= tick_upper

        # Check if rebalancing needed (price out of range)
        if rebalance_on_boundary and not in_range:
            # Calculate current position value before rebalancing (no fee reinvestment)
            sqrt_price_now = tick_to_sqrt(current_tick)
            amt0, amt1 = L_to_amounts(sqrt_price_now, sqrt_price_lower, sqrt_price_upper, L)

            if invert_price:
                lp_value_now = (amt0 / 10**token0_decimals) + (amt1 / 10**token1_decimals) * current_price
            else:
                lp_value_now = (amt0 / 10**token0_decimals) * current_price + (amt1 / 10**token1_decimals)

            # Calculate IL for this position before closing
            if invert_price:
                hodl_value_now = (hodl_token0 / 10**token0_decimals) + (hodl_token1 / 10**token1_decimals) * current_price
            else:
                hodl_value_now = (hodl_token0 / 10**token0_decimals) * current_price + (hodl_token1 / 10**token1_decimals)
            position_il = hodl_value_now - lp_value_now
            total_il_usd += position_il

            # Set new range around current price
            current_Pa = current_price * (1 - range_pct)
            current_Pb = current_price * (1 + range_pct)

            # Convert to ticks
            new_tick_lower = snap_tick(price_to_tick(current_Pb, token0_decimals, token1_decimals), tick_spacing)
            new_tick_upper = snap_tick(price_to_tick(current_Pa, token0_decimals, token1_decimals), tick_spacing)
            if new_tick_lower > new_tick_upper:
                new_tick_lower, new_tick_upper = new_tick_upper, new_tick_lower

            # Calculate tokens needed for new range (use current LP value, no fee reinvestment)
            new_x_human, new_y_human = split_tokens(lp_value_now, current_price, current_Pa, current_Pb)
            new_x_amount = int(new_x_human * (10 ** token0_decimals))
            new_y_amount = int(new_y_human * (10 ** token1_decimals))

            # Calculate swap cost
            old_tokens = (amt0 / 10**token0_decimals, amt1 / 10**token1_decimals)
            new_tokens = (new_x_human, new_y_human)
            swap_cost = calc_swap_cost(old_tokens, new_tokens, current_price)

            # Apply costs to LP value
            total_gas_usd += gas_cost_usd + swap_cost
            lp_value_now -= gas_cost_usd + swap_cost

            # Calculate new liquidity with reduced value
            new_x_human, new_y_human = split_tokens(lp_value_now, current_price, current_Pa, current_Pb)
            new_x_amount = int(new_x_human * (10 ** token0_decimals))
            new_y_amount = int(new_y_human * (10 ** token1_decimals))

            sqrt_price_lower = tick_to_sqrt(new_tick_lower)
            sqrt_price_upper = tick_to_sqrt(new_tick_upper)
            L = amounts_to_L(sqrt_price_now, sqrt_price_lower, sqrt_price_upper, new_x_amount, new_y_amount)

            # Update HODL baseline for new position
            hodl_token0 = new_x_amount
            hodl_token1 = new_y_amount

            tick_lower = new_tick_lower
            tick_upper = new_tick_upper
            rebalances += 1

            # Don't count fees for this hour (we were out of range when they accumulated)
            # Position will be in range starting from the NEXT hour

        if in_range:
            in_range_hours += 1

            curr_fg0 = int(row['feeGrowthGlobal0X128'])
            curr_fg1 = int(row['feeGrowthGlobal1X128'])

            delta_fg0 = fee_delta(curr_fg0, prev_fg0)
            delta_fg1 = fee_delta(curr_fg1, prev_fg1)

            fee0_raw = (L * delta_fg0) // Q128
            fee1_raw = (L * delta_fg1) // Q128

            if invert_price:
                fee0_usd = fee0_raw / 10**token0_decimals
                fee1_usd = (fee1_raw / 10**token1_decimals) * current_price
            else:
                fee0_usd = (fee0_raw / 10**token0_decimals) * current_price
                fee1_usd = fee1_raw / 10**token1_decimals

            total_fees_usd += fee0_usd + fee1_usd

        prev_fg0 = int(row['feeGrowthGlobal0X128'])
        prev_fg1 = int(row['feeGrowthGlobal1X128'])

    # Calculate final position value
    sqrt_price_final = tick_to_sqrt(end_tick)
    amt0, amt1 = L_to_amounts(sqrt_price_final, sqrt_price_lower, sqrt_price_upper, L)

    if invert_price:
        final_lp_value = (amt0 / 10**token0_decimals) + (amt1 / 10**token1_decimals) * P1
        final_hodl_value = (hodl_token0 / 10**token0_decimals) + (hodl_token1 / 10**token1_decimals) * P1
    else:
        final_lp_value = (amt0 / 10**token0_decimals) * P1 + (amt1 / 10**token1_decimals)
        final_hodl_value = (hodl_token0 / 10**token0_decimals) * P1 + (hodl_token1 / 10**token1_decimals)

    # Add IL for the final position
    final_position_il = final_hodl_value - final_lp_value
    total_il_usd += final_position_il

    # Net return = LP value + fees - initial
    # Note: Gas costs are ALREADY deducted from LP value during each rebalance
    final_total_value = final_lp_value + total_fees_usd
    net_return = final_total_value - initial_investment

    # vs HODL comparison (compare against initial HODL, not current position HODL)
    # Calculate what initial tokens would be worth now
    initial_x_human, initial_y_human = split_tokens(initial_investment, P0, Pa, Pb)
    if invert_price:
        initial_hodl_final = initial_x_human + initial_y_human * P1
    else:
        initial_hodl_final = initial_x_human * P1 + initial_y_human
    vs_hodl = final_total_value - initial_hodl_final

    return {
        'strategy': strategy_name,
        'range_pct': range_pct,
        'price_lower': current_Pa,
        'price_upper': current_Pb,
        'start_price': P0,
        'end_price': P1,
        'price_change_pct': (P1 - P0) / P0 * 100,
        'in_range_hours': in_range_hours,
        'in_range_pct': in_range_hours / (len(data) - 1) * 100,
        'initial_value': initial_investment,
        'final_lp_value': final_total_value,  # LP + fees (gas already in LP value)
        'final_hodl_value': initial_hodl_final,
        'fees': total_fees_usd,
        'il': total_il_usd,
        'gas': total_gas_usd,
        'net_return': net_return,
        'vs_hodl': vs_hodl,
        'return_pct': net_return / initial_investment * 100,
        'rebalances': rebalances,
    }


def evaluate_ml_model(model, env, initial_investment: float = 10000) -> dict:
    """Run ML model on environment and return metrics."""
    obs, _ = env.reset()

    # Save initial tokens BEFORE any rebalancing (for HODL comparison)
    initial_tokens = list(env.unwrapped.initial_token_amounts)
    initial_price = env.unwrapped.initial_price

    done = False
    total_fees = 0
    total_il = 0
    total_gas = 0
    total_swap_cost = 0
    rebalances = 0
    in_range_hours = 0
    total_hours = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_hours += 1

        if 'fees' in info:
            total_fees += info['fees']
        if 'il' in info:
            total_il += info['il']
        if 'gas' in info:
            total_gas += info['gas']
        if 'swap_cost' in info:
            total_swap_cost += info['swap_cost']
        if info.get('rebalanced', False):
            rebalances += 1
        if info.get('in_range', False):
            in_range_hours += 1

    # Get final LP value from environment (tracks actual position value)
    # Note: current_position_value already includes fees and deducts IL/gas
    final_lp_value = env.unwrapped.current_position_value

    # Get final price
    final_price = env.unwrapped.features_df.iloc[
        env.unwrapped.episode_start_idx + env.unwrapped.current_step
    ]['close']

    # Calculate HODL value for comparison using ORIGINAL initial tokens
    # For WETH-USDT pool (invert_price=True):
    #   token0 = WETH, token1 = USDT
    #   HODL value = WETH * final_price + USDT
    if env.unwrapped.invert_price:
        final_hodl_value = initial_tokens[0] * final_price + initial_tokens[1]
    else:
        final_hodl_value = initial_tokens[0] + initial_tokens[1] * final_price

    # Net return = final LP value - initial investment
    # Note: fees are already included in current_position_value
    net_return = final_lp_value - initial_investment

    # vs HODL comparison
    vs_hodl = final_lp_value - final_hodl_value

    return {
        'strategy': 'ML Model (PPO)',
        'initial_value': initial_investment,
        'final_lp_value': final_lp_value,
        'final_hodl_value': final_hodl_value,
        'start_price': initial_price,
        'end_price': final_price,
        'price_change_pct': (final_price - initial_price) / initial_price * 100,
        'fees': total_fees,
        'il': total_il,
        'gas': total_gas,
        'swap_cost': total_swap_cost,
        'net_return': net_return,
        'vs_hodl': vs_hodl,
        'return_pct': net_return / initial_investment * 100,
        'rebalances': rebalances,
        'in_range_hours': in_range_hours,
        'in_range_pct': in_range_hours / total_hours * 100 if total_hours > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with baselines')
    parser.add_argument('--model', type=str, help='Model filename (optional for baseline-only)')
    parser.add_argument('--fold', type=int, default=1, help='Fold number')
    parser.add_argument('--dataset', choices=['train', 'test', 'both', 'full'], default='test')
    parser.add_argument('--hours', type=int, default=0, help='Limit to N hours (0=no limit)')

    args = parser.parse_args()

    # Paths
    base_dir = project_root / "uniswap_v3/models/rolling_wfe"
    db_path = project_root / "uniswap_v3/data/pool_data.db"

    model_path = None
    if args.model:
        model_path = base_dir / "models" / args.model
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            sys.exit(1)

    # Load data
    print("Loading data...")
    data = load_data_from_db(db_path)

    datasets = {}

    if args.dataset == 'full':
        # Use full dataset
        if args.hours > 0:
            datasets['Full'] = data.iloc[:args.hours].reset_index(drop=True)
        else:
            datasets['Full'] = data.reset_index(drop=True)
    else:
        # Configuration for rolling window
        WINDOW_HOURS = 180 * 24  # 4320
        TEST_HOURS = 30 * 24     # 720

        # Calculate data ranges
        train_start = (args.fold - 1) * TEST_HOURS
        train_end = train_start + WINDOW_HOURS
        test_start = train_end
        test_end = test_start + TEST_HOURS

        if args.dataset in ['train', 'both']:
            datasets['Train'] = data.iloc[train_start:train_end].reset_index(drop=True)
        if args.dataset in ['test', 'both']:
            datasets['Test'] = data.iloc[test_start:test_end].reset_index(drop=True)

    for dataset_name, dataset_data in datasets.items():
        print(f"\n{'='*70}")
        print(f" {dataset_name.upper()} SET EVALUATION - {len(dataset_data)} hours ({len(dataset_data)/24:.0f} days)")
        print(f"{'='*70}")

        # Get pool config
        fee_tier = int(dataset_data['fee_tier'].iloc[0])
        token0_dec = int(dataset_data['token0_decimals'].iloc[0])
        token1_dec = int(dataset_data['token1_decimals'].iloc[0])

        # Calculate actual prices from ticks
        # tick_to_price returns price of token0 (WETH) in terms of token1 (USDT)
        # For WETH/USDT pool, this gives ETH price in USD directly
        start_price = tick_to_price(int(dataset_data['tick'].iloc[0]), token0_dec, token1_dec)
        end_price = tick_to_price(int(dataset_data['tick'].iloc[-1]), token0_dec, token1_dec)
        print(f"Price: ${start_price:,.2f} → ${end_price:,.2f} ({(end_price-start_price)/start_price*100:+.1f}%)")
        print()

        results = []

        # 1. HODL
        hodl = evaluate_hodl(
            dataset_data,
            token0_decimals=token0_dec,
            token1_decimals=token1_dec,
            invert_price=False  # tick_to_price already returns ETH/USD
        )
        results.append(hodl)

        # 2. Fixed-range LP strategies
        for range_pct in [0.10, 0.20, 0.30, 0.40, 0.50, 1.0]:
            fixed_lp = evaluate_fixed_range_lp(
                dataset_data, range_pct,
                fee_tier=fee_tier,
                token0_decimals=token0_dec,
                token1_decimals=token1_dec,
                invert_price=False  # tick_to_price already returns ETH/USD
            )
            if 'error' not in fixed_lp:
                results.append(fixed_lp)

        # 3. ML Model (optional)
        if model_path:
            print(f"Loading model: {args.model}")
            env = create_env(dataset_data)
            test_env = DummyVecEnv([lambda: create_env(dataset_data)])
            model = PPO.load(str(model_path), env=test_env, device="cuda")
            ml_result = evaluate_ml_model(model, env)
            ml_result['start_price'] = start_price
            ml_result['end_price'] = end_price
            results.append(ml_result)

        # Print results table
        print(f"{'Strategy':<20} {'Final Value':>12} {'Net Return':>12} {'Fees':>10} {'IL':>10} {'Gas':>8} {'Rebal':>6} {'In-Range':>10}")
        print("-" * 105)

        for r in results:
            # Final Value = final_lp_value (already includes fees for all strategies)
            final_total = r.get('final_lp_value', r.get('final_value', 10000))

            net = r.get('net_return', 0)
            fees = r.get('fees', 0)
            il = r.get('il', 0)
            gas = r.get('gas', 0)
            rebal = r.get('rebalances', 0)
            in_range = r.get('in_range_pct', 0)

            print(f"{r['strategy']:<20} ${final_total:>10,.0f} ${net:>+10,.0f} ${fees:>9,.0f} ${il:>9,.0f} ${gas:>7,.0f} {rebal:>6} {in_range:>9.1f}%")

        # Find best strategy
        print()
        best = max(results, key=lambda x: x.get('net_return', float('-inf')))
        print(f"🏆 Best Strategy: {best['strategy']} (${best['net_return']:+,.2f})")

        # ML vs HODL comparison
        ml = results[-1]
        hodl = results[0]
        diff = ml['net_return'] - hodl['net_return']
        print(f"📊 ML vs HODL: ${diff:+,.2f} ({'Better' if diff > 0 else 'Worse'})")


if __name__ == "__main__":
    main()
