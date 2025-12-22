#!/usr/bin/env python3
"""
Test script for backtest API fixes.

Tests:
1. IL calculation with price inversion
2. Realized vs unrealized IL separation
3. TVL-based fee calculation
4. Rebalancing detection
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from app.ml.environment import UniswapV3LPEnv
from stable_baselines3 import PPO


def test_environment_fixes():
    """Test environment fixes with real data"""
    print("=" * 60)
    print("Testing Environment Fixes")
    print("=" * 60)

    # Load config
    import yaml
    config_path = project_root / "config" / "training_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    pool_file = project_root / config['data']['pool_file']
    df = pd.read_parquet(pool_file)
    df = df.sort_values('periodStartUnix').reset_index(drop=True)

    # Use test split
    n = len(df)
    val_end = int(n * (config['data']['train_split'] + config['data']['val_split']))
    test_df = df.iloc[val_end:].copy()

    print(f"Test data: {len(test_df)} hours")
    print(f"Price range: {test_df['close'].min():.8f} - {test_df['close'].max():.8f}")
    print(f"Average price: {test_df['close'].mean():.8f}")

    # Pool config
    pool_config = {
        'protocol': int(test_df['protocol_id'].iloc[0]),
        'feeTier': int(test_df['fee_tier'].iloc[0]),
        'token0': {'decimals': int(test_df['token0_decimals'].iloc[0])},
        'token1': {'decimals': int(test_df['token1_decimals'].iloc[0])}
    }

    print(f"\nPool config:")
    print(f"  Protocol: {pool_config['protocol']}")
    print(f"  Fee tier: {pool_config['feeTier']} bps")
    print(f"  Token0 decimals: {pool_config['token0']['decimals']}")
    print(f"  Token1 decimals: {pool_config['token1']['decimals']}")

    # Create environment with debug mode
    env = UniswapV3LPEnv(
        historical_data=test_df,
        pool_config=pool_config,
        initial_investment=10000,
        episode_length_hours=100,  # Short episode for quick test
        reward_weights={'alpha': 1.0, 'beta': 0.8, 'gamma': 0.2},
        obs_dim=24,
        debug=True  # Enable debug logging
    )

    obs, info = env.reset()
    print(f"\n--- Episode Start ---")
    print(f"Initial price: {env.initial_price:.8f}")
    print(f"Initial price (human): ${1/env.initial_price:.2f} per token0")
    print(f"Position range: [{env.position_min_price:.8f}, {env.position_max_price:.8f}]")
    print(f"Initial tokens: {env.initial_token_amounts}")

    # Load model
    model_path = project_root / "models" / "ppo_arbitrum_usdc_weth_005.zip"
    if model_path.exists():
        model = PPO.load(str(model_path))
        print(f"\nModel loaded: {model_path}")
    else:
        print(f"\nModel not found: {model_path}")
        print("Using random actions instead")
        model = None

    # Run episode
    done = False
    step = 0
    rebalance_events = []

    print("\n--- Running Episode ---")

    while not done and step < 50:  # Limit to 50 steps for quick test
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        if info.get('rebalanced', False):
            rebalance_events.append({
                'step': step,
                'fees': info['cumulative_fees'],
                'realized_il': info.get('realized_il', 0),
                'unrealized_il': info.get('unrealized_il', 0)
            })
            print(f"\n  *** Rebalance at step {step} ***")

        # Print every 10 steps
        if step % 10 == 0:
            print(f"\n  Step {step}:")
            print(f"    Cumulative fees: ${info['cumulative_fees']:.2f}")
            print(f"    Realized IL: ${info.get('realized_il', info['cumulative_il']):.2f}")
            print(f"    Unrealized IL: ${info.get('unrealized_il', 0):.2f}")
            print(f"    Net return: ${info['net_return']:.2f}")

    print("\n--- Episode Summary ---")
    print(f"Steps completed: {step}")
    print(f"Total rebalances: {env.total_rebalances}")
    print(f"Cumulative fees: ${env.cumulative_fees:.2f}")
    print(f"Cumulative IL (realized): ${env.cumulative_il:.2f}")
    print(f"Cumulative gas: ${env.cumulative_gas:.2f}")
    print(f"Net return: ${env.cumulative_fees - env.cumulative_il - env.cumulative_gas:.2f}")

    # Verify calculations
    print("\n--- Verification ---")

    # Check if IL is reasonable
    if env.cumulative_il > env.initial_investment * 0.5:
        print(f"WARNING: Realized IL ({env.cumulative_il:.2f}) > 50% of investment")
    else:
        print(f"OK: Realized IL is reasonable ({env.cumulative_il:.2f})")

    # Check if fees are reasonable
    hours = step
    max_expected_fees = env.initial_investment * 0.01 * hours  # 1% per hour max
    if env.cumulative_fees > max_expected_fees:
        print(f"WARNING: Fees ({env.cumulative_fees:.2f}) seem high for {hours} hours")
    else:
        print(f"OK: Fees are reasonable ({env.cumulative_fees:.2f})")

    # Check if net return is reasonable
    net_return = env.cumulative_fees - env.cumulative_il - env.cumulative_gas
    if abs(net_return) > env.initial_investment * 0.5:
        print(f"WARNING: Net return ({net_return:.2f}) > 50% of investment")
    else:
        print(f"OK: Net return is reasonable ({net_return:.2f})")

    return True


def test_il_calculation():
    """Test IL calculation specifically"""
    print("\n" + "=" * 60)
    print("Testing IL Calculation")
    print("=" * 60)

    import math

    # Test case: WETH/USDC pool
    # Price is in token0/token1 (WETH/USDC) format
    initial_price = 0.000337  # 0.000337 WETH per USDC = 1/2967 USDC per WETH
    current_price = 0.000350  # Price moved up 3.8%

    # Convert to human-readable
    initial_price_human = 1 / initial_price  # ~2967 USDC per WETH
    current_price_human = 1 / current_price  # ~2857 USDC per WETH

    print(f"Initial price: {initial_price:.8f} (${initial_price_human:.2f} per WETH)")
    print(f"Current price: {current_price:.8f} (${current_price_human:.2f} per WETH)")

    # IL formula
    price_ratio = current_price_human / initial_price_human
    sqrt_ratio = math.sqrt(price_ratio)
    il_percentage = 2 * sqrt_ratio / (1 + price_ratio) - 1

    print(f"\nPrice ratio: {price_ratio:.4f}")
    print(f"IL percentage: {il_percentage * 100:.4f}%")

    # For a $10k position
    investment = 10000
    expected_il = abs(il_percentage) * investment
    print(f"Expected IL for $10k position: ${expected_il:.2f}")

    # Verify it's reasonable
    if expected_il < 100:
        print("OK: IL is reasonable for 3.8% price change")
    else:
        print("WARNING: IL seems too high")


if __name__ == "__main__":
    try:
        test_il_calculation()
        test_environment_fixes()
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
