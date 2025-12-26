#!/usr/bin/env python
"""
Evaluate a model variant (e.g., 1M, 1.5M steps) on test data for WFE validation.

Usage:
  python evaluate_model_variant.py --model fold_01_1000k.zip --fold 1
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

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from uniswap_v3.ml import UniswapV3LPEnv


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


def evaluate_model(model_name: str, fold: int):
    """Evaluate a model variant on test data."""

    # Paths
    base_dir = project_root / "uniswap_v3/models/rolling_wfe"
    model_path = base_dir / "models" / model_name
    db_path = project_root / "uniswap_v3/data/pool_data.db"

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Configuration
    WINDOW_DAYS = 180
    TEST_DAYS = 30
    WINDOW_HOURS = WINDOW_DAYS * 24
    TEST_HOURS = TEST_DAYS * 24

    # Calculate data ranges
    train_start = (fold - 1) * TEST_HOURS
    train_end = train_start + WINDOW_HOURS
    test_start = train_end
    test_end = test_start + TEST_HOURS

    # Load data
    print("Loading data...")
    data = load_data_from_db(db_path)
    train_data = data.iloc[train_start:train_end].reset_index(drop=True)
    test_data = data.iloc[test_start:test_end].reset_index(drop=True)

    print(f"Train data: {len(train_data)} hours ({len(train_data)/24:.0f} days)")
    print(f"Test data: {len(test_data)} hours ({len(test_data)/24:.0f} days)")

    # Load model
    print(f"\nLoading model: {model_path}")
    test_env = DummyVecEnv([lambda: create_env(test_data)])
    model = PPO.load(str(model_path), env=test_env, device="cuda")

    # Evaluate on training data
    print("\n=== Training Set Evaluation ===")
    train_env = create_env(train_data)
    train_return = evaluate_episode(model, train_env)

    # Evaluate on test data
    print("\n=== Test Set Evaluation ===")
    test_env_single = create_env(test_data)
    test_return = evaluate_episode(model, test_env_single)

    # WFE calculation
    wfe = 100.0 if test_return >= train_return else (test_return / train_return * 100) if train_return > 0 else 0

    print("\n" + "="*60)
    print(f"MODEL: {model_name}")
    print("="*60)
    print(f"In-Sample Return:  ${train_return:,.2f}")
    print(f"Out-Sample Return: ${test_return:,.2f}")
    print(f"WFE Score:         {wfe:.1f}%")

    if wfe >= 100:
        print("Rating: EXCELLENT (OOS >= IS)")
    elif wfe >= 70:
        print("Rating: GOOD")
    elif wfe >= 50:
        print("Rating: ACCEPTABLE")
    else:
        print("Rating: POOR (potential overfitting)")

    return {
        'model': model_name,
        'train_return': train_return,
        'test_return': test_return,
        'wfe': wfe
    }


def evaluate_episode(model, env):
    """Run one full episode and return net return."""
    obs, _ = env.reset()
    done = False
    total_fees = 0
    total_il = 0
    total_gas = 0
    rebalances = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if 'fees' in info:
            total_fees += info['fees']
        if 'il' in info:
            total_il += info['il']
        if 'gas' in info:
            total_gas += info['gas']
        if info.get('rebalanced', False):
            rebalances += 1

    net_return = total_fees - total_il - total_gas

    print(f"  Fees: ${total_fees:,.2f}")
    print(f"  IL: ${total_il:,.2f}")
    print(f"  Gas: ${total_gas:,.2f}")
    print(f"  Net Return: ${net_return:,.2f}")
    print(f"  Rebalances: {rebalances}")

    return net_return


def main():
    parser = argparse.ArgumentParser(description='Evaluate model variant')
    parser.add_argument('--model', type=str, required=True, help='Model filename (e.g., fold_01_1000k.zip)')
    parser.add_argument('--fold', type=int, required=True, help='Fold number')

    args = parser.parse_args()
    evaluate_model(args.model, args.fold)


if __name__ == "__main__":
    main()
