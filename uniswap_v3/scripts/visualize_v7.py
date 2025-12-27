#!/usr/bin/env python
"""
Visualize v7 model (discrete action space) on train/test sets.
"""
import sys
sys.path.insert(0, '/home/zekiya/liquidity/uniswap-v3-simulator')

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from stable_baselines3 import PPO
from uniswap_v3.ml import UniswapV3LPEnv


def load_data(db_path: str, start_idx: int = 0, length: int = 1000):
    """Load pool data from database."""
    conn = sqlite3.connect(db_path)
    pool_info = pd.read_sql('SELECT * FROM pools LIMIT 1', conn)
    df = pd.read_sql('SELECT * FROM pool_hour_data ORDER BY period_start_unix', conn)
    conn.close()

    token0_dec = int(pool_info['token0_decimals'].iloc[0])
    token1_dec = int(pool_info['token1_decimals'].iloc[0])
    fee_tier = int(pool_info['fee_tier'].iloc[0])

    df = df.rename(columns={
        'period_start_unix': 'periodStartUnix',
        'fee_growth_global_0_x128': 'feeGrowthGlobal0X128',
        'fee_growth_global_1_x128': 'feeGrowthGlobal1X128',
    })
    df['protocol_id'] = 0
    df['fee_tier'] = fee_tier
    df['token0_symbol'] = pool_info['token0_symbol'].iloc[0]
    df['token1_symbol'] = pool_info['token1_symbol'].iloc[0]
    df['token0_decimals'] = token0_dec
    df['token1_decimals'] = token1_dec

    end_idx = min(start_idx + length + 20, len(df))
    data = df.iloc[start_idx:end_idx].reset_index(drop=True)

    pool_config = {
        'protocol': 0,
        'feeTier': fee_tier,
        'token0': {'decimals': token0_dec},
        'token1': {'decimals': token1_dec}
    }

    return data, pool_config


def run_model_episode(model: PPO, data: pd.DataFrame, pool_config: dict,
                      episode_length: int = 720, seed: int = 42,
                      tick_width_options: list = None):
    """Run discrete action model on data."""
    env = UniswapV3LPEnv(
        historical_data=data,
        pool_config=pool_config,
        initial_investment=10000,
        episode_length_hours=episode_length
    )

    # Override tick width options if provided (for older models)
    if tick_width_options:
        env.TICK_WIDTH_OPTIONS = tick_width_options
        from gymnasium import spaces
        env.action_space = spaces.Discrete(len(tick_width_options))

    obs, _ = env.reset(seed=seed)

    # Get first action to set initial range
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)  # Discrete action

    # Set initial position with this action (simulate first step)
    if action > 0:
        tick_width = env.TICK_WIDTH_OPTIONS[action]
        current_tick = env._get_current_tick()
        half_width = tick_width // 2
        tick_spacing = env._get_tick_spacing()

        new_lower = ((current_tick - half_width) // tick_spacing) * tick_spacing
        new_upper = ((current_tick + half_width) // tick_spacing) * tick_spacing + tick_spacing

        from uniswap_v3.math import tick_to_price
        env.position_min_price = tick_to_price(new_lower)
        env.position_max_price = tick_to_price(new_upper)
        env.tick_lower = new_lower
        env.tick_upper = new_upper

    # Tracking data
    tracking = {
        'timestamp': [],
        'price': [],
        'min_range': [],
        'max_range': [],
        'in_range': [],
        'rebalanced': [],
        'fees': [],
        'cumulative_fees': [],
        'cumulative_il': [],
        'unrealized_il': [],
        'cumulative_gas': [],
        'net_return': [],
        'reward': [],
        'action': [],
        'tick_width': [],
    }

    done = False
    step = 0

    while not done and step < episode_length:
        current_idx = env.episode_start_idx + env.current_step
        if current_idx >= len(env.features_df):
            break
        current_data = env.features_df.iloc[current_idx]
        timestamp = datetime.fromtimestamp(current_data['periodStartUnix'])
        price = current_data['close']

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record tracking data
        tracking['timestamp'].append(timestamp)
        tracking['price'].append(price)
        tracking['min_range'].append(env.position_min_price)
        tracking['max_range'].append(env.position_max_price)
        tracking['in_range'].append(env.position_min_price < price < env.position_max_price)
        tracking['rebalanced'].append(info.get('rebalanced', False))
        tracking['fees'].append(info.get('fees', 0))
        tracking['cumulative_fees'].append(info.get('cumulative_fees', 0))
        tracking['cumulative_il'].append(info.get('cumulative_il', 0))
        tracking['unrealized_il'].append(info.get('unrealized_il', 0))
        tracking['cumulative_gas'].append(info.get('cumulative_gas', 0))
        tracking['net_return'].append(info.get('net_return', 0))
        tracking['reward'].append(reward)
        tracking['action'].append(action)
        tracking['tick_width'].append(env.TICK_WIDTH_OPTIONS[action] if action > 0 else 0)

        step += 1

    return pd.DataFrame(tracking)


def plot_model_behavior(df: pd.DataFrame, title: str = "Model Behavior",
                        save_path: str = None):
    """Create visualization."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # === Panel 1: Price with LP Range ===
    ax1 = axes[0]
    ax1.plot(df['timestamp'], df['price'], 'b-', linewidth=1.5, label='Price', zorder=3)
    ax1.fill_between(df['timestamp'], df['min_range'], df['max_range'],
                     alpha=0.3, color='green', label='LP Range', zorder=1)

    # Mark in-range periods
    in_range_mask = df['in_range']
    if in_range_mask.any():
        ax1.scatter(df.loc[in_range_mask, 'timestamp'],
                   df.loc[in_range_mask, 'price'],
                   c='green', s=10, alpha=0.5, label='In Range', zorder=2)

    # Mark rebalancing events
    rebal_mask = df['rebalanced']
    if rebal_mask.any():
        for ts in df.loc[rebal_mask, 'timestamp']:
            ax1.axvline(x=ts, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.scatter(df.loc[rebal_mask, 'timestamp'],
                   df.loc[rebal_mask, 'price'],
                   c='red', s=100, marker='v', label='Rebalance', zorder=4)

    ax1.set_ylabel('Price (USDT)', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Price & LP Range', fontsize=12)

    # === Panel 2: Cumulative Returns ===
    ax2 = axes[1]
    ax2.plot(df['timestamp'], df['cumulative_fees'], 'g-', linewidth=1.5, label='Fees')
    ax2.plot(df['timestamp'], -df['cumulative_il'], 'r-', linewidth=1.5, label='-IL (realized)')
    ax2.plot(df['timestamp'], -df['unrealized_il'], 'r--', linewidth=1, alpha=0.5, label='-IL (unrealized)')
    ax2.plot(df['timestamp'], -df['cumulative_gas'], 'orange', linewidth=1.5, label='-Gas')

    actual_return = df['cumulative_fees'] - df['cumulative_il'] - df['cumulative_gas']
    ax2.plot(df['timestamp'], actual_return, 'b-', linewidth=2, label='Net Return')

    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.fill_between(df['timestamp'], 0, actual_return,
                     where=actual_return > 0, alpha=0.3, color='green')
    ax2.fill_between(df['timestamp'], 0, actual_return,
                     where=actual_return < 0, alpha=0.3, color='red')

    ax2.set_ylabel('USD', fontsize=11)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Cumulative Returns', fontsize=12)

    # === Panel 3: Range Width ===
    ax3 = axes[2]
    range_width_pct = (df['max_range'] - df['min_range']) / df['price'] * 100
    ax3.fill_between(df['timestamp'], 0, range_width_pct, alpha=0.5, color='purple')
    ax3.plot(df['timestamp'], range_width_pct, 'purple', linewidth=1)

    if rebal_mask.any():
        for ts in df.loc[rebal_mask, 'timestamp']:
            ax3.axvline(x=ts, color='red', linestyle='--', alpha=0.7)

    ax3.set_ylabel('Range Width (%)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('LP Range Width (% of Price)', fontsize=12)

    # === Panel 4: In-Range Status ===
    ax4 = axes[3]
    in_range_numeric = df['in_range'].astype(int)
    ax4.fill_between(df['timestamp'], 0, in_range_numeric,
                     alpha=0.5, color='green', step='post')
    ax4.set_ylabel('In Range', fontsize=11)
    ax4.set_ylim(-0.1, 1.1)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Out', 'In'])
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Position In-Range Status', fontsize=12)
    ax4.set_xlabel('Time', fontsize=11)

    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()
    return fig


def print_summary(df: pd.DataFrame, dataset_name: str = ""):
    """Print episode summary."""
    print(f"\n{'='*60}")
    print(f"EPISODE SUMMARY - {dataset_name}")
    print("="*60)

    total_hours = len(df)
    in_range_hours = df['in_range'].sum()
    total_rebalances = df['rebalanced'].sum()

    print(f"Duration:        {total_hours} hours ({total_hours/24:.1f} days)")
    print(f"In-range hours:  {in_range_hours} ({in_range_hours/total_hours*100:.1f}%)")
    print(f"Rebalances:      {total_rebalances}")
    print()
    print(f"Total Fees:      ${df['cumulative_fees'].iloc[-1]:,.2f}")
    print(f"Realized IL:     ${df['cumulative_il'].iloc[-1]:,.2f}")
    print(f"Unrealized IL:   ${df['unrealized_il'].iloc[-1]:,.2f}")
    print(f"Total Gas:       ${df['cumulative_gas'].iloc[-1]:,.2f}")
    actual_net = df['cumulative_fees'].iloc[-1] - df['cumulative_il'].iloc[-1] - df['cumulative_gas'].iloc[-1]
    print(f"Net Return:      ${actual_net:,.2f}")
    print()
    print(f"Price Start:     ${df['price'].iloc[0]:,.2f}")
    print(f"Price End:       ${df['price'].iloc[-1]:,.2f}")
    print(f"Price Change:    {(df['price'].iloc[-1]/df['price'].iloc[0]-1)*100:+.2f}%")
    print()

    avg_width_pct = ((df['max_range'] - df['min_range']) / df['price']).mean() * 100
    print(f"Avg Range Width: {avg_width_pct:.2f}%")

    # Action distribution
    action_counts = df['action'].value_counts().sort_index()
    print(f"\nAction Distribution:")
    for action, count in action_counts.items():
        print(f"  Action {action}: {count} times ({count/len(df)*100:.1f}%)")


def main():
    db_path = "/home/zekiya/liquidity/uniswap-v3-simulator/uniswap_v3/data/pool_data.db"
    model_path = "/home/zekiya/liquidity/uniswap-v3-simulator/uniswap_v3/models/rolling_wfe/models/fold_01_v7_il_realized_500k.zip"

    # Fold 1 data ranges (7500h train, 1500h test per paper)
    TRAIN_START = 0
    TRAIN_LENGTH = 7500
    TEST_START = 7500
    TEST_LENGTH = 720  # 30 days for visualization

    # v7 model was trained with this action space
    V7_TICK_WIDTH_OPTIONS = [0, 500, 1000, 2000]

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    print(f"Model action space: {model.action_space}")
    print(f"Using tick width options: {V7_TICK_WIDTH_OPTIONS}")

    # === Training Set (last 30 days) ===
    train_vis_start = TRAIN_START + TRAIN_LENGTH - 720  # Last 30 days of training
    print(f"\nLoading training data from index {train_vis_start}...")
    train_data, pool_config = load_data(db_path, train_vis_start, 720)
    print(f"Training data range: {datetime.fromtimestamp(train_data['periodStartUnix'].iloc[0])} to "
          f"{datetime.fromtimestamp(train_data['periodStartUnix'].iloc[-1])}")

    print("\nRunning model on training set...")
    train_tracking = run_model_episode(model, train_data, pool_config, 720, seed=42,
                                        tick_width_options=V7_TICK_WIDTH_OPTIONS)
    print_summary(train_tracking, "TRAINING SET (last 30 days)")
    train_tracking.to_csv('fold1_v7_train_data.csv', index=False)
    plot_model_behavior(train_tracking,
                        title="V7 Model - Training Set (Last 30 Days)",
                        save_path="fold1_v7_train_evaluation.png")

    # === Test Set ===
    print(f"\nLoading test data from index {TEST_START}...")
    test_data, pool_config = load_data(db_path, TEST_START, TEST_LENGTH)
    print(f"Test data range: {datetime.fromtimestamp(test_data['periodStartUnix'].iloc[0])} to "
          f"{datetime.fromtimestamp(test_data['periodStartUnix'].iloc[-1])}")

    print("\nRunning model on test set...")
    test_tracking = run_model_episode(model, test_data, pool_config, TEST_LENGTH, seed=42,
                                       tick_width_options=V7_TICK_WIDTH_OPTIONS)
    print_summary(test_tracking, "TEST SET")
    test_tracking.to_csv('fold1_v7_test_data.csv', index=False)
    plot_model_behavior(test_tracking,
                        title="V7 Model - Test Set (30 Days)",
                        save_path="fold1_v7_test_evaluation.png")

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print("Saved files:")
    print("  - fold1_v7_train_evaluation.png")
    print("  - fold1_v7_train_data.csv")
    print("  - fold1_v7_test_evaluation.png")
    print("  - fold1_v7_test_data.csv")


if __name__ == '__main__':
    main()
