"""
Visualize trained model behavior on dataset.

Shows:
- Price movement with LP range overlay
- Rebalancing events
- Fee accumulation
- In-range periods
"""
import sys
sys.path.insert(0, '/home/zekiya/liquidity/uniswap-v3-simulator')

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

    # Prepare data
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

    # Slice data
    end_idx = min(start_idx + length + 20, len(df))  # Buffer for env
    data = df.iloc[start_idx:end_idx].reset_index(drop=True)

    pool_config = {
        'protocol': 0,
        'feeTier': fee_tier,
        'token0': {'decimals': token0_dec},
        'token1': {'decimals': token1_dec}
    }

    return data, pool_config


def run_model_episode(model: PPO, data: pd.DataFrame, pool_config: dict,
                      episode_length: int = 720, seed: int = 42):
    """Run model on data and collect tracking info."""
    env = UniswapV3LPEnv(
        historical_data=data,
        pool_config=pool_config,
        initial_investment=10000,
        episode_length_hours=episode_length
    )

    obs, _ = env.reset(seed=seed)

    # Set initial range from model
    action, _ = model.predict(obs, deterministic=True)
    env.set_initial_range_from_action(action)

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
        'cumulative_gas': [],
        'net_return': [],
        'reward': [],
        'action_width': [],
        'action_offset': [],
    }

    done = False
    step = 0

    while not done and step < episode_length:
        # Get current data
        current_idx = env.episode_start_idx + env.current_step
        if current_idx >= len(env.features_df):
            break
        current_data = env.features_df.iloc[current_idx]
        timestamp = datetime.fromtimestamp(current_data['periodStartUnix'])
        price = current_data['close']

        # Get model action
        action, _ = model.predict(obs, deterministic=True)

        # Step environment
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
        tracking['cumulative_gas'].append(info.get('cumulative_gas', 0))
        tracking['net_return'].append(info.get('net_return', 0))
        tracking['reward'].append(reward)
        tracking['action_width'].append(action[0])
        tracking['action_offset'].append(action[1])

        step += 1

    return pd.DataFrame(tracking)


def plot_model_behavior(df: pd.DataFrame, title: str = "Model Behavior",
                        save_path: str = None):
    """Create visualization of model behavior."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # === Panel 1: Price with LP Range ===
    ax1 = axes[0]

    # Plot price
    ax1.plot(df['timestamp'], df['price'], 'b-', linewidth=1.5, label='Price', zorder=3)

    # Fill LP range area
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
    ax2.plot(df['timestamp'], -df['cumulative_il'], 'r-', linewidth=1.5, label='-IL')
    ax2.plot(df['timestamp'], -df['cumulative_gas'], 'orange', linewidth=1.5, label='-Gas')

    # Calculate ACTUAL return (fees - IL - gas) instead of RL reward (fees - LVR - gas)
    actual_return = df['cumulative_fees'] - df['cumulative_il'] - df['cumulative_gas']
    ax2.plot(df['timestamp'], actual_return, 'b-', linewidth=2, label='Net Return (Actual)')

    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.fill_between(df['timestamp'], 0, actual_return,
                     where=actual_return > 0, alpha=0.3, color='green')
    ax2.fill_between(df['timestamp'], 0, actual_return,
                     where=actual_return < 0, alpha=0.3, color='red')

    ax2.set_ylabel('USD', fontsize=11)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Cumulative Returns', fontsize=12)

    # === Panel 3: Range Width (relative to price) ===
    ax3 = axes[2]

    range_width_pct = (df['max_range'] - df['min_range']) / df['price'] * 100
    ax3.fill_between(df['timestamp'], 0, range_width_pct, alpha=0.5, color='purple')
    ax3.plot(df['timestamp'], range_width_pct, 'purple', linewidth=1)

    # Mark rebalancing events
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

    # Rotate x-axis labels
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

    return fig


def print_summary(df: pd.DataFrame):
    """Print episode summary statistics."""
    print("\n" + "="*60)
    print("EPISODE SUMMARY")
    print("="*60)

    total_hours = len(df)
    in_range_hours = df['in_range'].sum()
    total_rebalances = df['rebalanced'].sum()

    print(f"Duration:        {total_hours} hours ({total_hours/24:.1f} days)")
    print(f"In-range hours:  {in_range_hours} ({in_range_hours/total_hours*100:.1f}%)")
    print(f"Rebalances:      {total_rebalances}")
    print()
    print(f"Total Fees:      ${df['cumulative_fees'].iloc[-1]:,.2f}")
    print(f"Total IL:        ${df['cumulative_il'].iloc[-1]:,.2f}")
    print(f"Total Gas:       ${df['cumulative_gas'].iloc[-1]:,.2f}")
    actual_net = df['cumulative_fees'].iloc[-1] - df['cumulative_il'].iloc[-1] - df['cumulative_gas'].iloc[-1]
    print(f"Net Return:      ${actual_net:,.2f} (Actual: fees - IL - gas)")
    print()
    print(f"Price Start:     ${df['price'].iloc[0]:,.2f}")
    print(f"Price End:       ${df['price'].iloc[-1]:,.2f}")
    print(f"Price Change:    {(df['price'].iloc[-1]/df['price'].iloc[0]-1)*100:+.2f}%")
    print()

    # Range statistics
    avg_width_pct = ((df['max_range'] - df['min_range']) / df['price']).mean() * 100
    print(f"Avg Range Width: {avg_width_pct:.2f}%")

    if total_rebalances > 0:
        rebal_times = df[df['rebalanced']]['timestamp'].tolist()
        print(f"\nRebalancing times:")
        for t in rebal_times:
            print(f"  - {t}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize model behavior')
    parser.add_argument('--model', type=str,
                       default='/home/zekiya/liquidity/uniswap-v3-simulator/uniswap_v3/models/rolling_wfe/fold_01.zip',
                       help='Path to trained model')
    parser.add_argument('--start', type=int, default=4320,
                       help='Start index in dataset (default: 4320 = Fold 1 test)')
    parser.add_argument('--length', type=int, default=720,
                       help='Episode length in hours (default: 720 = 30 days)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save figure')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    # Load data
    db_path = "/home/zekiya/liquidity/uniswap-v3-simulator/uniswap_v3/data/pool_data.db"
    print(f"Loading data from index {args.start}...")
    data, pool_config = load_data(db_path, args.start, args.length)

    print(f"Data range: {datetime.fromtimestamp(data['periodStartUnix'].iloc[0])} to "
          f"{datetime.fromtimestamp(data['periodStartUnix'].iloc[-1])}")

    # Run model
    print("\nRunning model episode...")
    tracking_df = run_model_episode(model, data, pool_config, args.length, args.seed)

    # Print summary
    print_summary(tracking_df)

    # Save tracking data
    tracking_path = args.save.replace('.png', '_data.csv') if args.save else 'model_behavior_data.csv'
    tracking_df.to_csv(tracking_path, index=False)
    print(f"\nTracking data saved: {tracking_path}")

    # Plot
    title = f"Model Behavior - {args.length} hours ({args.length//24} days)"
    save_path = args.save or 'model_behavior.png'
    plot_model_behavior(tracking_df, title=title, save_path=save_path)


if __name__ == '__main__':
    main()
