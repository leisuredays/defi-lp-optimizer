"""
Walk-Forward Fold Training

Rolling window approach:
- 5 folds with expanding training window
- Each fold: Train on past data, test on next period
- Training steps: 200k per fold (total 1M / 5)
"""
import sys
sys.path.insert(0, '/home/zekiya/liquidity/uniswap-v3-simulator')

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configuration
NUM_FOLDS = 5
TIMESTEPS_PER_FOLD = 200_000  # 1M / 5
CHECKPOINT_INTERVAL = 50_000
ENT_COEF = 0.01
WARMUP_PERIOD = 720  # 30 days for rolling features

# Learning rate schedule (higher for faster learning)
INITIAL_LR = 0.001  # 3x higher
FINAL_LR = 0.0001   # 10x decay

def linear_schedule(progress_remaining: float) -> float:
    return FINAL_LR + (INITIAL_LR - FINAL_LR) * progress_remaining

# Paths
project_root = Path('/home/zekiya/liquidity/uniswap-v3-simulator')
db_path = project_root / 'uniswap_v3/data/pool_data.db'

print("=" * 60)
print("WALK-FORWARD FOLD TRAINING")
print(f"  Folds: {NUM_FOLDS}")
print(f"  Steps per fold: {TIMESTEPS_PER_FOLD:,}")
print("=" * 60)

# Load data
print("\nLoading data...")
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

pool_config = {
    'protocol': int(df['protocol_id'].iloc[0]),
    'feeTier': int(df['fee_tier'].iloc[0]),
    'token0': {'decimals': int(df['token0_decimals'].iloc[0])},
    'token1': {'decimals': int(df['token1_decimals'].iloc[0])}
}

# Calculate fold boundaries
total_hours = len(df)
fold_size = total_hours // (NUM_FOLDS + 1)  # +1 for final test period

print(f"\n=== Dataset Info ===")
print(f"Total: {total_hours:,} hours ({total_hours/24:.0f} days)")
print(f"Fold size: {fold_size:,} hours ({fold_size/24:.0f} days)")

from uniswap_v3.ml import UniswapV3LPEnv

def create_fold_data(fold_idx):
    """
    Walk-forward fold:
    - Train: periods 0 to fold_idx (expanding)
    - Test: period fold_idx + 1
    """
    train_end = (fold_idx + 1) * fold_size
    test_start = train_end
    test_end = min(test_start + fold_size, total_hours)

    train_data = df.iloc[0:train_end].reset_index(drop=True)
    test_data = df.iloc[test_start:test_end].reset_index(drop=True)

    return train_data, test_data

def make_env(data):
    def _init():
        max_episode_length = len(data) - WARMUP_PERIOD - 10
        env = UniswapV3LPEnv(
            historical_data=data,
            pool_config=pool_config,
            initial_investment=10000,
            episode_length_hours=max_episode_length
        )
        return Monitor(env)
    return _init

def evaluate_model(model, data, name, max_hours=None):
    """Evaluate model and collect detailed metrics."""
    available_hours = len(data) - WARMUP_PERIOD - 10
    eval_hours = min(available_hours, max_hours) if max_hours else available_hours

    env = UniswapV3LPEnv(
        historical_data=data,
        pool_config=pool_config,
        initial_investment=10000,
        episode_length_hours=eval_hours
    )

    obs, info = env.reset()

    timestamps = []
    prices = []
    lower_bounds = []
    upper_bounds = []
    action_lower_pcts = []
    action_upper_pcts = []
    action_rebalance_signals = []
    cumulative_fees = []
    cumulative_il = []
    rebalance_timestamps = []
    rebalance_prices = []
    rewards = []
    cumulative_rewards = []

    done = False
    prev_rebalances = 0
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        raw_lower = (action[0] + 1) / 2 * 0.98 + 0.01
        raw_upper = (action[1] + 1) / 2 * 4.99 + 0.01
        raw_rebalance = (action[2] + 1) / 2

        action_lower_pcts.append(raw_lower * 100)
        action_upper_pcts.append(raw_upper * 100)
        action_rebalance_signals.append(raw_rebalance)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        current_idx = env.episode_start_idx + env.current_step - 1
        ts = pd.to_datetime(env.historical_data.iloc[current_idx]['periodStartUnix'], unit='s')

        timestamps.append(ts)
        prices.append(info['current_price'])
        lower_bounds.append(env.position_min_price)
        upper_bounds.append(env.position_max_price)
        cumulative_fees.append(info['cumulative_fees'])
        cumulative_il.append(env.cumulative_il)

        if env.total_rebalances > prev_rebalances:
            rebalance_timestamps.append(ts)
            rebalance_prices.append(info['current_price'])
            prev_rebalances = env.total_rebalances

        rewards.append(reward)
        total_reward += reward
        cumulative_rewards.append(total_reward)

    return {
        'name': name,
        'timestamps': timestamps,
        'prices': prices,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'action_lower_pcts': action_lower_pcts,
        'action_upper_pcts': action_upper_pcts,
        'action_rebalance_signals': action_rebalance_signals,
        'cumulative_fees': cumulative_fees,
        'cumulative_il': cumulative_il,
        'rebalance_timestamps': rebalance_timestamps,
        'rebalance_prices': rebalance_prices,
        'rewards': rewards,
        'cumulative_rewards': cumulative_rewards,
        'total_rebalances': env.total_rebalances,
        'final_lp': info['lp_value'],
        'final_hodl': info['hodl_value'],
        'excess_return': info['excess_return'] * 100,
    }

def create_visualization(train_results, test_results, fold_idx, step_k):
    """Create visualization for fold."""
    fig, axes = plt.subplots(5, 2, figsize=(20, 20))

    for col, results in enumerate([train_results, test_results]):
        name = results['name']

        # Row 1: Price with LP Range
        ax1 = axes[0, col]
        ax1.set_title(f'{name}: Price & LP Range', fontsize=11, fontweight='bold')
        ax1.plot(results['timestamps'], results['prices'], 'b-', label='ETH Price', linewidth=1)
        ax1.fill_between(results['timestamps'], results['lower_bounds'], results['upper_bounds'],
                         alpha=0.25, color='green', label='LP Range')
        if results['rebalance_timestamps']:
            ax1.scatter(results['rebalance_timestamps'], results['rebalance_prices'],
                       color='red', s=60, marker='o', label=f'Rebalance ({len(results["rebalance_timestamps"])})')
        ax1.set_ylabel('Price (USD)')
        ax1.legend(loc='upper right')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.grid(True, alpha=0.3)

        # Row 2: Actions
        ax2 = axes[1, col]
        ax2.set_title(f'{name}: Actions (Range %)', fontsize=11, fontweight='bold')
        ax2.plot(results['timestamps'], results['action_lower_pcts'], 'r-',
                 label=f'Lower (avg:{np.mean(results["action_lower_pcts"]):.1f}%)', linewidth=0.6)
        ax2.plot(results['timestamps'], results['action_upper_pcts'], 'g-',
                 label=f'Upper (avg:{np.mean(results["action_upper_pcts"]):.1f}%)', linewidth=0.6)
        ax2.set_ylabel('Range %')
        ax2.legend(loc='upper right')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.grid(True, alpha=0.3)

        # Row 3: Rebalance Signal
        ax3 = axes[2, col]
        ax3.set_title(f'{name}: Rebalance Signal', fontsize=11, fontweight='bold')
        ax3.fill_between(results['timestamps'], 0, results['action_rebalance_signals'],
                         alpha=0.5, color='orange')
        ax3.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax3.set_ylabel('Signal')
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(loc='upper right')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.grid(True, alpha=0.3)

        # Row 4: Fees vs IL
        ax4 = axes[3, col]
        ax4.set_title(f'{name}: Fees vs IL', fontsize=11, fontweight='bold')
        ax4.plot(results['timestamps'], results['cumulative_fees'], 'g-',
                 label=f'Fees: ${results["cumulative_fees"][-1]:.0f}', linewidth=1.5)
        ax4.plot(results['timestamps'], results['cumulative_il'], 'r-',
                 label=f'IL: ${results["cumulative_il"][-1]:.0f}', linewidth=1.5)
        net = np.array(results['cumulative_fees']) - np.array(results['cumulative_il'])
        ax4.plot(results['timestamps'], net, 'b--',
                 label=f'Net: ${net[-1]:.0f}', linewidth=1.5)
        ax4.axhline(0, color='black', linewidth=0.5)
        ax4.set_ylabel('USD')
        ax4.legend(loc='best')
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax4.grid(True, alpha=0.3)

        # Row 5: Cumulative Reward
        ax5 = axes[4, col]
        ax5.set_title(f'{name}: Cumulative Reward', fontsize=11, fontweight='bold')
        ax5.plot(results['timestamps'], results['cumulative_rewards'], 'purple',
                 label=f'Total: ${results["cumulative_rewards"][-1]:.0f}', linewidth=1.5)
        ax5.axhline(0, color='black', linewidth=0.5)
        ax5.set_ylabel('Reward (USD)')
        ax5.legend(loc='best')
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax5.grid(True, alpha=0.3)

    plt.tight_layout()

    summary = (
        f"TRAIN: Excess={train_results['excess_return']:+.1f}%, Rebal={train_results['total_rebalances']}, "
        f"Fees=${train_results['cumulative_fees'][-1]:.0f}, IL=${train_results['cumulative_il'][-1]:.0f}\n"
        f"TEST: Excess={test_results['excess_return']:+.1f}%, Rebal={test_results['total_rebalances']}, "
        f"Fees=${test_results['cumulative_fees'][-1]:.0f}, IL=${test_results['cumulative_il'][-1]:.0f}"
    )
    fig.suptitle(f'Walk-Forward Fold {fold_idx+1}/{NUM_FOLDS} - {step_k}k Steps\n{summary}',
                 fontsize=14, fontweight='bold', y=1.02)

    filename = f'walkforward_fold{fold_idx+1}_{step_k}k.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    return filename

class FoldCallback(BaseCallback):
    def __init__(self, fold_idx, train_data, test_data, verbose=0):
        super().__init__(verbose)
        self.fold_idx = fold_idx
        self.train_data = train_data
        self.test_data = test_data
        self.checkpoints = []

    def _on_step(self) -> bool:
        if self.num_timesteps % CHECKPOINT_INTERVAL == 0:
            step_k = self.num_timesteps // 1000

            print(f"\n{'='*60}")
            print(f"FOLD {self.fold_idx+1} - CHECKPOINT at {step_k}k steps")
            print(f"{'='*60}")

            # Episode reward
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                print(f"  Mean Episode Reward (last 10): {mean_reward:.2f}")

            # Evaluate
            print("  Evaluating Train...")
            train_results = evaluate_model(self.model, self.train_data, "Train")
            print(f"    Excess: {train_results['excess_return']:+.2f}%, Rebalances: {train_results['total_rebalances']}")

            print("  Evaluating Test...")
            test_results = evaluate_model(self.model, self.test_data, "Test")
            print(f"    Excess: {test_results['excess_return']:+.2f}%, Rebalances: {test_results['total_rebalances']}")

            print("  Creating visualization...")
            filename = create_visualization(train_results, test_results, self.fold_idx, step_k)
            print(f"  Saved: {filename}")

            self.checkpoints.append({
                'step': self.num_timesteps,
                'train_excess': train_results['excess_return'],
                'test_excess': test_results['excess_return'],
            })
            print(f"{'='*60}\n")

        return True

# Store results for all folds
all_fold_results = []

# Train each fold
for fold_idx in range(NUM_FOLDS):
    print(f"\n{'#'*60}")
    print(f"# FOLD {fold_idx + 1}/{NUM_FOLDS}")
    print(f"{'#'*60}")

    # Get fold data
    train_data, test_data = create_fold_data(fold_idx)

    train_start = pd.to_datetime(train_data.iloc[0]['periodStartUnix'], unit='s')
    train_end = pd.to_datetime(train_data.iloc[-1]['periodStartUnix'], unit='s')
    test_start = pd.to_datetime(test_data.iloc[0]['periodStartUnix'], unit='s')
    test_end = pd.to_datetime(test_data.iloc[-1]['periodStartUnix'], unit='s')

    print(f"Train: {len(train_data):,} hours ({train_start.strftime('%Y-%m-%d')} ~ {train_end.strftime('%Y-%m-%d')})")
    print(f"Test:  {len(test_data):,} hours ({test_start.strftime('%Y-%m-%d')} ~ {test_end.strftime('%Y-%m-%d')})")

    # Create environment
    env = DummyVecEnv([make_env(train_data)])

    # Create model (fresh for each fold)
    print("\nCreating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=nn.Tanh),
        learning_rate=linear_schedule,
        clip_range=0.2,
        ent_coef=ENT_COEF,
        gamma=0.999,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=0,
        device="cuda"
    )
    print(f"Device: {model.device}")

    # Train
    callback = FoldCallback(fold_idx, train_data, test_data)
    print(f"\nTraining for {TIMESTEPS_PER_FOLD:,} steps...")
    model.learn(total_timesteps=TIMESTEPS_PER_FOLD, callback=callback)

    # Save model
    model_path = f'walkforward_fold{fold_idx+1}.zip'
    model.save(model_path)
    print(f"\nModel saved: {model_path}")

    # Final evaluation
    print("\nFinal Evaluation...")
    train_results = evaluate_model(model, train_data, "Train")
    test_results = evaluate_model(model, test_data, "Test")

    all_fold_results.append({
        'fold': fold_idx + 1,
        'train_excess': train_results['excess_return'],
        'test_excess': test_results['excess_return'],
        'train_rebalances': train_results['total_rebalances'],
        'test_rebalances': test_results['total_rebalances'],
    })

    env.close()

# Summary
print("\n" + "=" * 60)
print("WALK-FORWARD TRAINING SUMMARY")
print("=" * 60)
print(f"{'Fold':<6} {'Train Excess':>14} {'Test Excess':>14} {'Train Rebal':>12} {'Test Rebal':>12}")
print("-" * 60)
for r in all_fold_results:
    print(f"{r['fold']:<6} {r['train_excess']:>+13.1f}% {r['test_excess']:>+13.1f}% "
          f"{r['train_rebalances']:>12} {r['test_rebalances']:>12}")

avg_train = np.mean([r['train_excess'] for r in all_fold_results])
avg_test = np.mean([r['test_excess'] for r in all_fold_results])
print("-" * 60)
print(f"{'AVG':<6} {avg_train:>+13.1f}% {avg_test:>+13.1f}%")
print("=" * 60)
