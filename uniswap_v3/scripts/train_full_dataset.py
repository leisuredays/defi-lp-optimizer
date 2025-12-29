"""
Train on Full Dataset - Single Train/Test Split
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
TRAIN_RATIO = 0.8  # 80% train, 20% test
TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_INTERVAL = 50_000  # Every 50k for detailed monitoring
ENT_COEF = 0.01  # Optimal entropy (0.05 was too high, caused non-convergence)

# Learning rate schedule (linear decay)
INITIAL_LR = 0.0003
FINAL_LR = 0.00003  # 10x decay

def linear_schedule(progress_remaining: float) -> float:
    """
    Linear learning rate schedule.
    progress_remaining: 1.0 at start, 0.0 at end
    """
    return FINAL_LR + (INITIAL_LR - FINAL_LR) * progress_remaining

# Paths
project_root = Path('/home/zekiya/liquidity/uniswap-v3-simulator')
db_path = project_root / 'uniswap_v3/data/pool_data.db'

# Load data
print("=" * 60)
print("FULL DATASET TRAINING - Single Train/Test Split")
print("=" * 60)

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

# Train/Test split
total_hours = len(df)
train_hours = int(total_hours * TRAIN_RATIO)
test_hours = total_hours - train_hours

train_data = df.iloc[0:train_hours].reset_index(drop=True)
test_data = df.iloc[train_hours:].reset_index(drop=True)

train_start = pd.to_datetime(train_data.iloc[0]['periodStartUnix'], unit='s')
train_end = pd.to_datetime(train_data.iloc[-1]['periodStartUnix'], unit='s')
test_start = pd.to_datetime(test_data.iloc[0]['periodStartUnix'], unit='s')
test_end = pd.to_datetime(test_data.iloc[-1]['periodStartUnix'], unit='s')

print(f"\n=== Dataset Split ===")
print(f"Total: {total_hours:,} hours ({total_hours/24:.0f} days)")
print(f"Train: {train_hours:,} hours ({train_hours/24:.0f} days) - {train_start.strftime('%Y-%m-%d')} ~ {train_end.strftime('%Y-%m-%d')}")
print(f"Test:  {test_hours:,} hours ({test_hours/24:.0f} days) - {test_start.strftime('%Y-%m-%d')} ~ {test_end.strftime('%Y-%m-%d')}")

pool_config = {
    'protocol': int(df['protocol_id'].iloc[0]),
    'feeTier': int(df['fee_tier'].iloc[0]),
    'token0': {'decimals': int(df['token0_decimals'].iloc[0])},
    'token1': {'decimals': int(df['token1_decimals'].iloc[0])}
}

from uniswap_v3.ml import UniswapV3LPEnv

WARMUP_PERIOD = 720  # 30 days for rolling features

def make_env():
    # Episode length must account for warmup period
    # Available: len(train_data) - warmup - buffer
    max_episode_length = len(train_data) - WARMUP_PERIOD - 10
    env = UniswapV3LPEnv(
        historical_data=train_data,
        pool_config=pool_config,
        initial_investment=10000,
        episode_length_hours=max_episode_length
    )
    return Monitor(env)

def evaluate_model(model, data, name, max_hours=None):
    """Evaluate model and collect detailed metrics."""
    # Account for warmup period
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
    lp_values = []
    cumulative_fees = []
    cumulative_il = []
    rebalance_timestamps = []
    rebalance_prices = []
    rewards = []  # Track rewards
    cumulative_rewards = []  # Cumulative rewards

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
        lp_values.append(info['lp_value'])
        cumulative_fees.append(info['cumulative_fees'])
        cumulative_il.append(env.cumulative_il)

        if env.total_rebalances > prev_rebalances:
            rebalance_timestamps.append(ts)
            rebalance_prices.append(info['current_price'])
            prev_rebalances = env.total_rebalances

        # Track rewards
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
        'lp_values': lp_values,
        'cumulative_fees': cumulative_fees,
        'cumulative_il': cumulative_il,
        'rebalance_timestamps': rebalance_timestamps,
        'rebalance_prices': rebalance_prices,
        'rewards': rewards,  # Individual step rewards
        'cumulative_rewards': cumulative_rewards,  # Cumulative rewards
        'total_rebalances': env.total_rebalances,
        'final_lp': info['lp_value'],
        'final_hodl': info['hodl_value'],
        'excess_return': info['excess_return'] * 100,
    }

def create_visualization(train_results, test_results, step_k):
    """Create visualization with rebalance points."""
    fig, axes = plt.subplots(5, 2, figsize=(20, 20))  # 5 rows for reward chart

    for col, results in enumerate([train_results, test_results]):
        name = results['name']

        # Row 1: Price with LP Range and Rebalance Points
        ax1 = axes[0, col]
        ax1.set_title(f'{name}: Price & LP Range', fontsize=11, fontweight='bold')
        ax1.plot(results['timestamps'], results['prices'], 'b-', label='ETH Price', linewidth=1, zorder=2)
        ax1.fill_between(results['timestamps'], results['lower_bounds'], results['upper_bounds'],
                         alpha=0.25, color='green', label='LP Range', zorder=1)

        # Rebalance points as RED DOTS
        if results['rebalance_timestamps']:
            ax1.scatter(results['rebalance_timestamps'], results['rebalance_prices'],
                       color='red', s=60, marker='o', label=f'Rebalance ({len(results["rebalance_timestamps"])})',
                       zorder=5, edgecolors='darkred', linewidths=1)

        ax1.set_ylabel('Price (USD)')
        ax1.legend(loc='upper right')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.grid(True, alpha=0.3)

        # Row 2: Model Actions
        ax2 = axes[1, col]
        ax2.set_title(f'{name}: Actions (Range %)', fontsize=11, fontweight='bold')
        ax2.plot(results['timestamps'], results['action_lower_pcts'], 'r-',
                 label=f'Lower (avg:{np.mean(results["action_lower_pcts"]):.1f}%)', linewidth=0.6, alpha=0.8)
        ax2.plot(results['timestamps'], results['action_upper_pcts'], 'g-',
                 label=f'Upper (avg:{np.mean(results["action_upper_pcts"]):.1f}%)', linewidth=0.6, alpha=0.8)
        ax2.set_ylabel('Range %')
        ax2.legend(loc='upper right')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.grid(True, alpha=0.3)

        for rt in results['rebalance_timestamps']:
            ax2.axvline(rt, color='red', alpha=0.3, linewidth=0.8)

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

        for rt in results['rebalance_timestamps']:
            ax3.axvline(rt, color='red', alpha=0.5, linewidth=1)

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

        # Row 5: Cumulative Rewards
        ax5 = axes[4, col]
        ax5.set_title(f'{name}: Cumulative Reward', fontsize=11, fontweight='bold')
        ax5.plot(results['timestamps'], results['cumulative_rewards'], 'purple',
                 label=f'Total: ${results["cumulative_rewards"][-1]:.0f}', linewidth=1.5)
        ax5.axhline(0, color='black', linewidth=0.5)
        ax5.set_ylabel('Reward (USD)')
        ax5.legend(loc='best')
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax5.grid(True, alpha=0.3)

        for rt in results['rebalance_timestamps']:
            ax5.axvline(rt, color='red', alpha=0.5, linewidth=1)

    plt.tight_layout()

    summary = (
        f"TRAIN: Excess={train_results['excess_return']:+.1f}%, Rebal={train_results['total_rebalances']}, "
        f"Fees=${train_results['cumulative_fees'][-1]:.0f}, IL=${train_results['cumulative_il'][-1]:.0f}\n"
        f"TEST: Excess={test_results['excess_return']:+.1f}%, Rebal={test_results['total_rebalances']}, "
        f"Fees=${test_results['cumulative_fees'][-1]:.0f}, IL=${test_results['cumulative_il'][-1]:.0f}"
    )
    fig.suptitle(f'Full Dataset Training - {step_k}k Steps (ent_coef={ENT_COEF})\n{summary}',
                 fontsize=13, fontweight='bold', y=1.02)

    filename = f'full_dataset_{step_k}k.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")
    return filename

# Callback
class CheckpointCallback(BaseCallback):
    def __init__(self, checkpoint_interval, train_data, test_data, pool_config, verbose=0):
        super().__init__(verbose)
        self.checkpoint_interval = checkpoint_interval
        self.train_data = train_data
        self.test_data = test_data
        self.pool_config = pool_config
        self.checkpoints = []
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])

        if self.num_timesteps > 0 and self.num_timesteps % self.checkpoint_interval == 0:
            step_k = self.num_timesteps // 1000
            print(f"\n{'='*60}")
            print(f"CHECKPOINT at {step_k}k steps")
            print(f"{'='*60}")

            if len(self.episode_rewards) > 0:
                print(f"  Mean Episode Reward (last 10): {np.mean(self.episode_rewards[-10:]):.2f}")

            # Evaluate on FULL periods
            print("  Evaluating Train (FULL)...")
            train_results = evaluate_model(self.model, self.train_data, "Train")
            print(f"    Excess: {train_results['excess_return']:+.2f}%, Rebalances: {train_results['total_rebalances']}")

            print("  Evaluating Test (FULL)...")
            test_results = evaluate_model(self.model, self.test_data, "Test")
            print(f"    Excess: {test_results['excess_return']:+.2f}%, Rebalances: {test_results['total_rebalances']}")

            print("  Creating visualization...")
            create_visualization(train_results, test_results, step_k)

            self.checkpoints.append({
                'step': self.num_timesteps,
                'train_excess': train_results['excess_return'],
                'test_excess': test_results['excess_return'],
                'train_rebalances': train_results['total_rebalances'],
                'test_rebalances': test_results['total_rebalances'],
            })
            print(f"{'='*60}\n")

        return True

# Create environment
env = DummyVecEnv([make_env])

# Create model
print("\nCreating PPO model...")
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=nn.Tanh),
    learning_rate=linear_schedule,  # Linear decay: 3e-4 → 3e-5
    clip_range=0.2,
    ent_coef=ENT_COEF,
    gamma=0.999,  # 장기 투자용 (기존 0.9565 → 0.999)
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=0,
    device="cuda"
)
print(f"Device: {model.device}")
print(f"Entropy Coef: {ENT_COEF}")
print(f"Learning Rate: {INITIAL_LR} → {FINAL_LR} (linear decay)")

# Train
print(f"\nTraining for {TOTAL_TIMESTEPS:,} steps...")
callback = CheckpointCallback(CHECKPOINT_INTERVAL, train_data, test_data, pool_config)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)

# Save model
from pathlib import Path
project_root = Path('/home/zekiya/liquidity/uniswap-v3-simulator')
save_path = project_root / 'uniswap_v3/models/full_dataset_1m.zip'
model.save(save_path)
print(f"\nModel saved to: {save_path}")

# Final full evaluation
print("\n" + "=" * 60)
print("FINAL EVALUATION (Full periods)")
print("=" * 60)

print("Evaluating full Train period...")
train_results = evaluate_model(model, train_data, f"Train ({train_hours/24:.0f}d)")
print(f"  Excess: {train_results['excess_return']:+.2f}%")
print(f"  Rebalances: {train_results['total_rebalances']}")
print(f"  Fees: ${train_results['cumulative_fees'][-1]:.0f}")
print(f"  IL: ${train_results['cumulative_il'][-1]:.0f}")

print("\nEvaluating full Test period...")
test_results = evaluate_model(model, test_data, f"Test ({test_hours/24:.0f}d)")
print(f"  Excess: {test_results['excess_return']:+.2f}%")
print(f"  Rebalances: {test_results['total_rebalances']}")
print(f"  Fees: ${test_results['cumulative_fees'][-1]:.0f}")
print(f"  IL: ${test_results['cumulative_il'][-1]:.0f}")

# Final visualization
print("\nCreating final visualization...")
create_visualization(train_results, test_results, 'final')

# Print rebalance details
print("\n" + "=" * 60)
print("REBALANCE POINTS")
print("=" * 60)
print(f"\nTrain ({train_results['total_rebalances']} rebalances):")
for i, (ts, price) in enumerate(zip(train_results['rebalance_timestamps'][:10], train_results['rebalance_prices'][:10])):
    print(f"  {i+1}. {ts.strftime('%Y-%m-%d %H:%M')} @ ${price:.2f}")
if train_results['total_rebalances'] > 10:
    print(f"  ... and {train_results['total_rebalances'] - 10} more")

print(f"\nTest ({test_results['total_rebalances']} rebalances):")
for i, (ts, price) in enumerate(zip(test_results['rebalance_timestamps'][:10], test_results['rebalance_prices'][:10])):
    print(f"  {i+1}. {ts.strftime('%Y-%m-%d %H:%M')} @ ${price:.2f}")
if test_results['total_rebalances'] > 10:
    print(f"  ... and {test_results['total_rebalances'] - 10} more")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for cp in callback.checkpoints:
    print(f"  {cp['step']//1000}k: Train {cp['train_excess']:+.1f}% ({cp['train_rebalances']} reb) | Test {cp['test_excess']:+.1f}% ({cp['test_rebalances']} reb)")
print(f"  Final: Train {train_results['excess_return']:+.1f}% ({train_results['total_rebalances']} reb) | Test {test_results['excess_return']:+.1f}% ({test_results['total_rebalances']} reb)")
