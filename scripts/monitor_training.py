#!/usr/bin/env python3
"""
Real-time Training Monitor for PPO Model

Periodically loads the latest checkpoint and visualizes model behavior.
Updates every N seconds with new episode data.
"""
import sys
import time
from pathlib import Path
from datetime import datetime
import argparse

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from uniswap_v3.ml.environment import UniswapV3LPEnv


class TrainingMonitor:
    def __init__(self, model_dir: Path, data_path: Path, update_interval: int = 30):
        self.model_dir = Path(model_dir)
        self.data_path = Path(data_path)
        self.update_interval = update_interval

        # Load test data
        df = pd.read_parquet(data_path)
        df = df.sort_values('periodStartUnix').reset_index(drop=True)
        n = len(df)
        self.test_df = df.iloc[int(n * 0.9):].copy()

        self.pool_config = {
            'protocol': 'ethereum',
            'feeTier': 3000,
            'token0': {'symbol': 'WETH', 'decimals': 18},
            'token1': {'symbol': 'USDT', 'decimals': 6}
        }

        # History for plotting
        self.history = {
            'timestamp': [],
            'checkpoint': [],
            'net_return': [],
            'fees': [],
            'il': [],
            'gas': [],
            'rebalances': [],
            'reward': []
        }

        # Episode step data for detailed view
        self.latest_episode = None

    def setup_plot(self):
        """Setup matplotlib figure with subplots."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('PPO Training Monitor', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def get_latest_checkpoint(self) -> Path:
        """Find the latest checkpoint file."""
        checkpoints_dir = self.model_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return None

        checkpoints = list(checkpoints_dir.glob("ppo_*.zip"))
        if not checkpoints:
            return None

        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-2]))
        return checkpoints[-1]

    def run_evaluation_episode(self, model: PPO) -> dict:
        """Run one evaluation episode and collect data."""
        # 전체 테스트 데이터를 1회 에피소드로 평가
        episode_length = len(self.test_df) - 10
        env = UniswapV3LPEnv(
            historical_data=self.test_df,
            pool_config=self.pool_config,
            initial_investment=10000,
            episode_length_hours=episode_length,
            debug=False
        )

        # Override gas cost to match training
        env.gas_costs['ethereum'] = 10
        env._is_position_failed = lambda price: False

        obs, info = env.reset(seed=42)  # Fixed seed for consistency

        done = False
        episode_reward = 0
        step_data = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = truncated
            episode_reward += reward

            # Record step data
            current_idx = env.episode_start_idx + env.current_step
            if current_idx < len(env.features_df) - 1:
                raw_price = env.features_df.iloc[current_idx]['close']
                display_price = 1.0 / raw_price if raw_price > 0 else 0
                display_min = 1.0 / env.position_max_price if env.position_max_price > 0 else 0
                display_max = 1.0 / env.position_min_price if env.position_min_price > 0 else 0

                step_data.append({
                    'step': env.current_step,
                    'price': display_price,
                    'min_price': display_min,
                    'max_price': display_max,
                    'cumulative_fees': env.cumulative_fees,
                    'cumulative_il': env.cumulative_il,
                    'cumulative_gas': env.cumulative_gas,
                    'rebalances': env.total_rebalances
                })

        return {
            'reward': episode_reward,
            'net_return': info['net_return'],
            'fees': info['cumulative_fees'],
            'il': info['cumulative_il'],
            'gas': info['cumulative_gas'],
            'rebalances': info['total_rebalances'],
            'steps': step_data
        }

    def update_plot(self):
        """Update all subplots with latest data and save to file."""
        self.setup_plot()  # Create fresh figure

        for ax in self.axes.flat:
            ax.clear()

        if len(self.history['timestamp']) == 0:
            return

        # Use evaluation numbers instead of timestamps for x-axis
        x_vals = list(range(1, len(self.history['timestamp']) + 1))

        # Plot 1: Net Return trend
        ax1 = self.axes[0, 0]
        ax1.plot(x_vals, self.history['net_return'], 'b-o', linewidth=2, markersize=6)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=1566, color='green', linestyle='--', alpha=0.5, label='Static ±10% benchmark')
        ax1.set_ylabel('Net Return ($)')
        ax1.set_xlabel('Evaluation #')
        ax1.set_title('Net Return per Episode')
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Fees vs IL vs Gas
        ax2 = self.axes[0, 1]
        ax2.plot(x_vals, self.history['fees'], 'g-o', label='Fees', linewidth=2, markersize=6)
        ax2.plot(x_vals, self.history['il'], 'r-o', label='IL', linewidth=2, markersize=6)
        ax2.plot(x_vals, self.history['gas'], 'orange', marker='o', label='Gas', linewidth=2, markersize=6)
        ax2.set_ylabel('USD')
        ax2.set_xlabel('Evaluation #')
        ax2.set_title('Fees / IL / Gas Breakdown')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Rebalances count
        ax3 = self.axes[1, 0]
        ax3.bar(x_vals, self.history['rebalances'], color='purple', alpha=0.7)
        ax3.set_ylabel('Rebalances')
        ax3.set_title('Rebalances per Episode')
        ax3.set_xlabel('Evaluation #')
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Latest episode detail (Price & Range)
        ax4 = self.axes[1, 1]
        if self.latest_episode and self.latest_episode['steps']:
            df = pd.DataFrame(self.latest_episode['steps'])
            ax4.plot(df['step'], df['price'], 'b-', label='Price', linewidth=1)
            ax4.fill_between(df['step'], df['min_price'], df['max_price'],
                           alpha=0.3, color='green', label='LP Range')

            # Mark rebalances
            rebalance_steps = df[df['rebalances'].diff() > 0]['step'].values
            for step in rebalance_steps:
                ax4.axvline(x=step, color='purple', alpha=0.5, linestyle='-', linewidth=2)

            ax4.set_ylabel('Price (USDT/WETH)')
            ax4.set_xlabel('Step (hours)')
            ax4.set_title(f'Latest Episode: Net ${self.latest_episode["net_return"]:.0f}')
            ax4.legend(loc='upper left', fontsize=8)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', fontsize=12)
            ax4.set_title('Latest Episode Detail')

        # Update title with latest info
        if self.history['checkpoint']:
            latest_ckpt = self.history['checkpoint'][-1]
            latest_return = self.history['net_return'][-1]
            self.fig.suptitle(
                f'PPO Training Monitor | Checkpoint: {latest_ckpt} | Latest Return: ${latest_return:.0f}',
                fontsize=12, fontweight='bold'
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save to file
        output_path = self.model_dir / "training_monitor.png"
        self.fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(self.fig)
        print(f"  → Updated: {output_path}", flush=True)

    def run(self):
        """Main monitoring loop."""
        print("="*60, flush=True)
        print(" PPO Training Monitor", flush=True)
        print(f" Model dir: {self.model_dir}", flush=True)
        print(f" Update interval: {self.update_interval}s", flush=True)
        print("="*60, flush=True)
        print("\nPress Ctrl+C to stop\n", flush=True)

        last_checkpoint = None

        try:
            while True:
                checkpoint_path = self.get_latest_checkpoint()

                if checkpoint_path and checkpoint_path != last_checkpoint:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading: {checkpoint_path.name}", flush=True)

                    try:
                        model = PPO.load(str(checkpoint_path))
                        result = self.run_evaluation_episode(model)

                        # Update history
                        self.history['timestamp'].append(datetime.now())
                        self.history['checkpoint'].append(checkpoint_path.stem)
                        self.history['net_return'].append(result['net_return'])
                        self.history['fees'].append(result['fees'])
                        self.history['il'].append(result['il'])
                        self.history['gas'].append(result['gas'])
                        self.history['rebalances'].append(result['rebalances'])
                        self.history['reward'].append(result['reward'])

                        self.latest_episode = result

                        print(f"  → Net: ${result['net_return']:.0f}, "
                              f"Fees: ${result['fees']:.0f}, "
                              f"IL: ${result['il']:.0f}, "
                              f"Rebal: {result['rebalances']}", flush=True)

                        self.update_plot()
                        last_checkpoint = checkpoint_path

                    except Exception as e:
                        print(f"  → Error: {e}", flush=True)

                elif checkpoint_path is None:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for checkpoints...", flush=True)

                time.sleep(self.update_interval)

        except KeyboardInterrupt:
            print("\n\nStopping monitor...", flush=True)

            # Save history
            history_df = pd.DataFrame(self.history)
            history_path = self.model_dir / "training_history.csv"
            history_df.to_csv(history_path, index=False)
            print(f"Saved: {history_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Monitor PPO training progress")
    parser.add_argument("--model-dir", type=Path, default=Path("models_v3"),
                       help="Model output directory")
    parser.add_argument("--data", type=Path,
                       default=Path("backend/data/training/ethereum_weth_usdt_03_data.parquet"),
                       help="Training data path")
    parser.add_argument("--interval", type=int, default=60,
                       help="Update interval in seconds")

    args = parser.parse_args()

    monitor = TrainingMonitor(
        model_dir=args.model_dir,
        data_path=args.data,
        update_interval=args.interval
    )
    monitor.run()


if __name__ == "__main__":
    main()
