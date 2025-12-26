#!/usr/bin/env python
"""
Continue training an existing fold model with additional timesteps.

Usage:
  python continue_fold_training.py --fold 1 --target 1000000
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from datetime import datetime


class ProgressCallback(BaseCallback):
    """Callback to log training progress."""

    def __init__(self, log_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.rewards = []

    def _on_step(self) -> bool:
        # Collect rewards from info
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.rewards.append(info['episode']['r'])

        # Log progress
        if self.num_timesteps % self.log_freq == 0:
            if self.rewards:
                mean_r = np.mean(self.rewards[-100:]) if len(self.rewards) >= 100 else np.mean(self.rewards)
                std_r = np.std(self.rewards[-100:]) if len(self.rewards) >= 100 else np.std(self.rewards)
                min_r = np.min(self.rewards[-100:]) if len(self.rewards) >= 100 else np.min(self.rewards)
                max_r = np.max(self.rewards[-100:]) if len(self.rewards) >= 100 else np.max(self.rewards)
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Steps: {self.num_timesteps:,} | Reward: {mean_r:.2f} ± {std_r:.2f} (min: {min_r:.2f}, max: {max_r:.2f})")
            else:
                print(f"Steps: {self.num_timesteps:,}")
            import sys
            sys.stdout.flush()

        return True

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from uniswap_v3.ml import UniswapV3LPEnv


def load_data_from_db(db_path: Path) -> pd.DataFrame:
    """Load all data from SQLite database."""
    conn = sqlite3.connect(db_path)

    pool_info = pd.read_sql('SELECT * FROM pools LIMIT 1', conn)
    df = pd.read_sql('''
        SELECT * FROM pool_hour_data
        ORDER BY period_start_unix
    ''', conn)
    conn.close()

    # Rename columns
    df = df.rename(columns={
        'period_start_unix': 'periodStartUnix',
        'fee_growth_global_0_x128': 'feeGrowthGlobal0X128',
        'fee_growth_global_1_x128': 'feeGrowthGlobal1X128',
    })

    # Add metadata
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


def continue_training(fold: int, target_steps: int):
    """Continue training a fold model to target total steps."""

    # Paths
    base_dir = project_root / "uniswap_v3/models/rolling_wfe"
    model_path = base_dir / "models" / f"fold_{fold:02d}.zip"
    db_path = project_root / "uniswap_v3/data/pool_data.db"

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Configuration
    WINDOW_DAYS = 180
    TEST_DAYS = 30
    WINDOW_HOURS = WINDOW_DAYS * 24
    TEST_HOURS = TEST_DAYS * 24

    # Calculate fold data range
    train_start = (fold - 1) * TEST_HOURS
    train_end = train_start + WINDOW_HOURS

    # Load data
    print("Loading data...")
    data = load_data_from_db(db_path)
    train_data = data.iloc[train_start:train_end].reset_index(drop=True)
    print(f"Training data: {len(train_data)} hours ({len(train_data)/24:.0f} days)")

    # Create environment
    env = DummyVecEnv([lambda: create_env(train_data)])

    # Load existing model
    print(f"\nLoading model: {model_path}")
    model = PPO.load(str(model_path), env=env, device="cuda")

    # Get current training steps
    current_steps = 500000  # Fold 1 was trained with 500K steps
    additional_steps = target_steps - current_steps

    if additional_steps <= 0:
        print(f"Model already trained to {current_steps} steps. Target: {target_steps}")
        sys.exit(0)

    print(f"\nContinuing training:")
    print(f"  Current steps: {current_steps:,}")
    print(f"  Target steps: {target_steps:,}")
    print(f"  Additional steps: {additional_steps:,}")

    # Callbacks
    checkpoint_dir = base_dir / "checkpoints"
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=str(checkpoint_dir),
        name_prefix=f"fold_{fold:02d}_continued"
    )
    progress_callback = ProgressCallback(log_freq=10000)
    callbacks = CallbackList([checkpoint_callback, progress_callback])

    # Continue training
    print("\nTraining...")
    model.learn(
        total_timesteps=additional_steps,
        callback=callbacks,
        progress_bar=False,
        reset_num_timesteps=False
    )

    # Save model with new name
    output_name = f"fold_{fold:02d}_{target_steps//1000}k.zip"
    output_path = base_dir / "models" / output_name
    model.save(str(output_path))
    print(f"\nModel saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Continue fold training')
    parser.add_argument('--fold', type=int, required=True, help='Fold number')
    parser.add_argument('--target', type=int, required=True, help='Target total timesteps')

    args = parser.parse_args()
    continue_training(args.fold, args.target)


if __name__ == "__main__":
    main()
