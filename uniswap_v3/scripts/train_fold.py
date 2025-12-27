#!/usr/bin/env python
"""
Train a fold model from scratch with specified total timesteps.
Saves checkpoints every 500K steps for evaluation.

Usage:
  python train_fold_fresh.py --fold 1 --total 2000000
"""
import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from uniswap_v3.ml import UniswapV3LPEnv


class ProgressCallback(BaseCallback):
    """Callback to log training progress."""

    def __init__(self, log_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.rewards = []

    def _on_step(self) -> bool:
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.rewards.append(info['episode']['r'])

        if self.num_timesteps % self.log_freq == 0:
            if self.rewards:
                mean_r = np.mean(self.rewards[-100:]) if len(self.rewards) >= 100 else np.mean(self.rewards)
                std_r = np.std(self.rewards[-100:]) if len(self.rewards) >= 100 else np.std(self.rewards)
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Steps: {self.num_timesteps:,} | Reward: {mean_r:.2f} ± {std_r:.2f}")
            else:
                print(f"Steps: {self.num_timesteps:,}")
            sys.stdout.flush()

        return True


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
        episode_length_hours=episode_length
    )
    return Monitor(env)


def train_fresh(fold: int, total_steps: int, version: str = "v4"):
    """Train a fold model from scratch (Paper: arXiv:2501.07508 style)."""

    base_dir = project_root / "uniswap_v3/models/rolling_wfe"
    db_path = project_root / "uniswap_v3/data/pool_data.db"

    # Configuration (Paper: arXiv:2501.07508, Section 6)
    # Training: 7,500 hours (~10 months), Testing: 1,500 hours (~2 months)
    WINDOW_HOURS = 7500  # ~312.5 days training
    TEST_HOURS = 1500    # ~62.5 days testing

    # Calculate fold data range
    train_start = (fold - 1) * TEST_HOURS
    train_end = train_start + WINDOW_HOURS

    # Load data
    print("Loading data...")
    data = load_data_from_db(db_path)
    train_data = data.iloc[train_start:train_end].reset_index(drop=True)
    print(f"Training data: {len(train_data)} hours ({len(train_data)/24:.0f} days)")

    # Create environment (no normalization - paper style)
    env = DummyVecEnv([lambda: create_env(train_data)])

    # Create NEW model from scratch
    # Paper hyperparameters (Table 1, 2 - arXiv:2501.07508)
    print(f"\nCreating new PPO model (version: {version})...")

    # Network architecture (Table 1)
    # Hidden layers: [6, 4], Activation: sigmoid
    import torch.nn as nn
    policy_kwargs = dict(
        net_arch=[6, 4],  # Hidden layers
        activation_fn=nn.Sigmoid  # Activation function
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.00005,  # Table 2: LR
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.999,            # Table 2: γ
        gae_lambda=0.95,
        clip_range=0.05,        # Table 2: Clip
        ent_coef=0.00001,       # Table 2: c2 (entropy)
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        device="cuda"
    )

    print(f"\nTraining for {total_steps:,} total steps...")
    print(f"  Version: {version} (Paper style: discrete action, no normalization)")

    # Callbacks - save every 500K
    checkpoint_dir = base_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path=str(checkpoint_dir),
        name_prefix=f"fold_{fold:02d}_{version}"
    )
    progress_callback = ProgressCallback(log_freq=10000)
    callbacks = CallbackList([checkpoint_callback, progress_callback])

    # Train
    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        progress_bar=False
    )

    # Save final model
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    final_name = f"fold_{fold:02d}_{version}_{total_steps//1000}k.zip"
    final_path = models_dir / final_name
    model.save(str(final_path))
    print(f"\nFinal model saved: {final_path}")

    return final_path


def main():
    parser = argparse.ArgumentParser(description='Train fold from scratch')
    parser.add_argument('--fold', type=int, default=1, help='Fold number')
    parser.add_argument('--total', type=int, default=100000, help='Total timesteps (paper: 100K)')
    parser.add_argument('--version', type=str, default='v4', help='Model version tag')

    args = parser.parse_args()

    print("="*60)
    print(f"Training Fold {args.fold} (Paper: arXiv:2501.07508)")
    print(f"  Total steps: {args.total:,}")
    print(f"  Version: {args.version}")
    print("  Hyperparameters (Table 1, 2):")
    print("    - Action space: {0, 20, 50}")
    print("    - Network: [6, 4] + sigmoid")
    print("    - LR: 0.00005, Clip: 0.05, γ: 0.999")
    print("    - Entropy: 0.00001")
    print("  Environment:")
    print("    - Observation: 11-dim")
    print("    - Reward: fees - LVR - I[a≠0]*gas")
    print("    - Gas: $5 per rebalance")
    print("="*60)

    train_fresh(args.fold, args.total, args.version)


if __name__ == "__main__":
    main()
