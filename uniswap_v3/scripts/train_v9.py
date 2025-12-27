#!/usr/bin/env python
"""
Train v9 model with Excess Return reward function.

v9 Reward: (LP_return - HODL_return) + time_in_range_bonus - gas_penalty

This addresses the fundamental issue where IL always dominates fees in v8.
By comparing LP to HODL, we get a relative performance metric that:
- Is positive when LP outperforms HODL (fees > IL)
- Is negative when LP underperforms HODL (fees < IL)
- Treats "do nothing" (HODL) as the baseline
"""
import sys
sys.path.insert(0, '/home/zekiya/liquidity/uniswap-v3-simulator')

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Import v9 environment
from uniswap_v3.ml.environment_v9 import UniswapV3LPEnvV9


class TrainingCallback(BaseCallback):
    """Print training progress."""
    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Steps: {self.num_timesteps:,} | Reward: {mean_reward:.4f}", flush=True)
        return True


def load_data(db_path: str, start_idx: int = 0, length: int = 7500):
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


def main():
    db_path = "/home/zekiya/liquidity/uniswap-v3-simulator/uniswap_v3/data/pool_data.db"
    model_dir = Path("/home/zekiya/liquidity/uniswap-v3-simulator/uniswap_v3/models/rolling_wfe/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Fold 1 data ranges
    TRAIN_START = 0
    TRAIN_LENGTH = 7500

    print("=" * 60, flush=True)
    print("Training v9 Model (Excess Return Reward)", flush=True)
    print("=" * 60, flush=True)
    print(f"Training data: {TRAIN_LENGTH} hours", flush=True)
    print("Reward: (LP_return - HODL_return) + in_range_bonus - gas_penalty", flush=True)
    print("Action space: [0, 4000, 8000, 16000] (No rebalance, ±20%, ±40%, ±80%)", flush=True)

    # Load training data
    train_data, pool_config = load_data(db_path, TRAIN_START, TRAIN_LENGTH)

    # Create v9 environment
    env = UniswapV3LPEnvV9(
        historical_data=train_data,
        pool_config=pool_config,
        initial_investment=10000,
        episode_length_hours=720,  # 30 days
        reward_weights={
            'in_range_bonus': 0.001,  # Small bonus for being in range
            'gas_penalty_scale': 1.0,
        }
    )

    print(f"Environment action space: {env.action_space}", flush=True)
    print(f"Environment observation space: {env.observation_space}", flush=True)

    # Create PPO model (paper hyperparameters)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        device='cuda',
    )

    # Train
    total_timesteps = 500_000
    print(f"\nTraining {total_timesteps//1000}k steps...", flush=True)

    callback = TrainingCallback(check_freq=10000)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    model_path = model_dir / "fold_01_v9_excess_return_500k.zip"
    model.save(str(model_path))
    print(f"\nSaved: {model_path}")


if __name__ == '__main__':
    main()
