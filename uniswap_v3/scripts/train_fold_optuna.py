#!/usr/bin/env python
"""
Train a fold model with Optuna hyperparameter optimization.
Optimizes action space, network architecture, and PPO hyperparameters.

Paper: arXiv:2501.07508 - Section 5 & Appendix A
- Action space optimized as hyperparameter
- 50 agents trained per fold, best selected

Usage:
  python train_fold_optuna.py --fold 1 --trials 50 --steps 200000
"""
import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from uniswap_v3.ml import UniswapV3LPEnv


class EvalCallback(BaseCallback):
    """Callback to track episode rewards during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
        return True

    def get_mean_reward(self, last_n=10):
        if len(self.episode_rewards) >= last_n:
            return np.mean(self.episode_rewards[-last_n:])
        elif len(self.episode_rewards) > 0:
            return np.mean(self.episode_rewards)
        return -np.inf


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


def create_env(data: pd.DataFrame):
    """Create environment with continuous action space."""
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
        episode_length_hours=len(data) - 10
    )
    # 3D Continuous action space is already set in environment:
    # action[0] = lower_pct (1% ~ 99%)
    # action[1] = upper_pct (1% ~ 500%)
    # action[2] = rebalance_signal (0 ~ 1)

    return Monitor(env)


def objective(trial, train_data, total_steps):
    """Optuna objective function - optimize hyperparameters for 3D continuous action space."""

    # 3D Action Space (환경에서 이미 설정됨):
    # action[0] = lower_pct (1% ~ 99%)
    # action[1] = upper_pct (1% ~ 500%)
    # action[2] = rebalance_signal (0 ~ 1)

    # 1. Network Architecture - 64x64 고정
    hidden_layers = [64, 64]

    # 2. Activation Function
    activation_name = trial.suggest_categorical('activation', ['relu', 'tanh'])
    activation_fn = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh
    }[activation_name]

    # 3. PPO Hyperparameters (연속 제어에 맞게 조정)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
    ent_coef = trial.suggest_float('ent_coef', 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)

    # Create environment (continuous action space)
    env = DummyVecEnv([lambda: create_env(train_data)])

    # Create model with separate policy/value networks
    policy_kwargs = dict(
        net_arch=dict(pi=hidden_layers, vf=hidden_layers),  # Separate actor/critic
        activation_fn=activation_fn
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        device="cuda"
    )

    # Train with callback
    callback = EvalCallback()
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)

    # Get mean reward from last episodes
    mean_reward = callback.get_mean_reward(last_n=10)

    env.close()

    # Store hyperparameters in trial for later retrieval
    trial.set_user_attr('activation', activation_name)

    return mean_reward


def train_with_optuna(fold: int, n_trials: int, total_steps: int):
    """Train fold with Optuna hyperparameter optimization."""

    base_dir = project_root / "uniswap_v3/models/rolling_wfe"
    db_path = project_root / "uniswap_v3/data/pool_data.db"

    # Configuration (Paper: arXiv:2501.07508, Section 6)
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

    # Create Optuna study
    print(f"\nStarting Optuna optimization with {n_trials} trials...")
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, train_data, total_steps),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Get best trial
    best_trial = study.best_trial
    print(f"\n{'='*60}")
    print("Best Trial Results:")
    print(f"{'='*60}")
    print(f"  Best Reward: {best_trial.value:.2f}")
    print(f"  Hidden Layers: [64, 64] (fixed)")
    print(f"  Activation: {best_trial.user_attrs['activation']}")
    print(f"  Learning Rate: {best_trial.params['learning_rate']}")
    print(f"  Clip Range: {best_trial.params['clip_range']}")
    print(f"  Entropy Coef: {best_trial.params['ent_coef']}")
    print(f"  Gamma: {best_trial.params['gamma']}")

    # Retrain best model with full steps
    print(f"\n{'='*60}")
    print("Retraining best model...")
    print(f"{'='*60}")

    hidden_layers = [64, 64]  # Fixed network size
    activation_name = best_trial.user_attrs['activation']
    activation_fn = {'relu': nn.ReLU, 'tanh': nn.Tanh}[activation_name]

    env = DummyVecEnv([lambda: create_env(train_data)])

    policy_kwargs = dict(
        net_arch=dict(pi=hidden_layers, vf=hidden_layers),  # Separate actor/critic
        activation_fn=activation_fn
    )

    final_model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=best_trial.params['learning_rate'],
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=best_trial.params['gamma'],
        gae_lambda=0.95,
        clip_range=best_trial.params['clip_range'],
        ent_coef=best_trial.params['ent_coef'],
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        device="cuda"
    )

    callback = EvalCallback()
    final_model.learn(total_timesteps=total_steps, callback=callback, progress_bar=True)

    # Save model
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = f"fold_{fold:02d}_optuna_{total_steps//1000}k.zip"
    model_path = models_dir / model_name
    final_model.save(str(model_path))
    print(f"\nModel saved: {model_path}")

    # Save hyperparameters
    config = {
        'fold': fold,
        'total_steps': total_steps,
        'n_trials': n_trials,
        'best_reward': best_trial.value,
        'action_space': '3D_continuous',  # [lower_pct, upper_pct, rebalance_signal]
        'hidden_layers': hidden_layers,
        'activation': activation_name,
        'learning_rate': best_trial.params['learning_rate'],
        'clip_range': best_trial.params['clip_range'],
        'ent_coef': best_trial.params['ent_coef'],
        'gamma': best_trial.params['gamma'],
    }

    config_path = models_dir / f"fold_{fold:02d}_optuna_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")

    env.close()
    return model_path, config


def main():
    parser = argparse.ArgumentParser(description='Train fold with Optuna optimization')
    parser.add_argument('--fold', type=int, default=1, help='Fold number')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials (paper: 50)')
    parser.add_argument('--steps', type=int, default=200000, help='Steps per trial (default: 200K)')

    args = parser.parse_args()

    print("="*60)
    print(f"Optuna Hyperparameter Optimization (3D Action Space)")
    print("="*60)
    print(f"  Fold: {args.fold}")
    print(f"  Trials: {args.trials}")
    print(f"  Steps per trial: {args.steps:,}")
    print()
    print("Configuration:")
    print("  - Action Space: 3D Continuous [lower_pct, upper_pct, rebalance_signal]")
    print("  - Network: [64, 64] (fixed)")
    print("  - Activation: relu, tanh")
    print("Optimizing:")
    print("  - LR, Clip, Entropy, Gamma")
    print("="*60)

    train_with_optuna(args.fold, args.trials, args.steps)


if __name__ == "__main__":
    main()
