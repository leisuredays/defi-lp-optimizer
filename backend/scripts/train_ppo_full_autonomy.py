"""
PPO Training Script for WETH/USDT 0.3% (Full Autonomy Mode)

Key differences from previous version:
- No forced rebalancing (model decides when to rebalance)
- No cooldown period
- Equal reward weights (alpha=beta=gamma=1.0)
- Model has full autonomy over rebalancing decisions

Usage:
  python train_ppo_full_autonomy.py                  # Start fresh
  python train_ppo_full_autonomy.py --resume latest  # Resume from checkpoint
"""
import sys
import argparse
from pathlib import Path
import yaml
import json
import pandas as pd
import numpy as np
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from app.ml.environment import UniswapV3LPEnv
from app.ml.callbacks import RewardLoggingCallback, ProgressCallback


def load_config(config_path: str = None) -> dict:
    """Load training configuration"""
    if config_path is None:
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / "config" / "training_config.yaml"

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_and_split_data(pool_file: Path, train_split: float, val_split: float):
    """Load pool data and split into train/val/test sets."""
    df = pd.read_parquet(pool_file)
    df = df.sort_values('periodStartUnix').reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def create_environment(pool_data: pd.DataFrame, config: dict, log_dir: Path = None):
    """Create and wrap RL environment"""

    pool_config = {
        'protocol': int(pool_data['protocol_id'].iloc[0]),
        'feeTier': int(pool_data['fee_tier'].iloc[0]),
        'token0': {'decimals': int(pool_data['token0_decimals'].iloc[0])},
        'token1': {'decimals': int(pool_data['token1_decimals'].iloc[0])}
    }

    env = UniswapV3LPEnv(
        historical_data=pool_data,
        pool_config=pool_config,
        initial_investment=config['environment']['initial_investment'],
        episode_length_hours=config['environment']['episode_length_hours'],
        reward_weights=config['environment']['reward_weights']
    )

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(log_dir / "monitor.csv"))
    else:
        env = Monitor(env)

    return env


def find_checkpoint(checkpoint_dir: Path, resume_arg: str) -> Path:
    """Find checkpoint file based on resume argument"""
    if resume_arg == "latest":
        checkpoints = list(checkpoint_dir.glob("ppo_weth_usdt_checkpoint_*_steps.zip"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")

        def get_steps(p):
            parts = p.stem.split("_")
            return int(parts[-2])

        return max(checkpoints, key=get_steps)
    else:
        step_count = int(resume_arg)
        pattern = f"ppo_weth_usdt_checkpoint_{step_count}_steps.zip"
        matches = list(checkpoint_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Checkpoint not found for step {step_count}")
        return matches[0]


def main():
    """Main training pipeline"""

    parser = argparse.ArgumentParser(description='PPO Full Autonomy Training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint: step count or "latest"')
    args = parser.parse_args()

    print("="*70)
    print(" " * 6 + "PPO TRAINING - WETH/USDT 0.3% (FULL AUTONOMY)")
    print("="*70)
    print("\nKey Features:")
    print("  - NO forced rebalancing (model decides)")
    print("  - NO cooldown period (model decides)")
    print("  - Equal reward weights (alpha=beta=gamma=1.0)")
    print("  - Objective: Maximize (Fees - IL - Gas)")

    # [1/6] Load configuration
    print("\n[1/6] Loading configuration...")
    print("-"*70)

    try:
        config = load_config()
        print(f"  Model: {config['output']['model_name']}")
        print(f"  Total timesteps: {config['training']['total_timesteps']:,}")
        print(f"  Reward weights: alpha={config['environment']['reward_weights']['alpha']}, "
              f"beta={config['environment']['reward_weights']['beta']}, "
              f"gamma={config['environment']['reward_weights']['gamma']}")
    except Exception as e:
        print(f"  Error: {e}")
        return

    # [2/6] Load and split data
    print("\n[2/6] Loading pool data...")
    print("-"*70)

    pool_file = Path(config['data']['pool_file'])
    if not pool_file.exists():
        pool_file = Path(__file__).parent.parent / config['data']['pool_file']

    if not pool_file.exists():
        print(f"  Pool data not found: {pool_file}")
        print(f"  Run: python scripts/collect_weth_usdt_data.py")
        return

    try:
        train_df, val_df, test_df = load_and_split_data(
            pool_file,
            config['data']['train_split'],
            config['data']['val_split']
        )

        print(f"  Loaded: {pool_file.name}")
        print(f"  Pool: {train_df['token0_symbol'].iloc[0]}/{train_df['token1_symbol'].iloc[0]}")
        print(f"  Fee tier: 0.3%")
        print(f"\n  Data split:")
        print(f"    Training: {len(train_df):,} hours ({len(train_df)/24:.0f} days)")
        print(f"    Validation: {len(val_df):,} hours ({len(val_df)/24:.0f} days)")
        print(f"    Test: {len(test_df):,} hours ({len(test_df)/24:.0f} days)")

    except Exception as e:
        print(f"  Error: {e}")
        return

    # [3/6] Create environments
    print("\n[3/6] Creating environments...")
    print("-"*70)

    log_dir = Path(config['output']['tensorboard_log']) / "train"
    train_env = create_environment(train_df, config, log_dir)
    train_env = DummyVecEnv([lambda: train_env])
    print(f"  Training environment created")

    eval_log_dir = Path(config['output']['tensorboard_log']) / "eval"
    eval_env = create_environment(val_df, config, eval_log_dir)
    eval_env = DummyVecEnv([lambda: eval_env])
    print(f"  Evaluation environment created")

    # [4/6] Initialize PPO
    print("\n[4/6] Initializing PPO model...")
    print("-"*70)

    resume_steps = 0
    try:
        if args.resume:
            checkpoint_dir = Path(config['output']['checkpoint_dir'])
            checkpoint_path = find_checkpoint(checkpoint_dir, args.resume)
            parts = checkpoint_path.stem.split("_")
            resume_steps = int(parts[-2])

            print(f"  Loading checkpoint: {checkpoint_path.name}")
            print(f"  Resuming from step: {resume_steps:,}")

            model = PPO.load(
                checkpoint_path,
                env=train_env,
                tensorboard_log=config['output']['tensorboard_log'],
                device=config['hardware']['device'],
                verbose=1
            )
            print(f"  PPO resumed from checkpoint")
        else:
            policy_kwargs = config['ppo']['policy_kwargs'].copy()
            policy_kwargs.update({
                'activation_fn': nn.SiLU,
                'normalize_images': False,
                'ortho_init': True,
            })

            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                learning_rate=config['ppo']['learning_rate'],
                n_steps=config['ppo']['n_steps'],
                batch_size=config['ppo']['batch_size'],
                n_epochs=config['ppo']['n_epochs'],
                gamma=config['ppo']['gamma'],
                gae_lambda=config['ppo']['gae_lambda'],
                clip_range=config['ppo']['clip_range'],
                ent_coef=config['ppo']['ent_coef'],
                vf_coef=config['ppo']['vf_coef'],
                max_grad_norm=config['ppo']['max_grad_norm'],
                policy_kwargs=policy_kwargs,
                tensorboard_log=config['output']['tensorboard_log'],
                device=config['hardware']['device'],
                verbose=1
            )
            print(f"  PPO initialized (fresh)")

        total_params = sum(p.numel() for p in model.policy.parameters())
        print(f"  Network: obs -> {config['ppo']['policy_kwargs']['net_arch']} -> 3")
        print(f"  Activation: SiLU (Swish)")
        print(f"  Parameters: ~{total_params:,}")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # [5/6] Setup callbacks
    print("\n[5/6] Setting up callbacks...")
    print("-"*70)

    Path(config['output']['model_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['output']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=config['training']['checkpoint_freq'],
        save_path=config['output']['checkpoint_dir'],
        name_prefix="ppo_weth_usdt_checkpoint",
        verbose=1
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=config['output']['model_dir'],
        log_path=config['output']['model_dir'],
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        deterministic=True,
        verbose=1
    )

    reward_cb = RewardLoggingCallback(verbose=1)
    progress_cb = ProgressCallback(
        total_timesteps=config['training']['total_timesteps'],
        check_freq=50000,
        verbose=1
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb, reward_cb, progress_cb])
    print(f"  Callbacks configured")

    # [6/6] Train
    print("\n[6/6] Starting training...")
    print("="*70)

    total_timesteps = config['training']['total_timesteps']
    remaining_timesteps = total_timesteps - resume_steps

    print(f"\nModel: {config['output']['model_name']}")
    if resume_steps > 0:
        print(f"Resuming from: {resume_steps:,} steps")
        print(f"Remaining: {remaining_timesteps:,} steps")
    else:
        print(f"Timesteps: {total_timesteps:,}")
    print(f"Monitor: tensorboard --logdir {config['output']['tensorboard_log']}")
    print("\n" + "="*70 + "\n")

    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False if resume_steps > 0 else True
        )
        print("\n" + "="*70)
        print("  Training completed!")

    except KeyboardInterrupt:
        print("\n" + "="*70)
        print("  Training interrupted")

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save final model
    print("\n[Saving] Final model...")
    print("-"*70)

    model_path = Path(config['output']['model_dir']) / config['output']['model_name']
    model.save(str(model_path))
    print(f"  Saved: {model_path}.zip")

    # Save training stats
    stats = {
        'pool': 'WETH/USDT 0.3%',
        'pool_id': config['pool']['pool_id'],
        'protocol_id': config['pool']['protocol_id'],
        'fee_tier': config['pool']['fee_tier'],
        'total_timesteps': config['training']['total_timesteps'],
        'train_hours': len(train_df),
        'val_hours': len(val_df),
        'test_hours': len(test_df),
        'reward_weights': config['environment']['reward_weights'],
        'features': {
            'full_autonomy': True,
            'no_forced_rebalancing': True,
            'no_cooldown': True,
        }
    }

    stats_path = Path(config['output']['eval_results_path'])
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Stats saved: {stats_path}")

    print("\n" + "="*70)
    print(" " * 25 + "SUMMARY")
    print("="*70)
    print(f"\nPool: WETH/USDT 0.3% (Ethereum)")
    print(f"Model: {model_path}.zip")
    print(f"\nNext steps:")
    print(f"  1. Review: tensorboard --logdir {config['output']['tensorboard_log']}")
    print(f"  2. Evaluate: python scripts/evaluate_single_pool.py")
    print("="*70)


if __name__ == "__main__":
    main()
