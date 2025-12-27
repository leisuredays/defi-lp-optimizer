"""
Rolling Window Walk-Forward Training with WFE Analysis

Implements Robert Pardo's Walk-Forward Efficiency methodology:
- Rolling window: 300 days train, 100 days test (default)
- Train:Test ratio = 3:1 (increased training emphasis)
- WFE = (Out-of-Sample Return) / (In-Sample Return) × 100%

Configuration Options:
- Default (300/100): 10 folds, 67% overlap, 3:1 ratio
- Alternative (270/90): 11 folds, 67% overlap, 3:1 ratio
- Legacy (180/30): 38 folds, 83% overlap - more data points

WFE Interpretation:
- ≥ 50%: Strategy valid, consider for production
- 30-50%: Caution, possible overfitting
- < 30%: Overfitting, discard

Usage:
  python train_wfe.py                         # Default: 300/100
  python train_wfe.py --window 300 --test 100 # Explicit 300/100
  python train_wfe.py --window 180 --test 30  # Legacy config
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import sqlite3
import pandas as pd
import numpy as np
import torch.nn as nn
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from uniswap_v3.ml import UniswapV3LPEnv


class TrainingProgressCallback(BaseCallback):
    """Callback for logging training progress with rewards."""

    def __init__(self, fold_num: int, log_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.fold_num = fold_num
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_log_step = 0

    def _on_step(self) -> bool:
        # Collect episode info
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                if 'l' in info:
                    self.episode_lengths.append(info['l'])

        # Log progress at intervals
        if self.num_timesteps - self.last_log_step >= self.log_freq:
            self._log_progress()
            self.last_log_step = self.num_timesteps

        return True

    def _log_progress(self):
        timestamp = datetime.now().strftime("%H:%M:%S")

        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-10:]  # Last 10 episodes
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            max_reward = np.max(recent_rewards)
            min_reward = np.min(recent_rewards)

            print(f"[{timestamp}] Fold {self.fold_num} | "
                  f"Steps: {self.num_timesteps:,} | "
                  f"Episodes: {len(self.episode_rewards)} | "
                  f"Reward: {mean_reward:.2f} ± {std_reward:.2f} "
                  f"(min: {min_reward:.2f}, max: {max_reward:.2f})")
        else:
            print(f"[{timestamp}] Fold {self.fold_num} | "
                  f"Steps: {self.num_timesteps:,} | "
                  f"Collecting experience...")

        sys.stdout.flush()

    def _on_training_end(self):
        if len(self.episode_rewards) > 0:
            print(f"\n  Training Summary (Fold {self.fold_num}):")
            print(f"    Total Episodes: {len(self.episode_rewards)}")
            print(f"    Final Mean Reward: {np.mean(self.episode_rewards[-10:]):.2f}")
            print(f"    Best Episode Reward: {np.max(self.episode_rewards):.2f}")
            print(f"    Worst Episode Reward: {np.min(self.episode_rewards):.2f}")
            sys.stdout.flush()


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


def train_model(train_data: pd.DataFrame, timesteps: int = 500000,
                fold_num: int = 1, checkpoint_dir: Path = None,
                checkpoint_freq: int = 100000, version: str = "v1") -> tuple:
    """Train a new PPO model on training data with checkpoints.

    Returns:
        tuple: (model, vec_normalize) - trained model and normalization wrapper
    """
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

    env = DummyVecEnv([lambda: create_env(train_data)])

    # Wrap with VecNormalize for reward normalization
    # norm_obs=True: normalize observations
    # norm_reward=True: normalize rewards (running mean/std)
    # clip_obs=10.0: clip normalized observations
    # gamma=0.99: discount factor for reward normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={'net_arch': [256, 256], 'activation_fn': nn.SiLU},
        device='cuda',
        verbose=0
    )

    # Progress callback (log every 10K steps)
    progress_cb = TrainingProgressCallback(fold_num=fold_num, log_freq=10000)
    callbacks = [progress_cb]

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix=f"fold_{fold_num:02d}_{version}",
            verbose=1
        )
        callbacks.append(checkpoint_cb)

    try:
        model.learn(total_timesteps=timesteps, callback=CallbackList(callbacks), progress_bar=True)
    except Exception as e:
        print(f"\n❌ ERROR during training fold {fold_num}:")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        raise

    return model, env  # Return both model and VecNormalize wrapper


def evaluate_model(model: PPO, data: pd.DataFrame, label: str = "") -> dict:
    """Evaluate model on data and return metrics."""
    try:
        env = create_env(data)
        obs, info = env.reset(seed=42)

        # Set initial range from model
        action, _ = model.predict(obs, deterministic=True)
        env.unwrapped.set_initial_range_from_action(action)

        total_reward = 0
        done = False
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step_count += 1

        result = {
            'total_reward': total_reward,
            'net_return': info['net_return'],
            'fees': info['cumulative_fees'],
            'il': info['cumulative_il'],
            'gas': info['cumulative_gas'],
            'rebalances': info['total_rebalances'],
            'hours': len(data),
            'steps': step_count
        }

        # Print detailed metrics
        print(f"    {label}: Return=${result['net_return']:+,.2f} | "
              f"Fees=${result['fees']:,.2f} | IL=${result['il']:,.2f} | "
              f"Gas=${result['gas']:,.2f} | Rebal={result['rebalances']}")
        sys.stdout.flush()

        return result

    except Exception as e:
        print(f"\n❌ ERROR during evaluation {label}:")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        raise


def save_fold_tracking(model: PPO, train_data: pd.DataFrame, test_data: pd.DataFrame,
                       fold_num: int, output_dir: Path) -> None:
    """Save tracking data for fold analysis."""
    from datetime import datetime as dt

    def run_tracking(model, data, label):
        """Run episode and collect tracking data."""
        env = create_env(data)
        obs, _ = env.reset(seed=42)
        unwrapped = env.unwrapped

        tracking = []
        action, _ = model.predict(obs, deterministic=True)
        unwrapped.set_initial_range_from_action(action)

        done = False
        while not done:
            current_idx = unwrapped.episode_start_idx + unwrapped.current_step
            current_data = unwrapped.features_df.iloc[current_idx]
            timestamp = dt.fromtimestamp(current_data['periodStartUnix'])
            price = current_data['close']

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            tracking.append({
                'timestamp': timestamp,
                'price': price,
                'min_range': unwrapped.position_min_price,
                'max_range': unwrapped.position_max_price,
                'in_range': unwrapped.position_min_price < price < unwrapped.position_max_price,
                'rebalanced': info.get('rebalanced', False),
                'fees': info.get('fees', 0),
                'cumulative_fees': info.get('cumulative_fees', 0),
                'cumulative_il': info.get('cumulative_il', 0),
                'net_return': info.get('net_return', 0),
                'reward': reward
            })
            done = terminated or truncated

        return pd.DataFrame(tracking)

    # Save test data tracking (more important for WFE analysis)
    test_df = run_tracking(model, test_data, "test")
    test_path = output_dir / "tracking_data" / f"fold_{fold_num:02d}_tracking.csv"
    test_df.to_csv(test_path, index=False)
    print(f"  Test tracking saved: tracking_data/{test_path.name}")


def rolling_window_wfe(
    data: pd.DataFrame,
    window_days: int = 180,
    test_days: int = 30,
    timesteps_per_fold: int = 500000,
    output_dir: Path = None,
    version: str = "v1",
    start_fold: int = 1,
    end_fold: int = None
) -> list:
    """
    Perform rolling window Walk-Forward Evaluation.

    Args:
        data: Full dataset (hourly)
        window_days: Training window size in days
        test_days: Test period size in days
        timesteps_per_fold: Training steps per fold
        output_dir: Directory to save fold models

    Returns:
        List of fold results with WFE metrics
    """
    window_hours = window_days * 24
    test_hours = test_days * 24

    total_hours = len(data)
    n_folds = (total_hours - window_hours) // test_hours

    print("="*70)
    print(" ROLLING WINDOW WALK-FORWARD EVALUATION")
    print("="*70)
    print(f"\nData: {total_hours:,} hours ({total_hours/24:.0f} days)")
    print(f"Window: {window_days} days train, {test_days} days test")
    print(f"Folds: {n_folds}")
    print(f"Timesteps per fold: {timesteps_per_fold:,}")
    print("="*70)
    sys.stdout.flush()

    results = []

    # Start from specified fold (1-indexed input, 0-indexed loop)
    start_idx_fold = start_fold - 1

    # End at specified fold (inclusive)
    if end_fold is not None:
        max_fold_idx = min(end_fold, n_folds)
    else:
        max_fold_idx = n_folds

    if start_idx_fold > 0:
        print(f"\n⚠️ Resuming from Fold {start_fold} (skipping Folds 1-{start_fold-1})")
    if end_fold is not None:
        print(f"📌 Training Folds {start_fold} to {max_fold_idx}")

    for fold in range(start_idx_fold, max_fold_idx):
        start_idx = fold * test_hours
        train_end = start_idx + window_hours
        test_end = train_end + test_hours

        if test_end > total_hours:
            break

        train_data = data.iloc[start_idx:train_end].copy().reset_index(drop=True)
        test_data = data.iloc[train_end:test_end].copy().reset_index(drop=True)

        train_start_date = datetime.fromtimestamp(train_data['periodStartUnix'].iloc[0])
        test_end_date = datetime.fromtimestamp(test_data['periodStartUnix'].iloc[-1])

        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"Train: {train_start_date.strftime('%Y-%m-%d')} ({len(train_data)} hours)")
        print(f"Test:  {test_end_date.strftime('%Y-%m-%d')} ({len(test_data)} hours)")
        print("-"*70)
        sys.stdout.flush()

        # Train
        print("\n[Training...]")
        sys.stdout.flush()
        checkpoint_dir = output_dir / "checkpoints" if output_dir else None
        model, vec_normalize = train_model(
            train_data,
            timesteps_per_fold,
            fold_num=fold + 1,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=500000,  # Save every 500K steps
            version=version
        )

        # Evaluate on train (in-sample)
        print("[Evaluating...]")
        sys.stdout.flush()
        in_sample = evaluate_model(model, train_data, label="In-Sample")
        out_sample = evaluate_model(model, test_data, label="Out-Sample")

        # Calculate WFE
        if in_sample['net_return'] > 0:
            wfe = (out_sample['net_return'] / in_sample['net_return']) * 100
        else:
            wfe = 0 if out_sample['net_return'] <= 0 else 100

        # Determine WFE rating
        if wfe >= 50:
            rating = "✅ VALID"
        elif wfe >= 30:
            rating = "⚠️ CAUTION"
        else:
            rating = "❌ OVERFIT"

        fold_result = {
            'fold': fold + 1,
            'train_start': train_start_date.isoformat(),
            'test_end': test_end_date.isoformat(),
            'in_sample': in_sample,
            'out_sample': out_sample,
            'wfe': wfe,
            'rating': rating
        }
        results.append(fold_result)

        # Print fold summary
        print(f"\n{'─'*40}")
        print(f"  WFE: {wfe:.1f}% {rating}")
        print(f"{'─'*40}")
        sys.stdout.flush()

        # Save fold model
        if output_dir:
            model_path = output_dir / "models" / f"fold_{fold+1:02d}_{version}_{timesteps_per_fold//1000}k.zip"
            model.save(str(model_path))
            # Save VecNormalize stats for inference
            vec_norm_path = output_dir / "models" / f"fold_{fold+1:02d}_{version}_{timesteps_per_fold//1000}k_vecnorm.pkl"
            vec_normalize.save(str(vec_norm_path))
            print(f"  Model saved: models/{model_path.name}")
            print(f"  VecNormalize saved: models/{vec_norm_path.name}")
            sys.stdout.flush()

        # Save intermediate results
        if output_dir:
            results_path = output_dir / "wfe_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Results updated: {results_path.name}")

        # Save tracking data for analysis (all folds)
        if output_dir:  # Save for all folds including Fold 1
            try:
                save_fold_tracking(model, train_data, test_data, fold + 1, output_dir)
            except Exception as e:
                print(f"  Warning: Could not save tracking data: {e}")

    # Summary
    print("\n" + "="*70)
    print(" WALK-FORWARD SUMMARY")
    print("="*70)

    wfe_scores = [r['wfe'] for r in results]
    avg_wfe = np.mean(wfe_scores)

    valid_folds = sum(1 for r in results if r['wfe'] >= 50)
    caution_folds = sum(1 for r in results if 30 <= r['wfe'] < 50)
    overfit_folds = sum(1 for r in results if r['wfe'] < 30)

    print(f"\nAverage WFE: {avg_wfe:.1f}%")
    print(f"Valid folds (≥50%): {valid_folds}/{len(results)}")
    print(f"Caution folds (30-50%): {caution_folds}/{len(results)}")
    print(f"Overfit folds (<30%): {overfit_folds}/{len(results)}")

    # Overall verdict
    if avg_wfe >= 50 and valid_folds >= len(results) * 0.6:
        verdict = "✅ STRATEGY VALIDATED - Ready for production"
    elif avg_wfe >= 30:
        verdict = "⚠️ NEEDS IMPROVEMENT - Review and tune"
    else:
        verdict = "❌ OVERFITTING DETECTED - Redesign strategy"

    print(f"\nVerdict: {verdict}")
    print("="*70)

    return results


def main():
    parser = argparse.ArgumentParser(description='Rolling Window WFE Training')
    parser.add_argument('--window', type=int, default=300, help='Training window (days), default: 300')
    parser.add_argument('--test', type=int, default=100, help='Test period (days), default: 100')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Timesteps per fold (default: 1M)')
    parser.add_argument('--version', type=str, default='v5_300_100', help='Version tag for model naming')
    parser.add_argument('--start-fold', type=int, default=1, help='Start from this fold number (for resuming)')
    parser.add_argument('--end-fold', type=int, default=None, help='End at this fold number (inclusive)')
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "uniswap_v3/data/pool_data.db"
    output_dir = project_root / "uniswap_v3/models/rolling_wfe"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create subdirectories
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "tracking_data").mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    data = load_data_from_db(db_path)
    print(f"Loaded {len(data):,} hours ({len(data)/24:.0f} days)")

    # Run rolling window WFE
    print(f"Version: {args.version}")
    print(f"Timesteps per fold: {args.timesteps:,}")
    if args.start_fold > 1:
        print(f"Starting from fold: {args.start_fold}")
    if args.end_fold:
        print(f"Ending at fold: {args.end_fold}")

    results = rolling_window_wfe(
        data=data,
        window_days=args.window,
        test_days=args.test,
        timesteps_per_fold=args.timesteps,
        output_dir=output_dir,
        version=args.version,
        start_fold=args.start_fold,
        end_fold=args.end_fold
    )

    # Save results
    results_path = output_dir / "wfe_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
