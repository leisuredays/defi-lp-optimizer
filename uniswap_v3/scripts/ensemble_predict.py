"""
Ensemble Prediction using Fold 1-8 Models

Uses trained fold models as an ensemble to predict on remaining data (after Fold 8).
Ensemble strategy: Average action outputs from all models.
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import sqlite3
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

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
        episode_length_hours=episode_length,
        obs_dim=28
    )
    return Monitor(env)


class EnsemblePredictor:
    """Ensemble predictor using multiple fold models."""

    def __init__(self, model_paths: list, weights: list = None, method: str = 'mean'):
        """
        Args:
            model_paths: List of paths to model .zip files
            weights: Optional weights for each model (default: equal weights)
            method: Ensemble method ('mean', 'median', 'vote')
        """
        self.models = []
        self.method = method
        for path in model_paths:
            print(f"Loading model: {path}")
            model = PPO.load(str(path))
            self.models.append(model)

        if weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)
        else:
            self.weights = np.array(weights) / np.sum(weights)

        print(f"Loaded {len(self.models)} models")
        print(f"Weights: {self.weights}")
        print(f"Method: {self.method}")

    def predict(self, obs, deterministic=True, method=None):
        """
        Ensemble prediction.

        Args:
            obs: Observation
            deterministic: Use deterministic prediction
            method: 'mean', 'median', or 'vote' (default: use instance method)

        Returns:
            Ensemble action
        """
        if method is None:
            method = self.method

        actions = []
        for model in self.models:
            action, _ = model.predict(obs, deterministic=deterministic)
            actions.append(action)

        actions = np.array(actions)

        if method == 'mean':
            # Weighted average
            ensemble_action = np.average(actions, axis=0, weights=self.weights)
        elif method == 'median':
            ensemble_action = np.median(actions, axis=0)
        elif method == 'vote':
            # Majority vote for discrete part (rebalance)
            ensemble_action = np.mean(actions, axis=0)
        else:
            ensemble_action = np.mean(actions, axis=0)

        return ensemble_action, None


def run_ensemble_evaluation(
    ensemble: EnsemblePredictor,
    data: pd.DataFrame,
    start_hour: int,
    test_hours: int = 720,
    label: str = ""
) -> dict:
    """
    Run ensemble evaluation on a test period.

    Args:
        ensemble: EnsemblePredictor instance
        data: Full dataset
        start_hour: Start hour for test period
        test_hours: Number of hours to test
        label: Label for logging

    Returns:
        Evaluation metrics
    """
    test_data = data.iloc[start_hour:start_hour + test_hours].copy().reset_index(drop=True)

    if len(test_data) < test_hours:
        print(f"  Warning: Only {len(test_data)} hours available (requested {test_hours})")
        if len(test_data) < 100:
            return None

    env = create_env(test_data)
    obs, info = env.reset(seed=42)
    unwrapped = env.unwrapped

    # Set initial range using ensemble
    action = ensemble.predict(obs, deterministic=True)[0]
    unwrapped.set_initial_range_from_action(action)

    total_reward = 0
    done = False
    step_count = 0

    tracking = []

    while not done:
        current_idx = unwrapped.episode_start_idx + unwrapped.current_step
        current_data = unwrapped.features_df.iloc[current_idx]
        timestamp = datetime.fromtimestamp(current_data['periodStartUnix'])
        price = current_data['close']

        action = ensemble.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step_count += 1

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

    result = {
        'total_reward': total_reward,
        'net_return': info['net_return'],
        'fees': info['cumulative_fees'],
        'il': info['cumulative_il'],
        'lvr': info.get('cumulative_lvr', 0),
        'gas': info['cumulative_gas'],
        'rebalances': info['total_rebalances'],
        'hours': len(test_data),
        'steps': step_count,
        'tracking': pd.DataFrame(tracking)
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Ensemble Prediction')
    parser.add_argument('--start-fold', type=int, default=9,
                        help='Start prediction from this fold (default: 9)')
    parser.add_argument('--end-fold', type=int, default=None,
                        help='End prediction at this fold (default: all)')
    parser.add_argument('--ensemble-size', type=int, default=8,
                        help='Number of recent folds to use in ensemble (default: 8)')
    parser.add_argument('--method', type=str, default='mean',
                        choices=['mean', 'median', 'vote'],
                        help='Ensemble method (default: mean)')
    parser.add_argument('--save-tracking', action='store_true',
                        help='Save tracking data for each fold')
    parser.add_argument('--window', type=int, default=300,
                        help='Training window in days (default: 300)')
    parser.add_argument('--test', type=int, default=100,
                        help='Test period in days (default: 100)')
    parser.add_argument('--version', type=str, default='v5_300_100',
                        help='Model version tag (default: v5_300_100)')
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "uniswap_v3/data/pool_data.db"
    models_dir = project_root / "uniswap_v3/models/rolling_wfe/models"
    output_dir = project_root / "uniswap_v3/models/rolling_wfe/ensemble_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    data = load_data_from_db(db_path)
    total_hours = len(data)
    print(f"Loaded {total_hours:,} hours ({total_hours/24:.0f} days)")

    # Parameters (configurable rolling window)
    window_hours = args.window * 24
    test_hours = args.test * 24
    n_folds = (total_hours - window_hours) // test_hours

    print(f"Rolling window: {args.window} days train, {args.test} days test")

    print(f"\nTotal possible folds: {n_folds}")
    print(f"Starting from fold: {args.start_fold}")

    # Load ensemble models
    model_paths = []
    for fold in range(1, args.ensemble_size + 1):
        # Try new version first, then fallback to legacy version
        model_path = models_dir / f"fold_{fold:02d}_{args.version}_1000k.zip"
        if not model_path.exists():
            model_path = models_dir / f"fold_{fold:02d}_v3_paper_lvr_1000k.zip"

        if model_path.exists():
            model_paths.append(model_path)
            print(f"  Loaded: {model_path.name}")
        else:
            print(f"Warning: Model not found for fold {fold}")

    if len(model_paths) < 2:
        print("Error: Need at least 2 models for ensemble")
        return

    print(f"\nCreating ensemble with {len(model_paths)} models...")
    ensemble = EnsemblePredictor(model_paths, method=args.method)

    # Run predictions on remaining folds
    results = []
    end_fold = args.end_fold if args.end_fold else n_folds

    print("\n" + "="*70)
    print(" ENSEMBLE PREDICTION")
    print("="*70)

    for fold in range(args.start_fold, end_fold + 1):
        # Calculate test period
        test_start = window_hours + (fold - 1) * test_hours
        test_end = test_start + test_hours

        if test_end > total_hours:
            print(f"\nFold {fold}: Not enough data (need {test_end}, have {total_hours})")
            break

        # Get dates
        test_start_ts = data.iloc[test_start]['periodStartUnix']
        test_end_ts = data.iloc[min(test_end-1, len(data)-1)]['periodStartUnix']
        test_start_date = datetime.fromtimestamp(test_start_ts)
        test_end_date = datetime.fromtimestamp(test_end_ts)

        print(f"\n{'─'*70}")
        print(f"Fold {fold}: {test_start_date.strftime('%Y-%m-%d')} → {test_end_date.strftime('%Y-%m-%d')}")
        print(f"  Hours: {test_start} - {test_end}")

        result = run_ensemble_evaluation(
            ensemble, data, test_start, test_hours, f"Fold {fold}"
        )

        if result is None:
            continue

        print(f"  Return: ${result['net_return']:+,.2f} | "
              f"Fees: ${result['fees']:,.2f} | "
              f"IL: ${result['il']:,.2f} | "
              f"LVR: ${result.get('lvr', 0):,.2f} | "
              f"Gas: ${result['gas']:,.2f} | "
              f"Rebal: {result['rebalances']}")

        fold_result = {
            'fold': fold,
            'test_start': test_start_date.isoformat(),
            'test_end': test_end_date.isoformat(),
            'net_return': result['net_return'],
            'fees': result['fees'],
            'il': result['il'],
            'lvr': result.get('lvr', 0),
            'gas': result['gas'],
            'rebalances': result['rebalances'],
            'reward': result['total_reward']
        }
        results.append(fold_result)

        # Save tracking if requested
        if args.save_tracking:
            tracking_path = output_dir / f"ensemble_fold_{fold:02d}_tracking.csv"
            result['tracking'].to_csv(tracking_path, index=False)

    # Summary
    print("\n" + "="*70)
    print(" ENSEMBLE SUMMARY")
    print("="*70)

    if results:
        returns = [r['net_return'] for r in results]
        fees = [r['fees'] for r in results]
        ils = [r['il'] for r in results]

        print(f"\nFolds evaluated: {len(results)}")
        print(f"Average Return: ${np.mean(returns):+,.2f} ± ${np.std(returns):,.2f}")
        print(f"Total Return: ${np.sum(returns):+,.2f}")
        print(f"Win Rate: {sum(1 for r in returns if r > 0) / len(returns) * 100:.1f}%")
        print(f"Average Fees: ${np.mean(fees):,.2f}")
        print(f"Average IL: ${np.mean(ils):,.2f}")

        # Best/Worst
        best_idx = np.argmax(returns)
        worst_idx = np.argmin(returns)
        print(f"\nBest Fold: {results[best_idx]['fold']} (${returns[best_idx]:+,.2f})")
        print(f"Worst Fold: {results[worst_idx]['fold']} (${returns[worst_idx]:+,.2f})")

    # Save results
    results_path = output_dir / f"ensemble_results_fold{args.start_fold}_to_{end_fold}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # Create summary DataFrame
    if results:
        df_results = pd.DataFrame(results)
        csv_path = output_dir / f"ensemble_results_fold{args.start_fold}_to_{end_fold}.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
