"""
Evaluate Single Pool Model

Evaluates trained PPO model on Arbitrum USDC/WETH 0.05% test data.
"""
import sys
from pathlib import Path
import json
import yaml
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from app.ml.environment import UniswapV3LPEnv


def load_config():
    """Load training config"""
    # Determine path relative to script location
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "training_config.yaml"

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_model(model: PPO, test_data: pd.DataFrame,
                   n_episodes: int = 20) -> dict:
    """
    Evaluate model on test data.

    Args:
        model: Trained PPO model
        test_data: Test dataset
        n_episodes: Number of episodes

    Returns:
        Evaluation metrics
    """
    pool_config = {
        'protocol': int(test_data['protocol_id'].iloc[0]),
        'feeTier': int(test_data['fee_tier'].iloc[0]),
        'token0': {'decimals': int(test_data['token0_decimals'].iloc[0])},
        'token1': {'decimals': int(test_data['token1_decimals'].iloc[0])}
    }

    env = UniswapV3LPEnv(
        historical_data=test_data,
        pool_config=pool_config,
        initial_investment=10000,
        episode_length_hours=400,
        reward_weights={'alpha': 1.0, 'beta': 0.8, 'gamma': 0.2}
    )

    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        results.append({
            'reward': episode_reward,
            'fees': info['cumulative_fees'],
            'il': info['cumulative_il'],
            'gas': info['cumulative_gas'],
            'rebalances': info['total_rebalances'],
            'net_return': info['net_return']
        })

    # Aggregate
    rewards = np.array([r['reward'] for r in results])
    fees = np.array([r['fees'] for r in results])
    il = np.array([r['il'] for r in results])
    gas = np.array([r['gas'] for r in results])
    rebalances = np.array([r['rebalances'] for r in results])
    net_returns = np.array([r['net_return'] for r in results])

    return {
        'episodes': n_episodes,
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_fees': float(np.mean(fees)),
        'std_fees': float(np.std(fees)),
        'mean_il': float(np.mean(il)),
        'std_il': float(np.std(il)),
        'mean_gas': float(np.mean(gas)),
        'mean_rebalances': float(np.mean(rebalances)),
        'std_rebalances': float(np.std(rebalances)),
        'mean_net_return': float(np.mean(net_returns)),
        'std_net_return': float(np.std(net_returns)),
        'sharpe_ratio': float(np.mean(rewards) / (np.std(rewards) + 1e-8)),
        'success_rate': float(np.sum(net_returns > 0) / len(net_returns)),
        'max_return': float(np.max(net_returns)),
        'min_return': float(np.min(net_returns))
    }


def main():
    """Main evaluation"""

    print("="*70)
    print(" " * 15 + "MODEL EVALUATION")
    print(" " * 10 + "Arbitrum USDC/WETH 0.05%")
    print("="*70)

    # Load config
    print("\n[1/4] Loading configuration...")
    print("-"*70)

    config = load_config()
    model_name = config['output']['model_name']

    # Try both relative paths
    model_path = Path(config['output']['model_dir']) / f"{model_name}.zip"
    if not model_path.exists():
        model_path = Path("backend") / config['output']['model_dir'] / f"{model_name}.zip"

    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        print(f"  Train first: python scripts/train_ppo_single_pool.py")
        return

    print(f"✓ Config loaded")
    print(f"  Model: {model_name}")
    print(f"  Path: {model_path}")

    # Load model
    print("\n[2/4] Loading trained model...")
    print("-"*70)

    try:
        model = PPO.load(str(model_path))
        print(f"✓ Model loaded")
    except Exception as e:
        print(f"✗ Error: {e}")
        return

    # Load test data
    print("\n[3/4] Loading test data...")
    print("-"*70)

    pool_file = Path(config['data']['pool_file'])
    df = pd.read_parquet(pool_file)
    df = df.sort_values('periodStartUnix').reset_index(drop=True)

    # Split
    n = len(df)
    train_end = int(n * config['data']['train_split'])
    val_end = int(n * (config['data']['train_split'] + config['data']['val_split']))
    test_df = df.iloc[val_end:].copy()

    print(f"✓ Test data loaded")
    print(f"  Hours: {len(test_df):,} ({len(test_df)/24:.0f} days)")
    print(f"  Pool: {test_df['token0_symbol'].iloc[0]}/{test_df['token1_symbol'].iloc[0]}")

    # Evaluate
    print("\n[4/4] Evaluating model (20 episodes)...")
    print("-"*70)

    try:
        metrics = evaluate_model(model, test_df, n_episodes=20)

        print(f"\n✓ Evaluation complete")
        print(f"\nResults:")
        print(f"  Mean Net Return: ${metrics['mean_net_return']:.2f} ± ${metrics['std_net_return']:.2f}")
        print(f"  Best: ${metrics['max_return']:.2f}, Worst: ${metrics['min_return']:.2f}")
        print(f"\n  Mean Fees: ${metrics['mean_fees']:.2f}")
        print(f"  Mean IL: ${metrics['mean_il']:.2f}")
        print(f"  Mean Gas: ${metrics['mean_gas']:.2f}")
        print(f"\n  Mean Rebalances: {metrics['mean_rebalances']:.1f} per episode (400 hours)")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Success Rate: {metrics['success_rate']*100:.1f}% (profitable episodes)")

        # APR estimation (400 hours = 16.67 days)
        apr = (metrics['mean_net_return'] / 10000) * (365 / (400/24))
        print(f"\n  Estimated APR: {apr*100:.2f}%")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save results
    print("\n[Saving] Results...")
    print("-"*70)

    output_path = Path("backend/data/evaluation") / f"{model_name}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'pool': 'Arbitrum USDC/WETH 0.05%',
        'model': model_name,
        'test_hours': len(test_df),
        'metrics': metrics
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved: {output_path}")

    # Summary
    print("\n" + "="*70)
    print(" " * 25 + "SUMMARY")
    print("="*70)
    print(f"\nPool: Arbitrum USDC/WETH 0.05%")
    print(f"Net Return: ${metrics['mean_net_return']:.2f} per episode (400 hours, on $10k)")
    print(f"APR: {apr*100:.2f}%")
    print(f"Success Rate: {metrics['success_rate']*100:.0f}%")

    if metrics['mean_net_return'] > 0:
        print(f"\n✅ Model is profitable!")
    else:
        print(f"\n⚠ Model needs improvement")

    print("="*70)


if __name__ == "__main__":
    main()
