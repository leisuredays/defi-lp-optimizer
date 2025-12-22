"""
Model Evaluation Script

Evaluates trained PPO model on test pools and compares with baseline strategies.
"""
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from app.ml.environment import UniswapV3LPEnv


def evaluate_on_pool(model: PPO, pool_data: pd.DataFrame,
                     pool_config: dict, n_episodes: int = 10,
                     deterministic: bool = True) -> dict:
    """
    Evaluate model on a single pool.

    Args:
        model: Trained PPO model
        pool_data: Pool historical data
        pool_config: Pool configuration dict
        n_episodes: Number of episodes to run
        deterministic: Use deterministic actions

    Returns:
        Dictionary with evaluation metrics
    """
    env = UniswapV3LPEnv(
        historical_data=pool_data,
        pool_config=pool_config,
        initial_investment=10000,
        episode_length_hours=720,
        reward_weights={'alpha': 1.0, 'beta': 0.8, 'gamma': 0.2}
    )

    episode_returns = []
    episode_metrics = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1

        # Store episode results
        episode_returns.append(episode_reward)
        episode_metrics.append({
            'fees': info['cumulative_fees'],
            'il': info['cumulative_il'],
            'gas': info['cumulative_gas'],
            'rebalances': info['total_rebalances'],
            'net_return': info['net_return'],
            'steps': step_count
        })

    # Calculate aggregate statistics
    returns_arr = np.array(episode_returns)
    fees_arr = np.array([m['fees'] for m in episode_metrics])
    il_arr = np.array([m['il'] for m in episode_metrics])
    gas_arr = np.array([m['gas'] for m in episode_metrics])
    rebalances_arr = np.array([m['rebalances'] for m in episode_metrics])
    net_return_arr = np.array([m['net_return'] for m in episode_metrics])

    return {
        'mean_return': float(np.mean(returns_arr)),
        'std_return': float(np.std(returns_arr)),
        'mean_fees': float(np.mean(fees_arr)),
        'std_fees': float(np.std(fees_arr)),
        'mean_il': float(np.mean(il_arr)),
        'std_il': float(np.std(il_arr)),
        'mean_gas': float(np.mean(gas_arr)),
        'std_gas': float(np.std(gas_arr)),
        'mean_rebalances': float(np.mean(rebalances_arr)),
        'std_rebalances': float(np.std(rebalances_arr)),
        'mean_net_return': float(np.mean(net_return_arr)),
        'std_net_return': float(np.std(net_return_arr)),
        'sharpe_ratio': float(np.mean(returns_arr) / (np.std(returns_arr) + 1e-8)),
        'success_rate': float(np.sum(net_return_arr > 0) / len(net_return_arr))  # % episodes with profit
    }


def evaluate_baseline(pool_data: pd.DataFrame, pool_config: dict,
                     n_episodes: int = 10) -> dict:
    """
    Evaluate a simple baseline strategy (fixed ±1 std dev range).

    Args:
        pool_data: Pool historical data
        pool_config: Pool configuration dict
        n_episodes: Number of episodes to run

    Returns:
        Dictionary with evaluation metrics
    """
    # TODO: Implement baseline evaluation
    # For now, return dummy values
    return {
        'mean_return': 0.0,
        'std_return': 0.0,
        'mean_fees': 0.0,
        'std_fees': 0.0,
        'mean_il': 0.0,
        'std_il': 0.0,
        'mean_gas': 0.0,
        'std_gas': 0.0,
        'mean_rebalances': 0.0,
        'std_rebalances': 0.0,
        'mean_net_return': 0.0,
        'std_net_return': 0.0,
        'sharpe_ratio': 0.0,
        'success_rate': 0.0
    }


def main():
    """Main evaluation pipeline"""

    print("="*70)
    print(" " * 20 + "MODEL EVALUATION")
    print("="*70)

    # [1/4] Load trained model
    print("\n[1/4] Loading trained model...")
    print("-"*70)

    model_path = "backend/models/ppo_uniswap_v3_v1.0.0.zip"

    if not Path(model_path).exists():
        print(f"✗ Model not found: {model_path}")
        print(f"  Train the model first: python scripts/train_ppo.py")
        return

    try:
        model = PPO.load(model_path)
        print(f"✓ Model loaded from {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # [2/4] Load test pools
    print("\n[2/4] Loading test pools...")
    print("-"*70)

    data_dir = Path("backend/data/training")
    pool_files = sorted(data_dir.glob("pool_*_data.parquet"))

    if len(pool_files) < 3:
        print(f"✗ Insufficient pool data (need at least 3 pools)")
        return

    # Use last 2 pools for testing (10% split)
    test_pools = pool_files[-2:]

    print(f"✓ Found {len(test_pools)} test pools:")
    for pool_file in test_pools:
        print(f"  - {pool_file.name}")

    # [3/4] Evaluate on test pools
    print("\n[3/4] Evaluating model on test pools...")
    print("-"*70)

    results = {
        'ppo': [],
        'baseline': [],
        'pool_names': []
    }

    for i, pool_file in enumerate(test_pools):
        print(f"\n[Test Pool {i+1}/{len(test_pools)}] {pool_file.stem}")

        try:
            # Load pool data
            pool_data = pd.read_parquet(pool_file)

            pool_config = {
                'protocol': int(pool_data['protocol_id'].iloc[0]),
                'feeTier': int(pool_data['fee_tier'].iloc[0]),
                'token0': {'decimals': int(pool_data['token0_decimals'].iloc[0])},
                'token1': {'decimals': int(pool_data['token1_decimals'].iloc[0])}
            }

            pool_name = f"{pool_data['token0_symbol'].iloc[0]}/{pool_data['token1_symbol'].iloc[0]}"
            results['pool_names'].append(pool_name)

            print(f"  Pool: {pool_name}")
            print(f"  Protocol: {pool_data['protocol_id'].iloc[0]}")
            print(f"  Fee tier: {pool_data['fee_tier'].iloc[0]/10000:.2f}%")

            # Evaluate PPO
            print(f"  Evaluating PPO (10 episodes)...")
            ppo_results = evaluate_on_pool(model, pool_data, pool_config, n_episodes=10)
            results['ppo'].append(ppo_results)

            print(f"    ✓ Mean net return: ${ppo_results['mean_net_return']:.2f} "
                  f"± ${ppo_results['std_net_return']:.2f}")
            print(f"    ✓ Mean fees: ${ppo_results['mean_fees']:.2f}")
            print(f"    ✓ Mean IL: ${ppo_results['mean_il']:.2f}")
            print(f"    ✓ Mean rebalances: {ppo_results['mean_rebalances']:.1f}")
            print(f"    ✓ Sharpe ratio: {ppo_results['sharpe_ratio']:.3f}")

            # Evaluate baseline (placeholder)
            # baseline_results = evaluate_baseline(pool_data, pool_config, n_episodes=10)
            # results['baseline'].append(baseline_results)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # [4/4] Save and display results
    print("\n[4/4] Saving results...")
    print("-"*70)

    # Save results
    output_path = Path("backend/data/evaluation/test_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {output_path}")

    # Display summary
    print("\n" + "="*70)
    print(" " * 22 + "EVALUATION SUMMARY")
    print("="*70)

    if len(results['ppo']) > 0:
        # Aggregate across all test pools
        all_returns = [r['mean_net_return'] for r in results['ppo']]
        all_fees = [r['mean_fees'] for r in results['ppo']]
        all_il = [r['mean_il'] for r in results['ppo']]
        all_gas = [r['mean_gas'] for r in results['ppo']]
        all_rebalances = [r['mean_rebalances'] for r in results['ppo']]
        all_sharpe = [r['sharpe_ratio'] for r in results['ppo']]
        all_success = [r['success_rate'] for r in results['ppo']]

        print(f"\nAggregate Performance (PPO):")
        print(f"  Mean Net Return: ${np.mean(all_returns):.2f} ± ${np.std(all_returns):.2f}")
        print(f"  Mean Fees: ${np.mean(all_fees):.2f}")
        print(f"  Mean IL: ${np.mean(all_il):.2f}")
        print(f"  Mean Gas: ${np.mean(all_gas):.2f}")
        print(f"  Mean Rebalances: {np.mean(all_rebalances):.1f} per episode")
        print(f"  Mean Sharpe Ratio: {np.mean(all_sharpe):.3f}")
        print(f"  Success Rate: {np.mean(all_success)*100:.1f}% (profitable episodes)")

        # Per-pool breakdown
        print(f"\nPer-Pool Breakdown:")
        for i, (pool_name, ppo_res) in enumerate(zip(results['pool_names'], results['ppo'])):
            print(f"\n  Pool {i+1}: {pool_name}")
            print(f"    Net Return: ${ppo_res['mean_net_return']:.2f}")
            print(f"    Fees: ${ppo_res['mean_fees']:.2f}, "
                  f"IL: ${ppo_res['mean_il']:.2f}, "
                  f"Gas: ${ppo_res['mean_gas']:.2f}")
            print(f"    Rebalances: {ppo_res['mean_rebalances']:.1f}, "
                  f"Sharpe: {ppo_res['sharpe_ratio']:.3f}")

        # APR estimation
        # Assuming 720 hours = 30 days per episode
        mean_return_pct = np.mean(all_returns) / 10000  # % of $10k investment
        apr = mean_return_pct * (365 / 30)  # Annualized
        print(f"\nEstimated APR: {apr*100:.2f}%")

    else:
        print("\n✗ No successful evaluations")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
