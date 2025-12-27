#!/usr/bin/env python
"""
Compare all models (v8, v9) vs baselines on test period.
"""
import sys
sys.path.insert(0, '/home/zekiya/liquidity/uniswap-v3-simulator')

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO
from uniswap_v3.ml.environment import UniswapV3LPEnv
from uniswap_v3.ml.environment_v9 import UniswapV3LPEnvV9


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


def evaluate_with_v9_env(model_path, data, pool_config, initial_investment=10000, episode_length=720, model_name="Model"):
    """Evaluate any model using v9 environment (tracks HODL properly)."""
    if model_path:
        model = PPO.load(model_path)
    else:
        model = None  # For baseline (always action 0)

    env = UniswapV3LPEnvV9(
        historical_data=data,
        pool_config=pool_config,
        initial_investment=initial_investment,
        episode_length_hours=episode_length,
        reward_weights={
            'in_range_bonus': 0.001,
            'gas_penalty_scale': 1.0,
        }
    )

    obs, _ = env.reset()
    lp_values = []
    hodl_values = []
    actions = []
    rebalance_count = 0
    in_range_count = 0
    fees_collected = 0.0

    for _ in range(episode_length):
        if model:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = 0  # Baseline: no rebalance

        actions.append(action)

        obs, reward, done, truncated, info = env.step(action)
        lp_values.append(env.current_position_value)
        hodl_values.append(env.hodl_value)
        fees_collected += info.get('fees', 0)

        if action > 0:
            rebalance_count += 1
        if info.get('in_range', False):
            in_range_count += 1

        if done or truncated:
            break

    return {
        'name': model_name,
        'final_value': lp_values[-1] if lp_values else initial_investment,
        'hodl_value': hodl_values[-1] if hodl_values else initial_investment,
        'lp_values': lp_values,
        'hodl_values': hodl_values,
        'return_pct': (lp_values[-1] / initial_investment - 1) * 100 if lp_values else 0,
        'hodl_return_pct': (hodl_values[-1] / initial_investment - 1) * 100 if hodl_values else 0,
        'excess_return_pct': ((lp_values[-1] / initial_investment) - (hodl_values[-1] / initial_investment)) * 100 if lp_values else 0,
        'in_range_pct': in_range_count / len(lp_values) * 100 if lp_values else 0,
        'rebalances': rebalance_count,
        'fees': fees_collected,
        'action_dist': pd.Series(actions).value_counts().to_dict() if actions else {},
    }


def main():
    db_path = "/home/zekiya/liquidity/uniswap-v3-simulator/uniswap_v3/data/pool_data.db"
    model_dir = Path("/home/zekiya/liquidity/uniswap-v3-simulator/uniswap_v3/models/rolling_wfe/models")

    # Test period: hours 7500-8220 (30 days after training)
    TEST_START = 7500
    TEST_LENGTH = 720

    print("=" * 70)
    print("Model Comparison: Test Period (Fold 1)")
    print("=" * 70)

    # Load test data
    test_data, pool_config = load_data(db_path, TEST_START, TEST_LENGTH)
    print(f"Test period: {TEST_LENGTH} hours starting at index {TEST_START}")

    results = []

    # 1. Baseline (20% range, no rebalance)
    print("\n[1/3] Evaluating Baseline (20% range, hold)...")
    baseline_result = evaluate_with_v9_env(None, test_data, pool_config, model_name="20% Hold")
    results.append(baseline_result)
    print(f"  LP: ${baseline_result['final_value']:.2f} ({baseline_result['return_pct']:.2f}%)")
    print(f"  HODL: ${baseline_result['hodl_value']:.2f} ({baseline_result['hodl_return_pct']:.2f}%)")
    print(f"  Excess Return: {baseline_result['excess_return_pct']:.2f}%")
    print(f"  In-Range: {baseline_result['in_range_pct']:.1f}%")

    # 2. v8 model
    v8_path = model_dir / "fold_01_v8_il_fix_500k.zip"
    if v8_path.exists():
        print("\n[2/3] Evaluating v8 model (IL Fix)...")
        v8_result = evaluate_with_v9_env(str(v8_path), test_data, pool_config, model_name="v8 (IL Fix)")
        results.append(v8_result)
        print(f"  LP: ${v8_result['final_value']:.2f} ({v8_result['return_pct']:.2f}%)")
        print(f"  HODL: ${v8_result['hodl_value']:.2f} ({v8_result['hodl_return_pct']:.2f}%)")
        print(f"  Excess Return: {v8_result['excess_return_pct']:.2f}%")
        print(f"  In-Range: {v8_result['in_range_pct']:.1f}%, Rebalances: {v8_result['rebalances']}")
        print(f"  Actions: {v8_result['action_dist']}")
    else:
        print(f"\n[2/3] v8 model not found")

    # 3. v9 model
    v9_path = model_dir / "fold_01_v9_excess_return_500k.zip"
    if v9_path.exists():
        print("\n[3/3] Evaluating v9 model (Excess Return)...")
        v9_result = evaluate_with_v9_env(str(v9_path), test_data, pool_config, model_name="v9 (Excess Return)")
        results.append(v9_result)
        print(f"  LP: ${v9_result['final_value']:.2f} ({v9_result['return_pct']:.2f}%)")
        print(f"  HODL: ${v9_result['hodl_value']:.2f} ({v9_result['hodl_return_pct']:.2f}%)")
        print(f"  Excess Return: {v9_result['excess_return_pct']:.2f}%")
        print(f"  In-Range: {v9_result['in_range_pct']:.1f}%, Rebalances: {v9_result['rebalances']}")
        print(f"  Actions: {v9_result['action_dist']}")
    else:
        print(f"\n[3/3] v9 model not found")

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY - Test Period Performance")
    print("=" * 90)
    print(f"{'Model':<22} {'LP Value':>12} {'HODL Value':>12} {'LP Return':>10} {'Excess':>10} {'In-Range':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<22} ${r['final_value']:>10.2f} ${r['hodl_value']:>10.2f} {r['return_pct']:>9.2f}% {r['excess_return_pct']:>9.2f}% {r['in_range_pct']:>9.1f}%")

    # Create comparison visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: LP Value over time
    ax1 = axes[0, 0]
    for r in results:
        ax1.plot(r['lp_values'], label=f"{r['name']} LP", linewidth=1.5)
    # Add HODL reference (same for all)
    ax1.plot(results[0]['hodl_values'], label='HODL', linestyle='--', color='black', alpha=0.7, linewidth=1.5)
    ax1.axhline(y=10000, color='gray', linestyle=':', alpha=0.5, label='Initial')
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Portfolio Value (USD)')
    ax1.set_title('Portfolio Value Over Time (Test Period)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final values comparison
    ax2 = axes[0, 1]
    x = np.arange(len(results))
    width = 0.35
    lp_vals = [r['final_value'] for r in results]
    hodl_vals = [r['hodl_value'] for r in results]
    names = [r['name'] for r in results]

    bars1 = ax2.bar(x - width/2, lp_vals, width, label='LP Value', color='#3498db')
    bars2 = ax2.bar(x + width/2, hodl_vals, width, label='HODL Value', color='#2ecc71')
    ax2.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax2.set_ylabel('Value (USD)')
    ax2.set_title('Final Values: LP vs HODL')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.legend()
    ax2.set_ylim(min(lp_vals + hodl_vals) * 0.95, max(lp_vals + hodl_vals) * 1.05)

    # Plot 3: Excess Return comparison
    ax3 = axes[1, 0]
    excess_returns = [r['excess_return_pct'] for r in results]
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in excess_returns]
    bars = ax3.bar(names, excess_returns, color=colors)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Excess Return (%)')
    ax3.set_title('Excess Return (LP - HODL)')
    for bar, ret in zip(bars, excess_returns):
        ypos = bar.get_height() + 0.2 if ret >= 0 else bar.get_height() - 0.5
        ax3.text(bar.get_x() + bar.get_width()/2, ypos,
                f'{ret:.2f}%', ha='center', va='bottom', fontsize=10)

    # Plot 4: In-range and rebalances
    ax4 = axes[1, 1]
    x = np.arange(len(results))
    width = 0.35
    in_ranges = [r['in_range_pct'] for r in results]
    rebalances = [r['rebalances'] for r in results]

    ax4_twin = ax4.twinx()
    bars1 = ax4.bar(x - width/2, in_ranges, width, label='In-Range %', color='#3498db')
    bars2 = ax4_twin.bar(x + width/2, rebalances, width, label='Rebalances', color='#e74c3c')

    ax4.set_ylabel('In-Range %', color='#3498db')
    ax4_twin.set_ylabel('Rebalances', color='#e74c3c')
    ax4.set_title('In-Range % vs Rebalances')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=15, ha='right')
    ax4.set_ylim(0, 110)
    ax4_twin.set_ylim(0, max(rebalances) * 1.2 if max(rebalances) > 0 else 10)

    # Add legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.suptitle('Model Comparison: Test Period (Fold 1)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = '/home/zekiya/liquidity/uniswap-v3-simulator/model_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
