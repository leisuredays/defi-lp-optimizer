#!/usr/bin/env python3
"""Test environment for debugging NaN issues"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from app.ml.environment import UniswapV3LPEnv

# Load data
data_path = "data/training/arbitrum_usdc_weth_005_data.parquet"
df = pd.read_parquet(data_path)

# Create environment
env = UniswapV3LPEnv(
    historical_data=df,
    pool_config={
        'pool_id': 'test',
        'token0': {'symbol': 'USDC', 'decimals': 6},
        'token1': {'symbol': 'WETH', 'decimals': 18},
        'feeTier': 500,
        'protocol': 'arbitrum'
    },
    episode_length_hours=400,
    initial_investment=10000
)

print("=" * 70)
print("ENVIRONMENT INITIALIZATION TEST")
print("=" * 70)

# Reset environment
obs, info = env.reset()
print(f"\n1. Reset successful")
print(f"   Observation shape: {obs.shape}")
print(f"   Observation dtype: {obs.dtype}")
print(f"   Has NaN: {np.isnan(obs).any()}")
print(f"   Has Inf: {np.isinf(obs).any()}")
print(f"   Min: {obs.min():.4f}, Max: {obs.max():.4f}")
print(f"   Sample values: {obs[:5]}")

# Try a step with zero action (3D)
print(f"\n2. Testing zero action (no rebalance, medium range)...")
action = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # Don't rebalance
obs2, reward, terminated, truncated, info = env.step(action)
print(f"   Step successful")
print(f"   Rebalanced: {info.get('rebalanced', False)}")
print(f"   Observation has NaN: {np.isnan(obs2).any()}")
print(f"   Observation has Inf: {np.isinf(obs2).any()}")
print(f"   Reward: {reward:.4f}")
print(f"   Sample values: {obs2[:5]}")

# Try rebalancing action (3D)
print(f"\n3. Testing rebalancing action (yes rebalance, narrow-wide range)...")
action = np.array([1.0, -0.5, 0.5], dtype=np.float32)  # Rebalance with asymmetric range
obs3, reward, terminated, truncated, info = env.step(action)
print(f"   Step successful")
print(f"   Rebalanced: {info.get('rebalanced', False)}")
print(f"   Observation has NaN: {np.isnan(obs3).any()}")
print(f"   Observation has Inf: {np.isinf(obs3).any()}")
print(f"   Reward: {reward:.4f}")

print(f"\n4. Environment state:")
print(f"   cumulative_fees: {env.cumulative_fees:.4f}")
print(f"   cumulative_il: {env.cumulative_il:.4f}")
print(f"   cumulative_gas: {env.cumulative_gas:.4f}")
print(f"   total_rebalances: {env.total_rebalances}")
print(f"   time_since_rebalance: {env.time_since_last_rebalance}h")

# Test after 24 hours
print(f"\n5. Testing rebalancing after 24h cooldown...")
for i in range(22):  # Skip to hour 24
    action = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # Don't rebalance
    obs, reward, terminated, truncated, info = env.step(action)

print(f"   Time since rebalance: {env.time_since_last_rebalance}h")
# Now try to rebalance with significant change
action = np.array([1.0, -0.8, 0.8], dtype=np.float32)  # Rebalance with wide range
obs, reward, terminated, truncated, info = env.step(action)
print(f"   Tried to rebalance: action[0] = {action[0]}")
print(f"   Actually rebalanced: {info['rebalanced']}")
print(f"   Total rebalances: {info['total_rebalances']}")

print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
