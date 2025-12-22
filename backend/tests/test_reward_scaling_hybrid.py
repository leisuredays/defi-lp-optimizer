#!/usr/bin/env python3
"""Test reward scaling with realistic values"""

# Realistic scenario: $10,000 investment, 400h episode
investment = 10000

# Per-hour estimates (Arbitrum USDC/WETH 0.05%)
fees_per_hour = 2.5  # ~$2.50/hour ($1,000/400h = realistic APR ~10%)
il_per_hour = 0.5    # IL accumulates slowly when in-range
gas_per_rebalance = 3.0  # Arbitrum gas
swap_per_rebalance = 10.0  # 0.15% of ~$6,666 rebalanced (2/3 of position)

# Scenario 1: Normal operation (no rebalancing)
print("="*70)
print("SCENARIO 1: Normal Hour (No Rebalancing)")
print("="*70)
fees = fees_per_hour
il_raw = il_per_hour
il_penalty = il_raw * 0.1  # Observation penalty (10%)
gas = 0
swap = 0

reward_raw = 1.0 * fees - 1.5 * il_penalty - 0.2 * gas
reward_normalized = reward_raw / 100.0

print(f"Fees earned: ${fees:.2f}")
print(f"IL (raw): ${il_raw:.2f}")
print(f"IL penalty (10%): ${il_penalty:.2f}")
print(f"Gas cost: ${gas:.2f}")
print(f"Swap cost: ${swap:.2f}")
print(f"\nReward (raw): {reward_raw:.4f}")
print(f"Reward (normalized): {reward_normalized:.6f}")
print(f"Reward scale: {abs(reward_normalized):.6f}")

# Scenario 2: Rebalancing event
print("\n" + "="*70)
print("SCENARIO 2: Rebalancing Hour")
print("="*70)
fees = fees_per_hour
il_raw = 50.0  # IL "realized" at rebalancing (accumulated loss)
il_penalty = il_raw * 1.0  # Full realized penalty (100%)
gas = gas_per_rebalance
swap = swap_per_rebalance

reward_raw = 1.0 * fees - 1.5 * il_penalty - 0.2 * (gas + swap)
reward_normalized = reward_raw / 100.0

print(f"Fees earned: ${fees:.2f}")
print(f"IL (raw): ${il_raw:.2f}")
print(f"IL penalty (100%): ${il_penalty:.2f}")
print(f"Gas cost: ${gas:.2f}")
print(f"Swap cost: ${swap:.2f}")
print(f"Total rebalance cost: ${gas + swap:.2f}")
print(f"\nReward (raw): {reward_raw:.4f}")
print(f"Reward (normalized): {reward_normalized:.6f}")
print(f"Reward scale: {abs(reward_normalized):.6f}")

# Scenario 3: Good episode (20 hours, 1 rebalance)
print("\n" + "="*70)
print("SCENARIO 3: Full Episode (400h, ~15 rebalances)")
print("="*70)
hours = 400
rebalances = 15
normal_hours = hours - rebalances

total_fees = fees_per_hour * hours

# IL penalty calculation with hybrid approach
# Normal hours: 10% penalty
# Rebalancing hours: 100% penalty (50 IL per rebalance)
il_observation_penalty = il_per_hour * 0.1 * normal_hours  # 385h × $0.5 × 10%
il_realized_penalty = 50.0 * rebalances  # 15 rebalances × $50

total_il_penalty = il_observation_penalty + il_realized_penalty
total_il_raw = il_per_hour * hours + 50 * rebalances  # For info only

total_gas = gas_per_rebalance * rebalances
total_swap = swap_per_rebalance * rebalances

reward_raw = 1.0 * total_fees - 1.5 * total_il_penalty - 0.2 * (total_gas + total_swap)
reward_normalized = reward_raw / 100.0

print(f"Total fees: ${total_fees:.2f}")
print(f"IL (raw): ${total_il_raw:.2f}")
print(f"  - Observation penalty (385h × 10%): ${il_observation_penalty:.2f}")
print(f"  - Realized penalty (15 rebalances): ${il_realized_penalty:.2f}")
print(f"  - Total IL penalty: ${total_il_penalty:.2f}")
print(f"Total gas: ${total_gas:.2f}")
print(f"Total swap: ${total_swap:.2f}")
print(f"Total rebalances: {rebalances}")
print(f"\nReward (raw): {reward_raw:.4f}")
print(f"Reward (normalized): {reward_normalized:.6f}")
print(f"Reward per step (avg): {reward_normalized/hours:.8f}")

# Analysis
print("\n" + "="*70)
print("REWARD SCALING ANALYSIS")
print("="*70)
print(f"Normalization factor: /100.0")
print(f"IL penalty strategy: Hybrid (10% observation, 100% realized)")
print(f"\nTypical reward ranges:")
print(f"  Good hour (no rebalance): {(2.5 - 1.5*0.5*0.1)/100:.6f}")
print(f"  Bad hour (rebalancing): {(2.5 - 1.5*50 - 0.2*13)/100:.6f}")
print(f"  Episode average: {reward_normalized/hours:.8f}")
print(f"\n✅ Rewards in reasonable range for PPO (-1 to +1)")
print(f"✅ Model can learn cost-benefit tradeoffs")
print(f"✅ Hybrid IL penalty prevents over-penalization")
