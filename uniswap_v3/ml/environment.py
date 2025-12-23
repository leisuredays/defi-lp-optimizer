"""
Uniswap V3 LP Position Optimization - Gymnasium Environment

RL environment for learning optimal liquidity range strategies.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import math

# uniswap_v3.math에서 저수준 함수 import
from ..math import (
    price_to_tick,
    tick_to_price,
    split_tokens as optimal_token_split,
)

# uniswap_v3.ml.adapter에서 고수준 래퍼 import
from .adapter import (
    snap_price as round_to_nearest_tick,
    get_tick_from_price,
    calc_liquidity as calculate_liquidity,
    get_token_amounts,
    calc_il as calculate_il_whitepaper,
    calc_fees as calculate_fees_whitepaper,
    active_ratio as active_liquidity_for_candle,
    action_to_range,
    tokens_for_strategy,
    liquidity_for_strategy,
)


class UniswapV3LPEnv(gym.Env):
    """
    Reinforcement Learning Environment for Uniswap V3 LP Position Optimization

    State Space (28 dimensions):
        - Market Features (10): price, volatility, momentum, TVL, volume, liquidity_depth, fee_tier, time features
        - Position Features (10): ticks, width, distances, in_range, time_since_rebalance, cumulative metrics
        - Forward-Looking (8): predictions, liquidity concentration, growth rates

    Action Space: Continuous [-500, 500] for min/max tick deltas

    Reward: α*fees - β*IL - γ*gas_costs + δ*in_range_bonus - ε*rebalancing_penalty

    IL/Fee Calculation: Whitepaper formulas (Section 6.3, 6.4.1)
        - IL: Token amount-based (L = Investment / (2√P₀ - √Pa - P₀/√Pb))
        - Fees: feeGrowthGlobal delta × L × active_ratio / Q128
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, historical_data: pd.DataFrame, pool_config: Dict[str, Any],
                 initial_investment: float = 10000,
                 episode_length_hours: int = 720,  # 30 days
                 reward_weights: Optional[Dict[str, float]] = None,
                 obs_dim: int = 28,
                 debug: bool = False):
        """
        Initialize the Uniswap V3 LP optimization environment.

        Args:
            historical_data: DataFrame with columns: periodStartUnix, close, high, low, liquidity,
                           feeGrowthGlobal0X128, feeGrowthGlobal1X128, pool (nested dict)
            pool_config: Pool configuration with token0, token1, feeTier
            initial_investment: Starting capital (USD)
            episode_length_hours: Episode duration in hours (default: 30 days = 720 hours)
            reward_weights: Dict with α, β, γ, δ, ε for reward function tuning
            obs_dim: Observation dimension (24 or 28). Use 24 for models trained before v3.
            debug: Enable debug logging for rebalancing decisions
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.debug = debug

        # Environment configuration
        self.historical_data = historical_data
        self.pool_config = pool_config
        self.initial_investment = initial_investment
        self.episode_length = episode_length_hours

        # Reward function weights
        # reward = α*fees - β*IL - γ*gas
        # Equal weights for pure profit maximization (Fees - IL - Gas)
        self.reward_weights = reward_weights or {
            'alpha': 1.0,    # Fees coefficient
            'beta': 1.0,     # IL coefficient (equal weight)
            'gamma': 1.0     # Gas costs coefficient (equal weight)
        }

        # Protocol ID to name mapping
        self.protocol_names = {
            0: 'ethereum',
            1: 'optimism',
            2: 'arbitrum',
            3: 'polygon',
            4: 'perpetual',
            5: 'celo'
        }

        # Gas cost estimates by protocol (in USD)
        self.gas_costs = {
            'ethereum': 100,
            'polygon': 2,
            'arbitrum': 3,
            'optimism': 3,
            'celo': 1
        }

        # Get protocol (handle both int ID and string name)
        protocol_raw = pool_config.get('protocol', 'ethereum')
        if isinstance(protocol_raw, int):
            self.protocol = self.protocol_names.get(protocol_raw, 'ethereum')
        else:
            self.protocol = protocol_raw

        # Action space: [should_rebalance, min_range_factor, max_range_factor]
        # 3D action space - model learns WHEN and WHERE to rebalance
        # action[0]: Rebalancing decision (-1=no, +1=yes)
        # action[1]: Lower range factor (-1=narrow, +1=wide)
        # action[2]: Upper range factor (-1=narrow, +1=wide)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space: configurable dimensions (24 or 28) for model compatibility
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.episode_start_idx = 0
        self.position_min_price = 0.0
        self.position_max_price = 0.0
        self.liquidity = 0.0
        self.cumulative_fees = 0.0
        self.cumulative_il = 0.0  # Sum of IL from each position (accumulated when rebalancing)
        self.unrealized_il = 0.0  # Current unrealized IL (for display, not accumulated)
        self.cumulative_gas = 0.0
        self.time_since_last_rebalance = 0
        self.total_rebalances = 0
        self.initial_price = 0.0
        self.initial_token_amounts = [0.0, 0.0]  # Current position's mint tokens (updates at each rebalancing)
        self.current_position_value = 0.0  # Actual position value (decreases with IL/gas, increases with fees)
        self.mint_price = 0.0  # Price when current position was minted (for IL calculation)

        # Precompute features for efficiency
        self._precompute_features()

    def _precompute_features(self):
        """Precompute volatility, momentum, and other features from historical data"""
        df = self.historical_data.copy()

        # Price features
        df['returns'] = df['close'].pct_change()
        df['volatility_24h'] = df['returns'].rolling(24).std()
        df['momentum_1h'] = df['close'].pct_change(1)
        df['momentum_24h'] = df['close'].pct_change(24)

        # Volume/TVL features (if available)
        if 'volume' in df.columns:
            df['volume_24h'] = df['volume'].rolling(24).sum()
        else:
            df['volume_24h'] = 0

        # Time features (cyclical encoding)
        df['hour'] = pd.to_datetime(df['periodStartUnix'], unit='s').dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week'] = pd.to_datetime(df['periodStartUnix'], unit='s').dt.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Fill NaN values (backward fill then forward fill, then 0)
        df = df.bfill().ffill().fillna(0)

        self.features_df = df

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset environment to random episode start point.

        Returns:
            Initial observation (28-dim state vector)
        """
        super().reset(seed=seed)

        # Select episode start (ensuring enough data for full episode)
        # Use features_df length since that's what _get_observation uses
        required_length = self.episode_length + 2  # Extra buffer for next_data
        if len(self.features_df) < required_length:
            raise ValueError(f"Not enough data: {len(self.features_df)} rows, need {required_length}")

        # For backtest/evaluation: always start from beginning (deterministic)
        # For training: random start for data augmentation
        max_start = len(self.features_df) - required_length
        if max_start > 0:
            self.episode_start_idx = np.random.randint(0, max_start)
        else:
            self.episode_start_idx = 0  # Start from beginning if data is tight
        self.current_step = 0

        # Get initial market data
        current_data = self.features_df.iloc[self.episode_start_idx]
        self.initial_price = current_data['close']
        volatility = current_data['volatility_24h']

        # Initialize with default ranges (will be replaced by AI prediction in backtest)
        # These are only used during training when no initial action is provided
        range_pct = max(volatility / self.initial_price * 2, 0.05)  # At least 5%
        range_pct = min(range_pct, 0.20)  # At most 20%

        self.position_min_price = self.initial_price * (1 - range_pct)
        self.position_max_price = self.initial_price * (1 + range_pct)
        self._initial_range_set = False  # Flag to track if AI range was set

        # Ensure min_price is always positive (important for log calculations in tick conversion)
        if self.position_min_price <= 0:
            # Fallback to 10% below current price if calculation results in negative/zero
            self.position_min_price = self.initial_price * 0.9

        # Ensure max_price is reasonable
        if self.position_max_price <= self.position_min_price:
            self.position_max_price = self.initial_price * 1.1

        # Round to valid ticks
        self.position_min_price = round_to_nearest_tick(
            self.position_min_price,
            self.pool_config['feeTier'],
            self.pool_config['token0']['decimals'],
            self.pool_config['token1']['decimals']
        )
        self.position_max_price = round_to_nearest_tick(
            self.position_max_price,
            self.pool_config['feeTier'],
            self.pool_config['token0']['decimals'],
            self.pool_config['token1']['decimals']
        )

        # Calculate initial liquidity
        tokens = self._calculate_tokens_for_range(
            self.initial_investment,
            self.initial_price,
            self.position_min_price,
            self.position_max_price
        )
        self.liquidity = liquidity_for_strategy(
            self.initial_price,
            self.position_min_price,
            self.position_max_price,
            tokens[0],
            tokens[1],
            self.pool_config['token0']['decimals'],
            self.pool_config['token1']['decimals']
        )
        self.initial_token_amounts = tokens  # Mint tokens for this position
        self.mint_price = self.initial_price  # Price when position was minted (for IL calculation)

        # Reset episode trackers
        self.cumulative_fees = 0.0
        self.cumulative_il = 0.0  # Sum of IL from all positions
        self.unrealized_il = 0.0  # Current unrealized IL
        self.cumulative_gas = 0.0
        self.time_since_last_rebalance = 0
        self.total_rebalances = 0
        self.current_position_value = self.initial_investment  # Start with full investment

        # Gymnasium requires reset() to return (observation, info)
        info = {
            'episode_start_idx': self.episode_start_idx,
            'initial_price': self.initial_price,
            'position_min_price': self.position_min_price,
            'position_max_price': self.position_max_price
        }

        return self._get_observation(), info

    def set_initial_range_from_action(self, action: np.ndarray) -> None:
        """
        Set initial LP range based on AI model action (for backtest evaluation).

        This method should be called immediately after reset() in backtest scenarios
        to use AI-predicted ranges instead of the default formula.

        Args:
            action: [rebalance_confidence, min_multiplier, max_multiplier] from model
        """
        # action_to_range is already imported from uniswap_v3_adapter at top of file

        current_data = self.features_df.iloc[self.episode_start_idx]
        current_price_raw = current_data['close']
        volatility_pct = current_data['volatility_24h']  # This is std dev of returns (percentage)

        # CRITICAL: The Graph returns prices in token1/token0 format
        # For USDC/WETH pool: token0=USDC (6 decimals), token1=WETH (18 decimals)
        # price = 2747 means 2747 USDC per WETH (already human-readable!)
        # NO INVERSION NEEDED - use price as-is
        current_price = current_price_raw

        # volatility_24h is std dev of percentage returns (dimensionless)
        # Convert to absolute price volatility by multiplying by price
        volatility = volatility_pct * current_price if volatility_pct > 0 else current_price * 0.01

        # Use shared action_to_range logic with human-readable prices
        self.position_min_price, self.position_max_price = action_to_range(
            action,
            current_price,
            volatility,
            self.pool_config['feeTier'],
            self.pool_config['token0']['decimals'],
            self.pool_config['token1']['decimals']
        )

        # action_to_range returns range in human-readable format (token1/token0)
        # Keep it as-is since environment uses token1/token0 format
        # No conversion needed!

        # Update initial_price to match
        self.initial_price = current_price_raw

        # Recalculate initial liquidity with new ranges
        tokens = self._calculate_tokens_for_range(
            self.initial_investment,
            self.initial_price,
            self.position_min_price,
            self.position_max_price
        )
        # liquidity_for_strategy is already imported from uniswap_v3_adapter at top of file
        self.liquidity = liquidity_for_strategy(
            self.initial_price,
            self.position_min_price,
            self.position_max_price,
            tokens[0],
            tokens[1],
            self.pool_config['token0']['decimals'],
            self.pool_config['token1']['decimals']
        )
        self.initial_token_amounts = tokens  # Mint tokens for this position
        self._initial_range_set = True

        if self.debug:
            print(f"[Env] Initial range set by AI:")
            print(f"  current_price: ${current_price:.2f}")
            print(f"  AI range: [${self.position_min_price:.2f}, ${self.position_max_price:.2f}]")
            print(f"  Initial liquidity: {self.liquidity:.2e}")
            print(f"  Initial tokens: token0={tokens[0]:.6f}, token1={tokens[1]:.6f}")
            print(f"  Decimals: token0={self.pool_config['token0']['decimals']}, token1={self.pool_config['token1']['decimals']}")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action (adjust position ranges) and simulate 1 hour of trading.

        Args:
            action: [min_tick_delta, max_tick_delta] in range [-500, 500]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Get current market data
        current_idx = self.episode_start_idx + self.current_step
        current_data = self.features_df.iloc[current_idx]
        next_data = self.features_df.iloc[current_idx + 1] if current_idx + 1 < len(self.features_df) else current_data

        # CRITICAL: Calculate IL BEFORE applying action (using OLD range)
        prev_min = self.position_min_price
        prev_max = self.position_max_price
        prev_liquidity = self.liquidity

        # Calculate IL with current position (before rebalancing)
        il_raw = self._calculate_il(current_data['close'], prev_min, prev_max, prev_liquidity)

        # Now apply action to adjust position ranges
        did_rebalance = self._apply_action(action, debug=self.debug)

        # Simulate trading for 1 hour
        fees_earned = self._calculate_fees(current_data, next_data)
        gas_cost = self.gas_costs[self.protocol] if did_rebalance else 0.0

        # Update unrealized IL (current snapshot, not accumulated)
        self.unrealized_il = il_raw

        # IL Logic:
        # - Unrealized IL: Current IL that would be incurred if we rebalanced now
        # - Realized IL: IL that was actually incurred when rebalancing (locked in)
        #
        # For reward calculation:
        # - Full IL penalty when rebalancing (realizes the IL)
        # - Small observation penalty (10%) otherwise (awareness for model)
        if did_rebalance:
            il_incurred = il_raw  # Full IL penalty at rebalancing
            self.cumulative_il += il_raw  # Add to realized IL only when rebalancing
        else:
            il_incurred = il_raw * 0.1  # Small observation penalty for reward only

        # Update cumulative metrics
        self.cumulative_fees += fees_earned
        # NOTE: cumulative_il is now only updated when rebalancing (above)
        self.cumulative_gas += gas_cost

        # Update current position value (reflects actual portfolio value)
        # This happens every step: fees increase value, but IL/gas only realized on rebalance
        if did_rebalance:
            # On rebalance: apply IL and gas cost to position value
            self.current_position_value = self.current_position_value - il_raw - gas_cost + fees_earned
        else:
            # No rebalance: only fees affect position value (IL is unrealized)
            self.current_position_value = self.current_position_value + fees_earned

        # Ensure position value doesn't go negative
        self.current_position_value = max(0.0, self.current_position_value)

        if did_rebalance:
            self.time_since_last_rebalance = 0
            self.total_rebalances += 1

            # CRITICAL: Reset IL tracking after rebalancing
            # When rebalancing, IL is "realized" and we start fresh with new position
            current_price = current_data['close']

            # Step 1: Calculate current position tokens using ACTUAL position value
            # (Not initial_investment - use current_position_value which reflects losses)
            old_tokens = self._calculate_tokens_for_range(
                self.current_position_value,  # Use ACTUAL current value
                current_price,
                prev_min,  # Old ranges
                prev_max
            )

            # Step 2: Calculate total value (should match current_position_value)
            # Price is already in token1/token0 format (USDC/WETH)
            price_human = current_price
            # token0=USDC (already USD), token1=WETH
            position_value = old_tokens[0] + (old_tokens[1] * price_human)

            # Step 3: Calculate required tokens for new range
            # This uses Uniswap V3 math to get proper token ratio
            new_tokens_needed = self._calculate_tokens_for_range(
                position_value,
                current_price,
                self.position_min_price,  # New ranges
                self.position_max_price
            )

            # Step 4: Calculate SWAP needed to reach required ratio
            swap_cost = self._calculate_swap_cost(
                old_tokens,
                new_tokens_needed,
                current_price
            )

            # Add swap cost to gas (already tracked in cumulative_gas above)
            # Note: swap_cost is additional cost on top of base gas

            # Recalculate liquidity for new position
            self.liquidity = liquidity_for_strategy(
                current_price,
                self.position_min_price,  # New ranges (already updated in _apply_action)
                self.position_max_price,
                new_tokens_needed[0],
                new_tokens_needed[1],
                self.pool_config['token0']['decimals'],
                self.pool_config['token1']['decimals']
            )

            # CRITICAL: Update initial token amounts to new position (new mint baseline)
            # This resets the HODL baseline for the next position's IL calculation
            self.initial_token_amounts = new_tokens_needed
            self.mint_price = current_price  # New mint price for IL calculation
        else:
            self.time_since_last_rebalance += 1

        # Calculate reward
        reward = self._calculate_reward(fees_earned, il_incurred, gas_cost, did_rebalance, current_data)

        # Check episode termination
        self.current_step += 1

        # Gymnasium uses separate terminated/truncated flags
        position_failed = self._is_position_failed(current_data['close'])
        time_limit_reached = self.current_step >= self.episode_length

        terminated = position_failed  # Episode ended naturally (failure)
        truncated = time_limit_reached and not position_failed  # Episode cut short by time limit

        # Info dict
        info = {
            'fees': fees_earned,
            'il': il_incurred,
            'realized_il': self.cumulative_il,  # Total IL realized through rebalancing
            'unrealized_il': self.unrealized_il,  # Current unrealized IL
            'gas': gas_cost,
            'rebalanced': did_rebalance,
            'total_rebalances': self.total_rebalances,
            'cumulative_fees': self.cumulative_fees,
            'cumulative_il': self.cumulative_il,  # Keep for backward compat (now = realized only)
            'cumulative_gas': self.cumulative_gas,
            'net_return': self.cumulative_fees - self.cumulative_il - self.cumulative_gas,
            'current_position_value': self.current_position_value  # Actual portfolio value
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _apply_action(self, action: np.ndarray, debug: bool = False) -> bool:
        """
        Apply action to adjust position ranges using VOLATILITY-BASED scaling.

        **3D Action Space** - Model learns WHEN and WHERE to rebalance:
        - action[0]: should_rebalance (-1=no, +1=yes)
        - action[1]: min_range_factor (-1=narrow 0.5σ, +1=wide 5.0σ)
        - action[2]: max_range_factor (-1=narrow 0.5σ, +1=wide 5.0σ)

        Rebalancing decision flow:
        1. Emergency: Out of range → FORCE rebalance (safety override)
        2. Cooldown: <24h since last → BLOCK rebalance (cost control)
        3. Model: action[0] > 0 → Allow if change ≥2%

        Args:
            action: [should_rebalance, min_range_factor, max_range_factor]
            debug: If True, print debug information

        Returns:
            True if position was rebalanced
        """
        # Calculate current volatility (recent price std dev)
        current_idx = self.episode_start_idx + self.current_step

        # Use recent 24-hour window for volatility
        lookback = min(24, current_idx)
        if lookback < 2:
            lookback = 2

        recent_prices = self.features_df.iloc[current_idx-lookback:current_idx+1]['close'].values
        volatility = np.std(recent_prices)
        current_price = self.features_df.iloc[current_idx]['close']

        # Handle edge cases
        if volatility <= 0 or np.isnan(volatility):
            volatility = current_price * 0.05  # Default to 5% volatility
        if current_price <= 0 or np.isnan(current_price):
            return False  # Invalid price

        # Extract rebalancing decision and range factors from 3D action
        should_rebalance_raw = action[0]  # -1 (no) to +1 (yes)
        min_range_factor = action[1]      # -1 (narrow) to +1 (wide)
        max_range_factor = action[2]      # -1 (narrow) to +1 (wide)

        # Map range factors [-1, 1] to volatility multiplier [0.5, 5.0]
        # factor = -1 → multiplier = 0.5σ (narrow)
        # factor =  0 → multiplier = 2.75σ (medium)
        # factor = +1 → multiplier = 5.0σ (wide)
        min_multiplier = 0.5 + (min_range_factor + 1) / 2 * 4.5
        max_multiplier = 0.5 + (max_range_factor + 1) / 2 * 4.5

        # Calculate new ranges
        new_min = current_price - (volatility * min_multiplier)
        new_max = current_price + (volatility * max_multiplier)

        # Safety bounds
        if new_min <= 0:
            new_min = current_price * 0.1  # At least 10% of current price
        if new_max <= new_min:
            new_max = new_min * 1.5  # Ensure valid range

        # Ensure range is not absurdly far from current price
        if new_max > current_price * 10 or new_min < current_price * 0.1:
            return False  # Range too extreme

        # Round to valid ticks
        fee_tier = self.pool_config['feeTier']
        new_min = round_to_nearest_tick(
            new_min,
            fee_tier,
            self.pool_config['token0']['decimals'],
            self.pool_config['token1']['decimals']
        )
        new_max = round_to_nearest_tick(
            new_max,
            fee_tier,
            self.pool_config['token0']['decimals'],
            self.pool_config['token1']['decimals']
        )

        # ===================================================================
        # REBALANCING LOGIC: Full Model Autonomy
        # No forced rebalancing, no cooldown - model has full control
        # ===================================================================

        # Check if price is out of range (informational only, no forced action)
        is_out_of_range = (current_price < self.position_min_price) or (current_price > self.position_max_price)

        # Debug logging
        if debug:
            print(f"[Rebalance Debug] Step {self.current_step}:")
            print(f"  Price: {current_price:.8f}, Range: [{self.position_min_price:.8f}, {self.position_max_price:.8f}]")
            print(f"  Out of range: {is_out_of_range}, Time since rebalance: {self.time_since_last_rebalance}h")
            print(f"  Model action[0]: {should_rebalance_raw:.4f} (> 0 = wants rebalance)")

        # MODEL DECISION: Let AI decide based on learned cost-benefit analysis
        # No forced rebalancing, no cooldown - model has full autonomy
        model_wants_rebalance = should_rebalance_raw > 0

        if not model_wants_rebalance:
            if debug:
                print(f"  → BLOCKED: Model says no (action[0]={should_rebalance_raw:.4f})")
            return False

        # Validate range change is meaningful (physical constraint, not policy)
        old_width = self.position_max_price - self.position_min_price
        new_width = new_max - new_min

        if old_width <= 0:
            if debug:
                print(f"  → BLOCKED: Invalid old range (width={old_width})")
            return False

        # Calculate change percentages
        width_change_pct = abs(new_width - old_width) / old_width
        old_center = (self.position_min_price + self.position_max_price) / 2
        new_center = (new_min + new_max) / 2
        center_shift_pct = abs(new_center - old_center) / old_width

        # Physical constraint: ignore if change too small (<2%)
        # This prevents wasting gas on micro-adjustments
        if width_change_pct < 0.02 and center_shift_pct < 0.02:
            if debug:
                print(f"  → BLOCKED: Change too small (width: {width_change_pct:.2%}, center: {center_shift_pct:.2%})")
            return False

        if debug:
            print(f"  → ALLOWED: Model rebalance (width Δ: {width_change_pct:.2%}, center Δ: {center_shift_pct:.2%})")

        # Apply new ranges
        self.position_min_price = new_min
        self.position_max_price = new_max

        if debug:
            print(f"  ✓ REBALANCED: New range [{new_min:.8f}, {new_max:.8f}]")

        return True  # Rebalancing occurred

    def _calculate_fees(self, current_data: pd.Series, next_data: pd.Series) -> float:
        """
        Calculate fees using Whitepaper formula (Section 6.3, 6.4).

        백서 공식:
            f_r = f_g - f_b(i_l) - f_a(i_u)  (범위 내 수수료 성장률)
            fees = L × (f_r(t₁) - f_r(t₀)) / Q128

        단순화 버전 (feeGrowthOutside 없을 때):
            fees = L × Δf_g × active_ratio / Q128

        Args:
            current_data: Current hour data
            next_data: Next hour data

        Returns:
            Fees earned (USD)
        """
        # get_tick_from_price, active_liquidity_for_candle already imported from uniswap_v3_adapter

        # Get feeGrowthGlobal data
        fg_start_0 = int(current_data.get('feeGrowthGlobal0X128', 0) or 0)
        fg_end_0 = int(next_data.get('feeGrowthGlobal0X128', 0) or 0)
        fg_start_1 = int(current_data.get('feeGrowthGlobal1X128', 0) or 0)
        fg_end_1 = int(next_data.get('feeGrowthGlobal1X128', 0) or 0)

        # If feeGrowthGlobal not available, fallback to TVL-based calculation
        if fg_end_0 == 0 and fg_end_1 == 0:
            return self._calculate_fees_tvl_fallback(current_data, next_data)

        # Get price range for this hour
        low_price = float(next_data.get('low', next_data['close']))
        high_price = float(next_data.get('high', next_data['close']))
        current_price = float(next_data['close'])

        if np.isnan(low_price) or np.isnan(high_price) or low_price <= 0 or high_price <= 0:
            return 0.0

        # Convert prices to ticks for active liquidity calculation
        try:
            low_tick = get_tick_from_price(low_price, self.pool_config, 0)
            high_tick = get_tick_from_price(high_price, self.pool_config, 0)
            min_tick = get_tick_from_price(self.position_min_price, self.pool_config, 0)
            max_tick = get_tick_from_price(self.position_max_price, self.pool_config, 0)
        except Exception:
            return 0.0

        # Calculate active liquidity percentage (0-100)
        active_liq = active_liquidity_for_candle(min_tick, max_tick, low_tick, high_tick)

        if active_liq <= 0:
            return 0.0  # Position was out of range entire hour

        # Get token decimals
        token0_decimals = self.pool_config['token0']['decimals']
        token1_decimals = self.pool_config['token1']['decimals']

        # Use whitepaper formula for fee calculation
        fees_usd = calculate_fees_whitepaper(
            liquidity=self.liquidity,
            fg_start_0=fg_start_0,
            fg_end_0=fg_end_0,
            fg_start_1=fg_start_1,
            fg_end_1=fg_end_1,
            active_pct=active_liq,
            decimals0=token0_decimals,
            decimals1=token1_decimals,
            price=current_price
        )

        # Sanity check: cap at 1% of investment per hour (realistic maximum)
        max_reasonable = self.initial_investment * 0.01
        fees_usd = min(fees_usd, max_reasonable)

        # Replace NaN with 0
        if np.isnan(fees_usd) or np.isinf(fees_usd):
            return 0.0

        return fees_usd

    def _calculate_fees_tvl_fallback(self, current_data: pd.Series, next_data: pd.Series) -> float:
        """
        Fallback fee calculation using TVL when feeGrowthGlobal not available.

        Formula:
        1. total_pool_fees = volumeUSD * fee_tier
        2. basic_share = investment / TVL (your share of the pool)
        3. concentration_factor = accounts for concentrated liquidity advantage
        4. position_fees = total_pool_fees * basic_share * concentration_factor * active_liq

        Args:
            current_data: Current hour data
            next_data: Next hour data

        Returns:
            Fees earned (USD)
        """
        # get_tick_from_price, active_liquidity_for_candle already imported from uniswap_v3_adapter

        # Get hourly volume
        volume_usd = float(next_data.get('volumeUSD', 0))
        if volume_usd <= 0 or np.isnan(volume_usd):
            return 0.0

        # Total fees generated by the pool this hour
        fee_tier_bps = self.pool_config['feeTier']  # e.g., 500 for 0.05%
        total_pool_fees = volume_usd * (fee_tier_bps / 1_000_000)  # Convert bps to decimal

        # Get price range for this hour
        low_price = float(next_data.get('low', next_data['close']))
        high_price = float(next_data.get('high', next_data['close']))

        if np.isnan(low_price) or np.isnan(high_price) or low_price <= 0 or high_price <= 0:
            return 0.0

        # Convert prices to ticks for active liquidity calculation
        try:
            low_tick = get_tick_from_price(low_price, self.pool_config, 0)
            high_tick = get_tick_from_price(high_price, self.pool_config, 0)
            min_tick = get_tick_from_price(self.position_min_price, self.pool_config, 0)
            max_tick = get_tick_from_price(self.position_max_price, self.pool_config, 0)
        except Exception:
            return 0.0

        # Calculate active liquidity percentage (0-100)
        active_liq = active_liquidity_for_candle(min_tick, max_tick, low_tick, high_tick)

        if active_liq <= 0:
            return 0.0  # Position was out of range entire hour

        # --- TVL-based share calculation ---
        pool_info = next_data.get('pool', {})
        if isinstance(pool_info, dict):
            tvl_usd = float(pool_info.get('totalValueLockedUSD', 0))
        else:
            tvl_usd = 0

        if tvl_usd <= 0 or np.isnan(tvl_usd):
            tvl_usd = 100_000_000  # Fallback to $100M (typical for large pools)

        # Calculate basic share (unbounded LP position)
        basic_share = self.initial_investment / tvl_usd

        # --- Concentration factor ---
        current_price = float(next_data['close'])
        if current_price <= 0:
            return 0.0

        # Calculate concentration factor based on range width
        range_width = (self.position_max_price - self.position_min_price) / current_price
        concentration_factor = min(20.0, 1.0 / max(range_width, 0.05))

        # Calculate effective share with concentration
        effective_share = basic_share * concentration_factor
        effective_share = min(effective_share, 0.10)  # Cap at 10%

        # Calculate position fees
        position_fees = total_pool_fees * effective_share * (active_liq / 100.0)

        # Sanity check: cap at 1% of investment per hour
        max_reasonable = self.initial_investment * 0.01
        position_fees = min(position_fees, max_reasonable)

        # Replace NaN with 0
        if np.isnan(position_fees) or np.isinf(position_fees):
            return 0.0

        return position_fees

    def _calculate_il(self, current_price: float, position_min: float, position_max: float, position_liquidity: float) -> float:
        """
        Calculate impermanent loss using Whitepaper formula (Section 6.4.1).

        백서 공식 (토큰 양 기반):
            L = Investment / (2√P₀ - √Pa - P₀/√Pb)
            x = L × (1/√P - 1/√Pb)
            y = L × (√P - √Pa)
            HODL = x₀ × P₁ + y₀
            LP   = x₁ × P₁ + y₁
            IL   = (LP - HODL) / HODL

        Args:
            current_price: Current market price
            position_min: Position min price
            position_max: Position max price
            position_liquidity: Position liquidity (unused but kept for API compatibility)

        Returns:
            IL amount (USD, positive = loss) for this position
        """
        # Handle invalid prices
        if np.isnan(current_price) or current_price <= 0:
            return 0.0
        if not hasattr(self, 'mint_price') or self.mint_price <= 0:
            return 0.0
        if position_min <= 0 or position_max <= 0 or position_min >= position_max:
            return 0.0

        # Use whitepaper formula for IL calculation
        result = calculate_il_whitepaper(
            investment=self.current_position_value,
            mint_price=self.mint_price,
            current_price=current_price,
            pa=position_min,
            pb=position_max
        )

        # IL is negative when LP < HODL (loss)
        # We return positive value as loss
        il_pct = result['il_pct']

        # IL is always expressed as a loss (positive value)
        il_usd = abs(il_pct) * self.current_position_value

        # Sanity check: IL can't exceed position value
        il_usd = min(il_usd, self.current_position_value * 0.5)  # Cap at 50%

        # Replace NaN/inf with 0
        if np.isnan(il_usd) or np.isinf(il_usd):
            return 0.0

        return il_usd

    def _calculate_reward(self, fees: float, il: float, gas: float,
                         did_rebalance: bool, current_data: pd.Series) -> float:
        """
        Calculate reward: α*fees - β*IL - γ*gas

        Rebalancing costs:
        - IL: Realized at rebalancing (included in il parameter)
        - Gas: Charged if did_rebalance=True
        - NO slippage: Uniswap V3 burn/mint has no slippage (not a swap)

        Args:
            fees: Fees earned this step
            il: IL incurred this step (realized if rebalanced)
            gas: Gas cost if rebalanced
            did_rebalance: Whether rebalancing occurred
            current_data: Current market data (unused but kept for compatibility)

        Returns:
            Reward value
        """
        w = self.reward_weights

        # Replace any NaN inputs with 0
        fees = 0.0 if np.isnan(fees) else fees
        il = 0.0 if np.isnan(il) else il
        gas = 0.0 if np.isnan(gas) else gas

        # Reward: fees - IL - gas
        # Beta increased to 1.5 (from 0.8) to prioritize capital preservation
        reward = (
            w['alpha'] * fees -
            w['beta'] * il -
            w['gamma'] * gas
        )

        # Replace NaN reward with 0
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        # Rescale reward using tanh for bounded output (-1, 1)
        # Scale factor: $1000/hour → tanh(1) ≈ 0.76
        # This provides smooth, bounded rewards regardless of fee magnitude
        reward_scaled = reward / 1000.0
        reward_bounded = np.tanh(reward_scaled)

        return reward_bounded

    def _is_position_failed(self, current_price: float) -> bool:
        """
        Check if position is in failed state (out of range too long).

        Args:
            current_price: Current market price

        Returns:
            True if position has failed
        """
        # Out of range for >48 hours
        if current_price <= self.position_min_price or current_price >= self.position_max_price:
            if self.time_since_last_rebalance > 48:
                return True

        return False

    def _calculate_tokens_for_range(self, total_value: float, current_price: float,
                                    range_min: float, range_max: float) -> Tuple[float, float]:
        """
        Calculate required token amounts for a specific range using Uniswap V3 math.

        Uses direct calculation from frontend's tokensForStrategy function.
        This determines the optimal token ratio for a given investment and price range.

        Args:
            total_value: Total position value (USD)
            current_price: Current market price
            range_min: Lower bound of range
            range_max: Upper bound of range

        Returns:
            (token0_amount, token1_amount) required for this range
        """
        decimal = self.pool_config['token1']['decimals'] - self.pool_config['token0']['decimals']

        return tokens_for_strategy(
            range_min,
            range_max,
            total_value,
            current_price,
            decimal
        )

    def _calculate_swap_cost(self, current_tokens: Tuple[float, float],
                            needed_tokens: Tuple[float, float],
                            price: float) -> float:
        """
        Calculate the cost of swapping tokens to reach needed ratio.

        SWAP cost = swap_amount × (pool_fee + slippage)

        Args:
            current_tokens: (token0, token1) currently held
            needed_tokens: (token0, token1) needed for new range
            price: Current market price (token0/token1 format)

        Returns:
            Total swap cost (USD)
        """
        # Calculate token deltas
        delta0 = needed_tokens[0] - current_tokens[0]
        delta1 = needed_tokens[1] - current_tokens[1]

        # Price is already in token1/token0 format (USDC/WETH)
        price_human = price

        # Determine which token needs to be swapped
        # For USDC/WETH pool: token0=USDC, token1=WETH
        if delta0 > 0 and delta1 < 0:
            # Need more token0 (USDC) → swap token1 (WETH) to token0
            swap_amount_usd = abs(delta1) * price_human  # WETH * USDC/WETH = USDC
        elif delta0 < 0 and delta1 > 0:
            # Need more token1 (WETH) → swap token0 (USDC) to token1
            swap_amount_usd = abs(delta0)  # USDC amount (already USD)
        else:
            # No swap needed (both increase or both decrease - shouldn't happen)
            return 0.0

        # Calculate swap costs
        fee_tier = self.pool_config['feeTier']
        pool_fee_rate = fee_tier / 1_000_000  # Convert bps to decimal (500 → 0.0005)
        slippage_rate = 0.001  # 0.1% slippage (conservative estimate for medium-sized pools)

        # Total swap cost = swap amount × (fee + slippage)
        swap_cost = swap_amount_usd * (pool_fee_rate + slippage_rate)

        return swap_cost

    def _get_observation(self) -> np.ndarray:
        """
        Extract state vector from current environment state.
        Supports both 24-dim (legacy) and 28-dim (current) observations.

        Returns:
            numpy array with obs_dim features (normalized)
        """
        current_idx = self.episode_start_idx + self.current_step
        current_data = self.features_df.iloc[current_idx]
        current_price = current_data['close']

        # Market features (10)
        f1 = current_price / self.initial_price  # Normalized price
        f2 = current_data['volatility_24h'] / current_price  # Relative volatility
        f3 = current_data['momentum_1h']
        f4 = current_data['momentum_24h']
        f5 = np.log1p(float(current_data.get('tvl', 1000000))) / 20  # Log-normalized TVL
        f6 = np.log1p(float(current_data.get('volume_24h', 10000))) / 15  # Log-normalized volume
        f7 = float(current_data.get('liquidity', 1000000)) / 1e6  # Liquidity depth
        f8 = self.pool_config['feeTier'] / 10000  # Fee tier (0.05, 0.30, 1.0)
        f9 = current_data['hour_sin']
        f10 = current_data['day_sin']

        # Position features (10)
        min_tick = get_tick_from_price(self.position_min_price, self.pool_config, 0)
        max_tick = get_tick_from_price(self.position_max_price, self.pool_config, 0)
        current_tick = get_tick_from_price(current_price, self.pool_config, 0)

        f11 = (min_tick - current_tick) / 10000  # Normalized distance to min
        f12 = (max_tick - current_tick) / 10000  # Normalized distance to max
        f13 = (max_tick - min_tick) / 10000  # Position width
        f14 = 1.0 if self.position_min_price < current_price < self.position_max_price else 0.0
        f15 = self.time_since_last_rebalance / 168  # Normalized (1 week = 168 hours)
        f16 = self.cumulative_fees / self.initial_investment
        f17 = self.cumulative_il / self.initial_investment
        f18 = self.cumulative_gas / self.initial_investment
        f19 = self.total_rebalances / 30  # Normalized (30 rebalances = 1.0)
        f20 = self.current_step / self.episode_length  # Episode progress

        # Forward-looking features (4 for 24-dim, 8 for 28-dim)
        f21 = current_data.get('momentum_1h', 0)  # Predicted price 1h (use momentum)
        f22 = current_data.get('momentum_24h', 0)  # Predicted price 24h
        f23 = current_data['volatility_24h'] / current_price  # Predicted vol 1h
        f24 = current_data['volatility_24h'] / current_price  # Predicted vol 24h

        if self.obs_dim == 24:
            # 24-dim: Market(10) + Position(10) + Forward(4)
            obs = np.array([
                f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
                f21, f22, f23, f24
            ], dtype=np.float32)
        else:
            # 28-dim: Additional forward-looking features
            f25 = 0.5  # Liquidity concentration above (placeholder)
            f26 = 0.5  # Liquidity concentration below (placeholder)
            f27 = current_data.get('returns', 0) * 24  # Recent fee growth rate
            f28 = current_data.get('momentum_1h', 0)  # Volume trend

            obs = np.array([
                f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
                f21, f22, f23, f24, f25, f26, f27, f28
            ], dtype=np.float32)

        # Replace NaN and inf values with 0 (critical for stable training)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

        # Clip to reasonable bounds
        obs = np.clip(obs, -10, 10)

        return obs

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            current_idx = self.episode_start_idx + self.current_step
            current_data = self.features_df.iloc[current_idx]
            print(f"\nStep: {self.current_step}/{self.episode_length}")
            print(f"Price: {current_data['close']:.2f}")
            print(f"Range: [{self.position_min_price:.2f}, {self.position_max_price:.2f}]")
            print(f"In Range: {self.position_min_price < current_data['close'] < self.position_max_price}")
            print(f"Cumulative Fees: ${self.cumulative_fees:.2f}")
            print(f"Cumulative IL: ${self.cumulative_il:.2f}")
            print(f"Cumulative Gas: ${self.cumulative_gas:.2f}")
            print(f"Net Return: ${self.cumulative_fees - self.cumulative_il - self.cumulative_gas:.2f}")
            print(f"Total Rebalances: {self.total_rebalances}")
