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
    tick_to_sqrt,
    snap_tick,
    amounts_to_L,
    L_to_amounts,
    fee_delta,
    split_tokens as optimal_token_split,
)
from ..constants import Q128, TICK_SPACINGS

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

# HMM regime detection
from .regime import RegimeDetector


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
                 debug: bool = False,
                 invert_price: bool = False):
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
            invert_price: If True, invert prices (token0 is stablecoin case)
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.debug = debug
        self.invert_price = invert_price

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
        # Lowered to $10 for Ethereum to encourage more rebalancing
        self.gas_costs = {
            'ethereum': 10,
            'polygon': 1,
            'arbitrum': 1,
            'optimism': 1,
            'celo': 0.5
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
        self.cumulative_lvr = 0.0  # Sum of LVR penalties (accumulated every step)
        self.cumulative_gas = 0.0
        self.time_since_last_rebalance = 0
        self.total_rebalances = 0
        self.initial_price = 0.0
        self.initial_token_amounts = [0.0, 0.0]  # Current position's mint tokens (updates at each rebalancing)
        self.current_position_value = 0.0  # Actual position value (decreases with IL/gas, increases with fees)
        self.mint_price = 0.0  # Price when current position was minted (for IL calculation)

        # Fee growth tracking (backtest-accurate fee calculation)
        self.prev_fg0 = 0  # Previous feeGrowthGlobal0X128
        self.prev_fg1 = 0  # Previous feeGrowthGlobal1X128
        self.tick_lower = 0  # Position tick lower bound
        self.tick_upper = 0  # Position tick upper bound
        self.backtest_L = 0  # Liquidity calculated exactly like backtest.py

        # Precompute features for efficiency
        self._precompute_features()

        # Initialize HMM regime detector
        self._init_regime_detector()

    def _precompute_features(self):
        """Precompute volatility, momentum, and other features from historical data"""
        df = self.historical_data.copy()

        # Convert price from token0/token1 to token1/token0 format
        # Database stores WETH/USDT (small numbers like 0.0003)
        # We need USDT/WETH (large numbers like 2500) for human-readable prices
        self.invert_price = False  # Track if we're inverting
        if df['close'].iloc[0] < 1.0:  # Prices are in token0/token1 format
            self.invert_price = True
            df['close'] = 1.0 / df['close']
            if 'high' in df.columns:
                # Note: high/low are swapped when inverting
                df['high'], df['low'] = 1.0 / df['low'], 1.0 / df['high']
            # NOTE: tick does NOT need inversion - tick_to_price already handles decimals correctly
            # tick=-195238 with d0=18, d1=6 gives price=3321 USDT/WETH (correct!)

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

    def _init_regime_detector(self):
        """Initialize and fit HMM regime detector on historical returns."""
        # Get returns from preprocessed features
        returns = self.features_df['returns'].values

        # Create and fit detector
        self.regime_detector = RegimeDetector(n_regimes=2, lookback=100)

        try:
            self.regime_detector.fit(returns)
            print("✅ HMM Regime Detector initialized")
        except Exception as e:
            print(f"⚠️ HMM fitting failed: {e}. Using uniform priors.")
            # Detector will return uniform probabilities if not fitted

        # Cache for regime probabilities (updated each step)
        self._regime_proba_cache = None
        self._regime_momentum_cache = None

    def _get_regime_features(self) -> tuple:
        """
        Get HMM regime features for current observation.

        Returns:
            Tuple of (regime_proba, regime_momentum), each shape (2,)
        """
        current_idx = self.episode_start_idx + self.current_step

        # Lookback window for regime detection
        lookback = min(100, current_idx)  # Use up to 100 past returns
        start_idx = max(0, current_idx - lookback)

        # Get recent returns
        recent_returns = self.features_df['returns'].iloc[start_idx:current_idx + 1].values

        if len(recent_returns) < 10:
            # Not enough data for meaningful regime detection
            return np.array([0.5, 0.5]), np.array([0.0, 0.0])

        # Get regime features
        regime_proba, regime_momentum = self.regime_detector.get_regime_features(
            recent_returns, momentum_lookback=10
        )

        return regime_proba, regime_momentum

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
        range_pct = max(volatility / self.initial_price * 2, 0.10)  # At least 10% (±5%)
        range_pct = min(range_pct, 0.50)  # At most 50%

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
        self.cumulative_lvr = 0.0  # Sum of LVR penalties
        self.cumulative_gas = 0.0
        self.time_since_last_rebalance = 0
        self.total_rebalances = 0
        self.current_position_value = self.initial_investment  # Start with full investment

        # Initialize fee growth tracking (backtest-accurate fee calculation)
        self.prev_fg0 = int(current_data.get('feeGrowthGlobal0X128', 0) or 0)
        self.prev_fg1 = int(current_data.get('feeGrowthGlobal1X128', 0) or 0)

        # Initialize tick bounds for position
        self._update_tick_bounds()

        # Calculate backtest-compatible liquidity
        self._update_backtest_L()

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

        CRITICAL: Uses SAME logic as _apply_action() for consistency!
        - action[0]: rebalance decision (ignored for initial range)
        - action[1]: min_range_factor (-1=narrow 0.5σ, +1=wide 5.0σ)
        - action[2]: max_range_factor (-1=narrow 0.5σ, +1=wide 5.0σ)

        Args:
            action: [rebalance_confidence, min_range_factor, max_range_factor] from model
        """
        current_data = self.features_df.iloc[self.episode_start_idx]
        current_price = current_data['close']

        # Calculate volatility (SAME as _apply_action)
        lookback = min(24, self.episode_start_idx)
        if lookback < 2:
            lookback = 2

        start_idx = max(0, self.episode_start_idx - lookback)
        recent_prices = self.features_df.iloc[start_idx:self.episode_start_idx + 1]['close'].values
        volatility = np.std(recent_prices)

        # Handle edge cases
        if volatility <= 0 or np.isnan(volatility):
            volatility = current_price * 0.05  # Default to 5% volatility

        # Extract range factors (SAME as _apply_action)
        # action[0] is rebalance decision - ignored for initial range
        min_range_factor = float(action[1])
        max_range_factor = float(action[2])

        # Clip to valid range
        min_range_factor = max(-1, min(1, min_range_factor))
        max_range_factor = max(-1, min(1, max_range_factor))

        # Map to volatility multiplier [0.5, 5.0] (SAME as _apply_action)
        min_multiplier = 0.5 + (min_range_factor + 1) / 2 * 4.5
        max_multiplier = 0.5 + (max_range_factor + 1) / 2 * 4.5

        # Calculate ranges (SAME as _apply_action)
        new_min = current_price - (volatility * min_multiplier)
        new_max = current_price + (volatility * max_multiplier)

        # Safety bounds
        if new_min <= 0:
            new_min = current_price * 0.1
        if new_max <= new_min:
            new_max = new_min * 1.5

        # Round to valid ticks
        fee_tier = self.pool_config['feeTier']
        self.position_min_price = round_to_nearest_tick(
            new_min,
            fee_tier,
            self.pool_config['token0']['decimals'],
            self.pool_config['token1']['decimals']
        )
        self.position_max_price = round_to_nearest_tick(
            new_max,
            fee_tier,
            self.pool_config['token0']['decimals'],
            self.pool_config['token1']['decimals']
        )

        # Enforce minimum range width (SAME as _apply_action)
        MIN_RANGE_WIDTH_PCT = 0.10
        range_width = (self.position_max_price - self.position_min_price) / current_price
        if range_width < MIN_RANGE_WIDTH_PCT:
            half_width = current_price * MIN_RANGE_WIDTH_PCT / 2
            self.position_min_price = round_to_nearest_tick(
                current_price - half_width, fee_tier,
                self.pool_config['token0']['decimals'],
                self.pool_config['token1']['decimals']
            )
            self.position_max_price = round_to_nearest_tick(
                current_price + half_width, fee_tier,
                self.pool_config['token0']['decimals'],
                self.pool_config['token1']['decimals']
            )

        # Update initial_price
        self.initial_price = current_price

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

        # Update tick bounds for fee calculation
        self._update_tick_bounds()

        # Update backtest-compatible liquidity
        self._update_backtest_L()

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
        base_gas_cost = self.gas_costs[self.protocol] if did_rebalance else 0.0
        swap_cost = 0.0  # Will be calculated in rebalance block

        # Update unrealized IL (current snapshot, not accumulated)
        self.unrealized_il = il_raw

        # IL Logic:
        # - Unrealized IL: Current IL that would be incurred if we rebalanced now
        # - Realized IL: IL that was actually incurred when rebalancing (locked in)
        #
        # For reward calculation:
        # - IL is ONLY penalized when rebalancing (when it becomes realized)
        # - No per-step penalty: IL is a cumulative snapshot, not incremental
        # - Penalizing snapshot IL every step would double-count the loss
        if did_rebalance:
            il_incurred = il_raw  # Full IL penalty at rebalancing (realized)
            self.cumulative_il += il_raw  # Add to realized IL only when rebalancing
        else:
            il_incurred = 0.0  # No IL penalty: position not closed, loss not realized

        # Update cumulative metrics
        self.cumulative_fees += fees_earned
        # NOTE: cumulative_il is now only updated when rebalancing (above)
        self.cumulative_gas += base_gas_cost

        # Update current position value (reflects actual portfolio value)
        # This happens every step: fees increase value, but IL/gas only realized on rebalance
        if did_rebalance:
            # On rebalance: apply IL and gas cost to position value
            self.current_position_value = self.current_position_value - il_raw - base_gas_cost + fees_earned
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

            # Add swap cost to cumulative gas and position value
            self.cumulative_gas += swap_cost
            self.current_position_value -= swap_cost

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

        # Total gas cost for this step (base gas + swap cost)
        total_gas_cost = base_gas_cost + swap_cost

        # =================================================================
        # LVR CALCULATION (Paper: arXiv:2208.06046, arXiv:2501.07508)
        # LVR is calculated EVERY step (unlike IL which was only on rebalance)
        # IMPORTANT: Use RETURNS volatility σ, NOT price volatility
        # =================================================================
        current_price = current_data['close']

        # Calculate RETURNS volatility (24-hour window)
        # σ = std(returns) where returns = (P_t - P_{t-1}) / P_{t-1}
        lookback = min(24, current_idx)
        if lookback < 3:  # Need at least 3 points for meaningful returns std
            lookback = 3
        recent_prices = self.features_df.iloc[current_idx-lookback:current_idx+1]['close'].values

        # Calculate hourly returns
        returns = np.diff(recent_prices) / recent_prices[:-1]
        returns_volatility = np.std(returns)

        # Handle edge case: use default 0.2% hourly volatility
        # (approximately 2% daily volatility / sqrt(24))
        if returns_volatility <= 0 or np.isnan(returns_volatility):
            returns_volatility = 0.002

        # Calculate LVR penalty for this step using exact paper formula
        lvr = self._calculate_lvr(current_price, returns_volatility)

        # Track cumulative LVR
        self.cumulative_lvr += lvr

        # Calculate reward using LVR-based formula
        # R = fees - LVR - gas (gas only when rebalancing)
        reward = self._calculate_reward(fees_earned, lvr, total_gas_cost, did_rebalance, current_data)

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
            'lvr': lvr,  # LVR penalty this step
            'il': il_incurred,  # Keep IL for backward compat
            'realized_il': self.cumulative_il,  # Total IL realized through rebalancing
            'unrealized_il': self.unrealized_il,  # Current unrealized IL
            'cumulative_lvr': self.cumulative_lvr,  # Total LVR accumulated
            'volatility': returns_volatility,  # Returns volatility σ (for debugging)
            'gas': total_gas_cost,  # Includes base gas + swap cost
            'swap_cost': swap_cost,  # Swap cost only (for debugging)
            'rebalanced': did_rebalance,
            'total_rebalances': self.total_rebalances,
            'cumulative_fees': self.cumulative_fees,
            'cumulative_il': self.cumulative_il,  # Keep for backward compat
            'cumulative_gas': self.cumulative_gas,
            'net_return': self.cumulative_fees - self.cumulative_lvr - self.cumulative_gas,  # Use LVR
            'current_position_value': self.current_position_value,  # Actual portfolio value
            'in_range': self.tick_lower <= int(current_data.get('tick', 0)) <= self.tick_upper
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
        # MINIMUM RANGE WIDTH ENFORCEMENT
        # Prevent 0-width or too narrow positions that can't earn fees
        # ===================================================================
        MIN_RANGE_WIDTH_PCT = 0.16  # Minimum 16% width (±8%)

        range_width = (new_max - new_min) / current_price if current_price > 0 else 0

        if range_width < MIN_RANGE_WIDTH_PCT:
            # Expand range symmetrically to meet minimum width
            half_width = current_price * MIN_RANGE_WIDTH_PCT / 2
            new_min = current_price - half_width
            new_max = current_price + half_width

            # Re-round to valid ticks after expansion
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

            if self.debug:
                print(f"  [MIN_WIDTH] Expanded range to {MIN_RANGE_WIDTH_PCT*100:.1f}%: "
                      f"[{new_min:.4f}, {new_max:.4f}]")

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

        # Update tick bounds for fee calculation
        self._update_tick_bounds()

        # Update backtest-compatible liquidity
        self._update_backtest_L()

        return True  # Rebalancing occurred

    def _update_tick_bounds(self) -> None:
        """
        Update tick bounds for the current position.

        Called after position range changes (reset, rebalance).

        When invert_price=True:
            - position_min_price/max_price are in token1/token0 format (e.g., USDT/WETH)
            - price_to_tick expects token1/token0 format
            - No inversion needed, use directly
        When invert_price=False:
            - position_min_price/max_price are in token0/token1 format
            - Need to invert before calling price_to_tick
        """
        token0_decimals = self.pool_config['token0']['decimals']
        token1_decimals = self.pool_config['token1']['decimals']
        fee_tier = self.pool_config['feeTier']
        tick_spacing = TICK_SPACINGS.get(fee_tier, 60)

        if self.position_min_price > 0 and self.position_max_price > 0:
            if self.invert_price:
                # Prices are already in token1/token0 format (USDT/WETH)
                # Use directly with price_to_tick
                tick_lower = price_to_tick(self.position_min_price, token0_decimals, token1_decimals)
                tick_upper = price_to_tick(self.position_max_price, token0_decimals, token1_decimals)
            else:
                # Prices are in token0/token1 format, need to invert
                inverted_min = 1.0 / self.position_min_price
                inverted_max = 1.0 / self.position_max_price
                tick_lower = price_to_tick(inverted_max, token0_decimals, token1_decimals)
                tick_upper = price_to_tick(inverted_min, token0_decimals, token1_decimals)
        else:
            tick_lower = 0
            tick_upper = 0

        # Snap to valid tick spacing
        tick_lower = snap_tick(tick_lower, tick_spacing)
        tick_upper = snap_tick(tick_upper, tick_spacing)

        # Ensure correct order
        if tick_lower > tick_upper:
            tick_lower, tick_upper = tick_upper, tick_lower

        self.tick_lower = tick_lower
        self.tick_upper = tick_upper

    def _update_backtest_L(self) -> None:
        """
        Calculate liquidity EXACTLY like backtest.py.

        Uses amounts_to_L with the same methodology as backtest.py.
        """
        token0_decimals = self.pool_config['token0']['decimals']
        token1_decimals = self.pool_config['token1']['decimals']

        # Get current tick from environment
        current_idx = self.episode_start_idx + self.current_step
        current_data = self.features_df.iloc[current_idx]

        if 'tick' in current_data.index:
            current_tick = int(current_data['tick'])
        else:
            self.backtest_L = 0
            return

        # Get price from tick
        # tick_to_price returns token1/token0 format (USDT/WETH for WETH-USDT pool)
        # This is already in the correct human-readable format (e.g., 3000 USDT per WETH)
        # DO NOT invert - tick_to_price already handles decimals correctly
        P0 = tick_to_price(current_tick, token0_decimals, token1_decimals)

        # Get range prices
        # After _precompute_features fix, position_min/max_price are already
        # in USDT/WETH format (e.g., 2500, 3500) when invert_price=True
        # DO NOT invert - they're already correct
        if self.position_min_price > 0 and self.position_max_price > 0:
            Pa = self.position_min_price  # Lower bound (already USDT/WETH)
            Pb = self.position_max_price  # Upper bound (already USDT/WETH)
            if Pa > Pb:
                Pa, Pb = Pb, Pa
        else:
            self.backtest_L = 0
            return

        # Calculate optimal token split (EXACT backtest.py)
        x_human, y_human = optimal_token_split(self.current_position_value, P0, Pa, Pb)
        x_amount = int(x_human * (10 ** token0_decimals))
        y_amount = int(y_human * (10 ** token1_decimals))

        # Calculate liquidity (EXACT backtest.py)
        sqrt_price_initial = tick_to_sqrt(current_tick)
        sqrt_price_lower = tick_to_sqrt(self.tick_lower)
        sqrt_price_upper = tick_to_sqrt(self.tick_upper)

        self.backtest_L = amounts_to_L(
            sqrt_price_initial,
            sqrt_price_lower,
            sqrt_price_upper,
            x_amount,
            y_amount
        )

    def _get_price_from_tick(self, tick: int) -> float:
        """
        Convert tick to human-readable price, respecting invert_price setting.
        """
        token0_decimals = self.pool_config['token0']['decimals']
        token1_decimals = self.pool_config['token1']['decimals']
        p = tick_to_price(tick, token0_decimals, token1_decimals)
        return 1/p if self.invert_price and p != 0 else p

    def _calculate_fees(self, current_data: pd.Series, next_data: pd.Series) -> float:
        """
        Calculate fees - EXACT copy of backtest.py logic.

        수수료 공식 (backtest.py 동일):
            1. tick 기반 in_range 체크 (binary)
            2. in_range일 때만 수수료 적립
            3. fees = L × Δfee_growth / Q128
            4. prev_fg는 매 step 업데이트 (out of range 포함)

        Args:
            current_data: Current hour data
            next_data: Next hour data

        Returns:
            Fees earned (USD)
        """
        fees_usd = 0.0

        # Get current tick from next_data
        if 'tick' not in next_data.index:
            self._update_prev_fg(next_data)
            return 0.0

        current_tick = int(next_data['tick'])
        token0_decimals = self.pool_config['token0']['decimals']
        token1_decimals = self.pool_config['token1']['decimals']

        # Get current price from tick
        # tick_to_price returns token1/token0 format (e.g., USDT/WETH ~3000)
        # DO NOT invert - this is already the correct human-readable format
        current_price = tick_to_price(current_tick, token0_decimals, token1_decimals)

        # Binary in_range check (backtest.py style)
        in_range = self.tick_lower <= current_tick <= self.tick_upper

        if in_range:
            # Get current fee growth values
            curr_fg0 = int(next_data.get('feeGrowthGlobal0X128', 0) or 0)
            curr_fg1 = int(next_data.get('feeGrowthGlobal1X128', 0) or 0)

            # If feeGrowthGlobal not available, fallback
            if curr_fg0 == 0 and curr_fg1 == 0:
                fees_usd = self._calculate_fees_tvl_fallback(current_data, next_data)
            else:
                # Calculate fee delta (handles wrap-around)
                delta_fg0 = fee_delta(curr_fg0, self.prev_fg0)
                delta_fg1 = fee_delta(curr_fg1, self.prev_fg1)

                # Calculate raw fee amounts (EXACT backtest.py formula)
                L = self.backtest_L  # Use backtest-compatible L
                if L > 0:
                    fee0_raw = (L * delta_fg0) // Q128
                    fee1_raw = (L * delta_fg1) // Q128

                    # Convert to USD
                    # For WETH-USDT pool (invert_price=True):
                    #   token0 = WETH → multiply by current_price (USDT/WETH) to get USD
                    #   token1 = USDT → already in USD
                    # For USDT-WETH pool (invert_price=False):
                    #   token0 = USDT → already in USD
                    #   token1 = WETH → multiply by current_price to get USD
                    if self.invert_price:
                        # token0=WETH needs price conversion, token1=USDT is stablecoin
                        fee0_usd = (fee0_raw / 10**token0_decimals) * current_price
                        fee1_usd = fee1_raw / 10**token1_decimals
                    else:
                        # token0 is stablecoin, token1 needs price conversion
                        fee0_usd = fee0_raw / 10**token0_decimals
                        fee1_usd = (fee1_raw / 10**token1_decimals) * current_price

                    fees_usd = fee0_usd + fee1_usd

        # ALWAYS update prev_fg at end of step (even when out of range)
        self._update_prev_fg(next_data)

        # Replace NaN/inf with 0
        if np.isnan(fees_usd) or np.isinf(fees_usd):
            return 0.0

        return fees_usd

    def _update_prev_fg(self, data: pd.Series) -> None:
        """Update previous fee growth values from data row."""
        self.prev_fg0 = int(data.get('feeGrowthGlobal0X128', 0) or 0)
        self.prev_fg1 = int(data.get('feeGrowthGlobal1X128', 0) or 0)

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

    def _calculate_lvr(self, current_price: float, returns_volatility: float) -> float:
        """
        Calculate Loss-Versus-Rebalancing (LVR) penalty using EXACT paper formula.

        Paper: "Loss-Versus-Rebalancing" (arXiv:2208.06046)
        Also used in: arXiv:2501.07508

        EXACT FORMULA:
            ℓ_t(σ, p_t) = (L × σ²) / 4 × √p_t

        Where:
            - L: Liquidity in USD-equivalent units
            - σ: Volatility (returns standard deviation, NOT price std dev)
            - p_t: Current price (token1/token0, e.g., USDT/WETH)

        For hourly data, the formula naturally applies per hour when σ is hourly volatility.

        Args:
            current_price: Current market price (token1/token0)
            returns_volatility: Returns volatility σ (std dev of hourly returns)

        Returns:
            LVR penalty amount (USD) per hour
        """
        # Handle invalid inputs
        if current_price <= 0 or np.isnan(current_price):
            return 0.0
        if returns_volatility <= 0 or np.isnan(returns_volatility):
            return 0.0
        if self.backtest_L <= 0:
            return 0.0

        # =================================================================
        # EXACT LVR FORMULA FROM PAPER
        # ℓ = (L × σ²) / 4 × √p
        # =================================================================

        # Step 1: Convert L from raw on-chain units to USD-equivalent
        # L is stored in units of sqrt(token0_amount × token1_amount)
        # To convert to USD scale: divide by 10^((d0 + d1) / 2)
        # For WETH/USDT: (18 + 6) / 2 = 12
        token0_decimals = self.pool_config['token0']['decimals']
        token1_decimals = self.pool_config['token1']['decimals']
        decimal_adjustment = (token0_decimals + token1_decimals) / 2
        L_usd = self.backtest_L / (10 ** decimal_adjustment)

        # Step 2: Calculate σ² (returns volatility squared)
        # returns_volatility is already the std dev of returns (e.g., 0.002 for 0.2%)
        sigma_squared = returns_volatility ** 2

        # Step 3: Calculate √p (square root of current price)
        sqrt_price = np.sqrt(current_price)

        # Step 4: Apply the exact paper formula
        # ℓ = (L_usd × σ²) / 4 × √p
        lvr = (L_usd * sigma_squared * sqrt_price) / 4.0

        # =================================================================
        # Sanity checks and caps
        # =================================================================
        # Cap LVR at 1% of position value per hour (extreme volatility protection)
        max_lvr = self.current_position_value * 0.01
        lvr_capped = min(lvr, max_lvr)

        # Replace NaN/inf with 0
        if np.isnan(lvr_capped) or np.isinf(lvr_capped):
            return 0.0

        return lvr_capped

    def _calculate_reward(self, fees: float, lvr: float, gas: float,
                         did_rebalance: bool, current_data: pd.Series) -> float:
        """
        Calculate reward using LVR-based formula from paper.

        Paper: arXiv:2501.07508
        R_{t+1} = f_t - ℓ_t(σ,p) - I[a_t ≠ 0] × g

        Where:
        - f_t: trading fees earned
        - ℓ_t(σ,p): LVR penalty (applied every step)
        - g: gas fee (only when rebalancing)

        Key differences from previous IL-based approach:
        - LVR is applied EVERY step (not just on rebalancing)
        - LVR is volatility-based (encourages wider ranges in volatile markets)
        - Simpler formula without separate α, β, γ coefficients

        Args:
            fees: Fees earned this step
            lvr: LVR penalty this step (applied every step)
            gas: Gas cost (only if rebalanced)
            did_rebalance: Whether rebalancing occurred
            current_data: Current market data

        Returns:
            Reward value (bounded by tanh)
        """
        # Replace any NaN inputs with 0
        fees = 0.0 if np.isnan(fees) else fees
        lvr = 0.0 if np.isnan(lvr) else lvr
        gas = 0.0 if np.isnan(gas) else gas

        # =================================================================
        # PAPER-BASED REWARD: R = fees - LVR - gas
        # arXiv:2501.07508
        # =================================================================
        # Key: LVR is applied every step (volatility-based penalty)
        # Gas is only applied when rebalancing (I[a_t ≠ 0] × g)
        reward = fees - lvr - gas

        # Replace NaN reward with 0
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        # Rescale reward using tanh for bounded output (-1, 1)
        # Scale factor adjusted for typical hourly fees ($1-5)
        # $10 change = tanh(1.0) ≈ 0.76 (provides good gradient)
        reward_scaled = reward / 10.0
        reward_bounded = np.tanh(reward_scaled)

        return reward_bounded

    def _is_position_failed(self, current_price: float) -> bool:
        """
        Check if position is in failed state.

        NOTE: Early termination disabled to allow model to learn
        long-term rebalancing strategies without artificial cutoff.

        Args:
            current_price: Current market price

        Returns:
            Always False (no early termination)
        """
        # Disabled: Let model learn to handle out-of-range situations
        # Previously: terminated if out of range > 48 hours
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
        # When rebalancing: LP withdraws → swaps → re-deposits
        # During swap, LP pays pool fee (no longer providing liquidity at that moment)
        fee_tier = self.pool_config['feeTier']
        pool_fee_rate = fee_tier / 1_000_000  # e.g., 3000 → 0.003 (0.3%)
        slippage_rate = 0.001  # 0.1% price impact

        # Total swap cost = swap amount × (pool_fee + slippage)
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
            # 28-dim: HMM regime features (replaces placeholders)
            regime_proba, regime_momentum = self._get_regime_features()
            f25 = regime_proba[0]  # P(low volatility regime)
            f26 = regime_proba[1]  # P(high volatility regime)
            f27 = regime_momentum[0]  # Low vol regime transition speed
            f28 = regime_momentum[1]  # High vol regime transition speed

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
