#!/usr/bin/env python
"""
UniswapV3LPEnvV9: Excess Return Reward Environment

v9 Reward Function:
- 기본 보상 = LP_return - HODL_return (excess return)
- 범위 내 보너스 = in_range * 0.001
- 가스 패널티 = rebalance_cost

핵심 차이점 (vs v8):
- HODL을 베이스라인으로 사용하여 상대적 성과 측정
- LP가 HODL보다 좋으면 positive, 나쁘면 negative
- "아무것도 안하기"(HODL)를 기준점으로 설정

Actions: [0, 4000, 8000, 16000] tick widths
- 0: No rebalance (hold current position)
- 4000: ±20% range (narrow, high fee capture)
- 8000: ±40% range (medium)
- 16000: ±80% range (wide, low IL risk)
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

from uniswap_v3.math.convert import tick_to_price, price_to_tick
from uniswap_v3.math.calc import amounts_to_L, L_to_amounts


# Constants
MIN_RANGE_WIDTH_PCT = 0.02  # Minimum 2% range width


class UniswapV3LPEnvV9(gym.Env):
    """
    Uniswap V3 LP Environment with Excess Return Reward.

    Reward = (LP_return - HODL_return) + time_in_range_bonus - gas_penalty

    This compares LP performance against HODL baseline:
    - Positive when LP outperforms HODL (fees > IL)
    - Negative when LP underperforms HODL (fees < IL)
    - time_in_range_bonus = in_range * 0.001 (hourly bonus)
    - gas_penalty = gas_cost / initial_investment

    Action Space: Discrete(4)
        0: No rebalance (hold)
        1: Rebalance to ±20% range (4000 ticks)
        2: Rebalance to ±40% range (8000 ticks)
        3: Rebalance to ±80% range (16000 ticks)

    Observation Space: Box(28,) normalized features
    """

    metadata = {"render_modes": ["human"]}

    # Action definitions: tick widths for each action
    ACTION_TICK_WIDTHS = [0, 4000, 8000, 16000]

    def __init__(
        self,
        historical_data: pd.DataFrame,
        pool_config: Dict[str, Any],
        initial_investment: float = 10000.0,
        episode_length_hours: int = 720,
        gas_cost_usd: float = 5.0,
        swap_fee_pct: float = 0.003,
        reward_weights: Optional[Dict[str, float]] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.historical_data = historical_data
        self.pool_config = pool_config
        self.initial_investment = initial_investment
        self.episode_length_hours = episode_length_hours
        self.gas_cost_usd = gas_cost_usd
        self.swap_fee_pct = swap_fee_pct
        self.render_mode = render_mode

        # Reward weights
        self.reward_weights = reward_weights or {
            'in_range_bonus': 0.001,  # Hourly bonus for being in range (0.1% of excess return scale)
            'gas_penalty_scale': 1.0,
        }

        # Pool configuration
        self.fee_tier = pool_config.get('feeTier', 3000)
        self.token0_decimals = pool_config.get('token0', {}).get('decimals', 18)
        self.token1_decimals = pool_config.get('token1', {}).get('decimals', 6)

        # Determine price inversion (for WETH/USDT style pairs)
        self.invert_price = self.token0_decimals > self.token1_decimals

        # Action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)

        # Observation space: 28 features (normalized)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32
        )

        # Preprocess data
        self._preprocess_data()

        # Episode state (initialized in reset)
        self.current_step = 0
        self.episode_start_idx = 0
        self.current_position_value = initial_investment
        self.hodl_value = initial_investment
        self.initial_hodl_tokens = (0.0, 0.0)  # (token0, token1) amounts
        self.position_liquidity = 0.0
        self.tick_lower = 0
        self.tick_upper = 0
        self.total_fees = 0.0
        self.total_gas_spent = 0.0
        self.rebalance_count = 0
        self.time_in_range = 0  # Total hours in range

    def _preprocess_data(self):
        """Preprocess historical data for features."""
        df = self.historical_data.copy()

        # Ensure required columns exist
        required_cols = ['close', 'periodStartUnix']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Calculate returns and volatility
        df['returns'] = df['close'].pct_change().fillna(0)
        df['volatility_24h'] = df['returns'].rolling(24, min_periods=1).std()
        df['ma_24h'] = df['close'].rolling(24, min_periods=1).mean()
        df['ma_168h'] = df['close'].rolling(168, min_periods=1).mean()

        self.features_df = df

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Random start within valid range
        max_start = len(self.features_df) - self.episode_length_hours - 10
        if max_start <= 0:
            self.episode_start_idx = 0
        else:
            self.episode_start_idx = self.np_random.integers(0, max_start)

        self.current_step = 0

        # Get initial price
        current_data = self.features_df.iloc[self.episode_start_idx]
        current_price = current_data['close']

        # Initialize position (±20% range by default)
        self._initialize_position(current_price, tick_width=4000)

        # Initialize HODL baseline (50/50 split)
        token0_value = self.initial_investment / 2
        token1_value = self.initial_investment / 2

        if self.invert_price:
            # token0 is volatile (WETH), token1 is stablecoin (USDT)
            self.initial_hodl_tokens = (token0_value / current_price, token1_value)
        else:
            # token0 is stablecoin, token1 is volatile
            self.initial_hodl_tokens = (token0_value, token1_value / current_price)

        self.hodl_value = self.initial_investment

        # Reset tracking
        self.total_fees = 0.0
        self.total_gas_spent = 0.0
        self.rebalance_count = 0
        self.time_in_range = 0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _initialize_position(self, current_price: float, tick_width: int = 4000):
        """Initialize LP position around current price."""
        # Convert price to tick
        if self.invert_price:
            current_tick = price_to_tick(current_price, self.token0_decimals, self.token1_decimals)
        else:
            current_tick = price_to_tick(1.0 / current_price, self.token0_decimals, self.token1_decimals)

        # Set tick range
        half_width = tick_width // 2
        self.tick_lower = current_tick - half_width
        self.tick_upper = current_tick + half_width

        # Calculate liquidity from initial investment
        self.position_liquidity = self._calculate_liquidity(
            self.initial_investment, current_price
        )

        self.current_position_value = self.initial_investment
        self.mint_price = current_price

    def _calculate_liquidity(self, value_usd: float, current_price: float) -> float:
        """Calculate liquidity L from USD value."""
        # Split value 50/50
        token0_value = value_usd / 2
        token1_value = value_usd / 2

        if self.invert_price:
            token0_amount = token0_value / current_price
            token1_amount = token1_value
        else:
            token0_amount = token0_value
            token1_amount = token1_value / current_price

        # Get tick prices
        price_lower = tick_to_price(self.tick_lower, self.token0_decimals, self.token1_decimals)
        price_upper = tick_to_price(self.tick_upper, self.token0_decimals, self.token1_decimals)

        sqrt_price = np.sqrt(current_price)
        sqrt_lower = np.sqrt(price_lower)
        sqrt_upper = np.sqrt(price_upper)

        # Calculate L
        L = amounts_to_L(
            token0_amount, token1_amount,
            sqrt_price, sqrt_lower, sqrt_upper
        )

        return max(L, 1e-10)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.current_step += 1
        current_idx = self.episode_start_idx + self.current_step

        if current_idx >= len(self.features_df):
            return self._get_observation(), 0.0, True, False, self._get_info()

        current_data = self.features_df.iloc[current_idx]
        current_price = current_data['close']

        # Calculate current tick
        if self.invert_price:
            current_tick = price_to_tick(current_price, self.token0_decimals, self.token1_decimals)
        else:
            current_tick = price_to_tick(1.0 / current_price, self.token0_decimals, self.token1_decimals)

        # Check if in range
        in_range = self.tick_lower <= current_tick <= self.tick_upper

        # Handle rebalancing
        gas_penalty = 0.0
        if action > 0:
            tick_width = self.ACTION_TICK_WIDTHS[action]
            gas_penalty = self._rebalance(current_price, current_tick, tick_width)
            self.rebalance_count += 1
            # Recalculate in_range after rebalance
            in_range = self.tick_lower <= current_tick <= self.tick_upper

        # Update position value
        self._update_position_value(current_price, current_tick, in_range)

        # Update HODL value
        if self.invert_price:
            # token0 is volatile (needs price conversion), token1 is stablecoin
            self.hodl_value = (self.initial_hodl_tokens[0] * current_price) + self.initial_hodl_tokens[1]
        else:
            # token0 is stablecoin, token1 is volatile (needs price conversion)
            self.hodl_value = self.initial_hodl_tokens[0] + (self.initial_hodl_tokens[1] * current_price)

        # Track time in range
        if in_range:
            self.time_in_range += 1

        # Calculate reward: excess return + in_range_bonus - gas_penalty
        # LP return (this step)
        lp_return = (self.current_position_value - self.initial_investment) / self.initial_investment

        # HODL return (this step)
        hodl_return = (self.hodl_value - self.initial_investment) / self.initial_investment

        # Excess return = LP - HODL
        excess_return = lp_return - hodl_return

        # Time-in-range bonus (small per-hour bonus)
        in_range_bonus = self.reward_weights['in_range_bonus'] if in_range else 0.0

        # Gas penalty (normalized by initial investment)
        gas_penalty_normalized = gas_penalty / self.initial_investment * self.reward_weights['gas_penalty_scale']

        # Final reward
        reward = excess_return + in_range_bonus - gas_penalty_normalized

        # Check termination
        done = self.current_step >= self.episode_length_hours
        truncated = self.current_position_value < self.initial_investment * 0.1  # 90% loss

        obs = self._get_observation()
        info = {
            'lp_value': self.current_position_value,
            'hodl_value': self.hodl_value,
            'excess_return': excess_return,
            'in_range': in_range,
            'time_in_range': self.time_in_range,
            'rebalance_count': self.rebalance_count,
            'total_fees': self.total_fees,
            'total_gas': self.total_gas_spent,
            'fees': 0.0,  # Per-step fees (simplified)
        }

        return obs, reward, done, truncated, info

    def _rebalance(self, current_price: float, current_tick: int, tick_width: int) -> float:
        """Rebalance position to new range. Returns gas cost."""
        # Calculate swap cost
        old_tokens = self._get_position_tokens(current_price)

        # Set new range
        half_width = tick_width // 2
        self.tick_lower = current_tick - half_width
        self.tick_upper = current_tick + half_width

        # Calculate new token amounts needed
        new_tokens = self._calculate_tokens_for_range(
            self.current_position_value, current_price
        )

        # Calculate swap amount
        delta0 = new_tokens[0] - old_tokens[0]
        delta1 = new_tokens[1] - old_tokens[1]

        # Swap cost calculation (fixed for invert_price)
        if delta0 > 0 and delta1 < 0:
            # Swapping token1 to token0
            if self.invert_price:
                swap_amount_usd = abs(delta1)  # delta1 is already in USD (USDT)
            else:
                swap_amount_usd = abs(delta1) * current_price
        elif delta0 < 0 and delta1 > 0:
            # Swapping token0 to token1
            if self.invert_price:
                swap_amount_usd = abs(delta0) * current_price  # convert WETH to USD
            else:
                swap_amount_usd = abs(delta0)
        else:
            swap_amount_usd = 0.0

        swap_fee = swap_amount_usd * self.swap_fee_pct
        total_cost = self.gas_cost_usd + swap_fee

        # Deduct cost from position
        self.current_position_value -= total_cost
        self.total_gas_spent += total_cost

        # Update liquidity
        self.position_liquidity = self._calculate_liquidity(
            self.current_position_value, current_price
        )
        self.mint_price = current_price

        return total_cost

    def _get_position_tokens(self, current_price: float) -> Tuple[float, float]:
        """Get current token amounts in position."""
        price_lower = tick_to_price(self.tick_lower, self.token0_decimals, self.token1_decimals)
        price_upper = tick_to_price(self.tick_upper, self.token0_decimals, self.token1_decimals)

        sqrt_price = np.sqrt(current_price)
        sqrt_lower = np.sqrt(price_lower)
        sqrt_upper = np.sqrt(price_upper)

        token0, token1 = L_to_amounts(
            self.position_liquidity,
            sqrt_price, sqrt_lower, sqrt_upper
        )

        return token0, token1

    def _calculate_tokens_for_range(self, value_usd: float, current_price: float) -> Tuple[float, float]:
        """Calculate token amounts for given value and current range."""
        token0_value = value_usd / 2
        token1_value = value_usd / 2

        if self.invert_price:
            token0_amount = token0_value / current_price
            token1_amount = token1_value
        else:
            token0_amount = token0_value
            token1_amount = token1_value / current_price

        return token0_amount, token1_amount

    def _update_position_value(self, current_price: float, current_tick: int, in_range: bool):
        """Update current position value based on price and fees."""
        # Get token amounts
        token0, token1 = self._get_position_tokens(current_price)

        # Calculate USD value
        if self.invert_price:
            self.current_position_value = token0 * current_price + token1
        else:
            self.current_position_value = token0 + token1 * current_price

        # Add fees if in range (simplified fee model)
        if in_range:
            # Estimate hourly fees based on position value and fee tier
            hourly_fee_rate = (self.fee_tier / 1e6) * 0.01  # Simplified
            fees = self.current_position_value * hourly_fee_rate
            self.current_position_value += fees
            self.total_fees += fees

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        current_idx = self.episode_start_idx + self.current_step
        if current_idx >= len(self.features_df):
            current_idx = len(self.features_df) - 1

        current_data = self.features_df.iloc[current_idx]
        current_price = current_data['close']

        # Calculate tick
        if self.invert_price:
            current_tick = price_to_tick(current_price, self.token0_decimals, self.token1_decimals)
        else:
            current_tick = price_to_tick(1.0 / current_price, self.token0_decimals, self.token1_decimals)

        in_range = self.tick_lower <= current_tick <= self.tick_upper

        # Build observation (28 features)
        obs = np.zeros(28, dtype=np.float32)

        # Price features (0-4)
        obs[0] = current_price / 2000.0  # Normalized price
        obs[1] = current_data.get('returns', 0.0)
        obs[2] = current_data.get('volatility_24h', 0.02)
        obs[3] = current_data.get('ma_24h', current_price) / current_price
        obs[4] = current_data.get('ma_168h', current_price) / current_price

        # Position features (5-14)
        obs[5] = self.current_position_value / self.initial_investment
        obs[6] = self.hodl_value / self.initial_investment
        obs[7] = (self.current_position_value - self.hodl_value) / self.initial_investment  # Excess
        obs[8] = float(in_range)
        obs[9] = self.time_in_range / max(self.current_step, 1)  # In-range ratio

        # Range features (10-14)
        price_lower = tick_to_price(self.tick_lower, self.token0_decimals, self.token1_decimals)
        price_upper = tick_to_price(self.tick_upper, self.token0_decimals, self.token1_decimals)
        range_width = (price_upper - price_lower) / current_price
        obs[10] = range_width
        obs[11] = (current_price - price_lower) / (price_upper - price_lower + 1e-10)  # Position in range
        obs[12] = self.rebalance_count / 100.0  # Normalized rebalance count
        obs[13] = self.total_gas_spent / self.initial_investment
        obs[14] = self.total_fees / self.initial_investment

        # Time features (15-19)
        obs[15] = self.current_step / self.episode_length_hours
        obs[16] = (self.episode_length_hours - self.current_step) / self.episode_length_hours

        # Historical returns (17-27)
        for i in range(10):
            idx = current_idx - i
            if idx >= 0:
                obs[17 + i] = self.features_df.iloc[idx].get('returns', 0.0)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        return {
            'lp_value': self.current_position_value,
            'hodl_value': self.hodl_value,
            'excess_return': (self.current_position_value - self.hodl_value) / self.initial_investment,
            'time_in_range': self.time_in_range,
            'rebalance_count': self.rebalance_count,
            'total_fees': self.total_fees,
            'total_gas': self.total_gas_spent,
        }

    @property
    def is_in_range(self) -> bool:
        """Check if current price is in range."""
        current_idx = self.episode_start_idx + self.current_step
        if current_idx >= len(self.features_df):
            return False

        current_data = self.features_df.iloc[current_idx]
        current_price = current_data['close']

        if self.invert_price:
            current_tick = price_to_tick(current_price, self.token0_decimals, self.token1_decimals)
        else:
            current_tick = price_to_tick(1.0 / current_price, self.token0_decimals, self.token1_decimals)

        return self.tick_lower <= current_tick <= self.tick_upper
