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

# TA-Lib for technical indicators (Paper: arXiv:2501.07508)
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

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

    Action Space: Continuous [-1, 1]^3
        - action[0]: Rebalancing decision (-1=no, +1=yes)
        - action[1]: Lower range factor (-1=narrow, +1=wide)
        - action[2]: Upper range factor (-1=narrow, +1=wide)

    Reward: R = fees - β*LVR (gas removed)
        - β: LVR weight (default 2.0) - higher = more conservative, minimizes IL indirectly
        - LVR: Loss-Versus-Rebalancing (arXiv:2208.06046)
        - Gas removed to focus on core LP economics

    Fee Calculation: Whitepaper formulas (Section 6.3, 6.4.1)
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
            reward_weights: Dict with 'lvr_weight' (β) for LVR penalty tuning (default: 1.0)
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

        # Reward function weights (Paper: arXiv:2501.07508)
        # reward = fees - β*LVR (gas excluded to avoid saturation)
        # β=1.0 per paper: treats LVR as true opportunity cost
        self.reward_weights = reward_weights or {
            'lvr_weight': 1.0,    # LVR coefficient (β) - paper uses 1.0
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
        # Paper (arXiv:2501.07508): $5 based on Etherscan gas tracker
        self.gas_costs = {
            'ethereum': 5,
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

        # Action space: Continuous (Box) - 3D Action Space
        # action[0] = lower_pct: 하한선 (현재가 - X%), 범위 1%~99%
        # action[1] = upper_pct: 상한선 (현재가 + Y%), 범위 1%~500%
        # action[2] = rebalance_signal: 리밸런싱 신호 (0~1)
        #             > 0.5: 즉시 리밸런싱 실행
        #             < 0.5: auto-rebalancing 로직 따름
        #
        # LP 특성:
        # - 하한선: 가격 > 0 이어야 함 → 최대 99%까지만 가능
        # - 상한선: 이론상 무한대, 실용적으로 500% (5배) 이상은 full range
        # - 예: [0.10, 0.30, 0.7] → 현재가 -10% ~ +30% 범위, 즉시 리밸런싱
        self.action_space = spaces.Box(
            low=np.array([0.01, 0.01, 0.0], dtype=np.float32),
            high=np.array([0.99, 5.00, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        # Store desired bounds for auto-rebalancing logic
        self.desired_lower_pct = 0.10  # Default 10%
        self.desired_upper_pct = 0.10  # Default 10%
        self.rebalance_signal = 0.0    # Model's rebalancing urgency

        # Observation space: 12 dimensions (장기 투자용)
        # [price, tick, width, liquidity, volatility_7d, volatility_30d,
        #  price_to_ma7d, price_to_ma30d, momentum_7d, momentum_30d,
        #  edge_proximity_lower, edge_proximity_upper]
        self.obs_dim = 12
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
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
        self.prev_unrealized_il = 0.0  # Previous step's unrealized IL (for delta calculation)
        self.cumulative_il_delta = 0.0  # Sum of IL deltas (for tracking total IL-based penalty)
        self.cumulative_lvr = 0.0  # Sum of LVR penalties (accumulated every step)
        self.cumulative_gas = 0.0
        self.time_since_last_rebalance = 0
        self.total_rebalances = 0
        self.initial_price = 0.0
        self.initial_token_amounts = [0.0, 0.0]  # Current position's mint tokens (updates at each rebalancing)
        self.current_position_value = 0.0  # Actual position value (decreases with IL/gas, increases with fees)
        self.mint_price = 0.0  # Price when current position was minted (for IL calculation)
        self.position_investment = 0.0  # Investment at position creation (fixed until rebalance, for IL calculation)

        # HODL tracking for LP vs HODL excess return calculation
        self.hodl_token0 = 0.0  # Initial token0 amount (fixed throughout episode)
        self.hodl_token1 = 0.0  # Initial token1 amount (fixed throughout episode)
        self.prev_excess_return = 0.0  # Previous step's excess return (for delta calculation)

        # Auto-rebalancing based on IL graph analysis (70% threshold)
        # See docs/il_graph.png - Position Quality vs IL Risk
        # Continuous action space: [lower_pct, upper_pct]
        self.current_lower_pct = 0.10  # Current position's lower bound %
        self.current_upper_pct = 0.10  # Current position's upper bound %
        self.rebalance_threshold = 0.70  # 70% edge proximity triggers rebalance

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

        # === LONG-TERM FEATURES (for LP investment) ===
        # Volatility: 7-day and 30-day
        df['volatility_7d'] = df['returns'].rolling(168).std()   # 7 days = 168 hours
        df['volatility_30d'] = df['returns'].rolling(720).std()  # 30 days = 720 hours

        # Moving averages: 7-day and 30-day
        df['ma_7d'] = df['close'].rolling(168).mean()   # 7 days
        df['ma_30d'] = df['close'].rolling(720).mean()  # 30 days

        # Price momentum: 7-day and 30-day
        df['momentum_7d'] = df['close'].pct_change(168)   # 7-day return
        df['momentum_30d'] = df['close'].pct_change(720)  # 30-day return

        # Price relative to moving averages (trend indicators)
        df['price_to_ma7d'] = df['close'] / df['ma_7d']
        df['price_to_ma30d'] = df['close'] / df['ma_30d']

        # BOP, BB, ADXR, DX 삭제 - 장기 투자 특성만 사용

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

        # Warmup period: 720 hours (30 days) for rolling features
        # 30-day volatility, 30-day MA, 30-day momentum all need 720 hours of history
        warmup_period = 720

        if len(self.features_df) < warmup_period + required_length:
            raise ValueError(f"Not enough data: {len(self.features_df)} rows, need {warmup_period + required_length}")

        # For training: random start after warmup period
        # Episode start must be >= warmup_period to ensure valid rolling features
        max_start = len(self.features_df) - required_length
        if max_start > warmup_period:
            self.episode_start_idx = np.random.randint(warmup_period, max_start)
        else:
            self.episode_start_idx = warmup_period  # Start right after warmup
        self.current_step = 0

        # Get initial market data
        current_data = self.features_df.iloc[self.episode_start_idx]
        self.initial_price = current_data['close']
        volatility = current_data['volatility_7d']

        # Initialize with wide range: ±100% (conservative start)
        # Model learns to narrow down the range for higher fee concentration
        default_lower_pct = 0.99  # -99% from current price (near 0)
        default_upper_pct = 1.00  # +100% from current price (2x)
        self.current_lower_pct = default_lower_pct
        self.current_upper_pct = default_upper_pct
        self.desired_lower_pct = default_lower_pct
        self.desired_upper_pct = default_upper_pct

        # Calculate position from percentage bounds
        fee_tier = self.pool_config['feeTier']
        tick_spacing = TICK_SPACINGS.get(fee_tier, 60)
        token0_decimals = self.pool_config['token0']['decimals']
        token1_decimals = self.pool_config['token1']['decimals']

        # Calculate price bounds from percentages
        price_lower = self.initial_price * (1.0 - default_lower_pct)
        price_upper = self.initial_price * (1.0 + default_upper_pct)

        # Convert prices to ticks
        if self.invert_price:
            tick_lower = price_to_tick(price_lower, token0_decimals, token1_decimals)
            tick_upper = price_to_tick(price_upper, token0_decimals, token1_decimals)
        else:
            tick_lower = price_to_tick(1.0 / price_upper, token0_decimals, token1_decimals)
            tick_upper = price_to_tick(1.0 / price_lower, token0_decimals, token1_decimals)

        # Snap to valid tick spacing
        tick_lower = (tick_lower // tick_spacing) * tick_spacing
        tick_upper = (tick_upper // tick_spacing) * tick_spacing

        if tick_upper <= tick_lower:
            tick_upper = tick_lower + tick_spacing

        new_min = tick_to_price(tick_lower, token0_decimals, token1_decimals)
        new_max = tick_to_price(tick_upper, token0_decimals, token1_decimals)

        if self.invert_price:
            self.position_min_price = new_min
            self.position_max_price = new_max
        else:
            self.position_min_price = 1.0 / new_max if new_max > 0 else self.initial_price * 0.9
            self.position_max_price = 1.0 / new_min if new_min > 0 else self.initial_price * 1.1

        self.tick_lower = tick_lower
        self.tick_upper = tick_upper
        self._initial_range_set = False  # Flag to track if AI range was set

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
        self.prev_unrealized_il = 0.0  # Previous step's unrealized IL (for delta)
        self.cumulative_il_delta = 0.0  # Sum of IL deltas (IL-based penalty)
        self.cumulative_lvr = 0.0  # Sum of LVR penalties
        self.cumulative_gas = 0.0
        self.time_since_last_rebalance = 0
        self.total_rebalances = 0
        self.current_position_value = self.initial_investment  # Start with full investment
        self.position_investment = self.initial_investment  # Investment at position creation (for IL calc)

        # Initialize HODL portfolio (50:50 split at initial price)
        # HODL = 초기 투자금을 50:50으로 나눠서 토큰 보유
        # token0 = stablecoin (USDC), token1 = ETH (when invert_price=True)
        half_investment = self.initial_investment / 2
        self.hodl_token0 = half_investment  # $5000 worth of token0
        self.hodl_token1 = half_investment / self.initial_price  # $5000 worth of token1
        self.prev_excess_return = 0.0  # Reset excess return tracking

        # Note: current_tick_width and desired_tick_width are set earlier
        # when initializing the position with default_tick_width

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

        # Update unrealized IL (current snapshot for display)
        self.unrealized_il = il_raw

        # Update cumulative metrics
        self.cumulative_fees += fees_earned
        self.cumulative_gas += base_gas_cost

        # IL is realized when rebalancing (calculated below from actual tokens)
        # Note: cumulative_il is updated in the rebalance block using actual token values

        if did_rebalance:
            self.time_since_last_rebalance = 0
            self.total_rebalances += 1

            # CRITICAL: Get ACTUAL tokens from current liquidity position
            # This is the real LP value (IL is naturally included in token amounts)
            current_price = current_data['close']

            # Step 1: Get ACTUAL tokens from liquidity (NOT from running total!)
            # get_token_amounts returns real token amounts based on L and current price
            old_tokens = get_token_amounts(
                int(self.liquidity),
                current_price,
                prev_min,  # Old ranges
                prev_max,
                self.pool_config['token0']['decimals'],
                self.pool_config['token1']['decimals']
            )

            # Step 2: Calculate ACTUAL LP value from tokens
            # token0 = WETH count, token1 = USDT value (already USD)
            # This value naturally includes IL (lower than HODL when price moved)
            actual_lp_value = (old_tokens[0] * current_price) + old_tokens[1]

            # Step 3: Calculate IL for this position (for tracking)
            # IL = LP value - HODL value (negative = loss)
            # HODL = initial tokens at current price
            hodl_value = (self.initial_token_amounts[0] * current_price) + self.initial_token_amounts[1]
            realized_il = hodl_value - actual_lp_value  # Positive = loss
            self.cumulative_il += max(0, realized_il)  # Only track losses

            # Step 4: Apply gas cost and add this step's fees
            # The actual value going into new position
            position_value_after_costs = actual_lp_value - base_gas_cost + fees_earned
            position_value_after_costs = max(0.0, position_value_after_costs)

            # Step 5: Calculate required tokens for new range
            new_tokens_needed = self._calculate_tokens_for_range(
                position_value_after_costs,
                current_price,
                self.position_min_price,  # New ranges
                self.position_max_price
            )

            # Step 6: Calculate SWAP cost
            swap_cost = self._calculate_swap_cost(
                old_tokens,
                new_tokens_needed,
                current_price
            )

            # Add swap cost to cumulative gas
            self.cumulative_gas += swap_cost

            # Final position value for new position
            final_position_value = position_value_after_costs - swap_cost
            final_position_value = max(0.0, final_position_value)

            # Recalculate tokens after swap cost deduction
            new_tokens_needed = self._calculate_tokens_for_range(
                final_position_value,
                current_price,
                self.position_min_price,
                self.position_max_price
            )

            # Recalculate liquidity for new position
            self.liquidity = liquidity_for_strategy(
                current_price,
                self.position_min_price,
                self.position_max_price,
                new_tokens_needed[0],
                new_tokens_needed[1],
                self.pool_config['token0']['decimals'],
                self.pool_config['token1']['decimals']
            )

            # CRITICAL: Reset for new position
            self.initial_token_amounts = new_tokens_needed
            self.mint_price = current_price
            self.position_investment = final_position_value
            self.current_position_value = final_position_value
        else:
            self.time_since_last_rebalance += 1
            # No rebalance: position value unchanged (IL unrealized, tracked separately)

        # Total gas cost for this step (base gas + swap cost)
        total_gas_cost = base_gas_cost + swap_cost

        # =================================================================
        # IL DELTA CALCULATION (User Request: Use IL instead of LVR)
        # IL is recalculated every step, and the DELTA is used as penalty
        # CRITICAL: When rebalancing, IL is REALIZED (locked in as real loss)
        # =================================================================
        current_price = current_data['close']
        current_tick = int(current_data.get('tick', 0))

        # Check if position is In-Range
        in_range = self.tick_lower <= current_tick <= self.tick_upper

        # =================================================================
        # IL DELTA CALCULATION (v8 fix: no double-counting)
        #
        # IL is penalized incrementally every step as il_delta.
        # When rebalancing, we DON'T apply full IL again (that would double-count).
        # Instead, we just apply the final delta for this step.
        #
        # The cumulative IL penalty = sum of all il_deltas over time
        # This equals the total IL experienced, without double-counting.
        # =================================================================

        # Calculate this step's IL change (same logic for both cases)
        il_delta = self.unrealized_il - self.prev_unrealized_il

        # Track cumulative IL delta
        self.cumulative_il_delta += il_delta

        if did_rebalance:
            # Reset prev_unrealized_il to 0 for next step
            # (new position starts with IL = 0 since mint_price = current_price)
            self.prev_unrealized_il = 0.0
        else:
            # Update prev_unrealized_il for next step
            self.prev_unrealized_il = self.unrealized_il

        # Calculate RETURNS volatility for info (keep for debugging)
        lookback = min(24, current_idx)
        if lookback < 3:
            lookback = 3
        recent_prices = self.features_df.iloc[current_idx-lookback:current_idx+1]['close'].values
        returns = np.diff(recent_prices) / recent_prices[:-1]
        returns_volatility = np.std(returns)
        if returns_volatility <= 0 or np.isnan(returns_volatility):
            returns_volatility = 0.002

        # LVR calculation (kept for comparison in info dict, not used in reward)
        if in_range:
            lvr = self._calculate_lvr(current_price, returns_volatility)
        else:
            lvr = 0.0

        # Track cumulative LVR (for info/comparison only)
        self.cumulative_lvr += lvr

        # =================================================================
        # REWARD CALCULATION: LP vs HODL Excess Return
        # - LP 수익률 vs HODL 수익률 비교
        # - 시장 방향성 제거 (상승장/하락장 무관)
        # - LP가 HODL보다 좋으면 +, 나쁘면 -
        # =================================================================
        reward = self._calculate_reward_excess_return(
            fees_earned, current_data['close'], did_rebalance, total_gas_cost
        )

        # Check episode termination
        self.current_step += 1

        # Gymnasium uses separate terminated/truncated flags
        position_failed = self._is_position_failed(current_data['close'])
        time_limit_reached = self.current_step >= self.episode_length

        terminated = position_failed  # Episode ended naturally (failure)
        truncated = time_limit_reached and not position_failed  # Episode cut short by time limit

        # Calculate position quality for info dict
        range_center = (self.position_min_price + self.position_max_price) / 2
        range_half = (self.position_max_price - self.position_min_price) / 2
        if range_half > 0 and self.position_min_price <= current_data['close'] <= self.position_max_price:
            distance = abs(current_data['close'] - range_center) / range_half
            position_quality = 1.0 - distance
        elif range_half > 0:
            if current_data['close'] < self.position_min_price:
                overshoot = (self.position_min_price - current_data['close']) / range_half
            else:
                overshoot = (current_data['close'] - self.position_max_price) / range_half
            position_quality = -min(overshoot, 2.0)
        else:
            position_quality = 0.0

        # Info dict
        lvr_weight = self.reward_weights.get('lvr_weight', 1.0)
        il_weight = self.reward_weights.get('il_weight', 0.05)  # Updated default for reactive reward
        info = {
            'fees': fees_earned,
            'il_delta': il_delta,  # Change in IL this step
            'lvr': lvr,  # LVR (kept for comparison)
            'lvr_weight': lvr_weight,
            'il': il_raw,  # Unrealized IL at this step
            'realized_il': self.cumulative_il,
            'unrealized_il': self.unrealized_il,
            'cumulative_il_delta': self.cumulative_il_delta,
            'cumulative_lvr': self.cumulative_lvr,
            'volatility': returns_volatility,
            'gas': total_gas_cost,
            'swap_cost': swap_cost,
            'rebalanced': did_rebalance,
            'total_rebalances': self.total_rebalances,
            'cumulative_fees': self.cumulative_fees,
            'cumulative_il': self.cumulative_il,
            'cumulative_gas': self.cumulative_gas,
            'net_return': self.cumulative_fees - self.cumulative_il_delta - self.cumulative_gas,
            'reward_net_return': self.cumulative_fees - il_weight * self.cumulative_il_delta,
            'current_position_value': self.current_position_value,
            'in_range': self.tick_lower <= int(current_data.get('tick', 0)) <= self.tick_upper,
            # Position quality metrics
            'position_quality': position_quality,  # 1=중앙, 0=경계, <0=이탈
            'range_width_pct': 2 * range_half / current_data['close'] if current_data['close'] > 0 else 0,
            # LP vs HODL excess return metrics
            'hodl_value': self._get_hodl_value(current_data['close']),
            'lp_value': self._get_lp_value(current_data['close']),  # 토큰 가격 반영된 LP 가치
            'excess_return': self._get_excess_return(current_data['close']),  # LP - HODL 수익률
            # Auto-rebalancing metrics (3D action space)
            'edge_proximity': self._calculate_edge_proximity(current_data['close']),
            'current_lower_pct': self.current_lower_pct,
            'current_upper_pct': self.current_upper_pct,
            'desired_lower_pct': self.desired_lower_pct,
            'desired_upper_pct': self.desired_upper_pct,
            'rebalance_signal': self.rebalance_signal,
            # Price info for visualization
            'current_price': current_data['close'],
            'position_min_price': self.position_min_price,
            'position_max_price': self.position_max_price,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _apply_action(self, action: np.ndarray, debug: bool = False) -> bool:
        """
        Apply continuous action with AUTO-REBALANCING based on IL graph analysis.

        **Continuous Action Space**:
        - action[0] = lower_pct: 현재가 대비 하한선 비율 (1% ~ 60%)
        - action[1] = upper_pct: 현재가 대비 상한선 비율 (1% ~ 60%)
        - 예: action=[0.10, 0.20] → 현재가 -10% ~ +20% 범위 (비대칭)

        **Auto-Rebalancing Logic** (see docs/il_graph.png):
        - Model selects desired range bounds
        - Actual rebalancing only occurs when:
          1. No position exists (first step)
          2. Edge proximity > threshold (IL acceleration zone)
          3. Range bounds changed significantly AND price near edge

        Args:
            action: numpy array [lower_pct, upper_pct]
            debug: If True, print debug information

        Returns:
            True if position was rebalanced
        """
        # Extract and clip action values (3D Action Space)
        # action[0] = lower_pct: 1%~99% (가격 > 0 제약)
        # action[1] = upper_pct: 1%~500% (실용적 상한)
        # action[2] = rebalance_signal: 0~1 (>0.5 시 즉시 리밸런싱)
        lower_pct = float(np.clip(action[0], 0.01, 0.99))
        upper_pct = float(np.clip(action[1], 0.01, 5.00))
        rebalance_signal = float(np.clip(action[2], 0.0, 1.0))

        # Store model's preference
        self.desired_lower_pct = lower_pct
        self.desired_upper_pct = upper_pct
        self.rebalance_signal = rebalance_signal

        # Get current price
        current_idx = self.episode_start_idx + self.current_step
        current_price = self.features_df.iloc[current_idx]['close']

        if current_price <= 0 or np.isnan(current_price):
            return False

        # Calculate edge proximity (how close to boundary)
        edge_proximity = self._calculate_edge_proximity(current_price)

        # === ASYMMETRIC AUTO-REBALANCING DECISION ===
        # Based on IL asymmetry analysis (docs/updownildifference.png):
        # - Price DOWN (→ lower bound): IL is ~1.8x more sensitive
        # - Price UP (→ upper bound): IL is gentler

        no_position = self.position_min_price <= 0

        # Check if bounds changed significantly (>5% difference)
        bounds_changed = (
            abs(lower_pct - getattr(self, 'current_lower_pct', 0)) > 0.05 or
            abs(upper_pct - getattr(self, 'current_upper_pct', 0)) > 0.05
        )

        # Determine direction and apply asymmetric threshold
        if no_position:
            in_danger_zone = True  # Force initial position
            direction = "first"
        else:
            center = (self.position_min_price + self.position_max_price) / 2
            if current_price < center:
                # Approaching LOWER bound → more dangerous (IL ~1.8x more sensitive)
                threshold = 0.65  # Earlier trigger (65%)
                direction = "down"
            else:
                # Approaching UPPER bound → less dangerous
                threshold = 0.80  # Later trigger (80%)
                direction = "up"
            in_danger_zone = edge_proximity > threshold

        in_warning_zone = edge_proximity > 0.5  # > 50%

        # 모델의 rebalance_signal이 0.5 초과하면 즉시 리밸런싱
        model_wants_rebalance = rebalance_signal > 0.5  # As documented: > 0.5 triggers rebalancing

        # 자동 리밸런싱 제거 - 모델이 직접 결정
        need_rebalance = (
            no_position or                          # First position only
            model_wants_rebalance                   # Model explicitly requests (signal > 0.9)
        )

        if not need_rebalance:
            if debug:
                print(f"[NoRebalance] Step {self.current_step}: edge={edge_proximity:.1%}, "
                      f"signal={rebalance_signal:.2f}, lower={lower_pct:.1%}, upper={upper_pct:.1%}")
            return False

        # === EXECUTE REBALANCING ===
        fee_tier = self.pool_config['feeTier']
        tick_spacing = TICK_SPACINGS.get(fee_tier, 60)
        token0_decimals = self.pool_config['token0']['decimals']
        token1_decimals = self.pool_config['token1']['decimals']

        # Calculate asymmetric price bounds
        new_min_price = current_price * (1.0 - lower_pct)
        new_max_price = current_price * (1.0 + upper_pct)

        # Convert prices to ticks
        if self.invert_price:
            new_tick_lower = price_to_tick(new_min_price, token0_decimals, token1_decimals)
            new_tick_upper = price_to_tick(new_max_price, token0_decimals, token1_decimals)
        else:
            new_tick_lower = price_to_tick(1.0 / new_max_price, token0_decimals, token1_decimals)
            new_tick_upper = price_to_tick(1.0 / new_min_price, token0_decimals, token1_decimals)

        # Snap to valid tick spacing
        new_tick_lower = (new_tick_lower // tick_spacing) * tick_spacing
        new_tick_upper = (new_tick_upper // tick_spacing) * tick_spacing

        # Ensure minimum width
        if new_tick_upper <= new_tick_lower:
            new_tick_upper = new_tick_lower + tick_spacing

        # Convert ticks back to prices
        new_min = tick_to_price(new_tick_lower, token0_decimals, token1_decimals)
        new_max = tick_to_price(new_tick_upper, token0_decimals, token1_decimals)

        # Update position prices
        if self.invert_price:
            self.position_min_price = new_min
            self.position_max_price = new_max
        else:
            self.position_min_price = 1.0 / new_max if new_max > 0 else 0
            self.position_max_price = 1.0 / new_min if new_min > 0 else 0

        # Update current bounds
        self.current_lower_pct = lower_pct
        self.current_upper_pct = upper_pct

        if debug:
            if no_position:
                reason = "first"
            elif model_wants_rebalance:
                reason = "signal"
            elif in_danger_zone:
                reason = "danger"
            else:
                reason = "bounds_change"
            print(f"[Rebalance:{reason}] Step {self.current_step}: edge={edge_proximity:.1%}, "
                  f"signal={rebalance_signal:.2f}, lower={lower_pct:.1%}, upper={upper_pct:.1%}")
            print(f"  Price: {current_price:.2f}, Range: [{self.position_min_price:.2f}, {self.position_max_price:.2f}]")

        # Update tick bounds for fee calculation
        self.tick_lower = new_tick_lower
        self.tick_upper = new_tick_upper

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
        # CRITICAL: Use position_investment (fixed at position creation), NOT current_position_value
        result = calculate_il_whitepaper(
            investment=self.position_investment,
            mint_price=self.mint_price,
            current_price=current_price,
            pa=position_min,
            pb=position_max
        )

        # IL is negative when LP < HODL (loss)
        # We return positive value as loss
        # Use il_usd directly from result (LP - HODL in USD)
        il_usd = result.get('il_usd', 0.0)

        # IL is typically negative (LP < HODL), so take abs for "loss" amount
        il_usd = abs(il_usd)

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

    def _calculate_reward_excess_return(self, fees: float, current_price: float,
                                        did_rebalance: bool, gas: float) -> float:
        """
        LP vs HODL 초과수익률 기반 보상 함수.

        핵심 원리:
        - LP 수익률 = (LP가치 + 누적수수료 - 초기투자) / 초기투자
        - HODL 수익률 = (HODL가치 - 초기투자) / 초기투자
        - 초과수익률 = LP 수익률 - HODL 수익률

        장점:
        - 시장 방향성 제거 (상승장/하락장 무관)
        - LP의 존재 이유 (HODL 대비 수익)를 직접 평가
        - fees가 IL을 상쇄하는지 자동 반영

        Args:
            fees: Fees earned this step (USD)
            current_price: Current market price
            did_rebalance: Whether a rebalance happened this step
            gas: Gas cost if rebalancing occurred (USD)

        Returns:
            Reward value (excess return delta)
        """
        # Sanitize inputs
        fees = 0.0 if np.isnan(fees) else fees
        gas = 0.0 if np.isnan(gas) else gas
        current_price = self.initial_price if (np.isnan(current_price) or current_price <= 0) else current_price

        # === 1. HODL 포트폴리오 현재 가치 ===
        # HODL = 초기 50:50으로 나눈 토큰을 그대로 보유
        # token0 = stablecoin (가치 불변), token1 = ETH (가격 변동)
        hodl_value = self.hodl_token0 + (self.hodl_token1 * current_price)

        # === 2. LP 포트폴리오 현재 가치 ===
        # _get_lp_value 사용: V3 유동성 공식으로 정확한 토큰 가치 계산
        # (current_position_value는 근사값이므로 사용하지 않음)
        lp_value = self._get_lp_value(current_price)

        # === 3. 수익률 계산 ===
        if self.initial_investment > 0:
            lp_return = (lp_value - self.initial_investment) / self.initial_investment
            hodl_return = (hodl_value - self.initial_investment) / self.initial_investment
            excess_return = lp_return - hodl_return
        else:
            excess_return = 0.0

        # === 4. 보상 = 초과수익률의 변화 (delta) ===
        # 매 스텝 초과수익률의 "변화량"을 보상으로 사용
        # 초과수익률이 증가하면 양수 보상, 감소하면 음수 보상
        excess_return_delta = excess_return - self.prev_excess_return
        self.prev_excess_return = excess_return

        # 스케일링: 수익률 변화를 USD 단위로 변환
        # 1% 초과수익률 변화 = $100 (투자금 $10,000 기준)
        reward = excess_return_delta * self.initial_investment

        # === 5. 가스/수수료 처리 ===
        # 가스: position_value에 이미 반영되어 excess_return에 자동 포함
        # 수수료: cumulative_fees → lp_value에 이미 반영되어 excess_return에 자동 포함
        # 추가 보너스/패널티 없음 (이중 계산 방지)

        # Note: 클리핑 제거 - 실제 excess return을 그대로 반영

        # Replace NaN reward with 0
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        return reward

    def _calculate_edge_proximity(self, current_price: float) -> float:
        """
        Calculate maximum edge proximity (for rebalancing decision).
        Returns how close the price is to ANY boundary.

        Returns:
            0.0 = at center
            1.0 = at boundary (lower or upper)
            >1.0 = out of range
        """
        lower = self._calculate_edge_proximity_lower(current_price)
        upper = self._calculate_edge_proximity_upper(current_price)
        # Return max deviation from center (0.5)
        # At center: lower=0.5, upper=0.5 → max(|0.5-0.5|, |0.5-0.5|)*2 = 0
        # At lower: lower=1.0, upper=0.0 → max(|1-0.5|, |0-0.5|)*2 = 1
        return max(abs(lower - 0.5), abs(upper - 0.5)) * 2

    def _calculate_edge_proximity_lower(self, current_price: float) -> float:
        """
        Calculate proximity to lower boundary.

        Returns:
            0.0 = at upper boundary
            0.5 = at center
            1.0 = at lower boundary
            >1.0 = below lower boundary (out of range)
        """
        if self.position_min_price <= 0 or self.position_max_price <= 0:
            return 0.5

        range_width = self.position_max_price - self.position_min_price
        if range_width <= 0:
            return 0.5

        # Distance from upper boundary, normalized
        distance_from_upper = self.position_max_price - current_price
        proximity = distance_from_upper / range_width

        return max(0.0, proximity)  # 0 at upper, 1 at lower, >1 below lower

    def _calculate_edge_proximity_upper(self, current_price: float) -> float:
        """
        Calculate proximity to upper boundary.

        Returns:
            0.0 = at lower boundary
            0.5 = at center
            1.0 = at upper boundary
            >1.0 = above upper boundary (out of range)
        """
        if self.position_min_price <= 0 or self.position_max_price <= 0:
            return 0.5

        range_width = self.position_max_price - self.position_min_price
        if range_width <= 0:
            return 0.5

        # Distance from lower boundary, normalized
        distance_from_lower = current_price - self.position_min_price
        proximity = distance_from_lower / range_width

        return max(0.0, proximity)  # 0 at lower, 1 at upper, >1 above upper

    def _get_hodl_value(self, current_price: float) -> float:
        """현재 가격에서 HODL 포트폴리오 가치 계산"""
        return self.hodl_token0 + (self.hodl_token1 * current_price)

    def _get_lp_value(self, current_price: float) -> float:
        """
        현재 가격에서 LP 포지션 가치 계산.

        LP value = 현재 토큰 × 현재 가격 + 누적 수수료 - 누적 가스비

        토큰 개수는 유동성 L과 현재 가격으로 V3 공식을 사용해 계산.
        이 방식으로 매 step마다 가격 변동이 LP 가치에 반영됨.

        Args:
            current_price: 현재 시장 가격

        Returns:
            LP 포지션 총 가치 (USD)
        """
        if self.liquidity <= 0:
            # 유동성 0 = 포지션 없음. 실제 남은 가치 반환 (누적 수수료 - 누적 가스비)
            # 주의: 포지션이 청산된 경우 current_position_value가 실제 남은 자금
            remaining_value = self.current_position_value + self.cumulative_fees - self.cumulative_gas
            return max(0.0, remaining_value)

        # 현재 유동성과 가격으로 토큰 수량 계산
        tokens = get_token_amounts(
            int(self.liquidity),
            current_price,
            self.position_min_price,
            self.position_max_price,
            self.pool_config['token0']['decimals'],
            self.pool_config['token1']['decimals']
        )

        # LP 포지션 가치 계산
        # tokens[0] = token0 (WETH) → 가격 곱해서 USD로 변환
        # tokens[1] = token1 (USDT) → 이미 USD
        # 가격 = USDT/WETH (예: 3000)
        if self.invert_price:
            # token0=WETH, token1=USDT, price=USDT/WETH
            position_value = (tokens[0] * current_price) + tokens[1]
        else:
            # token0=USDT, token1=WETH, price=USDT/WETH
            position_value = tokens[0] + (tokens[1] * current_price)

        # 총 LP 가치 = 포지션 가치 + 누적 수수료 - 누적 가스비
        lp_value = position_value + self.cumulative_fees - self.cumulative_gas

        return max(0.0, lp_value)  # 음수 방지

    def _get_excess_return(self, current_price: float) -> float:
        """현재 LP vs HODL 초과수익률 계산"""
        hodl_value = self._get_hodl_value(current_price)
        lp_value = self._get_lp_value(current_price)  # 가격 변동 반영된 LP 가치

        if self.initial_investment > 0:
            lp_return = (lp_value - self.initial_investment) / self.initial_investment
            hodl_return = (hodl_value - self.initial_investment) / self.initial_investment
            return lp_return - hodl_return
        return 0.0

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
        # split_tokens returns: (token0_count, token1_usd_value)
        # For WETH/USDT pool: token0=WETH (count), token1=USDT (USD value)
        delta0 = needed_tokens[0] - current_tokens[0]  # WETH count change
        delta1 = needed_tokens[1] - current_tokens[1]  # USDT value change (already USD)

        price_human = price  # USDT per WETH

        # Determine which token needs to be swapped
        if delta0 > 0 and delta1 < 0:
            # Need more WETH (token0) → sell USDT (token1) to get WETH
            swap_amount_usd = abs(delta1)  # delta1 is already USD
        elif delta0 < 0 and delta1 > 0:
            # Need more USDT (token1) → sell WETH (token0) to get USDT
            swap_amount_usd = abs(delta0) * price_human  # WETH count * price = USD
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
        Paper-style observation (arXiv:2501.07508, Section 5).

        State Space:
        1. Market price p_t (normalized)
        2. Tick index i (normalized)
        3. Interval width w_t (current tick width / tick_spacing)
        4. Current liquidity level L_t (normalized)
        5. Exponentially weighted volatility σ_t
        6. 24-hour moving average
        7. 168-hour moving average
        8-11. Technical indicators: BB, ADXR, BOP, DX

        Returns:
            numpy array with 11 features
        """
        current_idx = self.episode_start_idx + self.current_step
        current_data = self.features_df.iloc[current_idx]
        current_price = current_data['close']

        # Get tick spacing
        fee_tier = self.pool_config['feeTier']
        tick_spacing = TICK_SPACINGS.get(fee_tier, 60)

        # 1. Market price (normalized by initial price)
        f1 = current_price / self.initial_price if self.initial_price > 0 else 1.0

        # 2. Tick index (normalized)
        token0_dec = self.pool_config['token0']['decimals']
        token1_dec = self.pool_config['token1']['decimals']
        if self.invert_price:
            current_tick = price_to_tick(current_price, token0_dec, token1_dec)
        else:
            current_tick = price_to_tick(1.0 / current_price, token0_dec, token1_dec) if current_price > 0 else 0
        f2 = current_tick / 100000  # Normalize tick index

        # 3. Interval width (current width in tick_spacing units)
        interval_width = (self.tick_upper - self.tick_lower) / tick_spacing if tick_spacing > 0 else 0
        f3 = interval_width / 100  # Normalize (100 tick_spacings = 1.0)

        # 4. Current liquidity level (normalized)
        f4 = np.log1p(self.backtest_L) / 50 if hasattr(self, 'backtest_L') and self.backtest_L > 0 else 0

        # 5. 7-day volatility (이미 수익률의 표준편차이므로 그대로 사용)
        # 일반적으로 0.005~0.03 범위 (0.5%~3% daily std)
        f5 = current_data['volatility_7d']

        # 6. 30-day volatility
        f6 = current_data['volatility_30d']

        # 7. Price / 7-day MA (>1 = above MA, <1 = below MA)
        f7 = current_data['price_to_ma7d']

        # 8. Price / 30-day MA (>1 = above MA, <1 = below MA)
        f8 = current_data['price_to_ma30d']

        # 9. 7-day momentum (7일 수익률)
        f9 = current_data['momentum_7d']

        # 10. 30-day momentum (30일 수익률)
        f10 = current_data['momentum_30d']

        # 11. Lower edge proximity (하단 경계 근접도)
        # 0 = 상단, 0.5 = 중앙, 1 = 하단 경계, >1 = 하단 이탈
        f11 = self._calculate_edge_proximity_lower(current_price)

        # 12. Upper edge proximity (상단 경계 근접도)
        # 0 = 하단, 0.5 = 중앙, 1 = 상단 경계, >1 = 상단 이탈
        f12 = self._calculate_edge_proximity_upper(current_price)

        obs = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12], dtype=np.float32)

        # Replace NaN and inf values
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
