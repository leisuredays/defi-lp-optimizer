"""
Backtest API Endpoint

Runs model on test data and returns rebalancing history for visualization.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

router = APIRouter()

# Global cache for model and data
_model_cache = {}
_data_cache = {}


class BacktestRequest(BaseModel):
    """Backtest request parameters"""
    pool_id: str  # Pool address (e.g., "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8")
    protocol_id: int  # Protocol ID (0=Arbitrum, 1=Ethereum, etc.)
    model_name: str = "ppo_arbitrum_usdc_weth_005"
    n_episodes: int = 1
    episode_length_days: int = 18  # Number of days to backtest (max 30)
    debug: bool = False  # Enable debug logging


class RangeEvent(BaseModel):
    """Single range/rebalance event"""
    timestamp: int
    hour: int
    price: float
    min_price: float
    max_price: float
    rebalanced: bool
    in_range: bool
    fees_earned: float
    il_incurred: float
    cumulative_fees: float
    cumulative_il: float  # Realized IL (only from rebalancing)
    unrealized_il: float = 0.0  # Current unrealized IL
    position_value: float
    token0_amount: float = 0.0  # Amount of token0 in position
    token1_amount: float = 0.0  # Amount of token1 in position
    token0_value: float = 0.0   # USD value of token0
    token1_value: float = 0.0   # USD value of token1


class EpisodeResult(BaseModel):
    """Single episode result"""
    episode: int
    start_timestamp: int
    end_timestamp: int
    initial_price: float
    final_price: float
    total_fees: float
    total_il: float
    total_gas: float
    net_return: float
    rebalance_count: int
    token0_symbol: str = ""
    token1_symbol: str = ""
    events: List[RangeEvent]


class BacktestResponse(BaseModel):
    """Backtest response with full history"""
    status: str
    pool: str
    model_name: str
    test_period_days: int
    episodes: List[EpisodeResult]
    summary: dict


def load_config():
    """Load training config"""
    config_paths = [
        Path("config/training_config.yaml"),
        Path("backend/config/training_config.yaml")
    ]

    for path in config_paths:
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)

    raise FileNotFoundError("Training config not found")


def load_model(model_name: str):
    """Load PPO model with caching"""
    if model_name in _model_cache:
        return _model_cache[model_name]

    from stable_baselines3 import PPO

    model_paths = [
        Path(f"models/{model_name}.zip"),
        Path(f"backend/models/{model_name}.zip")
    ]

    for path in model_paths:
        if path.exists():
            model = PPO.load(str(path))
            _model_cache[model_name] = model
            return model

    raise FileNotFoundError(f"Model not found: {model_name}")


async def load_live_data(pool_id: str, protocol_id: int, days: int):
    """
    Load live data from The Graph API for backtesting.

    Args:
        pool_id: Pool address
        protocol_id: Protocol identifier
        days: Number of days of historical data (max 30)

    Returns:
        DataFrame with hourly pool data
    """
    from app.core.graph_client import fetch_pool_data_for_optimization

    # Cache key based on pool, protocol, and days
    cache_key = f"{pool_id}_{protocol_id}_{days}"
    if cache_key in _data_cache:
        print(f"[Backtest] Using cached data for {cache_key}")
        return _data_cache[cache_key]

    # Limit to 30 days max
    days = min(days, 30)

    # Add buffer for feature calculation (volatility_24h needs 24 hours history)
    # Fetch extra data to ensure we have enough for episode
    fetch_days = days + 2  # Add 2 extra days for buffer

    print(f"[Backtest] Fetching {fetch_days} days of live data for pool {pool_id} on protocol {protocol_id}")

    # Fetch live data from The Graph
    pool_data = await fetch_pool_data_for_optimization(
        pool_id=pool_id,
        protocol_id=protocol_id,
        days=fetch_days
    )

    if not pool_data['hourly_data'] or len(pool_data['hourly_data']) == 0:
        raise ValueError(f"No hourly data available for pool {pool_id}")

    # Convert to DataFrame
    hourly_data = pool_data['hourly_data']
    pool_info = pool_data['pool_info']

    # Build DataFrame with required columns
    df = pd.DataFrame(hourly_data)

    # Add pool metadata
    df['pool_id'] = pool_id
    df['protocol_id'] = protocol_id
    df['fee_tier'] = int(pool_info.get('feeTier', 500))
    df['token0_symbol'] = pool_info['token0']['symbol']
    df['token1_symbol'] = pool_info['token1']['symbol']
    df['token0_decimals'] = int(pool_info['token0']['decimals'])
    df['token1_decimals'] = int(pool_info['token1']['decimals'])

    # Ensure proper types
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['open'] = df['open'].astype(float)
    df['periodStartUnix'] = df['periodStartUnix'].astype(int)
    df['liquidity'] = df['liquidity'].astype(float)
    df['volumeUSD'] = df.get('volumeUSD', 0).astype(float) if 'volumeUSD' in df.columns else 0

    # Sort by time (oldest first)
    df = df.sort_values('periodStartUnix').reset_index(drop=True)

    # Cache for future requests
    _data_cache[cache_key] = df

    print(f"[Backtest] Loaded {len(df)} hours of live data")

    return df


def run_episode_with_history(model, test_data: pd.DataFrame,
                              episode_length: int = 400,
                              obs_dim: int = 24,
                              debug: bool = False) -> EpisodeResult:
    """
    Run single episode and collect detailed history.

    Args:
        model: Trained PPO model
        test_data: Test DataFrame
        episode_length: Number of hours per episode
        obs_dim: Observation dimension (24 for legacy models, 28 for new)
        debug: Enable debug logging for rebalancing decisions

    Returns:
        EpisodeResult with full event history
    """
    from app.ml.environment import UniswapV3LPEnv
    from app.ml.uniswap_v3_adapter import tokens_for_strategy

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
        episode_length_hours=episode_length,
        reward_weights={'alpha': 1.0, 'beta': 0.8, 'gamma': 0.2},
        obs_dim=obs_dim,
        debug=debug
    )

    obs, init_info = env.reset()

    # STEP 1: Use AI model to predict initial LP range
    initial_action, _ = model.predict(obs, deterministic=True)
    env.set_initial_range_from_action(initial_action)

    # Count initial range setting as first rebalance
    if hasattr(env, 'total_rebalances'):
        env.total_rebalances += 1
    else:
        env.total_rebalances = 1

    # Record initial state (Day 0)
    initial_idx = env.episode_start_idx
    initial_data = env.features_df.iloc[initial_idx]
    initial_price = initial_data['close']
    initial_timestamp = int(initial_data['periodStartUnix'])

    # Calculate initial position value and tokens
    # Price is already in token1/token0 format (USDC/WETH)
    price_human = initial_price
    initial_position_value = 10000.0  # Always start with initial investment

    # Get initial token amounts from environment (already correctly calculated)
    # env.initial_token_amounts is set by set_initial_range_from_action() using tokens_for_strategy
    initial_tokens = env.initial_token_amounts

    # Calculate decimal difference for tokens_for_strategy
    decimal = pool_config['token1']['decimals'] - pool_config['token0']['decimals']

    # For USDC/WETH pool: token0=USDC, token1=WETH, price_human=USDC/WETH
    initial_token0_value = initial_tokens[0]  # USDC (already USD)
    initial_token1_value = initial_tokens[1] * price_human  # WETH * USDC/WETH = USDC

    events = []
    day = 0

    events.append(RangeEvent(
        timestamp=initial_timestamp,
        hour=0,
        price=float(initial_price),
        min_price=float(env.position_min_price),
        max_price=float(env.position_max_price),
        rebalanced=True,  # Initial range setting counts as rebalancing
        in_range=True,
        fees_earned=0.0,
        il_incurred=0.0,
        cumulative_fees=0.0,
        cumulative_il=0.0,
        position_value=float(initial_position_value),
        token0_amount=float(initial_tokens[0]),
        token1_amount=float(initial_tokens[1]),
        token0_value=float(initial_token0_value),
        token1_value=float(initial_token1_value)
    ))

    # STEP 2: Daily simulation loop (24-hour intervals)
    total_hours = episode_length
    hours_per_day = 24
    done = False
    hour = 0

    while hour < total_hours and not done:
        # Simulate 24 hours (or remaining hours if less than 24)
        hours_this_iteration = min(hours_per_day, total_hours - hour)

        # Accumulate metrics over the day
        day_fees = 0.0
        day_il = 0.0
        day_rebalanced = False
        end_of_day_price = initial_price
        end_of_day_in_range = True

        for h in range(hours_this_iteration):
            # For intra-day steps, use no-op action (model only decides daily)
            # But on the last hour of the day, use model prediction
            if h == hours_this_iteration - 1 and hour + hours_this_iteration < total_hours:
                # End of day: Ask model for rebalancing decision
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Intra-day: No action (maintain current range)
                action = np.array([0.0, 0.0, 0.0])  # No rebalancing

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Accumulate daily metrics
            day_fees += info.get('fees', 0)
            day_il += info.get('il', 0)
            day_rebalanced = day_rebalanced or info.get('rebalanced', False)

            # Get end-of-day price
            current_idx = env.episode_start_idx + env.current_step
            if current_idx < len(env.features_df):
                current_data = env.features_df.iloc[current_idx]
                end_of_day_price = current_data['close']
                end_of_day_in_range = env.position_min_price <= end_of_day_price <= env.position_max_price

            if done:
                break

        # Record daily event
        hour += hours_this_iteration
        day += 1

        current_idx = env.episode_start_idx + env.current_step
        if current_idx < len(env.features_df):
            current_data = env.features_df.iloc[current_idx]
            current_timestamp = int(current_data['periodStartUnix'])

            # Calculate position value and token amounts using tokens_for_strategy
            # Use env.current_position_value which is already tracked by the environment
            try:
                # Get actual position value from environment (already accounts for IL, Gas, Fees)
                current_position_value = env.current_position_value

                # Calculate token distribution for current position value
                tokens = tokens_for_strategy(
                    env.position_min_price,
                    env.position_max_price,
                    current_position_value,  # Use actual position value from env
                    end_of_day_price,
                    decimal
                )
                # Price is already in token1/token0 format (USDC/WETH)
                price_human = end_of_day_price
                # For USDC/WETH pool: token0=USDC, token1=WETH
                token0_value = tokens[0]  # USDC (already USD)
                token1_value = tokens[1] * price_human  # WETH * USDC/WETH = USDC
                position_value = token0_value + token1_value

                if debug and day <= 2:  # Log first 2 days
                    print(f"[Backtest] Day {day}: tokens={tokens}, price_human={price_human:.2f}, position_value=${position_value:.2f}")
                    print(f"  IL={env.cumulative_il:.2f}, Gas={env.cumulative_gas:.2f}, Fees={env.cumulative_fees:.2f}")
            except Exception as e:
                if debug:
                    print(f"[Backtest] Position value calculation error: {e}")
                tokens = (0.0, 0.0)
                token0_value = 0.0
                token1_value = 0.0
                position_value = env.current_position_value

            events.append(RangeEvent(
                timestamp=current_timestamp,
                hour=hour,
                price=float(end_of_day_price),
                min_price=float(env.position_min_price),
                max_price=float(env.position_max_price),
                rebalanced=day_rebalanced,
                in_range=end_of_day_in_range,
                fees_earned=float(day_fees),
                il_incurred=float(day_il),
                cumulative_fees=float(env.cumulative_fees),
                cumulative_il=float(env.cumulative_il),
                unrealized_il=float(env.unrealized_il),
                position_value=float(position_value),
                token0_amount=float(tokens[0]),
                token1_amount=float(tokens[1]),
                token0_value=float(token0_value),
                token1_value=float(token1_value)
            ))

    # Final metrics
    final_data = env.features_df.iloc[min(env.episode_start_idx + env.current_step, len(env.features_df) - 1)]

    # Get token symbols from test_data
    token0_symbol = test_data['token0_symbol'].iloc[0] if 'token0_symbol' in test_data.columns else ""
    token1_symbol = test_data['token1_symbol'].iloc[0] if 'token1_symbol' in test_data.columns else ""

    return EpisodeResult(
        episode=0,
        start_timestamp=initial_timestamp,
        end_timestamp=int(final_data['periodStartUnix']),
        initial_price=float(initial_price),
        final_price=float(final_data['close']),
        total_fees=float(env.cumulative_fees),
        total_il=float(env.cumulative_il),
        total_gas=float(env.cumulative_gas),
        net_return=float(env.cumulative_fees - env.cumulative_il - env.cumulative_gas),
        rebalance_count=int(env.total_rebalances),
        token0_symbol=token0_symbol,
        token1_symbol=token1_symbol,
        events=events
    )


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """
    Run model backtest on live data with detailed history.

    Fetches recent historical data from The Graph API and simulates
    AI model performance over the selected period.

    Returns price history, range history, and rebalancing events
    for visualization in frontend charts.
    """
    try:
        print(f"[Backtest] Starting backtest with model: {request.model_name}")
        print(f"[Backtest] Pool: {request.pool_id}, Protocol: {request.protocol_id}")
        print(f"[Backtest] Period: {request.episode_length_days} days")

        # Load model
        model = load_model(request.model_name)
        print(f"[Backtest] Model loaded")

        # Detect model's observation dimension from its policy network
        obs_dim = model.observation_space.shape[0]
        print(f"[Backtest] Model observation dimension: {obs_dim}")

        # Load live data from The Graph API
        test_df = await load_live_data(
            pool_id=request.pool_id,
            protocol_id=request.protocol_id,
            days=request.episode_length_days
        )
        print(f"[Backtest] Live data loaded: {len(test_df)} hours")

        # Convert days to hours
        episode_length_hours = request.episode_length_days * 24

        # Run episodes
        episodes = []
        for ep in range(request.n_episodes):
            print(f"[Backtest] Running episode {ep + 1}/{request.n_episodes}")
            result = run_episode_with_history(
                model, test_df, episode_length_hours, obs_dim=obs_dim, debug=request.debug
            )
            result.episode = ep
            episodes.append(result)

        # Calculate summary
        total_fees = sum(ep.total_fees for ep in episodes)
        total_il = sum(ep.total_il for ep in episodes)
        total_gas = sum(ep.total_gas for ep in episodes)
        total_rebalances = sum(ep.rebalance_count for ep in episodes)
        net_returns = [ep.net_return for ep in episodes]

        summary = {
            'total_episodes': len(episodes),
            'avg_net_return': float(np.mean(net_returns)),
            'std_net_return': float(np.std(net_returns)),
            'avg_fees': float(total_fees / len(episodes)),
            'avg_il': float(total_il / len(episodes)),
            'avg_gas': float(total_gas / len(episodes)),
            'avg_rebalances': float(total_rebalances / len(episodes)),
            'success_rate': float(sum(1 for r in net_returns if r > 0) / len(net_returns)),
            'sharpe_ratio': float(np.mean(net_returns) / (np.std(net_returns) + 1e-8))
        }

        # Token info
        pool_name = f"{test_df['token0_symbol'].iloc[0]}/{test_df['token1_symbol'].iloc[0]}"

        print(f"[Backtest] Complete. Net return: ${summary['avg_net_return']:.2f}")

        return BacktestResponse(
            status="success",
            pool=pool_name,
            model_name=request.model_name,
            test_period_days=int(len(test_df) / 24),
            episodes=episodes,
            summary=summary
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"[Backtest] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/status")
async def backtest_status():
    """Check if backtest data and model are available"""
    try:
        config = load_config()

        # Check model
        model_name = config['output']['model_name']
        model_paths = [
            Path(f"models/{model_name}.zip"),
            Path(f"backend/models/{model_name}.zip")
        ]
        model_exists = any(p.exists() for p in model_paths)

        # Check data
        pool_paths = [
            Path(config['data']['pool_file']),
            Path("backend") / config['data']['pool_file']
        ]
        data_exists = any(p.exists() for p in pool_paths)

        return {
            "status": "ready" if (model_exists and data_exists) else "not_ready",
            "model_available": model_exists,
            "data_available": data_exists,
            "model_name": model_name
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
