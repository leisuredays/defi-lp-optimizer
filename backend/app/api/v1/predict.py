"""
AI Prediction Endpoint

Uses trained PPO model to predict optimal LP ranges for Uniswap V3 pools.
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

from app.api.schemas import (
    PredictRequest,
    PredictResponse,
    RangeRecommendation,
    ErrorResponse
)
from app.core.graph_client import fetch_pool_data_for_optimization, fetch_pool_info
from app.config import settings
from app.ml.uniswap_v3_adapter import get_tick_from_price, tick_to_price as get_price_from_tick, action_to_range

router = APIRouter()

# Global model cache (loaded on first request)
_model_cache = {}


def _get_model_metadata(model) -> dict:
    """
    Extract metadata from a loaded PPO model (V2).

    Returns:
        dict with observation_dim, action_dim
    """
    obs_dim = model.observation_space.shape[0]
    action_dim = model.action_space.shape[0]

    return {
        "observation_dim": obs_dim,
        "action_dim": action_dim
    }


def _load_model(model_name: str):
    """
    Load PPO model from disk (with caching).

    Returns:
        dict with 'model' and 'metadata' keys
    """
    if model_name in _model_cache:
        print(f"[Predict] Using cached model: {model_name}")
        return _model_cache[model_name]

    # Try multiple possible paths
    possible_paths = [
        Path(f"models/{model_name}.zip"),
        Path(f"backend/models/{model_name}.zip"),
        Path(f"{model_name}.zip")
    ]

    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break

    if not model_path:
        raise FileNotFoundError(f"Model not found: {model_name}. Searched paths: {[str(p) for p in possible_paths]}")

    print(f"[Predict] Loading model from: {model_path}")

    try:
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path))

        # Extract metadata
        metadata = _get_model_metadata(model)

        # Validate observation dimension (V2 expects 24-dim)
        if metadata['observation_dim'] != 24:
            raise ValueError(
                f"Unsupported observation dimension: {metadata['observation_dim']}. "
                f"Expected 24-dim (V2)"
            )

        # Cache as dict
        model_data = {
            'model': model,
            'metadata': metadata
        }
        _model_cache[model_name] = model_data

        print(f"[Predict] Model loaded successfully: {model_name} "
              f"[{metadata['observation_dim']}-dim, {metadata['action_dim']} actions]")

        return model_data

    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")


def _extract_state_features(pool_data: dict, investment: float) -> np.ndarray:
    """
    Extract 24-dimensional state features (V2).

    State features (24 dims):
    - Market features (10): normalized price, volatility, momentum, TVL, volume, liquidity, fee tier, time
    - Position features (9): tick distances, in-range, time since rebalance, fee ROI, IL rate, gas, rebalances
    - Forward-looking (5): expected volatility, fee growth rate, rolling metrics, z-score
    """
    hourly_data = pool_data['hourly_data']
    pool_info = pool_data['pool_info']

    if not hourly_data or len(hourly_data) == 0:
        raise ValueError("No hourly data available")

    # Get recent prices
    recent_prices = [float(h['close']) for h in hourly_data[:168]]  # Last 7 days
    if len(recent_prices) == 0:
        raise ValueError("No price data available")

    current_price = recent_prices[0]
    initial_price = current_price  # At prediction start, normalized to 1.0
    price_24h_ago = recent_prices[23] if len(recent_prices) >= 24 else current_price
    price_24h = recent_prices[:24] if len(recent_prices) >= 24 else recent_prices

    # --- Market Features (f1-f10) ---

    # f1: Normalized price (1.0 at start)
    f1 = current_price / initial_price

    # f2: Relative volatility (std / price)
    volatility_abs = np.std(recent_prices) if len(recent_prices) > 1 else current_price * 0.01
    f2 = volatility_abs / current_price if current_price > 0 else 0.01

    # f3: 1h momentum (%)
    price_1h_ago = recent_prices[1] if len(recent_prices) >= 2 else current_price
    f3 = (current_price - price_1h_ago) / price_1h_ago if price_1h_ago > 0 else 0.0

    # f4: 24h momentum (%)
    f4 = (current_price - price_24h_ago) / price_24h_ago if price_24h_ago > 0 else 0.0

    # f5: log(TVL) / 20 (normalized)
    tvl = float(pool_info.get('totalValueLockedUSD', 1000000))
    f5 = np.log(max(tvl, 1.0)) / 20.0

    # f6: log(volume_24h) / 15 (normalized)
    volume_24h = sum([float(h.get('volumeUSD', 0)) for h in hourly_data[:24]]) if len(hourly_data) >= 24 else 0
    f6 = np.log(max(volume_24h, 1.0)) / 15.0

    # f7: Liquidity / 1M (normalized)
    pool_liquidity = float(pool_info.get('liquidity', 1000000))
    f7 = pool_liquidity / 1_000_000.0

    # f8: Fee tier / 10000 (normalized)
    fee_tier = int(pool_info.get('feeTier', 500))
    f8 = fee_tier / 10000.0

    # f9: sin(hour of day)
    hour = datetime.now().hour
    f9 = np.sin(2 * np.pi * hour / 24.0)

    # f10: sin(day of week)
    weekday = datetime.now().weekday()
    f10 = np.sin(2 * np.pi * weekday / 7.0)

    # --- Position Features (f11-f19) - Initialize with defaults ---

    # f11-f13: Tick distances and width (default: ±1σ range)
    f11 = -1.0  # min tick distance (normalized)
    f12 = 1.0   # max tick distance (normalized)
    f13 = 2.0   # tick width (normalized)

    # f14: In range (1.0 = yes, 0.0 = no)
    f14 = 1.0

    # f15: Time since last rebalance (hours, normalized by 100)
    f15 = 0.0  # Start of episode

    # f16: Fee ROI (cumulative fees / investment)
    f16 = 0.0

    # f17: IL rate (recent IL as % of position value)
    f17 = 0.0

    # f18: Avg gas cost per rebalance (normalized by investment)
    f18 = 0.0

    # f19: Rebalance count (normalized by 10)
    f19 = 0.0

    # --- Forward-looking Features (f20-f24) ---

    # f20: Expected volatility (= current volatility)
    f20 = f2

    # f21: Fee growth rate (volume / liquidity * fee_tier)
    if pool_liquidity > 0:
        f21 = (volume_24h / pool_liquidity) * (fee_tier / 1_000_000.0)
    else:
        f21 = 0.0

    # f22: Rolling volatility (24h average)
    if len(price_24h) > 1:
        rolling_volatilities = []
        for i in range(len(price_24h) - 1):
            window = price_24h[i:i+2]
            if len(window) > 1:
                rolling_volatilities.append(np.std(window) / np.mean(window))
        f22 = np.mean(rolling_volatilities) if rolling_volatilities else f2
    else:
        f22 = f2

    # f23: Rolling fee rate (24h average)
    if len(hourly_data) >= 24:
        rolling_volumes = [float(h.get('volumeUSD', 0)) for h in hourly_data[:24]]
        avg_volume = np.mean(rolling_volumes) if rolling_volumes else 0
        f23 = (avg_volume / pool_liquidity) * (fee_tier / 1_000_000.0) if pool_liquidity > 0 else 0.0
    else:
        f23 = f21

    # f24: Volatility z-score (current vs. rolling mean/std)
    if len(price_24h) > 2:
        rolling_mean = np.mean([v for v in price_24h])
        rolling_std = np.std([v for v in price_24h])
        if rolling_std > 0:
            f24 = (current_price - rolling_mean) / rolling_std
        else:
            f24 = 0.0
    else:
        f24 = 0.0

    # Construct 24-dimensional state vector
    state = np.array([
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,      # Market (10)
        f11, f12, f13, f14, f15, f16, f17, f18, f19,  # Position (9)
        f20, f21, f22, f23, f24                        # Forward-looking (5)
    ], dtype=np.float32)

    # Handle NaN/Inf values
    state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)

    return state


@router.post("/predict", response_model=PredictResponse)
async def predict_optimal_range(request: PredictRequest):
    """
    Use trained PPO model (V2) to predict optimal LP range for a pool.

    Flow:
    1. Fetch pool data from The Graph
    2. Extract 24-dimensional state features (V2)
    3. Load PPO model
    4. Run inference to get action
    5. Convert action to price range using log-scale mapping
    6. Estimate expected performance
    7. Return prediction

    Args:
        request: PredictRequest with pool_id, protocol_id, model_name

    Returns:
        PredictResponse with predicted range and expected performance
    """
    try:
        # Step 1: Fetch pool data
        print(f"[Predict] Fetching data for pool {request.pool_id} on protocol {request.protocol_id}")

        pool_data = await fetch_pool_data_for_optimization(
            pool_id=request.pool_id,
            protocol_id=request.protocol_id,
            days=7  # 7 days for feature extraction
        )

        if not pool_data['pool_info']:
            raise HTTPException(
                status_code=404,
                detail=f"Pool {request.pool_id} not found on protocol {request.protocol_id}"
            )

        if not pool_data['hourly_data'] or len(pool_data['hourly_data']) < 24:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data. Need at least 24 hours, got {len(pool_data.get('hourly_data', []))}"
            )

        pool_info = pool_data['pool_info']
        hourly_data = pool_data['hourly_data']

        print(f"[Predict] Fetched {len(hourly_data)} hours of data")

        # Step 2: Load model
        print(f"[Predict] Loading model: {request.model_name}")
        model_data = _load_model(request.model_name)
        model = model_data['model']
        metadata = model_data['metadata']

        print(f"[Predict] Model loaded: {metadata['observation_dim']}-dim")

        # Step 3: Extract state features (24-dim V2)
        print(f"[Predict] Extracting state features")
        state = _extract_state_features(pool_data, request.investment)

        # Extract price and volatility (V2: normalized features)
        # V2: f1 is normalized price (1.0), need actual price from pool data
        current_price = float(hourly_data[0]['close'])
        # f2 is relative volatility, convert to absolute
        relative_volatility = float(state[1])
        volatility = relative_volatility * current_price

        print(f"[Predict] Current price: {current_price:.6f}, Volatility: {volatility:.6f}")

        # Step 4: Run inference
        print(f"[Predict] Running model inference")
        action, _states = model.predict(state, deterministic=True)

        print(f"[Predict] Model action: {action}")

        # Step 5: Convert action to price range (log-scale V2)
        fee_tier = int(pool_info.get('feeTier', 500))
        token0_decimals = int(pool_info['token0']['decimals'])
        token1_decimals = int(pool_info['token1']['decimals'])

        min_price, max_price = action_to_range(
            action, current_price, volatility,
            fee_tier, token0_decimals, token1_decimals
        )

        print(f"[Predict] Predicted range: [{min_price:.6f}, {max_price:.6f}]")

        # Calculate confidence based on action magnitude
        # Small actions (near 0) = high confidence (range is optimal)
        action_magnitude = np.abs(action).mean()
        confidence = max(0.7, min(0.95, 1.0 - action_magnitude * 0.5))

        predicted_range = RangeRecommendation(
            min=min_price,
            max=max_price,
            confidence=round(confidence, 2)
        )

        # Build response
        # Get actual liquidity from pool_info (not normalized state value)
        pool_liquidity = float(pool_info.get('liquidity', 1000000))

        response = PredictResponse(
            status="success",
            pool_info={
                "pool_id": request.pool_id,
                "token0": pool_info['token0']['symbol'],
                "token1": pool_info['token1']['symbol'],
                "fee_tier": int(pool_info.get('feeTier', 500)),
                "current_price": current_price
            },
            current_state={
                "price": current_price,
                "volatility": volatility,
                "liquidity": pool_liquidity
            },
            predicted_range=predicted_range,
            model_name=request.model_name,
            model_version="v2",  # V2 models only
            timestamp=datetime.utcnow()
        )

        print(f"[Predict] Prediction successful")
        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Predict] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/models/available")
async def list_available_models():
    """
    List all available V2 trained models (24-dim).

    Returns:
        List of V2 models with name, observation_dim, and size
    """
    try:
        models_dir = Path("models")
        if not models_dir.exists():
            models_dir = Path("backend/models")

        if not models_dir.exists():
            return {
                "status": "error",
                "message": "Models directory not found",
                "models": []
            }

        # Find all .zip model files
        model_files = list(models_dir.glob("*.zip"))

        models = []
        for model_file in model_files:
            model_name = model_file.stem

            # Try to load model and detect metadata
            try:
                model_data = _load_model(model_name)
                metadata = model_data['metadata']

                models.append({
                    "name": model_name,
                    "observation_dim": metadata['observation_dim'],
                    "action_dim": metadata['action_dim'],
                    "size_mb": round(model_file.stat().st_size / 1024 / 1024, 2)
                })
            except Exception as load_error:
                # Skip V1 models (28-dim) or invalid models
                print(f"[Models] Skipping {model_name}: {load_error}")
                continue

        return {
            "status": "success",
            "models": models,
            "count": len(models)
        }

    except Exception as e:
        print(f"[Models] Error listing models: {e}")
        return {
            "status": "error",
            "message": str(e),
            "models": []
        }
