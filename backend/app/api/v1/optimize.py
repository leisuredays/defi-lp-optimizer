"""
Optimization Endpoint

Main endpoint for ML-based range optimization.
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
import numpy as np

from app.api.schemas import (
    OptimizeRequest,
    OptimizeResponse,
    RangeRecommendation,
    ExpectedPerformance,
    ErrorResponse
)
from app.core.graph_client import fetch_pool_data_for_optimization
from app.config import settings
from app.ml.uniswap_v3_adapter import round_to_nearest_tick

router = APIRouter()


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_ranges(request: OptimizeRequest):
    """
    Get ML-optimized range recommendations for a Uniswap V3 pool

    Flow:
    1. Fetch historical pool data from The Graph
    2. Extract features from data
    3. Run PPO model inference (currently using heuristic baseline)
    4. Round recommendations to valid ticks
    5. Simulate backtest to estimate performance
    6. Return recommendations with expected metrics

    Args:
        request: OptimizeRequest with pool_id, current_ranges, etc.

    Returns:
        OptimizeResponse with recommended ranges and expected performance
    """
    try:
        # Step 1: Fetch pool data from The Graph
        print(f"[Optimize] Fetching data for pool {request.pool_id} on protocol {request.protocol_id}")
        pool_data = await fetch_pool_data_for_optimization(
            pool_id=request.pool_id,
            protocol_id=request.protocol_id,
            days=request.days_history
        )

        if not pool_data['pool_info']:
            raise HTTPException(
                status_code=404,
                detail=f"Pool {request.pool_id} not found on protocol {request.protocol_id}"
            )

        if not pool_data['hourly_data'] or len(pool_data['hourly_data']) < 24:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient historical data. Need at least 24 hours, got {len(pool_data.get('hourly_data', []))}"
            )

        pool_info = pool_data['pool_info']
        hourly_data = pool_data['hourly_data']

        print(f"[Optimize] Fetched {len(hourly_data)} hours of data")

        # Step 2: Extract features and calculate volatility
        prices = [float(h['close']) for h in hourly_data[:168]]  # Last 7 days
        volatility = np.std(prices)
        mean_price = np.mean(prices)

        print(f"[Optimize] Price: {request.current_price:.2f}, Volatility: {volatility:.2f}")

        # Step 3: Run optimization (currently heuristic, will be replaced with PPO model)
        recommendations = _generate_recommendations(
            current_price=request.current_price,
            volatility=volatility,
            mean_price=mean_price,
            current_ranges=request.current_ranges,
            fee_tier=request.fee_tier,
            pool_info=pool_info
        )

        # Step 4: Estimate expected performance
        expected_performance = _estimate_performance(
            recommendations=recommendations,
            hourly_data=hourly_data,
            current_price=request.current_price,
            investment=request.investment,
            protocol_id=request.protocol_id
        )

        print(f"[Optimize] Generated recommendations: S1=[{recommendations['S1'].min:.2f}, {recommendations['S1'].max:.2f}]")

        return OptimizeResponse(
            status="success",
            recommendations=recommendations,
            expected_performance=expected_performance,
            model_version=settings.MODEL_VERSION,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Optimize] Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


def _generate_recommendations(current_price: float, volatility: float,
                              mean_price: float, current_ranges: dict,
                              fee_tier: int, pool_info: dict) -> dict:
    """
    Generate range recommendations using heuristic baseline.

    TODO: Replace with actual PPO model inference

    Args:
        current_price: Current market price
        volatility: Price volatility (std dev)
        mean_price: Mean price over period
        current_ranges: Current user ranges
        fee_tier: Pool fee tier
        pool_info: Pool configuration

    Returns:
        Dict of RangeRecommendation by strategy ID
    """
    # Get token decimals
    decimal0 = int(pool_info['token0']['decimals'])
    decimal1 = int(pool_info['token1']['decimals'])

    # Heuristic: Adjust ranges based on volatility
    # - High volatility (>5%): Wider ranges (±10% from current price)
    # - Medium volatility (2-5%): Medium ranges (±7% from current price)
    # - Low volatility (<2%): Tighter ranges (±5% from current price)

    volatility_pct = (volatility / current_price) * 100

    if volatility_pct > 5:
        # High volatility - wide ranges
        s1_pct = 0.10  # ±10%
        s2_pct = 0.15  # ±15%
        confidence = 0.75
    elif volatility_pct > 2:
        # Medium volatility - medium ranges
        s1_pct = 0.07  # ±7%
        s2_pct = 0.12  # ±12%
        confidence = 0.82
    else:
        # Low volatility - tight ranges
        s1_pct = 0.05  # ±5%
        s2_pct = 0.08  # ±8%
        confidence = 0.88

    # Calculate S1 ranges
    s1_min = current_price * (1 - s1_pct)
    s1_max = current_price * (1 + s1_pct)

    # Round to valid ticks
    s1_min = round_to_nearest_tick(s1_min, fee_tier, decimal0, decimal1)
    s1_max = round_to_nearest_tick(s1_max, fee_tier, decimal0, decimal1)

    # Calculate S2 ranges (wider)
    s2_min = current_price * (1 - s2_pct)
    s2_max = current_price * (1 + s2_pct)

    # Round to valid ticks
    s2_min = round_to_nearest_tick(s2_min, fee_tier, decimal0, decimal1)
    s2_max = round_to_nearest_tick(s2_max, fee_tier, decimal0, decimal1)

    return {
        "S1": RangeRecommendation(
            min=s1_min,
            max=s1_max,
            confidence=confidence
        ),
        "S2": RangeRecommendation(
            min=s2_min,
            max=s2_max,
            confidence=confidence * 0.95  # S2 slightly lower confidence
        )
    }


def _estimate_performance(recommendations: dict, hourly_data: list,
                         current_price: float, investment: float,
                         protocol_id: int) -> ExpectedPerformance:
    """
    Estimate expected performance of recommended ranges.

    TODO: Use actual backtest simulation with calc_fees()

    Args:
        recommendations: Recommended ranges
        hourly_data: Historical pool data
        current_price: Current market price
        investment: Investment amount
        protocol_id: Protocol ID for gas costs

    Returns:
        ExpectedPerformance with estimated metrics
    """
    # Simple heuristic estimation (will be replaced with actual backtest)
    # Estimate based on recent fee growth and in-range probability

    s1_range = recommendations['S1']

    # Calculate in-range probability for last 7 days
    in_range_hours = 0
    total_hours = min(168, len(hourly_data))  # 7 days

    for h in hourly_data[:total_hours]:
        price = float(h['close'])
        if s1_range.min <= price <= s1_range.max:
            in_range_hours += 1

    in_range_pct = (in_range_hours / total_hours) * 100 if total_hours > 0 else 50

    # Estimate APR based on fee tier and in-range %
    # - 0.05% fee tier: ~20% base APR
    # - 0.30% fee tier: ~40% base APR
    # - 1.00% fee tier: ~80% base APR
    fee_tier_apr = {
        500: 20,
        3000: 40,
        10000: 80
    }

    base_apr = fee_tier_apr.get(s1_range.min, 40)  # Default to 0.3% tier
    estimated_apr = base_apr * (in_range_pct / 100) * s1_range.confidence

    # Estimate fees for 30 days
    expected_fees = (investment * estimated_apr / 100) * (30 / 365)

    # Estimate IL (simple approximation based on range width)
    range_width_pct = ((s1_range.max - s1_range.min) / current_price) * 100
    expected_il = investment * 0.01 * (10 / range_width_pct)  # Tighter range = more IL

    # Gas costs
    gas_cost = settings.get_gas_cost(protocol_id)

    # Net return
    net_return = expected_fees - expected_il - gas_cost

    return ExpectedPerformance(
        apr=round(estimated_apr, 2),
        expected_fees=round(expected_fees, 2),
        expected_il=round(expected_il, 2),
        gas_costs=round(gas_cost, 2),
        net_return=round(net_return, 2)
    )
