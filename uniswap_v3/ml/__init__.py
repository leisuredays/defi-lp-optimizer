"""
Uniswap V3 ML Module

RL 환경, 학습, 유틸리티.

Usage:
    # 환경 사용
    from uniswap_v3.ml import UniswapV3LPEnv

    # 수학 함수 (저수준 - math 모듈에서 직접)
    from uniswap_v3.math import price_to_tick, tick_to_price, split_tokens

    # 수학 함수 (고수준 - adapter)
    from uniswap_v3.ml import calc_liquidity, calc_il, calc_fees

    # 학습
    python -m uniswap_v3.ml.trainer --data data.parquet --steps 1000000
"""

# 저수준 수학 함수 (math 모듈에서 re-export)
from ..math import (
    price_to_tick,
    tick_to_price,
    tick_to_sqrt,
    snap_tick,
    amounts_to_L,
    L_to_amounts,
    fee_delta,
    split_tokens,
)

# 고수준 ML 래퍼 (adapter)
from .adapter import (
    snap_price,
    snap_price as round_to_nearest_tick,  # Alias for compatibility
    get_tick_from_price,
    calc_liquidity,
    get_token_amounts,
    calc_il,
    calc_fees,
    active_ratio,
    action_to_range,
    tokens_for_strategy,
    liquidity_for_strategy,
)

# 환경 및 학습
from .environment import UniswapV3LPEnv
from .environment_v9 import UniswapV3LPEnvV9  # v9: Excess Return reward
from .callbacks import RewardLoggingCallback, ProgressCallback
from .trainer import train
# from .visualize import create_visualization  # Module not yet implemented

# HMM Regime Detection
from .regime import RegimeDetector, create_regime_detector
